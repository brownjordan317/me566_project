#!/usr/bin/env python3
"""Adapt DA3's depth head to one scene, supervised by the scene's own multi-model evidence.

The two depth models (VGGT, DA3) err in different places, and the photometric
fusion knows, region by region, which one the images support
(recon.depth.fusion.filter). Where that evidence is decisive, the fused and
refined depth is a better estimate than DA3's; this stage teaches DA3 those
corrections, so that it also improves where the evidence is weak (test-time
adaptation, self-supervised from the scene: Sun et al., "Test-Time Training with
Self-Supervision", ICML 2020; for depth, e.g. Tonioni et al., CVPR 2019).

DA3's 1.1B-parameter backbone, where frames exchange information and our
cameras enter, stays frozen: its features are computed once per window (the
same windows as recon.depth.multiview.da3) and cached. Only the 50M-parameter
DPT depth head is trained, from those features, which makes each step cheap.

Labels: object pixels whose fusion confidence (pooled photometric margin of the
winner over the runner-up) is at least --label-quantile of the frame's, with the
refined depth as target. Each frame's targets are mapped into DA3's output scale
by an affine fit to the unadapted prediction. Loss: absolute log-depth error on
labels, plus --anchor times the log difference from the unadapted prediction on
all other pixels, so the head cannot drift where nothing supervises it.

Output: the adapted head's weights, loaded by recon.depth.multiview.da3
--head-weights to predict the scene's depth again.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from recon.common.colmap import parse_model
from recon.common.progress import bar, done, spinner
from recon.depth.multiview.da3 import windows


def affine_fit(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Least-squares scale and shift mapping target into prediction's space (target = s * prediction + b)."""
    A = np.stack([prediction, np.ones_like(prediction)], 1)
    (scale, shift), *_ = np.linalg.lstsq(A, target, rcond=None)
    return float(scale), float(shift)


def window_features(api, images: list[np.ndarray], extrinsics: np.ndarray, matrices: np.ndarray,
                    process_res: int, head=None) -> tuple[list[torch.Tensor], tuple[int, int], torch.Tensor]:
    """One window through DA3's frozen backbone: its feature layers (fp16, CPU), the processed image size
    and the depth (S, H, W) of `head` (default: DA3's own). Call under torch.no_grad()."""
    net = api.model
    with contextlib.redirect_stdout(io.StringIO()):
        imgs_cpu, ex, intr = api._preprocess_inputs(images, extrinsics, matrices, process_res, 'upper_bound_resize')
        imgs, ex_t, in_t = api._prepare_model_inputs(imgs_cpu, ex, intr)
        ex_n = api._normalize_extrinsics(ex_t.clone())
    with torch.autocast('cuda', dtype=torch.bfloat16):
        with torch.autocast('cuda', enabled=False):
            token = net.cam_enc(ex_n, in_t, imgs.shape[-2:])
        feats, _ = net.backbone(imgs, cam_token=token, export_feat_layers=[], ref_view_strategy='saddle_balanced')
    height, width = imgs.shape[-2:]
    layers = [feat[0].half().cpu() for feat in feats]                 # (1, S, N, C) each
    base = (head or net.head)([(layer.cuda().float(),) for layer in layers], height, width, patch_start_idx=0)['depth']
    del feats, imgs
    torch.cuda.empty_cache()
    return layers, (height, width), base[0].float().cpu()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('sequence', type=Path)
    p.add_argument('--colmap-model', type=Path, required=True)
    p.add_argument('--target-dir', type=Path, required=True, help='Refined fused depth (labels)')
    p.add_argument('--confidence-dir', type=Path, required=True, help='Fusion confidence maps')
    p.add_argument('--output', type=Path, required=True, help='Adapted head weights (.pt)')
    p.add_argument('--checkpoint', default='depth-anything/DA3-GIANT')
    p.add_argument('--window', type=int, default=16)
    p.add_argument('--step', type=int, default=8)
    p.add_argument('--process-res', type=int, default=756)
    p.add_argument('--label-quantile', type=float, default=.5)
    p.add_argument('--anchor', type=float, default=.1)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--frames-per-step', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-5)
    a = p.parse_args()
    from depth_anything_3.api import DepthAnything3
    intrinsics, _, frames = parse_model(a.colmap_model)
    with spinner(f'Loading {a.checkpoint}'), contextlib.redirect_stdout(io.StringIO()):
        api = DepthAnything3.from_pretrained(a.checkpoint).to('cuda').eval()
    net = api.model
    images = [cv2.cvtColor(cv2.imread(str(a.sequence / 'frames' / f'{number:08d}.png')), cv2.COLOR_BGR2RGB)
              for number, *_ in frames]
    extrinsics = np.tile(np.eye(4, dtype=np.float32), (len(frames), 1, 1))
    for i, (_, rotation, translation, _) in enumerate(frames):
        extrinsics[i, :3, :3], extrinsics[i, :3, 3] = rotation, translation
    matrices = np.stack([intrinsics[number] for number, *_ in frames]).astype(np.float32)

    # 1. Frozen backbone, once per window; cache features and the unadapted head output.
    cache = []
    started = time.time()
    with torch.no_grad():
        for window in bar(windows(len(frames), a.window, a.step), desc='DA3 features', unit='window'):
            index = list(window)
            layers, size, base = window_features(api, [images[i] for i in index], extrinsics[index],
                                                 matrices[index], a.process_res)
            cache.append({'index': index, 'layers': layers, 'size': size, 'base': base})

    # 2. Labels per (window, slot): targets in DA3's output scale, label mask, anchor mask.
    samples = []
    for item in cache:
        height, width = item['base'].shape[-2:]                              # head output resolution
        for slot, i in enumerate(item['index']):
            number = frames[i][0]
            target_path = a.target_dir / f'{number:08d}_grounded.npy'
            confidence_path = a.confidence_dir / f'{number:08d}_confidence.npy'
            mask = cv2.imread(str(a.sequence / 'masks' / f'{number:08d}.png'), cv2.IMREAD_GRAYSCALE)
            if not target_path.is_file() or not confidence_path.is_file() or mask is None:
                continue
            resize = lambda x: cv2.resize(x.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
            target, confidence = resize(np.load(target_path)), resize(np.load(confidence_path))
            obj = resize((mask > 127).astype(np.float32)) > .5
            base = item['base'][slot].numpy()
            valid = obj & (target > 0) & (base > 0)
            if valid.sum() < 500:
                continue
            label = valid & (confidence >= np.quantile(confidence[valid], a.label_quantile))
            scale, shift = affine_fit(base[label], target[label])
            if scale <= 0:
                continue
            target_head = np.where(label, (target - shift) / scale, 0).astype(np.float32)
            label &= target_head > 0
            samples.append({'window': item, 'slot': slot, 'target': torch.from_numpy(target_head),
                            'label': torch.from_numpy(label), 'anchor': torch.from_numpy(~label & (base > 0))})
    if not samples:
        raise SystemExit('No labelled frames')

    # 3. Train the head.
    head = net.head
    head.train()
    for parameter in head.parameters():
        parameter.requires_grad_(True)
    optimiser = torch.optim.Adam(head.parameters(), lr=a.lr)
    rng = np.random.default_rng(0)
    losses = []
    for step in bar(range(a.steps), desc='Adapting DA3 head', unit='step'):
        item = samples[rng.integers(len(samples))]['window']
        choices = [s for s in samples if s['window'] is item]
        chosen = [choices[j] for j in rng.choice(len(choices), min(a.frames_per_step, len(choices)), replace=False)]
        slots = [s['slot'] for s in chosen]
        layers = [(layer[:, slots].cuda().float(),) for layer in item['layers']]
        height, width = item['size']
        prediction = head(layers, height, width, patch_start_idx=0)['depth'][0]   # (k, H, W)
        target = torch.stack([s['target'] for s in chosen]).cuda()
        label = torch.stack([s['label'] for s in chosen]).cuda()
        anchor = torch.stack([s['anchor'] for s in chosen]).cuda()
        base = item['base'][slots].cuda()
        log_pred = torch.log(prediction.clamp(min=1e-6))
        loss = (log_pred - torch.log(target.clamp(min=1e-6))).abs()[label].mean()
        if anchor.any():
            loss = loss + a.anchor * (log_pred - torch.log(base.clamp(min=1e-6))).abs()[anchor].mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(float(loss))
    head.eval()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, a.output)
    report = {'checkpoint': a.checkpoint, 'frames': len(samples), 'steps': a.steps, 'lr': a.lr,
              'label_quantile': a.label_quantile, 'anchor': a.anchor, 'seconds': time.time() - started,
              'loss_first_20': float(np.mean(losses[:20])), 'loss_last_20': float(np.mean(losses[-20:]))}
    a.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    done(f"DA3 head adapted on {len(samples)} frames: loss {report['loss_first_20']:.4f} -> {report['loss_last_20']:.4f}")


if __name__ == '__main__':
    main()
