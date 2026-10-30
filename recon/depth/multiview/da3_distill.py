#!/usr/bin/env python3
"""Train DA3's depth head across scenes on true depth (research only).

The per-scene adaptation (recon.depth.multiview.da3_adapt) teaches DA3 one scene's corrections from that
scene's own fused depth. This stage asks whether training once, across many scenes, on true depth does the
same job before any scene is seen: the first student of the distillation plan (docs/PIPELINE.md).

Data: the MobileBrick training split (Li et al., "MobileBrick: Building LEGO for 3D Reconstruction on
Mobile Devices", CVPR 2023; runs/real/mobilebrick_dataset/train, never the test scenes, DTU or Tanks and
Temples). Each scene gives full-resolution photos, true cameras (pose/*.txt camera-to-world, intrinsic/*.txt)
and depth rendered from the true mesh (gt_depth/*.png, millimetres, object only). DA3 sees the photos with
those cameras, as the pipeline's DA3 stage sees ours.

As in da3_adapt, DA3's backbone stays frozen and only the DPT depth head (50M parameters) trains: each
window's features are computed once (recon.depth.multiview.da3_adapt.window_features) and the head takes
--steps-per-window steps on them before the next window. Loss: scale-invariant absolute log-depth error on
pixels with true depth away from depth edges (each frame's median log ratio removed, since the pipeline
grounds every depth map to its own anchors anyway), plus --anchor times the log difference from the
unadapted head on all other pixels, so the background does not drift. Whole set groups (scenes sharing a
set number) are held out for validation (--val-every), scored before training and after each epoch as the
median-scaled absolute depth error in millimetres.

Output: head weights for recon.depth.multiview.da3 --head-weights, and a JSON report beside them. DA3-Giant
weights are CC BY-NC 4.0, so the trained head is for research only.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from recon.common.progress import bar, done, spinner
from recon.depth.multiview.da3 import windows
from recon.depth.multiview.da3_adapt import window_features


def set_split(names: list[str], val_every: int) -> tuple[list[str], list[str]]:
    """Training and held-out scene names; every val_every-th set number (sorted) is held out whole."""
    groups = sorted({name.split('_')[0] for name in names})
    held = set(groups[val_every // 2::val_every])
    return ([name for name in names if name.split('_')[0] not in held],
            [name for name in names if name.split('_')[0] in held])


def split_scenes(root: Path, val_every: int) -> tuple[list[Path], list[Path]]:
    """Training and held-out scenes under root (set_split)."""
    scenes = sorted(path for path in root.iterdir() if (path / 'gt_depth').is_dir())
    train, held = set_split([path.name for path in scenes], val_every)
    return [root / name for name in train], [root / name for name in held]


def load_scene(scene: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Frame names, world-to-camera extrinsics (N, 4, 4) and intrinsics (N, 3, 3)."""
    names = sorted(path.stem for path in (scene / 'image').glob('*.jpg'))
    extrinsics = np.stack([np.linalg.inv(np.loadtxt(scene / 'pose' / f'{name}.txt')) for name in names])
    matrices = np.stack([np.loadtxt(scene / 'intrinsic' / f'{name}.txt')[:3, :3] for name in names])
    return names, extrinsics.astype(np.float32), matrices.astype(np.float32)


def true_depth(scene: Path, name: str, size: tuple[int, int], edge: float = .02) -> np.ndarray:
    """True depth (metres) on the head's grid, zero off the object and where two surfaces meet; the silhouette
    against the background keeps its depth."""
    depth = cv2.imread(str(scene / 'gt_depth' / f'{name}.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
    depth = cv2.resize(depth, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((3, 3), np.uint8)
    high, low = cv2.dilate(depth, kernel), cv2.erode(np.where(depth > 0, depth, 1e9).astype(np.float32), kernel)
    return np.where((depth > 0) & (high - low < edge * depth), depth, 0).astype(np.float32)


def scene_windows(api, scene: Path, window: int, process_res: int, frozen):
    """Yield (frame names, feature layers, size, unadapted depth from `frozen`) for each window of the scene."""
    names, extrinsics, matrices = load_scene(scene)
    for span in windows(len(names), window, window):
        index = list(span)
        if len(index) < 3:
            continue
        images = [cv2.cvtColor(cv2.imread(str(scene / 'image' / f'{names[i]}.jpg')), cv2.COLOR_BGR2RGB)
                  for i in index]
        with torch.no_grad():
            layers, size, base = window_features(api, images, extrinsics[index], matrices[index], process_res, frozen)
        yield [names[i] for i in index], layers, size, base


def log_error(log_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-pixel absolute log ratio after removing the frame's median (scale-invariant)."""
    residual = log_pred - torch.log(target)
    return (residual - residual.median().detach()).abs()


def scaled_error_mm(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Absolute error (mm) of the prediction after scaling it by the median true/predicted ratio."""
    valid = (target > 0) & (prediction > 0)
    scaled = prediction[valid] * np.median(target[valid] / prediction[valid])
    return np.abs(scaled - target[valid]) * 1000


def validate(api, scenes: list[Path], window: int, process_res: int, frozen) -> dict:
    """Median-scaled depth error of the unadapted and current heads on the held-out scenes."""
    head = api.model.head
    head.eval()
    per_scene, pooled = {}, {'base': [], 'head': []}
    for scene in scenes:
        errors = {'base': [], 'head': []}
        for names, layers, size, base in scene_windows(api, scene, window, process_res, frozen):
            with torch.no_grad():
                current = head([(layer.cuda().float(),) for layer in layers], *size,
                               patch_start_idx=0)['depth'][0].float().cpu().numpy()
            for slot, name in enumerate(names):
                target = true_depth(scene, name, size)
                if (target > 0).sum() < 200:
                    continue
                errors['base'].append(scaled_error_mm(base[slot].numpy(), target))
                errors['head'].append(scaled_error_mm(current[slot], target))
        if errors['base']:
            per_scene[scene.name] = {key: float(np.median(np.concatenate(value))) for key, value in errors.items()}
            for key in pooled:
                pooled[key].append(np.concatenate(errors[key]))
    result = {}
    for key, value in pooled.items():
        value = np.concatenate(value)
        result[key] = {'median_mm': float(np.median(value)), 'p90_mm': float(np.quantile(value, .9))}
    result['scenes_improved'] = sum(row['head'] < row['base'] for row in per_scene.values())
    result['scenes'] = len(per_scene)
    result['per_scene'] = per_scene
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--train-root', type=Path, default=Path('runs/real/mobilebrick_dataset/train'))
    p.add_argument('--output', type=Path, required=True, help='Trained head weights (.pt)')
    p.add_argument('--checkpoint', default='depth-anything/DA3-GIANT')
    p.add_argument('--window', type=int, default=16)
    p.add_argument('--process-res', type=int, default=756)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--steps-per-window', type=int, default=4)
    p.add_argument('--frames-per-step', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--anchor', type=float, default=.1)
    p.add_argument('--val-every', type=int, default=8, help='Hold out every n-th set number for validation')
    p.add_argument('--max-scenes', type=int, help='Train on only this many scenes (smoke test)')
    a = p.parse_args()
    from depth_anything_3.api import DepthAnything3
    train, held = split_scenes(a.train_root, a.val_every)
    if a.max_scenes:
        train, held = train[:a.max_scenes], held[:max(1, a.max_scenes // 4)]
    with spinner(f'Loading {a.checkpoint}'), contextlib.redirect_stdout(io.StringIO()):
        api = DepthAnything3.from_pretrained(a.checkpoint).to('cuda').eval()
    head = api.model.head
    frozen = copy.deepcopy(head).eval()          # the unadapted head: validation baseline and anchor
    started = time.time()
    report = {'checkpoint': a.checkpoint, 'train_scenes': [s.name for s in train],
              'held_out_scenes': [s.name for s in held], 'window': a.window, 'process_res': a.process_res,
              'epochs': a.epochs, 'steps_per_window': a.steps_per_window, 'frames_per_step': a.frames_per_step,
              'lr': a.lr, 'anchor': a.anchor, 'validation': [validate(api, held, a.window, a.process_res, frozen)]}
    print(f"Held out {len(held)} scenes: median {report['validation'][0]['base']['median_mm']:.3f} mm before training")
    for parameter in head.parameters():
        parameter.requires_grad_(True)
    optimiser = torch.optim.Adam(head.parameters(), lr=a.lr)
    rng = np.random.default_rng(0)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    losses = []
    for epoch in range(a.epochs):
        head.train()
        for scene in bar([train[i] for i in rng.permutation(len(train))], desc=f'Epoch {epoch + 1}', unit='scene'):
            for names, layers, size, base in scene_windows(api, scene, a.window, a.process_res, frozen):
                targets = [torch.from_numpy(true_depth(scene, name, size)) for name in names]
                usable = [slot for slot, target in enumerate(targets) if (target > 0).sum() >= 200]
                if not usable:
                    continue
                for _ in range(a.steps_per_window):
                    slots = sorted(rng.choice(usable, min(a.frames_per_step, len(usable)), replace=False).tolist())
                    prediction = head([(layer[:, slots].cuda().float(),) for layer in layers], *size,
                                      patch_start_idx=0)['depth'][0]
                    log_pred = torch.log(prediction.clamp(min=1e-6))
                    unadapted = torch.log(base[slots].cuda().clamp(min=1e-6))
                    loss = 0.
                    for k, slot in enumerate(slots):
                        target = targets[slot].cuda()
                        label = target > 0
                        loss = loss + log_error(log_pred[k][label], target[label]).mean()
                        loss = loss + a.anchor * (log_pred[k] - unadapted[k]).abs()[~label].mean()
                    loss = loss / len(slots)
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                    losses.append(float(loss))
        torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, a.output)
        report['validation'].append(validate(api, held, a.window, a.process_res, frozen))
        report['loss_by_epoch'] = report.get('loss_by_epoch', []) + [float(np.mean(losses[-1000:]))]
        report['seconds'] = time.time() - started
        a.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
        last = report['validation'][-1]
        print(f"Epoch {epoch + 1}: held-out median {last['base']['median_mm']:.3f} -> {last['head']['median_mm']:.3f} mm, "
              f"p90 {last['base']['p90_mm']:.3f} -> {last['head']['p90_mm']:.3f}, "
              f"{last['scenes_improved']}/{last['scenes']} scenes better")
    done(f'DA3 head trained on {len(train)} scenes -> {a.output}')


if __name__ == '__main__':
    main()
