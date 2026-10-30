#!/usr/bin/env python3
"""Fuse depth candidates from any sources with the learned judge (recon.depth.judge.train).

The judge scores every candidate and "none of these" on the evidence grid; --smooth averages the scores
over a window of grid pixels, the scores are brought up to the working frame and, at each object pixel,
the top candidate's depth is kept, averaged (by the judge's weights) only with candidates within --agree
of it (0: the top pick alone), so two different surfaces are never blended into a lump between them.
Where "none" outscores every candidate the pixel is marked in <frame>_none.png; with --abstain keep
(default) it still takes the top candidate's depth, since an empty pixel can cost a whole part of the
object downstream (space_shuttle lost an 8% piece that way), and the cross-view check that follows
removes depth no other view supports; --abstain empty leaves it empty. Writes <frame>_grounded.npy and
<frame>_confidence.npy (the judge's probability that the depth is within 1 mm, same names as
recon.depth.fusion.filter, so it drops in where the fused depth goes). Candidates' self-reports
(<frame>_selfcheck.npz beside their depth) are read when the weights were trained with them, and so
are the photo's appearance features (recon.depth.judge.appearance, with the projection saved in the
weights) and the full-resolution photo.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from recon.common.colmap import parse_model
from recon.common.progress import bar, done
from recon.depth.fusion.photometric import Images
from recon.depth.fusion.views import neighbours
from recon.depth.judge.appearance import load_backbone, patch_features, photo_files, project
from recon.depth.judge.evidence import IMAGE_FEATURES, Capture, candidate_evidence, grid, image_evidence, self_report
from recon.depth.judge.network import JudgeNet
from recon.depth.judge.train import CANDIDATE_FEATURES, appearance_grid, judge_inputs, photo_grid


def fuse(depth: np.ndarray, probability: np.ndarray, agree: float, keep: bool = False,
         margin: float = 0.) -> tuple[np.ndarray, np.ndarray]:
    """Fused depth (H, W) and the "none" mask from candidate depth (C, H, W) and probabilities (C + 1, H, W);
    keep fills "none" pixels with the top candidate's depth instead of leaving them empty. With a margin,
    a pixel leaves the frame's anchor (the candidate with the highest mean probability over the frame)
    only for a candidate scoring at least `margin` more there."""
    present = depth > 0
    scores = np.where(present, probability[:-1], 0)
    top = scores.argmax(0)
    if margin > 0 and present.any():
        anchor = int(np.argmax(scores.sum((1, 2))))
        stay = present[anchor] & (np.take_along_axis(scores, top[None], 0)[0] - scores[anchor] < margin)
        top = np.where(stay, anchor, top)
    top_depth = np.take_along_axis(depth, top[None], 0)[0]
    near = present & (np.abs(depth - top_depth) <= agree * top_depth)
    weight = np.where(near, scores, 0)
    fused = (weight * depth).sum(0) / np.maximum(weight.sum(0), 1e-9)
    nothing = probability[-1] > scores.max(0)
    return np.where(present.any(0) & (keep | ~nothing), fused, 0).astype(np.float32), nothing & present.any(0)


def pool_scores(probability: torch.Tensor, present: torch.Tensor, window: int) -> torch.Tensor:
    """Probabilities (1, C + 1, h, w) averaged over window x window grid pixels where each candidate is
    present, so the choice follows regions rather than single pixels."""
    if window <= 1:
        return probability
    weight = torch.cat([present.float(), torch.ones_like(present[:, :1]).float()], 1)
    total = F.avg_pool2d(probability * weight, window, 1, window // 2, count_include_pad=False)
    count = F.avg_pool2d(weight, window, 1, window // 2, count_include_pad=False)
    pooled = total / count.clamp(min=1e-6) * weight
    return pooled / pooled.sum(1, keepdim=True).clamp(min=1e-9)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('sequence', type=Path)
    p.add_argument('--colmap-model', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True, help='Full-resolution photos')
    p.add_argument('--candidate', nargs=2, action='append', required=True, metavar=('NAME', 'DIR'))
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--stride', type=int, default=4, help='Evidence grid, as in training')
    p.add_argument('--agree', type=float, default=.003,
                   help='Relative depth within which candidates are averaged (0: the top pick alone)')
    p.add_argument('--smooth', type=int, default=1, help='Window (grid pixels) the scores are averaged over')
    p.add_argument('--margin', type=float, default=0.,
                   help='Leave the frame\'s most trusted candidate only for one scoring this much more')
    p.add_argument('--abstain', choices=('keep', 'empty'), default='keep',
                   help='Where "none" wins: keep the top candidate\'s depth (marked), or leave the pixel empty')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    intrinsics, _, frames = parse_model(a.colmap_model)
    numbers = [number for number, *_ in frames]
    K = {i: intrinsics[n] for i, n in enumerate(numbers)}
    R = {i: f[1] for i, f in enumerate(frames)}
    t = {i: f[2] for i, f in enumerate(frames)}
    masks = {i: cv2.imread(str(a.sequence / 'masks' / f'{n:08d}.png'), cv2.IMREAD_GRAYSCALE) > 127
             for i, n in enumerate(numbers)}
    height, width = masks[0].shape
    candidates = {}
    for name, folder in a.candidate:
        candidates[name] = {i: np.load(Path(folder) / f'{n:08d}_grounded.npy').astype(np.float32)
                            for i, n in enumerate(numbers) if (Path(folder) / f'{n:08d}_grounded.npy').is_file()}
    capture = Capture(numbers, K, R, t, Images(a.images, numbers, width, a.device), masks, candidates, a.device)
    near = neighbours(np.stack([R[i] for i in range(len(numbers))]), np.stack([t[i] for i in range(len(numbers))]),
                      6, (10., 35.))
    state = torch.load(a.weights, map_location=a.device, weights_only=False)
    if 'config' in state:
        config, projection, state = state['config'], state['projection'], state['state']
    else:                                                        # v1 (no self-reports) and v3 weights
        features = state['encoder.down1.0.weight'].shape[1] - len(IMAGE_FEATURES)
        config, projection = {'candidate_features': features, 'image_features': len(IMAGE_FEATURES),
                              'photo_features': 0, 'extra': []}, None
    with_reports = config['candidate_features'] == CANDIDATE_FEATURES
    net = JudgeNet(config['candidate_features'], config['image_features'],
                   photo_features=config['photo_features']).to(a.device).eval()
    net.load_state_dict(state)
    if 'appearance' in config['extra']:
        backbone = load_backbone(a.device)
    if config['extra']:
        files = photo_files(a.images)
    rows, cols = grid(height, width, a.stride)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    with torch.inference_mode():
        for index, number in enumerate(bar(numbers, desc='Judging depth', unit='frame')):
            own, pairs, depth_grid, region = candidate_evidence(capture, index, near[index], a.stride)
            photo = cv2.imread(str(a.sequence / 'frames' / f'{number:08d}.png'))
            T = lambda x: torch.as_tensor(x, dtype=torch.float32, device=a.device)[None]
            report = (T(np.stack([self_report(Path(folder), number, rows, cols) for _, folder in a.candidate]))
                      if with_reports else None)
            inputs, present = judge_inputs(T(own), T(pairs), T(depth_grid), report)
            image = image_evidence(photo, masks[index], rows, cols)
            if config['extra']:
                full = cv2.imread(str(files[number]))
            if 'appearance' in config['extra']:
                appearance = project(patch_features(backbone, full, a.device), projection)
                image = np.concatenate([image, appearance_grid(appearance, image.shape[-2:])])
            photo_in = (torch.from_numpy(photo_grid(full, image.shape[-2:])).to(a.device)[None].float() / 255
                        if config['photo_features'] else None)
            logits, confidence = net(inputs, T(image), present, photo_in)
            probability = pool_scores(torch.softmax(logits, 1).nan_to_num(0), present, a.smooth)
            # The grid holds every stride-th pixel; upsample to the frame, aligned on those pixels.
            size = ((len(rows) - 1) * a.stride + 1, (len(cols) - 1) * a.stride + 1)
            up = lambda x: F.pad(F.interpolate(x, size=size, mode='bilinear', align_corners=True),
                                 (0, width - size[1], 0, height - size[0]), mode='replicate')
            probability = up(probability)[0].cpu().numpy()
            sure = torch.sigmoid(up(confidence[:, None]))[0, 0].cpu().numpy()
            depth = np.stack([candidates[name].get(index, np.zeros((height, width), np.float32)) for name in candidates])
            depth = np.where(masks[index][None], depth, 0)
            fused, nothing = fuse(depth, probability, a.agree, a.abstain == 'keep', a.margin)
            np.save(a.output_dir / f'{number:08d}_grounded.npy', fused)
            np.save(a.output_dir / f'{number:08d}_confidence.npy', np.where(fused > 0, sure, 0).astype(np.float32))
            cv2.imwrite(str(a.output_dir / f'{number:08d}_none.png'), nothing.astype(np.uint8) * 255)
            chosen = np.where(fused > 0, np.where(depth > 0, probability[:-1], 0).argmax(0), -1)
            reports.append({'frame': number, 'pixels': int((fused > 0).sum()), 'none': int(nothing.sum()),
                            'share': {name: float((chosen == c).sum() / max(1, (fused > 0).sum()))
                                      for c, name in enumerate(candidates)}})
    summary = {'method': 'learned judge', 'weights': str(a.weights), 'candidates': {n: d for n, d in a.candidate},
               'agree': a.agree, 'smooth': a.smooth, 'abstain': a.abstain, 'margin': a.margin, 'self_reports': with_reports,
               'inputs': config['extra'],
               'none_pixels': sum(r['none'] for r in reports),
               'share': {name: float(np.mean([r['share'][name] for r in reports])) for name in candidates},
               'per_frame': reports}
    (a.output_dir / 'hybrid_depth.json').write_text(json.dumps(summary, indent=2) + '\n')
    done(f"Judge: {len(reports)} frames; share {', '.join(f'{k} {v:.0%}' for k, v in summary['share'].items())}")


if __name__ == '__main__':
    main()
