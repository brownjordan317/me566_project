#!/usr/bin/env python3
"""Pick, region by region, the depth model the images support.

All candidate depth maps (VGGT, DA3, MoGe, Depth Pro, ...) are grounded in the
same map frame. On a grid of object pixels every candidate's depth is scored
photometrically against neighbouring full-resolution views
(recon.depth.fusion.photometric). Scores are pooled over a window, a
per-model prior is added (log-scale reliability measured on development
scenes: a weak model must be clearly better supported by the images to win),
and every object pixel takes the depth of the winning candidate. Measured on
MobileBrick development scenes, almost all of the gain comes from choosing
between the two multi-view models; the per-pixel choice is too noisy, the
pooled choice is not.

Pixels outside the mask keep the base (hybrid) depth. Outputs:
<frame>_grounded.npy, <frame>_confidence.npy (pooled score margin of the
winner over the runner-up) and hybrid_depth.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import warnings

import numpy as np
import torch

from recon.common.colmap import parse_model
from recon.common.progress import bar, done, spinner
from recon.depth.fusion.photometric import Images, ncc_scores, pool
from recon.depth.fusion.views import neighbours


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True, help='COLMAP text model (map frame)')
    p.add_argument('--masks-dir', type=Path, required=True)
    p.add_argument('--base-depth', type=Path, required=True, help='Depth kept outside the mask')
    p.add_argument('--images', type=Path, required=True, help='Full-resolution frames (numbered files)')
    p.add_argument('--candidate', nargs=3, action='append', metavar=('NAME', 'DIR', 'PRIOR'),
                   required=True, help='A grounded depth model and its prior (score units; 0 for the best)')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--neighbours', type=int, default=6)
    p.add_argument('--angle-band', type=float, nargs=2, metavar=('LOW', 'HIGH'),
                   help='Prefer neighbours whose viewing direction differs by this many degrees')
    p.add_argument('--radius', type=int, default=4, help='Patch radius in full-resolution pixels')
    p.add_argument('--stride', type=int, default=2, help='Score every STRIDE-th working pixel')
    p.add_argument('--window', type=int, default=17, help='Pooling window in working pixels')
    p.add_argument('--combine', choices=('score', 'mean', 'median'), default='score',
                   help="How to settle on a depth per pixel: the pooled patch score's winner (default), or "
                        'the candidates averaged or their median, which need no photometric comparison')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    device = a.device
    intrinsics, _, frames = parse_model(a.model)
    numbers = [number for number, *_ in frames]
    names = [name for name, _, _ in a.candidate]
    priors = np.array([float(prior) for _, _, prior in a.candidate], np.float32)
    T = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    K = {i: T(intrinsics[number]) for i, number in enumerate(numbers)}
    R = {i: T(rotation) for i, (_, rotation, _, _) in enumerate(frames)}
    t = {i: T(translation) for i, (_, _, translation, _) in enumerate(frames)}
    near = neighbours(np.stack([f[1] for f in frames]), np.stack([f[2] for f in frames]), a.neighbours,
                      tuple(a.angle_band) if a.angle_band else None)
    first_base = np.load(a.base_depth / f'{numbers[0]:08d}_grounded.npy')
    with spinner('Loading full-resolution images'):
        images = Images(a.images, numbers, first_base.shape[1], device)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    wins = np.zeros(len(names))
    reports = []
    with torch.inference_mode():
        for index, number in enumerate(bar(numbers, desc='Photometric fusion', unit='frame')):
            base_path = a.base_depth / f'{number:08d}_grounded.npy'
            if not base_path.is_file():
                continue
            base = np.load(base_path).astype(np.float32)
            height, width = base.shape
            mask = cv2.imread(str(a.masks_dir / f'{number:08d}.png'), cv2.IMREAD_GRAYSCALE) > 127
            stack = np.zeros((len(names), height, width), np.float32)
            for c, (_, directory, _) in enumerate(a.candidate):
                path = Path(directory) / f'{number:08d}_grounded.npy'
                if path.is_file():
                    stack[c] = np.load(path)
            present = stack > 0
            grid = np.zeros((height, width), bool)
            grid[::a.stride, ::a.stride] = True
            pooled = np.full((len(names), height, width), np.nan, np.float32)
            for c in range(len(names)):
                sampled = mask & grid & present[c]
                v, u = np.nonzero(sampled)
                if not len(u):
                    continue
                score = ncc_scores(images, index, near[index], K, R, t, T(stack[c][v, u]), T(u), T(v),
                                   a.radius).cpu().numpy()
                values = np.zeros((height, width), np.float32)
                values[v, u] = score
                pooled[c] = pool(values, sampled, a.window)
            ranked = np.where(present & np.isfinite(pooled), pooled + priors[:, None, None], -np.inf)
            choice = ranked.argmax(0)
            ordered = np.sort(ranked, axis=0)
            margin = np.where(np.isfinite(ordered[-2]), ordered[-1] - ordered[-2], 0)
            object_pixels = mask & np.isfinite(ranked.max(0))
            if a.combine == 'score':
                chosen = np.take_along_axis(stack, choice[None], 0)[0]
            else:
                # Photometry-free alternatives, for captures whose neighbours are too far apart for the
                # patch score to mean much: combine the candidates instead of choosing between them.
                covered = present.sum(0)
                if a.combine == 'mean':
                    # Summed and divided explicitly: nanmean warns on every pixel no candidate covers.
                    chosen = (np.where(present, stack, 0).sum(0)
                              / np.maximum(covered, 1)).astype(np.float32)
                else:
                    candidates = np.where(present, stack, np.nan)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN pixels
                        chosen = np.nanmedian(candidates, 0).astype(np.float32)
                chosen = np.where((covered > 0) & np.isfinite(chosen), chosen, 0)
                object_pixels = mask & (chosen > 0)
            fused = np.where(mask, 0, base).astype(np.float32)
            fused[object_pixels] = chosen[object_pixels]
            confidence = np.where(object_pixels, margin, 0).astype(np.float32)
            np.save(a.output_dir / f'{number:08d}_grounded.npy', fused)
            np.save(a.output_dir / f'{number:08d}_confidence.npy', confidence)
            counts = np.bincount(choice[object_pixels], minlength=len(names))
            wins += counts
            reports.append({'frame': number, 'object_pixels': int(object_pixels.sum()),
                            'chosen': {name: int(count) for name, count in zip(names, counts)}})
    share = wins / max(wins.sum(), 1)
    summary = {'method': 'pooled photometric selection between depth models', 'candidates': names,
               'priors': priors.tolist(), 'neighbours': a.neighbours, 'radius': a.radius,
               'stride': a.stride, 'window': a.window, 'image_scale': images.scale,
               'chosen_share': {name: float(x) for name, x in zip(names, share)},
               'frames': len(reports), 'per_frame': reports}
    (a.output_dir / 'hybrid_depth.json').write_text(json.dumps(summary, indent=2) + '\n')
    done('Photometric fusion: ' + ', '.join(f'{name} {x:.0%}' for name, x in zip(names, share)))


if __name__ == '__main__':
    main()
