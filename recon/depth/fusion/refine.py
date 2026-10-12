#!/usr/bin/env python3
"""Fine-tune fused object depth against full-resolution images (plane sweep).

After the model choice (recon.depth.fusion.filter), each object pixel's depth
is swept over a narrow band of relative offsets around it. Every offset is
scored photometrically against neighbouring views
(recon.depth.fusion.photometric), the scores are averaged over a window (cost
aggregation, as in local stereo; Scharstein and Szeliski, IJCV 2002), a
quadratic penalty keeps depth near the chosen model's, and the best offset is
refined to sub-step precision with a parabola through its neighbours. A pixel
moves only if the images prefer the new depth to the old one by a margin, so
textureless regions keep the model depth. Neighbours are taken 10-35 degrees
away by default: at a few degrees, as in densely sampled captures, a
millimetre of depth moves a patch by a fraction of a pixel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from recon.common.colmap import parse_model
from recon.common.progress import bar, done, spinner
from recon.depth.fusion.photometric import Images, camera_normals, ncc_scores, pool
from recon.depth.fusion.views import neighbours


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--masks-dir', type=Path, required=True)
    p.add_argument('--depth-dir', type=Path, required=True, help='Fused depth to refine')
    p.add_argument('--images', type=Path, required=True, help='Full-resolution frames (numbered files)')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--range', type=float, default=.006, help='Largest relative depth change tried')
    p.add_argument('--steps', type=int, default=13, help='Depths tried across the band (odd)')
    p.add_argument('--window', type=int, default=15, help='Cost aggregation window in working pixels')
    p.add_argument('--penalty', type=float, default=.02,
                   help='Score cost of moving the full range away from the model depth')
    p.add_argument('--margin', type=float, default=.01,
                   help='Required score gain over the model depth before a pixel moves')
    p.add_argument('--slanted', action=argparse.BooleanOptionalAction, default=True,
                   help='Tilt patches to the surface normal of the fused depth')
    p.add_argument('--neighbours', type=int, default=6)
    p.add_argument('--angle-band', type=float, nargs=2, metavar=('LOW', 'HIGH'), default=(10., 35.),
                   help='Prefer neighbours whose viewing direction differs by this many degrees; '
                        'close views cannot resolve millimetre depth changes')
    p.add_argument('--radius', type=int, default=4)
    p.add_argument('--stride', type=int, default=2)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if a.steps < 3 or a.steps % 2 == 0:
        p.error('--steps must be odd and at least 3')
    device = a.device
    intrinsics, _, frames = parse_model(a.model)
    numbers = [number for number, *_ in frames]
    T = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    K = {i: T(intrinsics[number]) for i, number in enumerate(numbers)}
    R = {i: T(rotation) for i, (_, rotation, _, _) in enumerate(frames)}
    t = {i: T(translation) for i, (_, _, translation, _) in enumerate(frames)}
    near = neighbours(np.stack([f[1] for f in frames]), np.stack([f[2] for f in frames]), a.neighbours,
                      tuple(a.angle_band))
    available = [i for i, number in enumerate(numbers) if (a.depth_dir / f'{number:08d}_grounded.npy').is_file()]
    width = np.load(a.depth_dir / f'{numbers[available[0]]:08d}_grounded.npy').shape[1]
    with spinner('Loading full-resolution images'):
        images = Images(a.images, numbers, width, device)
    offsets = np.linspace(-a.range, a.range, a.steps)
    centre = a.steps // 2
    penalty = a.penalty * (offsets / a.range) ** 2
    a.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    with torch.inference_mode():
        for index in bar(available, desc='Photometric refinement', unit='frame'):
            number = numbers[index]
            depth = np.load(a.depth_dir / f'{number:08d}_grounded.npy').astype(np.float32)
            height, width = depth.shape
            mask = (cv2.imread(str(a.masks_dir / f'{number:08d}.png'), cv2.IMREAD_GRAYSCALE) > 127) & (depth > 0)
            grid = np.zeros_like(mask)
            grid[::a.stride, ::a.stride] = True
            v, u = np.nonzero(mask & grid)
            if not len(u):
                np.save(a.output_dir / f'{number:08d}_grounded.npy', depth)
                continue
            base = T(depth[v, u])
            normals = T(camera_normals(depth, intrinsics[number])[v, u]) if a.slanted else None
            cost = np.empty((a.steps, height, width), np.float32)
            for k, offset in enumerate(offsets):
                score = ncc_scores(images, index, near[index], K, R, t, base * float(np.exp(offset)),
                                   T(u), T(v), a.radius, normals=normals).cpu().numpy()
                values = np.zeros((height, width), np.float32)
                values[v, u] = score
                cost[k] = pool(values, mask & grid, a.window) - penalty[k]
            cost = np.nan_to_num(cost, nan=-np.inf)
            best = cost.argmax(0)
            inner = np.clip(best, 1, a.steps - 2)
            left = np.take_along_axis(cost, (inner - 1)[None], 0)[0]
            middle = np.take_along_axis(cost, inner[None], 0)[0]
            right = np.take_along_axis(cost, (inner + 1)[None], 0)[0]
            curvature = left - 2 * middle + right
            with np.errstate(all='ignore'):
                shift = np.where(curvature < 0, .5 * (left - right) / curvature, 0)
            shift = np.where((best == inner) & np.isfinite(shift), np.clip(shift, -.5, .5), 0)
            step = offsets[1] - offsets[0]
            change = offsets[best] + shift * step
            gain = cost.max(0) - cost[centre]
            moved = mask & np.isfinite(gain) & (gain > a.margin)
            refined = np.where(moved, depth * np.exp(change), depth).astype(np.float32)
            np.save(a.output_dir / f'{number:08d}_grounded.npy', refined)
            reports.append({'frame': number, 'object_pixels': int(mask.sum()), 'moved': int(moved.sum()),
                            'median_change_pct': float(np.median(np.abs(change[moved])) * 100) if moved.any() else 0.})
    moved = sum(row['moved'] for row in reports) / max(sum(row['object_pixels'] for row in reports), 1)
    summary = {'method': 'aggregated photometric plane sweep around fused depth', 'range': a.range,
               'steps': a.steps, 'window': a.window, 'penalty': a.penalty, 'margin': a.margin,
               'moved_fraction': moved, 'frames': len(reports), 'per_frame': reports}
    (a.output_dir / 'hybrid_depth.json').write_text(json.dumps(summary, indent=2) + '\n')
    done(f'Photometric refinement: {moved:.0%} of object pixels moved')


if __name__ == '__main__':
    main()
