#!/usr/bin/env python3
"""Depth Anything 3 multi-view depth, conditioned on our COLMAP cameras.

DA3 (Lin et al., "Depth Anything 3: Recovering the Visual Space from Any
Views", 2025; https://github.com/ByteDance-Seed/Depth-Anything-3, code
Apache-2.0) predicts consistent depth for many views at once and can take known
camera poses and intrinsics, returning depth at their scale. Frames run in
overlapping windows (like the VGGT stage), each with our cameras supplied; the
depth is resized back to the working frames and grounded to the object's sparse
points like every other candidate. Weights: DA3-Giant and DA3-Large are
CC BY-NC 4.0, DA3-Base Apache-2.0.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from recon.common.colmap import parse_model
from recon.common.progress import bar, done, spinner
from recon.depth.grounding import ground, object_anchors, point_table


def windows(count: int, size: int, step: int) -> list[range]:
    starts = list(range(0, max(count - size, 0) + 1, step))
    if starts[-1] + size < count:
        starts.append(count - size)
    return [range(start, min(start + size, count)) for start in starts]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sequence', type=Path)
    p.add_argument('--colmap-model', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--checkpoint', default='depth-anything/DA3-GIANT')
    p.add_argument('--window', type=int, default=16)
    p.add_argument('--step', type=int, default=8)
    p.add_argument('--process-res', type=int, default=504)
    p.add_argument('--no-poses', action='store_true', help='Let DA3 estimate cameras itself')
    p.add_argument('--max-relative-error', type=float, default=.03)
    p.add_argument('--raw', action='store_true',
                   help='Save DA3 depth as predicted at our cameras\' scale, without the anchor fit')
    a = p.parse_args()
    from depth_anything_3.api import DepthAnything3
    intrinsics, points, frames = parse_model(a.colmap_model)
    points = point_table(points)
    with spinner(f'Loading {a.checkpoint}'), contextlib.redirect_stdout(io.StringIO()):
        model = DepthAnything3.from_pretrained(a.checkpoint).to('cuda').eval()
    images = [cv2.cvtColor(cv2.imread(str(a.sequence / 'frames' / f'{number:08d}.png')), cv2.COLOR_BGR2RGB)
              for number, *_ in frames]
    height, width = images[0].shape[:2]
    extrinsics = np.tile(np.eye(4, dtype=np.float32), (len(frames), 1, 1))   # world-to-camera, 4x4
    for i, (_, rotation, translation, _) in enumerate(frames):
        extrinsics[i, :3, :3], extrinsics[i, :3, 3] = rotation, translation
    matrices = np.stack([intrinsics[number] for number, *_ in frames]).astype(np.float32)
    sums = np.zeros((len(frames), height, width), np.float32)
    counts = np.zeros(len(frames), np.int32)
    with torch.inference_mode():
        for window in bar(windows(len(frames), a.window, a.step), desc='DA3 depth', unit='window'):
            index = list(window)
            with contextlib.redirect_stdout(io.StringIO()):
                prediction = model.inference(
                    [images[i] for i in index],
                    extrinsics=None if a.no_poses else extrinsics[index],
                    intrinsics=None if a.no_poses else matrices[index],
                    process_res=a.process_res)
            for slot, i in enumerate(index):
                # Windows overlap; average a frame's predictions (all share our cameras' scale).
                sums[i] += cv2.resize(prediction.depth[slot].astype(np.float32), (width, height),
                                      interpolation=cv2.INTER_LINEAR)
                counts[i] += 1
    a.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for i, (number, rotation, translation, observations) in enumerate(frames):
        depth = sums[i] / max(counts[i], 1)
        mask = cv2.imread(str(a.sequence / 'masks' / f'{number:08d}.png'), cv2.IMREAD_GRAYSCALE)
        try:
            coords, target = object_anchors(observations, points, rotation, translation, mask)
            grounded, report = ground(depth, 'depth', coords, target)
            rejected = report['relative_anchor_error'] > a.max_relative_error
        except ValueError as error:
            report, rejected = {'reason': str(error)}, True
        if a.raw:
            grounded, rejected = depth.astype(np.float32), False
        if not rejected:
            np.save(a.output_dir / f'{number:08d}_grounded.npy', grounded)
        reports.append({'frame': number, 'rejected': rejected, **report})
    kept = sum(not row['rejected'] for row in reports)
    summary = {'model': a.checkpoint, 'poses_given': not a.no_poses, 'window': a.window, 'step': a.step,
               'frames': len(reports), 'kept': kept,
               'median_scale': float(np.median([row['scale'] for row in reports if 'scale' in row])),
               'per_frame': reports}
    (a.output_dir / 'candidate.json').write_text(json.dumps(summary, indent=2) + '\n')
    done(f"DA3: {kept}/{len(reports)} frames grounded (fit scale {summary['median_scale']:.3f})")


if __name__ == '__main__':
    main()
