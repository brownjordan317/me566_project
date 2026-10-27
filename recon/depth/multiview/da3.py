#!/usr/bin/env python3
"""Depth Anything 3 multi-view depth, conditioned on our COLMAP cameras.

DA3 (Lin et al., "Depth Anything 3: Recovering the Visual Space from Any
Views", 2025; https://github.com/ByteDance-Seed/Depth-Anything-3, code
Apache-2.0) predicts consistent depth for many views at once and can take known
camera poses and intrinsics, returning depth at their scale. Frames run in
overlapping windows (like the VGGT stage), each with our cameras supplied; the
depth is resized back to the working frames and grounded to the object's sparse
points like every other candidate. The window is sized to the memory the card
has free rather than fixed (--window auto, capped at --max-window), and a window
that still will not fit is halved rather than abandoned, so the stage runs on a
smaller GPU without being retuned. --images gives DA3 the full-resolution photos
(with our intrinsics scaled to them) instead of the working frames: DA3 still
resizes to --process-res, but from the original pixels (colosseum median error
3.91 -> 3.22 mm, big_ben 2.96 -> 2.77, jeep 1.21 -> 1.19). Weights: DA3-Giant and DA3-Large are
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


# Peak allocation of one window, measured on this repository's card (25 GiB, DA3-Giant, 1600x1200 input
# at --process-res 756): 7.89 GiB at 3 frames, 8.81 at 4, 12.49 at 8, 13.33 at 12, 14.21 at 16, 15.93 at
# 24, of which 5.11 GiB is the weights. Beyond eight frames the cost is linear - DA3 does not pay
# attention over all views at once - so the activations fit 5.66 GiB + 0.215 GiB per frame, and that is
# what sizes the window. Both terms count transformer activations, which scale with the token count, so
# they scale with the square of the processing resolution.
WINDOW_FIXED_GIB, WINDOW_PER_FRAME_GIB, WINDOW_RES = 5.66, .215, 756
MIN_WINDOW = 3          # DA3 aligns its poses to ours by Umeyama, which is degenerate below 3 centres


def fitting_window(free_bytes: int, process_res: int, cap: int, fraction: float = .9) -> int:
    """The largest window whose activations fit `fraction` of the memory left after loading DA3."""
    tokens = (process_res / WINDOW_RES) ** 2
    budget = free_bytes / 2 ** 30 * fraction - WINDOW_FIXED_GIB * tokens
    return max(MIN_WINDOW, min(cap, int(budget / (WINDOW_PER_FRAME_GIB * tokens))))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sequence', type=Path)
    p.add_argument('--colmap-model', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--checkpoint', default='depth-anything/DA3-GIANT')
    p.add_argument('--window', default='auto',
                   help='Frames per window, or auto: the largest that fits the free GPU memory, capped '
                        'at --max-window (16 on a 25 GiB card, so this machine is unaffected)')
    p.add_argument('--max-window', type=int, default=16)
    p.add_argument('--step', type=int, help='Window stride (default: half the window)')
    p.add_argument('--process-res', type=int, default=504)
    p.add_argument('--no-poses', action='store_true', help='Let DA3 estimate cameras itself')
    p.add_argument('--grounding-regions', type=int, default=1,
                   help='Affine-fit the depth in this many regions of position and predicted depth, blended across the image, instead of one global fit per frame (1 keeps the global fit)')
    p.add_argument('--max-relative-error', type=float, default=.03)
    p.add_argument('--head-weights', type=Path,
                   help='Depth-head weights adapted to this scene (recon.depth.multiview.da3_adapt)')
    p.add_argument('--images', type=Path,
                   help='Full-resolution photos (numbered files) to feed DA3 instead of the working frames')
    p.add_argument('--raw', action='store_true',
                   help='Save DA3 depth as predicted at our cameras\' scale, without the anchor fit')
    a = p.parse_args()
    from depth_anything_3.api import DepthAnything3
    intrinsics, points, frames = parse_model(a.colmap_model)
    points = point_table(points)
    with spinner(f'Loading {a.checkpoint}'), contextlib.redirect_stdout(io.StringIO()):
        model = DepthAnything3.from_pretrained(a.checkpoint).to('cuda').eval()
    if a.head_weights is not None:
        model.model.head.load_state_dict(torch.load(a.head_weights, map_location='cuda'))
    if a.window == 'auto':
        a.window = fitting_window(torch.cuda.mem_get_info()[0], a.process_res, a.max_window)
        print(f'Window {a.window} frames ({torch.cuda.mem_get_info()[0] / 2 ** 30:.1f} GiB free)')
    else:
        a.window = max(MIN_WINDOW, int(a.window))
    if a.step is None:
        a.step = max(1, a.window // 2)
    height, width = cv2.imread(str(a.sequence / 'frames' / f'{frames[0][0]:08d}.png')).shape[:2]
    if a.images is not None:
        files = {int(path.stem): path for path in a.images.iterdir() if path.stem.isdigit()}
        paths = [files[number] for number, *_ in frames]
    else:
        paths = [a.sequence / 'frames' / f'{number:08d}.png' for number, *_ in frames]
    images = [cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB) for path in paths]
    scale = images[0].shape[1] / width                     # input pixels per working pixel
    extrinsics = np.tile(np.eye(4, dtype=np.float32), (len(frames), 1, 1))   # world-to-camera, 4x4
    for i, (_, rotation, translation, _) in enumerate(frames):
        extrinsics[i, :3, :3], extrinsics[i, :3, 3] = rotation, translation
    matrices = np.stack([intrinsics[number] for number, *_ in frames]).astype(np.float32)
    matrices[:, :2] *= scale
    sums = np.zeros((len(frames), height, width), np.float32)
    counts = np.zeros(len(frames), np.int32)
    def predict(index: list[int]) -> None:
        """Add one window's depth to the running sums, halving it if it will not fit."""
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                prediction = model.inference(
                    [images[i] for i in index],
                    extrinsics=None if a.no_poses else extrinsics[index],
                    intrinsics=None if a.no_poses else matrices[index],
                    process_res=a.process_res)
        except torch.OutOfMemoryError:
            # The sizing above uses the memory free at load time; a fragmented heap can still refuse a
            # window. Halving keeps the stage alive on a card the measured fit was too generous for,
            # with less multi-view context for those frames rather than no depth at all.
            if len(index) <= MIN_WINDOW * 2:
                raise
            torch.cuda.empty_cache()
            half = len(index) // 2
            print(f'Window of {len(index)} did not fit; halving')
            predict(index[:half + 1])
            return predict(index[half:])
        for slot, i in enumerate(index):
            # Windows overlap; average a frame's predictions (all share our cameras' scale).
            sums[i] += cv2.resize(prediction.depth[slot].astype(np.float32), (width, height),
                                  interpolation=cv2.INTER_LINEAR)
            counts[i] += 1

    with torch.inference_mode():
        for window in bar(windows(len(frames), a.window, a.step), desc='DA3 depth', unit='window'):
            predict(list(window))
    a.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for i, (number, rotation, translation, observations) in enumerate(frames):
        depth = sums[i] / max(counts[i], 1)
        mask = cv2.imread(str(a.sequence / 'masks' / f'{number:08d}.png'), cv2.IMREAD_GRAYSCALE)
        try:
            coords, target = object_anchors(observations, points, rotation, translation, mask)
            grounded, report = ground(depth, 'depth', coords, target, a.grounding_regions)
            rejected = report['relative_anchor_error'] > a.max_relative_error
        except ValueError as error:
            report, rejected = {'reason': str(error)}, True
        if a.raw:
            grounded, rejected = depth.astype(np.float32), False
        if not rejected:
            np.save(a.output_dir / f'{number:08d}_grounded.npy', grounded)
        reports.append({'frame': number, 'rejected': rejected, **report})
    kept = sum(not row['rejected'] for row in reports)
    summary = {'model': a.checkpoint, 'images': str(a.images) if a.images else 'working frames', 'head_weights': str(a.head_weights) if a.head_weights else None,
               'poses_given': not a.no_poses, 'window': a.window, 'step': a.step,
               'frames': len(reports), 'kept': kept,
               'median_scale': float(np.median([row['scale'] for row in reports if 'scale' in row])),
               'per_frame': reports}
    (a.output_dir / 'candidate.json').write_text(json.dumps(summary, indent=2) + '\n')
    done(f"DA3: {kept}/{len(reports)} frames grounded (fit scale {summary['median_scale']:.3f})")


if __name__ == '__main__':
    main()
