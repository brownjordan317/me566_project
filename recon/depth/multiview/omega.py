#!/usr/bin/env python3
"""VGGT-Omega multi-view depth, grounded to the object's sparse points.

VGGT-Omega (https://github.com/facebookresearch/vggt-omega, FAIR Noncommercial
Research License; weights gated on Hugging Face) predicts depth for a sequence.
Frames run in windows up to 128 by default, so benchmark scenes fit in one pass.
Balanced resizing preserves their aspect ratio. For longer sequences, each
window is grounded to object anchors before overlapping predictions are
averaged and regrounded.
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
from recon.depth.grounding import ground, object_anchors, point_table
from recon.depth.multiview.da3 import windows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sequence', type=Path)
    p.add_argument('--colmap-model', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--masks-dir', type=Path, help='Object masks; default SEQUENCE/masks')
    p.add_argument('--checkpoint', default='vggt_omega_1b_416_reproduce.pt', help='File in facebook/VGGT-Omega')
    p.add_argument('--resolution', type=int, default=416)
    p.add_argument('--resize-mode', choices=('balanced', 'max_size'), default='balanced')
    p.add_argument('--window', type=int, default=128)
    p.add_argument('--step', type=int, default=64)
    p.add_argument('--grounding-regions', type=int, default=1,
                   help='Affine-fit the depth in this many regions of position and predicted depth, blended across the image, instead of one global fit per frame (1 keeps the global fit)')
    p.add_argument('--max-relative-error', type=float, default=.03)
    p.add_argument('--images', type=Path,
                   help='Full-resolution photos (numbered files) to feed the model instead of the working frames')
    a = p.parse_args()
    from huggingface_hub import hf_hub_download
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    intrinsics, points, frames = parse_model(a.colmap_model)
    masks_dir = a.masks_dir or a.sequence / 'masks'
    points = point_table(points)
    with spinner('Loading VGGT-Omega'):
        model = VGGTOmega().to('cuda').eval()
        try:
            checkpoint = hf_hub_download('facebook/VGGT-Omega', a.checkpoint, local_files_only=True)
        except FileNotFoundError:
            checkpoint = hf_hub_download('facebook/VGGT-Omega', a.checkpoint)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
    paths = [str(a.sequence / 'frames' / f'{number:08d}.png') for number, *_ in frames]
    height, width = cv2.imread(paths[0]).shape[:2]
    if a.images is not None:
        # Same aspect ratio as the working frames; predictions are resized back to them.
        full = {int(path.stem): path for path in a.images.iterdir() if path.stem.isdigit()}
        paths = [str(full[number]) for number, *_ in frames]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    anchors = []
    for number, rotation, translation, observations in frames:
        mask = cv2.imread(str(masks_dir / f'{number:08d}.png'), cv2.IMREAD_GRAYSCALE)
        try:
            if mask is None or mask.shape != (height, width):
                raise ValueError(f'Missing/mismatched mask: {number}')
            anchors.append(object_anchors(observations, points, rotation, translation, mask))
        except ValueError as error:
            anchors.append(error)
    sums = np.zeros((len(frames), height, width), np.float32)
    counts = np.zeros(len(frames), np.int32)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for window in bar(windows(len(frames), a.window, a.step), desc='VGGT-Omega depth', unit='window'):
            index = list(window)
            images = load_and_preprocess_images([paths[i] for i in index], mode=a.resize_mode, image_resolution=a.resolution).to('cuda')
            depth = model(images)['depth'].float().reshape(len(index), *images.shape[-2:]).cpu().numpy()
            for slot, i in enumerate(index):
                if isinstance(anchors[i], ValueError):
                    continue
                prediction = cv2.resize(depth[slot], (width, height), interpolation=cv2.INTER_LINEAR)
                coords, target = anchors[i]
                try:
                    grounded, report = ground(prediction, 'depth', coords, target, a.grounding_regions)
                except ValueError:
                    continue
                if report['relative_anchor_error'] <= a.max_relative_error:
                    sums[i] += grounded
                    counts[i] += 1
            del images, depth
    reports = []
    for i, (number, *_rest) in enumerate(frames):
        try:
            if isinstance(anchors[i], ValueError):
                raise anchors[i]
            if not counts[i]:
                raise ValueError('No valid grounded window')
            coords, target = anchors[i]
            grounded, report = ground(sums[i] / counts[i], 'depth', coords, target, a.grounding_regions)
            rejected = report['relative_anchor_error'] > a.max_relative_error
        except ValueError as error:
            report, rejected = {'reason': str(error)}, True
        if not rejected:
            np.save(a.output_dir / f'{number:08d}_grounded.npy', grounded)
        reports.append({'frame': number, 'windows': int(counts[i]), 'rejected': rejected, **report})
    kept = sum(not row['rejected'] for row in reports)
    summary = {'model': f'VGGT-Omega {a.checkpoint}', 'resolution': a.resolution, 'resize_mode': a.resize_mode, 'window': a.window,
               'step': a.step, 'images': str(a.images) if a.images else 'working frames', 'frames': len(reports), 'kept': kept, 'per_frame': reports}
    (a.output_dir / 'candidate.json').write_text(json.dumps(summary, indent=2) + '\n')
    done(f'VGGT-Omega: {kept}/{len(reports)} frames grounded')


if __name__ == '__main__':
    main()
