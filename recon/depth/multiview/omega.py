#!/usr/bin/env python3
"""VGGT-Omega multi-view depth, grounded to the object's sparse points.

VGGT-Omega (https://github.com/facebookresearch/vggt-omega, FAIR Noncommercial
Research License; weights gated on Hugging Face) is VGGT's successor. Frames run
in overlapping windows like the VGGT stage; each window's depth (balanced
resize, so 4:3 frames are resized without cropping) is resized back to the
working frames, overlapping predictions of a frame are averaged after each is
grounded, and the result is grounded to the object's sparse points like every
other candidate.
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
    p.add_argument('--checkpoint', default='vggt_omega_1b_512.pt', help='File in facebook/VGGT-Omega')
    p.add_argument('--resolution', type=int, default=512)
    p.add_argument('--window', type=int, default=16)
    p.add_argument('--step', type=int, default=8)
    p.add_argument('--max-relative-error', type=float, default=.03)
    a = p.parse_args()
    from huggingface_hub import hf_hub_download
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    intrinsics, points, frames = parse_model(a.colmap_model)
    points = point_table(points)
    with spinner('Loading VGGT-Omega'):
        model = VGGTOmega().to('cuda').eval()
        state = torch.load(hf_hub_download('facebook/VGGT-Omega', a.checkpoint), map_location='cpu')
        model.load_state_dict(state)
    paths = [str(a.sequence / 'frames' / f'{number:08d}.png') for number, *_ in frames]
    height, width = cv2.imread(paths[0]).shape[:2]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    sums = np.zeros((len(frames), height, width), np.float32)
    counts = np.zeros(len(frames), np.int32)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for window in bar(windows(len(frames), a.window, a.step), desc='VGGT-Omega depth', unit='window'):
            index = list(window)
            images = load_and_preprocess_images([paths[i] for i in index], image_resolution=a.resolution).to('cuda')
            depth = model(images)['depth'].float().reshape(len(index), *images.shape[-2:]).cpu().numpy()
            for slot, i in enumerate(index):
                sums[i] += cv2.resize(depth[slot], (width, height), interpolation=cv2.INTER_LINEAR)
                counts[i] += 1
    reports = []
    for i, (number, rotation, translation, observations) in enumerate(frames):
        prediction = sums[i] / max(counts[i], 1)
        mask = cv2.imread(str(a.sequence / 'masks' / f'{number:08d}.png'), cv2.IMREAD_GRAYSCALE)
        try:
            coords, target = object_anchors(observations, points, rotation, translation, mask)
            grounded, report = ground(prediction, 'depth', coords, target)
            rejected = report['relative_anchor_error'] > a.max_relative_error
        except ValueError as error:
            report, rejected = {'reason': str(error)}, True
        if not rejected:
            np.save(a.output_dir / f'{number:08d}_grounded.npy', grounded)
        reports.append({'frame': number, 'rejected': rejected, **report})
    kept = sum(not row['rejected'] for row in reports)
    summary = {'model': f'VGGT-Omega {a.checkpoint}', 'resolution': a.resolution, 'window': a.window,
               'step': a.step, 'frames': len(reports), 'kept': kept, 'per_frame': reports}
    (a.output_dir / 'candidate.json').write_text(json.dumps(summary, indent=2) + '\n')
    done(f'VGGT-Omega: {kept}/{len(reports)} frames grounded')


if __name__ == '__main__':
    main()
