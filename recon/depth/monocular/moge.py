#!/usr/bin/env python3
"""Infer MoGe depth per image and align it to masked COLMAP tracks.

Saved *_grounded.npy maps feed the cloud, multiview refinement, and TSDF stages.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from recon.common.colmap import parse_model
from recon.common.progress import bar, done, spinner
from recon.depth.grounding import fit_metric_depth, object_anchors, point_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sequence', type=Path)
    parser.add_argument('--model', type=Path, required=True,
                        help='COLMAP sparse_txt with the calibrated per-image cameras')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path,
                        help='Local checkpoint; defaults to the cached MoGe-3 ViT-L weights')
    parser.add_argument('--num-tokens', type=int, default=2500)
    parser.add_argument('--version', choices=('v2', 'v3'), default='v3')
    parser.add_argument('--refine-steps', type=int, default=3)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--images', type=Path,
                        help='Full-resolution photos (numbered files) to feed MoGe instead of the working frames')
    parser.add_argument('--anchor-region', choices=('object', 'all'), default='object')
    parser.add_argument('--min-scale-share', type=float, default=.5,
                        help='Discard frames whose fit is mostly a constant depth')
    parser.add_argument('--reuse-raw-dir', type=Path,
                        help='Reuse cached raw MoGe maps and skip model inference')
    args = parser.parse_args()
    if args.checkpoint is None and args.version != 'v3':
        parser.error('--checkpoint is required for the MoGe-2 comparison')
    if args.checkpoint is None:
        from huggingface_hub import hf_hub_download
        try:
            args.checkpoint = Path(hf_hub_download(
                'Ruicheng/moge-3-vitl', 'model.pt', local_files_only=True))
        except Exception as error:
            parser.error(f'MoGe-3 weights are not cached: {error}')
    if not args.checkpoint.is_file():
        parser.error(f'Missing local MoGe checkpoint: {args.checkpoint}')
    model = None
    if args.reuse_raw_dir is None:
        if args.version == 'v3':
            from moge.model.v3 import MoGeModel
        else:
            from moge.model.v2 import MoGeModel
        torch.set_grad_enabled(False)
        with spinner('Loading MoGe'):
            model = MoGeModel.from_pretrained(args.checkpoint).to('cuda').eval()
    intrinsics, points, frames = parse_model(args.model)
    points = point_table(points)
    if args.max_frames is not None:
        frames = frames[:args.max_frames]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def load(number):
        if args.reuse_raw_dir is not None:
            raw = np.load(args.reuse_raw_dir / f'{number:08d}_moge_raw.npy',
                          allow_pickle=False)
        else:
            raw = None
        frame = full[number] if args.images else args.sequence / 'frames' / f'{number:08d}.png'
        return (cv2.imread(str(frame)),
                cv2.imread(str(args.sequence / 'masks' / f'{number:08d}.png'),
                           cv2.IMREAD_GRAYSCALE), raw)

    full = {int(path.stem): path for path in args.images.iterdir() if path.stem.isdigit()} if args.images else {}

    # Disk reads and writes overlap GPU inference; results are unchanged.
    reader = ThreadPoolExecutor(4)
    writer = ThreadPoolExecutor(4)
    prefetch = 8
    loads = [reader.submit(load, frame[0]) for frame in frames[:prefetch]]
    saves = []
    reports = []
    region = 'scene' if args.anchor_region == 'all' else args.anchor_region
    label = f'MoGe grounding ({region})' if args.reuse_raw_dir is not None else 'MoGe depth'
    for index, (number, rotation, translation, observations) in enumerate(
            bar(frames, desc=label, unit='frame')):
        if index + prefetch < len(frames):
            loads.append(reader.submit(load, frames[index + prefetch][0]))
        image, mask, raw = loads[index].result()
        loads[index] = None
        if (image is None or mask is None
                or abs(image.shape[1] / image.shape[0] - mask.shape[1] / mask.shape[0]) > 1e-3):
            raise ValueError(f'Missing/mismatched frame and mask {number}')
        height, width = mask.shape                         # the working frame
        focal = intrinsics[number][0, 0]
        fov_x = float(np.degrees(2 * np.arctan(width / (2 * focal))))
        if raw is None:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb.copy()).to('cuda', dtype=torch.float32)
            tensor = tensor.permute(2, 0, 1) / 255.
            infer_options = {'fov_x': fov_x, 'num_tokens': args.num_tokens,
                             'apply_mask': False}
            if args.version == 'v3':
                infer_options['refine_steps'] = args.refine_steps
            prediction = model.infer(tensor, **infer_options)
            raw = prediction['depth'].float().cpu().numpy()
            if raw.shape != (height, width):
                # Full-resolution input: the field of view is the same, so only the grid changes.
                raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)
        if raw.shape != (height, width):
            raise ValueError(f'MoGe shape mismatch in frame {number}: {raw.shape}')
        anchor_mask = (mask if args.anchor_region == 'object'
                       else np.full_like(mask, 255))
        if args.reuse_raw_dir is None:
            # Only the inference directory's raw maps are ever read back.
            saves.append(writer.submit(np.save, args.output_dir / f'{number:08d}_moge_raw.npy',
                                       raw.astype('float32')))
        try:
            coords, target = object_anchors(observations, points, rotation,
                                            translation, anchor_mask)
            grounded, report = fit_metric_depth(raw, coords, target)
            rejected = report['scale_share'] < args.min_scale_share
        except ValueError as error:
            # Too few sparse points to ground this frame: drop it like a
            # low-scale-share frame instead of failing the whole scene.
            report, rejected = {'reason': str(error)}, True
        if not rejected:
            saves.append(writer.submit(np.save, args.output_dir / f'{number:08d}_grounded.npy',
                                       grounded))
        report.update(frame=number, fov_x_deg=fov_x, rejected=bool(rejected),
                      object_pixels=int(np.count_nonzero(mask)))
        reports.append(report)
    reader.shutdown()
    for save in saves:
        save.result()
    writer.shutdown()
    summary = {'model': f'MoGe-{args.version[1:]}', 'checkpoint': str(args.checkpoint),
               'frames': len(reports), 'num_tokens': args.num_tokens,
               'min_scale_share': args.min_scale_share,
               'rejected_frames': [row['frame'] for row in reports if row['rejected']],
               'refine_steps': args.refine_steps if args.version == 'v3' else 0,
               'anchor_region': args.anchor_region,
               'reused_raw': args.reuse_raw_dir is not None,
               'median_anchor_error': float(np.median(
                   [row['median_anchor_error'] for row in reports if not row['rejected']]
                   or [np.nan])),
               'per_frame': reports}
    (args.output_dir / 'moge_inference.json').write_text(json.dumps(summary, indent=2) + '\n')
    kept = sum(not row['rejected'] for row in reports)
    done(f'{label}: {kept}/{len(reports)} frames kept, median anchor error '
         f"{summary['median_anchor_error']:.3f} map units")


if __name__ == '__main__':
    main()
