#!/usr/bin/env python3
"""Infer offline VGGT multiview depth and ground it to COLMAP feature tracks.

Uses VGGT's depth branch; COLMAP intrinsics and poses remain authoritative for
fusion. Overlapping windows provide context, while each frame is saved once.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

from recon.common.colmap import parse_model
from recon.common.progress import bar, done, spinner
from recon.depth.grounding import object_anchors, fit_regionwise


def unpad_depth(depth: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
    """Reverse VGGT's mode='pad' resize and centered padding.

    Inverts load_and_preprocess_images(mode='pad') in VGGT
    (vggt/utils/load_fn.py, https://github.com/facebookresearch/vggt; VGGT
    License): longest side resized to 518 px, the other rounded to a multiple
    of 14, then padded to a centred square.
    """
    height, width = original_shape
    side = 518
    if width >= height:
        resized_width = side
        resized_height = round(height * side / width / 14) * 14
    else:
        resized_height = side
        resized_width = round(width * side / height / 14) * 14
    top = (side - resized_height) // 2
    left = (side - resized_width) // 2
    crop = depth[top:top + resized_height, left:left + resized_width]
    if crop.shape != (resized_height, resized_width):
        raise ValueError('Unexpected VGGT depth shape')
    return cv2.resize(crop.astype(np.float32), (width, height),
                      interpolation=cv2.INTER_LINEAR)


def main(argv: list[str] | None = None, network=None):
    """Run the stage; `network` is an already-loaded VGGT to reuse (recon.depth.batch), and the loaded one is
    returned so the next job can."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sequence', type=Path)
    p.add_argument('--model', type=Path, required=True,
                   help='Registered COLMAP sparse_txt model')
    p.add_argument('--grounding-regions', type=int, default=1,
                   help='Affine-fit the depth in this many regions of position and predicted depth, blended across the image, instead of one global fit per frame (1 keeps the global fit)')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--checkpoint-dir', type=Path,
                   help='Local Hugging Face VGGT-1B snapshot; default cached model')
    p.add_argument('--window', type=int, default=16)
    p.add_argument('--step', type=int, default=8)
    p.add_argument('--max-frames', type=int)
    p.add_argument('--images', type=Path,
                   help='Full-resolution photos (numbered files) to feed VGGT instead of the working frames')
    args = p.parse_args(argv)
    if args.window < 2 or args.step < 1 or args.step > args.window:
        p.error('Need window >= 2 and 1 <= step <= window')
    if args.max_frames is not None and args.max_frames < 2:
        p.error('Need at least two frames')
    if not torch.cuda.is_available():
        p.error('VGGT requires CUDA for this pipeline')
    # VGGT itself calls the deprecated torch.cuda.amp.autocast; not actionable here.
    warnings.filterwarnings('ignore', category=FutureWarning, module='vggt')
    from huggingface_hub import snapshot_download
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    checkpoint = args.checkpoint_dir or Path(snapshot_download(
        'facebook/VGGT-1B', local_files_only=True))
    if not checkpoint.is_dir():
        p.error(f'Missing VGGT checkpoint directory: {checkpoint}')
    _, points, frames = parse_model(args.model)
    frames = [frame for frame in frames
              if (args.sequence / 'frames' / f'{frame[0]:08d}.png').is_file()]
    if args.max_frames:
        frames = frames[:args.max_frames]
    if len(frames) < 2:
        p.error('Need at least two registered frames')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_s = 0.
    if network is None:
        started = time.perf_counter()
        # huggingface_hub prints 'Loading weights from local directory' to stdout.
        with spinner('Loading VGGT'), contextlib.redirect_stdout(io.StringIO()):
            network = VGGT.from_pretrained(str(checkpoint)).cuda().eval()
        # Camera/point/track heads are unnecessary for this depth comparison.
        network.camera_head = None
        network.point_head = None
        network.track_head = None
        torch.cuda.synchronize()
        load_s = time.perf_counter() - started
    inference_started = time.perf_counter()
    # Same aspect ratio as the working frames, so unpad_depth maps the prediction back unchanged.
    full = {int(path.stem): path for path in args.images.iterdir() if path.stem.isdigit()} if args.images else {}
    reports = []
    progress = bar(total=len(frames), desc='VGGT depth', unit='frame')
    for first in range(0, len(frames), args.step):
        begin = max(0, min(first - (args.window - args.step) // 2,
                           len(frames) - args.window))
        end = min(len(frames), begin + args.window)
        paths = [full[row[0]] if args.images else args.sequence / 'frames' / f'{row[0]:08d}.png'
                 for row in frames[begin:end]]
        images = load_and_preprocess_images([str(path) for path in paths],
                                            mode='pad').cuda()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            predicted = network(images)['depth'][0, ..., 0].float().cpu().numpy()
        del images
        for index in range(first, min(first + args.step, len(frames))):
            number, rotation, translation, observations = frames[index]
            image = cv2.imread(str(args.sequence / 'frames' / f'{number:08d}.png'))
            mask = cv2.imread(str(args.sequence / 'masks' / f'{number:08d}.png'),
                              cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None or image.shape[:2] != mask.shape:
                raise ValueError(f'Missing/mismatched image or mask: {number}')
            depth = unpad_depth(predicted[index - begin], image.shape[:2])
            try:
                coords, target = object_anchors(observations, points,
                                                rotation, translation, mask)
                grounded, fit = fit_regionwise(depth, coords, target, args.grounding_regions)
                if fit['scale_share'] < .5:
                    raise ValueError(f'Degenerate depth fit: {fit["scale_share"]:.3f}')
            except ValueError as error:
                reports.append({'frame': number, 'saved': False, 'reason': str(error)})
                continue
            np.save(args.output_dir / f'{number:08d}_grounded.npy', grounded)
            reports.append({'frame': number, 'saved': True, **fit})
        progress.update(min(first + args.step, len(frames)) - first)
        torch.cuda.empty_cache()
    progress.close()
    summary = {'method': 'VGGT-1B multiview depth, affine COLMAP object-track grounding',
               'window': args.window, 'step': args.step,
               'images': str(args.images) if args.images else 'working frames',
               'registered_frames': len(frames),
               'saved_frames': sum(bool(row['saved']) for row in reports),
               'checkpoint': str(checkpoint), 'load_s': load_s, 'model_reused': load_s == 0.,
               'inference_s': time.perf_counter() - inference_started, 'reports': reports}
    (args.output_dir / 'vggt_inference.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    done(f"VGGT depth: {summary['saved_frames']}/{len(frames)} frames grounded")
    return network


if __name__ == '__main__':
    main()
