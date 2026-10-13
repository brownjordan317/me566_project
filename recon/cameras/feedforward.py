#!/usr/bin/env python3
"""Cameras from one feed-forward pass of VGGT-Omega, in the pipeline's camera formats.

VGGT-Omega (https://github.com/facebookresearch/vggt-omega; FAIR Noncommercial
Research License, weights gated on Hugging Face) predicts every frame's pose
and field of view at once. Its 9-D encoding (translation, quaternion, vertical
and horizontal field of view; world-to-camera, OpenCV) is decoded with the
package's own encoding_to_camera; the principal point is the image centre. Its
'balanced' preprocessing resizes each frame to a patch grid (16 px) holding about
--resolution^2 pixels, so focal lengths are rescaled per axis back to the
working frames. Up to --max-frames frames run in one pass; beyond that the
frames are subsampled evenly (only those get cameras).

Output (--output, a poses folder): sparse_txt/ (one PINHOLE camera per frame, no
points) and export/trajectory.csv, as written by the pipeline's camera stage, so
the pose metrics and later stages can read them.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from recon.common.progress import done, spinner
from recon.evaluation.baselines.vggt_feedforward import write_model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('sequence', type=Path)
    p.add_argument('--output', type=Path, required=True, help='Poses folder (sparse_txt/, export/)')
    p.add_argument('--checkpoint', default='vggt_omega_1b_512.pt', help='File in facebook/VGGT-Omega')
    p.add_argument('--resolution', type=int, default=512)
    p.add_argument('--max-frames', type=int, default=128)
    p.add_argument('--images', type=Path,
                   help='Full-resolution photos (numbered files) to feed the model instead of the working frames')
    p.add_argument('--register-only', action='store_true',
                   help='Exchange information between frames only through register tokens in every block '
                        '(the released model uses full global attention in most blocks)')
    a = p.parse_args()
    from huggingface_hub import hf_hub_download
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera
    paths = sorted((a.sequence / 'frames').glob('*.png'))
    if len(paths) > a.max_frames:
        paths = [paths[i] for i in np.unique(np.linspace(0, len(paths) - 1, a.max_frames).round().astype(int))]
    height, width = cv2.imread(str(paths[0])).shape[:2]          # working frames: the cameras' size
    if a.images is not None:
        full = {int(path.stem): path for path in a.images.iterdir() if path.stem.isdigit()}
        paths = [full[int(path.stem)] for path in paths]
    with spinner('Loading VGGT-Omega'):
        model = VGGTOmega().to('cuda').eval()
        state = torch.load(hf_hub_download('facebook/VGGT-Omega', a.checkpoint), map_location='cpu', weights_only=True)
        model.load_state_dict(state)
    if a.register_only:
        # The paper's register-only variant (about 6% of the FLOPs, accuracy near the original VGGT)
        # was trained that way; switching the released weights at inference is an experiment.
        model.aggregator.inter_frame_attention_types = ['register'] * len(model.aggregator.inter_frame_attention_types)
    torch.cuda.reset_peak_memory_stats()
    images = load_and_preprocess_images([str(path) for path in paths], mode='balanced',
                                        image_resolution=a.resolution).to('cuda')
    size = images.shape[-2:]
    started = time.time()
    with spinner(f'VGGT-Omega cameras for {len(paths)} frames'), torch.inference_mode(), \
            torch.autocast('cuda', dtype=torch.bfloat16):
        encoding = model(images)['pose_enc'].float()
    torch.cuda.synchronize()
    seconds, peak = time.time() - started, torch.cuda.max_memory_allocated() / 2 ** 30
    extrinsics, intrinsics = encoding_to_camera(encoding, size)
    extrinsics, intrinsics = extrinsics[0].cpu().numpy(), intrinsics[0].cpu().numpy()
    sx, sy = width / size[1], height / size[0]
    Ks, poses = [], []
    for E, K in zip(extrinsics, intrinsics):
        K = K.astype(np.float64).copy()
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2], K[1, 2] = (K[0, 2] + .5) * sx - .5, (K[1, 2] + .5) * sy - .5
        pose = np.eye(4)
        pose[:3, :4] = E
        Ks.append(K)
        poses.append(pose)
    frames = [int(path.stem) for path in paths]
    write_model(a.output, frames, Ks, poses, width, height)
    focal = np.array([K[0, 0] for K in Ks])
    report = {'model': f'VGGT-Omega {a.checkpoint}', 'resolution': a.resolution, 'frames': len(frames),
              'processed_size': list(size), 'register_only': a.register_only, 'forward_s': seconds,
              'peak_gib': peak, 'median_fx': float(np.median(focal)),
              'focal_spread_pct': float((focal.max() / focal.min() - 1) * 100)}
    (a.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    done(f"VGGT-Omega cameras: {len(frames)} frames, median focal {report['median_fx']:.1f} px")


if __name__ == '__main__':
    main()
