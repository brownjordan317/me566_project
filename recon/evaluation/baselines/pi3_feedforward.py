#!/usr/bin/env python3
"""Feed-forward baseline: π³X's own cameras and depth, fused with our surface stages.

One π³X forward pass over the sequence (all frames, or --max-frames spread evenly) predicts every camera pose,
camera rays and a local point map per frame (Wang et al., "π³: Scalable Permutation-Equivariant Visual Geometry
Learning", 2025; https://github.com/yyfz233/Pi3, code BSD-3-Clause, Pi3X weights CC BY-NC 4.0, comparison only).
Nothing from COLMAP, MoGe, VGGT, DA3 or our fusion is used. Depth is the local points' z; each frame's
intrinsics are recovered from its rays with π³'s recover_intrinsic_from_rays_d, then rescaled from the model's
grid (images resized to about 255k pixels, sides multiples of 14, as its load_images_as_tensor) to the working
frames. Everything after that - rescaling to --normalize-depth, the COLMAP text model, the scene TSDF, mask-vote
extraction and cleanup - is the VGGT feed-forward baseline's (recon.evaluation.baselines.vggt_feedforward).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch

from recon.common.progress import done, spinner
from recon.evaluation.baselines.vggt_feedforward import landmarks, write_model


def model_grid(width: int, height: int, pixels: int = 255000) -> tuple[int, int]:
    """π³'s uniform input size (pi3/utils/basic.py load_images_as_tensor): about `pixels`, sides multiples of 14."""
    scale = math.sqrt(pixels / (width * height))
    k, m = round(width * scale / 14), round(height * scale / 14)
    while (k * 14) * (m * 14) > pixels:
        if k / m > width / height:
            k -= 1
        else:
            m -= 1
    return max(1, k) * 14, max(1, m) * 14


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('sequence', type=Path, help='Prepared sequence with frames/ and masks/')
    p.add_argument('--masks', type=Path, help='Mask folder (default: sequence/masks)')
    p.add_argument('--output', type=Path, required=True, help='Run folder (poses/, depth/, work/, mesh/)')
    p.add_argument('--max-frames', type=int, default=96, help='Frames in the single forward pass (24 GB GPU)')
    p.add_argument('--normalize-depth', type=float, default=6.5)
    p.add_argument('--checkpoint', default='yyfz233/Pi3X')
    a = p.parse_args()
    from pi3.models.pi3x import Pi3X
    from pi3.utils.geometry import recover_intrinsic_from_rays_d

    masks_dir = a.masks or a.sequence / 'masks'
    paths = sorted((a.sequence / 'frames').glob('*.png'))
    if len(paths) > a.max_frames:
        paths = [paths[i] for i in np.unique(np.linspace(0, len(paths) - 1, a.max_frames).round().astype(int))]
    frames = [int(path.stem) for path in paths]
    height, width = cv2.imread(str(paths[0])).shape[:2]
    grid_w, grid_h = model_grid(width, height)
    images = np.stack([cv2.resize(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB), (grid_w, grid_h),
                                  interpolation=cv2.INTER_AREA) for path in paths]).astype(np.float32) / 255
    with spinner('Loading π³X'):
        network = Pi3X.from_pretrained(a.checkpoint).eval()
        network.disable_multimodal()
        network = network.cuda()
    tensor = torch.from_numpy(images).permute(0, 3, 1, 2)[None].cuda()
    with spinner(f'π³X forward pass over {len(paths)} frames'), torch.no_grad(), \
            torch.autocast('cuda', dtype=torch.bfloat16):
        predicted = network(imgs=tensor)
    local = predicted['local_points'][0].float()
    # Each pixel's local point is along its camera ray, so the points themselves give the intrinsics.
    K_grid = recover_intrinsic_from_rays_d(local).cpu().numpy()
    depth_grid = local[..., 2].cpu().numpy()
    c2w = predicted['camera_poses'][0].float().cpu().numpy()
    del predicted, tensor, network, local
    torch.cuda.empty_cache()
    sx, sy = width / grid_w, height / grid_h
    Ks, depths, poses = {}, {}, {}
    for i, frame in enumerate(frames):
        K = K_grid[i].astype(np.float64).copy()
        K[0, 0] *= sx
        K[0, 2] = (K[0, 2] + .5) * sx - .5
        K[1, 1] *= sy
        K[1, 2] = (K[1, 2] + .5) * sy - .5
        Ks[frame] = K
        depths[frame] = cv2.resize(depth_grid[i], (width, height), interpolation=cv2.INTER_LINEAR)
        poses[frame] = np.linalg.inv(c2w[i])
    scale = a.normalize_depth / float(np.median(np.concatenate([d[d > 0] for d in depths.values()])))
    masks = {frame: cv2.imread(str(masks_dir / f'{frame:08d}.png'), cv2.IMREAD_GRAYSCALE) > 127 for frame in frames}
    poses_dir, depth_dir = a.output / 'poses', a.output / 'depth/pi3_feedforward'
    depth_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        poses[frame][:3, 3] *= scale
        depths[frame] = np.where(depths[frame] > 0, depths[frame] * scale, 0).astype(np.float32)
        np.save(depth_dir / f'{frame:08d}_grounded.npy', depths[frame])
    write_model(poses_dir, frames, [Ks[f] for f in frames], [poses[f] for f in frames], width, height)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(landmarks(depths, masks, Ks, poses)))
    (poses_dir / 'export_object').mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(poses_dir / 'export_object/landmarks.ply'), cloud)
    (a.output / 'summary_pi3.json').write_text(json.dumps(
        {'frames': len(frames), 'grid': [grid_w, grid_h], 'scale_to_map': scale, 'checkpoint': a.checkpoint}, indent=2) + '\n')
    model, trajectory = poses_dir / 'sparse_txt', poses_dir / 'export/trajectory.csv'
    work, mesh = a.output / 'work', a.output / 'mesh'
    run = lambda *args: subprocess.run([sys.executable, '-m', *map(str, args)], check=True)
    run('recon.surface.tsdf', a.sequence, '--model', model, '--trajectory', trajectory, '--depth-dir', depth_dir,
        '--output', work / 'scene.ply', '--min-views', 3)
    run('recon.surface.object', work / 'scene.ply', '--bounds-cloud', poses_dir / 'export_object/landmarks.ply',
        '--masks-dir', masks_dir, '--depth-dir', depth_dir, '--trajectory', trajectory, '--model', model,
        '--output', work / 'object.ply', '--min-ratio', .8, '--min-component-support', .25)
    run('recon.surface.clean', work / 'object.ply', mesh / 'object.ply', '--hole-size', .08)
    done(f'π³X feed-forward baseline: {len(frames)} frames -> {mesh / "object.ply"}')


if __name__ == '__main__':
    main()
