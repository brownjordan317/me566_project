#!/usr/bin/env python3
"""Object TSDF with per-pixel confidence weights (dense GPU grid).

Open3D's VoxelBlockGrid weights every depth pixel equally. Here each pixel's
contribution to the truncated signed distance average (Curless and Levoy,
SIGGRAPH 1996) is weighted by how much it is trusted:

  agreement  exp(-(d_a - d_b)^2 / (sigma d)^2) for two independent depth
             models a, b at that pixel: where VGGT and DA3 agree the fused
             depth is likely right;
  angle      cos(angle between viewing ray and surface normal), floored:
             grazing views give noisier depth.

The grid covers only the object's bounds (landmarks plus a margin), so it can
be dense and fine. Every frame projects every voxel centre into the image
(nearest pixel). Voxels far behind the observed surface are skipped as
occluded. The surface is extracted by marching cubes (scikit-image) where the
at least --min-views frames observed it (weights change the average, not which
surface exists), coloured from the averaged image
colour, and written as a 32-bit PLY.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch

from recon.common.colmap import parse_model
from recon.common.ply import write_ply
from recon.common.progress import bar, done
from recon.depth.fusion.photometric import camera_normals
from recon.surface.tsdf import depth_validity


def object_box(landmarks: Path, margin: float) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(o3d.io.read_point_cloud(str(landmarks)).points)
    return np.quantile(points, .001, 0) - margin, np.quantile(points, .999, 0) + margin


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sequence', type=Path)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--depth-dir', type=Path, required=True, help='Depth to fuse (object pixels are used)')
    p.add_argument('--landmarks', type=Path, required=True, help='Object landmarks for the grid bounds')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--voxel', type=float, default=.02)
    p.add_argument('--truncation', type=float, default=.06,
                   help='Narrower than the scene TSDF (0.12): the weights keep noise down, so the band '
                        'can be tighter')
    p.add_argument('--margin', type=float, default=.2, help='Grid margin around the landmarks (map units)')
    p.add_argument('--agreement', nargs=2, type=Path, metavar=('DEPTH_A', 'DEPTH_B'),
                   help='Two independent depth models; weight pixels by their agreement')
    p.add_argument('--agreement-sigma', type=float, default=.01, help='Relative depth difference scale')
    p.add_argument('--angle-weight', action='store_true', help='Weight by viewing angle to the surface')
    p.add_argument('--min-views', type=int, default=3,
                   help='Frames that must observe a voxel for it to hold surface (unweighted, as in the '
                        'scene TSDF), so low confidence changes the average but never deletes surface')
    p.add_argument('--weight-floor', type=float, default=.05,
                   help='Smallest pixel weight, so every observation still counts a little')
    p.add_argument('--dilate', type=int, default=-1,
                   help='Use only object pixels, dilated by this many pixels; -1 (default) keeps the whole '
                        'image so background depth carves free space around the object')
    p.add_argument('--robust-passes', type=int, default=2,
                   help='2 or more: re-fuse, weighting each observation by its agreement with the '
                        'previous pass (iteratively reweighted averaging)')
    p.add_argument('--robust-scale', type=float, default=.5,
                   help='Disagreement scale for the robust passes, in truncation units')
    p.add_argument('--discontinuity-tau', type=float, default=.03)
    p.add_argument('--min-depth', type=float, default=.5)
    p.add_argument('--max-depth', type=float, default=30.)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    device = a.device
    intrinsics, _, frames = parse_model(a.model)
    low, high = object_box(a.landmarks, a.margin)
    shape = np.ceil((high - low) / a.voxel).astype(int) + 1
    if np.prod(shape) > 400e6:
        p.error(f'Grid {shape} too large; raise --voxel')
    axes = [torch.arange(n, device=device, dtype=torch.float32) * a.voxel + float(o) for n, o in zip(shape, low)]
    centres = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    previous = None
    for robust_pass in range(a.robust_passes):
        tsdf = torch.zeros(len(centres), device=device)
        weight = torch.zeros(len(centres), device=device)
        views = torch.zeros(len(centres), device=device, dtype=torch.int16)
        colour = torch.zeros(len(centres), 3, device=device)
        kernel = np.ones((2 * max(a.dilate, 0) + 1,) * 2, np.uint8)
        used = 0
        for number, rotation, translation, _ in bar(frames, desc=f'Weighted TSDF (pass {robust_pass + 1})', unit='frame'):
            path = a.depth_dir / f'{number:08d}_grounded.npy'
            if not path.is_file():
                continue
            depth = np.load(path).astype(np.float32)
            # Same pixel filter as the scene TSDF: drop depth jumps (flying pixels) and out-of-range depth.
            depth = np.where(depth_validity(depth, a.min_depth, a.max_depth, a.discontinuity_tau), depth, 0).astype(np.float32)
            mask = cv2.imread(str(a.sequence / 'masks' / f'{number:08d}.png'), cv2.IMREAD_GRAYSCALE) > 127
            if a.dilate >= 0:
                depth = np.where(cv2.dilate(mask.astype(np.uint8), kernel) > 0, depth, 0)
            K = intrinsics[number]
            pixel_weight = (depth > 0).astype(np.float32)
            if a.agreement:
                first, second = (np.load(d / f'{number:08d}_grounded.npy') if (d / f'{number:08d}_grounded.npy').is_file()
                                 else None for d in a.agreement)
                if first is not None and second is not None:
                    both = (first > 0) & (second > 0)
                    relative = np.where(both, np.log(np.maximum(first, 1e-6) / np.maximum(second, 1e-6)), 1.)
                    pixel_weight *= np.exp(-(relative / a.agreement_sigma) ** 2)
            if a.angle_weight:
                normals = camera_normals(depth, K)
                height, width = depth.shape
                v, u = np.mgrid[:height, :width].astype(np.float32)
                rays = np.stack(((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)), -1)
                rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
                pixel_weight *= np.clip(-(normals * rays).sum(-1), .1, 1)
            image = cv2.cvtColor(cv2.imread(str(a.sequence / 'frames' / f'{number:08d}.png')), cv2.COLOR_BGR2RGB)
            D, Wt = torch.as_tensor(depth, device=device), torch.as_tensor(pixel_weight, device=device)
            C = torch.as_tensor(image, device=device, dtype=torch.float32) / 255
            R, t, Kt = (torch.as_tensor(x, dtype=torch.float32, device=device) for x in (rotation, translation, K))
            camera = centres @ R.T + t
            z = camera[:, 2]
            u = torch.round(Kt[0, 0] * camera[:, 0] / z.clamp(min=1e-6) + Kt[0, 2]).long()
            v = torch.round(Kt[1, 1] * camera[:, 1] / z.clamp(min=1e-6) + Kt[1, 2]).long()
            height, width = depth.shape
            inside = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            index = torch.nonzero(inside).squeeze(1)
            observed = D[v[index], u[index]]
            sdf = observed - z[index]
            keep = (observed > 0) & (sdf > -a.truncation)
            index, sdf = index[keep], sdf[keep]
            w = Wt[v[index], u[index]].clamp(min=a.weight_floor)
            value = torch.clamp(sdf / a.truncation, max=1.)
            if previous is not None:
                # Robust pass: observations far from the previous estimate count less, so a few
                # off frames cannot pull a surface that accurate frames agree on.
                w = w * torch.exp(-((value - previous[index]) / a.robust_scale) ** 2).clamp(min=a.weight_floor)
            total = weight[index] + w
            tsdf[index] = torch.where(total > 0, (tsdf[index] * weight[index] + value * w) / total.clamp(min=1e-9), tsdf[index])
            colour[index] = torch.where(total[:, None] > 0, (colour[index] * weight[index, None]
                                        + C[v[index], u[index]] * w[:, None]) / total.clamp(min=1e-9)[:, None], colour[index])
            weight[index] = total
            views[index] += 1
            used += 1
        if robust_pass + 1 < a.robust_passes:
            previous = tsdf.clone()
    from skimage.measure import marching_cubes
    volume = tsdf.reshape(*shape).cpu().numpy()
    valid = (views.reshape(*shape) >= a.min_views).cpu().numpy()
    vertices, faces, _, _ = marching_cubes(volume, level=0., mask=valid, allow_degenerate=False)
    vertices = vertices * a.voxel + low
    grid = colour.reshape(*shape, 3).cpu().numpy()
    nearest = np.clip(np.round((vertices - low) / a.voxel).astype(int), 0, shape - 1)
    colours = grid[nearest[:, 0], nearest[:, 1], nearest[:, 2]]
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces[:, ::-1].copy()))
    mesh.compute_vertex_normals()
    write_ply(a.output, vertices, colours, np.asarray(mesh.vertex_normals), faces[:, ::-1])
    report = {'method': 'confidence-weighted dense object TSDF', 'grid': shape.tolist(), 'voxel': a.voxel,
              'truncation': a.truncation, 'agreement': bool(a.agreement), 'agreement_sigma': a.agreement_sigma,
              'angle_weight': a.angle_weight, 'min_views': a.min_views,
              'weight_floor': a.weight_floor, 'robust_passes': a.robust_passes,
              'robust_scale': a.robust_scale, 'frames': used,
              'faces': int(len(faces))}
    a.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    done(f"Weighted TSDF: {used} frames, grid {'x'.join(map(str, shape))}, {len(faces):,} faces")


if __name__ == '__main__':
    main()
