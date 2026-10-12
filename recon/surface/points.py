#!/usr/bin/env python3
"""Export coloured scene and object point clouds from the fused depth maps.

Every valid depth pixel is back-projected with its image colour and a normal
taken from the depth map, facing its camera. A pixel survives only if at least
--min-views reference views see a surface at the same depth, the same rule
that keeps one-view slivers out of the scene mesh. Survivors are averaged into
voxels, isolated points are dropped, and the object cloud is selected by the
same visible mask vote that extracts the object mesh.

Requiring several views to agree on a pixel's depth before keeping it is the
same rule as COLMAP's stereo fusion (min_num_pixels; Schoenberger et al.,
ECCV 2016); this is an independent GPU implementation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch

from recon.common.camera import frame_intrinsics
from recon.common.ply import write_ply
from recon.common.progress import bar, done, spinner
from recon.common.trajectory import load_poses
from recon.depth.consistency import project_votes
from recon.surface.object import object_bounds, visible_mask_votes
from recon.surface.tsdf import depth_validity


def voxel_average(points: torch.Tensor, attributes: torch.Tensor, voxel: float
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean position and attributes of the points falling in each voxel."""
    keys = torch.floor(points / voxel).long()
    _, inverse, counts = torch.unique(keys, dim=0, return_inverse=True, return_counts=True)
    values = torch.cat((points, attributes), 1)
    sums = torch.zeros(len(counts), values.shape[1], device=values.device, dtype=values.dtype)
    sums.index_add_(0, inverse, values)
    means = sums / counts[:, None]
    return means[:, :3], means[:, 3:]


def depth_normals(camera: torch.Tensor) -> torch.Tensor:
    """Per-pixel camera-space normals of an HxWx3 point map, facing the camera."""
    dx = torch.zeros_like(camera)
    dy = torch.zeros_like(camera)
    dx[:, 1:-1] = camera[:, 2:] - camera[:, :-2]
    dy[1:-1] = camera[2:] - camera[:-2]
    normals = torch.nn.functional.normalize(torch.cross(dx, dy, dim=2), dim=2)
    flip = (normals * camera).sum(2, keepdim=True) > 0
    return torch.where(flip, -normals, normals)


def frame_points(frame: int, depth: torch.Tensor, color: torch.Tensor, K: torch.Tensor,
                 rotation: torch.Tensor, center: torch.Tensor, depths: dict,
                 intrinsics: dict, poses: dict, references: list[int], min_views: int,
                 tolerance: float, voxel: float) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = depth.shape
    v, u = torch.meshgrid(torch.arange(height, device=depth.device),
                          torch.arange(width, device=depth.device), indexing='ij')
    camera = torch.stack(((u - K[0, 2]) * depth / K[0, 0],
                          (v - K[1, 2]) * depth / K[1, 1], depth), 2)
    normals = depth_normals(camera)
    valid = (depth > 0) & torch.isfinite(normals).all(2) & (normals.abs().sum(2) > 0)
    points = camera[valid] @ rotation.T + center
    supports = torch.zeros(len(points), dtype=torch.int32, device=depth.device)
    for other in references:
        if other == frame:
            continue
        other_rotation, other_center = poses[other]
        support, _ = project_votes(points, depths[other], intrinsics[other],
                                   other_rotation, other_center, tolerance)
        supports += support
    keep = supports >= min_views
    attributes = torch.cat((color[valid][keep], normals[valid][keep] @ rotation.T), 1)
    return voxel_average(points[keep], attributes, voxel)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sequence', type=Path)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--trajectory', type=Path, required=True)
    p.add_argument('--depth-dir', type=Path, required=True,
                   help='Consistency-filtered depth that is back-projected')
    p.add_argument('--vote-depth-dir', type=Path, required=True,
                   help='Unfiltered depth used for mask-vote visibility')
    p.add_argument('--masks-dir', type=Path, required=True)
    p.add_argument('--bounds-cloud', type=Path, required=True)
    p.add_argument('--scene-output', type=Path, required=True)
    p.add_argument('--object-output', type=Path, required=True)
    p.add_argument('--resolution', type=int, default=500,
                   help='Object point spacing is the object diagonal divided by this, so '
                        'clouds have similar point counts whatever the map scale')
    p.add_argument('--scene-factor', type=float, default=4.,
                   help='Scene point spacing as a multiple of the object spacing')
    p.add_argument('--min-views', type=int, default=3,
                   help='Other views that must see each point at the same depth')
    p.add_argument('--reference-views', type=int, default=24)
    p.add_argument('--tolerance', type=float, default=.01,
                   help='Relative depth difference counted as agreement')
    p.add_argument('--min-ratio', type=float, default=.8, help='Object mask-vote ratio')
    p.add_argument('--discontinuity-tau', type=float, default=.03)
    p.add_argument('--min-depth', type=float, default=.5)
    p.add_argument('--max-depth', type=float, default=30.)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    if (a.resolution < 1 or a.scene_factor < 1 or a.min_views < 1
            or not 0 < a.tolerance < 1 or not 0 < a.min_ratio <= 1):
        p.error('Invalid voxel, view, tolerance, or ratio settings')
    device = a.device
    box = object_bounds(a.bounds_cloud, 0.)
    a.voxel = float(np.linalg.norm(box.get_extent())) / a.resolution
    a.scene_voxel = a.voxel * a.scene_factor
    poses_np = load_poses(a.trajectory)
    intrinsics_np = frame_intrinsics(a.model)
    frames = [frame for frame in sorted(poses_np)
              if (a.depth_dir / f'{frame:08d}_grounded.npy').is_file()]
    if len(frames) < 2:
        p.error('Need at least two registered frames with depth')
    as_tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    depths = {}
    for frame in bar(frames, desc='Loading depth', unit='frame'):
        depth = np.load(a.depth_dir / f'{frame:08d}_grounded.npy', allow_pickle=False)
        depth = depth.astype('float32')
        valid = depth_validity(depth, a.min_depth, a.max_depth, a.discontinuity_tau)
        depths[frame] = as_tensor(np.where(valid, depth, 0))
    intrinsics = {frame: as_tensor(intrinsics_np[frame]) for frame in frames}
    poses = {frame: (as_tensor(poses_np[frame][0]), as_tensor(poses_np[frame][1]))
             for frame in frames}
    count = min(a.reference_views, len(frames))
    references = [frames[index] for index in
                  np.unique(np.linspace(0, len(frames) - 1, count).round().astype(int))]
    point_chunks, attribute_chunks = [], []
    with torch.inference_mode():
        for frame in bar(frames, desc='Point cloud', unit='frame'):
            image = cv2.imread(str(a.sequence / 'frames' / f'{frame:08d}.png'), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != depths[frame].shape:
                raise ValueError(f'Frame/depth mismatch {frame}')
            color = as_tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.)
            rotation, center = poses[frame]
            points, attributes = frame_points(
                frame, depths[frame], color, intrinsics[frame], rotation, center, depths,
                intrinsics, poses, references, a.min_views, a.tolerance, a.voxel)
            point_chunks.append(points)
            attribute_chunks.append(attributes)
        points, attributes = voxel_average(torch.cat(point_chunks), torch.cat(attribute_chunks),
                                           a.voxel)
    del depths
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.cpu().double().numpy()))
    attributes = attributes.cpu().double().numpy()
    cloud.colors = o3d.utility.Vector3dVector(attributes[:, :3])
    cloud.normals = o3d.utility.Vector3dVector(attributes[:, 3:])
    fused = len(cloud.points)
    with spinner('Removing isolated points'):
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.)
    cloud.normalize_normals()
    scene = cloud.voxel_down_sample(a.scene_voxel)
    scene.normalize_normals()
    write_ply(a.scene_output, np.asarray(scene.points), np.asarray(scene.colors),
              np.asarray(scene.normals))

    box = object_bounds(a.bounds_cloud, .4)  # same crop margin as the object mesh
    inside = np.asarray(box.get_point_indices_within_bounding_box(cloud.points), dtype=int)
    candidates = cloud.select_by_index(inside)
    positives, visible = visible_mask_votes(
        np.asarray(candidates.points), a.masks_dir, a.vote_depth_dir, poses_np, intrinsics_np,
        relative_tolerance=.06, absolute_tolerance=.12)
    ratio = positives / np.maximum(visible, 1)
    selected = np.flatnonzero((positives >= 2) & (ratio >= a.min_ratio))
    if not len(selected):
        raise ValueError('No points received enough object-mask votes')
    obj = candidates.select_by_index(selected)
    obj, _ = obj.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.)
    write_ply(a.object_output, np.asarray(obj.points), np.asarray(obj.colors),
              np.asarray(obj.normals))
    report = {'method': 'multi-view-supported depth back-projection, voxel averaged',
              'frames': len(frames), 'reference_views': len(references), 'object_voxel': a.voxel,
              'min_views': a.min_views, 'tolerance': a.tolerance, 'fused_points': fused,
              'scene_voxel': a.scene_voxel,
              'scene_points': len(scene.points), 'object_points': len(obj.points),
              'object_min_ratio': a.min_ratio}
    a.scene_output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    done(f'Point clouds: scene {len(scene.points):,} points, object {len(obj.points):,} points')


if __name__ == '__main__':
    main()
