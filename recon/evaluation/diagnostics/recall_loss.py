#!/usr/bin/env python3
"""Where is missing surface lost? (diagnostic, needs a run kept with KEEP_INTERMEDIATE=1)

Points are sampled on the reference surface, placed in our map frame by the
shape alignment (camera similarity refined by scaled ICP onto our mesh). For
every frame each point is tested in turn:

  seen       inside the image, facing the camera, and not hidden by the
             reference itself (a ray from the camera reaches it first);
  masked     seen and inside the object mask;
  depth      masked and the object depth input before the cross-view check
             (fused_refined, or hybrid without fusion) has a value there;
  kept       depth survives the cross-view check (depth/consistent);
  accurate   kept and within the tolerance of the point's true depth.

A point is recalled if our final mesh passes within the tolerance. Each missed
point is assigned to the first test that fewer than --min-views frames pass:
unobserved, outside the mask, no depth, removed by the cross-view check,
inaccurate depth, or (all tests passed) lost in meshing, extraction or cleanup,
split into surface present but 1-3 tolerances off and no surface nearby.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from recon.common.camera import frame_intrinsics
from recon.common.progress import bar, done
from recon.common.trajectory import load_poses
from recon.evaluation.diagnostics.oracle_depth import reference_mesh_in_map
from recon.evaluation.metrics.mesh import similarity

CAUSES = ('unobserved', 'outside mask', 'no depth', 'removed by cross-view check', 'inaccurate depth',
          'meshing: surface 1-3 tolerances away', 'meshing: no surface within 3 tolerances')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--sequence', type=Path, required=True, help='Prepared sequence (masks/)')
    p.add_argument('--pose-dir', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--tolerance-mm', type=float, default=1.)
    p.add_argument('--min-views', type=int, default=3)
    p.add_argument('--samples', type=int, default=60000)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    trajectory = a.run / 'poses/export/trajectory.csv'
    scale = similarity(trajectory, a.pose_dir)[3]                    # metres per map unit
    tolerance = a.tolerance_mm / 1000 / scale
    reference = reference_mesh_in_map(trajectory, a.pose_dir, a.reference, a.run / 'mesh/object.ply')
    reference.compute_triangle_normals()
    o3d.utility.random.seed(3)
    sampled = reference.sample_points_uniformly(a.samples, use_triangle_normal=True)
    points, normals = np.asarray(sampled.points), np.asarray(sampled.normals)
    occluder = o3d.t.geometry.RaycastingScene()
    occluder.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(reference))
    ours = o3d.t.geometry.RaycastingScene()
    ours.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(str(a.run / 'mesh/object.ply'))))
    to_ours = ours.compute_distance(o3d.core.Tensor(points.astype(np.float32))).numpy()
    recalled = to_ours < tolerance
    depth_dirs = [a.run / 'depth' / name for name in ('fused_refined', 'fused', 'hybrid')]
    raw_dir = next(d for d in depth_dirs if d.is_dir())
    kept_dir = a.run / 'depth/consistent'
    poses = load_poses(trajectory)
    intrinsics = frame_intrinsics(a.run / 'poses/sparse_txt')
    counts = np.zeros((5, len(points)), np.int32)
    for frame in bar(sorted(poses), desc='Recall loss', unit='frame'):
        rotation, centre = poses[frame]
        K = intrinsics[frame]
        camera = (points - centre) @ rotation
        z = camera[:, 2]
        u = K[0, 0] * camera[:, 0] / np.maximum(z, 1e-9) + K[0, 2]
        v = K[1, 1] * camera[:, 1] / np.maximum(z, 1e-9) + K[1, 2]
        mask = cv2.imread(str(a.sequence / 'masks' / f'{frame:08d}.png'), cv2.IMREAD_GRAYSCALE) > 127
        height, width = mask.shape
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        inside = (z > 0) & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
        facing = ((centre - points) * normals).sum(1) > 0
        candidates = np.flatnonzero(inside & facing)
        if not len(candidates):
            continue
        direction = points[candidates] - centre
        distance = np.linalg.norm(direction, axis=1)
        rays = np.concatenate((np.broadcast_to(centre, direction.shape), direction / distance[:, None]), 1)
        hit = occluder.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))['t_hit'].numpy()
        seen = candidates[hit > distance - 2 * tolerance]
        counts[0, seen] += 1
        in_mask = seen[mask[vi[seen], ui[seen]]]
        counts[1, in_mask] += 1
        raw_path, kept_path = raw_dir / f'{frame:08d}_grounded.npy', kept_dir / f'{frame:08d}_grounded.npy'
        if not raw_path.is_file():
            continue
        raw = np.load(raw_path)[vi[in_mask], ui[in_mask]]
        with_depth = in_mask[raw > 0]
        counts[2, with_depth] += 1
        if not kept_path.is_file():
            continue
        kept_depth = np.load(kept_path)[vi[with_depth], ui[with_depth]]
        kept = with_depth[kept_depth > 0]
        counts[3, kept] += 1
        accurate = kept[np.abs(kept_depth[kept_depth > 0] - z[kept]) < tolerance]
        counts[4, accurate] += 1
    passes = counts >= a.min_views
    meshing = np.where(to_ours < 3 * tolerance, len(CAUSES) - 2, len(CAUSES) - 1)
    cause = np.where(passes.all(0), meshing, np.argmin(passes, axis=0))
    missed = ~recalled
    report = {'tolerance_mm': a.tolerance_mm, 'min_views': a.min_views, 'samples': int(len(points)),
              'recall_pct': float(recalled.mean() * 100),
              'missed_by_cause_pct': {name: float(np.mean(missed & (cause == index)) * 100)
                                      for index, name in enumerate(CAUSES)}}
    if a.output:
        a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"recall {report['recall_pct']:.1f}%  missed: " + ', '.join(
        f'{name} {value:.1f}%' for name, value in report['missed_by_cause_pct'].items()))
    done(f"Recall loss: {100 - report['recall_pct']:.1f}% of the reference surface missed")


if __name__ == '__main__':
    main()
