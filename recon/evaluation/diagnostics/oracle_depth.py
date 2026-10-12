#!/usr/bin/env python3
"""Replace object depth with depth rendered from the reference mesh (diagnostic).

The reference mesh is moved into the run's map frame with the inverse of the
camera-centre similarity used for scoring, then ray-cast into every frame
with the run's own cameras. Pixels the mesh covers get its exact depth;
object-mask pixels it misses are cleared; everything else keeps the run's
depth. Fusing the result shows how much error is left once depth is perfect,
which separates depth error from fusion and extraction error.
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
from recon.evaluation.metrics.mesh import icp_similarity, similarity


def reference_mesh_in_map(trajectory: Path, pose_dir: Path, reference: Path,
                          refine_to: Path | None = None) -> o3d.geometry.TriangleMesh:
    """The reference mesh placed in the run's map frame.

    Placement starts from the camera-centre similarity used for scoring. With
    refine_to (our object mesh), it is refined by scaled ICP onto our surface,
    the alignment the shape metrics use: a dataset whose reference poses and
    reference mesh disagree slightly in scale (MobileBrick) otherwise shifts the
    reference depth by millimetres, and every candidate looks biased by it.
    """
    estimate_center, reference_center, rotation, scale, _ = similarity(trajectory, pose_dir)
    mesh = o3d.io.read_triangle_mesh(str(reference))
    vertices = ((np.asarray(mesh.vertices) - reference_center) / scale) @ rotation.T + estimate_center
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    if refine_to is not None:
        ours = np.asarray(o3d.io.read_triangle_mesh(str(refine_to)).sample_points_uniformly(200000).points)
        surface = np.asarray(mesh.sample_points_uniformly(200000).points)
        # 10 mm then 3 mm correspondence radii, in map units.
        transform = icp_similarity(surface, ours, (.01 / scale, .003 / scale))
        mesh.vertices = o3d.utility.Vector3dVector(vertices @ transform[:3, :3].T + transform[:3, 3])
    return mesh


def reference_in_map(trajectory: Path, pose_dir: Path, reference: Path,
                     refine_to: Path | None = None) -> o3d.t.geometry.RaycastingScene:
    """Ray-casting scene of reference_mesh_in_map."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
        reference_mesh_in_map(trajectory, pose_dir, reference, refine_to)))
    return scene


def render_depth(scene: o3d.t.geometry.RaycastingScene, K: np.ndarray, rotation: np.ndarray,
                 center: np.ndarray, height: int, width: int) -> np.ndarray:
    """Z-depth of the mesh in a camera; 0 where no surface is hit."""
    v, u = np.mgrid[:height, :width].astype(np.float32)
    directions = np.stack(((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1],
                           np.ones_like(u)), -1) @ rotation.T
    rays = np.concatenate((np.broadcast_to(center, directions.shape), directions), -1)
    hit = scene.cast_rays(o3d.core.Tensor(rays.reshape(-1, 6).astype(np.float32)))['t_hit'].numpy()
    # Directions have unit camera-space z, so the hit distance is the z-depth.
    return np.where(np.isfinite(hit), hit, 0).reshape(height, width).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--trajectory', type=Path, required=True)
    p.add_argument('--pose-dir', type=Path, required=True, help='Reference camera-to-world poses')
    p.add_argument('--reference', type=Path, required=True, help='Reference mesh')
    p.add_argument('--masks-dir', type=Path, required=True)
    p.add_argument('--depth-dir', type=Path, required=True, help='Depth to modify in place')
    p.add_argument('--refine-to', type=Path, help='Our object mesh: refine the placement by surface fit')
    a = p.parse_args()
    scene = reference_in_map(a.trajectory, a.pose_dir, a.reference, a.refine_to)
    poses = load_poses(a.trajectory)
    intrinsics = frame_intrinsics(a.model)
    replaced = cleared = 0
    frames = [frame for frame in sorted(poses)
              if (a.depth_dir / f'{frame:08d}_grounded.npy').is_file()]
    for frame in bar(frames, desc='Reference depth', unit='frame'):
        path = a.depth_dir / f'{frame:08d}_grounded.npy'
        depth = np.load(path)
        mask = cv2.imread(str(a.masks_dir / f'{frame:08d}.png'), cv2.IMREAD_GRAYSCALE) > 127
        frame_rotation, center = poses[frame]
        reference = render_depth(scene, intrinsics[frame], frame_rotation, center, *depth.shape)
        hit = reference > 0
        depth = np.where(hit, reference, np.where(mask, 0, depth)).astype(np.float32)
        replaced += int(hit.sum())
        cleared += int((mask & ~hit).sum())
        np.save(path, depth)
    report = {'reference': str(a.reference), 'frames': len(frames),
              'replaced_pixels': replaced, 'cleared_mask_pixels': cleared}
    (a.depth_dir / 'oracle_depth.json').write_text(json.dumps(report, indent=2) + '\n')
    done(f'Reference depth: {replaced:,} pixels replaced in {len(frames)} frames')


if __name__ == '__main__':
    main()
