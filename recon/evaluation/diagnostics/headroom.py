#!/usr/bin/env python3
"""How much object surface a partial capture cannot see, and how much of that symmetry could supply.

Answers whether completing unobserved geometry is worth anything on a capture, before building anything
that completes it. A contiguous arc of the capture is kept and the rest dropped, which is what an
incomplete scan looks like; dropping every n-th frame instead leaves full angular coverage and almost no
unobserved surface. Visibility is exact: the ground-truth mesh is ray-cast into each kept camera and a
sampled surface point counts as seen when its distance along the ray matches the first hit.

The symmetry share is the baseline any generative completion has to beat: where unseen surface is the
mirror image of surface that was seen, a reflection fills it and no learned prior is needed. The mirror
plane is the principal axis whose reflection lands closest to the real surface.

MobileBrick's official score excludes vertices outside the scene's visibility volume
(recon/evaluation/metrics/mobilebrick.py), so surface no view observed is not scored there at all.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def visibility(raycast: o3d.t.geometry.RaycastingScene, points: np.ndarray, pose: np.ndarray,
               K: np.ndarray, height: int, width: int, tolerance: float) -> np.ndarray:
    """Which sampled points one camera sees: inside the image and nothing closer along the ray."""
    rotation, centre = pose[:3, :3], pose[:3, 3]
    camera = (points - centre) @ rotation                                   # world -> camera
    z = camera[:, 2]
    u = K[0, 0] * camera[:, 0] / np.maximum(z, 1e-9) + K[0, 2]
    v = K[1, 1] * camera[:, 1] / np.maximum(z, 1e-9) + K[1, 2]
    inside = (z > 0) & (u >= 0) & (u < width - 1) & (v >= 0) & (v < height - 1)
    seen = np.zeros(len(points), bool)
    if not inside.any():
        return seen
    directions = points[inside] - centre
    distance = np.linalg.norm(directions, axis=1)
    rays = np.concatenate([np.broadcast_to(centre, directions.shape),
                           directions / distance[:, None]], 1).astype(np.float32)
    hit = raycast.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
    seen[inside] = np.abs(hit - distance) < tolerance
    return seen


def mirror_plane(points: np.ndarray, raycast: o3d.t.geometry.RaycastingScene) -> tuple[np.ndarray, np.ndarray]:
    """Principal axis whose reflection through the centroid lands closest to the surface, and that error."""
    centroid = points.mean(0)
    _, _, axes = np.linalg.svd(points - centroid, full_matrices=False)
    best = None
    for axis in axes:
        reflected = points - 2 * np.outer((points - centroid) @ axis, axis)
        residual = raycast.compute_distance(o3d.core.Tensor(reflected.astype(np.float32))).numpy()
        if best is None or np.median(residual) < np.median(best[1]):
            best = (axis, residual)
    return best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('scene', type=Path, help='Dataset scene with mesh/gt_mesh.ply, pose/, intrinsic/, mask/')
    p.add_argument('--arcs', type=float, nargs='+', default=(1., .75, .5, .25),
                   help='Fractions of the capture kept, as a contiguous arc')
    p.add_argument('--samples', type=int, default=120_000)
    p.add_argument('--tolerance', type=float, default=.002, help='Metres; depth agreement for "seen"')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    mesh = o3d.io.read_triangle_mesh(str(a.scene / 'mesh/gt_mesh.ply'))
    o3d.utility.random.seed(0)
    points = np.asarray(mesh.sample_points_uniformly(a.samples).points)
    raycast = o3d.t.geometry.RaycastingScene()
    raycast.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    frames = sorted(path.stem for path in (a.scene / 'pose').iterdir())
    seen_by = {}
    for name in frames:
        mask = cv2.imread(str(a.scene / 'mask' / f'{name}.png'), cv2.IMREAD_GRAYSCALE)
        height, width = mask.shape[:2]
        seen_by[name] = visibility(raycast, points, np.loadtxt(a.scene / 'pose' / f'{name}.txt'),
                                   np.loadtxt(a.scene / 'intrinsic' / f'{name}.txt'),
                                   height, width, a.tolerance)
    axis, residual = mirror_plane(points, raycast)
    centroid = points.mean(0)
    reflected = points - 2 * np.outer((points - centroid) @ axis, axis)
    search = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(points.astype(np.float32)))
    search.knn_index()
    partner = search.knn_search(o3d.core.Tensor(reflected.astype(np.float32)), 1)[0].numpy().ravel()
    report = {'scene': a.scene.name, 'frames': len(frames), 'samples': len(points),
              'symmetry_residual_mm': float(np.median(residual) * 1000), 'arcs': []}
    for arc in a.arcs:
        keep = frames[:max(2, int(round(len(frames) * arc)))]
        seen = np.zeros(len(points), bool)
        for name in keep:
            seen |= seen_by[name]
        unseen = ~seen
        # Recoverable by reflection: the mirror partner was seen and the object is symmetric there.
        recoverable = unseen & seen[partner] & (residual < 2 * a.tolerance)
        report['arcs'].append({'arc': arc, 'frames': len(keep), 'unseen_pct': float(unseen.mean() * 100),
                               'symmetric_share_pct': float(recoverable.sum() / max(1, unseen.sum()) * 100)})
    print(json.dumps(report, indent=2))
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
