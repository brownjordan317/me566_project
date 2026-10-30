#!/usr/bin/env python3
"""Is the reconstruction dented in or bulged out, and where? Signed surface error along the reference's normal.

Distances to the reference (the official metric, chamfer, F-score) are unsigned, so a recess filled in and a flat
panel sunk inward score alike. This samples the reference surface and, from each sample, casts a ray along its
outward normal both ways to our mesh: the nearest hit within --reach gives a signed offset - positive where our
surface lies outside the true one (a bulge, a filled recess), negative where it lies inside (a dent, a surface
seen through glass). Reports the shares of the surface dented and bulged beyond --tolerance, their typical
depth, and, split by how the photographs see the surface, where each happens. With --overlay it marks dents
(blue) and bulges (red) on the photographs that see the most of them.

Placement matters: a 1.4% scale error on a car reads as millimetres of "dent" at one end and "bulge" at the
other. So, like the shape metrics, the mesh is first placed by the camera similarity and then refined by ICP with
scale against the reference (--no-refine skips it), and only what remains is local shape.

The reference must be consistently oriented outward (MobileBrick's CAD-derived meshes are); the median offset of
a sound reconstruction is then close to zero, which the report includes as a check.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from recon.evaluation.diagnostics.headroom import visibility
from recon.evaluation.metrics.mesh import similarity


def signed_offsets(reference_points, normals, mesh, reach):
    """Offset along each normal to the nearest surface of `mesh` within reach (NaN where there is none)."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    best = np.full(len(reference_points), np.nan)
    for sign in (1, -1):
        rays = np.hstack([reference_points, sign * normals]).astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
        hit = np.where(hit <= reach, hit * sign, np.nan)
        closer = np.isnan(best) | (np.abs(hit) < np.abs(best))
        best = np.where(closer & ~np.isnan(hit), hit, best)
    return best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mesh', type=Path, help='Our mesh, in map units')
    p.add_argument('--run', type=Path, required=True, help='The run (poses/export/trajectory.csv) the mesh is from')
    p.add_argument('--scene', type=Path, required=True, help='Dataset scene (mesh/, pose/, intrinsic/, mask/, image/)')
    p.add_argument('--samples', type=int, default=150_000)
    p.add_argument('--tolerance', type=float, default=.002, help='Metres; beyond this a sample is dented or bulged')
    p.add_argument('--reach', type=float, default=.02, help='Metres searched along the normal')
    p.add_argument('--refine', action=argparse.BooleanOptionalAction, default=True,
                   help='Refine placement and scale by ICP before measuring, so only local shape remains')
    p.add_argument('--overlay', type=Path, help='Write an image marking dents and bulges on the photographs')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    reference = o3d.io.read_triangle_mesh(str(a.scene / 'mesh/gt_mesh.ply'))
    o3d.utility.random.seed(0)
    cloud = reference.sample_points_uniformly(a.samples, use_triangle_normal=True)
    points, normals = np.asarray(cloud.points), np.asarray(cloud.normals)
    centre, target, rotation, scale, _ = similarity(a.run / 'poses/export/trajectory.csv', a.scene / 'pose')
    ours = o3d.io.read_triangle_mesh(str(a.mesh))
    ours.vertices = o3d.utility.Vector3dVector((np.asarray(ours.vertices) - centre) @ rotation * scale + target)
    icp_scale = 1.
    if a.refine:
        source = ours.sample_points_uniformly(a.samples)
        result = o3d.pipelines.registration.registration_icp(
            source, cloud, .005, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
        ours.transform(result.transformation)
        icp_scale = float(np.cbrt(np.linalg.det(result.transformation[:3, :3])))
    offset = signed_offsets(points, normals, ours, a.reach)
    # How the capture sees each sample: how many frames, and their best viewing angle.
    raycast = o3d.t.geometry.RaycastingScene()
    raycast.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(reference))
    views = np.zeros(len(points), int)
    best = np.zeros(len(points))
    poses = {}
    for path in sorted((a.scene / 'pose').iterdir()):
        mask = cv2.imread(str(a.scene / 'mask' / f'{path.stem}.png'), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        pose = np.loadtxt(path)
        seen = visibility(raycast, points, pose, np.loadtxt(a.scene / 'intrinsic' / f'{path.stem}.txt'), *mask.shape[:2], .002)
        poses[path.stem] = pose
        views += seen
        centre_cam = pose[:3, 3]
        direction = centre_cam - points
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        best = np.where(seen, np.maximum(best, (normals * direction).sum(1)), best)
    seen = views > 0
    found = seen & ~np.isnan(offset)
    dent = found & (offset < -a.tolerance)
    bulge = found & (offset > a.tolerance)
    missing = seen & np.isnan(offset)
    mm = lambda x: float(np.median(np.abs(offset[x])) * 1000) if x.any() else float('nan')
    report = {'mesh': str(a.mesh), 'scene': a.scene.name, 'tolerance_mm': a.tolerance * 1000,
              'refined': a.refine, 'icp_scale_change_pct': (icp_scale - 1) * 100,
              'median_signed_offset_mm': float(np.nanmedian(offset[found]) * 1000),
              'seen_pct': float(seen.mean() * 100),
              'dented_pct': float(dent.sum() / max(seen.sum(), 1) * 100), 'dent_depth_median_mm': mm(dent),
              'bulged_pct': float(bulge.sum() / max(seen.sum(), 1) * 100), 'bulge_height_median_mm': mm(bulge),
              'no_surface_within_reach_pct': float(missing.sum() / max(seen.sum(), 1) * 100), 'by_view': {}}
    for label, pick in (('seen by 1-3 frames', seen & (views <= 3)), ('seen by 4+ frames', seen & (views > 3)),
                        ('best view within 45 deg', seen & (best > np.cos(np.radians(45)))),
                        ('best view beyond 45 deg', seen & (best <= np.cos(np.radians(45))))):
        total = max(pick.sum(), 1)
        report['by_view'][label] = {'share_pct': float(pick.sum() / max(seen.sum(), 1) * 100),
                                    'dented_pct': float((dent & pick).sum() / total * 100),
                                    'bulged_pct': float((bulge & pick).sum() / total * 100)}
    print(json.dumps(report, indent=2))
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + '\n')
    if a.overlay:
        scored = []
        for stem, pose in poses.items():
            cam = (points - pose[:3, 3]) @ pose[:3, :3]             # pose files are camera-to-world
            k = np.loadtxt(a.scene / 'intrinsic' / f'{stem}.txt')
            z = cam[:, 2]
            u = k[0, 0] * cam[:, 0] / z + k[0, 2]
            v = k[1, 1] * cam[:, 1] / z + k[1, 2]
            direction = points - pose[:3, 3]
            distance = np.linalg.norm(direction, axis=1)
            hit = raycast.cast_rays(o3d.core.Tensor(np.hstack([np.broadcast_to(pose[:3, 3], points.shape),
                                                                direction / distance[:, None]]).astype(np.float32)))['t_hit'].numpy()
            visible = (z > 0) & (np.abs(hit - distance) < .002)
            scored.append(((visible & (dent | bulge)).sum(), stem, u, v, visible))
        tiles = []
        for _, stem, u, v, visible in sorted(scored, key=lambda x: -x[0])[:4]:
            image = cv2.imread(str(next((a.scene / 'image').glob(f'{stem}.*'))))
            for pick, colour in ((dent, (255, 80, 0)), (bulge, (0, 0, 255))):
                for x, y in zip(u[visible & pick].astype(int), v[visible & pick].astype(int)):
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                        cv2.circle(image, (x, y), 3, colour, -1)
            mask = cv2.imread(str(a.scene / 'mask' / f'{stem}.png'), cv2.IMREAD_GRAYSCALE) > 127
            ys, xs = np.nonzero(mask)
            crop = image[max(ys.min() - 40, 0):ys.max() + 40, max(xs.min() - 40, 0):xs.max() + 40]
            tile = cv2.resize(crop, (480, int(480 * crop.shape[0] / crop.shape[1])))
            cv2.putText(tile, f'frame {stem}', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
            tiles.append(tile)
        height = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
        while len(tiles) < 4:
            tiles.append(np.full_like(tiles[0], 255))
        a.overlay.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(a.overlay), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))


if __name__ == '__main__':
    main()
