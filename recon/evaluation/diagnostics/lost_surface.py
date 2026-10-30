#!/usr/bin/env python3
"""Where does photographed surface get lost? Attribute it to the first stage that drops it.

On a full capture almost all the reference surface our mesh misses was photographed: the cameras saw it and the
reconstruction failed to turn it into surface (docs/PIPELINE.md). This finds which stage fails. The reference
surface is sampled and each sample marked seen or not by the capture (the capture-headroom visibility test). For
every seen sample farther than --tolerance from the final mesh, each stage is checked in pipeline order:

  depth models   does VGGT or DA3 put depth within --tolerance of the sample, in any frame that sees it?
  fusion         does the fused depth (the photometric choice between them)?
  refinement     does the plane-sweep-refined depth?
  gaussian       does the planar-Gaussian pass's depth?
  consistency    does the cross-view-checked depth, which the TSDF fuses?
  views          is it accurate there in at least --min-views frames, as the TSDF requires for surface?
  tsdf           is the weighted TSDF's surface within --tolerance?
  extraction     is it still in the extracted, cleaned object mesh?
  cut            is it still after the 3D cut?

The first check that fails is where the sample was lost. Everything is compared in the run's map units, with the
reference brought into them by the inverse of the similarity the scorers use. It follows the 2026-10-12
recall-loss diagnostic (removed then), rebuilt for the current scene-first chain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from recon.common.colmap import parse_model
from recon.evaluation.diagnostics.headroom import visibility
from recon.evaluation.metrics.mesh import similarity

DEPTH_STAGES = (('depth models', ('vggt', 'da3')), ('fusion', ('fused',)), ('refinement', ('fused_refined',)),
                ('gaussian', ('gaussian',)), ('consistency', ('consistent',)))


def distance_to(mesh_path: Path, points: np.ndarray) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene.compute_distance(o3d.core.Tensor(points.astype(np.float32))).numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path, help='A run kept with KEEP_INTERMEDIATE=1 (depth/, work/, mesh/, poses/)')
    p.add_argument('--scene', type=Path, required=True, help='Dataset scene (mesh/, pose/, intrinsic/, mask/)')
    p.add_argument('--final', default='mesh/object_segmented.ply', help='The finished mesh, relative to the run')
    p.add_argument('--samples', type=int, default=120_000)
    p.add_argument('--tolerance', type=float, default=.003, help='Metres: missing, and accurate depth')
    p.add_argument('--visibility-tolerance', type=float, default=.002, help='Metres; as the headroom test')
    p.add_argument('--min-views', type=int, default=3, help='The TSDF minimum (weighted_tsdf --min-views)')
    p.add_argument('--output', type=Path)
    a = p.parse_args()

    reference = o3d.io.read_triangle_mesh(str(a.scene / 'mesh/gt_mesh.ply'))
    o3d.utility.random.seed(0)
    points = np.asarray(reference.sample_points_uniformly(a.samples).points)
    raycast = o3d.t.geometry.RaycastingScene()
    raycast.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(reference))
    # Reference (metres) -> map units: invert ref = (ours - centre) @ rotation * scale + target.
    centre, target, rotation, scale, _ = similarity(a.run / 'poses/export/trajectory.csv', a.scene / 'pose')
    in_map = ((points - target) / scale) @ rotation.T + centre
    tolerance_map = a.tolerance / scale

    intrinsics, _, frames = parse_model(a.run / 'poses/sparse_txt')
    seen = np.zeros(len(points), bool)
    accurate = {stage: np.zeros(len(points), np.int32) for _, dirs in DEPTH_STAGES for stage in dirs}
    for number, R, t, _ in frames:
        name = f'{number:06d}'
        mask = cv2.imread(str(a.scene / 'mask' / f'{name}.png'), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        visible = visibility(raycast, points, np.loadtxt(a.scene / 'pose' / f'{name}.txt'),
                             np.loadtxt(a.scene / 'intrinsic' / f'{name}.txt'), *mask.shape[:2], a.visibility_tolerance)
        seen |= visible
        camera = in_map[visible] @ R.T + t
        K = intrinsics[number]
        z = camera[:, 2]
        u = np.round(K[0, 0] * camera[:, 0] / z + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * camera[:, 1] / z + K[1, 2]).astype(int)
        index = np.nonzero(visible)[0]
        for stage in accurate:
            path = a.run / 'depth' / stage / f'{number:08d}_grounded.npy'
            if not path.is_file():
                continue
            depth = np.load(path)
            inside = (z > 0) & (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
            d = np.zeros(len(z), np.float32)
            d[inside] = depth[v[inside], u[inside]]
            good = (d > 0) & (np.abs(d - z) < tolerance_map)
            accurate[stage][index[good]] += 1

    meshes = {'tsdf': a.run / 'work/object_tsdf.ply', 'extraction': a.run / 'mesh/object.ply', 'cut': a.run / a.final}
    near = {k: distance_to(path, in_map) < tolerance_map for k, path in meshes.items() if path.is_file()}
    lost = seen & ~near['cut']
    order = [(name, (sum(accurate[s] for s in dirs) > 0)) for name, dirs in DEPTH_STAGES]
    order.append(('views', accurate['consistent'] >= a.min_views))
    order += [(k, near[k]) for k in ('tsdf', 'extraction', 'cut') if k in near]
    remaining = lost.copy()
    report = {'run': str(a.run), 'scene': a.scene.name, 'samples': int(len(points)),
              'tolerance_mm': a.tolerance * 1000, 'seen_pct': float(seen.mean() * 100),
              'missed_pct': float((~near['cut']).mean() * 100), 'never_seen_missed_pct': float((~seen & ~near['cut']).mean() * 100),
              'seen_but_lost_pct': float(lost.mean() * 100), 'lost_at': {}}
    for name, passes in order:
        dropped = remaining & ~passes
        report['lost_at'][name] = float(dropped.mean() * 100)
        remaining &= passes
    report['lost_at']['unattributed'] = float(remaining.mean() * 100)
    print(json.dumps(report, indent=2))
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
