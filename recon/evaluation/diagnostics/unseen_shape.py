#!/usr/bin/env python3
"""How wrong is the closure's guess where no view looked?

The colour question has an answer: rendering the fitted Gaussians fills unobserved surface to within
about five points of colour error of observation itself (docs/PIPELINE.md), which leaves little for a
generative prior to win. Shape is the other half, and the held-out-arc protocol cannot reach it - an arc
leaves the mesh too open for the closure to accept at all. But the datasets ship complete ground truth,
so the question can be asked directly: sample the reference surface, mark which samples no view observed
(exactly as the capture-headroom diagnostic does), and compare the closed mesh's error on those against
its error where the views did look.

A closure that is nearly as accurate unobserved as observed leaves nothing for a prior to add; one that is
far worse marks the surface where a prior would earn its place, and by how much.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from recon.evaluation.diagnostics.headroom import visibility
from recon.evaluation.metrics.mesh import similarity


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mesh', type=Path, help='Our mesh, in map units')
    p.add_argument('--scene', type=Path, required=True, help='Dataset scene (mesh/, pose/, intrinsic/, mask/)')
    p.add_argument('--trajectory', type=Path, required=True)
    p.add_argument('--samples', type=int, default=120_000)
    p.add_argument('--tolerance', type=float, default=.002, help='Metres; depth agreement counted as seen')
    p.add_argument('--seen-by', choices=('capture', 'trajectory'), default='capture',
                   help="Which views count as having seen the surface: every frame of the dataset capture (the "
                        "capture's own headroom), or only the frames in --trajectory (what a held-out "
                        "reconstruction actually used, so the held-out views' surface counts as unseen)")
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    reference = o3d.io.read_triangle_mesh(str(a.scene / 'mesh/gt_mesh.ply'))
    o3d.utility.random.seed(0)
    points = np.asarray(reference.sample_points_uniformly(a.samples).points)
    raycast = o3d.t.geometry.RaycastingScene()
    raycast.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(reference))
    seen = np.zeros(len(points), bool)
    frames = sorted(path.stem for path in (a.scene / 'pose').iterdir())
    if a.seen_by == 'trajectory':
        used = {int(row['frame']) for row in csv.DictReader(a.trajectory.open())}
        frames = [name for name in frames if int(name) in used]
    for name in frames:
        mask = cv2.imread(str(a.scene / 'mask' / f'{name}.png'), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        seen |= visibility(raycast, points, np.loadtxt(a.scene / 'pose' / f'{name}.txt'),
                           np.loadtxt(a.scene / 'intrinsic' / f'{name}.txt'),
                           *mask.shape[:2], a.tolerance)
    # Our mesh lives in map units; the same similarity the scorers use puts it in the reference frame.
    centre, target, rotation, scale, _ = similarity(a.trajectory, a.scene / 'pose')
    ours = o3d.io.read_triangle_mesh(str(a.mesh))
    vertices = (np.asarray(ours.vertices) - centre) @ rotation * scale + target
    ours.vertices = o3d.utility.Vector3dVector(vertices)
    theirs = o3d.t.geometry.RaycastingScene()
    theirs.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(ours))
    distance = theirs.compute_distance(o3d.core.Tensor(points.astype(np.float32))).numpy() * 1000.
    report = {'mesh': str(a.mesh), 'scene': a.scene.name, 'samples': int(len(points)),
              'seen_by': a.seen_by, 'seeing_frames': len(frames),
              'unseen_pct': float((~seen).mean() * 100)}
    for name, picked in (('observed', seen), ('unobserved', ~seen)):
        if picked.any():
            report[name] = {'median_mm': float(np.median(distance[picked])),
                            'mean_mm': float(distance[picked].mean()),
                            'p90_mm': float(np.percentile(distance[picked], 90)),
                            'samples': int(picked.sum())}
    print(json.dumps(report, indent=2))
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
