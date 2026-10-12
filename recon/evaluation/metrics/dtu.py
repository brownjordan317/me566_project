#!/usr/bin/env python3
"""DTU chamfer evaluation (port of the official protocol, as in DTUeval-python).

The mesh is aligned to DTU's world from our camera centres, sampled at 0.2 mm,
and scored in millimetres: accuracy over points inside the observability mask,
completeness over reference points above the table plane, distances above
20 mm ignored, and chamfer as their mean.

Credits: protocol and constants (0.2 mm thinning, ObsMask observability test
with 60 mm patch margin, table Plane, 20 mm cap) follow DTUeval-python
(https://github.com/jzhangbs/DTUeval-python, eval.py; MIT, Copyright (c) 2021
Jingyang Zhang), itself a port of the official MATLAB evaluation of the DTU
benchmark (Jensen et al., CVPR 2014; Aanaes et al., IJCV 2016). One known
difference: meshes are sampled with Open3D's uniform random sampling at twice
the thinning density, not DTUeval-python's per-triangle grid, so scores can
differ slightly from that script's. See THIRD_PARTY_NOTICES.md.

pose_aligned_* repeats the score with the rotation taken from the camera
orientations (recon.evaluation.metrics.mesh.pose_similarity). It places the object
far better when the intrinsics are right (true intrinsics, our poses: mean 0.594
against 0.818 mm on DTU) and worse when an unmodelled principal-point offset
biases every orientation (estimated intrinsics); chamfer_mm keeps the centre-based
alignment.

shape_* repeats the score after refining that alignment, scale included, by
fitting the surface (ICP), so overall scale error from RGB-only cameras is
reported separately (size_error_pct, positive when ours is too large) instead of inflating the chamfer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.io import loadmat
from scipy.spatial import cKDTree

from recon.evaluation.metrics.mesh import pose_similarity, refine_alignment, similarity

DENSITY = .2
PATCH = 60
MAX_DIST = 20


def thin(points: np.ndarray, radius: float, seed: int = 0) -> np.ndarray:
    """Greedy random thinning: keep a point only if none kept lies within radius."""
    order = np.random.default_rng(seed).permutation(len(points))
    points = points[order]
    tree = cKDTree(points)
    keep = np.ones(len(points), dtype=bool)
    for index, neighbours in enumerate(tree.query_ball_point(points, radius, workers=-1)):
        if keep[index]:
            keep[neighbours] = False
            keep[index] = True
    return points[keep]


def evaluate(points: np.ndarray, reference: Path) -> dict:
    points = thin(points, DENSITY)
    mask = loadmat(reference / 'ObsMask.mat')
    obs, bounds, resolution = mask['ObsMask'], mask['BB'].astype(np.float64), float(np.asarray(mask['Res']).ravel()[0])
    inbound = np.all((points >= bounds[:1] - PATCH) & (points < bounds[1:] + 2 * PATCH), axis=1)
    data_in = points[inbound]
    grid = np.around((data_in - bounds[:1]) / resolution).astype(np.int64)
    in_grid = np.all((grid >= 0) & (grid < np.array(obs.shape)), axis=1)
    observed = obs[tuple(grid[in_grid].T)].astype(bool)
    data_obs = data_in[in_grid][observed]
    stl = np.asarray(o3d.io.read_point_cloud(str(reference / 'stl_total.ply')).points)
    accuracy = cKDTree(stl).query(data_obs, workers=-1)[0]
    plane = loadmat(reference / 'Plane.mat')['P'].reshape(4)
    above = stl[np.c_[stl, np.ones(len(stl))] @ plane > 0]
    completeness = cKDTree(data_in).query(above, workers=-1)[0]
    mean_accuracy = float(accuracy[accuracy < MAX_DIST].mean())
    mean_completeness = float(completeness[completeness < MAX_DIST].mean())
    return {'accuracy_mm': mean_accuracy, 'completeness_mm': mean_completeness,
            'chamfer_mm': (mean_accuracy + mean_completeness) / 2,
            'scored_points': int(len(data_obs)), 'reference_points': int(len(above))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mesh', type=Path)
    p.add_argument('--reference', type=Path, required=True,
                   help='Scan reference folder with stl_total.ply, ObsMask.mat and Plane.mat')
    p.add_argument('--trajectory', type=Path, required=True)
    p.add_argument('--pose-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    estimate_center, reference_center, rotation, scale, pose_errors = similarity(a.trajectory, a.pose_dir)
    mesh = o3d.io.read_triangle_mesh(str(a.mesh))
    if not len(mesh.triangles):
        p.error(f'Empty mesh: {a.mesh}')
    vertices = (np.asarray(mesh.vertices) - estimate_center) @ rotation * scale + reference_center
    mesh.vertices = o3d.utility.Vector3dVector(vertices * 1000.)    # metres to DTU millimetres
    area = mesh.get_surface_area()
    o3d.utility.random.seed(7)
    points = np.asarray(mesh.sample_points_uniformly(int(area / DENSITY ** 2 * 2)).points)
    report = evaluate(points, a.reference)
    stl = o3d.io.read_point_cloud(str(a.reference / 'stl_total.ply')).voxel_down_sample(.5)
    shaped, scale = refine_alignment(points, np.asarray(stl.points), (10., 2.))
    shape = evaluate(shaped, a.reference)
    report.update(shape_chamfer_mm=shape['chamfer_mm'], shape_accuracy_mm=shape['accuracy_mm'],
                  shape_completeness_mm=shape['completeness_mm'], size_error_pct=(1 / scale - 1) * 100)
    # Secondary placement: rotation from the camera orientations (see pose_similarity).
    center, target, rotation, scale, _ = pose_similarity(a.trajectory, a.pose_dir)
    posed = np.asarray(o3d.io.read_triangle_mesh(str(a.mesh)).vertices)
    mesh.vertices = o3d.utility.Vector3dVector(((posed - center) @ rotation * scale + target) * 1000.)
    o3d.utility.random.seed(7)
    oriented = evaluate(np.asarray(mesh.sample_points_uniformly(int(area / DENSITY ** 2 * 2)).points), a.reference)
    report.update(pose_aligned_chamfer_mm=oriented['chamfer_mm'], pose_aligned_accuracy_mm=oriented['accuracy_mm'],
                  pose_aligned_completeness_mm=oriented['completeness_mm'])
    report.update(median_pose_error_mm=float(np.median(pose_errors) * 1000),
                  localized_frames=len(pose_errors), mesh_faces=len(mesh.triangles))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
