#!/usr/bin/env python3
"""MobileBrick's official 3D evaluation, for comparison with published results.

Port of the benchmark's evaluations/evaluate_3d.py (https://github.com/ActiveVisionLab/
MobileBrick; MIT, Copyright (c) 2023 Active Vision Laboratory). Published methods
reconstruct in the reference frame from the dataset's poses; ours is placed there by
the similarity fitted to our camera centres (recon.evaluation.metrics.mesh.similarity),
then the official steps run unchanged:

- rigid point-to-point ICP of the reference vertices onto ours (2 cm threshold,
  10 iterations), applied only when its fitness exceeds 0.99;
- vertices outside the scene's visibility volume (visibility_mask.npy, nearest
  voxel) are removed;
- 100,000 Poisson-disk samples on each mesh;
- precision, recall and F1 at 2.5 mm and 5 mm, and chamfer as the *sum* of the
  two mean nearest-neighbour distances (metres in the paper; millimetres here).

The paper averages over the 18 test scenes. Differences from the script: nearest
neighbours come from SciPy's cKDTree instead of scikit-learn (identical distances),
the visibility lookup is done in NumPy instead of torch grid_sample, and sampling
is seeded. `scaled_*` repeats the score after our scaled ICP (as for the shape
metrics), so RGB-only scale error is reported apart from surface quality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from recon.evaluation.metrics.mesh import refine_alignment, similarity

SAMPLES = 100000
THRESHOLDS = (.0025, .005)


def visible(vertices: np.ndarray, volume: dict) -> np.ndarray:
    """Vertices whose nearest voxel of the visibility volume is set (out of bounds: no)."""
    resolution = np.asarray(volume['resolutions'], dtype=np.int64)
    mask = np.asarray(volume['mask']).reshape(resolution)
    index = np.rint((vertices - volume['min_pts']) / volume['voxel_size']).astype(np.int64)
    inside = np.all((index >= 0) & (index < resolution), axis=1)
    keep = np.zeros(len(vertices), dtype=bool)
    keep[inside] = mask[tuple(index[inside].T)] > 0
    return keep


def scores(predicted: np.ndarray, reference: np.ndarray) -> dict:
    to_reference = cKDTree(reference).query(predicted, workers=-1)[0]
    to_predicted = cKDTree(predicted).query(reference, workers=-1)[0]
    report = {'chamfer_mm': float(to_reference.mean() + to_predicted.mean()) * 1000}
    for threshold in THRESHOLDS:
        precision = float(np.mean(to_reference < threshold))
        recall = float(np.mean(to_predicted < threshold))
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        tag = f'{threshold * 1000:g}mm'
        report.update({f'precision_{tag}': precision * 100, f'recall_{tag}': recall * 100,
                       f'f1_{tag}': f1 * 100})
    return report


def official(mesh: o3d.geometry.TriangleMesh, scene: Path, seed: int = 0) -> dict:
    """Official scores of a mesh already in the reference frame (metres)."""
    o3d.utility.random.seed(seed)
    reference_mesh = o3d.io.read_triangle_mesh(str(scene / 'mesh/gt_mesh.ply'))
    reference = np.asarray(reference_mesh.sample_points_poisson_disk(SAMPLES).points)
    icp = o3d.pipelines.registration.registration_icp(
        o3d.geometry.PointCloud(reference_mesh.vertices), o3d.geometry.PointCloud(mesh.vertices), .02,
        np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10))
    mesh = o3d.geometry.TriangleMesh(mesh)
    if icp.fitness > .99:
        mesh.transform(np.linalg.inv(icp.transformation))
    volume = np.load(scene / 'visibility_mask.npy', allow_pickle=True).item()
    mesh.remove_vertices_by_mask(~visible(np.asarray(mesh.vertices), volume))
    if len(mesh.triangles):
        predicted = np.asarray(mesh.sample_points_poisson_disk(SAMPLES).points)
    else:
        predicted = np.random.default_rng(seed).permutation(np.asarray(mesh.vertices))[:SAMPLES]
    if not len(predicted):
        raise ValueError('No predicted surface inside the visibility volume')
    return {**scores(predicted, reference), 'icp_fitness': float(icp.fitness)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mesh', type=Path)
    p.add_argument('--scene', type=Path, required=True, help='MobileBrick scene with mesh/, pose/, visibility_mask.npy')
    p.add_argument('--trajectory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    center, target, rotation, scale, _ = similarity(a.trajectory, a.scene / 'pose')
    mesh = o3d.io.read_triangle_mesh(str(a.mesh))
    if not len(mesh.triangles):
        p.error(f'Empty mesh: {a.mesh}')
    vertices = (np.asarray(mesh.vertices) - center) @ rotation * scale + target
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    report = official(mesh, a.scene)
    reference = np.asarray(o3d.io.read_triangle_mesh(str(a.scene / 'mesh/gt_mesh.ply')).vertices)
    shaped, _ = refine_alignment(vertices, reference, (.01, .003))
    mesh.vertices = o3d.utility.Vector3dVector(shaped)
    report.update({f'scaled_{key}': value for key, value in official(mesh, a.scene).items()})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
