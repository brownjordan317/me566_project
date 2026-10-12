#!/usr/bin/env python3
"""Compare a reconstructed object mesh with a MobileBrick reference scan.

Two alignments are scored. The official one places the mesh with a similarity
fitted to our camera centres, so it also measures how well overall scale was
recovered. The shape metrics then refine that alignment, scale included, by
fitting the surface itself (ICP): from RGB alone absolute scale cannot be known,
and a 1% focal or scale offset otherwise reads as millimetres of surface error.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d
from scipy.spatial import cKDTree


def similarity(trajectory: Path, pose_dir: Path):
    """Similarity mapping our camera centres onto the reference ones.

    Closed-form least-squares rotation, translation and scale (Umeyama, TPAMI
    1991). Returns (estimate_center, reference_center, rotation, scale,
    per-camera residuals).
    """
    estimates, references = [], []
    with trajectory.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row.get('status') not in ('tracked', 'bootstrapped'):
                continue
            reference = pose_dir / f"{int(row['frame']):06d}.txt"
            if not reference.is_file():
                continue
            estimates.append([float(row[key]) for key in ('tx', 'ty', 'tz')])
            references.append(np.loadtxt(reference)[:3, 3])
    if len(estimates) < 3:
        raise ValueError('At least three matched camera poses are required')
    estimates, references = np.asarray(estimates), np.asarray(references)
    estimate_center = estimates.mean(axis=0)
    reference_center = references.mean(axis=0)
    u, singular, vt = np.linalg.svd(
        (estimates - estimate_center).T @ (references - reference_center))
    reflection = np.diag([1., 1., np.linalg.det(u @ vt)])
    rotation = u @ reflection @ vt
    scale = float(np.sum(singular * np.diag(reflection))
                  / np.sum((estimates - estimate_center) ** 2))
    aligned = (estimates - estimate_center) @ rotation * scale + reference_center
    pose_errors = np.linalg.norm(aligned - references, axis=1)
    return estimate_center, reference_center, rotation, scale, pose_errors


def pose_similarity(trajectory: Path, pose_dir: Path):
    """Similarity onto the reference cameras using their orientations as well as centres.

    Camera centres alone fix the rotation poorly when the cameras sit on a narrow
    cap far from the object (DTU: about 600 mm away, 30-degree field of view): a
    0.2-degree error there moves the object by millimetres. Each camera's
    orientation is solved far more precisely, so the rotation is the chordal mean
    of the per-camera rotations R_ref R_est^T (Hartley et al., "Rotation Averaging",
    IJCV 2013), and the scale is fitted to the centres given that rotation. Only
    valid when the intrinsics are right: an unmodelled principal-point offset tilts
    every estimated camera the same way. Returns the same tuple as similarity().
    """
    estimates, references, rotations = [], [], np.zeros((3, 3))
    with trajectory.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row.get('status') not in ('tracked', 'bootstrapped'):
                continue
            reference = pose_dir / f"{int(row['frame']):06d}.txt"
            if not reference.is_file():
                continue
            pose = np.loadtxt(reference)
            estimate = Rotation.from_quat([float(row[key]) for key in ('qx', 'qy', 'qz', 'qw')]).as_matrix()
            rotations += pose[:3, :3] @ estimate.T
            estimates.append([float(row[key]) for key in ('tx', 'ty', 'tz')])
            references.append(pose[:3, 3])
    if len(estimates) < 3:
        raise ValueError('At least three matched camera poses are required')
    estimates, references = np.asarray(estimates), np.asarray(references)
    u, _, vt = np.linalg.svd(rotations)
    rotation = (u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt).T     # row-vector convention
    estimate_center, reference_center = estimates.mean(axis=0), references.mean(axis=0)
    rotated = (estimates - estimate_center) @ rotation
    scale = float(np.sum(rotated * (references - reference_center)) / np.sum(rotated ** 2))
    pose_errors = np.linalg.norm(rotated * scale + reference_center - references, axis=1)
    return estimate_center, reference_center, rotation, scale, pose_errors


def icp_similarity(points: np.ndarray, reference: np.ndarray,
                   distances: tuple[float, ...]) -> np.ndarray:
    """4x4 similarity (rotation, translation, scale) fitting points onto reference points.

    Coarse-to-fine point-to-point ICP (Besl and McKay, TPAMI 1992) with Umeyama
    scale estimation, via Open3D's registration_icp.
    """
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(reference))
    transform = np.eye(4)
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
    for distance in distances:
        transform = o3d.pipelines.registration.registration_icp(
            source, target, distance, transform, estimation).transformation
    return transform


def refine_alignment(points: np.ndarray, reference: np.ndarray,
                     distances: tuple[float, ...]) -> tuple[np.ndarray, float]:
    """Coarse-to-fine scaled ICP of points onto reference points; returns (points, scale)."""
    transform = icp_similarity(points, reference, distances)
    scale = float(np.cbrt(np.linalg.det(transform[:3, :3])))
    return points @ transform[:3, :3].T + transform[:3, 3], scale


def evaluate(mesh_path: Path, reference_path: Path, trajectory: Path,
             pose_dir: Path, samples: int = 200000) -> dict:
    estimate_center, reference_center, rotation, scale, pose_errors = similarity(
        trajectory, pose_dir)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    reference = o3d.io.read_triangle_mesh(str(reference_path))
    if not len(mesh.triangles) or not len(reference.vertices):
        raise ValueError('Mesh or reference has no surface geometry')
    o3d.utility.random.seed(7)
    points = np.asarray(mesh.sample_points_uniformly(samples).points)
    points = (points - estimate_center) @ rotation * scale + reference_center
    vertices = np.asarray(reference.vertices)
    # Accuracy is the distance to the reference surface (its triangles). The
    # reference meshes are remeshed at about 1 mm edges, so distance to the
    # nearest vertex instead would score the reference itself at about 0.38 mm
    # median, a floor that hid real gains below it.
    surface_distance = o3d.t.geometry.RaycastingScene()
    surface_distance.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(reference))
    to_surface = lambda query: surface_distance.compute_distance(
        o3d.core.Tensor(query.astype(np.float32))).numpy().astype(np.float64)
    accuracy = to_surface(points)
    coverage = cKDTree(points).query(vertices, workers=-1)[0]
    surface = np.asarray(reference.sample_points_uniformly(400000).points)
    shaped, scale = refine_alignment(points, surface, (.01, .003))
    shape_accuracy = to_surface(shaped)
    # Balanced scores after shape alignment: precision (our surface near the reference),
    # recall (reference surface near ours) and their F-score, the usual way to judge an
    # accuracy/completeness trade-off (Knapitsch et al., Tanks and Temples, TOG 2017).
    shape_recall_distance = cKDTree(shaped).query(surface, workers=-1)[0]
    fscores = {}
    for millimetres in (1, 2):
        precision = float(np.mean(shape_accuracy < millimetres / 1000))
        recall = float(np.mean(shape_recall_distance < millimetres / 1000))
        fscores.update({f'precision_{millimetres}mm_pct': precision * 100, f'recall_{millimetres}mm_pct': recall * 100,
                        f'fscore_{millimetres}mm_pct': 200 * precision * recall / max(precision + recall, 1e-12)})
    faces = np.asarray(mesh.triangles)
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                                    faces[:, [2, 0]])), axis=1)
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    return {
        'localized_frames': len(pose_errors),
        'median_pose_error_mm': float(np.median(pose_errors) * 1000),
        'mesh_faces': len(mesh.triangles),
        'sampled_points': samples,
        'mesh_to_reference_median_mm': float(np.median(accuracy) * 1000),
        'mesh_to_reference_p90_mm': float(np.percentile(accuracy, 90) * 1000),
        'reference_vertices_within_10mm_pct': float(np.mean(coverage < .01) * 100),
        'shape_median_mm': float(np.median(shape_accuracy) * 1000),
        'shape_p90_mm': float(np.percentile(shape_accuracy, 90) * 1000),
        'size_error_pct': (1 / scale - 1) * 100,
        **fscores,
        'boundary_edges': int(np.count_nonzero(edge_counts == 1)),
        'nonmanifold_edges': int(np.count_nonzero(edge_counts > 2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mesh', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--pose-dir', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=200000)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    summary = evaluate(args.mesh, args.reference, args.trajectory,
                       args.pose_dir, args.samples)
    result = json.dumps(summary, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    print(result, end='')


if __name__ == '__main__':
    main()
