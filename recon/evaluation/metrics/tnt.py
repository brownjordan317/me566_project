#!/usr/bin/env python3
"""Tanks and Temples evaluation (training scenes), ported to current Open3D.

Port of the benchmark's python_toolbox/evaluation (run.py, registration.py,
evaluation.py; https://github.com/isl-org/TanksAndTemples, MIT, Copyright (c)
2017 Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, Vladlen Koltun), in the variant
2DGS ships (scripts/eval_tnt), which scores mesh vertices plus face centres. The
original calls Open3D's removed `o3d.registration` API; the steps are unchanged:

1. our camera centres are registered to the scene's reference COLMAP trajectory
   (<scene>_COLMAP_SfM.log, moved by <scene>_trans.txt) by RANSAC over frame
   correspondences with scale (6 points, 0.2 threshold);
2. three ICP refinements with scale against the ground-truth cloud inside the
   crop volume (<scene>.json): voxel tau at threshold 80 tau, voxel tau/2 at 20 tau,
   then uniform subsampling at 2 tau;
3. both clouds are cropped, voxel-downsampled at tau/2, and precision, recall and
   F-score are the shares of points within tau of the other cloud.

tau is the benchmark's per-scene threshold (metres). Frames are matched by order:
frame i of our run is image i of the scene's image set, as in the reference log.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from recon.common.trajectory import load_poses

TAU = {'Barn': .01, 'Caterpillar': .005, 'Church': .025, 'Courthouse': .025,
       'Ignatius': .003, 'Meetingroom': .01, 'Truck': .005}
MAX_POINTS = 4e6
registration = o3d.pipelines.registration


def read_log(path: Path) -> list[np.ndarray]:
    """Camera-to-world matrices of a Redwood .log trajectory."""
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return [np.array([[float(v) for v in lines[i + j].split()] for j in range(1, 5)]) for i in range(0, len(lines), 5)]


def points_of(cloud_or_mesh: Path) -> o3d.geometry.PointCloud:
    """Mesh vertices plus face centres (as in 2DGS's evaluation), or a point cloud as is."""
    mesh = o3d.io.read_triangle_mesh(str(cloud_or_mesh))
    if len(mesh.triangles):
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        points = np.concatenate([vertices, vertices[faces].mean(axis=1)])
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return o3d.io.read_point_cloud(str(cloud_or_mesh))


def crop_and_downsample(cloud, volume, method: str, voxel: float = .01, transform=np.eye(4)):
    cloud = copy.deepcopy(cloud)
    cloud.transform(transform)
    cloud = volume.crop_point_cloud(cloud)
    if method == 'voxel':
        return cloud.voxel_down_sample(voxel)
    if len(cloud.points) > MAX_POINTS:
        return cloud.uniform_down_sample(int(round(len(cloud.points) / MAX_POINTS)))
    return cloud


def refine(source, target, initial, volume, method: str, voxel: float, threshold: float, iterations: int):
    s = crop_and_downsample(source, volume, method, voxel, initial)
    t = crop_and_downsample(target, volume, method, voxel)
    result = registration.registration_icp(
        s, t, threshold, np.eye(4), registration.TransformationEstimationPointToPoint(True),
        registration.ICPConvergenceCriteria(1e-6, 1e-6, iterations))
    return result.transformation @ initial


def trajectory_alignment(ours: dict[int, np.ndarray], reference: list[np.ndarray], gt_transform: np.ndarray):
    frames = [index for index in range(len(reference)) if index in ours]
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array([ours[f] for f in frames])))
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array([reference[f][:3, 3] for f in frames])))
    target.transform(gt_transform)
    pairs = o3d.utility.Vector2iVector(np.array([[i, i] for i in range(len(frames))]))
    result = registration.registration_ransac_based_on_correspondence(
        source, target, pairs, .2, registration.TransformationEstimationPointToPoint(True), 6,
        criteria=registration.RANSACConvergenceCriteria(100000, .999))
    return result.transformation, len(frames)


def evaluate(mesh: Path, trajectory: Path, scene_dir: Path, scene: str) -> dict:
    tau = TAU[scene]
    ours = {frame: center for frame, (_, center) in load_poses(trajectory).items()}
    reference = read_log(scene_dir / f'{scene}_COLMAP_SfM.log')
    gt_transform = np.loadtxt(scene_dir / f'{scene}_trans.txt')
    volume = o3d.visualization.read_selection_polygon_volume(str(scene_dir / f'{scene}.json'))
    source, target = points_of(mesh), o3d.io.read_point_cloud(str(scene_dir / f'{scene}.ply'))
    transform, matched = trajectory_alignment(ours, reference, gt_transform)
    transform = refine(source, target, transform, volume, 'voxel', tau, 80 * tau, 20)
    transform = refine(source, target, transform, volume, 'voxel', tau / 2, 20 * tau, 20)
    transform = refine(source, target, transform, volume, 'uniform', 0, 2 * tau, 20)
    s = copy.deepcopy(source)
    s.transform(transform)
    s = volume.crop_point_cloud(s).voxel_down_sample(tau / 2)
    t = volume.crop_point_cloud(copy.deepcopy(target)).voxel_down_sample(tau / 2)
    to_target = np.asarray(s.compute_point_cloud_distance(t))
    to_source = np.asarray(t.compute_point_cloud_distance(s))
    precision = float(np.mean(to_target < tau)) if len(to_target) else 0.
    recall = float(np.mean(to_source < tau)) if len(to_source) else 0.
    fscore = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return {'scene': scene, 'tau_m': tau, 'precision': precision * 100, 'recall': recall * 100,
            'fscore': fscore * 100, 'matched_frames': matched, 'reference_frames': len(reference),
            'transform': transform.tolist()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mesh', type=Path)
    p.add_argument('--trajectory', type=Path, required=True)
    p.add_argument('--scene-dir', type=Path, required=True, help='Folder with <scene>.ply, .json, _trans.txt, _COLMAP_SfM.log')
    p.add_argument('--scene', choices=TAU, help='Default: the scene folder name')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = evaluate(a.mesh, a.trajectory, a.scene_dir, a.scene or a.scene_dir.name)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'transform'}, indent=2))


if __name__ == '__main__':
    main()
