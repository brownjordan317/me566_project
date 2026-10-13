#!/usr/bin/env python3
"""Same-camera baseline: 2D Gaussian Splatting trained on our estimated cameras.

2DGS (Huang et al., "2D Gaussian Splatting for Geometrically Accurate Radiance
Fields", SIGGRAPH 2024; https://github.com/hbb1/2d-gaussian-splatting, Gaussian-
Splatting License: research and evaluation only) is run unmodified from
third_party/ in its own environment (--python; it needs CUDA extensions built
against PyTorch with CUDA 12). It receives exactly what our pipeline's surface
stages receive: the working frames and our COLMAP model (cameras, poses, sparse
points) from an existing run. So the comparison isolates surface reconstruction.

Steps, following the 2DGS paper's DTU recipe (scripts/dtu_eval.py):
- the model is moved and scaled so the object landmarks fit a unit sphere (the
  normalisation its DTU data uses), and written as images/ + sparse/0; 2DGS ignores
  the principal point, so each image is shifted to centre it and cropped (with the
  dataset's DTU cameras it is about 12 px off at 800 px; ignoring it put the surface
  about 3 mm off on scan24);
- train.py: 30k iterations, --depth_ratio 1.0 --lambda_dist 1000, full working resolution;
- render.py: TSDF mesh with voxel 0.004, truncation 0.016, its largest cluster kept
  (fuse_post.ply); the depth cut-off is set from the cameras' distance to the object;
- meshing needs Open3D <= 0.19 in the 2DGS venv: 0.20's legacy ScalableTSDFVolume
  returns empty meshes (checked on a synthetic plane);
- the mesh is moved back to map units and culled with the object masks as in 2DGS's
  DTU evaluation (scripts/eval_dtu/evaluate_single_scene.py, cull_scan): a vertex is
  kept if it falls inside the mask, dilated by 24 px at 1600 px width, in every view
  that sees it in frame.
The result is RUN/mesh/object.ply with the source run's trajectory, so the standard
scorers apply (`recon.evaluation.metrics.{mesh,mobilebrick,dtu}`).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from recon.common.camera import frame_intrinsics
from recon.common.trajectory import load_poses


def read_images(model: Path) -> list[list[str]]:
    lines = [line for line in (model / 'images.txt').read_text().splitlines() if line and not line.startswith('#')]
    return [line.split() for line in lines[::2]]


def normalisation(landmarks: Path) -> tuple[np.ndarray, float]:
    points = np.asarray(o3d.io.read_point_cloud(str(landmarks)).points)
    center = np.median(points, axis=0)
    return center, float(np.quantile(np.linalg.norm(points - center, axis=1), .99))


def write_workspace(source: Path, sequence: Path, workspace: Path,
                    init_points: list[Path] = (), max_points: int = 300_000) -> tuple[np.ndarray, float, float]:
    """images/ and a normalised sparse/0 for 2DGS; returns centre, radius and depth cut-off.

    With init_points (our coloured point clouds), they replace the sparse SfM points as the
    Gaussians' starting positions: a warm start from our reconstruction."""
    model = source / 'poses/sparse_txt'
    center, radius = normalisation(source / 'poses/export_object/landmarks.ply')
    (workspace / 'sparse/0').mkdir(parents=True, exist_ok=True)
    (workspace / 'images').mkdir(exist_ok=True)
    # 2DGS reads PINHOLE only and, like 3DGS, ignores the principal point (it assumes the image centre).
    # (COLMAP convention: the centre is (w / 2, h / 2).) So every image is shifted to put its principal point at the centre and cropped by a margin that
    # covers the largest shift, and its camera is written for the cropped image.
    cameras = {}
    for line in (model / 'cameras.txt').read_text().splitlines():
        fields = line.split()
        if not fields or line.startswith('#'):
            continue
        width, height = int(fields[2]), int(fields[3])
        if fields[1] == 'SIMPLE_PINHOLE':
            fx = fy = float(fields[4]); cx, cy = map(float, fields[5:7])
        else:
            fx, fy, cx, cy = map(float, fields[4:8])
        cameras[fields[0]] = (width, height, fx, fy, cx, cy)
    shift = max(max(abs(cx - w / 2), abs(cy - h / 2)) for w, h, _, _, cx, cy in cameras.values())
    margin = int(np.ceil(shift))
    (workspace / 'sparse/0/cameras.txt').write_text(''.join(
        f'{key} PINHOLE {w - 2 * margin} {h - 2 * margin} {fx} {fy} {(w - 2 * margin) / 2} {(h - 2 * margin) / 2}\n'
        for key, (w, h, fx, fy, cx, cy) in cameras.items()))
    lines, distances = [], []
    for fields in read_images(model):
        qw, qx, qy, qz, tx, ty, tz = map(float, fields[1:8])
        rotation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        translation = (rotation @ center + np.array([tx, ty, tz])) / radius
        distances.append(np.linalg.norm(translation))
        lines += [' '.join(fields[:5] + [f'{value:.10f}' for value in translation] + fields[8:]), '']
        target = workspace / 'images' / fields[9]
        if not target.exists():
            w, h, _, _, cx, cy = cameras[fields[8]]
            frame = cv2.imread(str(sequence / 'frames' / fields[9]))
            moved = cv2.warpAffine(frame, np.float32([[1, 0, w / 2 - cx], [0, 1, h / 2 - cy]]), (w, h),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            cv2.imwrite(str(target), moved[margin:h - margin, margin:w - margin])
    (workspace / 'sparse/0/images.txt').write_text('\n'.join(lines) + '\n')
    points = []
    if init_points:
        clouds = [o3d.io.read_point_cloud(str(path)) for path in init_points]
        xyz = np.concatenate([np.asarray(cloud.points) for cloud in clouds])
        rgb = np.concatenate([np.asarray(cloud.colors) if cloud.has_colors() else np.full((len(cloud.points), 3), .5)
                              for cloud in clouds])
        pick = np.random.default_rng(0).permutation(len(xyz))[:max_points]
        for index, (point, colour) in enumerate(zip((xyz[pick] - center) / radius, rgb[pick] * 255), 1):
            points.append(f'{index} {point[0]:.8f} {point[1]:.8f} {point[2]:.8f} '
                          f'{int(colour[0])} {int(colour[1])} {int(colour[2])} 0')
    for line in ([] if init_points else (model / 'points3D.txt').read_text().splitlines()):
        if not line or line.startswith('#'):
            continue
        fields = line.split()
        xyz = (np.array(list(map(float, fields[1:4]))) - center) / radius
        points.append(' '.join([fields[0]] + [f'{value:.8f}' for value in xyz] + ['128', '128', '128'] + fields[7:]))
    (workspace / 'sparse/0/points3D.txt').write_text('\n'.join(points) + '\n')
    return center, radius, float(np.max(distances)) + 1.5


def cull(mesh: o3d.geometry.TriangleMesh, model: Path, trajectory: Path, masks: Path) -> o3d.geometry.TriangleMesh:
    """2DGS's DTU culling: keep vertices inside the dilated mask in every view seeing them."""
    vertices = np.asarray(mesh.vertices)
    keep = np.ones(len(vertices), dtype=bool)
    intrinsics, poses = frame_intrinsics(model), load_poses(trajectory)
    for frame, K in intrinsics.items():
        mask = cv2.imread(str(masks / f'{frame:08d}.png'), cv2.IMREAD_GRAYSCALE)
        if mask is None or frame not in poses:
            continue
        height, width = mask.shape
        radius = max(1, round(24 * width / 1600))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        mask = cv2.dilate((mask > 127).astype(np.uint8), kernel) > 0
        rotation, center = poses[frame]
        camera = (vertices - center) @ rotation
        z = np.maximum(camera[:, 2], 1e-6)
        u = np.rint(K[0, 0] * camera[:, 0] / z + K[0, 2]).astype(np.int64)
        v = np.rint(K[1, 1] * camera[:, 1] / z + K[1, 2]).astype(np.int64)
        inside = (camera[:, 2] > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        hit = np.zeros(len(vertices), dtype=bool)
        hit[inside] = mask[v[inside], u[inside]]
        keep &= hit | ~inside
    mesh.remove_vertices_by_mask(~keep)
    return mesh


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('source', type=Path, help='Finished run of ours (poses/sparse_txt, export, export_object)')
    p.add_argument('--sequence', type=Path, required=True)
    p.add_argument('--masks', type=Path, help='Default: sequence/masks')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--python', default=str(Path.home() / '.venvs/me566-2dgs/bin/python'))
    p.add_argument('--repo', type=Path, default=Path('third_party/2d-gaussian-splatting'))
    p.add_argument('--iterations', type=int, default=30000)
    p.add_argument('--init-points', type=Path, nargs='+', default=[],
                   help='Start from these point clouds (e.g. RUN/points/{object,scene}.ply) instead of the SfM points')
    p.add_argument('--train-args', default='', help="Extra train.py options as one string, e.g. '--opacity_reset_interval 100000'")
    a = p.parse_args()
    workspace = a.output / '2dgs'
    center, radius, depth_trunc = write_workspace(a.source, a.sequence, workspace, a.init_points)
    repo = a.repo.resolve()
    started = time.time()
    trained_model = workspace / f'model/point_cloud/iteration_{a.iterations}/point_cloud.ply'
    reused = trained_model.is_file()                                        # resume after a meshing failure
    if not reused:
        subprocess.run([a.python, 'train.py', '-s', str(workspace.resolve()), '-m', str((workspace / 'model').resolve()),
                        '--quiet', '--test_iterations', '-1', '--depth_ratio', '1.0', '-r', '1', '--lambda_dist', '1000',
                        '--iterations', str(a.iterations), '--save_iterations', str(a.iterations), *a.train_args.split()],
                       cwd=repo, check=True)
    trained = time.time()
    subprocess.run([a.python, 'render.py', '--iteration', str(a.iterations), '-s', str(workspace.resolve()),
                    '-m', str((workspace / 'model').resolve()), '--quiet', '--skip_train', '--skip_test',
                    '--depth_ratio', '1.0', '--num_cluster', '1', '--voxel_size', '0.004', '--sdf_trunc', '0.016',
                    '--depth_trunc', f'{depth_trunc:.3f}'], cwd=repo, check=True)
    meshed = time.time()
    mesh = o3d.io.read_triangle_mesh(str(workspace / f'model/train/ours_{a.iterations}/fuse_post.ply'))
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * radius + center)
    mesh = cull(mesh, a.source / 'poses/sparse_txt', a.source / 'poses/export/trajectory.csv',
                a.masks or a.sequence / 'masks')
    (a.output / 'mesh').mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(a.output / 'mesh/object.ply'), mesh)
    shutil.copytree(a.source / 'poses/export', a.output / 'poses/export', dirs_exist_ok=True)
    report = {'source': str(a.source), 'init_points': [str(path) for path in a.init_points], 'iterations': a.iterations,
              'train_args': a.train_args, 'train_s': trained - started if not reused else None, 'reused_model': reused, 'mesh_s': meshed - trained,
              'normalisation_center': center.tolist(), 'normalisation_radius': radius, 'depth_trunc': depth_trunc,
              'faces': len(mesh.triangles)}
    (a.output / 'twodgs.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
