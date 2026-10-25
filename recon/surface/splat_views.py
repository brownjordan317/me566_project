#!/usr/bin/env python3
"""Render a fitted Gaussian model from virtual cameras that face the mesh head-on.

Texturing from the photos alone leaves surface blank wherever every photo sees it at a grazing angle,
is occluded, or never looks at it: on jeep about 8% of on-surface texels end up with no colour, against
only 1.1% of the surface that no view observed at all (docs/PIPELINE.md). The photos' coverage, not the
capture's, is the limit.

The optimiser has already fitted a view-dependent radiance model to every photo, so it can be asked for
the view each texel actually wants. Virtual cameras are placed on a sphere around the object looking
inwards - evenly by a Fibonacci spiral, plus directions picked to face the mesh's worst-covered normals -
and rendered. The renders and a COLMAP text model are written in the layout `recon.surface.texture`
already consumes, so texturing is unchanged apart from being pointed at this folder.

Runs in the gsplat environment (make setup-gaussian), since it rasterises Gaussians.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation


def fibonacci_directions(count: int) -> np.ndarray:
    """`count` roughly even directions on the unit sphere."""
    index = np.arange(count) + .5
    phi = np.arccos(1 - 2 * index / count)
    theta = np.pi * (1 + 5 ** .5) * index
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)


def worst_covered_directions(mesh: o3d.geometry.TriangleMesh, poses: list[np.ndarray],
                             count: int) -> np.ndarray:
    """Face normals that the real cameras see most obliquely, clustered into `count` directions."""
    mesh.compute_triangle_normals()
    normals = np.asarray(mesh.triangle_normals)
    centres = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)].mean(1)
    best = np.full(len(normals), -1.)
    for pose in poses:
        to_camera = pose[:3, 3] - centres
        to_camera /= np.linalg.norm(to_camera, axis=1, keepdims=True) + 1e-9
        best = np.maximum(best, (normals * to_camera).sum(1))      # 1 = head-on, 0 = edge-on
    poor = normals[best < .35]                                      # never seen better than ~70 degrees
    if len(poor) < count:
        return np.zeros((0, 3))
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(poor))
    labels = np.asarray(cloud.cluster_dbscan(eps=.35, min_points=10))
    picked = []
    for label in range(labels.max() + 1):
        direction = poor[labels == label].mean(0)
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            picked.append(direction / norm)
    order = np.argsort([-(labels == label).sum() for label in range(labels.max() + 1)])
    return np.array([picked[i] for i in order[:count] if i < len(picked)]) if picked else np.zeros((0, 3))


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera-to-world pose looking from `eye` at `target`, y roughly down."""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    up = np.array([0., 0., 1.]) if abs(forward[2]) < .95 else np.array([0., 1., 0.])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, down, forward, eye
    return pose


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('gaussians', type=Path, help='3DGS-layout PLY (gaussian_refine --export-gaussians)')
    p.add_argument('--mesh', type=Path, required=True, help='Mesh to be textured; sets the framing')
    p.add_argument('--output', type=Path, required=True, help='Folder for frames/, sparse_txt/ and trajectory.csv')
    p.add_argument('--trajectory', type=Path, help='Real cameras, to aim extra views at what they miss')
    p.add_argument('--even', type=int, default=32, help='Evenly spread virtual views')
    p.add_argument('--targeted', type=int, default=12, help='Extra views aimed at the worst-covered normals')
    p.add_argument('--size', type=int, default=1600, help='Rendered image size (square)')
    p.add_argument('--margin', type=float, default=1.25, help='Framing: object radius x this fits the image')
    p.add_argument('--sh-degree', type=int, default=3)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    from gsplat import rasterization
    from recon.depth.gaussian_refine import load_gaussians

    mesh = o3d.io.read_triangle_mesh(str(a.mesh))
    vertices = np.asarray(mesh.vertices)
    centre = (vertices.min(0) + vertices.max(0)) / 2
    radius = float(np.linalg.norm(vertices - centre, axis=1).max())
    directions = fibonacci_directions(a.even)
    if a.trajectory is not None and a.targeted > 0:
        from recon.common.trajectory import load_poses
        poses = []
        for rotation, position in load_poses(a.trajectory).values():
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = rotation, position
            poses.append(pose)
        extra = worst_covered_directions(mesh, poses, a.targeted)
        if len(extra):
            directions = np.concatenate([directions, extra])
    gaussians = load_gaussians(a.gaussians, a.device)
    gaussians.sh_degree = a.sh_degree
    # One focal for all virtual views: the object's bounding sphere fills the frame.
    # Cameras sit at 2.2 x radius, so the object's radius projects to focal / 2.2 pixels; that must fit
    # half the image with the margin to spare.
    distance = 2.2 * radius
    focal = a.size * (distance / radius) / 2 / a.margin
    K = np.array([[focal, 0, a.size / 2 - .5], [0, focal, a.size / 2 - .5], [0, 0, 1.]])
    frames = a.output / 'frames'
    frames.mkdir(parents=True, exist_ok=True)
    lines, camera_lines, camera_poses = [], [], []
    for index, direction in enumerate(directions):
        eye = centre + direction * distance
        pose = look_at(eye, centre)
        camera_poses.append(pose)
        rotation = pose[:3, :3].T                              # world-to-camera
        translation = -rotation @ pose[:3, 3]
        viewmat = np.eye(4)
        viewmat[:3, :3], viewmat[:3, 3] = rotation, translation
        with torch.no_grad():
            colours = torch.cat([gaussians.sh0, gaussians.shN], 1)
            image, _, _ = rasterization(
                gaussians.means, torch.nn.functional.normalize(gaussians.quats, dim=-1),
                torch.exp(gaussians.scales), torch.sigmoid(gaussians.opacities), colours,
                torch.tensor(viewmat, dtype=torch.float32, device=a.device)[None],
                torch.tensor(K, dtype=torch.float32, device=a.device)[None],
                a.size, a.size, sh_degree=a.sh_degree, render_mode='RGB', packed=False)
        rgb = (image[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(frames / f'{index:08d}.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        q = Rotation.from_matrix(rotation).as_quat()            # x, y, z, w
        # Each image needs a non-empty observations line: our COLMAP reader drops blank lines and then
        # pairs what is left, so a blank second line would shift every image onto the wrong camera.
        lines += [f'{index + 1} {q[3]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} '
                  f'{translation[0]:.9f} {translation[1]:.9f} {translation[2]:.9f} '
                  f'{index + 1} {index:08d}.png', '0.0 0.0 -1']
        camera_lines.append(f'{index + 1} PINHOLE {a.size} {a.size} {focal:.9f} {focal:.9f} '
                            f'{a.size / 2 - .5:.9f} {a.size / 2 - .5:.9f}')
    model = a.output / 'sparse_txt'
    model.mkdir(parents=True, exist_ok=True)
    (model / 'cameras.txt').write_text('\n'.join(camera_lines) + '\n')
    (model / 'images.txt').write_text('\n'.join(lines) + '\n')
    (model / 'points3D.txt').write_text('')
    # Our trajectory CSV holds the camera-to-world rotation and the camera centre (recon.common.trajectory),
    # which is not COLMAP's world-to-camera convention used in images.txt above.
    rows = ['frame,qw,qx,qy,qz,tx,ty,tz']
    for index, (direction, pose) in enumerate(zip(directions, camera_poses)):
        q = Rotation.from_matrix(pose[:3, :3]).as_quat()        # x, y, z, w
        eye = pose[:3, 3]
        rows.append(f'{index},{q[3]:.9f},{q[0]:.9f},{q[1]:.9f},{q[2]:.9f},'
                    f'{eye[0]:.9f},{eye[1]:.9f},{eye[2]:.9f}')
    (a.output / 'trajectory.csv').write_text('\n'.join(rows) + '\n')
    (a.output / 'splat_views.json').write_text(json.dumps(
        {'views': len(directions), 'even': a.even, 'targeted': len(directions) - a.even,
         'size': a.size, 'radius': radius, 'focal': float(focal)}, indent=2) + '\n')
    print(f'{len(directions)} virtual views ({a.even} even, {len(directions) - a.even} targeted) '
          f'at {a.size}px -> {a.output}')


if __name__ == '__main__':
    main()
