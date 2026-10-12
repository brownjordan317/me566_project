#!/usr/bin/env python3
"""Texture a mesh from the calibrated full-resolution photos.

Vertex colours can only be as sharp as the mesh is dense; a texture keeps the
photos' detail. The mesh is UV-unwrapped with xatlas (https://github.com/jpcy/xatlas,
MIT; Python bindings by mworchel), which tolerates the non-manifold vertices marching
cubes leaves (Open3D's UVAtlas port crashes on them), and a subset of frames spread
over the sequence is
projected onto the texture and blended (TriangleMesh.project_images_to_albedo,
Open3D; MIT). The result is written as a binary glTF (.glb), which Blender
imports with its texture.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import xatlas

from recon.common.camera import frame_intrinsics
from recon.common.progress import done, spinner
from recon.common.trajectory import load_poses


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mesh', type=Path)
    p.add_argument('--model', type=Path, required=True, help='COLMAP text model (working-frame intrinsics)')
    p.add_argument('--trajectory', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True, help='Numbered full-resolution frames')
    p.add_argument('--sequence', type=Path, required=True,
                   help='Prepared sequence; its frames give the width the intrinsics refer to')
    p.add_argument('--output', type=Path, required=True, help='.glb file')
    p.add_argument('--frames', type=int, default=24, help='Frames projected, spread over the sequence')
    p.add_argument('--texture-size', type=int, default=4096)
    p.add_argument('--max-faces', type=int, default=300000, help='Simplify larger meshes before unwrapping')
    a = p.parse_args()
    legacy = o3d.io.read_triangle_mesh(str(a.mesh))
    if len(legacy.triangles) > a.max_faces:
        legacy = legacy.simplify_quadric_decimation(a.max_faces)
    legacy.remove_degenerate_triangles()
    legacy.remove_unreferenced_vertices()
    with spinner(f'UV unwrapping {len(legacy.triangles):,} faces'):
        mapping, faces, uvs = xatlas.parametrize(np.asarray(legacy.vertices, np.float32),
                                                 np.asarray(legacy.triangles, np.uint32))
    # xatlas splits vertices along UV seams; rebuild the mesh on its vertex mapping.
    vertices = np.asarray(legacy.vertices)[mapping]
    seamed = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                       o3d.utility.Vector3iVector(faces.astype(np.int32)))
    seamed.compute_vertex_normals()
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(seamed)
    mesh.triangle.texture_uvs = o3d.core.Tensor(uvs[faces].astype(np.float32))
    working_width = cv2.imread(str(next(iter(sorted((a.sequence / 'frames').glob('*.png')))))).shape[1]
    poses = load_poses(a.trajectory)
    intrinsics = frame_intrinsics(a.model)
    files = {int(path.stem): path for path in a.images.iterdir()
             if path.suffix.lower() in ('.png', '.jpg', '.jpeg') and path.stem.isdigit()}
    frames = [frame for frame in sorted(poses) if frame in files]
    chosen = [frames[i] for i in np.unique(np.linspace(0, len(frames) - 1, min(a.frames, len(frames))).round().astype(int))]
    images, Ks, extrinsics = [], [], []
    for frame in chosen:
        image = cv2.cvtColor(cv2.imread(str(files[frame])), cv2.COLOR_BGR2RGB)
        scale = image.shape[1] / working_width
        K = intrinsics[frame].copy()
        K[:2] *= scale
        K[0, 2] += .5 * (scale - 1)                                        # pixel-centre convention
        K[1, 2] += .5 * (scale - 1)
        rotation, centre = poses[frame]                                    # camera-to-world rotation, centre
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3], world_to_camera[:3, 3] = rotation.T, -rotation.T @ centre
        images.append(o3d.t.geometry.Image(np.ascontiguousarray(image)))
        Ks.append(o3d.core.Tensor(K, o3d.core.float32))
        extrinsics.append(o3d.core.Tensor(world_to_camera, o3d.core.float32))
    with spinner(f'Projecting {len(chosen)} photos onto a {a.texture_size}px texture'):
        mesh.project_images_to_albedo(images, Ks, extrinsics, a.texture_size)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.t.io.write_triangle_mesh(str(a.output), mesh):
        raise OSError(a.output)
    report = {'faces': len(legacy.triangles), 'frames': chosen, 'texture_size': a.texture_size}
    a.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    done(f'Texture: {len(chosen)} photos onto {len(legacy.triangles):,} faces -> {a.output.name}')


if __name__ == '__main__':
    main()
