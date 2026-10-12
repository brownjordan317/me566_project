#!/usr/bin/env python3
"""Open a mesh or point cloud in Open3D's native viewer, in its own colours.

    make view_ply PATH=runs/real/mobilebrick_sweep_support/jeep/mesh/object.ply
    python3 -m recon.apps.view_ply path/to/file.{ply,glb,obj}

The viewer (Open3D's O3DVisualizer) renders on the GPU with OpenGL, so full
meshes and multi-million-point clouds stay interactive without decimation.
Meshes show their vertex colours, or their texture for .glb/.obj models;
point clouds show their point colours. Uncoloured geometry is shaded by
height. The settings panel (lighting, point size, background) is on the right;
mouse: left drag rotates, right drag pans, wheel zooms.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def height_colours(points: np.ndarray) -> np.ndarray:
    z = points[:, 2]
    t = (z - z.min()) / max(float(np.ptp(z)), 1e-12)
    return np.stack((t, .3 + .4 * (1 - abs(2 * t - 1)), 1 - t), axis=1)


def load(path: Path):
    """(name, geometry, summary) for the viewer: a textured model, a mesh or a point cloud."""
    if path.suffix.lower() in ('.glb', '.gltf', '.obj'):
        model = o3d.io.read_triangle_model(str(path))
        if len(model.meshes):
            faces = sum(len(item.mesh.triangles) for item in model.meshes)
            return model, f'{faces:,} faces, textured'
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.triangles):
        if not mesh.has_vertex_colors():
            mesh.vertex_colors = o3d.utility.Vector3dVector(height_colours(np.asarray(mesh.vertices)))
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        return mesh, f'{len(mesh.triangles):,} faces'
    cloud = o3d.io.read_point_cloud(str(path))
    if not len(cloud.points):
        raise ValueError(f'No geometry in {path}')
    if not cloud.has_colors():
        cloud.colors = o3d.utility.Vector3dVector(height_colours(np.asarray(cloud.points)))
    return cloud, f'{len(cloud.points):,} points'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('path', type=Path)
    p.add_argument('--point-size', type=int, default=2)
    a = p.parse_args()
    if not a.path.is_file():
        p.error(f'No such file: {a.path}')
    geometry, summary = load(a.path)
    print(f'{a.path.name}: {summary}')
    o3d.visualization.draw(geometry, title=f'{a.path.name}: {summary}',
                           show_skybox=False, bg_color=(1., 1., 1., 1.), point_size=a.point_size,
                           show_ui=True, width=1600, height=1000)


if __name__ == '__main__':
    main()
