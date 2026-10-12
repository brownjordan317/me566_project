"""Compact binary PLY output for final meshes and point clouds.

Open3D writes positions and normals as 64-bit doubles. Blender, MeshLab and
CloudCompare all expect 32-bit floats, and doubles double the file size for no
gain at millimetre scale. Colours are written as uchar red/green/blue, which
Blender imports as a colour attribute.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None,
              normals: np.ndarray | None = None, triangles: np.ndarray | None = None) -> None:
    """Write float32 xyz (+ normals), uchar rgb in [0, 1] input, and int32 faces."""
    points = np.asarray(points, dtype='<f4').reshape(-1, 3)
    fields = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
    if normals is not None and len(normals):
        fields += [('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4')]
    if colors is not None and len(colors):
        fields += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertices = np.empty(len(points), dtype=fields)
    vertices['x'], vertices['y'], vertices['z'] = points.T
    if 'nx' in vertices.dtype.names:
        vertices['nx'], vertices['ny'], vertices['nz'] = np.asarray(normals, dtype='<f4').T
    if 'red' in vertices.dtype.names:
        rgb = np.clip(np.round(np.asarray(colors) * 255), 0, 255).astype('u1')
        vertices['red'], vertices['green'], vertices['blue'] = rgb.T
    names = {'<f4': 'float', 'u1': 'uchar'}
    header = ['ply', 'format binary_little_endian 1.0', f'element vertex {len(points)}']
    header += [f'property {names[kind]} {name}' for name, kind in fields]
    faces = None
    if triangles is not None and len(triangles):
        faces = np.empty(len(triangles), dtype=[('count', 'u1'), ('indices', '<i4', (3,))])
        faces['count'], faces['indices'] = 3, triangles
        header += [f'element face {len(faces)}', 'property list uchar int vertex_indices']
    header.append('end_header')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        stream.write(('\n'.join(header) + '\n').encode('ascii'))
        stream.write(vertices.tobytes())
        if faces is not None:
            stream.write(faces.tobytes())


def write_mesh(path: Path, mesh) -> None:
    """Write a legacy Open3D TriangleMesh with 32-bit floats."""
    write_ply(path, np.asarray(mesh.vertices), np.asarray(mesh.vertex_colors),
              np.asarray(mesh.vertex_normals), np.asarray(mesh.triangles))


def write_cloud(path: Path, cloud) -> None:
    """Write a legacy Open3D PointCloud with 32-bit floats."""
    write_ply(path, np.asarray(cloud.points), np.asarray(cloud.colors), np.asarray(cloud.normals))
