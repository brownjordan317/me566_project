#!/usr/bin/env python3
"""What an asset looks like, as numbers rather than an impression.

The benchmark scores cannot see the defects that fail an eye test. Its official metric is a 2.5 mm
threshold and it excludes everything outside the visibility volume, so holes, blank texture and inferred
colour are nearly free - which is why the two stages that most improve appearance, closure and splat
texturing, read as neutral or slightly negative there (docs/PIPELINE.md). An audit of the best-scoring
meshes found the defects were not floaters or roughness but holes and vertex colours.

So this reports the quantities that actually decide how an asset looks:

  closed        boundary edges, and whether the surface is edge- and vertex-manifold and orientable
  pieces        connected components, and the share of triangles outside the largest
  roughness     mean disagreement between adjacent face normals, which reads as a bumpy surface
  texture       share of on-surface texels with no colour, and sharpness where they have it
  provenance    share of the texture that was photographed rather than inferred

Nothing here needs ground truth, so it can be run on any capture.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def geometry_report(mesh: o3d.geometry.TriangleMesh) -> dict:
    mesh.compute_triangle_normals()
    triangles = np.asarray(mesh.triangles)
    labels, counts, _ = mesh.cluster_connected_triangles()
    counts = np.asarray(counts)
    boundary = len(np.asarray(mesh.get_non_manifold_edges(allow_boundary_edges=False)))
    normals = np.asarray(mesh.triangle_normals)
    owner, dots = {}, []
    for index, triangle in enumerate(triangles):
        for edge in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            key = (min(edge), max(edge))
            if key in owner:
                dots.append(abs(float(np.dot(normals[index], normals[owner[key]]))))
            else:
                owner[key] = index
    return {'faces': int(len(triangles)), 'boundary_edges': boundary,
            'closed': bool(boundary == 0 and mesh.is_edge_manifold() and mesh.is_vertex_manifold()),
            'edge_manifold': bool(mesh.is_edge_manifold()),
            'vertex_manifold': bool(mesh.is_vertex_manifold()),
            'orientable': bool(mesh.is_orientable()),
            'pieces': int(len(counts)),
            'stray_faces_pct': float((1 - counts.max() / counts.sum()) * 100) if len(counts) else 0.,
            'roughness': float(1 - np.mean(dots)) if dots else float('nan')}


def texture_report(path: Path, source_map: Path | None) -> dict:
    mesh = o3d.io.read_triangle_mesh(str(path), True)
    if not mesh.has_triangle_uvs() or not mesh.textures:
        return {}
    albedo = np.asarray(mesh.textures[-1])
    grey = albedo[..., :3].mean(-1).astype(np.float32) if albedo.ndim == 3 else albedo.astype(np.float32)
    uvs = np.asarray(mesh.triangle_uvs).reshape(-1, 2)
    height, width = grey.shape
    x = np.clip((uvs[:, 0] * (width - 1)).astype(int), 0, width - 1)
    y = np.clip((uvs[:, 1] * (height - 1)).astype(int), 0, height - 1)
    covered = np.zeros(grey.shape, bool)
    covered[y, x] = True
    painted = covered & (grey > 8)
    report = {'texture_size': int(width),
              'blank_on_surface_pct': float(np.mean(grey[covered] < 8) * 100) if covered.any() else 0.,
              'sharpness': float(cv2.Laplacian(grey, cv2.CV_32F)[painted].std()) if painted.any() else 0.}
    if source_map and source_map.is_file():
        source = cv2.imread(str(source_map), cv2.IMREAD_GRAYSCALE)
        shares = defaultdict(float)
        marked = source > 0
        if marked.any():
            for name, value in (('photographed_pct', 1), ('mirrored_pct', 2), ('rendered_pct', 3)):
                shares[name] = float((source[marked] == value).mean() * 100)
        report['provenance'] = dict(shares)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mesh', type=Path, help='Mesh (.ply) or textured asset (.glb)')
    p.add_argument('--source-map', type=Path, help='<asset>_source.png; defaults beside a .glb')
    p.add_argument('--geometry', type=Path,
                   help='Untextured mesh to read topology from; a textured copy has seam-split vertices '
                        '(defaults to object_watertight.ply or object.ply beside the asset)')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    textured = a.mesh.suffix.lower() in ('.glb', '.gltf', '.obj')
    # UV unwrapping splits vertices along chart seams, so a textured copy reports thousands of boundary
    # edges and hundreds of "pieces" that are not holes at all. Topology is read from the untextured mesh.
    geometry_path = a.geometry
    if geometry_path is None and textured:
        geometry_path = next((c for c in (a.mesh.with_name('object_watertight.ply'),
                                          a.mesh.with_name('object.ply')) if c.is_file()), None)
    report = {'mesh': str(a.mesh), 'geometry_from': str(geometry_path) if geometry_path else str(a.mesh)}
    mesh = o3d.io.read_triangle_mesh(str(geometry_path or a.mesh))
    if not len(mesh.triangles):
        p.error(f'Empty mesh: {geometry_path or a.mesh}')
    report.update(geometry_report(mesh))
    if textured:
        if geometry_path is None:
            report['topology_warning'] = ('read from the textured copy, whose UV seams split vertices: '
                                          'boundary edges and pieces are not holes. Pass --geometry.')
        source = a.source_map or a.mesh.with_name(a.mesh.stem + '_source.png')
        report.update(texture_report(a.mesh, source))
    print(json.dumps(report, indent=2))
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
