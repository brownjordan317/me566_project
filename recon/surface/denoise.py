#!/usr/bin/env python3
"""Feature-preserving mesh denoising: bilateral normal filtering, then vertices moved to match the normals.

Fused depth carries a millimetre of view-to-view disagreement, so flat panels come out covered in small bumps and
pits (docs/PIPELINE.md), and Taubin smoothing removes them only by also rounding every crease. Bilateral
filtering of face normals (Zheng, Fu, Au and Tai, "Bilateral Normal Filtering for Mesh Denoising", IEEE TVCG
2011; reimplemented here from the paper) averages each face's normal with its neighbours', weighted both by
distance and by how alike the normals already are, so faces across a crease barely mix: flat regions flatten
while edges stay sharp. The vertices are then moved to agree with the filtered normals, each pulled onto the
planes of its incident faces (Sun, Rosin, Martin and Langbein, "Fast and Effective Feature-Preserving Mesh
Denoising", IEEE TVCG 2007).

Neighbours are faces sharing a vertex; the spatial scale is the mean edge length (--spatial in those units) and
the range scale --range in normal difference. Vertex colours are carried through unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy import sparse

from recon.common.ply import write_ply
from recon.common.progress import done, spinner


def face_geometry(vertices: np.ndarray, faces: np.ndarray):
    corners = vertices[faces]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    area = np.linalg.norm(cross, axis=1) / 2
    normal = cross / np.maximum(2 * area, 1e-20)[:, None]
    return corners.mean(1), normal, area


def face_neighbours(faces: np.ndarray, vertex_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Pairs (i, j) of faces sharing at least one vertex, i != j."""
    rows = np.repeat(np.arange(len(faces)), 3)
    incidence = sparse.csr_matrix((np.ones(len(rows)), (rows, faces.reshape(-1))), shape=(len(faces), vertex_count))
    shared = (incidence @ incidence.T).tocoo()
    keep = shared.row != shared.col
    return shared.row[keep], shared.col[keep]


def filter_normals(centres, normals, areas, i, j, sigma_s, sigma_r, iterations):
    for _ in range(iterations):
        distance = np.linalg.norm(centres[i] - centres[j], axis=1)
        difference = np.linalg.norm(normals[i] - normals[j], axis=1)
        weight = areas[j] * np.exp(-(distance / sigma_s) ** 2 / 2) * np.exp(-(difference / sigma_r) ** 2 / 2)
        blended = np.zeros_like(normals)
        np.add.at(blended, i, weight[:, None] * normals[j])
        blended += areas[:, None] * normals                                  # the face itself, weight 1 in range
        normals = blended / np.maximum(np.linalg.norm(blended, axis=1, keepdims=True), 1e-20)
    return normals


def update_vertices(vertices, faces, target, iterations):
    """Sun et al. 2007: move each vertex onto the average of its incident faces' planes.

    A step can carry a vertex past its neighbours and turn a thin triangle inside out - 1.4% of faces on jeep,
    seen as dark specks and as doubled roughness - so any move that flips one of the vertex's triangles against
    where that triangle faced before the step is undone for that iteration."""
    vertices = vertices.copy()
    counts = np.bincount(faces.reshape(-1), minlength=len(vertices)).astype(float)[:, None]
    for _ in range(iterations):
        centres = vertices[faces].mean(1)
        before = face_geometry(vertices, faces)[1]
        moved = np.zeros_like(vertices)
        for corner in range(3):
            index = faces[:, corner]
            np.add.at(moved, index, target * ((centres - vertices[index]) * target).sum(1, keepdims=True))
        candidate = vertices + moved / np.maximum(counts, 1)
        # Freezing a flipped face's vertices can leave a neighbour half-moved and flipped in turn, so repeat until
        # no face is inside out.
        for _ in range(20):
            flipped = (face_geometry(candidate, faces)[1] * before).sum(1) < 0
            if not flipped.any():
                break
            frozen = np.zeros(len(vertices), bool)
            frozen[faces[flipped].reshape(-1)] = True
            candidate[frozen] = vertices[frozen]
        vertices = candidate
    return vertices


def denoise(vertices, faces, spatial=1., range_=.2, normal_iterations=4, vertex_iterations=10):
    """Returns the moved vertices and the faces, consistently oriented first: these meshes have about 1% of
    faces wound against their neighbours, and a flipped normal among the neighbours would steer the filter."""
    import igl
    faces = igl.bfs_orient(np.ascontiguousarray(faces, dtype=np.int64))[0]
    centres, normals, areas = face_geometry(vertices, faces)
    edge = np.linalg.norm(vertices[faces] - vertices[np.roll(faces, 1, axis=1)], axis=2).mean()
    i, j = face_neighbours(faces, len(vertices))
    target = filter_normals(centres, normals, areas, i, j, spatial * edge, range_, normal_iterations)
    return update_vertices(vertices, faces, target, vertex_iterations), faces


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mesh', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--spatial', type=float, default=1., help='Spatial scale, in mean edge lengths')
    # Chosen by eye on jeep's hood and convertible's cockpit: 0.35 with 10 / 20 iterations merged the hood's studs
    # into ridges, because a reconstructed stud is a low bump the size of the noise; these keep them as bumps.
    p.add_argument('--range', type=float, default=.2, help='Normal-difference scale; smaller keeps more creases')
    p.add_argument('--normal-iterations', type=int, default=4)
    p.add_argument('--vertex-iterations', type=int, default=10)
    a = p.parse_args()
    mesh = o3d.io.read_triangle_mesh(str(a.mesh))
    vertices, faces = np.asarray(mesh.vertices, np.float64), np.asarray(mesh.triangles)
    colours = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else np.full((len(vertices), 3), .7)
    with spinner(f'Denoising {len(faces):,} faces'):
        out, faces = denoise(vertices, faces, a.spatial, a.range, a.normal_iterations, a.vertex_iterations)
    result = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(out), o3d.utility.Vector3iVector(faces))
    result.compute_vertex_normals()
    write_ply(a.output, out, colours, np.asarray(result.vertex_normals), faces)
    moved = np.linalg.norm(out - vertices, axis=1)
    a.output.with_suffix('.json').write_text(json.dumps({
        'method': 'bilateral normal filtering (Zheng et al. 2011) + vertex update (Sun et al. 2007)',
        'spatial': a.spatial, 'range': a.range, 'normal_iterations': a.normal_iterations,
        'vertex_iterations': a.vertex_iterations, 'faces': int(len(faces)),
        'moved_median': float(np.median(moved)), 'moved_p99': float(np.percentile(moved, 99))}, indent=2) + '\n')
    done(f'Denoised {len(faces):,} faces, vertices moved median {np.median(moved):.5f}')


if __name__ == '__main__':
    main()
