#!/usr/bin/env python3
"""Cut the object out of a reconstructed scene, using the fitted Gaussians and the 2D object masks.

Scene-first reconstruction fuses everything the cameras saw, with no mask anywhere, so the object has to
be extracted afterwards. The masks become supervision for a 3D labelling instead of an authority over
which pixels each stage may use: a mask that clips the object in one frame is outvoted by the others
rather than deleting that surface, which is what costs object mode 12.6 official points for a 2 px
erosion (docs/PIPELINE.md).

Each Gaussian of the fit is labelled in closed form (`recon.depth.gaussian_refine.solve_labels`, after
FlashSplat): the rendered mask is linear in the labels, so a Gaussian is foreground when its alpha-blending
weight over mask pixels exceeds its weight over background pixels times --bias. Rasterisation decides which
Gaussian each pixel sees, so this is occlusion-aware where a visual hull is not - a hull keeps whatever lies
inside every silhouette cone, including the table under the object, and lost 11.8 points of recall to that
on jeep. Each mesh vertex then takes the labels of its nearest Gaussians.

Runs in the planar-Gaussian environment (gsplat):

    ~/.venvs/me566-pgsr/bin/python -m recon.surface.segment SEQUENCE --model sparse_txt \\
        --gaussians gaussians.ply --mesh scene_mesh.ply --masks MASKS --output object.ply
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def vertex_scores(vertices: np.ndarray, means: np.ndarray, foreground: np.ndarray,
                  neighbours: int = 8) -> np.ndarray:
    """Foreground score per vertex: an inverse-distance-weighted mean over its nearest Gaussians.

    Small parts were the worry here, and averaging is not what loses them: on big_ben, cuts from 1, 2,
    4 and 8 neighbours all scored within 0.4 F@1 of each other (docs/PIPELINE.md), so the default keeps
    the average for its tolerance of a single mislabelled Gaussian.
    """
    from scipy.spatial import cKDTree
    distance, index = cKDTree(means).query(vertices, k=neighbours, workers=-1)
    if neighbours == 1:                    # k=1 returns flat arrays, and there is nothing to average
        return foreground[index].astype(np.float64)
    weight = 1. / np.maximum(distance, 1e-9)
    return (foreground[index] * weight).sum(1) / weight.sum(1)


def cut(mesh, keep: np.ndarray, largest: bool = True):
    """Drop the vertices not kept, and optionally every piece but the largest."""
    mesh.remove_vertices_by_mask(~keep)
    mesh.remove_unreferenced_vertices()
    if largest and len(mesh.triangles):
        labels, counts, _ = mesh.cluster_connected_triangles()
        labels, counts = np.asarray(labels), np.asarray(counts)
        mesh.remove_triangles_by_mask(labels != int(np.argmax(counts)))
        mesh.remove_unreferenced_vertices()
    return mesh


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('sequence', type=Path, help='Prepared sequence (frames/) the Gaussians were fitted to')
    p.add_argument('--model', type=Path, required=True, help='COLMAP text model (sparse_txt)')
    p.add_argument('--gaussians', type=Path, required=True, help='gaussian_refine --export-gaussians PLY')
    p.add_argument('--mesh', type=Path, required=True, help='The scene mesh to cut')
    p.add_argument('--masks', type=Path, required=True, help='Object masks, <frame>.png')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bias', type=float, default=1.,
                   help='Background weight in the label solve; above 1 trades recall for precision')
    p.add_argument('--threshold', type=float, default=.5, help='Vertex score kept as object')
    p.add_argument('--neighbours', type=int, default=8, help='Gaussians averaged per vertex')
    p.add_argument('--no-largest', dest='largest', action='store_false',
                   help='Keep every piece, not only the largest')
    a = p.parse_args()
    import open3d as o3d
    import torch
    from recon.common.progress import done, spinner
    from recon.depth.gaussian_refine import Views, load_gaussians, solve_labels

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with spinner('Loading the fit and the masks'):
        views = Views(a.sequence, a.model, device, masks=a.masks)
        gaussians = load_gaussians(a.gaussians, device)
    with spinner(f'Labelling {len(gaussians.means):,} Gaussians over {len(views.numbers)} views'):
        foreground = solve_labels(gaussians, views, a.bias).cpu().numpy()
        means = gaussians.means.detach().cpu().numpy()
    mesh = o3d.io.read_triangle_mesh(str(a.mesh))
    source_faces = len(mesh.triangles)
    score = vertex_scores(np.asarray(mesh.vertices), means, foreground, a.neighbours)
    mesh = cut(mesh, score >= a.threshold, a.largest)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(a.output), mesh)
    report = {'gaussians': int(len(foreground)), 'foreground_gaussians_pct': float(foreground.mean() * 100),
              'source_faces': source_faces, 'faces': len(mesh.triangles), 'bias': a.bias,
              'threshold': a.threshold, 'neighbours': a.neighbours, 'largest': a.largest}
    a.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    done(f'Segment: {len(mesh.triangles):,} of {source_faces:,} faces kept '
         f'({report["foreground_gaussians_pct"]:.1f}% of Gaussians foreground) -> {a.output.name}')


if __name__ == '__main__':
    main()
