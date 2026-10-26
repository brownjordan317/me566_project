#!/usr/bin/env python3
"""Texture a mesh from the calibrated full-resolution photos.

Vertex colours can only be as sharp as the mesh is dense; a texture keeps the
photos' detail. The mesh is UV-unwrapped with xatlas (https://github.com/jpcy/xatlas,
MIT; Python bindings by mworchel), which tolerates the non-manifold vertices marching
cubes leaves (Open3D's UVAtlas port crashes on them), and a subset of frames spread
over the sequence is
projected onto the texture and blended (TriangleMesh.project_images_to_albedo,
Open3D; MIT). With --fill-views, texels the photos leave unpainted are taken from
renders of the fitted Gaussians instead (recon.surface.splat_views), which covers
the grazing and unseen surface the photos miss without losing the photos' detail
where they have it. The result is written as a binary glTF (.glb), which Blender
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


def texel_points(vertices: np.ndarray, faces: np.ndarray, uvs: np.ndarray,
                 size: int) -> tuple[np.ndarray, np.ndarray]:
    """The surface point behind each texel, by rasterising the UV triangles.

    Returned as a (size, size, 3) array of positions and a mask of which texels a triangle covers.
    """
    points = np.zeros((size, size, 3), np.float32)
    covered = np.zeros((size, size), bool)
    pixel = uvs.copy()
    pixel[:, 0] *= size - 1
    pixel[:, 1] = (1 - pixel[:, 1]) * (size - 1)
    for triangle in faces:
        uv = pixel[triangle]
        low = np.floor(uv.min(0)).astype(int)
        high = np.ceil(uv.max(0)).astype(int)
        low = np.maximum(low, 0)
        high = np.minimum(high, size - 1)
        if np.any(high < low):
            continue
        xs = np.arange(low[0], high[0] + 1)
        ys = np.arange(low[1], high[1] + 1)
        grid = np.stack(np.meshgrid(xs, ys, indexing='xy'), -1).reshape(-1, 2).astype(np.float32)
        a0, b0, c0 = uv
        cross = lambda p, q: p[..., 0] * q[..., 1] - p[..., 1] * q[..., 0]   # NumPy 2 dropped 2-D cross
        area = float(cross(b0 - a0, c0 - a0))
        if abs(area) < 1e-9:
            continue
        w0 = cross(b0 - a0, grid - a0) / area
        w1 = cross(grid - a0, c0 - a0) / area
        inside = (w0 >= -.002) & (w1 >= -.002) & (w0 + w1 <= 1.002)
        if not inside.any():
            continue
        picked = grid[inside].astype(int)
        bary = np.stack([1 - w0[inside] - w1[inside], w1[inside], w0[inside]], 1)
        points[picked[:, 1], picked[:, 0]] = bary @ vertices[triangle]
        covered[picked[:, 1], picked[:, 0]] = True
    return points, covered


def mirror_texels(albedo: np.ndarray, blank: np.ndarray, vertices: np.ndarray, faces: np.ndarray,
                  uvs: np.ndarray, blank_level: float) -> tuple[np.ndarray, float]:
    """Fill unpainted texels from their mirror-symmetric partner, where that partner was painted."""
    from scipy.spatial import cKDTree
    size = albedo.shape[0]
    points, covered = texel_points(vertices, faces, uvs, size)
    target = blank & covered
    painted = covered & ~blank
    if not target.any() or not painted.any():
        return albedo, 0.
    centroid = vertices.mean(0)
    _, _, axes = np.linalg.svd(vertices - centroid, full_matrices=False)
    source = points[painted]
    tree = cKDTree(source)
    colours = albedo[painted]
    best = None
    for axis in axes:                                   # the axis whose reflection lands closest wins
        reflected = points[target] - 2 * np.outer((points[target] - centroid) @ axis, axis)
        distance, index = tree.query(reflected, workers=-1)
        if best is None or np.median(distance) < best[0]:
            best = (float(np.median(distance)), distance, index)
    _, distance, index = best
    extent = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
    close = distance < .01 * extent                     # only where the mirror really lands on surface
    filled = albedo.copy()
    rows, cols = np.nonzero(target)
    filled[rows[close], cols[close]] = colours[index[close]]
    return filled, close.sum() / max(1, covered.sum())


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
    p.add_argument('--provenance-map', action=argparse.BooleanOptionalAction, default=True,
                   help='Write <output>_source.png: per texel, 0 unpainted, 1 photographed, 2 mirrored, '
                        '3 rendered, so inferred colour is never mistaken for observed')
    p.add_argument('--mirror-fill', action=argparse.BooleanOptionalAction, default=False,
                   help="Fill unpainted texels from their mirror-symmetric partner's photographed colour "
                        'before falling back to renders; most scanned objects are largely symmetric')
    p.add_argument('--fill-views', type=Path,
                   help='recon.surface.splat_views output; its renders fill texels the photos leave blank')
    p.add_argument('--fill-frames', type=int, default=48, help='Rendered views projected for the filling')
    p.add_argument('--projection-batch', type=int, default=0,
                   help='Images projected at once; 0 scales the batch to the atlas so memory stays flat')
    p.add_argument('--blank-level', type=float, default=8.,
                   help='Mean channel value below which a texel counts as unpainted')
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
    def project(source_images: Path, source_model: Path, source_trajectory: Path, count: int,
                source_sequence: Path) -> tuple[np.ndarray, list[int], o3d.t.geometry.TriangleMesh]:
        """Bake one set of calibrated images onto a fresh copy of this mesh's UV layout.

        Each source gets its own copy: projecting onto a mesh that already carries an albedo blends into
        it, which contaminated the photos' texels with the renders and cost sharpness (Laplacian std 22.0
        against 24.5 when the two are baked separately).
        """
        target = o3d.t.geometry.TriangleMesh.from_legacy(seamed)
        target.triangle.texture_uvs = o3d.core.Tensor(uvs[faces].astype(np.float32))
        source_poses = load_poses(source_trajectory)
        source_intrinsics = frame_intrinsics(source_model)
        width = cv2.imread(str(next(iter(sorted((source_sequence / 'frames').glob('*.png')))))).shape[1]
        available = {int(path.stem): path for path in source_images.iterdir()
                     if path.suffix.lower() in ('.png', '.jpg', '.jpeg') and path.stem.isdigit()}
        order = [frame for frame in sorted(source_poses) if frame in available]
        if not order:
            raise ValueError(f'No calibrated images in {source_images}')
        picked = [order[i] for i in np.unique(np.linspace(0, len(order) - 1,
                                                          min(count, len(order))).round().astype(int))]
        images, Ks, extrinsics = [], [], []
        for frame in picked:
            image = cv2.cvtColor(cv2.imread(str(available[frame])), cv2.COLOR_BGR2RGB)
            scale = image.shape[1] / width
            K = source_intrinsics[frame].copy()
            K[:2] *= scale
            K[0, 2] += .5 * (scale - 1)                                    # pixel-centre convention
            K[1, 2] += .5 * (scale - 1)
            rotation, centre = source_poses[frame]                         # camera-to-world rotation, centre
            world_to_camera = np.eye(4)
            world_to_camera[:3, :3], world_to_camera[:3, 3] = rotation.T, -rotation.T @ centre
            images.append(o3d.t.geometry.Image(np.ascontiguousarray(image)))
            Ks.append(o3d.core.Tensor(K, o3d.core.float32))
            extrinsics.append(o3d.core.Tensor(world_to_camera, o3d.core.float32))
        # Open3D holds every image and the whole atlas in memory at once, so a large atlas with many
        # photos simply dies (8192 px with 43 images exhausts 23 GB). Projecting in batches and averaging
        # the batches where they overlap keeps memory flat in the image count, so the atlas size is a
        # choice rather than something the hardware decides. One batch reproduces the single call exactly.
        batch = a.projection_batch or max(1, int(len(images) * (4096 / a.texture_size) ** 2))
        groups = [range(start, min(start + batch, len(images))) for start in range(0, len(images), batch)]
        total = np.zeros((a.texture_size, a.texture_size, 3), np.float64)
        count = np.zeros((a.texture_size, a.texture_size), np.float64)
        label = f'Projecting {len(picked)} images onto a {a.texture_size}px texture'
        with spinner(label if len(groups) == 1 else f'{label} in {len(groups)} batches'):
            for index, group in enumerate(groups):
                # The first batch runs on the mesh itself, so its material carries a real albedo map for
                # the glTF writer; later batches use scratch copies and are merged into that map.
                if index == 0:
                    piece = target
                else:
                    piece = o3d.t.geometry.TriangleMesh.from_legacy(seamed)
                    piece.triangle.texture_uvs = o3d.core.Tensor(uvs[faces].astype(np.float32))
                piece.project_images_to_albedo([images[i] for i in group], [Ks[i] for i in group],
                                               [extrinsics[i] for i in group], a.texture_size)
                painted = np.asarray(piece.material.texture_maps['albedo'].to_legacy())
                # Unpainted texels come back exactly zero, so test against that rather than the blank
                # level: a dim but genuine texel is real coverage and must not be dropped from the merge.
                lit = painted[..., :3].any(-1)
                total[lit] += painted[lit, :3]
                count[lit] += 1
        albedo = np.zeros_like(total, np.uint8)
        painted = count > 0
        albedo[painted] = np.rint(total[painted] / count[painted, None]).astype(np.uint8)
        target.material.texture_maps['albedo'] = o3d.t.geometry.Image(np.ascontiguousarray(albedo))
        return albedo.copy(), picked, target

    albedo, chosen, mesh = project(a.images, a.model, a.trajectory, a.frames, a.sequence)
    filled = mirrored = None
    # Provenance, so nothing synthetic can be mistaken for evidence: 1 photographed, 2 mirrored from a
    # photographed counterpart, 3 rendered from the fitted Gaussians (plausible, not observed), 0 unpainted.
    PHOTO, MIRROR, RENDER = 1, 2, 3
    source = np.where(albedo[..., :3].mean(-1) < a.blank_level, 0, PHOTO).astype(np.uint8)
    if a.mirror_fill:
        # Photographed colour, reflected, beats anything synthetic: these objects are largely symmetric,
        # and a third to two thirds of the surface no view saw has a mirror counterpart that was seen
        # (recon/evaluation/diagnostics/headroom.py). So unpainted texels take their partner's colour
        # before the renders are consulted at all.
        blank = albedo[..., :3].mean(-1) < a.blank_level
        with spinner('Mirroring photographed colour into unpainted texels'):
            before = albedo[..., :3].mean(-1) < a.blank_level
            albedo, share = mirror_texels(albedo, blank, vertices, faces, uvs, a.blank_level)
            source[before & (albedo[..., :3].mean(-1) >= a.blank_level)] = MIRROR
        mirrored = {'share_of_texture_pct': float(share * 100)}
    if a.fill_views is not None:
        # The photos leave surface blank wherever they only see it at a grazing angle or not at all: on
        # jeep 8% of on-surface texels against the 1.1% no view observed. Renders of the fitted Gaussians
        # from cameras facing the mesh (recon.surface.splat_views) cover it, at slightly lower sharpness,
        # so they are used only where the photos have nothing.
        fill, fill_frames, _ = project(a.fill_views / 'frames', a.fill_views / 'sparse_txt',
                                       a.fill_views / 'trajectory.csv', a.fill_frames, a.fill_views)
        blank = albedo[..., :3].mean(-1) < a.blank_level
        # One texel of dilation puts the seam inside the filled region rather than on its boundary.
        blank = cv2.dilate(blank.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        albedo[blank] = fill[blank]
        source[blank & (albedo[..., :3].mean(-1) >= a.blank_level)] = RENDER
        filled = {'views': fill_frames, 'share_of_texture_pct': float(blank.mean() * 100)}
        mesh.material.texture_maps['albedo'] = o3d.t.geometry.Image(np.ascontiguousarray(albedo))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.t.io.write_triangle_mesh(str(a.output), mesh):
        raise OSError(a.output)
    painted = source > 0
    provenance = {name: float((source[painted] == value).mean() * 100) if painted.any() else 0.
                  for name, value in (('photographed_pct', PHOTO), ('mirrored_pct', MIRROR),
                                      ('rendered_pct', RENDER))}
    if a.provenance_map:
        # Written beside the asset rather than into it: a viewer shows the albedo, and this says which of
        # it is evidence. 0 unpainted, 1 photographed, 2 mirrored, 3 rendered.
        cv2.imwrite(str(a.output.with_name(a.output.stem + '_source.png')), source)
    report = {'faces': len(legacy.triangles), 'frames': chosen, 'texture_size': a.texture_size,
              'texture_provenance': provenance,
              'disclaimer': 'Texels marked mirrored or rendered in the *_source.png map are inferred, not '
                            'observed: no view photographed that surface. Judge accuracy on the '
                            'photographed share only.',
              'filled_from_renders': filled, 'filled_from_mirror': mirrored}
    a.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    done(f'Texture: {len(chosen)} photos onto {len(legacy.triangles):,} faces -> {a.output.name}')


if __name__ == '__main__':
    main()
