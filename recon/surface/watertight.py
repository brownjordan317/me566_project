#!/usr/bin/env python3
"""Close the finished object mesh, leaving the measured surface where it is.

The object mesh comes out open: on jeep 2,240 boundary edges in 15 loops, the largest winding right
across the object rather than capping the unseen base. Nothing that rebuilds the surface from points or
from a field is a good answer. Hole filling closes 20% of the boundary, screened Poisson comes out open
as well (the cloud has no underside), and re-fusing depth into a new field loses what the pipeline's
accuracy actually comes from, which is the mask-vote extraction *after* the TSDF, not the TSDF itself
(the raw weighted TSDF mesh scores F@1 18 against the extracted mesh's 76).

So the input here is the finished mesh, and it is never re-meshed from evidence. Its generalised winding
number (Jacobson et al., SIGGRAPH 2013; the fast multipole form of Barill et al., SIGGRAPH 2018, through
libigl) is about 1 strictly inside a surface and 0 outside, and degrades gracefully where the surface is
open: an opening subtracts only its own solid angle, so a threshold still closes the hole. The visual
hull from the masks (shape from silhouette, Laurentini, TPAMI 1994) supplies the one thing the winding
number cannot know - that nothing may lie outside the silhouettes - which stops the closure bulging out
through a large opening such as the base an object stands on.

Defects are kept out of the field rather than repaired in the mesh: repairing afterwards reopens the
surface (`remove_non_manifold_edges` took jeep's hull from 9 boundary edges to 20), while one-voxel necks
and specks in the field never become bad edges at all.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from scipy import ndimage
from skimage.measure import marching_cubes

from recon.common.camera import frame_intrinsics
from recon.common.progress import spinner
from recon.common.trajectory import load_poses


def carve_hull(a, low: np.ndarray, shape: np.ndarray, voxel: float, device: str) -> np.ndarray:
    """Visual hull on a given grid: a voxel survives unless some view sees background through it."""
    intrinsics, poses = frame_intrinsics(a.model), load_poses(a.trajectory)
    points = grid_points(low, shape, voxel, device)
    inside = torch.ones(len(points), dtype=torch.bool, device=device)
    for frame, (rotation, centre) in sorted(poses.items()):
        mask_file = a.sequence / 'masks' / f'{frame:08d}.png'
        if frame not in intrinsics or not mask_file.is_file():
            continue
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        K = torch.tensor(intrinsics[frame], dtype=torch.float32, device=device)
        R = torch.tensor(rotation, dtype=torch.float32, device=device)
        t = torch.tensor(centre, dtype=torch.float32, device=device)
        camera = (points - t) @ R
        z = camera[:, 2]
        height, width = mask.shape
        u = (K[0, 0] * camera[:, 0] / z.clamp(min=1e-6) + K[0, 2]).round().long()
        v = (K[1, 1] * camera[:, 1] / z.clamp(min=1e-6) + K[1, 2]).round().long()
        in_view = (z > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        flat = torch.where(in_view, v * width + u, torch.zeros_like(u))
        silhouette = torch.tensor((mask > 127).reshape(-1), device=device)
        inside &= ~(in_view & ~silhouette[flat])
    return inside.reshape(tuple(shape)).cpu().numpy()


def carve_free_space(a, low: np.ndarray, shape: np.ndarray, voxel: float, device: str,
                    margin: float) -> np.ndarray:
    """Voxels some view saw through: nearer than its observed surface by more than `margin`.

    Silhouettes alone cannot carve a concavity, because a cavity that is only visible by looking into it
    still projects inside the object's outline from every direction; the visual hull therefore fills it
    (an open-top car's cockpit, convertible: official F1 55.4 -> 16.5 with the hull alone). Depth knows
    better: the space between a camera and the surface it measured is empty whatever the silhouette says.
    """
    intrinsics, poses = frame_intrinsics(a.model), load_poses(a.trajectory)
    points = grid_points(low, shape, voxel, device)
    free = torch.zeros(len(points), dtype=torch.bool, device=device)
    for frame, (rotation, centre) in sorted(poses.items()):
        depth_file = a.depth_dir / f'{frame:08d}_grounded.npy'
        if frame not in intrinsics or not depth_file.is_file():
            continue
        depth = np.load(depth_file).astype('float32')
        K = torch.tensor(intrinsics[frame], dtype=torch.float32, device=device)
        R = torch.tensor(rotation, dtype=torch.float32, device=device)
        t = torch.tensor(centre, dtype=torch.float32, device=device)
        camera = (points - t) @ R
        z = camera[:, 2]
        height, width = depth.shape
        u = (K[0, 0] * camera[:, 0] / z.clamp(min=1e-6) + K[0, 2]).round().long()
        v = (K[1, 1] * camera[:, 1] / z.clamp(min=1e-6) + K[1, 2]).round().long()
        in_view = (z > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        flat = torch.where(in_view, v * width + u, torch.zeros_like(u))
        observed = torch.tensor(depth.reshape(-1), device=device)[flat]
        free |= in_view & (observed > 0) & (z < observed - margin)
    return free.reshape(tuple(shape)).cpu().numpy()


def tidy_field(occupancy: np.ndarray) -> np.ndarray:
    """Largest blob with thin necks closed, so the level set cannot pick up specks or pinch points."""
    labels, count = ndimage.label(occupancy)
    if count > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        occupancy = labels == sizes.argmax()
    return ndimage.binary_closing(occupancy, iterations=1)


def finish(mesh: o3d.geometry.TriangleMesh, clean: bool) -> o3d.geometry.TriangleMesh:
    if clean:
        mesh.remove_duplicated_vertices(); mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles(); mesh.remove_unreferenced_vertices()
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    if len(counts) > 1:
        mesh.remove_triangles_by_mask(counts[labels] < counts.max())
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def close_by_winding(a) -> None:
    """Close the finished object mesh with its generalised winding number, bounded by the visual hull.

    The winding number of a triangle soup (Jacobson et al., SIGGRAPH 2013; the fast multipole form of
    Barill et al., SIGGRAPH 2018, via libigl) is about 1 strictly inside a surface and 0 outside, and it
    degrades gracefully rather than failing where the surface is open: an opening removes only its own
    solid angle from the integral, so thresholding the field still closes the hole. That is what makes it
    the right tool here, where the input is an accurate but open mesh and must stay exactly where it is.

    The visual hull then supplies what the winding number cannot know: nothing may lie outside the
    silhouettes, which stops the closure bulging out through a large opening such as an object's base.
    """
    import igl
    mesh = o3d.io.read_triangle_mesh(str(a.mesh))
    if not len(mesh.triangles):
        raise ValueError(f'Empty mesh: {a.mesh}')
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    low, high = vertices.min(0), vertices.max(0)
    pad = a.pad * (high - low)
    low, high = low - pad, high + pad
    voxel = float((high - low).max()) / a.resolution
    shape = np.maximum(np.ceil((high - low) / voxel).astype(int) + 1, 8)
    band = a.band * voxel
    device = a.device if torch.cuda.is_available() else 'cpu'
    with spinner('Carving the visual hull'):
        occupancy = carve_hull(a, low, shape, voxel, device)
    if a.depth_dir is not None:
        with spinner('Carving free space from depth'):
            occupancy &= ~carve_free_space(a, low, shape, voxel, device, a.free_margin * voxel)
    occupancy = tidy_field(occupancy)
    query = grid_points(low, shape, voxel, 'cpu').numpy().astype(np.float64)
    winding = np.empty(len(query))
    with spinner(f'Winding number at {len(query):,} points'):
        for start in range(0, len(query), a.chunk):                 # the whole grid at once exhausts memory
            winding[start:start + a.chunk] = igl.fast_winding_number(
                vertices, faces, query[start:start + a.chunk])
    # Orientation decides the winding number's sign, and these meshes are wound inconsistently: a single
    # probe at the centroid called big_ben positive while its interior is -0.690. Decide from the interior
    # itself - inside the hull, eroded away from its boundary - and flip the faces, not just the field,
    # because the signed distance takes its sign from the faces it is given.
    interior_mask = ndimage.binary_erosion(occupancy, iterations=max(1, int(a.band))).reshape(-1)
    flipped = interior_mask.sum() > 100 and float(np.median(winding[interior_mask])) < 0
    if flipped:
        faces = faces[:, ::-1].copy()
        winding = -winding
    distance = np.empty(len(query))
    if a.signed_distance:
        with spinner(f'Signed distance at {len(query):,} points'):
            for start in range(0, len(query), a.chunk):
                distance[start:start + a.chunk] = igl.signed_distance(
                    query[start:start + a.chunk], vertices, faces,
                    igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER)[0]
    # The winding number inside an *open* surface is 1 minus the openings' solid angle, which varies by
    # scene: 0.69 for audi and jeep, 0.51 satellite, 0.42 convertible, 0.14 big_ben. A fixed 0.5 therefore
    # finds nothing inside the more open meshes and the closure collapses onto the bare hull (convertible:
    # official F1 55.4 -> 16.5). So the threshold is calibrated per scene from the winding number deep
    # inside the hull, far from any surface, where the field is whatever "inside" means for this mesh.
    # The winding number's sign follows face orientation, and these meshes are wound inconsistently: a
    # single probe at the centroid called big_ben positive while its interior is -0.690. So orientation is
    # decided from the interior itself - voxels inside the hull and far from any surface. Flipping the
    # faces negates the winding exactly, so the field is simply negated rather than recomputed.
    deep = interior_mask & (np.abs(distance) > band if a.signed_distance else np.ones(len(query), bool))
    threshold = a.threshold
    interior = float(np.median(winding[deep])) if deep.sum() > 100 else float('nan')
    if a.adaptive_threshold and deep.sum() > 100:
        threshold = max(a.min_threshold, a.threshold_fraction * interior)
    # The whole method rests on the winding number separating inside from outside, and that separation
    # weakens as the input mesh gets more open: measured interior values are 0.69 (audi, jeep), 0.51
    # (satellite), 0.42 (convertible), 0.14 (big_ben). Below about 0.5 the field smears, the sign goes
    # wrong over whole regions and the output is a blob, not a surface - convertible's official F1 falls
    # 55.4 -> 16.5 and its chamfer rises 6.7 -> 24.6 mm, the same from four different variants of the
    # closure, so it is the winding number itself and not the clamp, threshold or carving. Refusing is
    # the honest behaviour: an open mesh beats a closed blob.
    if a.require_interior > 0 and interior == interior and interior < a.require_interior:
        raise SystemExit(
            f'Refusing to close {a.mesh}: the mesh is too open for its winding number to separate inside '
            f'from outside (interior winding {interior:.3f} < {a.require_interior}). Closing it would '
            f'produce a solid blob rather than a surface. Pass --require-interior 0 to override.')
    # A winding-number indicator only localises the surface to half a voxel, and smoothing blurs it
    # further; a true signed distance puts its zero set on the triangles themselves, and trilinear
    # interpolation of it is sub-voxel accurate, so the measured surface survives the re-meshing.
    if a.signed_distance:
        # Faces were already flipped to wind outwards, so this is negative inside without further work.
        field = np.clip(distance, -band, band).reshape(tuple(shape))
    else:
        field = (threshold - winding).reshape(tuple(shape))
    outside_value = band if a.signed_distance else threshold
    if a.hull_clamp:
        # Beyond the silhouettes nothing may exist, which stops the closure bulging through an opening.
        field = np.where(occupancy, field, np.maximum(field, outside_value))
    # Inside the hull but unreachable by the measured surface (the unseen base): the hull decides.
    if a.signed_distance:
        unresolved = (np.abs(distance.reshape(tuple(shape))) >= band) & (winding.reshape(tuple(shape)) < threshold)
        field = np.where(unresolved & occupancy, -band, field)
    if a.blur > 0:
        field = ndimage.gaussian_filter(field, a.blur)
    field = np.pad(field, 2, constant_values=outside_value)
    verts, faces_out, _, _ = marching_cubes(field, level=0., spacing=(voxel,) * 3)
    closed = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts + low - 2 * voxel),
                                       o3d.utility.Vector3iVector(faces_out))
    closed = finish(closed, a.clean)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(a.output), closed):
        raise OSError(a.output)
    report = {'source': str(a.mesh), 'voxel': voxel, 'grid': shape.tolist(), 'threshold': a.threshold,
              'hull_voxels_pct': float(occupancy.mean() * 100), 'depth_carved': a.depth_dir is not None,
              'inside_voxels_pct': float((winding > threshold).mean() * 100), 'threshold_used': threshold,
              'interior_winding': interior, 'orientation_flipped': bool(flipped),
              'winding_at_centroid': interior,
              'input_faces': len(faces), 'faces': len(closed.triangles),
              'boundary_edges': len(np.asarray(closed.get_non_manifold_edges(allow_boundary_edges=False))),
              'edge_manifold': bool(closed.is_edge_manifold()),
              'vertex_manifold': bool(closed.is_vertex_manifold()),
              'orientable': bool(closed.is_orientable())}
    print(json.dumps(report, indent=2))
    a.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')


def grid_points(low: np.ndarray, shape: np.ndarray, voxel: float, device: str) -> torch.Tensor:
    axes = [torch.arange(int(n), device=device, dtype=torch.float32) * voxel + low[i]
            for i, n in enumerate(shape)]
    mesh = torch.meshgrid(*axes, indexing='ij')
    return torch.stack(mesh, -1).reshape(-1, 3)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('sequence', type=Path, help='Prepared sequence (masks/)')
    p.add_argument('--mesh', type=Path, required=True, help='The finished open object mesh to close')
    p.add_argument('--model', type=Path, required=True, help='COLMAP text model (sparse_txt)')
    p.add_argument('--trajectory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resolution', type=int, default=320, help='Voxels across the longest side')
    p.add_argument('--pad', type=float, default=.08, help='Fraction of the extent added as free space')
    p.add_argument('--depth-dir', type=Path,
                   help='<frame>_grounded.npy; carves space seen through, which silhouettes cannot do '
                        '(concavities such as an open-top car\'s cockpit)')
    p.add_argument('--free-margin', type=float, default=1.5,
                   help='Voxels of slack before depth counts a voxel as free, so the surface is not eaten')
    p.add_argument('--signed-distance', action=argparse.BooleanOptionalAction, default=True,
                   help='Use the signed distance to the mesh (zero set on the triangles) rather than the '
                        'winding-number indicator, whose level set sits up to half a voxel off')
    p.add_argument('--band', type=float, default=3.,
                   help='Signed distance clamped to this many voxels, so far-field values cannot bias it')
    p.add_argument('--require-interior', type=float, default=.5,
                   help='Refuse to close when the measured interior winding is below this, where the '
                        'method degrades into a blob; 0 disables the check')
    p.add_argument('--adaptive-threshold', action=argparse.BooleanOptionalAction, default=True,
                   help='Calibrate the inside threshold from the winding number deep inside the hull, '
                        'instead of assuming the 0.5 that only holds for a nearly closed surface')
    p.add_argument('--threshold-fraction', type=float, default=.5,
                   help='Fraction of the measured interior winding counted as inside')
    p.add_argument('--min-threshold', type=float, default=.05,
                   help='Floor on the calibrated threshold, so noise cannot make everything inside')
    p.add_argument('--threshold', type=float, default=.5,
                   help='Winding number counted as inside; below 0.5 closes larger openings more boldly')
    p.add_argument('--blur', type=float, default=.6, help='Smoothing of the indicator, in voxels')
    p.add_argument('--chunk', type=int, default=2_000_000, help='Winding-number query points per batch')
    p.add_argument('--hull-clamp', action=argparse.BooleanOptionalAction, default=True,
                   help='Keep the closure inside the silhouettes, so it cannot bulge through an opening')
    p.add_argument('--clean', action=argparse.BooleanOptionalAction, default=True,
                   help='Merge duplicate and degenerate triangles after marching cubes')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    return close_by_winding(a)


if __name__ == '__main__':
    main()
