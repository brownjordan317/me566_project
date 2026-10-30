#!/usr/bin/env python3
"""Extract an object from a full-scene mesh by depth-visible 2D mask votes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from recon.common.camera import frame_intrinsics
from recon.common.progress import bar, done
from recon.surface.components import triangle_components
from recon.common.trajectory import load_poses


def mask_edge_band(mask: np.ndarray, band: float) -> np.ndarray:
    """Pixels within `band` pixels of the mask's edge, on either side."""
    inside = mask > 127
    to_outside = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 5)
    to_inside = cv2.distanceTransform((~inside).astype(np.uint8), cv2.DIST_L2, 5)
    return ((np.where(inside, to_outside, to_inside) <= band) * 255).astype(np.uint8)


def frame_votes(points: np.ndarray, mask: np.ndarray, depth: np.ndarray,
                rotation: np.ndarray, center: np.ndarray, K: np.ndarray,
                relative_tolerance: float, absolute_tolerance: float,
                chunk_size: int, edge_band: float = 0.) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-point visibility, object-mask hits, and whether the point lands clear of the mask's edge
    (more than `edge_band` pixels from it; always, when the band is 0) for one frame."""
    height, width = depth.shape
    band = mask_edge_band(mask, edge_band) if edge_band > 0 else None
    visible_all = np.zeros(len(points), dtype=bool)
    positive_all = np.zeros(len(points), dtype=bool)
    clear_all = np.ones(len(points), dtype=bool)
    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        camera = (points[start:end] - center) @ rotation
        z = camera[:, 2]
        u = (K[0, 0] * camera[:, 0] / np.maximum(z, 1e-6) + K[0, 2]).astype('float32')
        v = (K[1, 1] * camera[:, 1] / np.maximum(z, 1e-6) + K[1, 2]).astype('float32')
        inside = ((z > .1) & (u >= 1) & (u < width - 2)
                  & (v >= 1) & (v < height - 2))
        sampled_depth = cv2.remap(depth, u.reshape(-1, 1), v.reshape(-1, 1),
                                  cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT).ravel()
        sampled_mask = cv2.remap(mask, u.reshape(-1, 1), v.reshape(-1, 1),
                                 cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT).ravel() > 127
        tolerance = np.maximum(absolute_tolerance, relative_tolerance * z)
        visible = (inside & np.isfinite(sampled_depth)
                   & (sampled_depth > .1)
                   & (np.abs(sampled_depth - z) < tolerance))
        if band is not None:
            clear_all[start:end] = cv2.remap(band, u.reshape(-1, 1), v.reshape(-1, 1), cv2.INTER_NEAREST,
                                             borderMode=cv2.BORDER_CONSTANT).ravel() == 0
        visible_all[start:end] = visible
        positive_all[start:end] = visible & sampled_mask
    return visible_all, positive_all, clear_all


def visible_mask_votes(points: np.ndarray, masks_dir: Path, depth_dir: Path,
                       poses: dict, intrinsics: dict,
                       relative_tolerance: float, absolute_tolerance: float,
                       chunk_size: int = 20000, edge_band: float = 0., clear_share: float = 0.,
                       clear_min: int = 3
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Count object and visible observations for world-space surface points.

    With `edge_band`, views where a point lands within that many pixels of the mask's edge abstain: a surface
    seen edge-on projects onto the silhouette, where a mask a few pixels tight - or a pose a few pixels off -
    votes it out (on convertible, 91% of the votes against faces on the real object fell within 8 px of the
    mask). A point counts only its clear views if there are at least `clear_min` of them (and at least
    `clear_share` of its visible views); otherwise every view votes, so a point near the edge almost
    everywhere - the floor along the object's base - is not decided by one or two clear views. Without that
    guard, wider bands broke jeep (official F1 93.0 -> 78.4 at 16 px); with 3 it holds at 93.2."""
    def votes(frame_id):
        mask_path = masks_dir / f'{frame_id:08d}.png'
        depth_path = depth_dir / f'{frame_id:08d}_grounded.npy'
        if not mask_path.is_file() or not depth_path.is_file():
            return None
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        depth = np.load(depth_path, allow_pickle=False).astype('float32')
        if mask is None or mask.shape != depth.shape:
            raise ValueError(f'Mask/depth mismatch: {frame_id}')
        rotation, center = poses[frame_id]
        return frame_votes(points, mask, depth, rotation, center, intrinsics[frame_id],
                           relative_tolerance, absolute_tolerance, chunk_size, edge_band)

    positives = np.zeros(len(points), dtype=np.uint16)
    visible_count = np.zeros(len(points), dtype=np.uint16)
    clear_positives = np.zeros(len(points), dtype=np.uint16)
    clear_visible = np.zeros(len(points), dtype=np.uint16)
    # Frames are independent and votes are integer counts, so order does not matter.
    with ThreadPoolExecutor(os.cpu_count() or 4) as pool:
        for result in bar(pool.map(votes, sorted(poses)), total=len(poses),
                          desc='Mask votes', unit='frame'):
            if result is None:
                continue
            visible, positive, clear = result
            visible_count += visible
            positives += positive
            clear_visible += visible & clear
            clear_positives += positive & clear
    if edge_band <= 0:
        return positives, visible_count
    use_clear = (clear_visible >= clear_share * visible_count) & (clear_visible >= clear_min)
    return np.where(use_clear, clear_positives, positives), np.where(use_clear, clear_visible, visible_count)


def weakly_supported(component_ids: np.ndarray, sizes: np.ndarray, positives: np.ndarray,
                     min_support: float, report: dict) -> np.ndarray:
    """Faces of pieces masked in far fewer views than the largest piece.

    A mask that briefly covers something else (a chair the segmenter took for the
    object in a few frames) passes the per-face vote, because those few views all
    agree, but the piece it leaves is masked in only a handful of views. Real parts
    of the object are seen and masked about as often as its main body, so pieces
    whose median mask count is below min_support times the largest piece's are dropped.
    """
    order = np.argsort(component_ids, kind='stable')
    groups = np.split(positives[order], np.cumsum(np.bincount(component_ids, minlength=len(sizes)))[:-1])
    medians = np.array([np.median(group) if len(group) else 0. for group in groups])
    weak = medians < min_support * medians[np.argmax(sizes)]
    for component in np.flatnonzero(sizes >= 1000):
        report[int(component)] = {'faces': int(sizes[component]), 'median_mask_views': float(medians[component]),
                                  'dropped': bool(weak[component])}
    return weak[component_ids]


def object_bounds(cloud_path: Path, margin: float) -> o3d.geometry.AxisAlignedBoundingBox:
    cloud = o3d.io.read_point_cloud(str(cloud_path))
    points = np.asarray(cloud.points)
    if len(points) < 100:
        raise ValueError(f'Need object cloud for broad extraction bounds: {cloud_path}')
    lower = np.quantile(points, .001, axis=0) - margin
    upper = np.quantile(points, .999, axis=0) + margin
    return o3d.geometry.AxisAlignedBoundingBox(lower, upper)


def extract(scene_path: Path, bounds_cloud: Path, masks_dir: Path,
            depth_dir: Path, trajectory: Path, model: Path, output: Path,
            min_views: int, min_ratio: float, relative_tolerance: float,
            absolute_tolerance: float, bounds_margin: float,
            min_component_faces: int, min_component_support: float = 0., edge_band: float = 0.,
            clear_share: float = 0., clear_min: int = 3) -> dict:
    mesh = o3d.io.read_triangle_mesh(str(scene_path))
    if not len(mesh.triangles):
        raise ValueError(f'Empty scene mesh: {scene_path}')
    total_faces = len(mesh.triangles)
    mesh = mesh.crop(object_bounds(bounds_cloud, bounds_margin))
    if not len(mesh.triangles):
        raise ValueError('Object bounds contain no scene triangles')
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    centers = vertices[faces].mean(axis=1)
    positives, visible = visible_mask_votes(
        centers, masks_dir, depth_dir, load_poses(trajectory),
        frame_intrinsics(model), relative_tolerance, absolute_tolerance, edge_band=edge_band, clear_share=clear_share, clear_min=clear_min)
    ratio = positives / np.maximum(visible, 1)
    selected = ((visible >= min_views) & (positives >= min_views)
                & (ratio >= min_ratio))
    if not selected.any():
        raise ValueError('No scene triangles received enough object-mask votes')
    mesh.remove_triangles_by_mask(~selected)
    mesh.remove_unreferenced_vertices()
    component_ids, sizes = triangle_components(np.asarray(mesh.triangles))
    support = {}
    if len(sizes):
        cutoff = min(min_component_faces, int(np.max(sizes)))
        drop = sizes[component_ids] < cutoff
        if min_component_support > 0:
            drop |= weakly_supported(component_ids, sizes, positives[selected], min_component_support, support)
        mesh.remove_triangles_by_mask(drop)
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    if not len(mesh.triangles):
        raise ValueError('No object mesh after component filtering')
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output), mesh):
        raise OSError(output)
    report = {'method': 'full-scene mesh + visible mask-vote extraction',
              'scene_faces': total_faces, 'cropped_faces': len(faces),
              'mask_selected_faces': int(selected.sum()),
              'object_faces': len(mesh.triangles),
              'object_vertices': len(mesh.vertices),
              'min_views': min_views, 'min_ratio': min_ratio,
              'relative_visibility_tolerance': relative_tolerance,
              'absolute_visibility_tolerance_map_units': absolute_tolerance,
              'bounds_margin_map_units': bounds_margin,
              'min_component_faces': min_component_faces,
              'min_component_support': min_component_support, 'edge_band_px': edge_band, 'clear_share': clear_share, 'clear_min': clear_min,
              'components': support,
              'median_visible_views_selected': float(np.median(visible[selected])),
              'median_object_vote_ratio_selected': float(np.median(ratio[selected]))}
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('scene_mesh', type=Path)
    p.add_argument('--bounds-cloud', type=Path, required=True)
    p.add_argument('--masks-dir', type=Path, required=True)
    p.add_argument('--depth-dir', type=Path, required=True)
    p.add_argument('--trajectory', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--min-views', type=int, default=2)
    p.add_argument('--min-ratio', type=float, default=.6)
    p.add_argument('--relative-tolerance', type=float, default=.06)
    p.add_argument('--absolute-tolerance', type=float, default=.12)
    p.add_argument('--bounds-margin', type=float, default=.4)
    p.add_argument('--min-component-faces', type=int, default=300)
    p.add_argument('--min-component-support', type=float, default=.25,
                   help='Drop pieces masked in fewer views than this fraction of the largest piece (0: off)')
    p.add_argument('--edge-band', type=float, default=0.,
                   help='Pixels either side of the mask edge where a face abstains from the vote (0: off)')
    p.add_argument('--clear-share', type=float, default=0.,
                   help='With --edge-band: a face counts only its clear views if they are at least this share of its visible views')
    p.add_argument('--clear-min', type=int, default=3,
                   help='With --edge-band: and at least this many clear views; otherwise every view votes')
    a = p.parse_args()
    if (a.edge_band < 0 or a.min_views < 1 or not 0 < a.min_ratio <= 1
            or a.relative_tolerance <= 0 or a.absolute_tolerance <= 0
            or a.bounds_margin < 0 or a.min_component_faces < 0
            or not 0 <= a.min_component_support <= 1):
        p.error('Invalid vote, visibility, or bounds settings')
    report = extract(a.scene_mesh, a.bounds_cloud, a.masks_dir,
                             a.depth_dir, a.trajectory, a.model, a.output,
                             a.min_views, a.min_ratio, a.relative_tolerance,
                             a.absolute_tolerance, a.bounds_margin,
                             a.min_component_faces, a.min_component_support, a.edge_band, a.clear_share, a.clear_min)
    done(f"Object extraction: {report['object_faces']:,} faces kept of {report['cropped_faces']:,}")


if __name__ == '__main__':
    main()
