"""Fit predicted depth maps to the sparse COLMAP points they should agree with.

Each frame's depth prediction (MoGe, VGGT) is only correct up to an affine
change. The sparse points observed in that frame, optionally only those on the
object mask, give metric targets in map units; a robust scale-and-shift fit
maps the prediction onto them.

Least-squares scale-and-shift alignment of relative depth follows the
evaluation practice of MiDaS (Ranftl et al., TPAMI 2022); the robust trimming
and the per-frame rejection rule are this project's.
"""
from __future__ import annotations

import cv2
import numpy as np


def point_table(points: dict) -> tuple[np.ndarray, np.ndarray]:
    """Sorted point ids and their positions, for vectorized track lookups."""
    ids = np.fromiter(points, dtype='int64', count=len(points))
    order = np.argsort(ids)
    xyz = (np.asarray([points[point_id] for point_id in ids[order]], dtype='float64')
           .reshape(-1, 3))
    return ids[order], xyz


def object_anchors(observations, points, rotation, translation, mask):
    """points is a {id: xyz} dict or a point_table() result."""
    point_ids, xyz = point_table(points) if isinstance(points, dict) else points
    ids = observations[:, 2].astype('int64')
    if not len(point_ids):
        raise ValueError('No COLMAP points observed in frame')
    slot = np.minimum(np.searchsorted(point_ids, ids), len(point_ids) - 1)
    keep = point_ids[slot] == ids
    if not keep.any():
        raise ValueError('No COLMAP points observed in frame')
    coords = observations[keep, :2]
    camera = xyz[slot[keep]] @ rotation.T + translation
    x = np.rint(coords[:, 0]).astype(int)
    y = np.rint(coords[:, 1]).astype(int)
    inside = ((x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0]))
    on_object = np.zeros(len(coords), bool)
    on_object[inside] = mask[y[inside], x[inside]] > 0
    good = (camera[:, 2] > 0) & on_object
    if good.sum() < 30:
        raise ValueError(f'Only {good.sum()} object depth anchors')
    return coords[good], camera[good, 2]


def fit_metric_depth(predicted, coords, target):
    """Robust positive affine calibration from MoGe meters to COLMAP units."""
    x = cv2.remap(predicted, coords[:, 0].astype('float32').reshape(-1, 1),
                  coords[:, 1].astype('float32').reshape(-1, 1),
                  cv2.INTER_LINEAR).ravel().astype('float64')
    good = np.isfinite(x) & np.isfinite(target) & (x > 0) & (target > 0)
    x, y = x[good], target[good]
    if len(x) < 20:
        raise ValueError(f'Only {len(x)} valid MoGe depth anchors')
    accepted = np.ones(len(x), bool)
    for _ in range(8):
        scale, shift = np.linalg.lstsq(
            np.column_stack((x[accepted], np.ones(accepted.sum()))),
            y[accepted], rcond=None)[0]
        residual = np.abs(scale * x + shift - y)
        cutoff = max(float(np.quantile(residual[accepted], .7)),
                     float(.02 * np.median(y)))
        updated = residual <= cutoff
        if updated.sum() < 20 or np.array_equal(updated, accepted):
            break
        accepted = updated
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.median(y / x))
        shift = 0.
    depth = scale * predicted + shift
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype('float32')
    # Share of fitted depth carried by MoGe rather than the constant shift:
    # near 1 for a usable fit, near 0 when the fit collapses to a flat plane.
    typical = scale * float(np.median(x[accepted]))
    scale_share = typical / (typical + shift) if typical + shift > 0 else 0.
    return depth, {'scale': float(scale), 'shift': float(shift),
                   'scale_share': float(scale_share),
                   'anchors': int(accepted.sum()),
                   'median_anchor_error': float(np.median(np.abs(
                       scale * x + shift - y)))}


def ground(prediction: np.ndarray, space: str, coords: np.ndarray, target: np.ndarray):
    """Affine-fit a prediction to anchor depths in the space it is affine in ('depth' or 'inverse').

    Adds relative_anchor_error: the median anchor residual over the median
    anchor value in that space, comparable across depth and inverse-depth fits.
    """
    if space == 'depth':
        grounded, report = fit_metric_depth(prediction, coords, target)
        report['relative_anchor_error'] = report['median_anchor_error'] / float(np.median(target))
        return grounded, report
    inverse, report = fit_metric_depth(prediction, coords, 1. / target)
    report['relative_anchor_error'] = report['median_anchor_error'] / float(np.median(1. / target))
    return np.where(inverse > 0, 1. / np.maximum(inverse, 1e-9), 0).astype('float32'), report
