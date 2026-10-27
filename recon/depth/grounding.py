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


def fit_regionwise(predicted, coords, target, regions: int, min_anchors: int = 40):
    """Affine-fit each region of anchors separately, then blend the fits over the image.

    A single scale-and-shift per frame assumes the prediction is affine over the whole image. Monocular
    depth is only *locally* affine, so one global fit is mis-specified wherever a frame holds both a near
    surface and a far one - a table and the object standing on it, or a scene of any depth range. Measured
    on a matched pair of jeep runs (2026-10-26), widening the anchors from the object to the whole scene
    raised DA3's median relative anchor error from 0.0057 to 0.0139 and moved the fitted scale 7.5%, while
    the surface lost 4% of its area on big_ben, the scene with the largest depth range.

    Regions are clustered in **position and predicted depth together**, not position alone: a purely
    spatial cluster straddles a depth discontinuity, fits the two sides at once and comes out worse than
    the global fit it was meant to beat (measured while building this). Each pixel then takes a Gaussian-
    weighted blend of the fits in that same space, so a pixel on the near surface is served by the near
    surface's fit even where the far one is closer on screen. A cluster with too few anchors is dropped;
    with fewer than two survivors this is exactly the global fit.
    """
    if regions <= 1:
        return fit_metric_depth(predicted, coords, target)
    sampled = cv2.remap(predicted, coords[:, 0].astype('float32').reshape(-1, 1),
                        coords[:, 1].astype('float32').reshape(-1, 1), cv2.INTER_LINEAR).ravel()
    usable = np.isfinite(sampled) & (sampled > 0)
    if usable.sum() < max(min_anchors * 2, 40):
        return fit_metric_depth(predicted, coords, target)
    height, width = predicted.shape[:2]
    span = float(max(height, width))
    depth_scale = float(np.median(sampled[usable])) or 1.
    feature = lambda uv, d: np.column_stack((uv[:, 0] / span, uv[:, 1] / span, d / depth_scale))
    points = feature(coords[usable, :2].astype('float64'), sampled[usable].astype('float64'))
    centres = points[np.linspace(0, len(points) - 1, regions).astype(int)].copy()
    for _ in range(12):                       # Lloyd iterations; converges fast in three dimensions
        assignment = np.argmin(((points[:, None, :] - centres[None]) ** 2).sum(-1), 1)
        moved = np.stack([points[assignment == k].mean(0) if (assignment == k).any() else centres[k]
                          for k in range(regions)])
        if np.allclose(moved, centres):
            break
        centres = moved
    index = np.flatnonzero(usable)
    fits, kept = [], []
    for k in range(regions):
        member = index[assignment == k]
        if len(member) < min_anchors:
            continue
        try:
            _, fit = fit_metric_depth(predicted, coords[member], target[member])
        except ValueError:
            continue
        fits.append(fit)
        kept.append(centres[k])
    if len(fits) < 2:
        return fit_metric_depth(predicted, coords, target)
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype='float32'), np.arange(height, dtype='float32'))
    pixels = np.stack((grid_x.ravel() / span, grid_y.ravel() / span,
                       predicted.reshape(-1).astype('float32') / depth_scale), 1)
    centre_array = np.stack(kept).astype('float32')
    spacing = np.linalg.norm(centre_array[:, None, :] - centre_array[None], axis=-1)
    np.fill_diagonal(spacing, np.inf)
    sigma = max(float(np.median(spacing.min(1))) / 2, 1e-3)
    # ||p - c||^2 as ||p||^2 - 2 p.c + ||c||^2, which is one small matrix product. Forming the
    # (pixels, regions, 3) difference array instead cost 0.2 s per frame - 27 s over a 127-frame scene -
    # for a 960x720 image, because that intermediate is 133 MB before it is ever reduced.
    # Written as one buffer reused in place, with the reductions as matrix products: the readable version
    # allocated five (pixels, regions) temporaries per frame and spent 143 ms on a 960x720 image against
    # the global fit's 5. Arithmetic is identical; only the traffic differs.
    weight = pixels @ (-2 * centre_array.T)
    weight += (centre_array ** 2).sum(1)[None, :]
    weight += (pixels ** 2).sum(1)[:, None]
    np.maximum(weight, 0, out=weight)
    weight *= np.float32(-1. / (2 * sigma ** 2))
    np.exp(weight, out=weight)
    weight /= np.maximum(weight.sum(1, keepdims=True), np.float32(1e-12))
    pair = np.array([[f['scale'], f['shift']] for f in fits], 'float32')
    blended = weight @ pair
    scale = blended[:, 0].reshape(predicted.shape[:2])
    shift = blended[:, 1].reshape(predicted.shape[:2])
    depth = scale * predicted + shift
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype('float32')
    # Report the blend the pixels actually received, measured at the anchors, so the numbers stay
    # comparable with the global fit's.
    final = cv2.remap(depth, coords[:, 0].astype('float32').reshape(-1, 1),
                      coords[:, 1].astype('float32').reshape(-1, 1), cv2.INTER_LINEAR).ravel()
    good = np.isfinite(final) & (final > 0) & np.isfinite(target) & (target > 0)
    median_scale, median_shift = float(np.median(scale)), float(np.median(shift))
    typical = median_scale * float(np.median(sampled[usable]))
    return depth, {'scale': median_scale, 'shift': median_shift,
                   'scale_share': typical / (typical + median_shift) if typical + median_shift > 0 else 0.,
                   'anchors': int(good.sum()), 'regions': len(fits),
                   'median_anchor_error': float(np.median(np.abs(final[good] - target[good])))}


def ground(prediction: np.ndarray, space: str, coords: np.ndarray, target: np.ndarray,
           regions: int = 1):
    """Affine-fit a prediction to anchor depths in the space it is affine in ('depth' or 'inverse').

    Adds relative_anchor_error: the median anchor residual over the median
    anchor value in that space, comparable across depth and inverse-depth fits.
    """
    if space == 'depth':
        grounded, report = fit_regionwise(prediction, coords, target, regions)
        report['relative_anchor_error'] = report['median_anchor_error'] / float(np.median(target))
        return grounded, report
    inverse, report = fit_regionwise(prediction, coords, 1. / target, regions)
    report['relative_anchor_error'] = report['median_anchor_error'] / float(np.median(1. / target))
    return np.where(inverse > 0, 1. / np.maximum(inverse, 1e-9), 0).astype('float32'), report
