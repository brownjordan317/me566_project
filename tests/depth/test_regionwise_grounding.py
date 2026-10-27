"""Region-wise grounding must beat one global affine where the depth range is split, and be inert at 1."""
import numpy as np
import pytest

from recon.depth.grounding import fit_metric_depth, fit_regionwise


def two_plane_frame(size=200, anchors=600, seed=0):
    """A prediction that needs a different affine on each half: the left half is compressed."""
    rng = np.random.default_rng(seed)
    # Each half must span a *range* of depths, or one affine fits both halves exactly (two constant
    # depths are two points, and two points determine a line) and the test proves nothing.
    truth = np.empty((size, size), np.float32)
    truth[:, :size // 2] = rng.uniform(1., 2., (size, size // 2))          # a near surface
    truth[:, size // 2:] = rng.uniform(3., 5., (size, size - size // 2))   # and a far one
    predicted = truth.copy()
    predicted[:, :size // 2] = truth[:, :size // 2] * .4 + .2     # each half affine, but differently
    predicted[:, size // 2:] = truth[:, size // 2:] * 1.3 - .5
    x = rng.integers(2, size - 2, anchors)
    y = rng.integers(2, size - 2, anchors)
    coords = np.column_stack((x, y)).astype('float64')
    target = truth[y, x].astype('float64')
    return predicted, coords, target


def test_one_region_is_exactly_the_global_fit():
    predicted, coords, target = two_plane_frame()
    a, ra = fit_regionwise(predicted, coords, target, 1)
    b, rb = fit_metric_depth(predicted, coords, target)
    assert np.array_equal(a, b)
    assert ra == rb


def test_regions_beat_the_global_fit_on_a_split_depth_range():
    predicted, coords, target = two_plane_frame()
    _, glob = fit_metric_depth(predicted, coords, target)
    _, local = fit_regionwise(predicted, coords, target, 4)
    assert local['regions'] >= 2
    # The global fit cannot serve two different affines at once; the blend should be far closer.
    assert local['median_anchor_error'] < glob['median_anchor_error'] / 2


def test_too_few_anchors_falls_back_rather_than_fitting_badly():
    predicted, coords, target = two_plane_frame(anchors=60)
    _, report = fit_regionwise(predicted, coords, target, 16, min_anchors=40)
    # 60 anchors over 16 clusters leaves none big enough, so it must return the global fit.
    assert 'regions' not in report


def test_a_single_affine_frame_is_not_made_worse():
    rng = np.random.default_rng(1)
    truth = rng.uniform(1., 3., (120, 120)).astype(np.float32)
    predicted = truth * .7 + .3
    x, y = rng.integers(2, 118, 500), rng.integers(2, 118, 500)
    coords = np.column_stack((x, y)).astype('float64')
    target = truth[y, x].astype('float64')
    _, glob = fit_metric_depth(predicted, coords, target)
    _, local = fit_regionwise(predicted, coords, target, 4)
    assert local['median_anchor_error'] <= max(glob['median_anchor_error'] * 3, 1e-3)


def test_blending_leaves_no_holes():
    predicted, coords, target = two_plane_frame()
    depth, _ = fit_regionwise(predicted, coords, target, 4)
    assert depth.shape == predicted.shape
    assert (depth > 0).mean() > .99
