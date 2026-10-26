"""The fusion's photometry-free combination rules."""
import numpy as np


import warnings


def combine(stack, present, rule):
    """The branch recon.depth.fusion.filter takes for --combine mean/median, in isolation."""
    covered = present.sum(0)
    if rule == 'mean':
        chosen = (np.where(present, stack, 0).sum(0) / np.maximum(covered, 1)).astype(np.float32)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            chosen = np.nanmedian(np.where(present, stack, np.nan), 0).astype(np.float32)
    return np.where((covered > 0) & np.isfinite(chosen), chosen, 0)


def test_mean_averages_the_candidates_that_are_present():
    stack = np.array([[[2., 4.]], [[4., 8.]]], np.float32)
    present = np.ones_like(stack, bool)
    assert np.allclose(combine(stack, present, 'mean'), [[3., 6.]])


def test_a_missing_candidate_is_ignored_rather_than_counted_as_zero():
    stack = np.array([[[2., 4.]], [[0., 8.]]], np.float32)
    present = np.array([[[True, True]], [[False, True]]])
    # The first pixel has only one candidate, so it keeps that depth instead of being halved.
    assert np.allclose(combine(stack, present, 'mean'), [[2., 6.]])


def test_a_pixel_no_candidate_covers_becomes_zero_not_nan():
    stack = np.zeros((2, 1, 1), np.float32)
    present = np.zeros((2, 1, 1), bool)
    out = combine(stack, present, 'mean')
    assert np.isfinite(out).all() and out[0, 0] == 0


def test_median_resists_one_wild_candidate_where_the_mean_does_not():
    stack = np.array([[[10.]], [[10.4]], [[90.]]], np.float32)
    present = np.ones_like(stack, bool)
    assert np.allclose(combine(stack, present, 'median'), [[10.4]])
    assert combine(stack, present, 'mean')[0, 0] > 30
