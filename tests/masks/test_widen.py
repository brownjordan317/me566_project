"""Mask widening in the automatic mask path."""
import numpy as np

from recon.masks.birefnet import widen


def test_widen_grows_the_mask_by_the_asked_margin():
    mask = np.zeros((100, 960), bool)
    mask[40:60, 400:500] = True
    grown = widen(mask, 2)
    assert grown[38:62, 398:502].all()                  # the 2 px ring is filled
    assert not grown[:36].any() and not grown[64:].any()
    assert grown.sum() > mask.sum()


def test_widen_scales_the_margin_with_frame_width():
    small = np.zeros((50, 480), bool); small[20:30, 200:240] = True
    large = np.zeros((100, 1920), bool); large[40:60, 800:960] = True
    # 480 px is half the reference width, 1920 double, so the rings differ by about four times.
    assert widen(small, 2).sum() - small.sum() < widen(large, 2).sum() - large.sum()


def test_widen_is_a_no_op_at_zero_or_below():
    mask = np.zeros((20, 20), bool); mask[5:15, 5:15] = True
    assert np.array_equal(widen(mask, 0), mask)
    assert np.array_equal(widen(mask, -3), mask)


def test_widen_keeps_an_empty_mask_empty():
    assert not widen(np.zeros((30, 960), bool), 2).any()
