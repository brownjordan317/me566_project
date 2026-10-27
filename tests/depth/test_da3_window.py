"""The DA3 window must be sized to the memory the card actually has."""
from recon.depth.multiview.da3 import MIN_WINDOW, fitting_window, windows

GIB = 2 ** 30


def test_a_large_card_keeps_the_measured_default():
    # 20 GiB free after the weights is what this repository's card leaves; window 16 measured 14.21 GiB
    # peak there, so auto must not reduce it and change the published results.
    assert fitting_window(20 * GIB, 756, 16) == 16


def test_a_small_card_is_given_a_window_it_can_hold():
    # 8 GiB free fits 5.66 + 0.215n within 7.2 GiB: seven frames, not the capped sixteen.
    assert fitting_window(8 * GIB, 756, 16) == 7


def test_a_tiny_card_still_gets_the_umeyama_minimum():
    # Below three camera centres DA3's pose alignment is degenerate, so refusing is not an option:
    # the floor is returned and the OOM halving in the stage is what catches a window that will not fit.
    assert fitting_window(2 * GIB, 756, 16) == MIN_WINDOW


def test_resolution_costs_its_square():
    # Doubling the processing resolution quadruples the tokens, so the same card holds a quarter as many
    # frames; at 1512 the fixed term alone (4 x 5.66) exceeds a 20 GiB budget.
    assert fitting_window(20 * GIB, 1512, 16) == MIN_WINDOW
    assert fitting_window(80 * GIB, 1512, 16) == 16


def test_the_cap_is_never_exceeded():
    assert fitting_window(400 * GIB, 756, 16) == 16


def test_windows_cover_every_frame():
    for count in (3, 7, 16, 17, 64):
        for size in (3, 4, 8, 16):
            covered = {i for window in windows(count, size, max(1, size // 2)) for i in window}
            assert covered == set(range(count)), (count, size)
