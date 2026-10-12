import numpy as np

from recon.depth.grounding import fit_metric_depth, object_anchors


def test_fit_metric_depth_rejects_bad_tracks_and_keeps_positive_scale():
    yy, xx = np.mgrid[:20, :20]
    predicted = (1. + xx * .03 + yy * .02).astype('float32')
    coords = np.column_stack((xx.ravel(), yy.ravel())).astype('float32')
    truth = 4. * predicted.ravel() + 2.
    truth[::13] += 12.
    fitted, report = fit_metric_depth(predicted, coords, truth)
    assert report['scale'] > 0
    assert report['anchors'] < len(coords)
    assert np.median(np.abs(fitted - (4. * predicted + 2.))) < .01


def test_object_anchors_never_use_background_tracks():
    mask = np.zeros((8, 8), dtype='uint8')
    mask[2:7, 2:7] = 255
    observations = np.array([[3, 3, index] for index in range(35)] +
                            [[0, 0, index] for index in range(35, 70)], float)
    points = {index: np.array([0., 0., 2.]) for index in range(70)}
    coords, depths = object_anchors(observations, points, np.eye(3),
                                    np.zeros(3), mask)
    assert len(coords) == 35
    np.testing.assert_array_equal(depths, np.full(35, 2.))


def test_scale_share_flags_fits_that_collapse_to_a_constant():
    yy, xx = np.mgrid[:20, :20]
    predicted = (1. + xx * .03 + yy * .02).astype('float32')
    coords = np.column_stack((xx.ravel(), yy.ravel())).astype('float32')
    _, proportional = fit_metric_depth(predicted, coords, 3. * predicted.ravel())
    # Room scan failures looked like scale ~0.002, shift ~1.65.
    flat = 1.65 + .002 * predicted.ravel()
    _, constant = fit_metric_depth(predicted, coords, flat)
    assert proportional['scale_share'] > .95
    assert constant['scale_share'] < .1
