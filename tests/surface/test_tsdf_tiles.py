"""Tiling helpers of the TSDF stage."""
import numpy as np

from recon.surface.tsdf import frame_extent, tile_boxes


def test_tile_boxes_partition_the_volume_without_gaps_or_overlap():
    bounds = np.array([[0., 0., 0.], [2., 4., 6.]])
    boxes = tile_boxes(bounds, 2)
    assert len(boxes) == 8
    volume = sum(float(np.prod(box[1] - box[0])) for box in boxes)
    assert np.isclose(volume, float(np.prod(bounds[1] - bounds[0])))
    assert np.allclose(np.min([b[0] for b in boxes], axis=0), bounds[0])
    assert np.allclose(np.max([b[1] for b in boxes], axis=0), bounds[1])
    assert len({tuple(np.round(box[0], 6)) for box in boxes}) == 8      # distinct corners, so no overlap


def test_tile_boxes_of_one_split_is_the_whole_box():
    bounds = np.array([[-1., -2., -3.], [1., 2., 3.]])
    (box,) = tile_boxes(bounds, 1)
    assert np.allclose(box, bounds)


def test_frame_extent_covers_the_back_projected_surface():
    K = np.array([[50., 0, 16], [0, 50., 12], [0, 0, 1.]])
    depth = np.full((24, 32), 3., dtype='float32')
    extent = frame_extent(depth, K, np.eye(3), np.array([1., 2., 3.]), stride=1)
    assert extent is not None
    assert np.allclose(extent[0][2], 6.) and np.allclose(extent[1][2], 6.)     # the plane sits at z = 3 + 3
    assert extent[0][0] < 1. < extent[1][0]                                    # spans the camera centre in x
    assert np.all(extent[1] - extent[0] >= 0)


def test_frame_extent_is_none_without_valid_depth():
    K = np.eye(3)
    assert frame_extent(np.zeros((8, 8), dtype='float32'), K, np.eye(3), np.zeros(3)) is None
