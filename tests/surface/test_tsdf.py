"""Missing depth must not look like a measured depth discontinuity."""
import numpy as np

from recon.surface.tsdf import depth_validity, frame_blocks


def test_missing_neighbor_does_not_reject_flat_surface():
    depth = np.full((15, 15), 2., np.float32)
    depth[7, 7] = 0
    valid = depth_validity(depth, .5, 30., .03)
    assert not valid[7, 7]
    assert valid[6:9, 6:9].sum() == 8


def test_actual_depth_step_is_rejected():
    depth = np.full((15, 15), 2., np.float32)
    depth[:, 8:] = 3.
    valid = depth_validity(depth, .5, 30., .03)
    assert not valid[7, 7]
    assert not valid[7, 8]
    assert valid[7, 3]
    assert valid[7, 12]


def test_frame_blocks_cover_every_surface_block_and_truncation_band():
    depth = np.ones((64, 64), np.float32)
    K = np.array([[100., 0, 32], [0, 100., 32], [0, 0, 1]])
    keys = frame_blocks(depth, depth > 0, K, np.eye(3), np.zeros(3),
                        block_size=.1, truncation=.02, stride=1)
    # The plane z=1 spans x,y in [-0.32, 0.31]; widened by the 0.02 truncation
    # band that is [-0.34, 0.33], 8x8 block columns. Along z the band crosses
    # the 0.9..1.0 / 1.0..1.1 block boundary, so two layers.
    assert len(keys) == 8 * 8 * 2
    assert len(np.unique(keys)) == len(keys)
