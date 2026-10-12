"""Checks camera reprojection, visibility, and full-image depth gating."""
from pathlib import Path

import cv2
import numpy as np

from recon.surface.object import visible_mask_votes, weakly_supported
from recon.surface.tsdf import depth_validity


def test_depth_validity_rejects_sobel_jumps():
    depth = np.full((7, 7), 2., dtype=np.float32)
    depth[:, 5:] = 3.
    valid = depth_validity(depth, .5, 30., .03)
    assert valid[3, 2]
    assert not valid[3, 4]
    assert not valid[3, 5]


def test_mask_votes_require_correct_warp_and_depth_visibility(tmp_path: Path):
    masks = tmp_path / 'masks'
    depths = tmp_path / 'depths'
    masks.mkdir()
    depths.mkdir()
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[3, 3] = 255
    cv2.imwrite(str(masks / '00000000.png'), mask)
    np.save(depths / '00000000_grounded.npy',
            np.full((7, 7), 2., dtype=np.float32))
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    center = np.array([1., 2., 0.])
    camera_points = np.array([[0., 0., 2.], [.5, 0., 2.], [0., 0., 3.]])
    world_points = center + camera_points @ rotation.T
    K = np.array([[4., 0., 3.], [0., 4., 3.], [0., 0., 1.]])
    positive, visible = visible_mask_votes(
        world_points, masks, depths, {0: (rotation, center)}, {0: K},
        relative_tolerance=.02, absolute_tolerance=.05)
    assert positive.tolist() == [1, 0, 0]
    assert visible.tolist() == [1, 1, 0]


def test_weakly_supported_drops_pieces_masked_in_few_views():
    # Piece 0: main body, masked in ~40 views. Piece 1: a part seen as often.
    # Piece 2: something the masks covered in only 4 views.
    ids = np.array([0] * 6 + [1] * 3 + [2] * 3)
    sizes = np.bincount(ids)
    positives = np.array([40, 42, 38, 41, 39, 40, 35, 30, 33, 4, 5, 3])
    report = {}
    drop = weakly_supported(ids, sizes, positives, .25, report)
    assert not drop[:9].any()
    assert drop[9:].all()
