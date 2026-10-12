"""Neighbour views, photometric scoring, pooling and candidate grounding."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from recon.depth.grounding import ground
from recon.depth.fusion.photometric import Images, ncc_scores, pool
from recon.depth.fusion.views import neighbours


def test_neighbours_prefer_similar_viewing_directions():
    angles = [0, 5, 40, 90]
    rotations = np.stack([Rotation.from_euler('y', a, degrees=True).as_matrix().T for a in angles])
    translations = np.stack([np.array([0, 0, 5.]) + i for i in range(4)])
    assert neighbours(rotations, translations, 2)[0] == [1, 2]


def test_correct_depth_scores_higher_than_wrong_depth():
    rng = np.random.default_rng(0)
    texture = rng.uniform(0, 255, (120, 160)).astype(np.float32)
    # Two cameras looking at a textured plane at depth 4; the second shifted sideways.
    K = np.array([[100., 0, 80], [0, 100., 60], [0, 0, 1]])
    shift = 0.4
    second = np.roll(texture, -int(round(100 * shift / 4)), axis=1)   # disparity f*b/z = 10 px
    images = Images.__new__(Images)
    images.gray, images.scale = {0: torch.tensor(texture), 1: torch.tensor(second)}, 1.
    T = lambda x: torch.as_tensor(x, dtype=torch.float32)
    R = {0: T(np.eye(3)), 1: T(np.eye(3))}
    t = {0: T(np.zeros(3)), 1: T([-shift, 0, 0])}
    Ks = {0: T(K), 1: T(K)}
    u, v = T([70., 80, 90]), T([50., 60, 70])
    right = ncc_scores(images, 0, [1], Ks, R, t, T([4., 4, 4]), u, v, radius=3)
    wrong = ncc_scores(images, 0, [1], Ks, R, t, T([5., 5, 5]), u, v, radius=3)
    assert (right > .9).all() and (right > wrong + .3).all()


def test_pool_averages_only_valid_samples():
    values = np.zeros((5, 5), np.float32)
    valid = np.zeros((5, 5), bool)
    values[2, 2], valid[2, 2] = 1., True
    pooled = pool(values, valid, 3)
    assert pooled[2, 2] == 1. and pooled[1, 1] == 1. and np.isnan(pooled[0, 4])


def test_inverse_depth_grounding_recovers_depth():
    depth = np.linspace(2, 4, 400, dtype=np.float32).reshape(20, 20)
    disparity = 3. / depth + .1                                      # affine in inverse depth
    coords = np.stack(np.meshgrid(np.arange(0, 20, 2), np.arange(0, 20, 2)), -1).reshape(-1, 2).astype(np.float32)
    target = depth[coords[:, 1].astype(int), coords[:, 0].astype(int)]
    grounded, report = ground(disparity, 'inverse', coords, target)
    np.testing.assert_allclose(grounded, depth, rtol=1e-3)
    assert report['relative_anchor_error'] < 1e-3


def test_angle_band_prefers_wider_baselines():
    angles = [0, 2, 4, 20, 25, 60]
    rotations = np.stack([Rotation.from_euler('y', a, degrees=True).as_matrix().T for a in angles])
    translations = np.stack([np.array([0, 0, 5.]) + i for i in range(len(angles))])
    assert set(neighbours(rotations, translations, 2, band=(10, 35))[0]) == {3, 4}
    assert neighbours(rotations, translations, 3, band=(10, 35))[0][2] == 1     # closest view fills in


def test_slanted_patches_match_a_tilted_plane_better():
    from recon.depth.fusion.photometric import camera_normals
    K = np.array([[100., 0, 40], [0, 100., 30], [0, 0, 1]])
    v, u = np.mgrid[:60, :80].astype(np.float32)
    rays = np.stack(((u - 40) / 100, (v - 30) / 100, np.ones_like(u)), -1)
    normal = np.array([.5, 0, -1.]) / np.linalg.norm([.5, 0, -1.])
    depth = (normal @ np.array([0, 0, 4.])) / (rays @ normal)          # plane through (0,0,4)
    estimated = camera_normals(depth.astype(np.float32), K)[30, 40]
    assert np.dot(estimated, normal) > .99
