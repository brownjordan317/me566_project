"""Relative-pose accuracy: exact poses score 100, a rotated camera is caught."""
import numpy as np
from scipy.spatial.transform import Rotation

from recon.evaluation.metrics.pose_auc import auc, pair_errors, world_to_camera


def cameras(count, rng):
    return [(Rotation.random(random_state=index).as_matrix(), rng.normal(size=3)) for index in range(count)]


def test_similarity_transformed_poses_have_no_error():
    rng = np.random.default_rng(0)
    reference = cameras(6, rng)
    turn = Rotation.from_euler('z', 40, degrees=True).as_matrix()
    estimate = [(turn @ rotation, 3. * turn @ center + 1.) for rotation, center in reference]
    r, t = pair_errors([world_to_camera(*pose) for pose in estimate], [world_to_camera(*pose) for pose in reference])
    assert r.max() < 1e-3 and t.max() < 1e-3
    assert auc(r, t, 5) == 1.


def test_one_rotated_camera_counts_in_its_pairs_only():
    rng = np.random.default_rng(1)
    reference = cameras(5, rng)
    estimate = list(reference)
    estimate[0] = (reference[0][0] @ Rotation.from_euler('x', 10, degrees=True).as_matrix(), reference[0][1])
    r, _ = pair_errors([world_to_camera(*pose) for pose in estimate], [world_to_camera(*pose) for pose in reference])
    assert np.sum(np.isclose(r, 10, atol=1e-3)) == 4 and np.sum(r < 1e-3) == 6
