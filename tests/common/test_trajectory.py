"""Camera-to-world trajectory convention."""
import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from recon.common.trajectory import load_poses


def test_camera_pose_convention(tmp_path: Path):
    path = tmp_path / 'trajectory.csv'
    quaternion = Rotation.from_euler('y', 30, degrees=True).as_quat()
    center = np.array([1., 2., 3.])
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['frame', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
        writer.writerow([5, *center, *quaternion])
    camera_to_world, loaded_center = load_poses(path)[5]
    world_to_camera = camera_to_world.T
    translation = -world_to_camera @ loaded_center
    np.testing.assert_allclose(world_to_camera,
                               Rotation.from_quat(quaternion).as_matrix().T)
    np.testing.assert_allclose(world_to_camera @ center + translation,
                               np.zeros(3), atol=1e-12)
