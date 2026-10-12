"""Load camera-to-world poses from the project trajectory CSV."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load_poses(trajectory: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    poses = {}
    with trajectory.open(newline='') as stream:
        for row in csv.DictReader(stream):
            frame = int(row['frame'])
            center = np.array([float(row[key]) for key in ('tx', 'ty', 'tz')])
            quaternion = [float(row[key]) for key in ('qx', 'qy', 'qz', 'qw')]
            poses[frame] = (Rotation.from_quat(quaternion).as_matrix(), center)
    return poses
