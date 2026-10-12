#!/usr/bin/env python3
"""Audit COLMAP world-to-image projection against its measured image points."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from recon.common.camera import frame_intrinsics


def audit(model: Path) -> dict:
    intrinsics = frame_intrinsics(model)
    points = {}
    for line in (model / 'points3D.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            fields = line.split()
            points[int(fields[0])] = np.asarray(fields[1:4], dtype=np.float64)
    lines = [line for line in (model / 'images.txt').read_text().splitlines()
             if line and not line.startswith('#')]
    errors = []
    for image_line, observation_line in zip(lines[::2], lines[1::2]):
        fields = image_line.split()
        frame = int(Path(fields[9]).stem)
        q = np.array([float(fields[i]) for i in range(1, 5)])
        rotation = Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix()
        translation = np.array([float(fields[i]) for i in range(5, 8)])
        observations = observation_line.split()
        ids = np.array([int(observations[i + 2]) for i in range(0, len(observations), 3)])
        good = np.array([i for i, point_id in enumerate(ids) if point_id in points])
        if not len(good):
            continue
        world = np.array([points[ids[i]] for i in good])
        measured = np.array([[float(observations[3*i]), float(observations[3*i+1])]
                             for i in good])
        camera = world @ rotation.T + translation
        valid = camera[:, 2] > 0
        if not np.any(valid):
            continue
        K = intrinsics[frame]
        projected = camera[valid, :2] / camera[valid, 2, None]
        projected = projected * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        errors.extend(np.linalg.norm(projected - measured[valid], axis=1))
    if not errors:
        raise ValueError('No visible COLMAP observations')
    return {'observations': len(errors), 'median_px': float(np.median(errors)),
            'p90_px': float(np.percentile(errors, 90)),
            'p99_px': float(np.percentile(errors, 99))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.model), indent=2))


if __name__ == '__main__':
    main()
