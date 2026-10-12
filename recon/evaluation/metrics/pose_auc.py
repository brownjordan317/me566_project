#!/usr/bin/env python3
"""Relative-pose accuracy in the protocol of recent feed-forward reconstruction papers.

For every pair of registered frames, the relative rotation and translation are
compared with the reference: the rotation error is the angle of R_est^T R_ref, and
the translation error is the angle between the two relative translation directions
(scale-free, folded so that a flipped direction counts as min(a, 180 - a)). Reported:

- RRA@15 / RTA@15: share of pairs with rotation / translation error under 15 degrees;
- AUC@1, @3, @5, @15, @30: the mean over 1-degree thresholds up to that limit of the
  share of pairs whose larger error (rotation or translation) is below it. AUC@30 is
  the usual headline; AUC@1 and @3 (as in image-matching benchmarks) separate methods
  whose poses are all within a few degrees, as on DTU and MobileBrick.

This reproduces calculate_auc_np, rotation_angle and translation_angle from Depth
Anything 3's benchmark utilities (src/depth_anything_3/bench/utils.py; Apache-2.0,
ByteDance), which follow VGGT's CO3D evaluation (Wang et al., CVPR 2025) and PoseDiffusion
(Wang et al., ICCV 2023). Frames missing from the reconstruction are skipped, so
check `pairs` against the full count when comparing methods that drop frames.
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def world_to_camera(rotation: np.ndarray, center: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation.T, -rotation.T @ center
    return pose


def load(trajectory: Path, pose_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """World-to-camera matrices of the frames present in both, estimate and reference."""
    estimates, references = [], []
    with trajectory.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row.get('status') not in ('tracked', 'bootstrapped'):
                continue
            reference = pose_dir / f"{int(row['frame']):06d}.txt"
            if not reference.is_file():
                continue
            pose = np.loadtxt(reference)                                   # camera-to-world
            rotation = Rotation.from_quat([float(row[key]) for key in ('qx', 'qy', 'qz', 'qw')]).as_matrix()
            center = np.array([float(row[key]) for key in ('tx', 'ty', 'tz')])
            estimates.append(world_to_camera(rotation, center))
            references.append(world_to_camera(pose[:3, :3], pose[:3, 3]))
    return estimates, references


def pair_errors(estimates: list[np.ndarray], references: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation-direction errors (degrees) over all frame pairs."""
    rotation_errors, translation_errors = [], []
    for i, j in combinations(range(len(estimates)), 2):
        estimate = estimates[j] @ np.linalg.inv(estimates[i])
        reference = references[j] @ np.linalg.inv(references[i])
        relative = estimate[:3, :3].T @ reference[:3, :3]
        rotation_errors.append(np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))
        a, b = estimate[:3, 3], reference[:3, 3]
        cosine = abs(a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-15)
        translation_errors.append(np.degrees(np.arccos(np.clip(cosine, 0, 1))))
    return np.array(rotation_errors), np.array(translation_errors)


def auc(rotation_errors: np.ndarray, translation_errors: np.ndarray, limit: int) -> float:
    worst = np.maximum(rotation_errors, translation_errors)
    histogram, _ = np.histogram(worst, bins=np.arange(limit + 1))
    return float(np.mean(np.cumsum(histogram / len(worst))))


def evaluate(trajectory: Path, pose_dir: Path) -> dict:
    estimates, references = load(trajectory, pose_dir)
    if len(estimates) < 2:
        raise ValueError('Need at least two matched frames')
    rotation_errors, translation_errors = pair_errors(estimates, references)
    return {'frames': len(estimates), 'pairs': len(rotation_errors),
            'rra_15': float(np.mean(rotation_errors < 15) * 100),
            'rta_15': float(np.mean(translation_errors < 15) * 100),
            **{f'auc_{limit}': auc(rotation_errors, translation_errors, limit) * 100 for limit in (1, 3, 5, 15, 30)},
            'median_rotation_error_deg': float(np.median(rotation_errors)),
            'median_translation_error_deg': float(np.median(translation_errors))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('trajectory', type=Path)
    p.add_argument('pose_dir', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    report = evaluate(a.trajectory, a.pose_dir)
    text = json.dumps(report, indent=2) + '\n'
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
