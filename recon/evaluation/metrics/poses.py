#!/usr/bin/env python3
"""Similarity-aligned camera position error for a MobileBrick sequence."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def evaluate(trajectory: Path, pose_dir: Path,
             landmarks: Path | None = None, mesh: Path | None = None) -> dict:
    estimates, references = [], []
    with trajectory.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["status"] not in {"bootstrapped", "tracked"}:
                continue
            frame = int(row["frame"])
            pose_file = pose_dir / f"{frame:06d}.txt"
            if not pose_file.is_file():
                raise FileNotFoundError(pose_file)
            estimates.append([float(row[name]) for name in ("tx", "ty", "tz")])
            references.append(np.loadtxt(pose_file)[:3, 3])
    if len(estimates) < 3:
        raise ValueError("At least three localized frames are required")
    estimated = np.asarray(estimates, np.float64)
    reference = np.asarray(references, np.float64)
    estimate_center = estimated.mean(axis=0)
    reference_center = reference.mean(axis=0)
    u, singular, vt = np.linalg.svd(
        (estimated - estimate_center).T @ (reference - reference_center))
    reflection = np.diag([1., 1., np.linalg.det(u @ vt)])
    rotation = u @ reflection @ vt
    scale = float(np.sum(singular * np.diag(reflection)) /
                  np.sum((estimated - estimate_center) ** 2))
    aligned = (estimated - estimate_center) @ rotation * scale + reference_center
    errors = np.linalg.norm(aligned - reference, axis=1)
    summary = {
        "localized_frames": len(errors),
        "similarity_scale": scale,
        "median_position_error_m": float(np.median(errors)),
        "rmse_position_error_m": float(np.sqrt(np.mean(errors ** 2))),
        "p90_position_error_m": float(np.percentile(errors, 90)),
        "max_position_error_m": float(np.max(errors)),
    }
    if landmarks is not None and mesh is not None:
        with landmarks.open() as stream:
            header_lines = 0
            while stream.readline().strip() != "end_header":
                header_lines += 1
        points = np.loadtxt(landmarks, skiprows=header_lines + 1,
                            usecols=(0, 1, 2))
        with mesh.open("rb") as stream:
            vertex_count = None
            while True:
                line = stream.readline()
                if line.startswith(b"element vertex "):
                    vertex_count = int(line.split()[2])
                if line.strip() == b"end_header":
                    break
            if vertex_count is None:
                raise ValueError("Reference mesh PLY has no vertices")
            # MobileBrick's reference PLY stores xyz and normals as six float32s.
            vertices = np.fromfile(stream, dtype="<f4", count=vertex_count * 6)
            vertices = vertices.reshape(-1, 6)[:, :3]
        aligned_points = (points - estimate_center) @ rotation * scale + reference_center
        distances = cKDTree(vertices).query(aligned_points, workers=-1)[0]
        summary["map_to_mesh_median_m"] = float(np.median(distances))
        summary["map_to_mesh_p90_m"] = float(np.percentile(distances, 90))
        summary["map_to_mesh_note"] = "one-sided sparse-point to reference-mesh distance"
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("pose_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--landmarks", type=Path)
    parser.add_argument("--mesh", type=Path)
    args = parser.parse_args()
    if (args.landmarks is None) != (args.mesh is None):
        parser.error("--landmarks and --mesh must be provided together")
    summary = evaluate(args.trajectory, args.pose_dir,
                       args.landmarks, args.mesh)
    result = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    print(result, end="")


if __name__ == "__main__":
    main()
