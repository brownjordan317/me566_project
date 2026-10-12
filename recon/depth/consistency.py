#!/usr/bin/env python3
"""Drop depth pixels that other registered views see straight through.

Each pixel's 3D point is projected into reference views spread over the whole
sequence. A view that observes a surface at the same depth supports the point;
a view that observes something clearly farther away along that ray has looked
through the point, so the point lies in that view's free space. Points with
more such violations than supports are removed before fusion. Agreement alone
is not enough: a thin part seen from one side can be misplaced consistently by
every view on that side, and only views from the other side expose it.

The support / free-space-violation vote follows visibility-based depth-map
fusion (Merrell et al., "Real-Time Visibility-Based Fusion of Depth Maps",
ICCV 2007); this is an independent GPU implementation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from recon.common.camera import frame_intrinsics
from recon.common.trajectory import load_poses
from recon.common.progress import bar, done


def project_votes(points: torch.Tensor, depth: torch.Tensor, K: torch.Tensor,
                  rotation: torch.Tensor, center: torch.Tensor, tolerance: float
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Support and free-space-violation masks of world points in one view."""
    camera = (points - center) @ rotation
    z = camera[:, 2]
    safe = torch.clamp(z, min=1e-6)
    u = torch.round(K[0, 0] * camera[:, 0] / safe + K[0, 2]).long()
    v = torch.round(K[1, 1] * camera[:, 1] / safe + K[1, 2]).long()
    height, width = depth.shape
    inside = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    observed = torch.zeros_like(z)
    observed[inside] = depth[v[inside], u[inside]]
    seen = inside & (observed > 0)
    support = seen & ((observed - z).abs() <= tolerance * z)
    violation = seen & (observed > z * (1 + tolerance))
    return support, violation


def filter_frame(frame: int, depths: dict, intrinsics: dict, poses: dict,
                 references: list[int], tolerance: float, device: str
                 ) -> tuple[np.ndarray, dict]:
    depth = depths[frame]
    K = intrinsics[frame]
    rotation, center = poses[frame]
    v, u = torch.nonzero(depth > 0, as_tuple=True)
    z = depth[v, u]
    camera = torch.stack(((u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z), 1)
    points = camera @ rotation.T + center
    supports = torch.zeros(len(z), dtype=torch.int32, device=device)
    violations = torch.zeros_like(supports)
    for other in references:
        if other == frame:
            continue
        other_rotation, other_center = poses[other]
        support, violation = project_votes(points, depths[other], intrinsics[other],
                                           other_rotation, other_center, tolerance)
        supports += support
        violations += violation
    rejected = violations > supports
    filtered = depth.clone()
    filtered[v[rejected], u[rejected]] = 0
    return filtered.cpu().numpy(), {
        'frame': frame, 'valid_pixels': int(len(z)), 'rejected_pixels': int(rejected.sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--depth-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference-views', type=int, default=24,
                        help='Views spread evenly over the sequence that vote on each pixel')
    parser.add_argument('--tolerance', type=float, default=.03,
                        help='Relative depth difference counted as agreement')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.reference_views < 2 or not 0 < args.tolerance < 1:
        parser.error('Need at least two reference views and a tolerance in (0, 1)')
    device = args.device
    poses = load_poses(args.trajectory)
    intrinsics = frame_intrinsics(args.model)
    frames = [frame for frame in sorted(poses)
              if (args.depth_dir / f'{frame:08d}_grounded.npy').is_file()]
    if len(frames) < 3:
        parser.error('Need at least three registered frames with depth')
    depths = {}
    for frame in bar(frames, desc='Loading depth', unit='frame'):
        depth = np.load(args.depth_dir / f'{frame:08d}_grounded.npy', allow_pickle=False)
        depths[frame] = torch.from_numpy(
            np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype('float32')).to(device)
    as_tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    intrinsics = {frame: as_tensor(intrinsics[frame]) for frame in frames}
    poses = {frame: (as_tensor(poses[frame][0]), as_tensor(poses[frame][1])) for frame in frames}
    count = min(args.reference_views, len(frames))
    references = [frames[index] for index in
                  np.unique(np.linspace(0, len(frames) - 1, count).round().astype(int))]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    with torch.inference_mode():
        for frame in bar(frames, desc='Cross-view depth check', unit='frame'):
            filtered, report = filter_frame(frame, depths, intrinsics, poses, references,
                                            args.tolerance, device)
            np.save(args.output_dir / f'{frame:08d}_grounded.npy', filtered)
            reports.append(report)
    rejected = sum(row['rejected_pixels'] for row in reports)
    total = sum(row['valid_pixels'] for row in reports)
    summary = {'method': 'free-space violation vote over reference views',
               'frames': len(reports), 'reference_views': len(references),
               'tolerance': args.tolerance, 'rejected_fraction': rejected / max(total, 1),
               'per_frame': reports}
    (args.output_dir / 'consistency.json').write_text(json.dumps(summary, indent=2) + '\n')
    done(f"Cross-view depth check: {summary['rejected_fraction']:.1%} of depth pixels rejected")


if __name__ == '__main__':
    main()
