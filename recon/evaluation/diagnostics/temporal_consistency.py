#!/usr/bin/env python3
"""Does a stage's depth flicker from frame to frame? Temporal alignment error (TAE) per depth stage.

Video Depth Anything (Chen et al., CVPR 2025) reports temporal consistency as TAE: each frame's depth is warped
into the next frame with the known camera motion and compared there, both ways (following NVDS, Wang et al.,
ICCV 2023). Our depth comes from several stages - two multi-view models, a per-frame monocular model, fusion,
refinement, the planar-Gaussian pass, the cross-view check - each grounded to the sparse map frame by frame, so
flicker could enter at any of them. This computes the same metric for every stage of a run kept with
KEEP_INTERMEDIATE=1, on object pixels, so a temporal fix can be aimed at the stage that needs one, if any.

    TAE = 1 / (2 (N - 1)) * sum_i [ AbsRel(warp(d_i -> i+1), d_i+1) + AbsRel(warp(d_i+1 -> i), d_i) ]

computed over pixels that land inside the other frame's object mask, have depth in both, and whose warped depth
agrees within --occlusion (relative), so surfaces hidden in one view are not counted as flicker. Reported per
stage as the mean over consecutive pairs, with the share of object pixels compared. With --scene, the reference
mesh is rendered into the same cameras and scored the same way: that is the floor the poses and pixel rounding
put under any depth, and only the excess over it is flicker.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from recon.common.colmap import parse_model

STAGES = ('moge_scene', 'vggt', 'da3', 'hybrid', 'fused', 'fused_refined', 'gaussian', 'consistent')


def warp_absrel(depth_a, depth_b, K_a, K_b, R_a, t_a, R_b, t_b, mask_b, occlusion):
    """AbsRel of frame a's depth warped into frame b against b's own depth, over b's object pixels."""
    height, width = depth_a.shape
    v, u = np.nonzero(depth_a > 0)
    z = depth_a[v, u]
    camera = np.stack(((u - K_a[0, 2]) / K_a[0, 0] * z, (v - K_a[1, 2]) / K_a[1, 1] * z, z), 1)
    world = (camera - t_a) @ R_a
    other = world @ R_b.T + t_b
    ub = np.round(K_b[0, 0] * other[:, 0] / other[:, 2] + K_b[0, 2]).astype(int)
    vb = np.round(K_b[1, 1] * other[:, 1] / other[:, 2] + K_b[1, 2]).astype(int)
    inside = (other[:, 2] > 0) & (ub >= 0) & (ub < width) & (vb >= 0) & (vb < height)
    ub, vb, warped = ub[inside], vb[inside], other[inside, 2]
    target = depth_b[vb, ub]
    keep = mask_b[vb, ub] & (target > 0)
    relative = np.abs(warped[keep] - target[keep]) / target[keep]
    visible = relative < occlusion                    # a larger gap is a surface hidden in one of the views
    covered = len(np.unique(vb[keep][visible] * width + ub[keep][visible]))
    return (float(relative[visible].mean()) if visible.any() else float('nan'),
            float(covered / max(int(mask_b.sum()), 1)))


def rendered_reference(a, intrinsics, frames, masks) -> dict[int, np.ndarray]:
    """The reference mesh's depth in each of our cameras, brought into map units by the scorers' similarity."""
    import open3d as o3d
    from recon.evaluation.metrics.mesh import similarity
    centre, target, rotation, scale, _ = similarity(a.run / 'poses/export/trajectory.csv', a.scene / 'pose')
    mesh = o3d.io.read_triangle_mesh(str(a.scene / 'mesh/gt_mesh.ply'))
    mesh.vertices = o3d.utility.Vector3dVector(((np.asarray(mesh.vertices) - target) / scale) @ rotation.T + centre)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    depth = {}
    for number, R, t, _ in frames:
        K = intrinsics[number]
        height, width = masks[number].shape
        v, u = np.mgrid[:height, :width].astype(np.float32)
        directions = np.stack(((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)), -1) @ R
        rays = np.concatenate([np.broadcast_to(-R.T @ t, directions.shape), directions], -1).reshape(-1, 6)
        hit = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))['t_hit'].numpy().reshape(height, width)
        depth[number] = np.where(np.isfinite(hit) & masks[number], hit, 0).astype(np.float32)
    return depth


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path, help='A run kept with KEEP_INTERMEDIATE=1')
    p.add_argument('--sequence', type=Path, required=True, help='Prepared sequence (masks/)')
    p.add_argument('--occlusion', type=float, default=.1, help='Relative gap treated as occlusion, not flicker')
    p.add_argument('--scene', type=Path, help='Dataset scene (mesh/gt_mesh.ply, pose/) for the ground-truth floor')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    intrinsics, _, frames = parse_model(a.run / 'poses/sparse_txt')
    frames = sorted(frames, key=lambda f: f[0])
    masks = {n: cv2.imread(str(a.sequence / 'masks' / f'{n:08d}.png'), cv2.IMREAD_GRAYSCALE) > 127 for n, *_ in frames}
    report = {'run': str(a.run), 'pairs': len(frames) - 1, 'stages': {}}
    reference = rendered_reference(a, intrinsics, frames, masks) if a.scene else None
    for stage in (*STAGES, 'reference'):
        folder = a.run / 'depth' / stage
        if stage == 'reference' and reference is None or stage != 'reference' and not folder.is_dir():
            continue
        errors, shares = [], []
        for (na, Ra, ta, _), (nb, Rb, tb, _) in zip(frames, frames[1:]):
            if stage == 'reference':
                da, db = reference[na], reference[nb]
            else:
                pa, pb = folder / f'{na:08d}_grounded.npy', folder / f'{nb:08d}_grounded.npy'
                if not (pa.is_file() and pb.is_file()):
                    continue
                da, db = np.load(pa), np.load(pb)
            forward = warp_absrel(da, db, intrinsics[na], intrinsics[nb], Ra, ta, Rb, tb, masks[nb], a.occlusion)
            backward = warp_absrel(db, da, intrinsics[nb], intrinsics[na], Rb, tb, Ra, ta, masks[na], a.occlusion)
            errors.append((forward[0] + backward[0]) / 2)
            shares.append((forward[1] + backward[1]) / 2)
        if errors:
            report['stages'][stage] = {'tae_pct': float(np.nanmean(errors) * 100), 'pairs': len(errors),
                                       'tae_p90_pct': float(np.nanpercentile(errors, 90) * 100),
                                       'compared_pct': float(np.mean(shares) * 100)}
    print(json.dumps(report, indent=2))
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
