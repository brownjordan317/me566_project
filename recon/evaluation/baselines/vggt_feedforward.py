#!/usr/bin/env python3
"""Feed-forward baseline: VGGT's own cameras and depth, fused with our surface stages.

One VGGT-1B forward pass over the sequence (all frames, or --max-frames spread
evenly) predicts every camera (pose and intrinsics) and a depth map per frame
(Wang et al., "VGGT: Visual Geometry Grounded Transformer", CVPR 2025; weights
CC BY-NC 4.0, comparison only). Nothing from COLMAP, MoGe, DA3 or our fusion is
used. The predictions are written in the pipeline's own formats (a COLMAP text
model with one PINHOLE camera per frame, trajectory.csv, grounded depth maps and
object landmarks), rescaled so the median depth is --normalize-depth map units as
for our captures, and then go through the same scene TSDF, mask-vote object
extraction and cleanup as the default pipeline. Scoring places the mesh by
VGGT's camera centres, exactly as ours is placed by our own.

Image handling follows VGGT's load_and_preprocess_images(mode='pad') (vggt/utils/
load_fn.py) and its pose decoding (vggt/utils/pose_enc.py, extrinsics world-to-
camera in OpenCV convention, intrinsics in the padded 518-pixel frame), inverted
here back to the working frames.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation

from recon.common.progress import done, spinner
from recon.depth.multiview.vggt import unpad_depth

SIDE = 518


def padded_geometry(width: int, height: int) -> tuple[float, int, int]:
    """Scale and padding of VGGT's mode='pad' preprocessing (as in unpad_depth)."""
    if width >= height:
        resized_width, resized_height = SIDE, round(height * SIDE / width / 14) * 14
    else:
        resized_width, resized_height = round(width * SIDE / height / 14) * 14, SIDE
    return resized_width / width, (SIDE - resized_width) // 2, (SIDE - resized_height) // 2


def working_intrinsics(K: np.ndarray, width: int, height: int) -> np.ndarray:
    """VGGT's padded-frame intrinsics expressed in working-frame pixels."""
    scale, left, top = padded_geometry(width, height)
    K = K.astype(np.float64).copy()
    K[0, 2] -= left
    K[1, 2] -= top
    K[:2] /= scale
    return K


def write_model(output: Path, frames: list[int], Ks: list[np.ndarray], extrinsics: list[np.ndarray],
                width: int, height: int) -> None:
    """COLMAP text model, one PINHOLE camera per frame, and the pipeline's trajectory.csv."""
    (output / 'sparse_txt').mkdir(parents=True, exist_ok=True)
    (output / 'export').mkdir(parents=True, exist_ok=True)
    cameras, images, rows = [], [], ['frame,status,tx,ty,tz,qx,qy,qz,qw,matches,inliers,landmarks,processing_ms']
    for index, (frame, K, extrinsic) in enumerate(zip(frames, Ks, extrinsics), start=1):
        cameras.append(f'{index} PINHOLE {width} {height} {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}')
        rotation, translation = extrinsic[:3, :3], extrinsic[:3, 3]
        qx, qy, qz, qw = Rotation.from_matrix(rotation).as_quat()
        images += [f'{index} {qw} {qx} {qy} {qz} {translation[0]} {translation[1]} {translation[2]} '
                   f'{index} {frame:08d}.png', '0 0 -1']                  # no observations (parsers skip blank lines)
        center = -rotation.T @ translation
        cx, cy, cz, cw = Rotation.from_matrix(rotation.T).as_quat()
        rows.append(f'{frame},tracked,{center[0]},{center[1]},{center[2]},{cx},{cy},{cz},{cw},0,0,0,0')
    (output / 'sparse_txt/cameras.txt').write_text('\n'.join(cameras) + '\n')
    (output / 'sparse_txt/images.txt').write_text('\n'.join(images) + '\n')
    (output / 'sparse_txt/points3D.txt').write_text('')
    (output / 'export/trajectory.csv').write_text('\n'.join(rows) + '\n')


def landmarks(depths: dict, masks: dict, Ks: dict, extrinsics: dict, count: int = 200000) -> np.ndarray:
    """Object points back-projected from masked depth, for the extraction bounds."""
    points = []
    for frame, depth in depths.items():
        v, u = np.nonzero((depth > 0) & masks[frame])
        if not len(u):
            continue
        keep = np.random.default_rng(frame).choice(len(u), min(len(u), count // len(depths) + 1), replace=False)
        u, v, z = u[keep], v[keep], depth[v[keep], u[keep]]
        K, extrinsic = Ks[frame], extrinsics[frame]
        camera = np.stack(((u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z), 1)
        points.append((camera - extrinsic[:3, 3]) @ extrinsic[:3, :3])
    return np.concatenate(points)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('sequence', type=Path, help='Prepared sequence with frames/ and masks/')
    p.add_argument('--masks', type=Path, help='Mask folder (default: sequence/masks)')
    p.add_argument('--output', type=Path, required=True, help='Run folder (poses/, depth/, work/, mesh/)')
    p.add_argument('--max-frames', type=int, default=96, help='Frames in the single forward pass (24 GB GPU)')
    p.add_argument('--normalize-depth', type=float, default=6.5)
    p.add_argument('--checkpoint-dir', type=Path)
    a = p.parse_args()
    warnings.filterwarnings('ignore', category=FutureWarning, module='vggt')
    from huggingface_hub import snapshot_download
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    masks_dir = a.masks or a.sequence / 'masks'
    paths = sorted((a.sequence / 'frames').glob('*.png'))
    if len(paths) > a.max_frames:
        paths = [paths[i] for i in np.unique(np.linspace(0, len(paths) - 1, a.max_frames).round().astype(int))]
    frames = [int(path.stem) for path in paths]
    height, width = cv2.imread(str(paths[0])).shape[:2]
    checkpoint = a.checkpoint_dir or Path(snapshot_download('facebook/VGGT-1B', local_files_only=True))
    with spinner('Loading VGGT'), contextlib.redirect_stdout(io.StringIO()):
        network = VGGT.from_pretrained(str(checkpoint)).cuda().eval()
    network.track_head = None
    network.point_head = None
    images = load_and_preprocess_images([str(path) for path in paths], mode='pad').cuda()
    with spinner(f'VGGT forward pass over {len(paths)} frames'), torch.inference_mode(), \
            torch.autocast('cuda', dtype=torch.bfloat16):
        predicted = network(images)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predicted['pose_enc'], images.shape[-2:])
    depth_maps = predicted['depth'][0, ..., 0].float().cpu().numpy()
    extrinsic = extrinsic[0].float().cpu().numpy()
    intrinsic = intrinsic[0].float().cpu().numpy()
    del predicted, images, network
    torch.cuda.empty_cache()
    depths = {frame: unpad_depth(depth, (height, width)) for frame, depth in zip(frames, depth_maps)}
    masks = {frame: cv2.imread(str(masks_dir / f'{frame:08d}.png'), cv2.IMREAD_GRAYSCALE) > 127 for frame in frames}
    scale = a.normalize_depth / float(np.median(np.concatenate([d[d > 0] for d in depths.values()])))
    Ks = {frame: working_intrinsics(K, width, height) for frame, K in zip(frames, intrinsic)}
    poses = {}
    for frame, matrix in zip(frames, extrinsic):
        pose = np.eye(4)
        pose[:3, :4] = matrix
        pose[:3, 3] *= scale
        poses[frame] = pose
    poses_dir, depth_dir = a.output / 'poses', a.output / 'depth/vggt_feedforward'
    depth_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        depths[frame] = (depths[frame] * scale).astype(np.float32)
        np.save(depth_dir / f'{frame:08d}_grounded.npy', depths[frame])
    write_model(poses_dir, frames, [Ks[f] for f in frames], [poses[f] for f in frames], width, height)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(landmarks(depths, masks, Ks, poses)))
    (poses_dir / 'export_object').mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(poses_dir / 'export_object/landmarks.ply'), cloud)
    (a.output / 'summary_vggt.json').write_text(json.dumps(
        {'frames': len(frames), 'scale_to_map': scale, 'checkpoint': str(checkpoint)}, indent=2) + '\n')
    model, trajectory = poses_dir / 'sparse_txt', poses_dir / 'export/trajectory.csv'
    work, mesh = a.output / 'work', a.output / 'mesh'
    run = lambda *args: subprocess.run([sys.executable, '-m', *map(str, args)], check=True)
    run('recon.surface.tsdf', a.sequence, '--model', model, '--trajectory', trajectory, '--depth-dir', depth_dir,
        '--output', work / 'scene.ply', '--min-views', 3)
    run('recon.surface.object', work / 'scene.ply', '--bounds-cloud', poses_dir / 'export_object/landmarks.ply',
        '--masks-dir', masks_dir, '--depth-dir', depth_dir, '--trajectory', trajectory, '--model', model,
        '--output', work / 'object.ply', '--min-ratio', .8, '--min-component-support', .25)
    run('recon.surface.clean', work / 'object.ply', mesh / 'object.ply', '--hole-size', .08)
    done(f'VGGT feed-forward baseline: {len(frames)} frames -> {mesh / "object.ply"}')


if __name__ == '__main__':
    main()
