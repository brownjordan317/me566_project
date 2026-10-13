#!/usr/bin/env python3
"""How close is the depth fusion to the best possible choice between its candidates? (diagnostic)

Needs a run kept with KEEP_INTERMEDIATE=1 (depth/vggt, depth/da3, depth/fused,
depth/fused_refined). The reference mesh, placed in our map by the shape
alignment (camera similarity refined by scaled ICP onto our mesh), is rendered
into every frame; on object pixels where it and both candidates have depth, the
absolute error of each depth is measured in millimetres:

  vggt, da3           the two candidates;
  fused               our photometric per-region choice between them;
  refined             fused after the plane-sweep refinement;
  oracle block        the better candidate per --block x --block pixel tile, chosen with the truth;
  oracle pixel        the better candidate per pixel, chosen with the truth.

fused vs oracle block is what a better selector could still gain; oracle block
(or pixel) vs zero is what only better candidates can gain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from recon.common.camera import frame_intrinsics
from recon.common.progress import bar, done
from recon.common.trajectory import load_poses
from recon.evaluation.diagnostics.oracle_depth import reference_in_map, render_depth
from recon.evaluation.metrics.mesh import similarity

ARMS = ('vggt', 'da3', 'fused', 'refined', 'oracle block', 'oracle pixel')


def block_choice(error_a: np.ndarray, error_b: np.ndarray, valid: np.ndarray, block: int) -> np.ndarray:
    """True where candidate a has the lower mean error over its block (blocks on a fixed grid)."""
    height, width = valid.shape
    rows, cols = -(-height // block), -(-width // block)
    pad = ((0, rows * block - height), (0, cols * block - width))
    tiles = lambda x: np.pad(x, pad).reshape(rows, block, cols, block).sum((1, 3))
    count = np.maximum(tiles(valid.astype(np.float64)), 1)
    a = tiles(np.where(valid, error_a, 0)) / count
    b = tiles(np.where(valid, error_b, 0)) / count
    return np.repeat(np.repeat(a <= b, block, 0), block, 1)[:height, :width]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--sequence', type=Path, required=True, help='Prepared sequence (masks/)')
    p.add_argument('--pose-dir', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--block', type=int, default=17, help='Tile size, as the fusion filter pools over 17x17')
    p.add_argument('--extra', nargs=2, action='append', default=[], metavar=('NAME', 'DIR'),
                   help='Also score another depth folder (e.g. a third model) on the same pixels')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    trajectory = a.run / 'poses/export/trajectory.csv'
    scale_mm = 1000. * similarity(trajectory, a.pose_dir)[3]                  # map units -> mm
    scene = reference_in_map(trajectory, a.pose_dir, a.reference, a.run / 'mesh/object.ply')
    poses, intrinsics = load_poses(trajectory), frame_intrinsics(a.run / 'poses/sparse_txt')
    depth = a.run / 'depth'
    errors = {arm: [] for arm in ARMS + tuple(name for name, _ in a.extra)}
    for frame in bar(sorted(poses), desc='Fusion ceiling', unit='frame'):
        files = {name: depth / name / f'{frame:08d}_grounded.npy' for name in ('vggt', 'da3', 'fused', 'fused_refined')}
        files.update({name: Path(folder) / f'{frame:08d}_grounded.npy' for name, folder in a.extra})
        mask = cv2.imread(str(a.sequence / 'masks' / f'{frame:08d}.png'), cv2.IMREAD_GRAYSCALE)
        if mask is None or not all(path.is_file() for path in files.values()) or frame not in intrinsics:
            continue
        maps = {name: np.load(path) for name, path in files.items()}
        rotation, center = poses[frame]
        truth = render_depth(scene, intrinsics[frame], rotation, center, *mask.shape)
        valid = (mask > 127) & (truth > 0) & (maps['vggt'] > 0) & (maps['da3'] > 0) & (maps['fused'] > 0)
        if not valid.any():
            continue
        err = {name: np.abs(value - truth) * scale_mm for name, value in maps.items()}
        pick_vggt = block_choice(err['vggt'], err['da3'], valid, a.block)
        oracle_block = np.where(pick_vggt, err['vggt'], err['da3'])
        oracle_pixel = np.minimum(err['vggt'], err['da3'])
        for arm, value in (('vggt', err['vggt']), ('da3', err['da3']), ('fused', err['fused']),
                           ('refined', err['fused_refined']), ('oracle block', oracle_block),
                           ('oracle pixel', oracle_pixel)):
            errors[arm].append(value[valid & (maps['fused_refined'] > 0 if arm == 'refined' else True)])
        for name, _ in a.extra:
            errors[name].append(err[name][valid & (maps[name] > 0)])
    report = {'run': str(a.run), 'block': a.block}
    for arm in errors:
        values = np.concatenate(errors[arm])
        report[arm] = {'median_mm': float(np.median(values)), 'p90_mm': float(np.quantile(values, .9)),
                       'within_1mm_pct': float(np.mean(values < 1) * 100), 'pixels': int(len(values))}
    text = json.dumps(report, indent=2) + '\n'
    if a.output:
        a.output.write_text(text)
    print(text, end='')
    done(f"Fusion ceiling: fused {report['fused']['median_mm']:.2f} mm, oracle block {report['oracle block']['median_mm']:.2f} mm")


if __name__ == '__main__':
    main()
