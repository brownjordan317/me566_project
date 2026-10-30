#!/usr/bin/env python3
"""Write the learned judge's training data for one finished run.

For every --every-th frame: the evidence of each depth candidate (recon.depth.judge.evidence), the
candidates' depth and the true depth on the evidence grid. The true depth is the reference mesh placed
in our map exactly as the shape metrics place it (camera similarity refined by scaled ICP onto our mesh,
recon.evaluation.diagnostics.oracle_depth) and rendered with our cameras, so the candidates are judged
in the cameras the judge will meet at scan time, errors and all.

Candidates are real (--candidate NAME DIR, grounded depth from any source) and --synthetic damaged
copies of the true depth (recon.depth.judge.corrupt), so the judge also learns failures no current source
shows. Output: <output>/<frame>.npz and meta.json (candidate names and kinds, feature names, mm per map
unit), and <frame>_self.npy, the candidates' self-reports (recon.depth.judge.self_reports). Use only
training scenes; the dev, test, DTU and Tanks and Temples scenes stay out of training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch

from recon.common.colmap import parse_model
from recon.common.progress import bar, done
from recon.depth.fusion.photometric import Images
from recon.depth.fusion.views import neighbours
from recon.depth.judge.corrupt import character, damage
from recon.depth.judge.evidence import (IMAGE_FEATURES, OWN_FEATURES, PAIR_FEATURES, Capture, candidate_evidence,
                                        grid, image_evidence)
from recon.depth.judge.self_reports import write_self_reports
from recon.evaluation.diagnostics.oracle_depth import reference_mesh_in_map, render_depth
from recon.evaluation.metrics.mesh import similarity


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path, help='Finished run (poses/sparse_txt, poses/export/trajectory.csv, mesh/object.ply)')
    p.add_argument('--sequence', type=Path, required=True, help='Prepared sequence (frames/, masks/)')
    p.add_argument('--images', type=Path, required=True, help='Full-resolution photos, for the patch evidence')
    p.add_argument('--reference', type=Path, required=True, help='Reference mesh')
    p.add_argument('--pose-dir', type=Path, required=True, help='Reference poses, for placing the mesh')
    p.add_argument('--candidate', nargs=2, action='append', default=[], metavar=('NAME', 'DIR'),
                   help='Grounded depth folder (<frame>_grounded.npy) of one source; repeat for each')
    p.add_argument('--synthetic', type=int, default=3, help='Damaged copies of the true depth to add')
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--every', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    intrinsics, _, frames = parse_model(a.run / 'poses/sparse_txt')
    numbers = [number for number, *_ in frames]
    K = {i: intrinsics[n] for i, n in enumerate(numbers)}
    R = {i: f[1] for i, f in enumerate(frames)}
    t = {i: f[2] for i, f in enumerate(frames)}
    trajectory = a.run / 'poses/export/trajectory.csv'
    mm_per_unit = 1000. * similarity(trajectory, a.pose_dir)[3]
    mesh = reference_mesh_in_map(trajectory, a.pose_dir, a.reference, a.run / 'mesh/object.ply')
    extent = float(np.linalg.norm(mesh.get_max_bound() - mesh.get_min_bound()))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    photos = {i: cv2.imread(str(a.sequence / 'frames' / f'{n:08d}.png')) for i, n in enumerate(numbers)}
    masks = {i: cv2.imread(str(a.sequence / 'masks' / f'{n:08d}.png'), cv2.IMREAD_GRAYSCALE) > 127
             for i, n in enumerate(numbers)}
    height, width = masks[0].shape
    truth = {i: render_depth(scene, K[i], R[i].T, -R[i].T @ t[i], height, width) for i in range(len(numbers))}
    candidates, kinds = {}, {}
    for name, folder in a.candidate:
        maps = {i: np.load(Path(folder) / f'{n:08d}_grounded.npy').astype(np.float32)
                for i, n in enumerate(numbers) if (Path(folder) / f'{n:08d}_grounded.npy').is_file()}
        if maps:
            candidates[name], kinds[name] = maps, 'real'
    traits = {}
    for k in range(a.synthetic):
        name = f'synthetic_{k}'
        traits[name] = character(rng)
        candidates[name] = {i: damage(truth[i], photos[i], K[i], R[i].T, -R[i].T @ t[i], extent, traits[name], rng)
                            for i in range(len(numbers))}
        kinds[name] = 'synthetic'
    images = Images(a.images, numbers, width, a.device)
    near = neighbours(np.stack([R[i] for i in range(len(numbers))]), np.stack([t[i] for i in range(len(numbers))]),
                      6, (10., 35.))
    capture = Capture(numbers, K, R, t, images, masks, candidates, a.device)
    a.output.mkdir(parents=True, exist_ok=True)
    rows, cols = grid(height, width, a.stride)
    written = []
    with torch.inference_mode():
        for index in bar(range(0, len(numbers), a.every), desc='Judge evidence', unit='frame'):
            own, pairs, depth, region = candidate_evidence(capture, index, near[index], a.stride)
            target = np.where(region, truth[index][np.ix_(rows, cols)], 0).astype(np.float32)
            if (target > 0).sum() < 100:
                continue
            np.savez_compressed(a.output / f'{numbers[index]:08d}.npz', own=own.astype(np.float16),
                                pairs=pairs.astype(np.float16), depth=depth, truth=target, region=region,
                                image=image_evidence(photos[index], masks[index], rows, cols).astype(np.float16))
            written.append(numbers[index])
    meta = {'run': str(a.run), 'names': list(candidates), 'kinds': kinds, 'mm_per_unit': mm_per_unit,
            'stride': a.stride, 'own_features': OWN_FEATURES, 'pair_features': PAIR_FEATURES,
            'image_features': IMAGE_FEATURES, 'frames': written,
            'synthetic': {name: {'damages': value['damages'], 'bias': value['bias'],
                                 'drift_scale': value['drift_scale']} for name, value in traits.items()}}
    (a.output / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    write_self_reports(a.output, {name: Path(folder) for name, folder in a.candidate}, height, width)
    done(f'Judge data: {len(written)} frames x {len(candidates)} candidates -> {a.output}')


if __name__ == '__main__':
    main()
