#!/usr/bin/env python3
"""How well the fusion's photometric score judges its candidate depth models, and what a perfect one buys.

The fusion scores each candidate's depth by patch NCC against neighbouring views, pools that over the
same window it uses in production and takes the winner (recon.depth.fusion.filter). This measures that
judge against the reference: per block, does the higher score belong to the candidate actually closer to
the truth? With --write-oracle it also writes the depth a perfect judge would have produced, so the
ceiling can be carried through the surface stages and read at the mesh.

Measured on audi and convertible: the judge is right 69.1% and 64.8% of the time against 50% for a coin,
beats averaging the candidates by 6-8%, and a perfect judge would gain a further 13-17% of depth error -
worth about 2 points of F@1 once the plane sweep has had its turn (docs/PIPELINE.md). Set up exactly as
the fusion runs, so the judge is measured as it behaves rather than as reimplemented.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from recon.common.colmap import parse_model
from recon.common.progress import spinner
from recon.depth.fusion.photometric import Images, ncc_scores, pool
from recon.depth.fusion.views import neighbours
from recon.evaluation.diagnostics.oracle_depth import reference_in_map, render_depth


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path, help='Finished run with poses/ and depth/<candidate>/')
    p.add_argument('--sequence', type=Path, required=True, help='Prepared sequence (frames/, masks/)')
    p.add_argument('--reference', type=Path, required=True, help='Reference mesh')
    p.add_argument('--pose-dir', type=Path, required=True, help='Dataset poses, for placing the reference')
    p.add_argument('--candidates', default='vggt,da3', help='Candidate depth folders under depth/')
    p.add_argument('--window', type=int, default=17, help='Pooling window, as the fusion uses')
    p.add_argument('--stride', type=int, default=2)
    p.add_argument('--radius', type=int, default=4, help='Patch radius in full-resolution pixels')
    p.add_argument('--neighbours', type=int, default=6)
    p.add_argument('--angle-band', type=float, nargs=2, default=(10., 35.))
    p.add_argument('--every', type=int, default=2, help='Describe every n-th frame')
    p.add_argument('--write-oracle', type=Path, help='Also write the depth a perfect judge would produce')
    p.add_argument('--output', type=Path)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    names = a.candidates.split(',')
    intrinsics, _, frames = parse_model(a.run / 'poses/sparse_txt')
    numbers = [n for n, *_ in frames]
    T = lambda v: torch.as_tensor(v, dtype=torch.float32, device=a.device)
    K = {i: T(intrinsics[n]) for i, n in enumerate(numbers)}
    R = {i: T(f[1]) for i, f in enumerate(frames)}
    t = {i: T(f[2]) for i, f in enumerate(frames)}
    near = neighbours(np.stack([f[1] for f in frames]), np.stack([f[2] for f in frames]),
                      a.neighbours, tuple(a.angle_band))
    scene = reference_in_map(a.run / 'poses/export/trajectory.csv', a.pose_dir, a.reference,
                             a.run / 'mesh/object.ply')
    first = np.load(a.run / f'depth/{names[0]}/{numbers[0]:08d}_grounded.npy')
    images = Images(a.sequence / 'frames', numbers, first.shape[1], a.device)
    if a.write_oracle:
        a.write_oracle.mkdir(parents=True, exist_ok=True)
    right = wrong = 0
    judged, oracle, averaged, per_model = [], [], [], []
    with torch.inference_mode(), spinner('Judging the depth candidates'):
        for index, number in enumerate(numbers):
            if index % a.every:
                continue
            maps = []
            for name in names:
                path = a.run / f'depth/{name}/{number:08d}_grounded.npy'
                if not path.is_file():
                    maps = []
                    break
                maps.append(np.load(path).astype('float32'))
            if not maps:
                continue
            mask = cv2.imread(str(a.sequence / 'masks' / f'{number:08d}.png'), cv2.IMREAD_GRAYSCALE) > 127
            truth = render_depth(scene, intrinsics[number], frames[index][1].T,
                                 -frames[index][1].T @ frames[index][2], *maps[0].shape)
            valid = mask & (truth > 0) & np.all([m > 0 for m in maps], 0)
            if valid.sum() < 500:
                continue
            grid = np.zeros_like(valid)
            grid[::a.stride, ::a.stride] = True
            pooled, errors = [], []
            for depth in maps:
                sampled = valid & grid
                v, u = np.nonzero(sampled)
                score = ncc_scores(images, index, [int(j) for j in near[index]], K, R, t, T(depth[v, u]),
                                   T(u.astype('float32')), T(v.astype('float32')), a.radius)
                values = np.zeros_like(depth)
                values[v, u] = score.cpu().numpy()
                pooled.append(pool(values, sampled, a.window))
                relative = np.abs(depth - truth) / np.maximum(truth, 1e-6)
                errors.append(pool(np.where(valid, relative, 0).astype(np.float32), valid, a.window))
            pooled, errors = np.stack(pooled), np.stack(errors)
            good = valid & np.all(np.isfinite(pooled), 0) & np.all(np.isfinite(errors), 0)
            if good.sum() < 200:
                continue
            picked, truly = pooled.argmax(0)[good], errors.argmin(0)[good]
            right += int((picked == truly).sum())
            wrong += int((picked != truly).sum())
            e = errors[:, good]
            judged.append(float(np.take_along_axis(e, picked[None], 0).mean()))
            oracle.append(float(e.min(0).mean()))
            averaged.append(float(e.mean(0).mean()))
            per_model.append([float(e[c].mean()) for c in range(len(names))])
            if a.write_oracle:
                choice = np.where(good, errors.argmin(0), 0)
                chosen = np.take_along_axis(np.stack(maps), choice[None], 0)[0]
                np.save(a.write_oracle / f'{number:08d}_grounded.npy',
                        np.where(mask & good, chosen, 0).astype(np.float32))
    total = right + wrong
    report = {'run': str(a.run), 'candidates': names, 'blocks': total,
              'judge_accuracy_pct': float(right / max(1, total) * 100),
              'relative_error': {'photometric_judge': float(np.mean(judged)) if judged else None,
                                 'oracle': float(np.mean(oracle)) if oracle else None,
                                 'averaged': float(np.mean(averaged)) if averaged else None,
                                 **{f'always_{n}': float(np.mean([m[i] for m in per_model]))
                                    for i, n in enumerate(names) if per_model}}}
    if a.write_oracle:
        (a.write_oracle / 'hybrid_depth.json').write_text(
            json.dumps({'method': f'oracle block-level judge over {names}'}, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
