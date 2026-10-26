#!/usr/bin/env python3
"""Judge a reconstruction from frames it never saw (no ground truth).

split   copy a finished camera solve to a new run without every --every-th frame
        (starting at --offset) and re-export it, so the depth and surface stages
        build the object from the remaining frames only. The cameras come from the
        full solve; only the reconstruction is held out.
score   render the object mesh of one or more such runs into the held-out frames
        and measure, per held-out pixel inside its mask, how well the held-out
        image agrees with its nearest training views at the rendered depth (the
        fusion's patch NCC, recon.depth.fusion.photometric). Mask pixels the mesh
        misses score -1. Stages that optimise photometric agreement only ever saw
        the training frames, so they cannot inflate this score, which is the
        problem with judging a reconstruction from the images it was built from.

Held-out validation views are the usual check for image-based reconstruction
(e.g. novel-view evaluation in NeRF, Mildenhall et al., ECCV 2020); the score
here is ours.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from recon.common.colmap import parse_model
from recon.common.trajectory import load_poses


def held_out(numbers: list[int], every: int, offset: int) -> list[int]:
    return [number for index, number in enumerate(sorted(numbers)) if index % every == offset % every]


def kept_arc(numbers: list[int], low: int, high: int) -> list[int]:
    """Hold out everything outside the frame range [low, high].

    A contiguous arc, unlike thinning or an evenly spread handful: the surface the dropped views saw is
    then genuinely unobserved, which is what a test of inferred colour or geometry needs. Interleaved
    held-out views see almost the same surface as the kept ones and test nothing.
    """
    return [number for number in sorted(numbers) if not low <= number <= high]


def kept_few(numbers: list[int], keep: int) -> list[int]:
    """Hold out everything except `keep` frames spread evenly over the sequence.

    The inverse of `held_out`: that thins a capture, this reduces it to a handful of views, which is the
    regime a multi-camera rig works in - a few simultaneous views rather than a dense orbit.
    """
    order = sorted(numbers)
    if keep >= len(order):
        return []
    chosen = {order[index] for index in np.unique(np.linspace(0, len(order) - 1, keep).round().astype(int))}
    return [number for number in order if number not in chosen]


def split(a: argparse.Namespace) -> None:
    source, target = a.poses, a.run / 'poses'
    _, _, frames = parse_model(source / 'sparse_txt')
    numbers = [number for number, *_ in frames]
    keep, arc = getattr(a, 'keep', 0), getattr(a, 'arc', None)
    if arc:
        hold = set(kept_arc(numbers, *arc))
    else:
        hold = set(kept_few(numbers, keep) if keep else held_out(numbers, a.every, a.offset))
    if target.exists():
        shutil.rmtree(target)
    (target / 'sparse_txt').mkdir(parents=True)
    shutil.copy(source / 'sparse_txt/cameras.txt', target / 'sparse_txt/cameras.txt')
    lines = [line for line in (source / 'sparse_txt/images.txt').read_text().splitlines() if not line.startswith('#')]
    kept, dropped = [], set()
    for header, observations in zip(lines[::2], lines[1::2]):
        if int(Path(header.split()[9]).stem) in hold:
            dropped.add(header.split()[0])
        else:
            kept += [header, observations]
    (target / 'sparse_txt/images.txt').write_text('\n'.join(kept) + '\n')
    # Tracks must not name removed images (pycolmap rejects the model otherwise).
    points = []
    for line in (source / 'sparse_txt/points3D.txt').read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        fields = line.split()
        track = [pair for pair in zip(fields[8::2], fields[9::2]) if pair[0] not in dropped]
        points.append(' '.join(fields[:8] + [value for pair in track for value in pair]))
    (target / 'sparse_txt/points3D.txt').write_text('\n'.join(points) + '\n')
    subprocess.run([sys.executable, '-m', 'recon.cameras.export', str(target / 'sparse_txt'), str(target / 'export')],
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, '-m', 'recon.cameras.export', str(target / 'sparse_txt'),
                    str(target / 'export_object'), '--sequence', str(a.sequence)], check=True, stdout=subprocess.DEVNULL)
    summary = json.loads((source / 'summary.json').read_text())
    summary.update({'held_out': sorted(hold), 'held_out_from': str(source)})
    (target / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Held out {len(hold)} of {len(frames)} frames: {sorted(hold)}')


def score(a: argparse.Namespace) -> None:
    import open3d as o3d
    from recon.depth.fusion.photometric import Images, ncc_scores
    from recon.depth.fusion.views import neighbours
    from recon.evaluation.diagnostics.oracle_depth import render_depth
    intrinsics, _, frames = parse_model(a.poses / 'sparse_txt')
    numbers = [number for number, *_ in frames]
    hold = held_out(numbers, a.every, a.offset)
    poses = load_poses(a.poses / 'export/trajectory.csv')
    T = lambda value: torch.as_tensor(value, dtype=torch.float32, device='cuda')
    K = {i: T(intrinsics[n]) for i, n in enumerate(numbers)}
    R = {i: T(frame[1]) for i, frame in enumerate(frames)}
    t = {i: T(frame[2]) for i, frame in enumerate(frames)}
    rotations, translations = np.stack([f[1] for f in frames]), np.stack([f[2] for f in frames])
    near = neighbours(rotations, translations, len(frames) - 1, tuple(a.angle_band))
    train = {i for i, n in enumerate(numbers) if n not in hold}
    masks = {n: cv2.imread(str(a.sequence / 'masks' / f'{n:08d}.png'), cv2.IMREAD_GRAYSCALE) > 127 for n in hold}
    height, width = next(iter(masks.values())).shape
    images = Images(a.images, numbers, width, 'cuda')
    results = []
    for run in a.runs:
        mesh = o3d.io.read_triangle_mesh(str(run / 'mesh/object.ply'))
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        values, covered, iou = [], [], []
        for number in hold:
            index = numbers.index(number)
            depth = render_depth(scene, intrinsics[number], *poses[number], height, width)
            mask, hit = masks[number], depth > 0
            grid = np.zeros_like(mask)
            grid[::a.stride, ::a.stride] = True
            v, u = np.nonzero(mask & grid)
            z = depth[v, u]
            score_pixels = np.full(len(z), -1.)
            seen = z > 0
            others = [j for j in near[index] if j in train][:a.neighbours]
            with torch.inference_mode():
                score_pixels[seen] = ncc_scores(images, index, others, K, R, t, T(z[seen]), T(u[seen]), T(v[seen]),
                                                a.radius).cpu().numpy()
            values.append(score_pixels)
            covered.append(seen.mean())
            iou.append((hit & mask).sum() / max((hit | mask).sum(), 1))
        value = np.concatenate(values)
        value = np.where(np.isfinite(value), value, -1.)
        row = {'run': str(run), 'score': float(value.mean()), 'score_on_surface': float(value[value > -1].mean()),
               'mask_coverage': float(np.mean(covered)), 'silhouette_iou': float(np.mean(iou)), 'held_out': hold}
        results.append(row)
        print(f"{run}: held-out score {row['score']:.4f} (on surface {row['score_on_surface']:.4f}, "
              f"mask coverage {row['mask_coverage']:.3f}, silhouette IoU {row['silhouette_iou']:.3f})")
    if a.output:
        a.output.write_text(json.dumps(results, indent=2) + '\n')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('split', 'score'):
        q = sub.add_parser(name)
        q.add_argument('--poses', type=Path, required=True, help='The full camera solve (poses/ of a finished run)')
        q.add_argument('--sequence', type=Path, required=True, help='Prepared sequence (masks/)')
        q.add_argument('--every', type=int, default=8)
        q.add_argument('--offset', type=int, default=4)
        q.add_argument('--arc', type=int, nargs=2, metavar=('LOW', 'HIGH'),
                       help='Keep only frames in this inclusive range, holding out the rest')
        q.add_argument('--keep', type=int, default=0,
                       help='Instead of thinning, keep only this many evenly spread frames (0: thin)')
    s = sub.choices['split']
    s.add_argument('--run', type=Path, required=True, help='New run folder; poses/ is written there')
    s = sub.choices['score']
    s.add_argument('--images', type=Path, required=True, help='Full-resolution frames')
    s.add_argument('runs', type=Path, nargs='+', help='Runs built from a split (mesh/object.ply)')
    s.add_argument('--neighbours', type=int, default=6)
    s.add_argument('--angle-band', type=float, nargs=2, default=(10., 35.))
    s.add_argument('--radius', type=int, default=4)
    s.add_argument('--stride', type=int, default=4)
    s.add_argument('--output', type=Path)
    a = p.parse_args()
    split(a) if a.command == 'split' else score(a)


if __name__ == '__main__':
    main()
