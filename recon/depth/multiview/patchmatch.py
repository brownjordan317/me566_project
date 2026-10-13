#!/usr/bin/env python3
"""Multi-view stereo depth from COLMAP's PatchMatch, plain or seeded with our depth.

COLMAP's PatchMatch stereo (Schönberger et al., "Pixelwise View Selection for
Unstructured Multi-View Stereo", ECCV 2016; BSD-3-Clause) runs on our own
registered model and frames, so its depth is already in the map's units and
needs no anchor fit.

Plain (default): the photometric pass starts every pixel from a random depth,
then the geometric pass makes each frame's depth agree with its neighbours'.
Its output is a fusion candidate (recon.depth.fusion.filter, PATCHMATCH=1).

Seeded (--seed-dir, Makefile GEOMETRIC=1): our depth and normals computed from it
are written in place of the photometric pass's output, which COLMAP then skips
(it reuses existing maps), and only the geometric pass runs, here for one
iteration. Pixels it leaves empty keep the seed depth. On jeep, seeded with our
refined depth: median error 0.95 to 0.65 mm, p90 3.56 to 2.35 mm, in about 30 s,
against 0.63 / 3.79 mm in 600 s for the plain default.

Dense stereo needs a CUDA build of COLMAP, which our COLMAP 3.9.1 and the main
environment's pycolmap are not. This module runs in its own environment
(pycolmap-cuda12, `make setup-patchmatch`) and imports only numpy, OpenCV and
pycolmap.

--images runs PatchMatch on the original photos: the cameras are rescaled to
them, seeds are upsampled (nearest) with normals computed at that size, and the
result is brought back to the working grid by averaging the valid pixels each
working pixel covers.
Reading and writing COLMAP's .bin dense maps follows `read_array` / `write_array`
in COLMAP's `scripts/python/read_write_dense.py` (BSD-3-Clause).

Output: <frame>_grounded.npy (0 = no depth); when seeded, <frame>_confirmed.npy
(pixels the geometric pass kept) and hybrid_depth.json, otherwise candidate.json.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

import cv2
import numpy as np


def read_array(path: Path) -> np.ndarray:
    """A COLMAP dense map: 'width&height&channels&' header, then float32 channel-major data."""
    with path.open('rb') as stream:
        header = b''
        while header.count(b'&') < 3:
            header += stream.read(1)
        width, height, channels = map(int, header.decode().split('&')[:3])
        data = np.fromfile(stream, np.float32)
    return data.reshape(channels, height, width).transpose(1, 2, 0).squeeze()


def write_array(path: Path, array: np.ndarray) -> None:
    array = array.astype(np.float32)
    if array.ndim == 2:
        array = array[..., None]
    height, width, channels = array.shape
    with path.open('wb') as stream:
        stream.write(f'{width}&{height}&{channels}&'.encode())
        array.transpose(2, 0, 1).tofile(stream)


def depth_normals(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Camera-frame unit normals from central differences of the back-projected depth, facing the camera."""
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    points = np.stack(((u - K[0, 2]) * depth / K[0, 0], (v - K[1, 2]) * depth / K[1, 1], depth), -1)
    du, dv = np.zeros_like(points), np.zeros_like(points)
    du[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dv[1:-1] = points[2:] - points[:-2]
    normals = np.cross(du, dv)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12
    normals = np.where((normals * points).sum(-1, keepdims=True) > 0, -normals, normals)
    normals[(depth <= 0) | (np.linalg.norm(normals, axis=-1) < .5)] = (0, 0, -1)
    return normals


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('sequence', type=Path, help='Prepared sequence (frames/)')
    p.add_argument('--colmap-model', type=Path, required=True, help='Registered model (sparse_txt)')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed-dir', type=Path, help='Depth (<frame>_grounded.npy) to start from; runs only the '
                                                 'geometric pass and keeps the seed where it finds nothing')
    p.add_argument('--workspace', type=Path, help='Undistorted images and raw maps (default: a temporary '
                                                  'folder in OUTPUT, removed afterwards)')
    p.add_argument('--images', type=Path,
                   help='Original photos (numbered files) to run on instead of the working frames')
    p.add_argument('--sources', type=int, default=8, help='Source views per frame (COLMAP picks them)')
    p.add_argument('--iterations', type=int, help='PatchMatch iterations (default 1 seeded, 5 plain)')
    p.add_argument('--window-step', type=int, default=2)
    p.add_argument('--window-radius', type=int, default=5, help='Matching window radius in input pixels')
    a = p.parse_args()
    import pycolmap
    if not pycolmap.has_cuda:
        p.error('This pycolmap has no CUDA; run with the pycolmap-cuda12 environment (make setup-patchmatch)')
    seeded = a.seed_dir is not None
    iterations = a.iterations or (1 if seeded else 5)
    workspace = a.workspace or a.output_dir / 'workspace'
    shutil.rmtree(workspace, ignore_errors=True)
    (workspace / 'sparse').mkdir(parents=True)
    model = pycolmap.Reconstruction(str(a.colmap_model))
    stems = {image.name: image.name.split('.')[0] for image in model.images.values()}   # frame numbers
    working = {camera_id: (camera.width, camera.height) for camera_id, camera in model.cameras.items()}
    image_dir = a.sequence / 'frames'
    if a.images is not None:
        files = {int(path.stem): path for path in a.images.iterdir() if path.stem.isdigit()}
        height, width = cv2.imread(str(next(iter(files.values())))).shape[:2]
        for camera in model.cameras.values():
            camera.rescale(width, height)
        for image in model.images.values():
            name = files[int(stems[image.name])].name
            stems[name] = stems.pop(image.name)
            image.name = name
        image_dir = a.images
    model.write(str(workspace / 'sparse'))
    started = time.time()
    dense = workspace / 'dense'
    pycolmap.undistort_images(str(dense), str(workspace / 'sparse'), str(image_dir))
    config = dense / 'stereo/patch-match.cfg'
    config.write_text(re.sub(r'__auto__, \d+', f'__auto__, {a.sources}', config.read_text()))
    seeds = {}
    if seeded:
        # SIMPLE_PINHOLE cameras: undistortion leaves frames and intrinsics unchanged.
        for image in model.images.values():
            seed = a.seed_dir / f'{stems[image.name]}_grounded.npy'
            if not seed.is_file():
                continue   # an earlier stage rejected this frame; COLMAP runs its photometric pass instead
            depth = np.load(seed).astype(np.float32)
            depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0)
            seeds[image.name] = depth
            camera = model.cameras[image.camera_id]
            K = camera.calibration_matrix()
            if depth.shape != (camera.height, camera.width):
                depth = cv2.resize(depth, (camera.width, camera.height), interpolation=cv2.INTER_NEAREST)
            write_array(dense / 'stereo/depth_maps' / f'{image.name}.photometric.bin', depth)
            write_array(dense / 'stereo/normal_maps' / f'{image.name}.photometric.bin', depth_normals(depth, K))
    options = pycolmap.PatchMatchOptions()
    options.geom_consistency = True
    options.num_iterations = iterations
    options.window_step = a.window_step
    options.window_radius = a.window_radius
    pycolmap.patch_match_stereo(str(dense), options=options)
    seconds = time.time() - started
    a.output_dir.mkdir(parents=True, exist_ok=True)
    coverage = []
    by_name = {image.name: image.camera_id for image in model.images.values()}
    for path in sorted((dense / 'stereo/depth_maps').glob('*.geometric.bin')):
        name = path.name[:-len('.geometric.bin')]
        depth = read_array(path)
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
        size = working[by_name[name]]
        if depth.shape[::-1] != size:
            # Back to the working grid: mean of the valid pixels each working pixel covers.
            valid = (depth > 0).astype(np.float32)
            total = cv2.resize(depth, size, interpolation=cv2.INTER_AREA)
            share = cv2.resize(valid, size, interpolation=cv2.INTER_AREA)
            depth = np.where(share > .5, total / np.maximum(share, 1e-6), 0).astype(np.float32)
        coverage.append(float((depth > 0).mean()))
        stem = stems[name]
        if seeded and name in seeds:
            # Which pixels the geometric pass confirmed (the rest keep the seed); a fusion confidence.
            np.save(a.output_dir / f'{stem}_confirmed.npy', depth > 0)
            depth = np.where(depth > 0, depth, seeds[name])
        np.save(a.output_dir / f'{stem}_grounded.npy', depth)
    if a.workspace is None:
        shutil.rmtree(workspace)
    summary = {'model': f'COLMAP PatchMatch stereo (pycolmap {pycolmap.__version__})',
               'seed': str(a.seed_dir) if seeded else None, 'frames': len(coverage), 'sources': a.sources,
               'iterations': iterations, 'window_step': a.window_step, 'window_radius': a.window_radius, 'seconds': seconds,
               'images': str(a.images) if a.images else 'working frames',
               'median_coverage': float(np.median(coverage))}
    (a.output_dir / ('hybrid_depth.json' if seeded else 'candidate.json')).write_text(json.dumps(summary, indent=2) + '\n')
    print(f"PatchMatch depth: {len(coverage)} frames in {seconds:.0f} s, median coverage {summary['median_coverage']:.0%}")


if __name__ == '__main__':
    main()
