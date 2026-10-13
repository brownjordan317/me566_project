#!/usr/bin/env python3
"""Build COLMAP poses from full-image LightGlue matches.

Frames of a prepared sequence are matched against their sequential neighbours
(or every pair). Cameras are fixed from --per-frame-intrinsics when given;
otherwise one shared SIMPLE_PINHOLE focal length is estimated, optionally
refined per frame.
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import subprocess

import numpy as np
import torch

from recon.cameras.matching import extract_multiscale, sequential_pairs
from recon.cameras.sfm import (create_database, initialised_model, install_per_image_cameras, normalize_depth,
                               replace_features, rescale_text_model, run_mapper, split_cameras,
                               verify_matches)
from recon.common.camera import load_dataset_intrinsics
from recon.common.progress import bar, done, spinner


@contextlib.contextmanager
def gpu_lock(path: Path | None):
    """Hold an exclusive flock on path (the Makefile's GPU lock) for the duration, if given."""
    if path is None:
        yield
        return
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', type=Path, required=True,
                        help='Prepared sequence with frames/')
    parser.add_argument('--overlap', type=int, default=10,
                        help='Sequential neighbours matched per frame')
    parser.add_argument('--normalize-depth', type=float,
                        help='Rescale the model so the median scene depth is this many map units '
                             '(for captures of unknown scale)')
    parser.add_argument('--camera-model', choices=('simple_pinhole', 'pinhole'), default='simple_pinhole',
                        help='One focal length, or separate x and y focal lengths')
    parser.add_argument('--refine-principal-point', action='store_true',
                        help='Also estimate the principal point (otherwise fixed at the image centre)')
    parser.add_argument('--loop', action='store_true',
                        help='Also match the end of the sequence to its start (orbit captures)')
    parser.add_argument('--feature-frames', type=Path,
                        help='Higher-resolution copies of the frames (matched by frame number). '
                             'Poses are solved at that resolution and the model is rescaled '
                             'to the working frames')
    parser.add_argument('--per-frame-focal', action='store_true',
                        help='After solving with one estimated focal length, refine a separate '
                             'focal per frame. Opt-in for phone captures without locked focus '
                             '(lens drift ~1%%); it invents drift and hurts on fixed lenses')
    parser.add_argument('--scales', default='1',
                        help='Comma-separated detection scales relative to the feature frames, '
                             'e.g. 1,0.5 for fine plus coarse keypoints')
    parser.add_argument('--pairs', choices=('sequential', 'exhaustive'), default='sequential',
                        help='Video-order neighbours, or every pair for unordered photo sets')
    parser.add_argument('--per-frame-intrinsics', type=Path,
                        help='Dataset directory of NNNNNN.txt camera matrices (comparison only)')
    parser.add_argument('--calibration', type=Path,
                        help='Calibration JSON whose image_size the per-frame matrices use')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--colmap', type=Path, default=Path('colmap'))
    parser.add_argument('--max-keypoints', type=int, default=4096)
    parser.add_argument('--features', choices=('aliked', 'sift', 'superpoint'), default='superpoint',
                        help='Keypoints matched by LightGlue; ALIKED and SIFT localise to sub-pixel precision')
    parser.add_argument('--threads', type=int, default=os.cpu_count() or 8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--init-cameras', type=Path,
                        help='Text model of cameras to start from instead of incremental mapping')
    parser.add_argument('--init-rounds', type=int, default=6,
                        help='With --init-cameras: bundle-adjustment rounds (0 keeps the given cameras)')
    parser.add_argument('--gpu-lock', type=Path, help='Lock file held while the GPU is used (as make GPU_LOCK)')
    args = parser.parse_args()
    if args.per_frame_intrinsics is not None and args.calibration is None:
        parser.error('--per-frame-intrinsics needs --calibration')
    if args.max_keypoints < 256:
        parser.error('--max-keypoints must be at least 256')
    if (args.output / 'database.db').exists():
        parser.error('Output already has a COLMAP database; choose a fresh directory')
    from lightglue import ALIKED, LightGlue, SIFT, SuperPoint
    from lightglue.utils import load_image

    image_dir = args.sequence / 'frames'
    names = sorted(path.name for path in image_dir.glob('*.png'))
    if len(names) < 3:
        parser.error(f'Need at least three PNG frames in {image_dir}')
    pairs = (sorted((a, b) for i, a in enumerate(names) for b in names[i + 1:])
             if args.pairs == 'exhaustive' else sequential_pairs(names, args.overlap, loop=args.loop))
    args.output.mkdir(parents=True, exist_ok=True)
    # Feature extraction and matching use the GPU; take the pipeline's GPU lock for them only, so
    # COLMAP's long CPU mapping does not hold the GPU (other scenes' DA3 needs ~20 GB).
    with gpu_lock(args.gpu_lock):
        extractor = {'aliked': ALIKED, 'sift': SIFT, 'superpoint': SuperPoint}[args.features](
            max_num_keypoints=args.max_keypoints).eval().to(args.device)
        matcher = LightGlue(features=args.features).eval().to(args.device)
        sources = {}
        if args.feature_frames is not None:
            sources = {int(path.stem): path for path in args.feature_frames.iterdir()
                       if path.suffix.lower() in ('.png', '.jpg', '.jpeg') and path.stem.isdigit()}
        detect_scales = [float(value) for value in args.scales.split(',')]
        if not detect_scales or any(not 0 < value <= 1 for value in detect_scales):
            parser.error('--scales must be in (0, 1]')
        features = {}
        scales = {}
        with torch.inference_mode():
            for name in bar(names, desc=f'{args.features} keypoints', unit='img'):
                working = load_image(image_dir / name)
                image = working
                if sources:
                    source = sources.get(int(Path(name).stem))
                    if source is None:
                        parser.error(f'No feature frame for {name} in {args.feature_frames}')
                    image = load_image(source)
                scales[name] = image.shape[-1] / working.shape[-1]
                features[name] = extract_multiscale(extractor, image.to(args.device), detect_scales)
        raw_matches = args.output / 'lightglue_raw_matches.txt'
        pair_counts = []
        blocks = []
        with torch.inference_mode():
            for name0, name1 in bar(pairs, desc='LightGlue matching', unit='pair'):
                result = matcher({'image0': features[name0], 'image1': features[name1]})
                matches = result['matches'][0].detach().cpu().numpy()
                if matches.ndim != 2 or matches.shape[1] != 2:
                    raise ValueError(f'Invalid LightGlue matches for {name0}, {name1}')
                blocks.append(f'{name0} {name1}\n'
                              + ''.join(f'{a} {b}\n' for a, b in matches) + '\n')
                pair_counts.append(len(matches))
        raw_matches.write_text(''.join(blocks))
        arrays = {name: {key: value[0].detach().cpu().numpy()
                         for key, value in feature.items() if key in ('keypoints', 'descriptors')}
                  for name, feature in features.items()}
        del extractor, matcher, features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    feature_scale = scales[names[0]]
    if any(abs(value - feature_scale) > 1e-6 for value in scales.values()):
        parser.error('Feature frames must share one resolution')
    database = args.output / 'database.db'
    height, width = load_image(image_dir / names[0]).shape[-2:]
    # COLMAP works at the keypoint resolution; the model is rescaled afterwards.
    width, height = round(width * feature_scale), round(height * feature_scale)
    create_database(database, args.colmap, names, width, height, args.camera_model)
    if args.per_frame_intrinsics is not None:
        calibration_size = json.loads(args.calibration.read_text())['image_size']
        install_per_image_cameras(database, load_dataset_intrinsics(
            args.per_frame_intrinsics, [int(Path(name).stem) for name in names],
            (width, height), tuple(calibration_size)), width, height)
    replace_features(database, arrays)
    # Only an estimated camera is refined; supplied cameras stay fixed.
    refine_focal = '0' if args.per_frame_intrinsics is not None else '1'
    (args.output / 'sparse').mkdir()
    executable = str(args.colmap)
    env = os.environ.copy()
    with spinner(f'COLMAP geometric verification ({len(blocks)} pairs)'):
        verify_matches(database, blocks, args.output / 'verify_chunks', executable,
                       args.threads, env, args.output / 'verify.log')
    if args.init_cameras is not None:
        # Cameras from another source (e.g. VGGT-Omega, recon.cameras.feedforward) replace incremental mapping.
        with spinner(f'COLMAP from given cameras ({args.init_rounds} adjustment rounds)'):
            registered = initialised_model(database, args.init_cameras, feature_scale, args.output / 'sparse',
                                           executable, env, args.init_rounds, refine_focal == '1',
                                           args.output / 'initialised.log')
    else:
        run_mapper([executable, 'mapper', '--database_path', str(database),
                        '--image_path', str(image_dir), '--output_path', str(args.output / 'sparse'),
                        '--Mapper.ba_refine_focal_length', refine_focal,
                        '--Mapper.ba_refine_principal_point',
                        '1' if args.refine_principal_point and refine_focal == '1' else '0',
                        '--Mapper.ba_refine_extra_params', '0',
                        # Global BA after 40% growth instead of 10%: 29% less mapper
                        # time over the 18 MobileBrick scenes, same pose accuracy.
                        '--Mapper.ba_global_images_ratio', '1.4',
                        '--Mapper.ba_global_points_ratio', '1.4',
                        '--Mapper.num_threads', str(args.threads)],
                   args.output / 'mapper.log', len(names), env)
        models = sorted((p for p in (args.output / 'sparse').iterdir()
                         if (p / 'images.bin').is_file()),
                        key=lambda p: (p / 'images.bin').stat().st_size, reverse=True)
        if not models:
            raise RuntimeError('COLMAP registered no model; see mapper.log')
        registered = models[0]
    if args.per_frame_focal and refine_focal == '1' and not (args.init_cameras is not None and args.init_rounds == 0):
        shared = args.output / 'shared_focal_txt'
        shared.mkdir()
        subprocess.run([executable, 'model_converter', '--input_path', str(registered),
                        '--output_path', str(shared), '--output_type', 'TXT'],
                       check=True, env=env)
        split_cameras(shared)
        registered = args.output / 'sparse_per_frame_focal'
        registered.mkdir()
        with spinner('COLMAP per-frame focal refinement'), \
                (args.output / 'per_frame_focal.log').open('w') as log:
            subprocess.run([executable, 'bundle_adjuster', '--input_path', str(shared),
                            '--output_path', str(registered),
                            '--BundleAdjustment.refine_focal_length', '1',
                            '--BundleAdjustment.refine_principal_point',
                            '1' if args.refine_principal_point else '0',
                            '--BundleAdjustment.refine_extra_params', '0'],
                           check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
    text_model = args.output / 'sparse_txt'
    text_model.mkdir()
    subprocess.run([executable, 'model_converter', '--input_path', str(registered),
                    '--output_path', str(text_model), '--output_type', 'TXT'],
                   check=True, env=env)
    if feature_scale != 1:
        rescale_text_model(text_model, 1 / feature_scale)
    depth_scale = normalize_depth(text_model, args.normalize_depth) if args.normalize_depth else 1.
    with sqlite3.connect(database) as connection:
        verified = connection.execute(
            'SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0').fetchone()[0]
    summary = {'images': len(names), 'attempted_pairs': len(pairs),
               'verified_pairs': verified, 'median_raw_matches_per_pair': float(np.median(pair_counts)),
               'model': f'{args.features} + LightGlue', 'max_keypoints': args.max_keypoints,
               'registered_model': str(registered), 'depth_normalization_factor': depth_scale}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    done(f"Camera poses: {summary['images']} frames, {summary['verified_pairs']} verified pairs")


if __name__ == '__main__':
    main()
