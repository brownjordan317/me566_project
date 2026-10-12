#!/usr/bin/env python3
"""Prepare an ordered image folder or a video for mask-free full-scene reconstruction."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np

from recon.common.progress import bar


def image_order(path: Path) -> tuple:
    parts = re.split(r'(\d+)', path.stem.lower())
    return tuple((1, int(part)) if part.isdigit() else (0, part)
                 for part in parts)


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}
# PNG is lossless at every level; level 1 only trades file size for speed.
PNG_FAST = [cv2.IMWRITE_PNG_COMPRESSION, 1]
VIDEO_SUFFIXES = {'.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm'}


def is_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES


def video_step(video: Path, sample_fps: float) -> int:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f'Cannot open video: {video}')
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return max(1, round(fps / sample_fps)) if fps > 0 else 1


def video_frames(video: Path, step: int):
    """Yield the sharpest frame of each run of step frames to avoid motion blur."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f'Cannot open video: {video}')
    best = None
    number = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        if best is None or sharpness > best[0]:
            best = (sharpness, number, frame)
        number += 1
        if number % step == 0:
            yield f'{video.name}@{best[1]}', best[2]
            best = None
    capture.release()
    if best is not None:
        yield f'{video.name}@{best[1]}', best[2]


def read_image(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f'Cannot read image: {path}')
    return frame


def write_frame(source, path: Path, max_edge: int, full_path: Path | None = None) -> tuple[int, int]:
    """Resize and save one frame (and, with full_path, an unresized copy); source is an array or path."""
    frame = read_image(source) if isinstance(source, Path) else source
    if full_path is not None and not cv2.imwrite(str(full_path), frame, PNG_FAST):
        raise OSError(full_path)
    height, width = frame.shape[:2]
    factor = min(1., max_edge / max(width, height))
    target = (max(1, round(width * factor)), max(1, round(height * factor)))
    if target != (width, height):
        frame = cv2.resize(frame, target, interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(path), frame, PNG_FAST):
        raise OSError(path)
    return target


def prepare(source: Path, output: Path, max_edge: int = 1200,
            sample_fps: float = 6., keep_full: bool = True) -> dict:
    """Resize an image folder, or frames sampled from a video, into a sequence.

    With keep_full, unresized copies go to full/ for feature matching, photometric
    fusion and texturing, which all gain from the original resolution.
    """
    if max_edge < 128:
        raise ValueError('max_edge must be at least 128 pixels')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Prepared output is not empty: {output}')
    if is_video(source):
        if sample_fps <= 0:
            raise ValueError('sample_fps must be positive')
        step = video_step(source, sample_fps)
        frames = video_frames(source, step)
    else:
        images = sorted((path for path in source.iterdir()
                         if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
                        key=image_order)
        if len(images) < 3:
            raise ValueError(f'Need at least three images in {source}')
        step = None
        frames = ((path.name, path) for path in images)
    frame_dir = output / 'frames'
    mask_dir = output / 'masks'
    frame_dir.mkdir(parents=True)
    mask_dir.mkdir()
    full_dir = output / 'full'
    if keep_full:
        full_dir.mkdir()
    workers = os.cpu_count() or 4
    labels = []
    sizes = []
    pending = deque()
    # Decoding, resizing, and PNG encoding release the GIL, so threads overlap them.
    with ThreadPoolExecutor(workers) as pool:
        for index, (label, source_frame) in enumerate(
                bar(frames, desc='Preparing frames', unit='frame')):
            labels.append(label)
            pending.append(pool.submit(write_frame, source_frame,
                                       frame_dir / f'{index:08d}.png', max_edge,
                                       full_dir / f'{index:08d}.png' if keep_full else None))
            # Bound decoded video frames held in memory.
            while len(pending) > 2 * workers:
                sizes.append(pending.popleft().result())
        sizes.extend(future.result() for future in pending)
    if len(labels) < 3:
        raise ValueError(f'Need at least three frames from {source}')
    size = sizes[0]
    if any(target != size for target in sizes):
        raise ValueError('Input images must share one aspect ratio and size')
    ok, mask_png = cv2.imencode('.png', np.full(size[::-1], 255, dtype=np.uint8), PNG_FAST)
    if not ok:
        raise OSError('Cannot encode mask')
    mapping = []
    for index, label in enumerate(labels):
        (mask_dir / f'{index:08d}.png').write_bytes(mask_png.tobytes())
        mapping.append({'frame': index, 'source': label})
    report = {'source': str(source.resolve()), 'frames': len(mapping),
              'size': list(size), 'max_edge': max_edge, 'video_step': step,
              'calibration': 'estimated by COLMAP',
              'masks': 'all-white compatibility masks for full-scene depth',
              'mapping': mapping}
    (output / 'sequence.json').write_text(json.dumps(report, indent=2) + '\n')
    return {key: value for key, value in report.items() if key != 'mapping'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--max-edge', type=int, default=1200)
    parser.add_argument('--sample-fps', type=float, default=6.,
                        help='Frames per second kept from a video source')
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.max_edge,
                             args.sample_fps), indent=2))


if __name__ == '__main__':
    main()
