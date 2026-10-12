#!/usr/bin/env python3
"""Prepare offline MobileBrick test sequences at half image resolution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from recon.common.progress import bar, done


def prepare_scene(source: Path, output: Path) -> dict:
    images = sorted((source / 'image').glob('*.jpg'))
    if not images:
        raise ValueError(f'No RGB images in {source}')
    first = cv2.imread(str(images[0]))
    if first is None:
        raise ValueError(images[0])
    source_height, source_width = first.shape[:2]
    width, height = source_width // 2, source_height // 2
    (output / 'frames').mkdir(parents=True, exist_ok=True)
    (output / 'masks').mkdir(exist_ok=True)
    intrinsics = []
    video = cv2.VideoWriter(str(output / 'scan.mp4'),
                            cv2.VideoWriter_fourcc(*'mp4v'), 10,
                            (width, height))
    if not video.isOpened():
        raise OSError(output / 'scan.mp4')
    for image_path in images:
        frame = int(image_path.stem)
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(source / 'mask' / f'{frame:06d}.png'), 0)
        matrix = np.loadtxt(source / 'intrinsic' / f'{frame:06d}.txt')
        if (image is None or mask is None or image.shape[:2] != (source_height, source_width)
                or mask.shape != image.shape[:2] or matrix.shape != (3, 3)):
            raise ValueError(f'Frame, mask, or intrinsic mismatch: {image_path}')
        small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        selected = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(output / 'frames' / f'{frame:08d}.png'), small)
        cv2.imwrite(str(output / 'masks' / f'{frame:08d}.png'), selected)
        video.write(small)
        intrinsics.append(matrix)
    video.release()
    median = np.median(np.stack(intrinsics), axis=0)
    calibration = {'camera_matrix': median.tolist(),
                   'image_size': [source_width, source_height],
                   'source': 'MobileBrick per-frame intrinsics median; use raw intrinsic/ for exact cameras'}
    (output / 'calibration.json').write_text(json.dumps(calibration, indent=2) + '\n')
    summary = {'scene': source.name, 'frames': len(images),
               'source_size': [source_width, source_height],
               'working_size': [width, height],
               'mask_source': 'MobileBrick reference mask for isolated geometry benchmark'}
    (output / 'sequence.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=Path('runs/real/mobilebrick_dataset/test'))
    parser.add_argument('--output', type=Path,
                        default=Path('runs/real/mobilebrick_benchmark'))
    args = parser.parse_args()
    scenes = [scene for scene in sorted(args.source.iterdir()) if scene.is_dir()]
    for scene in bar(scenes, desc='Preparing MobileBrick', unit='scene'):
        prepare_scene(scene, args.output / scene.name)
    done(f'Prepared {len(scenes)} MobileBrick scenes in {args.output}')


if __name__ == '__main__':
    main()
