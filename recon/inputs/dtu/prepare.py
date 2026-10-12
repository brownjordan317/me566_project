#!/usr/bin/env python3
"""Convert the NeuS/IDR DTU release into this project's dataset and sequence layout.

DTU stores one projection matrix per image in millimetres. Each is decomposed
into intrinsics and a camera-to-world pose; poses are written in metres so the
existing trajectory and mesh tools apply unchanged. Frames are halved like
MobileBrick, and the reference scan and evaluation masks are linked per scan.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from recon.common.progress import bar, done


def decompose(projection: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Intrinsics and camera-to-world pose (metres) of a DTU world_mat.

    Adapted from load_K_Rt_from_P in NeuS (models/dataset.py; MIT, Copyright (c)
    2021 Peng Wang), which comes from IDR (MIT, Copyright (c) 2020 Lior Yariv).
    See THIRD_PARTY_NOTICES.md.
    """
    K, rotation, center = cv2.decomposeProjectionMatrix(projection[:3])[:3]
    K = K / K[2, 2]
    pose = np.eye(4)
    pose[:3, :3] = rotation.T
    pose[:3, 3] = (center[:3] / center[3]).ravel() / 1000.
    return K, pose


def prepare_scan(source: Path, reference: Path, data: Path, sequence: Path) -> dict:
    scan = int(source.name.removeprefix('dtu_scan'))
    # Skip hidden files such as the macOS '._000.png' forks in scan55's masks.
    images = sorted(p for p in (source / 'image').glob('*.png') if not p.name.startswith('.'))
    masks = sorted(p for p in (source / 'mask').glob('*.png') if not p.name.startswith('.'))
    cameras = np.load(source / 'cameras_sphere.npz')
    if not images or len(images) != len(masks):
        raise ValueError(f'Image/mask count mismatch in {source}')
    for folder in ('intrinsic', 'pose'):
        (data / folder).mkdir(parents=True, exist_ok=True)
    (sequence / 'frames').mkdir(parents=True, exist_ok=True)
    (sequence / 'masks').mkdir(exist_ok=True)
    matrices = []
    for index, (image_path, mask_path) in enumerate(zip(images, masks)):
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or image.shape[:2] != mask.shape:
            raise ValueError(f'Bad frame or mask: {image_path}')
        K, pose = decompose(cameras[f'world_mat_{index}'])
        np.savetxt(data / 'intrinsic' / f'{index:06d}.txt', K, fmt='%.8f')
        np.savetxt(data / 'pose' / f'{index:06d}.txt', pose, fmt='%.10f')
        height, width = image.shape[:2]
        size = (width // 2, height // 2)
        cv2.imwrite(str(sequence / 'frames' / f'{index:08d}.png'),
                    cv2.resize(image, size, interpolation=cv2.INTER_AREA))
        cv2.imwrite(str(sequence / 'masks' / f'{index:08d}.png'),
                    np.where(cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST) > 127,
                             255, 0).astype(np.uint8))
        matrices.append(K)
    links = {'stl_total.ply': reference / 'Points/stl' / f'stl{scan:03d}_total.ply',
             'ObsMask.mat': reference / 'SampleSet/MVS Data/ObsMask' / f'ObsMask{scan}_10.mat',
             'Plane.mat': reference / 'SampleSet/MVS Data/ObsMask' / f'Plane{scan}.mat'}
    # Source images, so Makefile checks for $(DATA)/<scan>/image find the scan.
    image_link = data / 'image'
    image_link.unlink(missing_ok=True)
    image_link.symlink_to((source / 'image').resolve())
    (data / 'reference').mkdir(exist_ok=True)
    for name, target in links.items():
        if not target.is_file():
            raise FileNotFoundError(target)
        link = data / 'reference' / name
        link.unlink(missing_ok=True)
        link.symlink_to(target.resolve())
    calibration = {'camera_matrix': np.median(np.stack(matrices), axis=0).tolist(),
                   'image_size': [width, height],
                   'source': 'DTU per-image intrinsics median; use intrinsic/ for exact cameras'}
    (sequence / 'calibration.json').write_text(json.dumps(calibration, indent=2) + '\n')
    summary = {'scene': f'scan{scan}', 'frames': len(images), 'source_size': [width, height],
               'working_size': list(size), 'mask_source': 'IDR/NeuS DTU object masks'}
    (sequence / 'sequence.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='DTU root holding neus/, Points/ and SampleSet/')
    parser.add_argument('--data', type=Path, required=True,
                        help='Converted per-scan cameras, poses and reference links')
    parser.add_argument('--output', type=Path, default=Path('runs/real/dtu_benchmark'),
                        help='Prepared half-resolution sequences')
    args = parser.parse_args()
    scans = sorted((args.source / 'neus').glob('dtu_scan*'))
    for scan in bar(scans, desc='Preparing DTU', unit='scan'):
        name = 'scan' + scan.name.removeprefix('dtu_scan')
        prepare_scan(scan, args.source, args.data / name, args.output / name)
    done(f'Prepared {len(scans)} DTU scans in {args.output}')


if __name__ == '__main__':
    main()
