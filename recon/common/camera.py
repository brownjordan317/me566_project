"""Read fixed or per-image pinhole intrinsics from a COLMAP text model."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def parse_cameras(model_dir: Path) -> dict[int, np.ndarray]:
    cameras = {}
    for line in (model_dir / 'cameras.txt').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        fields = line.split()
        if fields[1] == 'PINHOLE':
            fx, fy, cx, cy = map(float, fields[4:8])
        elif fields[1] == 'SIMPLE_PINHOLE':
            fx, cx, cy = map(float, fields[4:7])
            fy = fx
        else:
            raise ValueError('Only PINHOLE and SIMPLE_PINHOLE COLMAP cameras are supported')
        cameras[int(fields[0])] = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1.]], dtype=np.float64)
    if not cameras:
        raise ValueError(f'No cameras in {model_dir}')
    return cameras


def frame_intrinsics(model_dir: Path) -> dict[int, np.ndarray]:
    """Map numbered image frames to their calibrated cameras."""
    cameras = parse_cameras(model_dir)
    lines = [line for line in (model_dir / 'images.txt').read_text().splitlines()
             if line and not line.startswith('#')]
    if len(lines) % 2:
        raise ValueError('Invalid COLMAP image/observation lines')
    result = {}
    for line in lines[::2]:
        fields = line.split()
        number = int(Path(fields[9]).stem)
        result[number] = cameras[int(fields[8])]
    if not result:
        raise ValueError(f'No registered images in {model_dir}')
    return result


def load_dataset_intrinsics(directory: Path, frame_ids: list[int],
                            image_size: tuple[int, int],
                            calibration_size: tuple[int, int]) -> dict[int, np.ndarray]:
    """Scale the dataset's per-frame 3x3 matrices to working image pixels."""
    width, height = image_size
    source_width, source_height = calibration_size
    matrices = {}
    for number in frame_ids:
        path = directory / f'{number:06d}.txt'
        if not path.is_file():
            raise FileNotFoundError(path)
        matrix = np.loadtxt(path, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError(f'Invalid intrinsic matrix: {path}')
        matrix = matrix.copy()
        matrix[0] *= width / source_width
        matrix[1] *= height / source_height
        if (matrix[0, 0] <= 0 or matrix[1, 1] <= 0
                or not 0 < matrix[0, 2] < width
                or not 0 < matrix[1, 2] < height):
            raise ValueError(f'Intrinsic matrix does not fit frame: {path}')
        matrices[number] = matrix
    return matrices
