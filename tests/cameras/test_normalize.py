"""Depth normalisation of COLMAP text models."""
from pathlib import Path

import numpy as np

from recon.cameras.sfm import normalize_depth


def test_normalize_depth_sets_median_depth_and_keeps_geometry(tmp_path: Path):
    # Two identity-rotation cameras looking down +z at points 4-6 units away.
    (tmp_path / 'cameras.txt').write_text('1 SIMPLE_PINHOLE 100 100 50 50 50\n')
    points = {1: (0, 0, 4.), 2: (1, 0, 5.), 3: (0, 1, 6.)}
    (tmp_path / 'points3D.txt').write_text(''.join(f'{i} {x} {y} {z} 255 255 255 0.1\n'
                                                   for i, (x, y, z) in points.items()))
    observations = ' '.join(f'{i}.0 {i}.0 {i}' for i in points)
    (tmp_path / 'images.txt').write_text(f'1 1 0 0 0 0 0 0 1 a.png\n{observations}\n'
                                         f'2 1 0 0 0 -1 0 0 1 b.png\n{observations}\n')
    factor = normalize_depth(tmp_path, 10.)
    assert np.isclose(factor, 2.)                                     # median depth was 5
    lines = (tmp_path / 'images.txt').read_text().split('\n')
    assert np.allclose([float(v) for v in lines[2].split()[5:8]], [-2, 0, 0])
    z = [float(line.split()[3]) for line in (tmp_path / 'points3D.txt').read_text().splitlines()]
    assert np.allclose(z, [8, 10, 12])
