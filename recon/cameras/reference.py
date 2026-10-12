#!/usr/bin/env python3
"""Replace estimated COLMAP poses with a dataset's reference poses.

The reference camera-to-world poses are expressed at the estimated model's
scale (map units differ from metres), so depth and fusion settings tuned in
map units still apply. Points are re-triangulated from the existing verified
matches with the reference poses held fixed. Use it to separate pose error from
depth and fusion error.

Cameras come from the estimated run's database, at the resolution its features
were detected at; with INTRINSICS=dataset those are the dataset's own cameras,
so the result has reference poses and reference intrinsics. The final text
model is rescaled to the prepared sequence's frames, as the pose stage does.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from recon.evaluation.metrics.mesh import similarity
from recon.cameras.sfm import rescale_text_model

CAMERA_MODELS = {0: 'SIMPLE_PINHOLE', 1: 'PINHOLE'}


def database_cameras(database: Path) -> str:
    """cameras.txt lines for the cameras stored in a COLMAP database."""
    with sqlite3.connect(database) as connection:
        rows = connection.execute('SELECT camera_id, model, width, height, params FROM cameras').fetchall()
    lines = []
    for camera_id, model, width, height, params in rows:
        if model not in CAMERA_MODELS:
            raise ValueError(f'Unsupported COLMAP camera model {model}')
        values = ' '.join(repr(float(value)) for value in np.frombuffer(params, dtype=np.float64))
        lines.append(f'{camera_id} {CAMERA_MODELS[model]} {width} {height} {values}')
    return '\n'.join(lines) + '\n'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--estimated', type=Path, required=True,
                   help='Estimated pose folder (database.db, sparse_txt, export/trajectory.csv)')
    p.add_argument('--pose-dir', type=Path, required=True, help='Reference NNNNNN.txt camera-to-world')
    p.add_argument('--frames', type=Path, required=True,
                   help='Images the database features were detected on')
    p.add_argument('--sequence', type=Path, required=True,
                   help='Prepared sequence whose frames/ the output model is scaled to')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--colmap', type=Path, default=Path('colmap'))
    a = p.parse_args()
    _, _, _, scale, _ = similarity(a.estimated / 'export/trajectory.csv', a.pose_dir)
    a.output.mkdir(parents=True, exist_ok=True)
    database = a.output / 'database.db'
    shutil.copy2(a.estimated / 'database.db', database)
    connection = sqlite3.connect(database)
    images = connection.execute('SELECT image_id, name, camera_id FROM images').fetchall()
    connection.close()
    init = a.output / 'reference_init'
    init.mkdir(exist_ok=True)
    (init / 'cameras.txt').write_text(database_cameras(database))
    lines = []
    for image_id, name, camera_id in images:
        c2w = np.loadtxt(a.pose_dir / f'{int(Path(name).stem):06d}.txt')
        rotation = c2w[:3, :3].T                        # world-to-camera
        center = c2w[:3, 3] / scale                     # metres to estimated map units
        translation = -rotation @ center
        qx, qy, qz, qw = Rotation.from_matrix(rotation).as_quat()
        lines.append(f'{image_id} {qw} {qx} {qy} {qz} {" ".join(map(str, translation))} {camera_id} {name}\n\n')
    (init / 'images.txt').write_text(''.join(lines))
    (init / 'points3D.txt').write_text('')
    env = os.environ.copy()
    sparse = a.output / 'sparse' / '0'
    sparse.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(a.colmap), 'point_triangulator', '--database_path', str(database),
                    '--image_path', str(a.frames), '--input_path', str(init),
                    '--output_path', str(sparse)], check=True, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    text = a.output / 'sparse_txt'
    text.mkdir(exist_ok=True)
    subprocess.run([str(a.colmap), 'model_converter', '--input_path', str(sparse),
                    '--output_path', str(text), '--output_type', 'TXT'], check=True, env=env,
                   stdout=subprocess.DEVNULL)
    working_width = cv2.imread(str(sorted((a.sequence / 'frames').glob('*.png'))[0])).shape[1]
    camera_width = int(next(line for line in (text / 'cameras.txt').read_text().splitlines()
                            if line and not line.startswith('#')).split()[2])
    if camera_width != working_width:
        rescale_text_model(text, working_width / camera_width)
    summary = json.loads((a.estimated / 'summary.json').read_text())
    summary.update(model='reference poses, re-triangulated LightGlue matches',
                   map_units_per_metre=1 / scale, registered_model=str(sparse))
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
