"""Fill frames may add surface where no real frame looked, and must not alter anything real frames observed."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

W, H, F = 40, 30, 20.


def scene(tmp: Path) -> tuple[Path, Path, Path, Path]:
    """One camera at the origin looking along +z. Frame 0 (real) sees a plane at z=3 across the left half of
    the image only; frame 100 (fill) sees a plane at z=3.5 across the whole image."""
    seq, model, depth = tmp / 'seq', tmp / 'model', tmp / 'depth'
    for d in (seq / 'frames', seq / 'masks', model, depth):
        d.mkdir(parents=True)
    (model / 'cameras.txt').write_text(f'1 PINHOLE {W} {H} {F} {F} {W / 2} {H / 2}\n')
    (model / 'images.txt').write_text(''.join(f'{i + 1} 1 0 0 0 0 0 0 1 {n:08d}.png\n0.0 0.0 -1\n'
                                              for i, n in enumerate((0, 100))))
    (model / 'points3D.txt').write_text('')
    real = np.zeros((H, W), np.float32); real[:, :W // 2] = 3.
    np.save(depth / '00000000_grounded.npy', real)
    np.save(depth / '00000100_grounded.npy', np.full((H, W), 3.5, np.float32))
    for n in (0, 100):
        cv2.imwrite(str(seq / 'frames' / f'{n:08d}.png'), np.full((H, W, 3), 128, np.uint8))
        cv2.imwrite(str(seq / 'masks' / f'{n:08d}.png'), np.full((H, W), 255, np.uint8))
    box = np.array([[x, y, z] for x in (-1, 1) for y in (-.7, .7) for z in (2.6, 3.9)], float)
    landmarks = tmp / 'landmarks.ply'
    o3d.io.write_point_cloud(str(landmarks), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(box)))
    return seq, model, depth, landmarks


def surface(tmp: Path, *extra: str) -> tuple[np.ndarray, dict]:
    seq, model, depth, landmarks = scene(tmp)
    out = tmp / 'mesh.ply'
    subprocess.run([sys.executable, '-m', 'recon.surface.weighted_tsdf', str(seq), '--model', str(model),
                    '--depth-dir', str(depth), '--landmarks', str(landmarks), '--output', str(out),
                    '--voxel', '.05', '--truncation', '.2', '--margin', '0', '--min-views', '1',
                    '--robust-passes', '1', '--min-depth', '.1', '--device', 'cpu', *extra],
                   check=True, capture_output=True)
    return np.asarray(o3d.io.read_triangle_mesh(str(out)).vertices), json.loads(out.with_suffix('.json').read_text())


def depth_by_half(vertices: np.ndarray) -> tuple[float, float]:
    """Median surface depth left and right of the image centre, away from the dividing line."""
    left, right = vertices[vertices[:, 0] < -.2], vertices[vertices[:, 0] > .2]
    return float(np.median(left[:, 2])), float(np.median(right[:, 2]))


def test_fill_frames_only_add_where_nothing_real_looked(tmp_path):
    vertices, report = surface(tmp_path, '--fill-from', '100')
    left, right = depth_by_half(vertices)
    assert abs(left - 3.) < .05           # what the real frame saw is exactly what the real frame says
    assert abs(right - 3.5) < .05         # the half no real ray reached comes from the fill frame
    # and the fill's plane does not reappear behind the real one, as a second skin inside the object
    assert not np.any((vertices[:, 0] < -.2) & (vertices[:, 2] > 3.3))
    assert report['fill_voxels'] > 0 and report['fill_from'] == 100


def test_without_the_flag_every_frame_is_an_equal(tmp_path):
    vertices, report = surface(tmp_path)
    # As an equal, the fill frame reshapes what the real frame observed: its plane appears behind the real one.
    assert np.any((vertices[:, 0] < -.2) & (vertices[:, 2] > 3.3))
    assert report['fill_voxels'] == 0
