"""Per-frame camera calibration must survive COLMAP export and projection."""
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from recon.common.camera import frame_intrinsics, load_dataset_intrinsics
from recon.cameras.sfm import install_per_image_cameras


class CameraModelTests(unittest.TestCase):
    def test_dataset_intrinsics_scale_to_working_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            np.savetxt(folder / '000000.txt',
                       [[1200, 0, 640], [0, 1202, 360], [0, 0, 1]])
            result = load_dataset_intrinsics(folder, [0], (640, 360), (1280, 720))
            np.testing.assert_allclose(result[0],
                                       [[600, 0, 320], [0, 601, 180], [0, 0, 1]])

    def test_each_image_receives_its_own_fixed_colmap_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            database = folder / 'database.db'
            with sqlite3.connect(database) as connection:
                connection.execute('CREATE TABLE cameras(camera_id INTEGER PRIMARY KEY, '
                                   'model INTEGER, width INTEGER, height INTEGER, '
                                   'params BLOB, prior_focal_length INTEGER)')
                connection.execute('CREATE TABLE images(image_id INTEGER PRIMARY KEY, '
                                   'name TEXT, camera_id INTEGER)')
                connection.execute('INSERT INTO cameras VALUES (1, 1, 640, 360, ?, 1)',
                                   (np.zeros(4, dtype='<f8').tobytes(),))
                connection.executemany('INSERT INTO images VALUES (?, ?, 1)',
                                       [(1, '00000000.png'), (2, '00000001.png')])
            matrices = {
                0: np.array([[600, 0, 320], [0, 601, 180], [0, 0, 1.]]),
                1: np.array([[610, 0, 321], [0, 611, 181], [0, 0, 1.]]),
            }
            install_per_image_cameras(database, matrices, 640, 360)
            with sqlite3.connect(database) as connection:
                rows = connection.execute('SELECT name, params FROM images '
                                          'JOIN cameras USING(camera_id) ORDER BY name').fetchall()
            self.assertEqual(len(rows), 2)
            np.testing.assert_array_equal(np.frombuffer(rows[0][1], dtype='<f8'),
                                          [600, 601, 320, 180])
            np.testing.assert_array_equal(np.frombuffer(rows[1][1], dtype='<f8'),
                                          [610, 611, 321, 181])

    def test_exported_image_uses_its_camera_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'cameras.txt').write_text(
                '2 PINHOLE 640 360 600 601 320 180\n'
                '3 PINHOLE 640 360 610 611 321 181\n')
            (folder / 'images.txt').write_text(
                '1 1 0 0 0 0 0 0 2 00000000.png\n0 0 -1\n'
                '2 1 0 0 0 0 0 0 3 00000001.png\n0 0 -1\n')
            result = frame_intrinsics(folder)
            self.assertEqual(result[0][0, 0], 600)
            self.assertEqual(result[1][0, 0], 610)


if __name__ == '__main__':
    unittest.main()
