import sqlite3
from pathlib import Path

import numpy as np

from recon.cameras.matching import sequential_pairs
from recon.cameras.sfm import create_database, replace_features, rescale_text_model


def test_replaced_keypoints_keep_image_coordinates_and_descriptor_shape(tmp_path: Path):
    db = tmp_path / 'features.db'
    with sqlite3.connect(db) as con:
        con.execute('CREATE TABLE images(image_id INTEGER, name TEXT)')
        con.execute('CREATE TABLE keypoints(image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB)')
        con.execute('CREATE TABLE descriptors(image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB)')
        con.execute('CREATE TABLE matches(pair_id INTEGER)')
        con.execute('CREATE TABLE two_view_geometries(pair_id INTEGER)')
        con.execute("INSERT INTO images VALUES (3, 'frame.png')")
    points = np.array([[10.25, 20.5], [30.75, 40.0]], dtype=np.float32)
    replace_features(db, {'frame.png': {'keypoints': points,
                                        'descriptors': np.full((2, 128), .25, np.float32)}})
    with sqlite3.connect(db) as con:
        rows, cols, data = con.execute('SELECT rows, cols, data FROM keypoints').fetchone()
        drows, dcols, desc = con.execute('SELECT rows, cols, data FROM descriptors').fetchone()
    assert (rows, cols, drows, dcols) == (2, 2, 2, 128)
    np.testing.assert_array_equal(np.frombuffer(data, dtype='<f4').reshape(2, 2), points)
    assert len(desc) == 2 * 128
    replace_features(db, {'frame.png': {'keypoints': points,
                                        'descriptors': np.ones((2, 256), np.float32)}})
    with sqlite3.connect(db) as con:
        cols, data = con.execute('SELECT cols, data FROM descriptors').fetchone()
    assert cols == 128 and set(data) == {0}


def test_sequential_pairs_cover_neighbours_and_power_of_two_jumps():
    names = [f'{index:08d}.png' for index in range(40)]
    pairs = sequential_pairs(names, overlap=3)
    first = {b for a, b in pairs if a == names[0]}
    assert first == {names[1], names[2], names[3], names[4]}
    assert all(names.index(a) < names.index(b) for a, b in pairs)
    assert len(pairs) == len(set(pairs))


def test_loop_pairs_close_the_sequence_without_duplicates():
    names = [f'{index:08d}.png' for index in range(40)]
    open_pairs = set(sequential_pairs(names, overlap=3))
    loop_pairs = set(sequential_pairs(names, overlap=3, loop=True))
    assert open_pairs < loop_pairs
    assert (names[0], names[39]) in loop_pairs and (names[1], names[38]) in loop_pairs
    assert all(names.index(a) < names.index(b) for a, b in loop_pairs)
    assert len(sequential_pairs(names, overlap=3, loop=True)) == len(loop_pairs)


def test_created_database_has_one_shared_estimated_camera(tmp_path: Path):
    import os
    roots = [Path.home() / '.local/colmap-3.9.1', Path('/tmp/colmap-local')]   # as the Makefile
    root = next((root for root in roots if (root / 'usr/bin/colmap').is_file()), None)
    if root is None:
        import pytest
        pytest.skip('COLMAP executable is not installed here')
    colmap = root / 'usr/bin/colmap'
    os.environ['LD_LIBRARY_PATH'] = (f'{root}/usr/lib/x86_64-linux-gnu:'
                                     + os.environ.get('LD_LIBRARY_PATH', ''))
    db = tmp_path / 'database.db'
    create_database(db, colmap, ['00000000.png', '00000001.png'], 960, 720)
    with sqlite3.connect(db) as con:
        cameras = con.execute('SELECT camera_id, model, width, height, params FROM cameras').fetchall()
        images = con.execute('SELECT image_id, name, camera_id FROM images').fetchall()
    assert [row[:4] for row in cameras] == [(1, 0, 960, 720)]
    np.testing.assert_allclose(np.frombuffer(cameras[0][4], dtype='<f8'), [1152, 480, 360])
    assert images == [(1, '00000000.png', 1), (2, '00000001.png', 1)]


def test_verify_copies_include_writes_still_in_the_wal(tmp_path: Path, monkeypatch):
    from recon.cameras import sfm as module
    db = tmp_path / 'database.db'
    keep = sqlite3.connect(db)
    keep.execute('PRAGMA journal_mode=WAL')
    keep.execute('CREATE TABLE images(image_id INTEGER, name TEXT)')
    keep.execute("INSERT INTO images VALUES (1, 'a.png')")
    keep.commit()          # committed but left open, so the row stays in the WAL
    seen = []

    class Done:
        args = None
        def wait(self):
            return 0

    def fake_popen(command, **kwargs):
        copy = command[command.index('--database_path') + 1]
        seen.append(sqlite3.connect(copy).execute('SELECT COUNT(*) FROM images').fetchone()[0])
        return Done()

    monkeypatch.setattr(module.subprocess, 'Popen', fake_popen)
    monkeypatch.setattr(module.sqlite3, 'connect', sqlite3.connect)
    try:
        module.verify_matches(db, ['a.png b.png\n0 0\n\n'] * 8, tmp_path / 'chunks', 'colmap',
                              1, {}, tmp_path / 'verify.log')
    except Exception:
        pass               # merging from the fake importer may fail; the copy is what matters
    keep.close()
    assert seen == [1]


def test_rescaled_model_halves_cameras_and_observations(tmp_path: Path):
    (tmp_path / 'cameras.txt').write_text('# header\n1 PINHOLE 1600 1200 2900 2890 823 619\n')
    (tmp_path / 'images.txt').write_text('# header\n1 1 0 0 0 0 0 0 1 a.png\n100 200 5 1600 1 -1\n')
    rescale_text_model(tmp_path, .5)
    camera = (tmp_path / 'cameras.txt').read_text().splitlines()[1].split()
    assert camera[2:4] == ['800', '600']
    np.testing.assert_allclose([float(v) for v in camera[4:]], [1450, 1445, 411.5, 309.5])
    pose, observations = (tmp_path / 'images.txt').read_text().splitlines()[1:3]
    assert pose == '1 1 0 0 0 0 0 0 1 a.png'
    np.testing.assert_allclose([float(v) for v in observations.split()], [50, 100, 5, 800, .5, -1])


def test_split_cameras_gives_each_image_its_own_camera(tmp_path: Path):
    from recon.cameras.sfm import split_cameras
    (tmp_path / 'cameras.txt').write_text('# c\n1 SIMPLE_PINHOLE 800 600 740 400 300\n')
    (tmp_path / 'images.txt').write_text('# i\n3 1 0 0 0 0 0 0 1 a.png\n10 20 -1\n'
                                         '7 1 0 0 0 0 0 0 1 b.png\n30 40 5\n')
    split_cameras(tmp_path)
    assert (tmp_path / 'cameras.txt').read_text().split('\n')[:2] == [
        '3 SIMPLE_PINHOLE 800 600 740 400 300', '7 SIMPLE_PINHOLE 800 600 740 400 300']
    lines = (tmp_path / 'images.txt').read_text().split('\n')
    assert lines[0].split()[8] == '3' and lines[2].split()[8] == '7'
    assert lines[1] == '10 20 -1' and lines[3] == '30 40 5'
