"""COLMAP structure from motion around externally matched features.

Creates the database, installs cameras and keypoints, verifies raw matches in
parallel, runs the incremental mapper with a progress bar (or builds the model from
given cameras, initialised_model), and prepares text
models for per-frame focal refinement and for rescaling to working frames.
"""
from __future__ import annotations

import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

import numpy as np

from recon.common.progress import bar

SIMPLE_PINHOLE = 0
PINHOLE = 1






def install_per_image_cameras(database: Path, matrices: dict[int, np.ndarray],
                              width: int, height: int) -> None:
    """Give every image its supplied, fixed PINHOLE camera."""
    with sqlite3.connect(database) as connection:
        images = connection.execute('SELECT image_id, name FROM images').fetchall()
        if len(images) != len(matrices):
            raise ValueError('Per-frame intrinsics and COLMAP images differ')
        for image_id, name in images:
            matrix = matrices[int(Path(name).stem)]
            params = np.array([matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]],
                              dtype='<f8')
            camera_id = image_id + 1
            connection.execute(
                'INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length) '
                'VALUES (?, 1, ?, ?, ?, 1)', (camera_id, width, height, params.tobytes()))
            connection.execute('UPDATE images SET camera_id=? WHERE image_id=?',
                               (camera_id, image_id))
        connection.execute('DELETE FROM cameras WHERE camera_id=1')


def create_database(database: Path, colmap: Path, names: list[str],
                    width: int, height: int, model: str = 'simple_pinhole') -> None:
    """Create a COLMAP database with one shared camera at COLMAP's default prior.

    simple_pinhole has one focal length; pinhole has separate x and y focal lengths
    (non-square pixels, as on DTU's camera)."""
    subprocess.run([str(colmap), 'database_creator', '--database_path', str(database)],
                   check=True, stdout=subprocess.DEVNULL)
    focal = 1.2 * max(width, height)
    model_id = {'simple_pinhole': SIMPLE_PINHOLE, 'pinhole': PINHOLE}[model]
    params = np.array([focal] * (1 + model_id) + [width / 2, height / 2], dtype='<f8')
    with sqlite3.connect(database) as connection:
        connection.execute(
            'INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length) '
            'VALUES (1, ?, ?, ?, ?, 0)', (model_id, width, height, params.tobytes()))
        connection.executemany('INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, 1)',
                               [(index + 1, name) for index, name in enumerate(names)])


def replace_features(database: Path, features: dict[str, dict[str, np.ndarray]]) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute('DELETE FROM two_view_geometries')
        connection.execute('DELETE FROM matches')
        connection.execute('DELETE FROM descriptors')
        connection.execute('DELETE FROM keypoints')
        for image_id, name in connection.execute('SELECT image_id, name FROM images'):
            feature = features[name]
            points = np.ascontiguousarray(feature['keypoints'], dtype='<f4')
            descriptor_values = feature['descriptors']
            # COLMAP's raw-match verifier uses keypoints; 128-byte descriptors
            # are retained only to satisfy its database schema.
            descriptors = (np.ascontiguousarray(np.clip(descriptor_values * 512, 0, 255), dtype=np.uint8)
                           if descriptor_values.shape[1] == 128
                           else np.zeros((len(points), 128), dtype=np.uint8))
            if points.ndim != 2 or points.shape[1] != 2 or descriptors.shape != (len(points), 128):
                raise ValueError(f'Invalid LightGlue features: {name}')
            connection.execute('INSERT INTO keypoints(image_id, rows, cols, data) VALUES (?, ?, 2, ?)',
                               (image_id, len(points), points.tobytes()))
            connection.execute('INSERT INTO descriptors(image_id, rows, cols, data) VALUES (?, ?, 128, ?)',
                               (image_id, len(points), descriptors.tobytes()))


def run_mapper(command: list[str], log: Path, images: int, env: dict) -> None:
    """Run COLMAP's mapper with a bar of registered images read from its log."""
    progress = bar(total=images, desc='COLMAP mapping', unit='img')
    registered = 0
    with log.open('w') as stream:
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        with log.open() as reader:
            while True:
                finished = process.poll() is not None
                for line in reader:
                    if 'Registering image #' in line:
                        count = int(line.rsplit('(', 1)[1].split(')')[0])
                        if count > registered:
                            progress.update(min(count, images) - min(registered, images))
                            registered = count
                if finished:
                    break
                time.sleep(.5)
    progress.close()
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def initialised_model(database: Path, cameras: Path, scale: float, output: Path, colmap: str, env: dict,
                      rounds: int, refine_focal: bool, log: Path) -> Path:
    """A registered model from given cameras instead of incremental mapping.

    cameras is a text model (e.g. recon.cameras.feedforward) at the working
    resolution; scale takes it to the database's keypoint resolution. Its poses
    and median focal length seed the database's single shared camera, the
    verified matches are triangulated with those cameras fixed, and with rounds
    > 0 bundle adjustment (focal refined if asked) alternates with re-triangulation
    from scratch, as incremental mapping does. Six rounds reached COLMAP's own
    optimum from VGGT-Omega's cameras on jeep and DTU scan37.
    """
    with sqlite3.connect(database) as connection:
        camera_id, width, height = connection.execute('SELECT camera_id, width, height FROM cameras').fetchone()
        ids = {name: image_id for image_id, name in connection.execute('SELECT image_id, name FROM images')}
    rows = [line.split() for line in (cameras / 'cameras.txt').read_text().splitlines()
            if line and not line.startswith('#')]
    focal = float(np.median([float(row[4]) for row in rows])) * scale
    poses = [line.split() for line in (cameras / 'images.txt').read_text().splitlines()
             if line and not line.startswith('#') and len(line.split()) >= 10]
    source = output / 'init'
    source.mkdir(parents=True, exist_ok=True)
    (source / 'cameras.txt').write_text(f'{camera_id} SIMPLE_PINHOLE {width} {height} {focal} {width / 2} {height / 2}\n')
    images = []
    for fields in poses:
        if fields[9] in ids:
            images += [' '.join([str(ids[fields[9]]), *fields[1:8], str(camera_id), fields[9]]), '']
    (source / 'images.txt').write_text('\n'.join(images) + '\n')
    (source / 'points3D.txt').write_text('')
    with log.open('w') as stream:
        run = lambda *args: subprocess.run([colmap, *map(str, args)], check=True, env=env, stdout=stream,
                                           stderr=subprocess.STDOUT)
        for index in range(max(rounds, 0) + 1):
            triangulated = output / f'triangulated_{index}'
            triangulated.mkdir(exist_ok=True)
            run('point_triangulator', '--database_path', database, '--image_path', output,
                '--input_path', source, '--output_path', triangulated)
            if index == rounds:
                return triangulated
            adjusted = output / f'adjusted_{index}'
            adjusted.mkdir(exist_ok=True)
            run('bundle_adjuster', '--input_path', triangulated, '--output_path', adjusted,
                '--BundleAdjustment.refine_focal_length', int(refine_focal),
                '--BundleAdjustment.refine_principal_point', 0, '--BundleAdjustment.refine_extra_params', 0,
                '--BundleAdjustment.max_num_iterations', 300)
            # Next round: the adjusted cameras, without points (re-triangulated from scratch).
            source = output / f'cameras_{index}'
            source.mkdir(exist_ok=True)
            run('model_converter', '--input_path', adjusted, '--output_path', source, '--output_type', 'TXT')
            # Pose lines have 10 fields; observation lines a multiple of 3 (possibly none).
            body = [line for line in (source / 'images.txt').read_text().splitlines()
                    if not line.startswith('#') and len(line.split()) == 10]
            (source / 'images.txt').write_text('\n'.join(sum(([pose, ''] for pose in body), [])) + '\n')
            (source / 'points3D.txt').write_text('')
    return source


def verify_matches(database: Path, blocks: list[str], workdir: Path, colmap: str,
                   workers: int, env: dict, log: Path) -> None:
    """Geometrically verify raw matches with parallel COLMAP importers.

    COLMAP 3.9 verifies imported raw matches on one thread, so pair chunks run in
    separate processes on database copies and their results are merged back.
    """
    workdir.mkdir()
    # COLMAP databases use WAL journaling, and 'with sqlite3.connect()' commits
    # without closing, so recent writes can sit in database.db-wal. Fold them in
    # before copying, or the copies can lack images and cameras entirely.
    connection = sqlite3.connect(database)
    connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    connection.close()
    chunks = [chunk for chunk in np.array_split(np.arange(len(blocks)),
                                                max(1, min(workers, len(blocks) // 8)))
              if len(chunk)]
    processes = []
    with log.open('w') as stream:
        for index, chunk in enumerate(chunks):
            chunk_db = workdir / f'{index}.db'
            chunk_list = workdir / f'{index}.txt'
            shutil.copy2(database, chunk_db)
            chunk_list.write_text(''.join(blocks[position] for position in chunk))
            processes.append(subprocess.Popen(
                [colmap, 'matches_importer', '--database_path', str(chunk_db),
                 '--match_list_path', str(chunk_list), '--match_type', 'raw',
                 '--SiftMatching.use_gpu', '0', '--SiftMatching.num_threads', '1'],
                env=env, stdout=stream, stderr=subprocess.STDOUT))
        failed = [process.args for process in processes if process.wait()]
    if failed:
        raise RuntimeError(f'COLMAP match verification failed; see {log}')
    with sqlite3.connect(database) as connection:
        for index in range(len(chunks)):
            connection.execute('ATTACH DATABASE ? AS chunk', (str(workdir / f'{index}.db'),))
            connection.execute('INSERT INTO matches SELECT * FROM chunk.matches')
            connection.execute('INSERT INTO two_view_geometries '
                               'SELECT * FROM chunk.two_view_geometries')
            connection.commit()
            connection.execute('DETACH DATABASE chunk')
    shutil.rmtree(workdir)


def split_cameras(model: Path) -> None:
    """Give every image its own copy of its camera so focal lengths can differ."""
    cameras = {}
    for line in (model / 'cameras.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            fields = line.split()
            cameras[fields[0]] = fields[1:]
    entries = [line for line in (model / 'images.txt').read_text().splitlines()
               if not line.startswith('#')]
    camera_lines, image_lines = [], []
    for pose, observations in zip(entries[::2], entries[1::2]):
        fields = pose.split()
        image_id, camera_id = fields[0], fields[8]
        camera_lines.append(' '.join([image_id] + cameras[camera_id]))
        image_lines += [' '.join(fields[:8] + [image_id] + fields[9:]), observations]
    (model / 'cameras.txt').write_text('\n'.join(camera_lines) + '\n')
    (model / 'images.txt').write_text('\n'.join(image_lines) + '\n')


def rescale_text_model(model: Path, scale: float) -> None:
    """Scale a PINHOLE/SIMPLE_PINHOLE text model's cameras and 2D observations.

    Intrinsics scale the same way dataset matrices are scaled to working frames
    (multiply focal and principal point), and so do the observed keypoints.
    """
    lines = []
    for line in (model / 'cameras.txt').read_text().splitlines():
        if not line or line.startswith('#'):
            lines.append(line)
            continue
        fields = line.split()
        size = [str(round(int(value) * scale)) for value in fields[2:4]]
        params = [repr(float(value) * scale) for value in fields[4:]]
        lines.append(' '.join(fields[:2] + size + params))
    (model / 'cameras.txt').write_text('\n'.join(lines) + '\n')
    lines = []
    body = [line for line in (model / 'images.txt').read_text().splitlines()]
    header = [line for line in body if line.startswith('#')]
    entries = [line for line in body if not line.startswith('#')]
    for pose, observations in zip(entries[::2], entries[1::2]):
        values = observations.split()
        for index in range(0, len(values), 3):
            values[index] = repr(float(values[index]) * scale)
            values[index + 1] = repr(float(values[index + 1]) * scale)
        lines += [pose, ' '.join(values)]
    (model / 'images.txt').write_text('\n'.join(header + lines) + '\n')


def normalize_depth(model: Path, target: float) -> float:
    """Rescale a text model so its sparse points' median depth in their cameras is target.

    COLMAP's unit is arbitrary (set by its first image pair). Fixing the median depth
    makes map-unit settings (voxel size, truncation, depth limits) apply to any capture.
    Returns the scale factor applied.
    """
    points = {}
    for line in (model / 'points3D.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            fields = line.split()
            points[int(fields[0])] = np.array([float(v) for v in fields[1:4]])
    body = (model / 'images.txt').read_text().splitlines()
    entries = [line for line in body if not line.startswith('#')]
    depths = []
    for pose, observations in zip(entries[::2], entries[1::2]):
        fields = pose.split()
        q = np.array([float(v) for v in fields[1:5]])
        w, x, y, z = q / np.linalg.norm(q)
        rotation = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                             [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                             [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        translation = np.array([float(v) for v in fields[5:8]])
        ids = [int(v) for v in observations.split()[2::3] if int(v) in points]
        if ids:
            depths.extend((np.stack([points[i] for i in ids]) @ rotation.T + translation)[:, 2])
    if not depths:
        raise ValueError(f'No observed points in {model}')
    factor = target / float(np.median(depths))
    lines = []
    for line in body:
        if line.startswith('#'):
            lines.append(line)
    for pose, observations in zip(entries[::2], entries[1::2]):
        fields = pose.split()
        fields[5:8] = [repr(float(v) * factor) for v in fields[5:8]]
        lines += [' '.join(fields), observations]
    (model / 'images.txt').write_text('\n'.join(lines) + '\n')
    lines = []
    for line in (model / 'points3D.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            fields = line.split()
            fields[1:4] = [repr(float(v) * factor) for v in fields[1:4]]
            line = ' '.join(fields)
        lines.append(line)
    (model / 'points3D.txt').write_text('\n'.join(lines) + '\n')
    return factor
