#!/usr/bin/env python3
"""Rank pipeline stages by how much a perfect input would improve the result.

Every scene is reconstructed four times:

  ours                our full pipeline
  poses               reference poses and intrinsics, our depth
  depth               our poses, object depth rendered from the reference
  poses_depth         both

The shape metrics (scale-aligned, see recon.evaluation.metrics) of each rung
show how much error each stage is responsible for: the gap from ours to
`poses` is what camera estimation costs, ours to `depth` what depth costs, and
what remains at `poses_depth` is lost in fusion, extraction and cleanup. Runs
go through the Makefile; each scene keeps only its metrics and object mesh.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from recon.common.progress import bar, done

RUNGS = ('ours', 'poses', 'depth', 'poses_depth')
DATASETS = {
    'mobilebrick': {'data': Path('runs/real/mobilebrick_dataset/test'),
                    'sequences': Path('runs/real/mobilebrick_benchmark'),
                    'make': [], 'evaluate': 'evaluate', 'metrics': 'object',
                    'columns': (('shape_median_mm', 'shape median'), ('shape_p90_mm', 'shape p90'),
                                ('reference_vertices_within_10mm_pct', 'coverage %'))},
    'dtu': {'data': Path('runs/real/dtu_dataset'), 'sequences': Path('runs/real/dtu_benchmark'),
            'make': ['PER_FRAME_FOCAL='], 'evaluate': 'evaluate-dtu', 'metrics': 'object_dtu',
            'columns': (('shape_chamfer_mm', 'shape chamfer'), ('shape_accuracy_mm', 'shape accuracy'),
                        ('shape_completeness_mm', 'shape completeness'))},
}
TRUTH_CAMERAS = ['INTRINSICS=dataset', 'PER_FRAME_FOCAL=']


class Scene:
    def __init__(self, name: str, config: dict, root: Path, surfaces: Path, colmap: str):
        self.name, self.config, self.root, self.colmap = name, config, root / name, colmap
        self.data = config['data'] / name
        self.sequence = config['sequences'] / name
        self.reference = (self.data / 'mesh/gt_mesh.ply' if config['metrics'] == 'object'
                          else surfaces / f'{name}.ply')
        self.env = dict(os.environ, NO_PROGRESS='1')

    def make(self, rung: str, targets: list[str], extra: list[str] = ()) -> None:
        command = ['make', '--no-print-directory', '-j1', f'SCENE={self.name}',
                   f'RUN={self.root / rung}', f'DATA={self.config["data"]}',
                   f'SEQ_ROOT={self.config["sequences"]}', f'MASKS={self.sequence / "masks"}',
                   'KEEP_INTERMEDIATE=1', *self.config['make'], *extra, *targets]
        self.run(rung, command)

    def run(self, rung: str, command: list[str]) -> None:
        with (self.root / f'{rung}.log').open('a') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=self.env, check=True)

    def truth_poses(self, rung: str) -> None:
        """Pose stage with dataset cameras, then reference poses swapped in."""
        self.make(rung, ['poses'], TRUTH_CAMERAS)
        poses = self.root / rung / 'poses'
        self.run(rung, [sys.executable, '-m', 'recon.cameras.reference', '--estimated', str(poses),
                        '--pose-dir', str(self.data / 'pose'), '--frames', str(self.data / 'image'),
                        '--sequence', str(self.sequence), '--output', str(poses / 'reference'),
                        '--colmap', self.colmap])
        for stale in ('sparse_txt', 'export', 'export_object'):
            shutil.rmtree(poses / stale)
        (poses / 'reference/sparse_txt').rename(poses / 'sparse_txt')
        shutil.copy2(poses / 'reference/summary.json', poses / 'summary.json')

    def truth_depth(self, base: str, rung: str) -> None:
        """Copy a finished rung and replace its object depth with reference depth."""
        target = self.root / rung
        shutil.copytree(self.root / base, target, symlinks=True)
        for stale in ('depth/consistent', 'work', 'mesh', 'points', 'metrics'):
            shutil.rmtree(target / stale, ignore_errors=True)
        self.run(rung, [sys.executable, '-m', 'recon.evaluation.diagnostics.oracle_depth',
                        '--model', str(target / 'poses/sparse_txt'),
                        '--trajectory', str(target / 'poses/export/trajectory.csv'),
                        '--pose-dir', str(self.data / 'pose'), '--reference', str(self.reference),
                        '--masks-dir', str(self.sequence / 'masks'),
                        '--depth-dir', str(target / 'depth/hybrid')])

    def rank(self) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        # Truth depth replaces the hybrid depth, so its rungs skip fusion (which would overwrite
        # the object pixels with VGGT or DA3 depth again).
        steps = (('ours', lambda: None, []), ('poses', lambda: self.truth_poses('poses'), TRUTH_CAMERAS),
                 ('depth', lambda: self.truth_depth('ours', 'depth'), ['FUSION=']),
                 ('poses_depth', lambda: self.truth_depth('poses', 'poses_depth'), TRUTH_CAMERAS + ['FUSION=']))
        results = {}
        for rung, prepare, extra in steps:
            if (self.root / rung / 'metrics' / f'{self.config["metrics"]}.json').is_file():
                results[rung] = self.metrics(rung)
                continue
            try:
                prepare()
                self.make(rung, ['reconstruct', self.config['evaluate']], extra)
                results[rung] = self.metrics(rung)
            except (subprocess.CalledProcessError, OSError, ValueError) as error:
                results[rung] = {'failed': f'{type(error).__name__}: see {self.root / rung}.log'}
        for rung in RUNGS:  # depth rungs are copies, so prune only once all have run
            self.prune(self.root / rung)
        return results

    def metrics(self, rung: str) -> dict:
        return json.loads((self.root / rung / 'metrics' / f'{self.config["metrics"]}.json').read_text())

    @staticmethod
    def prune(run: Path) -> None:
        if not run.is_dir():
            return
        for path in run.iterdir():
            if path.name == 'metrics':
                continue
            if path.name == 'mesh':
                for mesh in path.iterdir():
                    if mesh.name != 'object.ply':
                        mesh.unlink()
                continue
            shutil.rmtree(path) if path.is_dir() else path.unlink()


def summarize(results: dict, config: dict, root: Path) -> str:
    columns = config['columns']
    header = ['scene'] + [f'{rung} {title}' for rung in RUNGS for _, title in columns]
    rows = []
    for scene, rungs in sorted(results.items()):
        row = [scene]
        for rung in RUNGS:
            for key, _ in columns:
                value = rungs.get(rung, {}).get(key)
                row.append(float(value) if value is not None else float('nan'))
        rows.append(row)
    with (root / 'stage_ranking.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    key, title = columns[0]
    second_key, second_title = columns[1]
    lines = [f'| Scene | Ours {title} | Truth poses | Truth depth | Both | Ours {second_title} | '
             f'Truth poses | Truth depth | Both |', '|' + '---|' * 9]
    table = []
    for scene, rungs in sorted(results.items()):
        values = [rungs.get(rung, {}).get(name) for name in (key, second_key) for rung in RUNGS]
        table.append(values)
        lines.append(f'| {scene} | ' + ' | '.join('failed' if value is None else f'{value:.2f}'
                                                   for value in values) + ' |')
    array = np.array([[np.nan if value is None else value for value in row] for row in table], float)
    medians = np.nanmedian(array, axis=0)
    lines.append('| **median** | ' + ' | '.join(f'{value:.2f}' for value in medians) + ' |')
    ours, poses, depth, both = medians[:4]
    lines += ['', f'Median {title} (mm): ours {ours:.2f}; with truth poses {poses:.2f} '
              f'({ours - poses:+.2f}); with truth depth {depth:.2f} ({ours - depth:+.2f}); '
              f'with both {both:.2f}, the floor left to fusion, extraction and cleanup.']
    text = '\n'.join(lines) + '\n'
    (root / 'stage_ranking.md').write_text(text)
    return text


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--scenes', nargs='+', required=True)
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--surfaces', type=Path, default=Path('runs/real/dtu_reference_surfaces'),
                   help='DTU reference surfaces, built here when missing')
    p.add_argument('--colmap', default='colmap')
    a = p.parse_args()
    config = DATASETS[a.dataset]
    a.root.mkdir(parents=True, exist_ok=True)
    scenes = [Scene(name, config, a.root, a.surfaces, a.colmap) for name in a.scenes]
    if a.dataset == 'dtu':
        for scene in scenes:
            if not scene.reference.is_file():
                subprocess.run([sys.executable, '-m', 'recon.evaluation.diagnostics.reference_surface',
                                '--reference', str(scene.data / 'reference'),
                                '--pose-dir', str(scene.data / 'pose'),
                                '--output', str(scene.reference)], check=True)
    results = {}
    with ThreadPoolExecutor(max(1, a.jobs)) as pool:
        futures = {pool.submit(scene.rank): scene.name for scene in scenes}
        for future in bar(as_completed(futures), total=len(futures), desc='Stage ranking', unit='scene'):
            results[futures[future]] = future.result()
            (a.root / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(summarize(results, config, a.root))
    done(f'Stage ranking: {len(results)} scenes, tables in {a.root}')


if __name__ == '__main__':
    main()
