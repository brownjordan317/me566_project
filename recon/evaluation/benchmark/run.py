#!/usr/bin/env python3
"""Run every benchmark scene through the Makefile with a live rich dashboard.

Scenes run in parallel (GPU stages share a lock in the Makefile). Each scene's
full output goes to <root>/<scene>.log; the dashboard shows each scene's
current stage, taken from the Makefile's '▶ <stage>' labels, and its elapsed
time. The summary table is printed at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

STAGE = re.compile(r'▶ (.+?)\x1b\[0m')


class Scene:
    def __init__(self, name: str):
        self.name, self.status, self.stage = name, 'queued', ''
        self.start = self.end = None

    def elapsed(self) -> str:
        if self.start is None:
            return ''
        seconds = (self.end or time.monotonic()) - self.start
        return f'{int(seconds // 60)}:{int(seconds % 60):02d}'


def run_scene(scene: Scene, root: Path, make_vars: list[str], evaluation: list[list[str]],
              scores_only: bool = False, append: bool = False) -> None:
    scene.status, scene.start = 'running', time.monotonic()
    log = root / f'{scene.name}.log'
    env = dict(os.environ, NO_PROGRESS='1')
    base = ['make', '--no-print-directory', '-j1', f'SCENE={scene.name}',
            f'RUN={root / scene.name}', *make_vars]
    # Scores need only the object mesh; reconstruct also builds texture, point clouds and the scene mesh.
    steps = evaluation if scores_only else [['reconstruct'], *evaluation]
    ok = True
    with log.open('a' if append else 'w') as stream:          # the depth batch started this log
        for targets in steps:
            process = subprocess.Popen(base + targets, stdout=stream, stderr=subprocess.STDOUT,
                                       env=env)
            while process.poll() is None:
                scene.stage = latest_stage(log) or scene.stage
                time.sleep(1)
            if process.returncode:
                ok = False
                break
    scene.stage = latest_stage(log) or scene.stage
    scene.status, scene.end = ('ok' if ok else 'failed'), time.monotonic()


def batch_depth(scenes: list[Scene], root: Path, make_vars: list[str], seq_root: Path, jobs: int) -> None:
    """Run every scene's depth models in one process, model by model, so each model loads once per benchmark.

    Poses come first, for all scenes in parallel (the models need them); then each scene's depth jobs are
    collected from `make depth-jobs` and run by one recon.depth.batch process, which keeps a model only until
    the last scene that needs it. The per-scene pipelines then find their depth already built."""
    env = dict(os.environ, NO_PROGRESS='1')

    def scene_make(scene: Scene, *targets: str, capture: bool = False):
        command = ['make', '--no-print-directory', '-j1', f'SCENE={scene.name}', f'RUN={root / scene.name}',
                   *make_vars, f'MASKS={seq_root / scene.name / "masks"}', *targets]
        if capture:
            return subprocess.run(command + ['-s'], capture_output=True, text=True, env=env)
        with (root / f'{scene.name}.log').open('w' if 'poses' in targets else 'a') as stream:
            return subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env)

    def poses(scene: Scene) -> bool:
        scene.status, scene.stage = 'running', 'Camera poses (before the depth batch)'
        return scene_make(scene, 'poses').returncode == 0

    with ThreadPoolExecutor(max(1, jobs)) as pool:
        ready = [scene for scene, ok in zip(scenes, pool.map(poses, scenes)) if ok]
    queued = []
    for scene in ready:
        result = scene_make(scene, 'depth-jobs', capture=True)
        if result.returncode == 0:
            queued += json.loads(result.stdout.strip().splitlines()[-1])
        scene.stage = 'Depth models (batched across scenes)'
    if not queued:
        return
    job_file = root / 'depth_jobs.json'
    job_file.write_text(json.dumps(queued))
    lock = next((v.split('=', 1)[1] for v in make_vars if v.startswith('GPU_LOCK=')), '/tmp/me566_gpu.lock')
    with (root / 'depth_batch.log').open('w') as stream:
        subprocess.run(['flock', lock, 'env', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', sys.executable, '-m',
                        'recon.depth.batch', str(job_file), '--skip-done', '--report', str(root / 'depth_batch.json')],
                       stdout=stream, stderr=subprocess.STDOUT, env=env)


def latest_stage(log: Path) -> str:
    try:
        text = log.read_text(errors='replace')[-20000:]
    except FileNotFoundError:
        return ''
    found = STAGE.findall(text)
    return found[-1] if found else ''


def dashboard(scenes: list[Scene], progress: Progress) -> Group:
    table = Table(expand=False)
    for column in ('Scene', 'Status', 'Stage', 'Time'):
        table.add_column(column)
    style = {'queued': 'dim', 'running': 'yellow', 'ok': 'green', 'failed': 'bold red'}
    for scene in scenes:
        table.add_row(scene.name, f'[{style[scene.status]}]{scene.status}[/]',
                      scene.stage if scene.status == 'running' else '', scene.elapsed())
    return Group(table, progress)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True, help='Benchmark output folder')
    p.add_argument('--scenes', nargs='+', required=True)
    p.add_argument('--seq-root', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--dtu', action='store_true', help='Score with DTU chamfer instead')
    p.add_argument('--scores-only', action='store_true',
                   help='Build only what scoring needs (no texture, point clouds or scene mesh)')
    p.add_argument('--depth-batch', action=argparse.BooleanOptionalAction, default=True,
                   help='Run all scenes\' depth models in one process first, each model loaded once '
                        '(recon.depth.batch); off runs them inside each scene\'s pipeline')
    p.add_argument('make_vars', nargs='*', help='Extra NAME=value settings passed to make')
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    make_vars = [*a.make_vars, f'SEQ_ROOT={a.seq_root}']
    evaluation = [['evaluate-dtu']] if a.dtu else [['evaluate']]
    scenes = [Scene(name) for name in a.scenes]
    console = Console()
    progress = Progress(TextColumn('[bold]Benchmark'), BarColumn(), MofNCompleteColumn(),
                        TimeElapsedColumn(), console=console)
    task = progress.add_task('scenes', total=len(scenes))

    def work(scene: Scene) -> None:
        masks = f'MASKS={a.seq_root / scene.name / "masks"}'
        run_scene(scene, a.root, [*make_vars, masks], evaluation, a.scores_only, append=a.depth_batch)
        progress.advance(task)

    with Live(dashboard(scenes, progress), console=console, refresh_per_second=2) as live:
        stop = threading.Event()

        def refresh() -> None:
            while not stop.is_set():
                live.update(dashboard(scenes, progress))
                time.sleep(.5)

        updater = threading.Thread(target=refresh, daemon=True)
        updater.start()
        if a.depth_batch:
            batch_depth(scenes, a.root, make_vars, a.seq_root, a.jobs)
        with ThreadPoolExecutor(max(1, a.jobs)) as pool:
            list(pool.map(work, scenes))
        stop.set()
        updater.join()
        live.update(dashboard(scenes, progress))
    console.print()
    failed = [scene.name for scene in scenes if scene.status != 'ok']
    metrics = 'object_dtu' if a.dtu else 'object'
    subprocess.run([sys.executable, '-m', 'recon.evaluation.benchmark.summarize', str(a.root),
                    '--metrics', metrics], stdout=subprocess.DEVNULL, check=True)
    console.rule('[bold]Results')
    console.print(Markdown((a.root / f'summary_{metrics}.md').read_text()))
    if failed:
        console.print(f'[bold red]Failed:[/] {", ".join(failed)} (see the .log files in {a.root})')
        sys.exit(1)


if __name__ == '__main__':
    main()
