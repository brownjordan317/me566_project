#!/usr/bin/env python3
"""Compare benchmark sweeps: run-to-run spread of the default and paired ablations.

    python3 -m recon.evaluation.benchmark.compare --dataset mobilebrick \
        --runs runs/real/mobilebrick_sweep_support runs/real/mobilebrick_sweep_repeat2 ... \
        --arm "no fusion=runs/real/mobilebrick_ablation_nofusion" ... --output table.md

The default is run several times (--runs; COLMAP's mapper is not deterministic),
which gives each metric's dataset mean as mean ± standard deviation over runs, and
a per-scene noise level: the median over scenes of the standard deviation across
runs. Each arm (one change to the default, run once) is compared scene by scene
with the first default run: mean, difference, scenes better / worse, and the
two-sided Wilcoxon signed-rank p-value of the paired differences (Wilcoxon, 1945;
scipy.stats.wilcoxon). A difference smaller than the run-to-run spread is noise
whatever its p-value; both are reported so the table can be read either way.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

# (file, key, label, higher is better)
METRICS = {
    'mobilebrick': [('object_official.json', 'f1_2.5mm', 'F1@2.5mm (official)', True),
                    ('object_official.json', 'f1_5mm', 'F1@5mm (official)', True),
                    ('object_official.json', 'chamfer_mm', 'Chamfer mm (official)', False),
                    ('object.json', 'fscore_1mm_pct', 'F@1mm', True),
                    ('object.json', 'shape_median_mm', 'Shape median mm', False),
                    ('pose_auc.json', 'auc_3', 'Pose AUC@3', True)],
    'dtu': [('object_dtu.json', 'chamfer_mm', 'Chamfer mm (official)', False),
            ('object_dtu.json', 'accuracy_mm', 'Accuracy mm', False),
            ('object_dtu.json', 'completeness_mm', 'Completeness mm', False),
            ('object_dtu.json', 'shape_chamfer_mm', 'Shape chamfer mm', False),
            ('pose_auc.json', 'auc_3', 'Pose AUC@3', True)],
}


def load(sweep: Path, metrics: list) -> dict[str, dict[str, float]]:
    """{scene: {label: value}} for the scenes that have every metric file."""
    scenes = {}
    for scene in sorted(path for path in sweep.iterdir() if path.is_dir()):
        values = {}
        for name, key, label, _ in metrics:
            path = scene / 'metrics' / name
            if path.is_file():
                value = json.loads(path.read_text()).get(key)
                if value is not None:
                    values[label] = float(value)
        if len(values) == len(metrics):
            scenes[scene.name] = values
    return scenes


def spread_table(runs: list[dict], metrics: list) -> list[str]:
    scenes = sorted(set.intersection(*(set(run) for run in runs)))
    lines = [f'Default pipeline, {len(runs)} runs over {len(scenes)} scenes common to all runs.', '',
             '| Metric | Mean ± sd over runs | Per-scene sd (median) |', '|---|---|---|']
    for _, _, label, _ in metrics:
        table = np.array([[run[scene][label] for scene in scenes] for run in runs])
        means = table.mean(axis=1)
        per_scene = np.median(table.std(axis=0, ddof=1)) if len(runs) > 1 else float('nan')
        sd = means.std(ddof=1) if len(runs) > 1 else float('nan')
        lines.append(f'| {label} | {means.mean():.3f} ± {sd:.3f} | {per_scene:.3f} |')
    return lines


def arm_table(reference: dict, arms: dict[str, dict], metrics: list) -> list[str]:
    lines = ['| Change | Scenes | ' + ' | '.join(label for _, _, label, _ in metrics) + ' |',
             '|---|---|' + '---|' * len(metrics)]
    for name, arm in arms.items():
        scenes = sorted(set(reference) & set(arm))
        cells = []
        for _, _, label, higher in metrics:
            a = np.array([arm[scene][label] for scene in scenes])
            b = np.array([reference[scene][label] for scene in scenes])
            difference = a - b
            better = int(np.sum(difference > 0) if higher else np.sum(difference < 0))
            worse = int(np.sum(difference < 0) if higher else np.sum(difference > 0))
            p = wilcoxon(a, b).pvalue if np.any(difference) and len(scenes) > 5 else float('nan')
            cells.append(f'{a.mean():.3f} ({difference.mean():+.3f}; {better}↑ {worse}↓; p={p:.2g})')
        lines.append(f'| {name} | {len(scenes)} | ' + ' | '.join(cells) + ' |')
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=METRICS, required=True)
    p.add_argument('--runs', type=Path, nargs='+', required=True, help='Repeated runs of the default')
    p.add_argument('--arm', action='append', default=[], help='name=sweep folder (one change each)')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    metrics = METRICS[a.dataset]
    runs = [load(path, metrics) for path in a.runs]
    arms = {}
    for item in a.arm:
        name, _, path = item.partition('=')
        arms[name] = load(Path(path), metrics)
    lines = spread_table(runs, metrics)
    if arms:
        lines += ['', f'Each change against default run 1 ({a.runs[0]}): mean (difference; scenes better/worse; '
                  'Wilcoxon p).', ''] + arm_table(runs[0], arms, metrics)
    text = '\n'.join(lines) + '\n'
    if a.output:
        a.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
