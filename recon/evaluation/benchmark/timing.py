#!/usr/bin/env python3
"""Per-stage run time and peak GPU memory of finished runs.

Every Makefile stage appends "<unix time>\\t<label>" to RUN/timing.log as it
starts, so a stage lasts until the next one starts. Reconstruction ends where
scoring begins (the first "Scoring"/"DTU chamfer" stage), so time is reported
for reconstruction only. With --gpu-log (lines "<unix time>\\t<GiB in use>",
sampled while the runs went one at a time), each stage also gets its peak device
memory above the idle level (the lowest sample, i.e. the desktop's share).

    python3 -m recon.evaluation.benchmark.timing runs/real/mobilebrick_profile \\
        runs/real/dtu_profile --gpu-log runs/real/profile_gpu.tsv
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

SCORING = ('Scoring mesh', 'MobileBrick official score', 'DTU chamfer', 'Scoring poses', 'Reprojection check')


def stages(timing: Path) -> list[tuple[str, float, float]]:
    """(label, start, end) of each reconstruction stage of one run, in order."""
    rows = [line.split('\t') for line in timing.read_text().splitlines() if '\t' in line]
    rows = [(float(time), label) for time, label in rows]
    result = []
    for (start, label), (end, _) in zip(rows, rows[1:]):
        if label in SCORING:
            break
        result.append((label, start, end))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('sweeps', type=Path, nargs='+')
    p.add_argument('--gpu-log', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    samples = None
    if a.gpu_log and a.gpu_log.is_file():
        samples = np.loadtxt(a.gpu_log, ndmin=2)
        samples = samples[np.argsort(samples[:, 0])]
        idle = float(samples[:, 1].min())
    durations, peaks, totals = defaultdict(list), defaultdict(list), []
    for sweep in a.sweeps:
        for timing in sorted(sweep.glob('*/timing.log')):
            run = stages(timing)
            if not run:
                continue
            totals.append((sweep.name, timing.parent.name, run[-1][2] - run[0][1]))
            # A stage that runs twice (e.g. two exports) is summed per run.
            per_run = defaultdict(float)
            per_peak = defaultdict(float)
            for label, start, end in run:
                per_run[label] += end - start
                if samples is not None:
                    window = samples[(samples[:, 0] >= start) & (samples[:, 0] < end), 1]
                    if len(window):
                        per_peak[label] = max(per_peak[label], float(window.max()) - idle)
            for label, value in per_run.items():
                durations[label].append(value)
                if label in per_peak:
                    peaks[label].append(per_peak[label])
    order = list(dict.fromkeys(label for sweep in a.sweeps for timing in sorted(sweep.glob('*/timing.log'))
                               for label, *_ in stages(timing)))
    lines = ['| Stage | Median s | Mean s | Share | Peak GPU GiB |', '|---|---|---|---|---|']
    whole = sum(np.mean(values) for values in durations.values())
    for label in order:
        values = durations[label]
        peak = f'{max(peaks[label]):.1f}' if peaks.get(label) else '—'
        lines.append(f'| {label} | {np.median(values):.1f} | {np.mean(values):.1f} | '
                     f'{100 * np.mean(values) / whole:.0f}% | {peak} |')
    lines += ['', '| Run | Scene | Reconstruction s |', '|---|---|---|']
    lines += [f'| {sweep} | {scene} | {total:.0f} |' for sweep, scene, total in totals]
    seconds = np.array([total for *_, total in totals])
    lines += ['', f'Reconstruction per scene: median {np.median(seconds):.0f} s, '
              f'mean {seconds.mean():.0f} s, range {seconds.min():.0f}–{seconds.max():.0f} s over {len(seconds)} scenes.']
    if samples is not None:
        lines.append(f'Peak GPU memory above idle ({idle:.1f} GiB): {float(samples[:, 1].max()) - idle:.1f} GiB.')
    text = '\n'.join(lines) + '\n'
    if a.output:
        a.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
