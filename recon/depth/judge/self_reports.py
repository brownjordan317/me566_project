#!/usr/bin/env python3
"""Add the candidates' self-reports (recon.depth.judge.evidence.self_report) to judge data.

For each judge-data folder (recon.depth.judge.dataset output), every real candidate's <frame>_selfcheck.npz
under <run>/depth/<name>/ is sampled on the folder's evidence grid and written as <frame>_self.npy
(C, SELF_FEATURES, h, w) in the candidates' order; synthetic candidates and sources without a self-check
get zeros (their training-time reports are made in recon.depth.judge.train). The dataset writes these
itself; this adds them to data written before self-reports existed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recon.common.progress import bar, done
from recon.depth.judge.evidence import SELF_FEATURES, grid, self_report


def write_self_reports(folder: Path, sources: dict[str, Path], height: int, width: int) -> int:
    """Write <frame>_self.npy for every frame of one judge-data folder; returns how many candidates report."""
    meta = json.loads((folder / 'meta.json').read_text())
    rows, cols = grid(height, width, meta['stride'])
    reporting = 0
    for number in meta['frames']:
        out = np.zeros((len(meta['names']), len(SELF_FEATURES), len(rows), len(cols)), np.float32)
        for c, name in enumerate(meta['names']):
            if name in sources:
                out[c] = self_report(sources[name], number, rows, cols)
        reporting = max(reporting, int(out[:, 0].any((1, 2)).sum()))
        np.save(folder / f'{number:08d}_self.npy', out.astype(np.float16))
    return reporting


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('folders', type=Path, nargs='+', help='Judge-data folders (or one folder of them)')
    a = p.parse_args()
    folders = [f for f in a.folders if (f / 'meta.json').is_file()] or \
              sorted(f for f in a.folders[0].iterdir() if (f / 'meta.json').is_file())
    reporting = {}
    for folder in bar(folders, desc='Self-reports', unit='scene'):
        meta = json.loads((folder / 'meta.json').read_text())
        run = Path(meta['run'])
        sources = {n: run / 'depth' / n for n in meta['names'] if meta['kinds'][n] == 'real'}
        first = next(iter(np.load(run / 'depth' / n / f"{meta['frames'][0]:08d}_grounded.npy").shape
                          for n in sources if (run / 'depth' / n / f"{meta['frames'][0]:08d}_grounded.npy").is_file()))
        reporting[folder.name] = write_self_reports(folder, sources, *first)
    done(f'Self-reports: {len(folders)} scenes, {sum(v > 0 for v in reporting.values())} with a reporting source')


if __name__ == '__main__':
    main()
