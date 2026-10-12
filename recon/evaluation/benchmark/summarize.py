#!/usr/bin/env python3
"""Tabulate per-scene benchmark metrics written by `make benchmark`."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

DTU_COLUMNS = [
    ('scene', 'Scene'),
    ('pose_mm', 'Pose median (mm)'),
    ('chamfer_mm', 'Chamfer (mm)'),
    ('accuracy_mm', 'Accuracy (mm)'),
    ('completeness_mm', 'Completeness (mm)'),
    ('shape_chamfer_mm', 'Shape chamfer (mm)'),
    ('size_pct', 'Size error (%)'),
]
COLUMNS = [
    ('scene', 'Scene'),
    ('frames', 'Frames'),
    ('pose_mm', 'Pose median (mm)'),
    ('surface_mm', 'Surface median (mm)'),
    ('surface_p90_mm', 'Surface p90 (mm)'),
    ('coverage_pct', 'Coverage <=10 mm (%)'),
    ('fscore_1mm', 'F@1mm (%)'),
    ('fscore_2mm', 'F@2mm (%)'),
    ('recall_1mm', 'Recall@1mm (%)'),
    ('shape_mm', 'Shape median (mm)'),
    ('shape_p90_mm', 'Shape p90 (mm)'),
    ('size_pct', 'Size error (%)'),
    ('boundary_edges', 'Open edges'),
]


def scene_row(directory: Path, metrics: str = 'object') -> dict:
    metrics_path = directory / 'metrics' / f'{metrics}.json'
    if not metrics_path.is_file():
        return {'scene': directory.name, 'status': 'failed'}
    metrics = json.loads(metrics_path.read_text())
    if 'chamfer_mm' in metrics:
        return {'scene': directory.name, 'status': 'ok',
                'pose_mm': metrics['median_pose_error_mm'],
                'chamfer_mm': metrics['chamfer_mm'], 'accuracy_mm': metrics['accuracy_mm'],
                'completeness_mm': metrics['completeness_mm'],
                'shape_chamfer_mm': metrics.get('shape_chamfer_mm', float('nan')),
                'size_pct': metrics.get('size_error_pct', float('nan'))}
    return {'scene': directory.name, 'status': 'ok',
            'frames': metrics['localized_frames'],
            'pose_mm': metrics['median_pose_error_mm'],
            'surface_mm': metrics['mesh_to_reference_median_mm'],
            'surface_p90_mm': metrics['mesh_to_reference_p90_mm'],
            'coverage_pct': metrics['reference_vertices_within_10mm_pct'],
            'fscore_1mm': metrics.get('fscore_1mm_pct', float('nan')),
            'fscore_2mm': metrics.get('fscore_2mm_pct', float('nan')),
            'recall_1mm': metrics.get('recall_1mm_pct', float('nan')),
            'shape_mm': metrics.get('shape_median_mm', float('nan')),
            'shape_p90_mm': metrics.get('shape_p90_mm', float('nan')),
            'size_pct': metrics.get('size_error_pct', float('nan')),
            'boundary_edges': metrics['boundary_edges']}


def cell(value) -> str:
    if isinstance(value, float):
        return f'{value:.2f}'
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='Directory holding one output folder per scene')
    parser.add_argument('--metrics', default='object',
                        help='Which metrics/<name>.json to tabulate: object (default), '
                             'object_dtu, or a comparison arm such as moge')
    args = parser.parse_args()
    rows = [scene_row(path, args.metrics) for path in sorted(args.root.iterdir()) if path.is_dir()]
    if not rows:
        parser.error(f'No scene folders in {args.root}')
    done = [row for row in rows if row['status'] == 'ok']
    columns = DTU_COLUMNS if done and 'chamfer_mm' in done[0] else COLUMNS
    keys = [key for key, _ in columns]
    with (args.root / f'summary_{args.metrics}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, ['status', *keys])
        writer.writeheader()
        writer.writerows(rows)
    lines = ['| ' + ' | '.join(title for _, title in columns) + ' |',
             '|' + '---|' * len(columns)]
    for row in rows:
        if row['status'] != 'ok':
            lines.append(f"| {row['scene']} | failed |" + ' |' * (len(columns) - 2))
            continue
        lines.append('| ' + ' | '.join(cell(row[key]) for key in keys) + ' |')
    if done:
        medians = {key: float(np.median([row[key] for row in done])) for key in keys[1:]}
        lines.append('| **median** | ' + ' | '.join(cell(medians[key]) for key in keys[1:]) + ' |')
    table = '\n'.join(lines) + '\n'
    (args.root / f'summary_{args.metrics}.md').write_text(table)
    print(table)
    print(f'{len(done)}/{len(rows)} scenes evaluated; summary written to {args.root}')


if __name__ == '__main__':
    main()
