#!/usr/bin/env python3
"""Write docs/REPORT.md, the results report, from the finished runs.

    python3 -m recon.evaluation.benchmark.report            # or: make report

Every number comes from the run folders listed in docs/PIPELINE.md ("Runs and
data layout"), except the published results, which are constants below with
their sources. Sections whose runs are missing or unfinished are marked pending,
so the report can be regenerated at any point and fills in as runs complete.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from recon.evaluation.benchmark.compare import METRICS, arm_table, load, spread_table
from recon.evaluation.benchmark.timing import stages

RUNS = Path('runs/real')

# Published results, all on the standard splits and official protocols.
MOBILEBRICK_PUBLISHED = [  # method, poses, F1@2.5, F1@5, chamfer mm, time per object
    ('TSDF-Fusion (depth sensor)', 'given', 28.77, 53.39, 13.78, '—'),
    ('COLMAP MVS', 'given', 71.08, 88.71, 5.26, '—'),
    ('NeRF', 'given', 43.55, 73.45, 7.98, '>10 h'),
    ('NeuS', 'given', 73.74, 89.30, 4.74, '>10 h'),
    ('Voxurf', 'given', 75.13, 90.41, 4.71, '—'),
    ('Voxurf', 'ARKit', 69.18, 88.43, 5.30, '—'),
    ('Voxurf + PoRF', 'ARKit, refined', 75.67, 90.65, 4.67, '+2.5 h'),
]
DTU_PUBLISHED = [  # method, cameras, chamfer mm, time per scan
    ('3DGS', 'given', 1.96, '11 min'), ('NeRF', 'given', 1.49, '>12 h'), ('SuGaR', 'given', 1.33, '~1 h'),
    ('VolSDF', 'given', 0.86, '>12 h'), ('NeuS', 'given', 0.84, '>12 h'), ('2DGS', 'given', 0.80, '19 min'),
    ('GOF', 'given', 0.74, '1–2 h'), ('QGS', 'given', 0.73, '50 min'), ('Neuralangelo', 'given', 0.61, '>128 h'),
    ('PGSR', 'given', 0.53, '1 h'), ('DiGS', 'given', 0.46, '1.2 h'),
    ('Voxurf', 'COLMAP poses, true intrinsics', 3.48, '—'),
    ('Voxurf + PoRF', 'COLMAP poses refined, true intrinsics', 0.85, '+2.5 h'),
]
TNT_SCENES = ('Barn', 'Caterpillar', 'Courthouse', 'Ignatius', 'Meetingroom', 'Truck')
TNT_PUBLISHED = [  # method, per-scene F-score (TNT_SCENES order), published mean, time; all with COLMAP cameras
    ('3DGS', (.13, .08, .09, .04, .01, .19), .09, '14 min'), ('SuGaR', (.14, .16, .08, .33, .15, .26), .19, '>1 h'),
    ('2DGS', (.36, .23, .13, .44, .16, .26), .30, '34 min'), ('Geo-NeuS', (.33, .26, .12, .72, .20, .45), .35, '>24 h'),
    ('NeuS', (.29, .29, .17, .83, .24, .45), .38, '>24 h'), ('GOF', (.44, .41, .28, .68, .28, .58), .46, '—'),
    ('PGSR', (.66, .41, .21, .80, .29, .60), .50, '—'), ('Neuralangelo', (.70, .36, .28, .89, .32, .48), .50, '>24 h'),
    ('DiGS', (.68, .42, .25, .83, .28, .61), .51, '—'),
]  # means as published (2DGS's own table reports .30 although its rounded scene values average .26)
SOURCES = """Sources: MobileBrick (Li et al., CVPR 2023, Table 2); PoRF (Bian et al., ICLR 2024, Tables 2 and 4);
2DGS (Huang et al., SIGGRAPH 2024, Tables 1 and 2); QGS (Zhang et al., 2024); DiGS ("Accurate and Complete
Surface Reconstruction from 3D Gaussians via Direct SDF Learning", 2025, Tables I and II). Times are as reported
by each paper, on their hardware."""


def mean_json(pattern: str, key: str) -> tuple[float, int]:
    values = [json.loads(path.read_text()).get(key) for path in sorted(RUNS.glob(pattern))]
    values = [value for value in values if value is not None]
    return (float(np.mean(values)), len(values)) if values else (float('nan'), 0)


def complete(sweep: str, metric: str, scenes: int) -> bool:
    return len(list((RUNS / sweep).glob(f'*/metrics/{metric}'))) >= scenes


def pending(what: str) -> list[str]:
    return [f'*Pending: {what}.*', '']


def mobilebrick_section() -> list[str]:
    lines = ['## MobileBrick (18 test scenes, official protocol)', '',
             'Official protocol: the mesh is placed in the reference frame (ours by a similarity fitted to our '
             'estimated camera centres), rigid ICP, visibility crop, 100k samples each side; precision/recall/F1 at '
             '2.5 and 5 mm, chamfer as the sum of both directions. Ours estimates cameras and lens from the images; '
             '"scale removed" repeats the score after a scaled ICP, separating our RGB-only scale error from surface '
             'quality.', '',
             '| Method | Cameras | Masks | F1 @ 2.5 mm | F1 @ 5 mm | Chamfer mm | Time per object |',
             '|---|---|---|---|---|---|---|']
    for method, poses, f25, f5, chamfer, time in MOBILEBRICK_PUBLISHED:
        lines.append(f'| {method} | {poses} | dataset | {f25:.1f} | {f5:.1f} | {chamfer:.2f} | {time} |')
    latency = object_latency('mobilebrick_profile')
    for sweep, masks in (('mobilebrick_sweep_support', 'dataset'), ('mobilebrick_sweep_birefnet_support', 'BiRefNet (automatic)')):
        if not complete(sweep, 'object_official.json', 18):
            lines.append(f'| **Ours** | estimated | {masks} | pending | | | |')
            continue
        f25, _ = mean_json(f'{sweep}/*/metrics/object_official.json', 'f1_2.5mm')
        f5, _ = mean_json(f'{sweep}/*/metrics/object_official.json', 'f1_5mm')
        cd, _ = mean_json(f'{sweep}/*/metrics/object_official.json', 'chamfer_mm')
        s25, _ = mean_json(f'{sweep}/*/metrics/object_official.json', 'scaled_f1_2.5mm')
        s5, _ = mean_json(f'{sweep}/*/metrics/object_official.json', 'scaled_f1_5mm')
        scd, _ = mean_json(f'{sweep}/*/metrics/object_official.json', 'scaled_chamfer_mm')
        lines.append(f'| **Ours** | estimated | {masks} | **{f25:.1f}** | **{f5:.1f}** | **{cd:.2f}** | {latency} |')
        lines.append(f'| **Ours, scale removed** | estimated | {masks} | **{s25:.1f}** | **{s5:.1f}** | **{scd:.2f}** | {latency} |')
    for name, folder in (('VGGT feed-forward', 'mobilebrick_baseline_vggt'), ('2DGS on our cameras', 'mobilebrick_baseline_2dgs')):
        if complete(folder, 'object_official.json', 18):
            f25, _ = mean_json(f'{folder}/*/metrics/object_official.json', 'f1_2.5mm')
            f5, _ = mean_json(f'{folder}/*/metrics/object_official.json', 'f1_5mm')
            cd, _ = mean_json(f'{folder}/*/metrics/object_official.json', 'chamfer_mm')
            lines.append(f'| {name} (run here) | estimated | dataset | {f25:.1f} | {f5:.1f} | {cd:.2f} | {baseline_latency(folder)} |')
    return lines + ['']


def dtu_section() -> list[str]:
    lines = ['## DTU (15 scans, official protocol)', '',
             'Mean chamfer of the DTU evaluation (DTUeval-python port). Published methods use DTU\'s calibrated '
             'cameras; ours is shown with estimated cameras, with true intrinsics only, and with the true cameras.', '',
             '| Method | Cameras | Chamfer mm | Shape chamfer mm | Time per scan |', '|---|---|---|---|---|']
    for method, cameras, chamfer, time in DTU_PUBLISHED:
        lines.append(f'| {method} | {cameras} | {chamfer:.2f} | — | {time} |')
    latency = object_latency('dtu_profile')
    rows = (('estimated (no calibration)', 'dtu_sweep_support/*/metrics/object_dtu.json', 'chamfer_mm'),
            ('true intrinsics, our poses', 'dtu_sweep_truek/*/metrics/object_dtu.json', 'chamfer_mm'),
            ('true intrinsics, our poses, orientation alignment', 'dtu_sweep_truek/*/metrics/object_dtu.json', 'pose_aligned_chamfer_mm'),
            ('given (true intrinsics and poses)', 'dtu_sweep_truecams/*/cams/metrics/object_dtu.json', 'chamfer_mm'))
    for cameras, pattern, key in rows:
        chamfer, n = mean_json(pattern, key)
        shape, _ = mean_json(pattern, 'shape_chamfer_mm')
        lines.append(f'| **Ours** | {cameras} | ' + (f'**{chamfer:.3f}** | {shape:.3f} | {latency} |' if n == 15 else 'pending | | |'))
    for name, folder in (('VGGT feed-forward', 'dtu_baseline_vggt'), ('2DGS on our cameras', 'dtu_baseline_2dgs')):
        if complete(folder, 'object_dtu.json', 15):
            chamfer, _ = mean_json(f'{folder}/*/metrics/object_dtu.json', 'chamfer_mm')
            shape, _ = mean_json(f'{folder}/*/metrics/object_dtu.json', 'shape_chamfer_mm')
            lines.append(f'| {name} (run here) | estimated | {chamfer:.3f} | {shape:.3f} | {baseline_latency(folder)} |')
    lines += ['', 'Shape chamfer (after scaled ICP) is the same in all our camera settings, so the gap to '
              'the given-camera result is camera placement. DTU\'s principal point lies about 30 px from the image '
              'centre, which self-calibration from its narrow lens cannot recover; see PIPELINE.md, "Current results".', '']
    return lines


def tnt_section() -> list[str]:
    lines = ['## Tanks and Temples (6 training scenes, official protocol)', '',
             'F-score at each scene\'s threshold after the official trajectory alignment and ICP refinement. Every '
             'method here, ours included, uses cameras estimated from the images. Ours runs in scene mode (no masks, '
             'one shared focal) and is scored on its scene mesh.', '',
             '| Method | ' + ' | '.join(TNT_SCENES) + ' | Mean | Time per scene |', '|---|' + '---|' * (len(TNT_SCENES) + 2)]
    for method, scores, mean, time in TNT_PUBLISHED:
        lines.append(f'| {method} | ' + ' | '.join(f'{s:.2f}' for s in scores) + f' | {mean:.2f} | {time} |')
    ours, times = [], []
    for scene in TNT_SCENES:
        path = RUNS / 'tnt' / scene / 'metrics/tnt.json'
        ours.append(json.loads(path.read_text())['fscore'] / 100 if path.is_file() else None)
        latency = RUNS / 'tnt' / scene / 'latency.json'
        if latency.is_file():
            times.append(json.loads(latency.read_text())['wall_s'])
    cells = ' | '.join(f'**{s:.2f}**' if s is not None else 'pending' for s in ours)
    mean = f'**{np.mean([s for s in ours if s is not None]):.2f}**' if all(s is not None for s in ours) else 'pending'
    time = f'{np.median(times) / 60:.0f} min (median)' if times else 'pending'
    return lines + [f'| **Ours** | {cells} | {mean} | {time} |', '']


def object_latency(profile: str) -> str:
    totals = []
    for timing in sorted((RUNS / profile).glob('*/timing.log')):
        run = stages(timing)
        skip = sum(end - start for label, start, end in run
                   if label in ('Texture', 'TSDF fusion', 'Scene mesh cleanup', 'Point clouds'))
        totals.append(run[-1][2] - run[0][1] - skip)
    return f'{np.median(totals) / 60:.1f} min' if totals else 'pending'


def baseline_latency(folder: str) -> str:
    values = [json.loads(path.read_text())['wall_s'] for path in (RUNS / folder).glob('*/latency.json')]
    return f'{np.median(values) / 60:.1f} min' if values else '—'


def latency_section() -> list[str]:
    lines = ['## Latency', '',
             'Wall time per scene on one RTX 4090, one scene at a time, from the first stage to the finished '
             'object mesh (texture, scene mesh and point clouds excluded; listed separately). Measured with '
             '`recon/evaluation/benchmark/timing.py` from each run\'s `timing.log`.', '',
             '| Run | Scene | Frames | Object mesh s | Texture s | Total s |', '|---|---|---|---|---|---|']
    found = False
    for profile, data in (('mobilebrick_profile', 'mobilebrick_benchmark'), ('dtu_profile', 'dtu_benchmark')):
        for timing in sorted((RUNS / profile).glob('*/timing.log')):
            found = True
            run = stages(timing)
            durations = {}
            for label, start, end in run:
                durations[label] = durations.get(label, 0.) + end - start
            total = run[-1][2] - run[0][1]
            extra = sum(durations.get(label, 0.) for label in ('Texture', 'TSDF fusion', 'Scene mesh cleanup', 'Point clouds'))
            frames = len(list((RUNS / data / timing.parent.name / 'frames').glob('*.png')))
            lines.append(f'| {profile.split("_")[0]} | {timing.parent.name} | {frames} | {total - extra:.0f} | '
                         f'{durations.get("Texture", 0.):.0f} | {total:.0f} |')
    if not found:
        return lines[:2] + pending('timing runs')
    stage_totals = {}
    for profile in ('mobilebrick_profile', 'dtu_profile'):
        for timing in (RUNS / profile).glob('*/timing.log'):
            for label, start, end in stages(timing):
                stage_totals.setdefault(label, []).append(end - start)
    lines += ['', '| Stage | Median s per scene |', '|---|---|']
    for label, values in sorted(stage_totals.items(), key=lambda item: -np.median(item[1])):
        if np.median(values) >= 1:
            lines.append(f'| {label} | {np.median(values):.1f} |')
    lines += ['', 'Baselines run here, same GPU (median wall time per scene, including their meshing):', '',
              '| Method | MobileBrick | DTU |', '|---|---|---|',
              f'| VGGT feed-forward | {baseline_latency("mobilebrick_baseline_vggt")} | {baseline_latency("dtu_baseline_vggt")} |',
              f'| 2DGS (30k iterations) | {baseline_latency("mobilebrick_baseline_2dgs")} | {baseline_latency("dtu_baseline_2dgs")} |',
              f'| Ours (object mesh) | {object_latency("mobilebrick_profile")} | {object_latency("dtu_profile")} |', '']
    return lines


def pose_section() -> list[str]:
    lines = ['## Camera accuracy', '',
             'Relative-pose accuracy over all frame pairs (VGGT / PoseDiffusion protocol): AUC of the larger of '
             'rotation and translation-direction error up to 1°, 3° and 30°, and median errors.', '',
             '| Run | Cameras | AUC@1 | AUC@3 | AUC@30 | Rotation ° | Translation ° |', '|---|---|---|---|---|---|---|']
    for name, cameras, pattern in (('MobileBrick, ours', 'estimated', 'mobilebrick_sweep_support/*/metrics/pose_auc.json'),
                                   ('MobileBrick, VGGT', 'feed-forward', 'mobilebrick_baseline_vggt/*/metrics/pose_auc.json'),
                                   ('DTU, ours', 'estimated', 'dtu_sweep_support/*/metrics/pose_auc.json'),
                                   ('DTU, ours', 'true intrinsics', 'dtu_sweep_truek/*/metrics/pose_auc.json'),
                                   ('DTU, VGGT', 'feed-forward', 'dtu_baseline_vggt/*/metrics/pose_auc.json')):
        values = [mean_json(pattern, key) for key in ('auc_1', 'auc_3', 'auc_30', 'median_rotation_error_deg',
                                                       'median_translation_error_deg')]
        if values[0][1] == 0:
            lines.append(f'| {name} | {cameras} | pending | | | | |')
            continue
        lines.append(f'| {name} | {cameras} | ' + ' | '.join(f'{v:.2f}' for v, _ in values) + ' |')
    return lines + ['']


def ablation_section() -> list[str]:
    lines = ['## Ablations and run-to-run spread', '']
    for dataset, scenes, default in (('mobilebrick', 18, 'mobilebrick_sweep_support'), ('dtu', 15, 'dtu_sweep_support')):
        metrics = METRICS[dataset]
        runs = [RUNS / default] + [RUNS / f'{dataset}_sweep_repeat{r}' for r in (2, 3)]
        runs = [run for run in runs if run.is_dir() and len(load(run, metrics)) >= scenes]
        title = 'MobileBrick' if dataset == 'mobilebrick' else 'DTU'
        lines += [f'### {title}', '']
        if len(runs) < 3:
            lines += pending(f'{3 - len(runs)} more repeat run(s) of the default')
        else:
            lines += spread_table([load(run, metrics) for run in runs], metrics) + ['']
        arms = {}
        for name, folder in (('no fusion', 'nofusion'), ('no refinement', 'norefine'),
                             ('scene TSDF for the object', 'scenetsdf'), ('no cross-view check', 'nocheck')):
            path = RUNS / f'{dataset}_ablation_{folder}'
            if path.is_dir() and len(load(path, metrics)) >= scenes:
                arms[name] = load(path, metrics)
        if arms:
            lines += ['Each change against the default (run 1): mean (difference; scenes better/worse; Wilcoxon p).', '']
            lines += arm_table(load(RUNS / default, metrics), arms, metrics) + ['']
        if len(arms) < 4:
            lines += pending(f'{4 - len(arms)} of 4 ablations')
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, default=Path('docs/REPORT.md'))
    a = p.parse_args()
    lines = ['# Results report', '',
             f'Generated {date.today().isoformat()} by `make report` (`recon/evaluation/benchmark/report.py`) from '
             'the runs listed in [PIPELINE.md](PIPELINE.md); regenerate it after new runs instead of editing it. '
             'All numbers are ours unless marked published. Method details, conventions and the history behind '
             'each result are in PIPELINE.md.', '',
             '**Setting.** Our pipeline takes RGB images and object masks only: no calibration, no poses, no depth '
             'sensor. Published DTU and MobileBrick results are given the dataset\'s calibrated cameras (or ARKit '
             'poses); we show ours with estimated cameras and, on DTU, with the true cameras for a like-for-like '
             'comparison. On Tanks and Temples every method estimates its cameras.', '']
    lines += mobilebrick_section() + dtu_section() + tnt_section() + latency_section() + pose_section() + ablation_section()
    lines += [SOURCES, '']
    a.output.write_text('\n'.join(lines))
    print(f'Wrote {a.output}')


if __name__ == '__main__':
    main()
