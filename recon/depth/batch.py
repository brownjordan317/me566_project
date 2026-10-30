#!/usr/bin/env python3
"""Run the depth models' stages in one process, loading each model once and dropping it when nothing queued needs it.

Each depth stage (recon.depth.monocular.moge, recon.depth.multiview.vggt, recon.depth.multiview.da3) runs as its own
process in the pipeline, so every run pays Python start-up, CUDA initialisation and the model's weights loading
again - about 20 s of a scan's depth stages, against 30 s of inference (docs/PIPELINE.md). Here the stages run as
jobs in one process instead, model by model: every queued MoGe job, then every VGGT job, then every DA3 job.
A model is loaded for the first job that needs it, reused by every later job with the same weights, and dropped
- deleted and its CUDA cache released - after the last queued job that needs it, before the next model loads.
On a single scan that drops each model as soon as its stage is done; over a batch of scans it keeps each model
until the last scan that uses it. Each stage's code, licence and output are its own: the models share a process,
not a module.

Jobs come as --job STAGE 'ARGS' (the stage's own command line, as the Makefile writes it) or as JSON files of
[{"stage": "moge" | "vggt" | "da3", "argv": [...]}], and keep their order within a stage (MoGe's grounding job
reads the raw maps its inference job writes). --print emits the jobs as JSON instead of running them, which is
how a batch over many scans is assembled; --skip-done drops jobs whose stage summary already exists. Each
stage's summary records its own load_s and inference_s; this driver writes the whole batch's timings, load
separate from inference, to --report.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import shlex
import time
from pathlib import Path

STAGES = {'moge': 'recon.depth.monocular.moge', 'vggt': 'recon.depth.multiview.vggt',
          'da3': 'recon.depth.multiview.da3'}
# Arguments that decide which weights a stage loads; a job whose values differ cannot reuse the loaded model.
WEIGHTS = {'moge': ('--checkpoint', '--version'), 'vggt': ('--checkpoint-dir',),
           'da3': ('--checkpoint', '--head-weights')}
# MoGe jobs that reuse raw maps run no inference and need no model.
NO_MODEL = {'moge': '--reuse-raw-dir'}


def weights_key(stage: str, argv: list[str]) -> tuple:
    return tuple((flag, argv[argv.index(flag) + 1] if flag in argv else None) for flag in WEIGHTS[stage])


def needs_model(stage: str, argv: list[str]) -> bool:
    return NO_MODEL.get(stage) not in argv


def plan(jobs: list[dict]) -> list[tuple[str, list[dict]]]:
    """Group jobs model by model in pipeline order, keeping each stage's job order."""
    unknown = {job['stage'] for job in jobs} - set(STAGES)
    if unknown:
        raise ValueError(f'Unknown depth stage(s): {sorted(unknown)}')
    return [(stage, [job for job in jobs if job['stage'] == stage]) for stage in STAGES
            if any(job['stage'] == stage for job in jobs)]


def drop_points(stage: str, jobs: list[dict]) -> set[int]:
    """Indices of the jobs after which the loaded model is dropped: the last job needing each set of weights."""
    last = {}
    for index, job in enumerate(jobs):
        if needs_model(stage, job['argv']):
            last[weights_key(stage, job['argv'])] = index
    return set(last.values())


def summary_path(argv: list[str]) -> Path | None:
    """The summary a job's stage writes when it finishes, if it is there."""
    output = Path(argv[argv.index('--output-dir') + 1])
    return next((output / name for name in ('moge_inference.json', 'vggt_inference.json', 'candidate.json')
                 if (output / name).is_file()), None)


def release() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('files', type=Path, nargs='*', help='JSON lists of {"stage", "argv"}')
    p.add_argument('--job', nargs=2, action='append', default=[], metavar=('STAGE', 'ARGS'),
                   help="One job: the stage and its command line as one string")
    p.add_argument('--print', action='store_true', help='Print the jobs as JSON and exit')
    p.add_argument('--skip-done', action='store_true', help="Skip jobs whose stage summary already exists")
    p.add_argument('--report', type=Path, help='Write the batch timings here')
    a = p.parse_args()
    jobs = [job for path in a.files for job in json.loads(path.read_text())]
    jobs += [{'stage': stage, 'argv': shlex.split(args)} for stage, args in a.job]
    if a.print:
        print(json.dumps(jobs))
        return
    if a.skip_done:
        jobs = [job for job in jobs if summary_path(job['argv']) is None]
    started = time.perf_counter()
    report = {'jobs': len(jobs), 'stages': []}
    for stage, queue in plan(jobs):
        module = importlib.import_module(STAGES[stage])
        drops = drop_points(stage, queue)
        model, key, entry = None, None, {'stage': stage, 'loads': [], 'jobs': []}
        for index, job in enumerate(queue):
            argv = [str(x) for x in job['argv']]
            if needs_model(stage, argv) and model is not None and weights_key(stage, argv) != key:
                model = None
                release()
            result = module.main(argv, model)
            if needs_model(stage, argv):
                model, key = result, weights_key(stage, argv)
            summary = summary_path(argv)
            timing = json.loads(summary.read_text()) if summary else {}
            entry['jobs'].append({'output': argv[argv.index('--output-dir') + 1], 'load_s': timing.get('load_s'),
                                  'inference_s': timing.get('inference_s'), 'model_reused': timing.get('model_reused')})
            if timing.get('load_s'):
                entry['loads'].append(timing['load_s'])
            if index in drops:
                model = None
                release()
                entry['jobs'][-1]['dropped_model_after'] = True
        entry['load_s'] = sum(entry['loads'])
        entry['inference_s'] = sum(job['inference_s'] or 0. for job in entry['jobs'])
        report['stages'].append(entry)
        del module
    report['load_s'] = sum(stage['load_s'] for stage in report['stages'])
    report['inference_s'] = sum(stage['inference_s'] for stage in report['stages'])
    report['wall_s'] = time.perf_counter() - started
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(report, indent=2) + '\n')
    print(f"Depth batch: {len(jobs)} jobs, load {report['load_s']:.1f} s, inference {report['inference_s']:.1f} s, "
          f"wall {report['wall_s']:.1f} s")


if __name__ == '__main__':
    main()
