"""Benchmark comparison and stage timing helpers."""
import json

from recon.evaluation.benchmark.compare import METRICS, arm_table, load
from recon.evaluation.benchmark.timing import stages


def write_scene(root, scene, chamfer):
    metrics = root / scene / 'metrics'
    metrics.mkdir(parents=True)
    (metrics / 'object_dtu.json').write_text(json.dumps(
        {'chamfer_mm': chamfer, 'accuracy_mm': chamfer, 'completeness_mm': chamfer, 'shape_chamfer_mm': chamfer}))
    (metrics / 'pose_auc.json').write_text(json.dumps({'auc_3': 99.}))


def test_arm_counts_scenes_better_and_worse(tmp_path):
    for index in range(8):
        write_scene(tmp_path / 'default', f's{index}', 1.)
        write_scene(tmp_path / 'arm', f's{index}', .9 if index < 6 else 1.1)
    reference, arm = load(tmp_path / 'default', METRICS['dtu']), load(tmp_path / 'arm', METRICS['dtu'])
    row = arm_table(reference, {'change': arm}, METRICS['dtu'])[2]
    assert '6↑ 2↓' in row and row.startswith('| change | 8 |')


def test_stages_end_where_scoring_starts(tmp_path):
    log = tmp_path / 'timing.log'
    log.write_text('100\tCamera poses\n130\tMoGe depth\n150\tScoring mesh\n160\tScoring poses\n')
    assert stages(log) == [('Camera poses', 100., 130.), ('MoGe depth', 130., 150.)]
