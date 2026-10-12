"""Benchmark summary tables."""
import json
from pathlib import Path

from recon.evaluation.benchmark.summarize import main as summarize_main


def test_summary_marks_failed_scenes_and_reports_medians(tmp_path: Path, monkeypatch, capsys):
    for scene, surface in (('a', 1.), ('b', 3.)):
        folder = tmp_path / scene / 'metrics'
        folder.mkdir(parents=True)
        (folder / 'object.json').write_text(json.dumps({
            'localized_frames': 10, 'median_pose_error_mm': 2.,
            'mesh_to_reference_median_mm': surface, 'mesh_to_reference_p90_mm': 5.,
            'reference_vertices_within_10mm_pct': 90., 'boundary_edges': 100}))
    (tmp_path / 'c').mkdir()
    monkeypatch.setattr('sys.argv', ['summarize', str(tmp_path)])
    summarize_main()
    table = (tmp_path / 'summary_object.md').read_text()
    assert '| c | failed |' in table
    assert '| **median** | 10.00 | 2.00 | 2.00 |' in table
    assert '2/3 scenes evaluated' in capsys.readouterr().out
