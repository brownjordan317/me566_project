"""Held-out frame selection and the camera split for judging a reconstruction."""
import argparse

import numpy as np

from recon.evaluation.diagnostics.heldout import held_out, split


def test_every_eighth_frame_from_the_offset():
    assert held_out(list(range(20)), 8, 4) == [4, 12]
    assert held_out([30, 10, 20, 0], 2, 0) == [0, 20]


def test_split_drops_held_out_images(tmp_path, monkeypatch):
    source = tmp_path / 'poses'
    (source / 'sparse_txt').mkdir(parents=True)
    (source / 'sparse_txt/cameras.txt').write_text('1 SIMPLE_PINHOLE 8 6 5 4 3\n')
    (source / 'sparse_txt/points3D.txt').write_text('')
    images = ''.join(f'{i + 1} 1 0 0 0 0 0 0 1 {i:08d}.png\n1.0 2.0 -1\n' for i in range(4))
    (source / 'sparse_txt/images.txt').write_text('# header\n' + images)
    (source / 'summary.json').write_text('{}')
    monkeypatch.setattr('recon.evaluation.diagnostics.heldout.subprocess.run', lambda *a, **k: None)
    run = tmp_path / 'run'
    split(argparse.Namespace(poses=source, run=run, sequence=tmp_path, every=2, offset=1))
    kept = (run / 'poses/sparse_txt/images.txt').read_text().split()[9::13]
    assert kept == ['00000000.png', '00000002.png']


def test_kept_few_holds_out_everything_but_a_spread_handful():
    from recon.evaluation.diagnostics.heldout import kept_few
    numbers = list(range(20))
    hold = kept_few(numbers, 4)
    kept = sorted(set(numbers) - set(hold))
    assert len(kept) == 4
    assert kept[0] == 0 and kept[-1] == 19                 # spans the sequence
    gaps = np.diff(kept)
    assert gaps.max() - gaps.min() <= 1                    # evenly spread


def test_kept_few_holds_out_nothing_when_asked_for_everything():
    from recon.evaluation.diagnostics.heldout import kept_few
    assert kept_few([3, 1, 2], 3) == []
    assert kept_few([3, 1, 2], 9) == []
