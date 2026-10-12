"""Progress output falls back to plain start/finish lines outside a terminal."""
import pytest

from recon.common.progress import bar, done, heading, spinner


def test_bar_logs_start_and_summary(monkeypatch, capsys):
    monkeypatch.setenv('NO_PROGRESS', '1')
    progress = bar(range(3), desc='Frames', unit='frame')
    progress.set_postfix(kept=3)
    assert list(progress) == [0, 1, 2]
    err = capsys.readouterr().err
    assert 'Frames: started (3 frame)' in err
    assert 'Frames: done, 3 frame in' in err and 'kept 3' in err


def test_manual_bar_counts_updates(monkeypatch, capsys):
    monkeypatch.setenv('NO_PROGRESS', '1')
    progress = bar(total=10, desc='Windows', unit='frame')
    progress.update(4)
    progress.update(6)
    progress.close()
    progress.close()
    assert capsys.readouterr().err.count('Windows: done, 10 frame') == 1


def test_spinner_reports_success_and_failure(monkeypatch, capsys):
    monkeypatch.setenv('NO_PROGRESS', '1')
    with spinner('Mapping'):
        pass
    with pytest.raises(RuntimeError), spinner('Broken'):
        raise RuntimeError('boom')
    err = capsys.readouterr().err
    assert 'Mapping: done in' in err and 'Broken: failed after' in err


def test_summary_and_heading_go_to_stderr(capsys):
    done('Fusion: 5 frames')
    heading('TSDF fusion', 'scene.ply')
    out, err = capsys.readouterr()
    assert out == ''
    assert 'Fusion: 5 frames' in err and '▶ TSDF fusion' in err and 'scene.ply' in err
