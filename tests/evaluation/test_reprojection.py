from pathlib import Path
from recon.evaluation.metrics.reprojection import audit


def test_projection_with_nonzero_camera_translation(tmp_path: Path):
    (tmp_path / 'cameras.txt').write_text('1 PINHOLE 100 100 50 50 50 50\n')
    # World point (1, 0, 5), camera translated +1 in X: projection is (50, 50).
    (tmp_path / 'images.txt').write_text('1 1 0 0 0 -1 0 0 1 00000000.png\n50 50 1\n')
    (tmp_path / 'points3D.txt').write_text('1 1 0 5 255 255 255 0 1 0\n')
    report = audit(tmp_path)
    assert report['observations'] == 1
    assert report['median_px'] == 0
