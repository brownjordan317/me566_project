"""Scoring alignment."""
import numpy as np


def test_refine_alignment_recovers_scale_and_shift():
    from recon.evaluation.metrics.mesh import refine_alignment
    rng = np.random.default_rng(0)
    reference = rng.uniform(-.05, .05, (20000, 3)) * [1, 1, .3]
    moved = reference[::2] * 1.02 + [.001, -.0005, .0008]
    aligned, scale = refine_alignment(moved, reference, (.01, .003))
    assert abs(scale - 1 / 1.02) < 2e-3
    np.testing.assert_allclose(aligned, reference[::2], atol=5e-4)


def test_pose_similarity_recovers_rotation_from_orientations(tmp_path):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from recon.evaluation.metrics.mesh import pose_similarity
    rng = np.random.default_rng(0)
    true_rotation = Rotation.from_euler('xyz', [10, -20, 30], degrees=True).as_matrix()
    scale, offset = 2.5, np.array([.3, -.1, .7])
    rows = ['frame,status,tx,ty,tz,qx,qy,qz,qw']
    (tmp_path / 'pose').mkdir()
    for frame in range(12):
        centre = rng.normal(size=3)
        orientation = Rotation.random(random_state=frame).as_matrix()
        reference = np.eye(4)
        reference[:3, :3] = true_rotation @ orientation
        reference[:3, 3] = scale * true_rotation @ centre + offset
        np.savetxt(tmp_path / 'pose' / f'{frame:06d}.txt', reference)
        q = Rotation.from_matrix(orientation).as_quat()
        rows.append(f'{frame},tracked,{centre[0]},{centre[1]},{centre[2]},{q[0]},{q[1]},{q[2]},{q[3]}')
    (tmp_path / 'trajectory.csv').write_text('\n'.join(rows) + '\n')
    center, target, rotation, fitted_scale, errors = pose_similarity(tmp_path / 'trajectory.csv', tmp_path / 'pose')
    assert np.allclose(rotation.T, true_rotation, atol=1e-9)
    assert abs(fitted_scale - scale) < 1e-9
    assert errors.max() < 1e-9
