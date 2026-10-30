"""Cross-scene DA3 head training: held-out sets, true-depth labels and the scale-invariant loss."""
import cv2
import numpy as np
import torch

from recon.depth.multiview.da3_distill import log_error, scaled_error_mm, split_scenes, true_depth


def test_whole_set_groups_are_held_out(tmp_path):
    for name in ['0013', '0013_0023', '0342', '0342_0100', '1447', '2022', '2022_0001']:
        (tmp_path / name / 'gt_depth').mkdir(parents=True)
    train, held = split_scenes(tmp_path, val_every=2)
    groups = lambda scenes: {path.name.split('_')[0] for path in scenes}
    assert groups(train).isdisjoint(groups(held)) and groups(held)
    assert len(train) + len(held) == 7


def test_true_depth_drops_depth_edges_and_background(tmp_path):
    depth = np.zeros((40, 40), np.uint16)
    depth[:, :20], depth[:, 20:30] = 300, 400                     # millimetres; columns 30+ are background
    (tmp_path / 'gt_depth').mkdir()
    cv2.imwrite(str(tmp_path / 'gt_depth' / '000000.png'), depth)
    target = true_depth(tmp_path, '000000', (40, 40))
    assert np.allclose(target[5, 5], .3) and np.allclose(target[5, 25], .4)
    assert target[5, 19] == 0 and target[5, 20] == 0 and target[5, 35] == 0
    assert np.allclose(target[5, 29], .4)                     # the silhouette keeps its true depth


def test_loss_ignores_a_global_scale():
    target = torch.rand(100) + .5
    assert log_error(torch.log(target * 3.7), target).max() < 1e-5
    assert log_error(torch.log(target), target * 1.1).max() < 1e-5


def test_scaled_error_is_zero_for_a_scaled_copy():
    target = np.random.default_rng(0).uniform(.3, .5, (20, 20)).astype(np.float32)
    assert scaled_error_mm(target * 2, target).max() < 1e-3
