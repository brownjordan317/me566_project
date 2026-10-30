"""The learned judge: evidence for any set of candidates, a set network, and fusion without lumps."""
import numpy as np
import torch

from recon.depth.judge.corrupt import character, damage
from recon.depth.judge.apply import fuse
from recon.depth.judge.evidence import CROSS_FEATURES, combine_pairs
from recon.depth.judge.network import JudgeNet
from recon.depth.judge.train import CANDIDATE_FEATURES, loss_terms


def random_set(C=4, h=6, w=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    depth = 1 + .01 * torch.rand(1, C, h, w, generator=g)
    depth[0, 0, :2] = 0
    pairs = torch.rand(1, 2, C, C, h, w, generator=g)
    return depth, pairs


def test_cross_evidence_follows_its_candidate_when_reordered():
    depth, pairs = random_set()
    order = torch.tensor([2, 0, 3, 1])
    cross = combine_pairs(depth, pairs)
    shuffled = combine_pairs(depth[:, order], pairs[:, :, order][:, :, :, order])
    assert cross.shape[2] == len(CROSS_FEATURES)
    assert torch.allclose(cross[:, order], shuffled, atol=1e-6)


def test_a_left_out_candidate_leaves_no_trace_in_the_others():
    depth, pairs = random_set()
    kept = [0, 1, 2]
    alone = combine_pairs(depth[:, kept], pairs[:, :, kept][:, :, :, kept])
    changed = depth.clone()
    changed[:, 3] *= 1.5
    pairs_changed = pairs.clone()
    pairs_changed[:, :, :, 3] = 0
    with_other = combine_pairs(changed, pairs_changed)[:, kept]
    assert not torch.allclose(alone, with_other)        # the full set sees candidate 3...
    again = combine_pairs(changed[:, kept], pairs_changed[:, :, kept][:, :, :, kept])
    assert torch.allclose(alone, again)                  # ...a set without it does not


def test_judge_scores_any_number_of_candidates_in_any_order():
    torch.manual_seed(0)
    net = JudgeNet(CANDIDATE_FEATURES, 6).eval()
    for C in (1, 3, 5):
        x = torch.randn(1, C, CANDIDATE_FEATURES, 8, 8)
        image = torch.randn(1, 6, 8, 8)
        present = torch.ones(1, C, 8, 8, dtype=torch.bool)
        present[0, 0, 0, 0] = False
        logits, confidence = net(x, image, present)
        assert logits.shape == (1, C + 1, 8, 8) and confidence.shape == (1, 8, 8)
        assert torch.isinf(logits[0, 0, 0, 0])
        order = torch.randperm(C)
        shuffled, _ = net(x[:, order], image, present[:, order])
        assert torch.allclose(logits[:, :-1][:, order], shuffled[:, :-1], atol=1e-4)


def test_fusion_never_averages_two_surfaces():
    depth = np.stack([np.full((2, 2), 1.), np.full((2, 2), 1.001), np.full((2, 2), 1.2)]).astype(np.float32)
    probability = np.stack([np.full((2, 2), .4), np.full((2, 2), .2), np.full((2, 2), .35), np.full((2, 2), .05)])
    fused, nothing = fuse(depth, probability, .003)
    assert np.allclose(fused, (.4 * 1 + .2 * 1.001) / .6) and not nothing.any()
    probability[-1] = .9
    fused, nothing = fuse(depth, probability, .003)
    assert (fused == 0).all() and nothing.all()


def test_abstaining_is_the_target_only_when_every_candidate_is_far_off():
    error = torch.tensor([[[[.2]], [[5.]]], [[[4.]], [[6.]]]])          # (B=2, C=2, 1, 1) mm
    truth = torch.ones(2, 1, 1)
    abstain = torch.tensor([0., 0., 5.]).reshape(1, 3, 1, 1).repeat(2, 1, 1, 1)
    pick_first = torch.tensor([5., 0., 0.]).reshape(1, 3, 1, 1).repeat(2, 1, 1, 1)
    confidence = torch.zeros(2, 1, 1)
    # Sample 0 should pick its 0.2 mm candidate, sample 1 (nothing within 3 mm) should abstain.
    _, a0 = loss_terms(abstain[:1], confidence[:1], error[:1], truth[:1], 3.)
    _, b0 = loss_terms(pick_first[:1], confidence[:1], error[:1], truth[:1], 3.)
    _, a1 = loss_terms(abstain[1:], confidence[1:], error[1:], truth[1:], 3.)
    _, b1 = loss_terms(pick_first[1:], confidence[1:], error[1:], truth[1:], 3.)
    assert b0['choice'] < a0['choice'] and a1['choice'] < b1['choice']


def test_synthetic_damage_stays_near_the_truth_and_keeps_the_background_mostly_empty():
    rng = np.random.default_rng(1)
    truth = np.zeros((240, 320), np.float32)
    truth[60:180, 80:240] = 2.
    photo = np.full((240, 320, 3), 128, np.uint8)
    K = np.array([[300., 0, 160], [0, 300., 120], [0, 0, 1]])
    for _ in range(10):
        out = damage(truth, photo, K, np.eye(3), np.zeros(3), 1., character(rng), rng)
        kept = (out > 0) & (truth > 0)
        assert kept.sum() > .3 * (truth > 0).sum()
        assert np.median(np.abs(out[kept] / truth[kept] - 1)) < .04
        assert (out[:40] == 0).all()


def test_abstaining_keeps_the_top_candidate_when_asked():
    from recon.depth.judge.apply import fuse
    depth = np.stack([np.full((2, 2), 1.), np.full((2, 2), 1.2)]).astype(np.float32)
    probability = np.stack([np.full((2, 2), .2), np.full((2, 2), .1), np.full((2, 2), .7)])
    empty, nothing = fuse(depth, probability, .003)
    kept, _ = fuse(depth, probability, .003, keep=True)
    assert nothing.all() and (empty == 0).all() and np.allclose(kept, 1.)


def test_pooled_scores_follow_the_region():
    from recon.depth.judge.apply import pool_scores
    probability = torch.zeros(1, 3, 5, 5)
    probability[0, 0] = 1                          # candidate 0 wins everywhere...
    probability[0, :, 2, 2] = torch.tensor([0., 1., 0.])   # ...except one pixel
    present = torch.ones(1, 2, 5, 5, dtype=torch.bool)
    pooled = pool_scores(probability, present, 3)
    assert pooled[0, :, 2, 2].argmax() == 0 and torch.allclose(pooled.sum(1), torch.ones(1, 5, 5))


def test_smoothness_counts_a_flip_but_not_across_an_image_edge():
    from recon.depth.judge.train import smoothness
    probability = torch.zeros(1, 3, 1, 4)
    probability[0, 0, :, :2], probability[0, 1, :, 2:] = 1, 1          # flips between columns 1 and 2
    present = torch.ones(1, 2, 1, 4, dtype=torch.bool)
    labelled = torch.ones(1, 1, 4, dtype=torch.bool)
    flat = torch.zeros(1, 6, 1, 4)
    edge = flat.clone()
    edge[0, :3, :, 2:] = 1
    assert smoothness(probability, present, flat, labelled) > 0.3
    assert smoothness(probability, present, edge, labelled) < 1e-3


def test_synthetic_reports_are_honest_uninformative_or_absent():
    from recon.depth.judge.train import synthetic_report
    truth = np.ones((40, 40), np.float32)
    depth = truth.copy()
    depth[:, 20:] *= 1.01                         # 10 mm off on the right at 1000 mm per unit
    modes = set()
    for seed in range(30):
        report = synthetic_report(depth, truth, 1000., np.random.default_rng(seed))
        if not report[0].any():
            modes.add('none')
        elif report[3][:, 25:].mean() > 5 * report[3][:, :15].mean():
            modes.add('honest')
        else:
            modes.add('uninformative')
    assert modes == {'none', 'honest', 'uninformative'}


def test_a_margin_keeps_the_anchor_unless_another_is_clearly_better():
    from recon.depth.judge.apply import fuse
    depth = np.stack([np.full((1, 3), 1.), np.full((1, 3), 1.1)]).astype(np.float32)
    probability = np.array([[[.9, .45, .2]], [[.1, .55, .8]], [[0., 0., 0.]]])   # anchor 0, then a near tie, then 1
    plain, _ = fuse(depth, probability, 0.)
    held, _ = fuse(depth, probability, 0., margin=.2)
    assert np.allclose(plain, [[1., 1.1, 1.1]]) and np.allclose(held, [[1., 1., 1.1]])


def test_photo_branch_and_appearance_reach_the_grid():
    from recon.depth.judge.network import PHOTO_SCALE, JudgeNet
    from recon.depth.judge.train import appearance_grid, photo_grid
    h, w = 12, 16
    photo = np.zeros((PHOTO_SCALE * h, PHOTO_SCALE * w, 3), np.uint8)
    photo[..., 0] = 255                                       # blue in BGR
    rgb = photo_grid(photo, (h, w))
    assert rgb.shape == (3, PHOTO_SCALE * h, PHOTO_SCALE * w) and rgb[2].min() == 255 and rgb[0].max() == 0
    assert photo_grid(np.zeros((50, 70, 3), np.uint8), (h, w)).shape == rgb.shape
    assert appearance_grid(np.ones((64, 5, 7), np.float16), (h, w)).shape == (64, h, w)
    net = JudgeNet(5, 6, width=32, photo_features=32)
    logits, confidence = net(torch.zeros(1, 3, 5, h, w), torch.zeros(1, 6, h, w), torch.ones(1, 3, h, w, dtype=torch.bool),
                             torch.from_numpy(rgb)[None].float() / 255)
    assert logits.shape == (1, 4, h, w) and confidence.shape == (1, h, w)
