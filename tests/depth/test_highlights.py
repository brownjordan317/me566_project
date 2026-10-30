"""A highlight seen in one view is removed; a bright material seen bright in every view is kept."""
import numpy as np
import torch

from recon.depth.fusion.highlights import consensus, neighbours


def test_a_view_dependent_highlight_takes_the_other_views_colour():
    own = torch.tensor([[.95, .95, .95]])                    # bright here ...
    samples = torch.tensor([[[.4, .2, .2]], [[.42, .21, .2]], [[.38, .2, .19]]])   # ... red everywhere else
    diffuse, count = consensus(own, samples, torch.ones(3, 1, dtype=torch.bool), .5)
    assert count.item() == 4 and diffuse[0, 0] < .5


def test_a_white_brick_is_white_in_every_view_and_stays_white():
    own = torch.tensor([[.92, .92, .92]])
    samples = torch.tensor([[[.9, .9, .9]], [[.93, .93, .93]], [[.91, .91, .91]]])
    diffuse, _ = consensus(own, samples, torch.ones(3, 1, dtype=torch.bool), .5)
    assert diffuse[0, 0] > .89


def test_missing_views_are_never_chosen():
    own = torch.tensor([[.5, .5, .5]])
    samples = torch.tensor([[[0., 0., 0.]], [[1., 1., 1.]]])
    diffuse, count = consensus(own, samples, torch.tensor([[False], [False]]), .5)
    assert count.item() == 1 and torch.allclose(diffuse[0], own[0])


def test_neighbours_are_close_and_facing_the_same_way():
    angle = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    centres = np.stack([np.cos(angle), np.sin(angle), np.full_like(angle, .5)], 1)
    axes = -centres / np.linalg.norm(centres, axis=1, keepdims=True)           # all looking at the origin
    near = neighbours(centres, axes, 2)
    assert sorted(near[0]) == [1, 11]                                         # the two adjacent cameras
    assert all(6 not in row for row in near[:1])                              # never the one opposite
