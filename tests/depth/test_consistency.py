"""Cross-view free-space votes."""


def test_consistency_votes_flag_points_a_view_sees_through():
    import torch
    from recon.depth.consistency import project_votes
    depth = torch.full((64, 64), 2.)
    K = torch.tensor([[100., 0, 32], [0, 100., 32], [0, 0, 1]])
    # Points on the optical axis: in front of, on, and behind the observed surface.
    points = torch.tensor([[0., 0, 1], [0, 0, 2], [0, 0, 3]])
    support, violation = project_votes(points, depth, K, torch.eye(3), torch.zeros(3), .03)
    assert support.tolist() == [False, True, False]
    # Only the nearer point lies in free space; the farther one is merely occluded.
    assert violation.tolist() == [True, False, False]
