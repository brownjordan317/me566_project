"""The GPU bake paints each texel with the photograph's colour where it sees the surface, and nothing hidden."""
import numpy as np
import pytest

from recon.surface.texture import bake_gpu, texel_faces


def test_texel_faces_cover_a_triangle_with_valid_barycentrics():
    uvs = np.array([[.1, .1], [.9, .1], [.1, .9]])
    ids, bary = texel_faces(np.array([[0, 1, 2]]), uvs, 64)
    covered = ids >= 0
    assert .3 < covered.mean() < .4                         # about half of the 0.8 x 0.8 corner square
    assert np.allclose(bary[covered].sum(1), 1, atol=1e-5) and (bary[covered] > -1e-5).all()


def quad(z, half=1.):
    """A square at depth z facing a camera at the origin, with a two-triangle UV atlas covering [0, 1]^2."""
    vertices = np.array([[-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uvs = np.array([[0., 0], [1, 0], [1, 1], [0, 1]])
    return vertices, faces, uvs


def camera(width=80, height=80, focal=60.):
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.]]), np.eye(4)


def test_bake_reads_the_photograph_where_the_surface_is_seen():
    vertices, faces, uvs = quad(3.)
    K, extrinsic = camera()
    v, u = np.mgrid[:80, :80]
    image = np.stack([u * 3, v * 3, np.full_like(u, 100)], -1).astype(np.uint8)   # a known gradient
    albedo = bake_gpu(vertices, faces, uvs, [image], [K], [extrinsic], 32, power=4.)
    painted = albedo.any(-1)
    assert painted.mean() > .9
    # Texel (x, y) is the point (-1 + 2x/31, -1 + 2y/31, 3), projecting to pixel 40 + 20 * coordinate; the
    # albedo comes back in Open3D's row order, so that texel sits in row 31 - y.
    y, x = 16, 8
    expected_u, expected_v = 40 + 20 * (-1 + 2 * x / 31), 40 + 20 * (-1 + 2 * y / 31)
    row = 31 - y
    assert abs(int(albedo[row, x, 0]) - 3 * expected_u) <= 3 and abs(int(albedo[row, x, 1]) - 3 * expected_v) <= 3


def test_hidden_surface_is_left_unpainted():
    front, faces, _ = quad(3., half=2.)                              # a large square in front ...
    back, _, uvs = quad(5.)                                          # ... hides this one entirely
    vertices = np.concatenate([back, front])
    all_faces = np.concatenate([faces, faces + 4])
    # Only the back square gets atlas space; the front one is collapsed into a corner texel.
    atlas = np.concatenate([uvs * .98 + .01, np.zeros((4, 2))])
    K, extrinsic = camera()
    image = np.full((80, 80, 3), 200, np.uint8)
    albedo = bake_gpu(vertices, all_faces, atlas, [image], [K], [extrinsic], 32, power=4.)
    assert not albedo[2:-2, 2:-2].any()


def test_padding_grows_charts_with_their_own_edge_colour():
    from recon.surface.texture import pad_charts
    albedo = np.zeros((16, 16, 3), np.uint8)
    albedo[4:8, 4:8] = (200, 100, 50)
    padded = pad_charts(albedo, 2)
    assert (padded[2:10, 2:10] == (200, 100, 50)).all()        # grown two texels, in the chart's colour
    assert not padded[:1].any() and not padded[11:].any()       # and no further
