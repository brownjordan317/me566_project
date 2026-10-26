"""UV rasterising and mirror filling behind the textured asset's provenance."""
import numpy as np

from recon.surface.texture import mirror_texels, texel_points


def unit_square():
    """Two triangles covering the whole atlas, lying in the z = 0 plane."""
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uvs = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    return vertices, faces, uvs


def test_texel_points_covers_the_atlas_and_places_points_on_the_surface():
    vertices, faces, uvs = unit_square()
    points, covered = texel_points(vertices, faces, uvs, 16)
    assert covered.mean() > .95                                  # two triangles tile the square
    assert np.allclose(points[covered][:, 2], 0.)                # the surface is the z = 0 plane
    assert points[covered][:, :2].min() >= -1e-5
    assert points[covered][:, :2].max() <= 1 + 1e-5


def test_texel_points_uses_the_albedo_row_order_not_a_flipped_one():
    # Here uv equals the vertex's own (x, y), so texel (col, row) must hold the point
    # (col, row) / (size - 1). A flipped v would mirror every point vertically instead.
    vertices, faces, uvs = unit_square()
    size = 16
    points, covered = texel_points(vertices, faces, uvs, size)
    rows, cols = np.nonzero(covered)
    expected = np.stack([cols / (size - 1), rows / (size - 1)], 1)
    assert np.abs(points[covered][:, :2] - expected).max() < .02


def test_texel_points_leaves_texels_no_triangle_covers_uncovered():
    vertices, faces, uvs = unit_square()
    uvs = uvs * .5                                               # charts now fill only a corner
    _, covered = texel_points(vertices, faces, uvs, 16)
    assert covered.any() and covered.mean() < .4
    assert not covered[-1, -1]


def test_texel_points_skips_a_degenerate_uv_triangle():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    faces = np.array([[0, 1, 2]])
    uvs = np.array([[0., 0.], [.5, 0.], [1., 0.]])               # zero area in UV space
    _, covered = texel_points(vertices, faces, uvs, 8)
    assert not covered.any()                                     # skipped rather than dividing by zero


def test_mirror_texels_fills_a_blank_half_from_the_painted_half():
    # A strip symmetric about x = 0; the left half is painted, the right half blank.
    size = 32
    xs = np.linspace(-1, 1, 4)
    vertices = np.array([[x, y, 0.] for x in xs for y in (0., 1.)])
    faces = []
    for i in range(len(xs) - 1):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        faces += [[a, b, c], [b, d, c]]
    faces = np.array(faces)
    uvs = np.stack([(vertices[:, 0] + 1) / 2, vertices[:, 1]], 1)
    albedo = np.zeros((size, size, 3), np.uint8)
    albedo[:, : size // 2] = 200                                 # the left half carries colour
    blank = albedo[..., :3].mean(-1) < 8
    filled, share = mirror_texels(albedo, blank, vertices, faces, uvs, 8.)
    assert share > .2                                            # a good part of the blank half is filled
    assert filled[size // 2, size // 2 + 4].mean() > 100         # and it took the painted colour


def test_mirror_texels_is_a_no_op_when_nothing_is_blank():
    vertices, faces, uvs = unit_square()
    albedo = np.full((16, 16, 3), 180, np.uint8)
    filled, share = mirror_texels(albedo, np.zeros((16, 16), bool), vertices, faces, uvs, 8.)
    assert share == 0. and np.array_equal(filled, albedo)
