"""The 3D cut must follow the Gaussians' labels and keep only the object's piece."""
import numpy as np
import open3d as o3d

from recon.surface.segment import cut, vertex_scores


def two_boxes():
    """An object box and a separate table slab, as one mesh."""
    object_box = o3d.geometry.TriangleMesh.create_box(1., 1., 1.)
    table = o3d.geometry.TriangleMesh.create_box(4., 4., .1).translate((-1.5, -1.5, -1.))
    return object_box + table


def test_each_vertex_takes_its_nearest_gaussians_label():
    means = np.array([[0., 0., 0.], [10., 0., 0.]])
    foreground = np.array([1., 0.])
    vertices = np.array([[.1, 0., 0.], [9.9, 0., 0.]])
    for k in (1, 2):
        score = vertex_scores(vertices, means, foreground, k)
        assert score[0] > .5 > score[1], k


def test_one_neighbour_needs_no_averaging():
    # cKDTree returns flat arrays at k=1; the score must still be one value per vertex.
    means = np.random.default_rng(0).normal(size=(50, 3))
    foreground = (means[:, 0] > 0).astype(float)
    score = vertex_scores(means + 1e-6, means, foreground, 1)
    assert score.shape == (50,)
    assert np.array_equal(score, foreground)


def test_the_cut_drops_the_table_and_keeps_the_object():
    mesh = two_boxes()
    vertices = np.asarray(mesh.vertices)
    on_object = vertices[:, 2] > -.5
    cut(mesh, on_object, largest=True)
    kept = np.asarray(mesh.vertices)
    assert len(kept) and (kept[:, 2] > -.5).all()


def test_largest_drops_a_stray_piece_the_labels_kept():
    mesh = two_boxes()
    keep = np.ones(len(mesh.vertices), bool)          # labels kept both pieces
    faces_before = len(mesh.triangles)
    cut(mesh, keep, largest=True)
    assert 0 < len(mesh.triangles) < faces_before     # only one of the two boxes survives
