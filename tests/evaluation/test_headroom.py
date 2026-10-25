"""Visibility and mirror-plane helpers of the capture-headroom diagnostic."""
import numpy as np
import open3d as o3d

from recon.evaluation.diagnostics.headroom import mirror_plane, visibility


def scene_of(mesh):
    raycast = o3d.t.geometry.RaycastingScene()
    raycast.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return raycast


def test_visibility_hides_points_behind_a_wall():
    # Two parallel squares at z = 1 and z = 2; a camera at the origin sees only the near one.
    near = o3d.geometry.TriangleMesh.create_box(1., 1., .01).translate((-.5, -.5, 1.))
    far = o3d.geometry.TriangleMesh.create_box(1., 1., .01).translate((-.5, -.5, 2.))
    raycast = scene_of(near + far)
    points = np.array([[0., 0., 1.], [0., 0., 2.]])
    K = np.array([[100., 0, 50], [0, 100., 50], [0, 0, 1.]])
    seen = visibility(raycast, points, np.eye(4), K, 100, 100, tolerance=.01)
    assert seen[0] and not seen[1]


def test_visibility_excludes_points_outside_the_image():
    box = o3d.geometry.TriangleMesh.create_box(.1, .1, .01).translate((2., 0., 1.))
    K = np.array([[100., 0, 50], [0, 100., 50], [0, 0, 1.]])
    seen = visibility(scene_of(box), np.array([[2., 0., 1.]]), np.eye(4), K, 100, 100, tolerance=.01)
    assert not seen[0]                                   # projects far outside a 100 x 100 image


def test_mirror_plane_finds_the_symmetry_of_a_symmetric_mesh():
    box = o3d.geometry.TriangleMesh.create_box(2., 1., 1.).translate((-1., -.5, -.5))
    o3d.utility.random.seed(0)
    points = np.asarray(box.sample_points_uniformly(4000).points)
    axis, residual = mirror_plane(points, scene_of(box))
    # A box reflects onto itself; the residual left is the sampled centroid's offset, ~1% of the extent.
    assert np.median(residual) < .02 * 2.
    assert np.isclose(np.linalg.norm(axis), 1.)
    assert abs(axis[0]) > .99                            # the long axis, which is also the symmetry it finds
