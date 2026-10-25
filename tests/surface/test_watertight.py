"""Helpers of the watertight closure that need no GPU or libigl."""
import numpy as np
import open3d as o3d

from recon.surface.splat_views import fibonacci_directions, look_at
from recon.surface.watertight import finish, tidy_field


def test_tidy_field_keeps_the_largest_blob_and_drops_specks():
    field = np.zeros((20, 20, 20), bool)
    field[4:14, 4:14, 4:14] = True                      # the object
    field[18, 18, 18] = True                            # a speck that would become its own surface
    tidy = tidy_field(field)
    assert tidy[9, 9, 9] and not tidy[18, 18, 18]
    assert tidy.sum() >= 10 ** 3                        # closing may thicken, never delete


def test_tidy_field_closes_a_one_voxel_neck():
    field = np.zeros((12, 12, 12), bool)
    field[2:10, 2:10, 2:5] = True
    field[2:10, 2:10, 7:10] = True
    field[5, 5, 5:7] = True                             # a thin bridge between the two slabs
    assert tidy_field(field)[5, 5, 6]                    # kept as one component, not split away


def test_finish_keeps_only_the_largest_component():
    # Subdivided, so the two components differ in face count and "largest" is meaningful.
    big = o3d.geometry.TriangleMesh.create_box(1., 1., 1.).subdivide_midpoint(2)
    small = o3d.geometry.TriangleMesh.create_box(.1, .1, .1).translate((5., 5., 5.))
    mesh = finish(big + small, clean=True)
    assert len(mesh.triangles) == len(big.triangles)
    assert np.asarray(mesh.vertices).max() < 2.


def test_fibonacci_directions_are_unit_and_spread():
    directions = fibonacci_directions(32)
    assert directions.shape == (32, 3)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.)
    assert np.abs(directions.mean(0)).max() < .2        # roughly balanced over the sphere


def test_look_at_faces_the_target():
    pose = look_at(np.array([0., 0., -5.]), np.zeros(3))
    assert np.allclose(pose[:3, 3], [0., 0., -5.])
    assert np.allclose(pose[:3, 2], [0., 0., 1.])        # camera z points at the target
    assert np.allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-6)
