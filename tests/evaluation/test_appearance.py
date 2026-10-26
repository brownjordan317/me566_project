"""The appearance report's geometry half, which needs no texture or ground truth."""
import numpy as np
import open3d as o3d

from recon.evaluation.diagnostics.appearance import geometry_report


def test_a_closed_box_reads_as_closed():
    report = geometry_report(o3d.geometry.TriangleMesh.create_box(1., 1., 1.))
    assert report['closed'] and report['boundary_edges'] == 0
    assert report['pieces'] == 1 and report['stray_faces_pct'] == 0.
    assert report['edge_manifold'] and report['vertex_manifold']


def test_a_box_with_a_face_removed_reads_as_open():
    box = o3d.geometry.TriangleMesh.create_box(1., 1., 1.)
    drop = np.zeros(len(box.triangles), bool); drop[:2] = True       # one square side
    box.remove_triangles_by_mask(drop)
    box.remove_unreferenced_vertices()
    report = geometry_report(box)
    assert not report['closed'] and report['boundary_edges'] > 0


def test_stray_pieces_are_reported_as_a_share_of_faces():
    big = o3d.geometry.TriangleMesh.create_box(1., 1., 1.).subdivide_midpoint(2)
    speck = o3d.geometry.TriangleMesh.create_box(.05, .05, .05).translate((9., 9., 9.))
    report = geometry_report(big + speck)
    assert report['pieces'] == 2
    assert 0 < report['stray_faces_pct'] < 10                        # the speck is a small minority


def test_roughness_is_low_for_a_smooth_sphere_and_higher_for_a_box():
    sphere = geometry_report(o3d.geometry.TriangleMesh.create_sphere(1., resolution=30))
    box = geometry_report(o3d.geometry.TriangleMesh.create_box(1., 1., 1.))
    assert sphere['roughness'] < box['roughness']                    # a box's 90-degree edges dominate
    assert sphere['roughness'] >= 0.
