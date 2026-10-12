"""The viewer loads meshes and clouds with their colours, and shades uncoloured ones."""
import numpy as np
import open3d as o3d

from recon.apps.view_ply import load


def test_loads_coloured_mesh_and_cloud_and_shades_plain_mesh(tmp_path):
    mesh = o3d.geometry.TriangleMesh.create_box()
    plain = tmp_path / 'plain.ply'
    o3d.io.write_triangle_mesh(str(plain), mesh)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile([[1., 0., 0.]], (len(mesh.vertices), 1)))
    o3d.io.write_triangle_mesh(str(tmp_path / 'box.ply'), mesh)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.random.default_rng(0).random((50, 3))))
    cloud.colors = o3d.utility.Vector3dVector(np.tile([[0., 0., 1.]], (50, 1)))
    o3d.io.write_point_cloud(str(tmp_path / 'cloud.ply'), cloud)
    box, summary = load(tmp_path / 'box.ply')
    assert summary == '12 faces' and np.allclose(np.asarray(box.vertex_colors), [1, 0, 0])
    points, summary = load(tmp_path / 'cloud.ply')
    assert summary == '50 points' and np.allclose(np.asarray(points.colors), [0, 0, 1])
    shaded, _ = load(plain)
    assert shaded.has_vertex_colors() and np.ptp(np.asarray(shaded.vertex_colors)) > 0
