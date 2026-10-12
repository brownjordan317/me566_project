"""32-bit PLY output and point-cloud fusion helpers."""
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from recon.common.ply import write_ply
from recon.surface.points import depth_normals, voxel_average


def test_ply_is_float32_with_uchar_colours_and_round_trips(tmp_path: Path):
    path = tmp_path / 'mesh.ply'
    points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9.25]])
    write_ply(path, points, np.array([[1, 0, .5]] * 3), np.array([[0, 0, 1.]] * 3),
              np.array([[0, 1, 2]]))
    header = path.read_bytes().split(b'end_header')[0].decode()
    assert 'property float x' in header and 'property uchar red' in header
    assert 'double' not in header
    mesh = o3d.io.read_triangle_mesh(str(path))
    np.testing.assert_allclose(np.asarray(mesh.vertices), points)
    np.testing.assert_allclose(np.asarray(mesh.vertex_colors)[0], [1, 0, 128 / 255])
    cloud_path = tmp_path / 'cloud.ply'
    write_ply(cloud_path, points, np.zeros((3, 3)))
    assert len(o3d.io.read_point_cloud(str(cloud_path)).points) == 3


def test_voxel_average_merges_points_sharing_a_voxel():
    points = torch.tensor([[.01, .01, .01], [.03, .03, .03], [1.5, 0, 0]])
    attributes = torch.tensor([[0.], [1.], [5.]])
    merged, values = voxel_average(points, attributes, .1)
    order = torch.argsort(merged[:, 0])
    torch.testing.assert_close(merged[order], torch.tensor([[.02, .02, .02], [1.5, 0, 0]]))
    torch.testing.assert_close(values[order], torch.tensor([[.5], [5.]]))


def test_depth_normals_face_the_camera():
    v, u = torch.meshgrid(torch.arange(8.), torch.arange(8.), indexing='ij')
    depth = 2 + .1 * u  # plane tilted about the y axis
    camera = torch.stack(((u - 4) * depth / 10, (v - 4) * depth / 10, depth), 2)
    normals = depth_normals(camera)[1:-1, 1:-1]
    assert ((normals * camera[1:-1, 1:-1]).sum(2) < 0).all()
    torch.testing.assert_close(normals.norm(dim=2), torch.ones(6, 6))
