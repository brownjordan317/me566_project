"""Helpers of the planar-Gaussian refinement that run without gsplat."""
import numpy as np
import torch

from recon.depth.gaussian_refine import depth_normals, quaternion_from_normal, rotation_matrices, ssim


def rotate_z(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def test_quaternions_turn_the_z_axis_onto_each_normal():
    normals = np.array([[0, 0, 1.], [0, 0, -1.], [1, 0, 0.], [0, .6, .8]])
    for normal, quaternion in zip(normals, quaternion_from_normal(normals)):
        assert np.allclose(np.linalg.norm(quaternion), 1)
        assert np.allclose(rotate_z(quaternion), normal, atol=1e-6)


def test_ssim_is_one_for_identical_images_and_lower_for_noise():
    image = torch.rand(3, 32, 32)
    assert torch.isclose(ssim(image, image), torch.tensor(1.), atol=1e-4)
    assert ssim(image, torch.rand(3, 32, 32)) < .5


def test_rotation_matrices_match_the_quaternion_z_axis():
    normals = np.array([[0, 0, 1.], [1, 0, 0.], [0, .6, .8]])
    rotation = rotation_matrices(torch.tensor(quaternion_from_normal(normals), dtype=torch.float32))
    assert torch.allclose(rotation @ rotation.transpose(1, 2), torch.eye(3).expand(3, 3, 3), atol=1e-5)
    assert torch.allclose(rotation[:, :, 2], torch.tensor(normals, dtype=torch.float32), atol=1e-5)


def test_depth_normals_of_a_tilted_plane_face_the_camera():
    K = torch.tensor([[100., 0, 32], [0, 100., 24], [0, 0, 1]])
    normal = torch.tensor([0., .6, -.8])                     # plane n·X = -4, seen from the origin
    v, u = torch.meshgrid(torch.arange(48.), torch.arange(64.), indexing='ij')
    ray = torch.stack(((u - 32) / 100, (v - 24) / 100, torch.ones_like(u)), -1)
    depth = -4 / (ray @ normal)
    estimated = depth_normals(depth, K)[5:-5, 5:-5]
    assert torch.allclose(estimated, normal.expand_as(estimated), atol=1e-4)

