"""COLMAP dense-map files and the seed normals for PatchMatch."""
import numpy as np

from recon.depth.multiview.patchmatch import depth_normals, read_array, write_array


def test_dense_map_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    depth = rng.uniform(1, 2, (5, 7)).astype(np.float32)
    normals = rng.normal(size=(5, 7, 3)).astype(np.float32)
    write_array(tmp_path / 'd.bin', depth)
    write_array(tmp_path / 'n.bin', normals)
    assert np.array_equal(read_array(tmp_path / 'd.bin'), depth)
    assert np.array_equal(read_array(tmp_path / 'n.bin'), normals)


def test_fronto_parallel_plane_faces_the_camera():
    K = np.array([[100., 0, 32], [0, 100., 24], [0, 0, 1]])
    normals = depth_normals(np.full((48, 64), 2., np.float32), K)
    assert np.allclose(normals[10:-10, 10:-10], (0, 0, -1), atol=1e-6)


def test_tilted_plane_normal_and_missing_depth():
    K = np.array([[100., 0, 32], [0, 100., 24], [0, 0, 1]])
    # Plane z = 2 + 0.5 x: depth along each pixel's ray.
    u = np.arange(64)[None].repeat(48, 0)
    depth = (2 / (1 - .5 * (u - 32) / 100)).astype(np.float32)
    depth[0, 0] = 0
    normals = depth_normals(depth, K)
    expected = np.array([.5, 0, -1]) / np.linalg.norm([.5, 0, -1])
    assert np.allclose(normals[20, 30], expected, atol=1e-3)
    assert np.allclose(normals[0, 0], (0, 0, -1))
