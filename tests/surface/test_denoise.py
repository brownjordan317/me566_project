"""Bilateral normal filtering flattens noise on faces and keeps a cube's edges sharp."""
import numpy as np
import open3d as o3d

from recon.surface.denoise import denoise


def noisy_cube(seed=0):
    box = o3d.geometry.TriangleMesh.create_box(1, 1, 1).subdivide_midpoint(4)
    clean = np.asarray(box.vertices).copy()
    return clean, clean + np.random.default_rng(seed).normal(0, .01, clean.shape), np.asarray(box.triangles)


def test_flat_faces_come_out_flatter():
    clean, vertices, faces = noisy_cube()
    top = np.isclose(clean[:, 2], 1) & np.all((clean[:, :2] > .05) & (clean[:, :2] < .95), 1)   # inside the top face
    out, _ = denoise(vertices, faces)
    assert np.abs(out[top, 2] - 1).mean() < .6 * np.abs(vertices[top, 2] - 1).mean()


def test_edges_are_not_rounded_off():
    _, vertices, faces = noisy_cube()
    out, _ = denoise(vertices, faces)
    corner = np.argmin(np.linalg.norm(vertices - [1, 1, 1], axis=1))
    # Taubin-style smoothing pulls a cube corner inward; bilateral filtering should keep it near where it was.
    assert np.linalg.norm(out[corner] - [1, 1, 1]) < .05


def test_no_face_is_turned_inside_out():
    from recon.surface.denoise import face_geometry
    _, vertices, faces = noisy_cube(3)
    out, oriented = denoise(vertices, faces)
    before, after = face_geometry(vertices, oriented)[1], face_geometry(out, oriented)[1]
    assert ((before * after).sum(1) < 0).mean() < .001
