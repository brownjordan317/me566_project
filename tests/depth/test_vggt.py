"""VGGT depth must map back to the whole original frame."""
import numpy as np

from recon.depth.multiview.vggt import unpad_depth


def test_unpad_depth_landscape():
    image = np.full((518, 518), 5., np.float32)
    image[:63] = 100.
    image[455:] = 100.
    restored = unpad_depth(image, (720, 960))
    assert restored.shape == (720, 960)
    assert np.allclose(restored, 5.)


def test_unpad_depth_portrait():
    image = np.full((518, 518), 5., np.float32)
    image[:, :63] = 100.
    image[:, 455:] = 100.
    restored = unpad_depth(image, (960, 720))
    assert restored.shape == (960, 720)
    assert np.allclose(restored, 5.)
