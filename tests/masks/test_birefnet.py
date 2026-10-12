"""Guided BiRefNet masks keep the object's pieces and drop other salient things."""
import numpy as np

from recon.masks.birefnet import keep_guided


def test_keep_guided_completes_object_and_drops_other_foreground():
    full = np.zeros((40, 60), bool)
    full[5:25, 5:25] = True                # the object, larger than its supplied mask
    full[5:25, 40:55] = True               # a chair the supplied mask never touches
    guide = np.zeros_like(full)
    guide[8:20, 8:20] = True
    kept = keep_guided(full, guide)
    assert kept[5:25, 5:25].all()
    assert not kept[:, 40:].any()


def test_keep_guided_falls_back_to_the_supplied_mask():
    full = np.zeros((20, 20), bool)
    full[0:5, 0:5] = True
    guide = np.zeros_like(full)
    guide[10:15, 10:15] = True
    assert (keep_guided(full, guide) == guide).all()
