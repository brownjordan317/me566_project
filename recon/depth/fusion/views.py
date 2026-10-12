"""Neighbouring views for multi-view checks."""
from __future__ import annotations

import numpy as np


def neighbours(rotations: np.ndarray, translations: np.ndarray, count: int,
               band: tuple[float, float] | None = None) -> list[list[int]]:
    """For each frame, the neighbouring frames to compare it with (excluding itself).

    rotations and translations are world-to-camera, as in COLMAP models. By
    default the frames with the most similar viewing direction are taken. With
    band=(low, high) degrees, frames whose direction differs by an angle in the
    band are preferred, closest to its middle first: a depth change only moves a
    patch across neighbouring views in proportion to the angle between them, so
    near-duplicate views cannot resolve millimetre differences. Frames outside
    the band fill in when it holds fewer than count.
    """
    directions = rotations[:, 2, :]                    # camera z axis in world (row 3 of world-to-camera)
    centres = -np.einsum('nji,nj->ni', rotations, translations)
    result = []
    for index in range(len(rotations)):
        angle = np.degrees(np.arccos(np.clip(directions @ directions[index], -1, 1)))
        baseline = np.linalg.norm(centres - centres[index], axis=1)
        angle[index] = np.inf
        angle[baseline < 1e-6] = np.inf               # identical positions give no parallax
        order = np.argsort(angle)
        if band is not None:
            low, high = band
            inside = [j for j in np.argsort(np.abs(angle - (low + high) / 2))
                      if low <= angle[j] <= high]
            order = inside + [j for j in order if j not in inside]
        result.append([int(j) for j in order[:count] if np.isfinite(angle[j])])
    return result
