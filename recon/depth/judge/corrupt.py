"""Synthetic depth candidates: true depth damaged the way real depth sources fail.

The judge must hold up on sources that do not exist yet, so it is also trained on candidates whose
failures no current model shows in that mix. Each synthetic candidate starts from the true depth in every
frame and gets:

  always   a per-frame scale and shift drift (an imperfect grounding fit), a smooth bias that is fixed
           in the world, so it is the same error in every view, like a model's systematic bias
           (random Fourier features of the world position), and a smooth bias of each view alone, so
           the candidate disagrees with itself across views as real models do;
  some of  whole regions offset (a model wrong over a part, the heavy tail of real errors), softened depth edges, thin parts lost (to a hole or to the surface behind), floaters pulled
           towards the camera, holes, highlights pushed behind the surface (a reflection's depth),
           lumpy high-frequency noise, and edges bled onto the background.

The parameters are drawn once per candidate (its "character"); blobs are drawn per frame, as real
floaters and holes do not repeat across views.
"""
from __future__ import annotations

import cv2
import numpy as np

DAMAGES = ('soften', 'thin', 'floaters', 'holes', 'glossy', 'lumps', 'bleed', 'regions')


def character(rng: np.random.Generator) -> dict:
    """One synthetic candidate's failure parameters."""
    chosen = rng.choice(DAMAGES, size=rng.integers(1, 4), replace=False).tolist()
    if rng.random() < .5 and 'regions' not in chosen:
        chosen.append('regions')
    return {'drift_scale': float(rng.uniform(.001, .008)), 'drift_shift': float(rng.uniform(0, .004)),
            'bias': float(rng.uniform(.001, .008)), 'bias_frequency': float(rng.uniform(2, 12)),
            'view_bias': float(rng.uniform(.001, .006)),
            'frequencies': rng.normal(size=(6, 3)), 'phases': rng.uniform(0, 2 * np.pi, 6),
            'lumps': float(rng.uniform(.0005, .002)), 'damages': chosen}


def world_field(points: np.ndarray, frequencies: np.ndarray, phases: np.ndarray, scale: float) -> np.ndarray:
    """A smooth field over world points (N, 3) in [-1, 1], from random Fourier features."""
    return np.sin(points @ (frequencies.T * scale) + phases).mean(1) * np.sqrt(2)


def blobs(shape: tuple[int, int], count: int, radius: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """A mask of `count` random discs."""
    out = np.zeros(shape, np.uint8)
    for _ in range(count):
        cv2.circle(out, (int(rng.integers(shape[1])), int(rng.integers(shape[0]))), int(rng.integers(*radius)), 1, -1)
    return out > 0


def damage(truth: np.ndarray, photo: np.ndarray, K: np.ndarray, rotation: np.ndarray, center: np.ndarray,
           extent: float, traits: dict, rng: np.random.Generator) -> np.ndarray:
    """One frame of a synthetic candidate.

    truth is the true depth (0 off the object), photo the working-size BGR frame, rotation and center the
    camera-to-world pose, extent the object's size in map units (sets the bias field's frequency).
    """
    depth = truth.astype(np.float32).copy()
    valid = depth > 0
    if not valid.any():
        return depth
    height, width = depth.shape
    v, u = np.nonzero(valid)
    z = depth[v, u]
    points = np.stack(((u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z), 1) @ rotation.T + center
    field = world_field(points, traits['frequencies'], traits['phases'], traits['bias_frequency'] / extent)
    depth[v, u] = z * (1 + traits['bias'] * field)
    # A smooth error of this view alone: real models disagree with themselves across views.
    coarse = rng.normal(size=(4, 4)).astype(np.float32)
    depth[valid] *= 1 + traits['view_bias'] * cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)[valid]
    damages = traits['damages']
    if 'lumps' in damages:
        noise = cv2.GaussianBlur(rng.normal(size=depth.shape).astype(np.float32), (0, 0), 1.5)
        depth[valid] *= 1 + traits['lumps'] * noise[valid] / max(noise[valid].std(), 1e-6)
    if 'regions' in damages:
        for _ in range(int(rng.integers(1, 4))):
            region = cv2.GaussianBlur(blobs(depth.shape, 1, (30, 90), rng).astype(np.float32), (0, 0), 8)
            depth[valid] *= 1 + rng.choice([-1, 1]) * rng.uniform(.005, .02) * region[valid]
    if 'soften' in damages:
        filled = cv2.GaussianBlur(np.where(valid, depth, 0), (0, 0), 3)
        weight = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), 3)
        smooth = filled / np.maximum(weight, 1e-6)
        edges = cv2.dilate((cv2.Laplacian(np.where(valid, depth, 0), cv2.CV_32F).__abs__() > .01 * depth.max())
                           .astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        depth = np.where(valid & edges, smooth, depth)
    if 'thin' in damages:
        opened = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)) > 0
        thin = valid & ~opened
        behind = cv2.dilate(np.where(opened, depth, 0), np.ones((15, 15), np.uint8))
        depth[thin] = np.where(rng.random() < .5, 0, behind[thin])
    if 'floaters' in damages:
        spots = blobs(depth.shape, int(rng.integers(1, 6)), (3, 25), rng) & valid
        depth[spots] *= 1 - rng.uniform(.01, .05)
    if 'holes' in damages:
        depth[blobs(depth.shape, int(rng.integers(1, 6)), (5, 40), rng)] = 0
    if 'glossy' in damages:
        hsv = cv2.cvtColor(photo, cv2.COLOR_BGR2HSV)
        shiny = cv2.dilate(((hsv[..., 2] > 200) & (hsv[..., 1] < 60)).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        depth[shiny & valid] *= 1 + rng.uniform(.01, .03)
    if 'bleed' in damages:
        reach = int(rng.integers(2, 8))
        grown = cv2.dilate(np.where(depth > 0, depth, 0), np.ones((2 * reach + 1, 2 * reach + 1), np.uint8))
        depth = np.where(depth > 0, depth, grown)
    scale = 1 + rng.uniform(-1, 1) * traits['drift_scale']
    shift = rng.uniform(-1, 1) * traits['drift_shift'] * float(np.median(z))
    return np.where(depth > 0, depth * scale + shift, 0).astype(np.float32)
