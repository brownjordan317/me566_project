"""Photometric evidence for depth hypotheses from full-resolution images.

A pixel's depth hypothesis places a small patch in 3D, either fronto-parallel
to the reference camera or tilted to a given surface normal (the slanted
support planes of PatchMatch stereo, Bleyer et al., BMVC 2011); projecting it into neighbouring views and comparing the
image patches by normalized cross-correlation (NCC) scores how well the
images support that depth, the matching cost of plane-sweep stereo (Collins,
CVPR 1996) and PatchMatch MVS (Schoenberger et al., ECCV 2016). The best few
neighbours are averaged so one occluded view does not veto a correct depth.
A single patch is weak evidence where models differ by a millimetre or two,
so scores are pooled over a window before comparing hypotheses.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def bilinear(image: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Bilinear samples of a (H, W) image at pixel coordinates x, y of any shape."""
    height, width = image.shape
    grid = torch.stack((x / (width - 1) * 2 - 1, y / (height - 1) * 2 - 1), -1)
    return F.grid_sample(image[None, None], grid.reshape(1, 1, -1, 2), align_corners=True).reshape(x.shape)


class Images:
    """Grayscale frames at full resolution, matched to frame numbers by file stem.

    gray is keyed by position in numbers, like the K, R, t dictionaries of the
    callers, so frame numbers with gaps (unregistered or held-out frames) work.
    """

    def __init__(self, directory: Path, numbers: list[int], working_width: int, device: str):
        files = {int(path.stem): path for path in directory.iterdir()
                 if path.suffix.lower() in ('.png', '.jpg', '.jpeg') and path.stem.isdigit()}
        missing = [number for number in numbers if number not in files]
        if missing:
            raise FileNotFoundError(f'No full-resolution image for frames {missing[:5]} in {directory}')
        self.gray = {}
        for index, number in enumerate(numbers):
            image = cv2.imread(str(files[number]), cv2.IMREAD_GRAYSCALE)
            self.gray[index] = torch.as_tensor(image, dtype=torch.float32, device=device)
        self.scale = next(iter(self.gray.values())).shape[1] / working_width


def camera_normals(depth: np.ndarray, K: np.ndarray, smooth: int = 5) -> np.ndarray:
    """Per-pixel camera-space surface normals (H, W, 3) of a depth map, facing the camera.

    Depth is smoothed over valid pixels first, so a normal describes the local
    surface rather than per-pixel noise. Pixels without depth get (0, 0, -1).
    """
    valid = (depth > 0).astype(np.float32)
    weight = cv2.blur(valid, (smooth, smooth))
    smoothed = cv2.blur(depth * valid, (smooth, smooth)) / np.maximum(weight, 1e-6)
    height, width = depth.shape
    v, u = np.mgrid[:height, :width].astype(np.float32)
    points = np.stack(((u - K[0, 2]) / K[0, 0] * smoothed, (v - K[1, 2]) / K[1, 1] * smoothed, smoothed), -1)
    dx = np.zeros_like(points)
    dy = np.zeros_like(points)
    dx[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dy[1:-1] = points[2:] - points[:-2]
    normals = np.cross(dx, dy)
    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.where(length > 0, normals / np.maximum(length, 1e-12), [0, 0, -1])
    normals = np.where((normals * points).sum(-1, keepdims=True) > 0, -normals, normals)
    normals[weight < .5] = [0, 0, -1]
    return normals.astype(np.float32)


def ncc_scores(images: Images, reference: int, others: list[int], K: dict, R: dict, t: dict,
               depth: torch.Tensor, u: torch.Tensor, v: torch.Tensor, radius: int = 4,
               best: int = 3, normals: torch.Tensor | None = None) -> torch.Tensor:
    """Mean of the best NCCs over neighbouring views for each pixel's depth hypothesis.

    u, v are working-frame pixels; K are working-frame intrinsics, scaled here to
    the full-resolution images; R, t are world-to-camera. normals (N, 3), in the
    reference camera, tilt each patch to the surface; without them patches are
    fronto-parallel. Returns (N,), in [-1, 1].
    """
    device = depth.device
    s = images.scale
    Ki = K[reference].clone()
    Ki[:2] *= s
    full_u, full_v = (u + .5) * s - .5, (v + .5) * s - .5
    steps = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    dy, dx = torch.meshgrid(steps, steps, indexing='ij')
    px, py = full_u[:, None] + dx.reshape(-1), full_v[:, None] + dy.reshape(-1)   # (N, P)
    patch = bilinear(images.gray[reference], px, py)
    rays = torch.stack(((px - Ki[0, 2]) / Ki[0, 0], (py - Ki[1, 2]) / Ki[1, 1], torch.ones_like(px)), -1)
    if normals is None:
        camera = rays * depth[:, None, None]
    else:
        # Intersect each patch ray with the plane through the centre point with this normal.
        centre_ray = torch.stack(((full_u - Ki[0, 2]) / Ki[0, 0], (full_v - Ki[1, 2]) / Ki[1, 1],
                                  torch.ones_like(full_u)), -1)
        offset = (normals * centre_ray).sum(-1) * depth                    # n . X0
        along = (rays * normals[:, None]).sum(-1)                           # n . r
        scale = offset[:, None] / torch.where(along.abs() > 1e-3, along, torch.full_like(along, 1e-3))
        scale = torch.where(scale > 0, scale, depth[:, None].expand_as(scale))
        camera = rays * scale[..., None]
    world = (camera - t[reference]) @ R[reference]                      # world = R^T (x - t)
    centred = patch - patch.mean(1, keepdim=True)
    scores = []
    for other in others:
        Ko = K[other].clone()
        Ko[:2] *= s
        projected = world @ R[other].T + t[other]
        z = projected[..., 2].clamp(min=1e-6)
        target = bilinear(images.gray[other], Ko[0, 0] * projected[..., 0] / z + Ko[0, 2],
                          Ko[1, 1] * projected[..., 1] / z + Ko[1, 2])
        target = target - target.mean(1, keepdim=True)
        scores.append((centred * target).sum(1) / (centred.norm(dim=1) * target.norm(dim=1) + 1e-3))
    stacked = torch.stack(scores)
    return stacked.topk(min(best, len(scores)), dim=0).values.mean(0)


def pool(values: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    """Windowed mean of sparse per-pixel values (H, W) over the pixels where valid."""
    weight = cv2.blur(valid.astype(np.float32), (window, window))
    total = cv2.blur(np.where(valid, values, 0).astype(np.float32), (window, window))
    return np.where(weight > 0, total / np.maximum(weight, 1e-6), np.nan)
