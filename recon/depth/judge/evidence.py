"""Source-agnostic evidence for each depth candidate, for the learned judge.

A candidate is a depth map per frame in our cameras, from any source (a depth model, stereo, rendered
Gaussians or meshes, a sensor). The judge is never told the source. What it sees is how the 3D point a
candidate puts at each pixel fares against the capture:

  own evidence (OWN_FEATURES, per candidate):
    present          the candidate has depth here
    ncc_1..ncc_6     slanted-patch NCC against each neighbouring view, best first (as the fusion's
                     recon.depth.fusion.photometric scores, but per view instead of the best-3 mean)
    silhouette       share of the reference views whose mask the point projects into
    facing           how squarely the candidate's surface faces the camera (-normal z)
    roughness        |depth - local mean| / depth, the lumpiness of the candidate's own surface
    slope            relative depth gradient
  pair evidence (PAIR_FEATURES, candidate x candidate):
    support          share of reference views where the other candidate's depth agrees with this point
    violation        share where the other candidate sees clearly past it (the point is in its free space;
                     Merrell et al., ICCV 2007, as recon.depth.consistency)
  The diagonal is the candidate's agreement with itself in other views.
  self-report (SELF_FEATURES, per candidate, optional): a source that can check itself writes
  <frame>_selfcheck.npz beside its depth, with any of
    confidence       how sure it is the surface is there (0..1; the Gaussian pass: accumulated opacity)
    residual         how badly it explains the photo there (the Gaussian pass: |render - photo|)
    normal           1 - cos between its own surface normals and those of its depth
  plus self_present. A source without one reports nothing (self_present 0). Training gives synthetic
  candidates honest, uninformative or no self-reports (recon.depth.judge.train), so the judge weighs a
  report against the other evidence instead of trusting whoever brings one.

Everything that compares candidates is kept per pair and only combined (combine_pairs) once the set
of candidates is chosen, so a candidate left out of a training sample, or a judge run without one, leaves
no trace in the others' evidence. Image evidence (IMAGE_FEATURES: colour, local contrast, highlight)
describes the pixel, whichever candidate is judged.

Evidence is computed on a grid of every --stride-th working pixel, on the object mask dilated by two
grid steps.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from recon.depth.consistency import project_votes
from recon.depth.fusion.photometric import Images, camera_normals, ncc_scores

NEIGHBOURS = 6
OWN_FEATURES = ('present', *(f'ncc_{i + 1}' for i in range(NEIGHBOURS)), 'silhouette', 'facing', 'roughness', 'slope')
PAIR_FEATURES = ('support', 'violation')
IMAGE_FEATURES = ('red', 'green', 'blue', 'contrast', 'highlight', 'mask')
CROSS_FEATURES = ('relative_depth', 'agreeing', 'support_mean', 'support_max', 'violation_mean', 'self_support',
                  'self_violation')
SELF_FEATURES = ('self_present', 'self_confidence', 'self_residual', 'self_normal')


def self_report(folder, number: int, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """(SELF_FEATURES, h, w) from <folder>/<frame>_selfcheck.npz on the evidence grid; zeros without one."""
    out = np.zeros((len(SELF_FEATURES), len(rows), len(cols)), np.float32)
    path = folder / f'{number:08d}_selfcheck.npz'
    if not path.is_file():
        return out
    checks = np.load(path)
    out[0] = 1
    for slot, key in enumerate(('confidence', 'residual', 'normal'), 1):
        if key in checks:
            out[slot] = checks[key].astype(np.float32)[np.ix_(rows, cols)]
    return out


def reference_views(count: int, wanted: int = 12) -> list[int]:
    """Frames spread evenly over the sequence, the views every point is checked against."""
    return sorted(set(np.linspace(0, count - 1, min(wanted, count)).round().astype(int).tolist()))


def grid(height: int, width: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows and columns of the evidence grid in working pixels."""
    return np.arange(0, height, stride), np.arange(0, width, stride)


def image_evidence(photo: np.ndarray, mask: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """(IMAGE_FEATURES, h, w) from the working-size colour frame (BGR) and its mask."""
    rgb = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    gray = rgb.mean(2)
    contrast = np.sqrt(np.maximum(cv2.blur(gray ** 2, (9, 9)) - cv2.blur(gray, (9, 9)) ** 2, 0))
    hsv = cv2.cvtColor(photo, cv2.COLOR_BGR2HSV).astype(np.float32) / 255
    highlight = ((hsv[..., 2] > .9) & (hsv[..., 1] < .2)).astype(np.float32)
    stack = np.concatenate([rgb, contrast[..., None], highlight[..., None], (mask > 0)[..., None].astype(np.float32)], 2)
    return stack[np.ix_(rows, cols)].transpose(2, 0, 1)


class Capture:
    """Cameras, photos, masks and candidate depth of one run, on the GPU.

    K are working-frame intrinsics, R and t world-to-camera, all keyed by position in numbers.
    candidates maps a name to {position: depth (H, W)}; only the positions asked for are loaded.
    """

    def __init__(self, numbers: list[int], intrinsics: dict, rotations: dict, translations: dict,
                 images: Images, masks: dict, candidates: dict, device: str = 'cuda'):
        T = lambda v: torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device)
        self.numbers, self.device = numbers, device
        self.K_np = intrinsics
        self.K = {i: T(k) for i, k in intrinsics.items()}
        self.R = {i: T(r) for i, r in rotations.items()}
        self.t = {i: T(x) for i, x in translations.items()}
        self.c2w = {i: (self.R[i].T, -self.R[i].T @ self.t[i]) for i in self.R}
        self.images, self.masks, self.candidates = images, masks, candidates
        self.references = reference_views(len(numbers))
        self.depth_gpu = {name: {i: T(d) for i, d in maps.items() if i in self.references}
                          for name, maps in candidates.items()}
        self.mask_gpu = {i: T(masks[i] > 0) for i in self.references}


def candidate_evidence(capture: Capture, index: int, near: list[int], stride: int, radius: int = 4,
                       tolerance: float = .01) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evidence of every candidate at one frame.

    Returns own (C, OWN_FEATURES, h, w), pairs (PAIR_FEATURES, C, C, h, w), depth (C, h, w) and the grid's
    region of interest (h, w); candidates are in capture.candidates order.
    """
    names = list(capture.candidates)
    mask = capture.masks[index] > 0
    height, width = mask.shape
    rows, cols = grid(height, width, stride)
    region = cv2.dilate(mask.astype(np.uint8), np.ones((4 * stride + 1, 4 * stride + 1), np.uint8)) > 0
    region = region[np.ix_(rows, cols)]
    h, w = region.shape
    own = np.zeros((len(names), len(OWN_FEATURES), h, w), np.float32)
    pairs = np.zeros((len(PAIR_FEATURES), len(names), len(names), h, w), np.float32)
    depth_grid = np.zeros((len(names), h, w), np.float32)
    others = [j for j in capture.references if j != index]
    T = lambda v: torch.as_tensor(v, dtype=torch.float32, device=capture.device)
    for c, name in enumerate(names):
        depth = capture.candidates[name].get(index)
        if depth is None:
            continue
        sampled = depth[np.ix_(rows, cols)]
        depth_grid[c] = np.where(region, sampled, 0)
        gy, gx = np.nonzero(region & (sampled > 0))
        if not len(gy):
            continue
        v, u = rows[gy], cols[gx]
        z = depth[v, u].astype(np.float32)
        own[c, 0, gy, gx] = 1
        normals = camera_normals(depth, capture.K_np[index])
        n = T(normals[v, u])
        for slot, other in enumerate(near[:NEIGHBOURS]):
            score = ncc_scores(capture.images, index, [other], capture.K, capture.R, capture.t, T(z),
                               T(u.astype(np.float32)), T(v.astype(np.float32)), radius, best=1, normals=n)
            own[c, 1 + slot, gy, gx] = score.cpu().numpy()
        scores = own[c][1:1 + NEIGHBOURS]                                   # view, so assignment sticks
        scores[:, gy, gx] = -np.sort(-scores[:, gy, gx], axis=0)
        K = capture.K_np[index]
        camera = np.stack(((u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z), 1)
        rotation, center = capture.c2w[index]
        points = T(camera) @ rotation.T + center
        inside_mask = torch.zeros(len(z), device=capture.device)
        in_view = torch.zeros(len(z), device=capture.device)
        support = torch.zeros((len(names), len(z)), device=capture.device)
        violation = torch.zeros_like(support)
        seen = torch.zeros_like(support)
        for other in others:
            other_rotation, other_center = capture.c2w[other]
            Ko = capture.K[other]
            cam = (points - other_center) @ other_rotation
            zz = cam[:, 2].clamp(min=1e-6)
            px = torch.round(Ko[0, 0] * cam[:, 0] / zz + Ko[0, 2]).long()
            py = torch.round(Ko[1, 1] * cam[:, 1] / zz + Ko[1, 2]).long()
            inside = (cam[:, 2] > 0) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
            in_view += inside
            hit = torch.zeros_like(zz)
            hit[inside] = capture.mask_gpu[other][py[inside], px[inside]]
            inside_mask += hit
            for k, other_name in enumerate(names):
                other_depth = capture.depth_gpu[other_name].get(other)
                if other_depth is None:
                    continue
                agree, past = project_votes(points, other_depth, Ko, other_rotation, other_center, tolerance)
                looked = inside & (other_depth[py.clamp(0, height - 1), px.clamp(0, width - 1)] > 0)
                support[k] += agree
                violation[k] += past
                seen[k] += looked
        own[c, 1 + NEIGHBOURS, gy, gx] = (inside_mask / in_view.clamp(min=1)).cpu().numpy()
        share = lambda x: (x / seen.clamp(min=1)).cpu().numpy()
        pairs[0, c][:, gy, gx], pairs[1, c][:, gy, gx] = share(support), share(violation)
        facing, roughness, slope = surface_shape(depth, normals)
        own[c, 2 + NEIGHBOURS, gy, gx] = facing[v, u]
        own[c, 3 + NEIGHBOURS, gy, gx] = roughness[v, u]
        own[c, 4 + NEIGHBOURS, gy, gx] = slope[v, u]
    return own, pairs, depth_grid, region


def surface_shape(depth: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Facing (-normal z), roughness (|depth - 5x5 mean| / depth) and relative slope of one depth map."""
    valid = (depth > 0).astype(np.float32)
    mean = cv2.blur(depth * valid, (5, 5)) / np.maximum(cv2.blur(valid, (5, 5)), 1e-6)
    safe = np.maximum(depth, 1e-6)
    roughness = np.where(depth > 0, np.minimum(np.abs(depth - mean) / safe, .05), 0)
    gy, gx = np.gradient(depth)
    slope = np.where(depth > 0, np.minimum(np.hypot(gx, gy) / safe, .2), 0)
    return -normals[..., 2], roughness.astype(np.float32), slope.astype(np.float32)


def combine_pairs(depth: torch.Tensor, pairs: torch.Tensor, agree: float = .005) -> torch.Tensor:
    """CROSS_FEATURES for a chosen set of candidates.

    depth (B, C, h, w) and pairs (B, PAIR_FEATURES, C, C, h, w) hold only the chosen candidates (absent
    pixels have depth 0). Returns (B, C, CROSS_FEATURES, h, w): log depth relative to the set's median,
    the share of other candidates within `agree` of it, the mean and max support from the others' depth in
    other views, the mean violation by them, and the candidate's agreement and violation with itself.
    """
    B, C = depth.shape[:2]
    present = depth > 0
    logd = torch.log(depth.clamp(min=1e-6))
    masked = torch.where(present, logd, torch.nan)
    median = torch.nanmedian(masked, dim=1, keepdim=True).values
    relative = torch.where(present, (logd - median).nan_to_num(0), 0).clamp(-.1, .1) * 10
    close = ((logd[:, :, None] - logd[:, None]).abs() < agree) & present[:, :, None] & present[:, None]
    eye = torch.eye(C, dtype=torch.bool, device=depth.device)[None, :, :, None, None]
    others = (~eye).expand_as(close) & present[:, None]
    count = others.sum(2).clamp(min=1)
    agreeing = (close & ~eye).sum(2) / count
    support, violation = pairs[:, 0], pairs[:, 1]
    support_mean = torch.where(others, support, 0).sum(2) / count
    support_max = torch.where(others, support, 0).amax(2)
    violation_mean = torch.where(others, violation, 0).sum(2) / count
    diagonal = lambda x: torch.diagonal(x, dim1=1, dim2=2).permute(0, 3, 1, 2)
    cross = torch.stack([relative, agreeing, support_mean, support_max, violation_mean,
                         diagonal(support), diagonal(violation)], 2)
    return torch.where(present[:, :, None], cross, 0)
