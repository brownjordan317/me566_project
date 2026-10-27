#!/usr/bin/env python3
"""Refine our reconstruction against every photo with planar Gaussians (PGSR's architecture, our implementation).

PGSR (Chen et al., "PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity
Surface Reconstruction", TVCG 2024) optimises flat Gaussians so that renderings match
the photos while depth stays consistent within each view and across neighbouring
views. Started from our reconstruction it improved our meshes (DTU chamfer −7%,
MobileBrick jeep F@1 +3.7), but its code is research-only and runs as a separate
program. This module reimplements the ideas; no PGSR code is used. Rendering is
gsplat's 2D-Gaussian (surfel) rasteriser (Ye et al., gsplat, Apache-2.0), which also
returns rendered normals and normals from its depth (Huang et al., 2DGS, SIGGRAPH 2024).

Per iteration, for a random training view and (from --geo-from) a random neighbour:
  photometric  0.8 L1 + 0.2 (1 − SSIM) against the photo;
  normal       1 − rendered normal · depth normal (2DGS's depth-normal consistency);
  geometric    a pixel's rendered depth is projected into the neighbour, its depth
               there back-projected and projected home; the round-trip error in
               pixels, weighted by exp(−error) and used where under --geo-threshold;
  ncc          7×7 patches warped into the neighbour by the homography of the local
               plane (rendered normal and depth), 1 − NCC weighted like the geometric
               term (PGSR's multi-view photometric consistency).
Gaussians start at our dense points (object and scene clouds), oriented along normals
estimated from the cloud, sized from their nearest neighbours, coloured from the cloud.
Cameras are our COLMAP model at the working frames (full intrinsic matrix, so no crop).
From --densify-from to --densify-until, every --densify-every iterations, Gaussians whose
mean screen-space gradient (gsplat's 2DGS gradient, scaled to pixels) exceeds
--grad-threshold are split in their plane if large or cloned if small, and nearly
transparent ones are pruned (3DGS's adaptive density control, Kerbl et al., SIGGRAPH 2023).

Output: OUTPUT/<frame>_grounded.npy (rendered median depth, where accumulated opacity
passes 0.5, as 2DGS meshes; --export expected for the alpha-weighted mean; only where
the alpha exceeds 0.5) and hybrid_depth.json, the format our cross-view check and
object TSDF read. Runs in an environment with gsplat (make setup-gaussians).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from recon.common.colmap import parse_model
from recon.depth.fusion.views import neighbours


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean SSIM of two (3, H, W) images with an 11×11 Gaussian window."""
    g = torch.exp(-((torch.arange(11, device=a.device) - 5.) ** 2) / (2 * 1.5 ** 2))
    window = (g[:, None] * g[None]) / g.sum() ** 2
    window = window.expand(3, 1, 11, 11)
    blur = lambda x: F.conv2d(x[None], window, padding=5, groups=3)[0]
    mu_a, mu_b = blur(a), blur(b)
    var_a, var_b, cov = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2, blur(a * b) - mu_a * mu_b
    c1, c2 = .01 ** 2, .03 ** 2
    return (((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))).mean()


def quaternion_from_normal(normals: np.ndarray) -> np.ndarray:
    """(w, x, y, z) rotations taking the z axis onto each normal."""
    z = np.array([0., 0., 1.])
    axis = np.cross(z, normals)
    sin = np.linalg.norm(axis, axis=1)
    cos = normals @ z
    angle = np.arctan2(sin, cos)
    axis = np.where(sin[:, None] > 1e-8, axis / np.maximum(sin[:, None], 1e-12), [1., 0., 0.])
    return np.concatenate([np.cos(angle / 2)[:, None], axis * np.sin(angle / 2)[:, None]], 1)


class Views:
    """Our cameras and photos on the GPU."""

    def __init__(self, sequence: Path, model: Path, device: str, principal_shift: float = 0.,
                 masks: Path | None = None):
        intrinsics, _, frames = parse_model(model)
        self.numbers = [number for number, *_ in frames]
        self.K = torch.tensor(np.stack([intrinsics[n] for n in self.numbers]), dtype=torch.float32, device=device)
        # A shift of the principal point, for Gaussians fitted in another camera frame: the 2DGS / PGSR workspace
        # (twodgs.write_workspace) puts it half a pixel from ours, and PGSR's Gaussians render as its own export with 0.5.
        self.K[:, :2, 2] += principal_shift
        view = np.tile(np.eye(4), (len(frames), 1, 1))
        for i, (_, rotation, translation, _) in enumerate(frames):
            view[i, :3, :3], view[i, :3, 3] = rotation, translation
        self.viewmats = torch.tensor(view, dtype=torch.float32, device=device)     # world-to-camera
        self.rotations = np.stack([f[1] for f in frames])
        self.translations = np.stack([f[2] for f in frames])
        images = []
        for number in self.numbers:
            rgb = cv2.cvtColor(cv2.imread(str(sequence / 'frames' / f'{number:08d}.png')), cv2.COLOR_BGR2RGB)
            images.append(torch.tensor(rgb, device=device))
        self.images = torch.stack(images).permute(0, 3, 1, 2).float() / 255             # (N, 3, H, W)
        self.gray = self.images.mean(1)                                                 # (N, H, W)
        self.height, self.width = self.images.shape[-2:]
        self.masks = None
        if masks is not None:
            layers = [cv2.imread(str(masks / f'{number:08d}.png'), cv2.IMREAD_GRAYSCALE) for number in self.numbers]
            if any(layer is None for layer in layers):
                raise FileNotFoundError(f'A mask is missing from {masks}')
            self.masks = (torch.tensor(np.stack(layers), device=device).float() / 255 > .5).float()   # (N, H, W)
        centres = -np.einsum('nji,nj->ni', self.rotations, self.translations)
        self.radius = 1.1 * float(np.max(np.linalg.norm(centres - centres.mean(0), axis=1)))
        self._scaled = {1.: self.gray}

    def gray_scaled(self, scale: float) -> torch.Tensor:
        """Grey images resampled by `scale` (area averaging), cached."""
        if scale not in self._scaled:
            size = (round(self.height * scale), round(self.width * scale))
            self._scaled[scale] = F.interpolate(self.gray[:, None], size=size, mode='area')[:, 0]
        return self._scaled[scale]

    def gradient_weight(self, index: int) -> torch.Tensor:
        """(1 - normalised image gradient)², low at edges (PGSR's weighting of its single-view loss)."""
        image = self.images[index]
        gx = (image[:, 1:-1, 2:] - image[:, 1:-1, :-2]).abs().mean(0)
        gy = (image[:, 2:, 1:-1] - image[:, :-2, 1:-1]).abs().mean(0)
        grad = torch.maximum(gx, gy)
        grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
        grad = F.pad(grad[None, None], (1, 1, 1, 1), value=1.)[0, 0]
        return ((1 - grad).clamp(0, 1) ** 2)


class Gaussians(torch.nn.Module):
    def __init__(self, xyz: np.ndarray, rgb: np.ndarray, normals: np.ndarray, spacing: np.ndarray, device: str,
                 opacity: float = .9, size_factor: float = .7):
        super().__init__()
        t = lambda x: torch.nn.Parameter(torch.tensor(x, dtype=torch.float32, device=device))
        self.means = t(xyz)
        self.quats = t(quaternion_from_normal(normals))
        size = np.log(np.clip(spacing * size_factor, 1e-6, None))
        self.scales = t(np.stack([size, size, size - 4], 1))          # flat: the third axis is ignored by 2DGS
        self.opacities = t(np.full(len(xyz), math.log(opacity / (1 - opacity))))   # logit
        # View-dependent colour as spherical harmonics (degree up to 3, as 3DGS and PGSR): without it, highlights
        # can only be explained by bending the surface.
        self.sh0 = t(((np.clip(rgb, .02, .98) - .5) / SH_C0)[:, None, :])
        self.shN = t(np.zeros((len(xyz), 15, 3)))
        self.sh_degree = 0
        # One foreground logit per Gaussian, supervised by the 2D masks and composited like colour
        # (the identity-feature idea of Gaussian Grouping, Ye et al. 2024, and SAGA, Cen et al. 2023;
        # implemented here rather than ported). It is a read-out: the label render detaches the geometry,
        # so no mask can move a surfel and the reconstruction stays exactly what it would be without it.
        self.labels = t(np.zeros((len(xyz), 1)))

    export = 'median'

    def render(self, views: Views, index: list[int], mode: str = 'RGB+ED'):
        from gsplat import rasterization_2dgs
        colors = torch.cat([self.sh0, self.shN], 1)
        return rasterization_2dgs(self.means, F.normalize(self.quats, dim=-1), torch.exp(self.scales),
                                  torch.sigmoid(self.opacities), colors, views.viewmats[index],
                                  views.K[index], views.width, views.height, render_mode=mode,
                                  sh_degree=self.sh_degree)

    def render_labels(self, views: Views, index: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Alpha-composite the foreground probability, with the geometry held fixed.

        Occlusion is the reason this beats a visual hull: the rasteriser decides which Gaussian each
        pixel sees, so a surfel on the table is supervised as background in the views that see it, while
        one buried under the object receives no gradient instead of being wrongly kept by a silhouette
        cone that cannot know it is hidden.
        """
        from gsplat import rasterization_2dgs
        rendered, alphas, *_ = rasterization_2dgs(
            self.means.detach(), F.normalize(self.quats, dim=-1).detach(), torch.exp(self.scales).detach(),
            torch.sigmoid(self.opacities).detach(),
            torch.sigmoid(self.labels)[None].expand(len(index), -1, -1),    # features need a camera axis
            views.viewmats[index],
            views.K[index], views.width, views.height, render_mode='RGB', sh_degree=None)
        return rendered[..., 0], alphas[..., 0]

    def forward(self, views: Views, index: int) -> dict:
        colors, alphas, normals, surf_normals, _, _, meta = self.render(views, [index])
        surface = surf_normals if surf_normals.dim() == 3 else surf_normals[0]    # no camera axis for one camera
        return {'image': colors[0, ..., :3].permute(2, 0, 1), 'depth': colors[0, ..., 3], 'alpha': alphas[0, ..., 0],
                'normal': normals[0], 'depth_normal': surface, 'gradient': meta['gradient_2dgs'],
                'radii': meta['radii'][0], 'regulariser': torch.zeros((), device=self.means.device)}

    def depth_only(self, views: Views, index: int, export: bool = False) -> torch.Tensor:
        colors, alphas, _, _, _, median, _ = self.render(views, [index], mode='ED')
        depth = median[0, ..., 0] if (export and self.export == 'median') else colors[0, ..., 0]
        return torch.where(alphas[0, ..., 0] > .5, depth, torch.zeros_like(depth)) if export else depth


def rotation_matrices(quats: torch.Tensor) -> torch.Tensor:
    w, x, y, z = F.normalize(quats, dim=-1).unbind(-1)
    return torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1).reshape(-1, 3, 3)


def load_gaussians(path: Path, device: str, center=np.zeros(3), radius: float = 1.) -> 'Gaussians3D':
    """Trained 3D Gaussians from a 3DGS-layout PLY (PGSR's point_cloud.ply), moved out of a normalised frame."""
    from plyfile import PlyData
    vertex = PlyData.read(str(path))['vertex']
    column = lambda *names: np.stack([np.asarray(vertex[n]) for n in names], 1).astype(np.float32)
    rest = column(*[f'f_rest_{i}' for i in range(45)]).reshape(-1, 3, 15).transpose(0, 2, 1)   # channel-major
    t = lambda x: torch.nn.Parameter(torch.tensor(np.ascontiguousarray(x), dtype=torch.float32, device=device))
    gaussians = Gaussians3D.__new__(Gaussians3D)
    torch.nn.Module.__init__(gaussians)
    gaussians.means = t(column('x', 'y', 'z') * radius + center)
    gaussians.quats = t(column('rot_0', 'rot_1', 'rot_2', 'rot_3'))
    gaussians.scales = t(column('scale_0', 'scale_1', 'scale_2') + math.log(radius))
    gaussians.opacities = t(column('opacity')[:, 0])
    gaussians.sh0, gaussians.shN = t(column('f_dc_0', 'f_dc_1', 'f_dc_2')[:, None]), t(rest)
    gaussians.sh_degree, gaussians.flat_weight = 3, 100.
    # __init__ is bypassed above, so the label logit it creates has to be made here too, or a loaded
    # file cannot be segmented (solve_labels on it would fail on the missing attribute).
    gaussians.labels = t(np.zeros((len(gaussians.means), 1)))
    return gaussians


def save_gaussians(gaussians: 'Gaussians', path: Path) -> None:
    """Write the fitted Gaussians in the 3DGS PLY layout, which `load_gaussians` reads back."""
    from plyfile import PlyData, PlyElement
    with torch.no_grad():
        means = gaussians.means.cpu().numpy()
        normals = np.zeros_like(means)
        sh0 = gaussians.sh0[:, 0].cpu().numpy()
        shN = gaussians.shN.permute(0, 2, 1).reshape(len(means), -1).cpu().numpy()   # channel-major, as 3DGS
        opacities = gaussians.opacities.cpu().numpy().reshape(-1, 1)
        scales = gaussians.scales.cpu().numpy()
        quats = F.normalize(gaussians.quats, dim=-1).cpu().numpy()
    names = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2']
    names += [f'f_rest_{i}' for i in range(shN.shape[1])]
    names += ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    columns = np.concatenate([means, normals, sh0, shN, opacities, scales, quats], 1).astype(np.float32)
    array = np.empty(len(columns), dtype=[(name, 'f4') for name in names])
    for index, name in enumerate(names):
        array[name] = columns[:, index]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(array, 'vertex')]).write(str(path))


def depth_normals(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Camera-frame normals (H, W, 3) from central differences of a z-depth map, facing the camera."""
    height, width = depth.shape
    v, u = torch.meshgrid(torch.arange(height, device=depth.device), torch.arange(width, device=depth.device),
                          indexing='ij')
    points = backproject(depth, K, u.float(), v.float())
    dx = torch.zeros_like(points); dy = torch.zeros_like(points)
    dx[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dy[1:-1] = points[2:] - points[:-2]
    normal = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    return torch.where(((normal * points).sum(-1, keepdim=True) > 0), -normal, normal)


class Gaussians3D(Gaussians):
    """Flattened 3D Gaussians with PGSR-style plane depth, on gsplat's 3D rasteriser.

    Each Gaussian's normal is the axis of its smallest scale, turned to face the camera; a second
    render blends the camera-frame normal and the plane distance |n·X|, and the plane depth is
    distance / |n · ray| (PGSR's unbiased depth). A scale loss on the smallest axis keeps them flat.
    """
    export = 'plane'

    def __init__(self, xyz, rgb, normals, spacing, device, opacity=.3, size_factor=.7, flat_weight=100.,
                 isotropic=False):
        super().__init__(xyz, rgb, normals, spacing, device, opacity, size_factor)
        self.flat_weight = flat_weight
        with torch.no_grad():
            if isotropic:                                    # as 3DGS / PGSR: round, unrotated
                self.scales[:, 2] = self.scales[:, 0]
                self.quats.copy_(torch.tensor([1., 0, 0, 0], device=device).expand_as(self.quats))
            else:                                            # the third (normal) axis starts thin, not ignored
                self.scales[:, 2] = self.scales[:, 0] - 2.3     # about a tenth of the in-plane size

    def _normals(self) -> torch.Tensor:
        rotation = rotation_matrices(self.quats)
        smallest = torch.exp(self.scales).argmin(-1)
        return torch.gather(rotation, 2, smallest[:, None, None].expand(-1, 3, 1))[..., 0]   # column = axis

    def _plane(self, views: Views, index: int, geometry_only: bool = False):
        from gsplat import rasterization
        view = views.viewmats[index]
        camera = self.means @ view[:3, :3].T + view[:3, 3]
        normal = self._normals() @ view[:3, :3].T
        normal = torch.where((normal * camera).sum(-1, keepdim=True) > 0, -normal, normal)   # face the camera
        distance = (normal * camera).sum(-1, keepdim=True).abs()
        quats, scales, opacities = F.normalize(self.quats, dim=-1), torch.exp(self.scales), torch.sigmoid(self.opacities)
        geometry, alphas, meta = rasterization(self.means, quats, scales, opacities, torch.cat([normal, distance], -1)[None],
                                               view[None], views.K[index][None], views.width, views.height,
                                               render_mode='RGB', packed=False)
        alpha = alphas[0, ..., 0]
        blended = geometry[0, ..., :3]
        n = F.normalize(blended, dim=-1)
        height, width = alpha.shape
        v, u = torch.meshgrid(torch.arange(height, device=alpha.device), torch.arange(width, device=alpha.device), indexing='ij')
        K = views.K[index]
        ray = torch.stack(((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], torch.ones_like(alpha)), -1)
        # Both blends unnormalised, so the weights cancel (normalising the normal alone shrinks the depth by |Σwn|/α).
        depth = geometry[0, ..., 3] / (blended * ray).sum(-1).abs().clamp(min=1e-6)
        depth = torch.where(alpha > 1e-3, depth, torch.zeros_like(depth))
        return depth, n, alpha, meta

    def forward(self, views: Views, index: int) -> dict:
        from gsplat import rasterization
        colors = torch.cat([self.sh0, self.shN], 1)
        image, alphas, meta = rasterization(self.means, F.normalize(self.quats, dim=-1), torch.exp(self.scales),
                                            torch.sigmoid(self.opacities), colors, views.viewmats[index][None],
                                            views.K[index][None], views.width, views.height, sh_degree=self.sh_degree,
                                            render_mode='RGB', packed=False)
        depth, n_camera, alpha, _ = self._plane(views, index)
        rotation = views.viewmats[index][:3, :3]
        normal = n_camera @ rotation                       # camera -> world
        depth_normal = depth_normals(depth, views.K[index]) @ rotation
        flat = torch.exp(self.scales).min(-1).values
        return {'image': image[0].permute(2, 0, 1), 'depth': depth, 'alpha': alpha,
                'normal': normal * alpha[..., None], 'depth_normal': depth_normal, 'gradient': meta['means2d'],
                'radii': meta['radii'][0], 'regulariser': self.flat_weight * flat.mean()}

    def depth_only(self, views: Views, index: int, export: bool = False) -> torch.Tensor:
        depth, _, alpha, _ = self._plane(views, index)
        return torch.where(alpha > .5, depth, torch.zeros_like(depth)) if export else depth


SH_C0 = 0.28209479177387814
NAMES = ('means', 'quats', 'scales', 'opacities', 'sh0', 'shN', 'labels')   # order matches the optimiser's
# param groups, which rebuild() zips against: a parameter missing here survives densification at its old
# length and the rasteriser then rejects it against the new Gaussian count.


def solve_labels(gaussians: 'Gaussians', views: Views, bias: float = 1.) -> torch.Tensor:
    """Assign each Gaussian a foreground label in closed form, instead of fitting one by gradient descent.

    FlashSplat's observation (Shen et al., ECCV 2024, "2D to 3D Gaussian Splatting Segmentation Solved
    Optimally"; formulation reimplemented here, none of its code used): the rendered mask is a *linear*
    function of the per-Gaussian labels, so the optimal binary assignment decouples - a Gaussian belongs
    to the object when its alpha-blending weight accumulated over foreground pixels exceeds its weight
    over background pixels, scaled by a bias that trades precision for recall.

    Those accumulated weights need no change to the rasteriser: rendering R = sum_i w_i l_i makes dR/dl_i
    exactly the blending weight w_i, so one backward pass of sum_pixels (mask - bias (1 - mask)) R per view
    yields the whole decision statistic, summed over views as the gradient accumulates. The sigmoid on the
    labels only multiplies each Gaussian's gradient by a positive constant, which cannot change the sign,
    so the existing render is reused unaltered.
    """
    if views.masks is None:
        raise ValueError('solve_labels needs masks (--labels)')
    gaussians.labels.grad = None
    with torch.enable_grad():
        for index in range(len(views.numbers)):
            rendered, alpha = gaussians.render_labels(views, [index])
            mask = views.masks[index]
            weight = torch.where(mask > .5, torch.ones_like(mask), -float(bias) * torch.ones_like(mask))
            (rendered[0] * weight * (alpha[0] > .01)).sum().backward()
    statistic = gaussians.labels.grad[:, 0].clone()
    gaussians.labels.grad = None
    return (statistic > 0).float()


def rebuild(gaussians: 'Gaussians', optimiser: torch.optim.Adam, keep: torch.Tensor, added: dict | None = None) -> None:
    """Keep the masked Gaussians, append `added`, and carry Adam's moments over (zero for new ones)."""
    for group, name in zip(optimiser.param_groups, NAMES):
        old = group['params'][0]
        extra = added[name] if added else old[:0]
        new = torch.nn.Parameter(torch.cat([old.data[keep], extra.detach()]))
        state = optimiser.state.pop(old, None)
        if state:
            for key in ('exp_avg', 'exp_avg_sq'):
                state[key] = torch.cat([state[key][keep], torch.zeros_like(extra)])
            optimiser.state[new] = state
        group['params'] = [new]
        setattr(gaussians, name, new)


def densify(gaussians: 'Gaussians', optimiser: torch.optim.Adam, gradient: torch.Tensor, threshold: float,
            split_size: float, min_opacity: float) -> tuple[int, int, int]:
    """Clone small and split large Gaussians whose mean screen gradient exceeds `threshold`; prune transparent ones."""
    with torch.no_grad():
        grow = gradient > threshold
        size = torch.exp(gaussians.scales).max(1).values
        large, small = grow & (size > split_size), grow & (size <= split_size)
        # Split: two surfels at random points of the parent's plane, 1.6× smaller.
        rotation = rotation_matrices(gaussians.quats[large])
        scale = torch.exp(gaussians.scales[large])
        if not isinstance(gaussians, Gaussians3D):
            scale = scale * torch.tensor([1., 1., 0.], device=scale.device)        # surfels: stay in the plane
        children = []
        for _ in range(2):
            offset = torch.randn_like(scale) * scale
            children.append(gaussians.means[large] + (rotation @ offset[..., None])[..., 0])
        added = {'means': torch.cat(children + [gaussians.means[small]]),
                 'quats': torch.cat([gaussians.quats[large]] * 2 + [gaussians.quats[small]]),
                 'scales': torch.cat([gaussians.scales[large] - math.log(1.6)] * 2 + [gaussians.scales[small]]),
                 'opacities': torch.cat([gaussians.opacities[large]] * 2 + [gaussians.opacities[small]]),
                 'sh0': torch.cat([gaussians.sh0[large]] * 2 + [gaussians.sh0[small]]),
                 'shN': torch.cat([gaussians.shN[large]] * 2 + [gaussians.shN[small]]),
                 'labels': torch.cat([gaussians.labels[large]] * 2 + [gaussians.labels[small]])}
        keep = ~large & (torch.sigmoid(gaussians.opacities) > min_opacity)
        pruned = int((~keep).sum()) - int(large.sum())
    rebuild(gaussians, optimiser, keep, added)
    return int(large.sum()), int(small.sum()), pruned


class Priors:
    """Depth from several sources, each gated per pixel (after HD-VGGT's feature modulation).

    Static gate: how far a source's depth at a pixel is supported by the same source in
    neighbouring views (projected point lands within --prior-tolerance of the neighbour's depth)
    rather than contradicted (the neighbour sees through it): supports / (supports + violations),
    0.5 where no neighbour observes it. Dynamic gate, per iteration: exp(-(e / sigma)^2) of the
    source's relative difference e from the rendered depth. The prior term is the gate-weighted mean
    of min(e, clip) over sources and pixels, so a source that is wrong somewhere loses its vote there.
    """

    def __init__(self, views: 'Views', sources: list[tuple[str, Path]], near: list[list[int]], tolerance: float,
                 sequence: Path, device: str):
        self.names = [name for name, _ in sources]
        frames = len(views.numbers)
        self.depth = torch.zeros(len(sources), frames, views.height, views.width, device=device)
        for s, (_, directory) in enumerate(sources):
            for i, number in enumerate(views.numbers):
                path = directory / f'{number:08d}_grounded.npy'
                if path.is_file():
                    self.depth[s, i] = torch.tensor(np.load(path), device=device)
        masks = torch.stack([torch.tensor(cv2.imread(str(sequence / 'masks' / f'{n:08d}.png'), cv2.IMREAD_GRAYSCALE) > 127,
                                          device=device) for n in views.numbers])
        self.depth *= masks[None]                                                      # object pixels only
        self.static = torch.full_like(self.depth, .5)
        v, u = torch.meshgrid(torch.arange(views.height, device=device), torch.arange(views.width, device=device),
                              indexing='ij')
        u, v = u.float(), v.float()
        for s in range(len(sources)):
            for i in range(frames):
                d = self.depth[s, i]
                if not (d > 0).any():
                    continue
                world = (backproject(d, views.K[i], u, v) - views.viewmats[i][:3, 3]) @ views.viewmats[i][:3, :3]
                support = torch.zeros_like(d)
                violation = torch.zeros_like(d)
                for j in near[i]:
                    camera = world @ views.viewmats[j][:3, :3].T + views.viewmats[j][:3, 3]
                    uj, vj, zj = project(camera, views.K[j])
                    inside = (zj > 0) & (uj >= 0) & (uj <= views.width - 1) & (vj >= 0) & (vj <= views.height - 1)
                    observed = torch.zeros_like(d)
                    observed[inside] = self.depth[s, j][vj[inside].round().long(), uj[inside].round().long()]
                    seen = inside & (observed > 0) & (d > 0)
                    support += (seen & ((observed - zj).abs() <= tolerance * zj)).float()
                    violation += (seen & (observed > zj * (1 + tolerance))).float()
                total = support + violation
                self.static[s, i] = torch.where(total > 0, support / total.clamp(min=1), torch.full_like(d, .5))
        self.static *= (self.depth > 0)

    def loss(self, index: int, rendered: torch.Tensor, sigma: float, clip: float) -> torch.Tensor:
        target = self.depth[:, index]
        valid = target > 0
        error = (rendered[None] - target).abs() / target.clamp(min=1e-6)
        gate = (self.static[:, index] * torch.exp(-(error.detach() / sigma) ** 2)) * valid
        total = gate.sum()
        return (gate * error.clamp(max=clip)).sum() / total if total > 0 else rendered.sum() * 0


def trim(gaussians: 'Gaussians', optimiser: torch.optim.Adam, views: 'Views', min_views: int,
         tolerance: float = .01) -> int:
    """Prune Gaussians on the visible surface of fewer than `min_views` training views (PGSR's multi-view trim).

    A Gaussian counts as seen by a view when its centre projects inside the image and lies within
    `tolerance` (relative) of that view's rendered median depth there.
    """
    with torch.no_grad():
        seen = torch.zeros(len(gaussians.means), device=gaussians.means.device)
        for index in range(len(views.numbers)):
            rendered = gaussians.depth_only(views, index, export=True)
            camera = gaussians.means @ views.viewmats[index][:3, :3].T + views.viewmats[index][:3, 3]
            u, v, z = project(camera, views.K[index])
            inside = (z > 0) & (u >= 0) & (u <= views.width - 1) & (v >= 0) & (v <= views.height - 1)
            depth = torch.zeros_like(z)
            depth[inside] = rendered[v[inside].round().long(), u[inside].round().long()]
            seen += (inside & (depth > 0) & ((depth - z).abs() <= tolerance * z)).float()
        keep = seen >= min_views
    rebuild(gaussians, optimiser, keep)
    return int((~keep).sum())


def load_points(paths: list[Path], max_points: int):
    clouds = [o3d.io.read_point_cloud(str(path)) for path in paths]
    cloud = clouds[0]
    for other in clouds[1:]:
        cloud += other
    if len(cloud.points) > max_points:
        cloud = cloud.random_down_sample(max_points / len(cloud.points))
    xyz = np.asarray(cloud.points)
    spacing = np.asarray(cloud.compute_nearest_neighbor_distance())
    if not cloud.has_normals():                       # our clouds carry the depth maps' normals
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(16))
    rgb = np.asarray(cloud.colors) if cloud.has_colors() else np.full_like(xyz, .5)
    return xyz, rgb, np.asarray(cloud.normals), spacing


def depth_seeds(views: 'Views', depth_dir: Path, sequence: Path, count: int, rng: np.random.Generator,
                region: str = 'object'):
    """Surfels from our depth maps: back-projected object (or background) pixels with the depth map's normal."""
    from recon.depth.fusion.photometric import camera_normals
    per_frame = max(1, count // len(views.numbers))
    xyz, rgb, normals = [], [], []
    images = (views.images.permute(0, 2, 3, 1).cpu().numpy())
    for index, number in enumerate(views.numbers):
        path = depth_dir / f'{number:08d}_grounded.npy'
        mask = cv2.imread(str(sequence / 'masks' / f'{number:08d}.png'), cv2.IMREAD_GRAYSCALE)
        if not path.is_file() or mask is None:
            continue
        depth = np.load(path).astype(np.float32)
        K = views.K[index].cpu().numpy().astype(np.float64)
        inside = mask > 127 if region == 'object' else mask <= 127
        v, u = np.nonzero(inside & (depth > 0))
        if not len(u):
            continue
        pick = rng.choice(len(u), min(per_frame, len(u)), replace=False)
        v, u = v[pick], u[pick]
        z = depth[v, u].astype(np.float64)
        camera = np.stack(((u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z), 1)
        R, t = views.rotations[index], views.translations[index]
        xyz.append((camera - t) @ R)                                     # camera -> world (R is world-to-camera)
        normals.append(camera_normals(depth, K)[v, u].astype(np.float64) @ R)
        rgb.append(images[index][v, u])
    xyz = np.concatenate(xyz)
    # Every frame seeds the same surface, so size surfels by their spacing among all seeds.
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    spacing = np.asarray(cloud.compute_nearest_neighbor_distance())
    return xyz, np.concatenate(rgb), np.concatenate(normals), spacing


def backproject(depth: torch.Tensor, K: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.stack(((u - K[0, 2]) / K[0, 0] * depth, (v - K[1, 2]) / K[1, 1] * depth, depth), -1)


def project(points: torch.Tensor, K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = points[..., 2].clamp(min=1e-6)
    return K[0, 0] * points[..., 0] / z + K[0, 2], K[1, 1] * points[..., 1] / z + K[1, 2], points[..., 2]


def sample(image: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Bilinear samples of a (H, W) image at pixel coordinates (any shape)."""
    height, width = image.shape
    grid = torch.stack((2 * u / (width - 1) - 1, 2 * v / (height - 1) - 1), -1).reshape(1, -1, 1, 2)
    return F.grid_sample(image[None, None], grid, align_corners=True)[0, 0, :, 0].reshape(u.shape)


def multiview_losses(views: Views, ref: int, other: int, depth_ref, depth_other, normal_ref, alpha_ref,
                     samples: int, threshold: float, stats: dict | None = None, patch_step: float = 2.,
                     all_pixels: bool = False, ncc_scale: float = 1., squared_ncc: bool = False):
    """PGSR-style geometric (round-trip pixels) and photometric (plane-homography NCC) losses.

    With all_pixels the geometric term uses every valid pixel (as PGSR) and only the NCC term is sampled.
    ncc_scale < 1 matches patches on a downscaled grey image, patch_step pixels apart there. (PGSR's ncc_scale 0.5
    is the opposite: a grey image at twice the training size, 1 px apart there, i.e. patch_step 0.5 here.)
    squared_ncc: PGSR's lncc cost 1 - cc², with patches whose cost is >= 0.9 (no correlation) left out.
    """
    device = depth_ref.device
    valid = (alpha_ref > .5) & (depth_ref > 0)
    v, u = torch.nonzero(valid, as_tuple=True)
    if len(u) < 100:
        return None
    if not all_pixels:
        pick = torch.randperm(len(u), device=device)[:samples]
        u, v = u[pick], v[pick]
    u, v = u.float(), v.float()
    K_r, K_o = views.K[ref], views.K[other]
    relative = views.viewmats[other] @ torch.linalg.inv(views.viewmats[ref])      # ref camera -> other camera
    R, t = relative[:3, :3], relative[:3, 3]
    d = depth_ref[v.long(), u.long()]
    point = backproject(d, K_r, u, v)
    uo, vo, zo = project(point @ R.T + t, K_o)
    inside = (zo > 0) & (uo >= 0) & (uo <= views.width - 1) & (vo >= 0) & (vo <= views.height - 1)
    d_other = sample(depth_other, uo, vo)
    back = (backproject(d_other, K_o, uo, vo) - t) @ R                             # other camera -> ref camera
    ur, vr, _ = project(back, K_r)
    error = torch.sqrt((ur - u) ** 2 + (vr - v) ** 2 + 1e-12)                    # finite gradient at zero
    use = inside & (d_other > 0) & (error < threshold)
    if stats is not None:
        stats['sampled'] += len(u); stats['used'] += int(use.sum())
        stats['error_median'].append(float(error[inside & (d_other > 0)].median()) if (inside & (d_other > 0)).any() else float('nan'))
    if use.sum() < 50:
        return None
    weight = torch.exp(-error).detach()
    geometric = (weight * error)[use].mean()
    if all_pixels:                                                                  # sample for the NCC term
        pick = torch.nonzero(use, as_tuple=True)[0]
        pick = pick[torch.randperm(len(pick), device=device)[:samples]]
        u, v, point, weight, use = u[pick], v[pick], point[pick], weight[pick], use[pick]
    # Plane through the point with the rendered normal (camera frame), homography ref -> other.
    n = normal_ref[v.long(), u.long()] @ views.viewmats[ref][:3, :3].T
    n = F.normalize(n, dim=-1)
    plane = (n * point).sum(-1, keepdim=True)                                       # n·X = plane
    offsets = torch.stack(torch.meshgrid(torch.arange(-3, 4, device=device), torch.arange(-3, 4, device=device),
                                         indexing='xy'), -1).reshape(-1, 2).float() * patch_step / ncc_scale
    pu, pv = u[:, None] + offsets[:, 0], v[:, None] + offsets[:, 1]
    rays = torch.stack(((pu - K_r[0, 2]) / K_r[0, 0], (pv - K_r[1, 2]) / K_r[1, 1], torch.ones_like(pu)), -1)
    facing = (rays * n[:, None]).sum(-1, keepdim=True)                             # negative: normals face the camera
    facing = torch.where(facing.abs() < 1e-6, torch.full_like(facing, -1e-6), facing)
    scale = plane[:, :, None] / facing                                              # distance along each patch ray
    hu, hv, hz = project((rays * scale) @ R.T + t, K_o)
    gray = views.gray_scaled(ncc_scale)
    to = lambda x: (x + .5) * ncc_scale - .5                                        # full-resolution pixel -> scaled grid
    a = sample(gray[ref], to(pu), to(pv))
    b = sample(gray[other], to(hu), to(hv))
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    if squared_ncc:                  # PGSR's lncc: 1 - cc², and patches that do not correlate (loss >= 0.9) are left out
        cost = (1 - (a * b).sum(1) ** 2 / ((a * a).sum(1) * (b * b).sum(1) + 1e-8)).clamp(0, 2)
        use = use & (cost < .9)
        if use.sum() < 50:
            return geometric, None
    else:
        cost = (1 - (a * b).sum(1) / torch.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-8)).clamp(0, 2)
    photometric = (weight * cost)[use].mean()
    return geometric, photometric


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('sequence', type=Path, help='Prepared sequence (frames/)')
    p.add_argument('--model', type=Path, required=True, help='Our COLMAP text model (sparse_txt)')
    p.add_argument('--init-points', type=Path, nargs='+', required=True, help='Our point clouds (object, scene)')
    p.add_argument('--init-depth', type=Path,
                   help='Seed the object from these depth maps (<frame>_grounded.npy) instead of the first point '
                        'cloud; the other clouds (e.g. the scene) are still used')
    p.add_argument('--depth-seeds', type=int, default=300_000, help='Object surfels drawn from --init-depth')
    p.add_argument('--background-seeds', type=int, default=0,
                   help='Background surfels drawn from --init-depth outside the masks (0: use the scene cloud)')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=3000)
    p.add_argument('--max-points', type=int, default=300_000)
    p.add_argument('--neighbours', type=int, default=6)
    p.add_argument('--geo-from', type=int, default=500)
    p.add_argument('--normal-weight', type=float, default=.015)
    p.add_argument('--normal-from', type=int, default=0, help='First iteration of the depth-normal loss')
    p.add_argument('--geo-weight', type=float, default=.03)
    p.add_argument('--ncc-weight', type=float, default=.15)
    p.add_argument('--geo-threshold', type=float, default=1.0, help='Round-trip error (pixels) still used')
    p.add_argument('--samples', type=int, default=102400)
    p.add_argument('--patch-step', type=float, default=.5,
                   help='Pixels between the 7×7 NCC patch samples (at --ncc-scale); 0.5 as PGSR (3.5-px patch)')
    p.add_argument('--ncc-scale', type=float, default=1., help='Resolution of the grey images patches are matched on')
    p.add_argument('--geo-all-pixels', action=argparse.BooleanOptionalAction, default=True,
                   help='Geometric term on every valid pixel (PGSR)')
    p.add_argument('--pgsr-normal-loss', action=argparse.BooleanOptionalAction, default=True,
                   help="L1 between rendered and depth normals weighted by (1 - image gradient)² (PGSR's single-view loss)")
    p.add_argument('--squared-ncc', action=argparse.BooleanOptionalAction, default=True,
                   help="PGSR's NCC cost: 1 - cc², leaving out patches that do not correlate (cost >= 0.9)")
    p.add_argument('--position-lr', type=float, nargs=2, default=(1.6e-4, 1.6e-6), metavar=('START', 'END'),
                   help='Position learning rate × scene radius, decayed exponentially (3DGS / PGSR defaults)')
    p.add_argument('--export-gaussians', type=Path,
                   help='Also write the fitted Gaussians as a 3DGS-layout PLY (recon.surface.splat_views)')
    p.add_argument('--principal-shift', type=float, default=0.,
                   help='Shift our principal point (pixels) for --init-gaussians fitted in another frame (PGSR: 0.5)')
    p.add_argument('--init-gaussians', type=Path,
                   help='Start from trained 3D Gaussians (3DGS / PGSR point_cloud.ply in a normalised frame) instead '
                        'of seeding; with --init-frame LANDMARKS the frame is undone as twodgs.normalisation')
    p.add_argument('--init-frame', type=Path, help='Landmarks the --init-gaussians frame was normalised by')
    p.add_argument('--trim-every', type=int, default=0, help='Multi-view trim every N iterations (PGSR: 1000; 0 = off)')
    p.add_argument('--trim-views', type=int, default=2, help='Prune Gaussians seen on the surface by fewer views')
    p.add_argument('--sh-degree', type=int, default=3, help='Spherical-harmonic colour degree, raised every 1000 iterations')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--log-every', type=int, default=0, help='Print loss terms and multi-view usage every N iterations')
    p.add_argument('--prior', nargs=2, action='append', default=[], metavar=('NAME', 'DIR'),
                   help='A depth source kept as a gated prior (repeat for several)')
    p.add_argument('--prior-weight', type=float, default=0.)
    p.add_argument('--labels', type=Path,
                   help='Object masks (<frame>.png) to supervise a per-Gaussian foreground label, for 3D '
                        'segmentation after an unmasked reconstruction; the label never moves the geometry')
    p.add_argument('--label-weight', type=float, default=.5)
    p.add_argument('--label-from', type=int, default=500, help='Iteration the label loss starts at')
    p.add_argument('--export-labels', type=Path, help='Write means and foreground probability as .npz')
    p.add_argument('--label-solver', choices=('fit', 'closed-form'), default='fit',
                   help="fit trains the label by cross-entropy; closed-form solves it optimally in one "
                        "pass per view (FlashSplat's formulation), which needs no label loss at all")
    p.add_argument('--label-bias', type=float, default=1.,
                   help='Background weight in the closed-form solve; above 1 demands more foreground '
                        'evidence (higher precision), below 1 keeps more surface (higher recall)')
    p.add_argument('--prior-sigma', type=float, default=.005, help='Dynamic gate width (relative depth)')
    p.add_argument('--prior-clip', type=float, default=.01, help='Cap on the per-pixel relative difference')
    p.add_argument('--prior-tolerance', type=float, default=.005, help='Static gate: relative agreement across views')
    p.add_argument('--prior-static', action=argparse.BooleanOptionalAction, default=True,
                   help='Use the cross-view static gate (off: every source pixel weighs 1)')
    p.add_argument('--densify-from', type=int, default=500)
    p.add_argument('--densify-until', type=int, default=1500, help='0 turns densification off')
    p.add_argument('--densify-every', type=int, default=100)
    p.add_argument('--grad-threshold', type=float, default=.0002, help='Mean screen-space gradient that triggers growth')
    p.add_argument('--split-size', type=float, default=.01, help='Surfels larger than this × scene radius are split')
    p.add_argument('--export', choices=('median', 'expected'), default='median',
                   help='Depth written out: where accumulated opacity passes 0.5 (2DGS), or the alpha-weighted mean')
    p.add_argument('--init-opacity', type=float, default=.9)
    p.add_argument('--init-size', type=float, default=.7, help='Starting surfel radius / nearest-neighbour spacing')
    a = p.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = 'cuda'
    started = time.time()
    views = Views(a.sequence, a.model, device, a.principal_shift, a.labels)
    near = neighbours(views.rotations, views.translations, a.neighbours, (10., 35.))
    if a.init_depth is not None:
        parts = [depth_seeds(views, a.init_depth, a.sequence, a.depth_seeds, rng)]
        if a.background_seeds > 0:
            parts.append(depth_seeds(views, a.init_depth, a.sequence, a.background_seeds, rng, 'background'))
        elif len(a.init_points) > 1:
            parts.append(load_points(a.init_points[1:], a.max_points))
        xyz, rgb, normals, spacing = (np.concatenate(values) for values in zip(*parts))
    else:
        xyz, rgb, normals, spacing = load_points(a.init_points, a.max_points)
    if a.init_gaussians is not None:
        from recon.evaluation.baselines.twodgs import normalisation
        center, radius = normalisation(a.init_frame) if a.init_frame is not None else (np.zeros(3), 1.)
        gaussians = load_gaussians(a.init_gaussians, device, center, radius)
    else:
        gaussians = Gaussians(xyz, rgb, normals, spacing, device, a.init_opacity, a.init_size)
        gaussians.export = a.export
    lr = lambda start, end, step: start * (end / start) ** min(step / a.iterations, 1)
    optimiser = torch.optim.Adam([
        {'params': [gaussians.means], 'lr': a.position_lr[0] * views.radius, 'name': 'means'},
        {'params': [gaussians.quats], 'lr': 1e-3}, {'params': [gaussians.scales], 'lr': 5e-3},
        {'params': [gaussians.opacities], 'lr': 5e-2}, {'params': [gaussians.sh0], 'lr': 2.5e-3},
        {'params': [gaussians.shN], 'lr': 2.5e-3 / 20},
        {'params': [gaussians.labels], 'lr': 5e-2}], eps=1e-15)
    priors = None
    if a.prior and a.prior_weight > 0:
        priors = Priors(views, [(name, Path(directory)) for name, directory in a.prior], near, a.prior_tolerance,
                        a.sequence, device)
        if not a.prior_static:
            priors.static = (priors.depth > 0).float()
    setup = time.time() - started
    order = []
    accum = torch.zeros(len(gaussians.means), device=device)
    count = torch.zeros_like(accum)
    grown = {'split': 0, 'cloned': 0, 'pruned': 0}
    stats = {'sampled': 0, 'used': 0, 'error_median': [], 'terms': {'rgb': [], 'normal': [], 'geo': [], 'ncc': []},
             'active': 0, 'steps': 0}
    start_degree = gaussians.sh_degree                    # loaded Gaussians keep theirs
    for step in range(a.iterations):
        optimiser.param_groups[0]['lr'] = lr(*a.position_lr, step) * views.radius
        gaussians.sh_degree = max(start_degree, min(a.sh_degree, step // 1000))
        if not order:
            order = list(rng.permutation(len(views.numbers)))
        ref = int(order.pop())
        out = gaussians(views, ref)
        densifying = a.densify_from <= step < a.densify_until
        if densifying:
            out['gradient'].retain_grad()
        image, depth, alpha = out['image'], out['depth'], out['alpha']
        normals_r, surface = out['normal'], out['depth_normal']
        target = views.images[ref]
        rgb_loss = .8 * (image - target).abs().mean() + .2 * (1 - ssim(image, target))
        if a.pgsr_normal_loss:
            normal_loss = (views.gradient_weight(ref) * (surface * alpha[..., None] - normals_r).abs().sum(-1)).mean()
        else:
            normal_loss = (alpha * (1 - (normals_r * surface).sum(-1))).mean()
        loss = rgb_loss + (a.normal_weight if step >= a.normal_from else 0.) * normal_loss + out['regulariser']
        if views.masks is not None and a.label_weight > 0 and step >= a.label_from:
            # Binary cross-entropy of the composited foreground probability against the mask, wherever the
            # render has coverage. Every view votes, so a mask that eats part of the object in one frame is
            # outvoted instead of deleting that surface, which is what costs 12.6 points in object mode.
            label, label_alpha = gaussians.render_labels(views, [ref])
            covered = label_alpha[0] > .1
            if covered.any():
                loss = loss + a.label_weight * F.binary_cross_entropy(
                    label[0][covered].clamp(1e-6, 1 - 1e-6), views.masks[ref][covered])
        if priors is not None:
            prior_loss = priors.loss(ref, depth, a.prior_sigma, a.prior_clip)
            loss = loss + a.prior_weight * prior_loss
        stats['terms']['rgb'].append(float(rgb_loss)); stats['terms']['normal'].append(float(normal_loss))
        if step >= a.geo_from and near[ref]:
            stats['steps'] += 1
            other = int(near[ref][rng.integers(len(near[ref]))])
            depth_o = gaussians.depth_only(views, other)
            terms = multiview_losses(views, ref, other, depth, depth_o, normals_r, alpha,
                                     a.samples, a.geo_threshold, stats, a.patch_step, a.geo_all_pixels, a.ncc_scale,
                                     a.squared_ncc)
            if terms is not None:
                stats['active'] += 1
                stats['terms']['geo'].append(float(terms[0]))
                loss = loss + a.geo_weight * terms[0]
                if terms[1] is not None:
                    stats['terms']['ncc'].append(float(terms[1]))
                    loss = loss + a.ncc_weight * terms[1]
        if a.log_every and (step + 1) % a.log_every == 0:
            mean = lambda x: float(np.mean(x[-a.log_every:])) if x else float('nan')
            t = stats['terms']
            print(f"step {step + 1}: rgb {mean(t['rgb']):.4f} normal {mean(t['normal']):.4f} "
                  f"geo {mean(t['geo']):.4f} ncc {mean(t['ncc']):.4f} | multi-view active {stats['active']}/{stats['steps']}, "
                  f"samples used {stats['used'] / max(stats['sampled'], 1):.0%}, round-trip median "
                  f"{np.nanmedian(stats['error_median'][-a.log_every:]) if stats['error_median'] else float('nan'):.2f} px, "
                  f"opacity mean {float(torch.sigmoid(gaussians.opacities).mean()):.2f}", flush=True)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        if densifying:
            with torch.no_grad():
                grad = out['gradient'].grad[0].clone()
                grad[:, 0] *= views.width / 2
                grad[:, 1] *= views.height / 2
                radii = out['radii']
                seen = (radii > 0).all(-1) if radii.dim() == 2 else radii > 0
                accum[seen] += grad.norm(dim=-1)[seen]
                count[seen] += 1
        optimiser.step()
        if a.trim_every and (step + 1) % a.trim_every == 0 and step + 1 < a.iterations:
            grown['trimmed'] = grown.get('trimmed', 0) + trim(gaussians, optimiser, views, a.trim_views)
            accum = torch.zeros(len(gaussians.means), device=device)
            count = torch.zeros_like(accum)
        if densifying and (step + 1) % a.densify_every == 0:
            counts = densify(gaussians, optimiser, accum / count.clamp(min=1), a.grad_threshold,
                             a.split_size * views.radius, .005)
            grown['split'] += counts[0]; grown['cloned'] += counts[1]; grown['pruned'] += counts[2]
            accum = torch.zeros(len(gaussians.means), device=device)
            count = torch.zeros_like(accum)
    trained = time.time() - started - setup
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index, number in enumerate(views.numbers):
            depth = gaussians.depth_only(views, index, export=True)
            np.save(a.output_dir / f'{number:08d}_grounded.npy', depth.cpu().numpy().astype(np.float32))
    if a.export_gaussians is not None:
        save_gaussians(gaussians, a.export_gaussians)
    if a.export_labels is not None:
        if a.label_solver == 'closed-form':
            solved = solve_labels(gaussians, views, a.label_bias)
            probability = solved.detach().cpu().numpy()
        else:
            with torch.no_grad():
                probability = torch.sigmoid(gaussians.labels[:, 0]).cpu().numpy()
        a.export_labels.parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.export_labels, means=gaussians.means.detach().cpu().numpy(), foreground=probability)
        print(f'Labels: {float((probability > .5).mean()) * 100:.1f}% of {len(probability):,} '
              f'Gaussians foreground -> {a.export_labels}')
    summary = {'method': 'planar Gaussian refinement (PGSR-style losses, gsplat 2DGS rasteriser)', 'gaussians': len(xyz),
               'final_gaussians': len(gaussians.means), 'densification': grown,
               'priors': [list(x) for x in a.prior] if priors is not None else [], 'prior_weight': a.prior_weight,
               'prior_static': a.prior_static,
               'prior_gate_means': {n: float(priors.static[k][priors.depth[k] > 0].mean()) for k, n in enumerate(priors.names)}
               if priors is not None else {},
               'iterations': a.iterations, 'setup_s': setup, 'train_s': trained, 'export_s': time.time() - started - setup - trained,
               'weights': {'normal': a.normal_weight, 'geometric': a.geo_weight, 'ncc': a.ncc_weight}, 'frames': len(views.numbers)}
    (a.output_dir / 'hybrid_depth.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
