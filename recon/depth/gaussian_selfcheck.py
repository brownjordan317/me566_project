#!/usr/bin/env python3
"""The planar-Gaussian pass's own account of each pixel, for the learned judge (recon.depth.judge).

The fitted surfels (`gaussian_refine --export-gaussians`) are rendered again from every camera and three
self-checks are written per frame to <frame>_selfcheck.npz, beside the pass's depth:

  confidence  accumulated opacity: how solidly the surface is there;
  residual    |render - photo|, mean over RGB and blurred over 5x5: where the fit could not explain the
              photo (a highlight, a moving shadow, a part it got wrong);
  normal      1 - cos between the rendered surfel normals and the normals of the rendered depth: where
              the surface the surfels describe and the depth they export disagree.

Runs in the gsplat environment (make setup-gaussian), like the pass itself.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from recon.common.progress import bar, done
from recon.depth.gaussian_refine import SH_C0, Gaussians, Views


def load_surfels(path: Path, device: str) -> Gaussians:
    """Surfels written by gaussian_refine.save_gaussians, as the 2DGS model that fitted them."""
    from plyfile import PlyData
    vertex = PlyData.read(str(path))['vertex']
    column = lambda *names: np.stack([np.asarray(vertex[n]) for n in names], 1).astype(np.float32)
    rest = column(*[f'f_rest_{i}' for i in range(45)]).reshape(-1, 3, 15).transpose(0, 2, 1)   # channel-major
    t = lambda x: torch.nn.Parameter(torch.tensor(np.ascontiguousarray(x), dtype=torch.float32, device=device))
    gaussians = Gaussians.__new__(Gaussians)
    torch.nn.Module.__init__(gaussians)
    gaussians.means = t(column('x', 'y', 'z'))
    gaussians.quats = t(column('rot_0', 'rot_1', 'rot_2', 'rot_3'))
    gaussians.scales = t(column('scale_0', 'scale_1', 'scale_2'))
    gaussians.opacities = t(column('opacity')[:, 0])
    gaussians.sh0, gaussians.shN = t(column('f_dc_0', 'f_dc_1', 'f_dc_2')[:, None]), t(rest)
    gaussians.sh_degree, gaussians.export = 3, 'median'
    return gaussians


def selfcheck(gaussians: Gaussians, views: Views, index: int) -> dict:
    """Confidence, residual and normal disagreement (H, W) of one frame."""
    colors, alphas, normals, surface, *_ = gaussians.render(views, [index])
    alpha = alphas[0, ..., 0]
    residual = (colors[0, ..., :3].permute(2, 0, 1) - views.images[index]).abs().mean(0)
    residual = F.avg_pool2d(residual[None, None], 5, 1, 2, count_include_pad=False)[0, 0]
    surface = surface if surface.dim() == 3 else surface[0]
    cosine = (F.normalize(normals[0], dim=-1) * F.normalize(surface, dim=-1)).sum(-1)
    solid = alpha > .5
    return {'confidence': alpha, 'residual': torch.where(solid, residual, 0),
            'normal': torch.where(solid, (1 - cosine).clamp(0, 2), 0)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('sequence', type=Path, help='Prepared sequence (frames/)')
    p.add_argument('--model', type=Path, required=True, help='Our COLMAP text model (sparse_txt)')
    p.add_argument('--gaussians', type=Path, required=True, help='gaussian_refine --export-gaussians PLY')
    p.add_argument('--output-dir', type=Path, required=True, help='Usually the pass\'s depth folder')
    p.add_argument('--check-depth', action='store_true',
                   help='Report how closely the re-rendered depth matches the pass\'s depth beside the PLY')
    a = p.parse_args()
    views = Views(a.sequence, a.model, 'cuda')
    gaussians = load_surfels(a.gaussians, 'cuda')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    differences = []
    with torch.no_grad():
        for index, number in enumerate(bar(views.numbers, desc='Gaussian self-check', unit='frame')):
            checks = {k: v.cpu().numpy().astype(np.float16) for k, v in selfcheck(gaussians, views, index).items()}
            np.savez_compressed(a.output_dir / f'{number:08d}_selfcheck.npz', **checks)
            saved = a.gaussians.parent / f'{number:08d}_grounded.npy'
            if a.check_depth and saved.is_file():
                depth = gaussians.depth_only(views, index, export=True).cpu().numpy()
                old = np.load(saved)
                both = (depth > 0) & (old > 0)
                differences.append(float(np.median(np.abs(depth[both] - old[both]) / old[both])) if both.any() else 0.)
    if differences:
        print(f'Re-rendered depth: median relative difference {np.median(differences):.2e}')
    done(f'Gaussian self-check: {len(views.numbers)} frames -> {a.output_dir}')


if __name__ == '__main__':
    main()
