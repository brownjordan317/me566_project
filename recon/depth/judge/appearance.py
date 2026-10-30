#!/usr/bin/env python3
"""What the surface looks like, for the learned judge: DINOv2 features of the full-resolution photo.

Which depth source fails where depends on the material (dark, glossy and transparent surface defeat the
models in different ways), which a few colour channels cannot tell. Each photo goes through DINOv2
ViT-L/14 at its own resolution (resized by at most 13 px to a multiple of the patch), and the patch
features are projected onto APPEARANCE principal components fitted on training-split photos only. The
judge samples them onto its evidence grid (recon.depth.judge.train.load_frame).

DINOv2: M. Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", TMLR 2024;
code and weights Apache-2.0 (github.com/facebookresearch/dinov2), loaded from the local torch hub cache.

  fit    sample patch features from the training sets' photos and save the projection (.npz);
  write  write <frame>_appearance.npy (APPEARANCE, th, tw) fp16 into judge-data folders, and link each
         folder's photos as photos/ for the judge's full-resolution colour branch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from recon.common.progress import bar, done

APPEARANCE = 64
PATCH = 14
HUB = Path.home() / '.cache/torch/hub/facebookresearch_dinov2_main'
MEAN, STD = np.array([.485, .456, .406], np.float32), np.array([.229, .224, .225], np.float32)


def load_backbone(device: str) -> torch.nn.Module:
    return torch.hub.load(str(HUB), 'dinov2_vitl14', source='local').to(device).eval()


def photo_files(directory: Path) -> dict[int, Path]:
    """{frame number: photo} by file stem, as recon.depth.fusion.photometric.Images matches them."""
    return {int(p.stem): p for p in directory.iterdir()
            if p.suffix.lower() in ('.png', '.jpg', '.jpeg') and p.stem.isdigit()}


@torch.no_grad()
def patch_features(backbone: torch.nn.Module, photo: np.ndarray, device: str) -> torch.Tensor:
    """DINOv2 patch features (1024, th, tw) of one full-resolution BGR photo."""
    height, width = photo.shape[:2]
    size = (width // PATCH * PATCH, height // PATCH * PATCH)
    rgb = cv2.cvtColor(cv2.resize(photo, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = torch.from_numpy((rgb.astype(np.float32) / 255 - MEAN) / STD).permute(2, 0, 1)[None].to(device)
    with torch.autocast('cuda', torch.bfloat16):
        tokens = backbone.forward_features(x)['x_norm_patchtokens'][0].float()
    return tokens.T.reshape(-1, size[1] // PATCH, size[0] // PATCH)


def project(features: torch.Tensor, projection: dict) -> np.ndarray:
    """Patch features (1024, th, tw) onto the fitted components, about unit variance: (APPEARANCE, th, tw)."""
    T = lambda key: torch.as_tensor(projection[key], device=features.device)
    flat = features.flatten(1).T - T('mean')
    out = (flat @ T('components').T) / T('scale')
    return out.T.reshape(-1, *features.shape[1:]).cpu().numpy().astype(np.float16)


def judge_folders(root: Path) -> list[Path]:
    return sorted(f for f in root.iterdir() if (f / 'meta.json').is_file())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=('fit', 'write'))
    p.add_argument('data', type=Path, help='Folder of judge-data folders (recon.depth.judge.dataset outputs)')
    p.add_argument('--photos', type=Path, required=True, help='Folder of <set>/image/ full-resolution photos')
    p.add_argument('--projection', type=Path, required=True, help='Projection (.npz) to fit or use')
    p.add_argument('--val-every', type=int, default=8, help='As recon.depth.judge.train: these sets are not fitted')
    p.add_argument('--per-frame', type=int, default=400, help='Patches sampled per photo when fitting')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    folders = judge_folders(a.data)
    backbone = load_backbone(a.device)
    if a.mode == 'fit':
        from recon.depth.multiview.da3_distill import set_split
        train, _ = set_split([f.name for f in folders], a.val_every)
        rng, samples = np.random.default_rng(0), []
        for folder in bar([f for f in folders if f.name in train], desc='Appearance samples', unit='set'):
            files = photo_files(a.photos / folder.name / 'image')
            frames = json.loads((folder / 'meta.json').read_text())['frames']
            for number in rng.choice(frames, min(4, len(frames)), replace=False):
                flat = patch_features(backbone, cv2.imread(str(files[int(number)])), a.device).flatten(1).T
                samples.append(flat[torch.from_numpy(rng.choice(len(flat), a.per_frame, replace=False))].cpu())
        x = torch.cat(samples).double()
        mean = x.mean(0)
        _, values, vectors = torch.linalg.svd(x - mean, full_matrices=False)
        components = vectors[:APPEARANCE]
        scale = values[:APPEARANCE] / np.sqrt(len(x) - 1)
        a.projection.parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.projection, mean=mean.float().numpy(), components=components.float().numpy(),
                 scale=scale.float().numpy())
        kept = float((values[:APPEARANCE] ** 2).sum() / (values ** 2).sum())
        done(f'Appearance projection from {len(x)} patches, {kept:.0%} of the variance kept -> {a.projection}')
        return
    projection = dict(np.load(a.projection))
    for folder in bar(folders, desc='Appearance', unit='set'):
        directory = (a.photos / folder.name / 'image').resolve()
        link = folder / 'photos'
        if not link.exists():
            link.symlink_to(directory)
        files = photo_files(directory)
        for number in json.loads((folder / 'meta.json').read_text())['frames']:
            out = folder / f'{number:08d}_appearance.npy'
            if not out.is_file():
                np.save(out, project(patch_features(backbone, cv2.imread(str(files[number])), a.device), projection))
    done(f'Appearance written for {len(folders)} sets')


if __name__ == '__main__':
    main()
