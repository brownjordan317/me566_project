#!/usr/bin/env python3
"""Remove specular highlights from a capture, frame by frame, using the other frames as evidence.

A single-image highlight remover (e.g. TSHRNet, Fu et al., ICCV 2023) has to guess whether a bright patch is a
reflection or a bright material, and on LEGO it guesses wrong: it greys white bricks (docs/PIPELINE.md). A video
does not have to guess. A highlight is view-dependent - it moves across the surface as the camera moves, so a
surface point is bright in a few frames and its own colour in the rest - while a white brick is white in every
frame. This is the classic multi-view separation (Lin et al., "Diffuse-Specular Separation and Depth Recovery
from Image Sequences", ECCV 2002), with the correspondence it needs taken from the pipeline instead of searched
for: the cross-view-checked depth, which sits at the reference's own frame-to-frame floor even on glossy scenes,
and the solved cameras.

For each frame, every pixel with depth is projected into its --neighbours nearest frames; views where something
else is in front (their depth disagrees by more than --occlusion) are dropped, and each neighbour's exposure is
matched to the frame's by the median ratio over the pixels both see. The pixel's view-independent colour is the
--quantile of luminance across the surviving views, its own included. Where the frame is brighter than that by
more than --threshold, with at least --min-views views, the excess is a highlight: the frame takes the consensus
colour there (feathered at the edges) and keeps its own pixels everywhere else. The correction is computed at the
working resolution and applied to the full-resolution photographs (--images), so texture detail is the
photograph's own. Writes the specular-free photographs, a highlight map per frame and a summary.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from recon.common.colmap import parse_model
from recon.common.progress import bar, done

LUMA = torch.tensor([.299, .587, .114])


def neighbours(centres: np.ndarray, axes: np.ndarray, count: int) -> list[list[int]]:
    """For each frame, the nearest frames that look the same way (camera centre distance, viewing direction)."""
    distance = np.linalg.norm(centres[:, None] - centres[None], axis=-1)
    facing = axes @ axes.T
    distance[facing < .5] = np.inf                         # more than 60 degrees apart sees different surface
    np.fill_diagonal(distance, np.inf)
    return [list(np.argsort(row)[:count][np.isfinite(np.sort(row)[:count])]) for row in distance]


def consensus(own: torch.Tensor, samples: torch.Tensor, valid: torch.Tensor, quantile: float):
    """Per pixel, the sample (own colour included) at the given luminance quantile, and how many views there were.

    own (P, 3), samples (V, P, 3), valid (V, P). Missing views take +inf luminance so they sort last and are
    never chosen while the quantile index stays inside the views that exist."""
    colours = torch.cat([own[None], samples], 0)                              # (V+1, P, 3)
    present = torch.cat([torch.ones_like(valid[:1]), valid], 0)
    luma = (colours * LUMA.to(colours)).sum(-1)
    luma = torch.where(present, luma, torch.full_like(luma, float('inf')))
    order = luma.argsort(0)
    count = present.sum(0)
    pick = ((count - 1).float() * quantile).round().long()                     # index into the sorted views
    chosen = order.gather(0, pick[None])[0]
    return colours[chosen, torch.arange(colours.shape[1])], count


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('sequence', type=Path, help='Prepared sequence (frames/, masks/)')
    p.add_argument('--model', type=Path, required=True, help='COLMAP text model (working-frame intrinsics)')
    p.add_argument('--depth-dir', type=Path, required=True, help='Grounded depth to align frames with')
    p.add_argument('--images', type=Path, help='Full-resolution photographs (numbered) to correct')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--neighbours', type=int, default=8)
    p.add_argument('--quantile', type=float, default=.5, help='Luminance quantile taken as the diffuse colour')
    p.add_argument('--threshold', type=float, default=.06, help='Excess brightness (0-1) counted as a highlight')
    p.add_argument('--min-views', type=int, default=3)
    p.add_argument('--occlusion', type=float, default=.01, help='Relative depth gap that marks a view occluded')
    p.add_argument('--feather', type=int, default=5, help='Pixels over which the correction fades out')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    device = a.device
    intrinsics, _, frames = parse_model(a.model)
    numbers = [n for n, *_ in frames if (a.depth_dir / f'{n:08d}_grounded.npy').is_file()]
    poses = {n: (R, t) for n, R, t, _ in frames}
    centres = np.array([-poses[n][0].T @ poses[n][1] for n in numbers])
    axes = np.array([poses[n][0][2] for n in numbers])
    near = neighbours(centres, axes, a.neighbours)
    # Every frame is a neighbour of about --neighbours others, so each is read once and kept on the GPU
    # (half precision) rather than read again for every frame that looks at it.
    pool = ThreadPoolExecutor(8)
    read = lambda n: cv2.cvtColor(cv2.imread(str(a.sequence / 'frames' / f'{n:08d}.png')), cv2.COLOR_BGR2RGB)
    frames_gpu = {n: torch.tensor(img, device=device).half() / 255 for n, img in zip(numbers, pool.map(read, numbers))}
    load = lambda n: frames_gpu[n].float()
    depth = {n: torch.tensor(d, dtype=torch.float32, device=device) for n, d in
             zip(numbers, pool.map(lambda n: np.load(a.depth_dir / f'{n:08d}_grounded.npy'), numbers))}
    full = {int(path.stem): path for path in a.images.iterdir() if path.stem.isdigit()} if a.images else {}
    a.output_dir.mkdir(parents=True, exist_ok=True)
    # Full-resolution photographs are read ahead and written behind on worker threads, so disk overlaps the GPU.
    photos = {n: pool.submit(lambda path: cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB), full[n])
              for n in numbers if n in full}
    writes = []
    reports = []
    for index, n in enumerate(bar(numbers, desc='Highlight removal', unit='frame')):
        image, d = load(n), depth[n]
        height, width = d.shape
        K = torch.tensor(intrinsics[n], dtype=torch.float32, device=device)
        R, t = (torch.tensor(x, dtype=torch.float32, device=device) for x in poses[n])
        v, u = torch.nonzero(d > 0, as_tuple=True)
        z = d[v, u]
        camera = torch.stack(((u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z), -1)
        world = (camera - t) @ R
        own = image[v, u]
        samples, valid = [], []
        for j in near[index]:
            m = numbers[j]
            Kj = torch.tensor(intrinsics[m], dtype=torch.float32, device=device)
            Rj, tj = (torch.tensor(x, dtype=torch.float32, device=device) for x in poses[m])
            other = world @ Rj.T + tj
            uj = Kj[0, 0] * other[:, 0] / other[:, 2] + Kj[0, 2]
            vj = Kj[1, 1] * other[:, 1] / other[:, 2] + Kj[1, 2]
            inside = (other[:, 2] > 0) & (uj >= 0) & (uj <= width - 1) & (vj >= 0) & (vj <= height - 1)
            seen = depth[m][vj.round().long().clamp(0, height - 1), uj.round().long().clamp(0, width - 1)]
            ok = inside & (seen > 0) & ((seen - other[:, 2]).abs() < a.occlusion * other[:, 2])
            grid = torch.stack([uj / (width - 1) * 2 - 1, vj / (height - 1) * 2 - 1], -1)[None, None]
            colour = F.grid_sample(load(m).permute(2, 0, 1)[None], grid, align_corners=True)[0, :, 0].T
            # Exposure drifts through a phone video: match this view to the frame on the pixels both see,
            # ignoring near-black and near-saturated ones.
            luma_own, luma_other = (own * LUMA.to(own)).sum(-1), (colour * LUMA.to(colour)).sum(-1)
            usable = ok & (luma_own > .05) & (luma_own < .95) & (luma_other > .05) & (luma_other < .95)
            gain = (luma_own[usable] / luma_other[usable]).median() if usable.sum() > 100 else torch.tensor(1., device=device)
            samples.append((colour * gain).clamp(0, 1)); valid.append(ok)
        if samples:
            diffuse, count = consensus(own, torch.stack(samples), torch.stack(valid), a.quantile)
            excess = ((own - diffuse) * LUMA.to(own)).sum(-1)
            highlight = (excess > a.threshold) & (count >= a.min_views + 0)
        else:
            diffuse, highlight = own, torch.zeros(len(own), dtype=torch.bool, device=device)
        # The correction (consensus minus frame) on highlight pixels, feathered so it fades rather than steps.
        delta = torch.zeros(height, width, 3, device=device)
        weight = torch.zeros(height, width, device=device)
        delta[v[highlight], u[highlight]] = diffuse[highlight] - own[highlight]
        weight[v[highlight], u[highlight]] = 1
        if a.feather > 0:
            size = 2 * a.feather + 1
            kernel = torch.ones(1, 1, size, size, device=device) / size ** 2
            soft = F.conv2d(weight[None, None], kernel, padding=a.feather)[0, 0]
            delta = F.conv2d(delta.permute(2, 0, 1)[:, None], kernel, padding=a.feather)[:, 0].permute(1, 2, 0) / soft.clamp(min=1e-6)[..., None]
            weight = soft.clamp(0, 1)
        correction = (delta * weight[..., None]).cpu().numpy()
        if n in full:
            photo = photos.pop(n).result()
            correction = cv2.resize(correction, (photo.shape[1], photo.shape[0]), interpolation=cv2.INTER_LINEAR)
            name = full[n].stem
        else:
            photo, name = (image.cpu().numpy() * 255), f'{n:08d}'
        corrected = np.clip(photo + correction * 255, 0, 255).round().astype(np.uint8)
        fast = [cv2.IMWRITE_PNG_COMPRESSION, 1]                             # lossless, a fraction of the default's time
        writes.append(pool.submit(cv2.imwrite, str(a.output_dir / f'{name}.png'), cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR), fast))
        writes.append(pool.submit(cv2.imwrite, str(a.output_dir / f'{name}_highlight.png'),
                                  (weight.cpu().numpy() * 255).round().astype(np.uint8), fast))
        reports.append({'frame': int(n), 'highlight_pct': float(highlight.float().mean() * 100),
                        'views_median': float(count.float().median()) if samples else 0.})
    for write in writes:
        write.result()
    pool.shutdown()
    summary = {'method': 'multi-view highlight removal by depth-aligned luminance consensus',
               'depth_dir': str(a.depth_dir), 'neighbours': a.neighbours, 'quantile': a.quantile,
               'threshold': a.threshold, 'min_views': a.min_views,
               'highlight_pct_median': float(np.median([r['highlight_pct'] for r in reports])), 'frames': reports}
    (a.output_dir / 'highlights.json').write_text(json.dumps(summary, indent=2) + '\n')
    done(f"Highlights: {len(reports)} frames, median {summary['highlight_pct_median']:.1f}% of depth pixels corrected")


if __name__ == '__main__':
    main()
