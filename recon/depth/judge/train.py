#!/usr/bin/env python3
"""Train the learned judge on recon.depth.judge.dataset output from the training scenes.

Each step takes random crops of random frames and a random subset of their candidates (any size from one
up, so the judge never relies on a particular pool), builds the cross-candidate evidence for just that
subset (recon.depth.judge.evidence.combine_pairs) and scores it against the truth:

  choice      cross-entropy towards the candidates closest to the truth (soft target exp(-error / 0.5 mm)),
              or towards "none" where no candidate is within --none-mm;
  expected    the error the judge's choice distribution expects, each candidate's error capped at 10 mm
              and "none" costing --none-mm, so abstaining pays off only where everything is worse;
  confidence  whether the picked candidate is within 1 mm.

Pixels are weighted up where the candidates disagree, and most where the best candidate disagrees with
the candidates' median (a minority is right), so the judge does not learn to follow the majority. One
term acts on the judged surface rather than on pixels alone:

  smooth       --smooth-weight × how much the choice changes between neighbouring pixels, less across image
               edges: each source has its own small bias, so a choice that flips pixel by pixel stitches a
               patchwork the TSDF turns into lumps.

Self-reports (recon.depth.judge.evidence.SELF_FEATURES): real ones are dropped from 30% of samples, and
each synthetic candidate gets an honest but noisy report (worse where its error is larger), an
uninformative one or none, in equal measure, so having a report is no sign of being right.
--exclude leaves sources out of training entirely; validation then reports how the judge treats them
when they appear, the test of whether it judges evidence rather than sources. --appearance adds what the
surface looks like (DINOv2 features of the full-resolution photo, recon.depth.judge.appearance) to the
image evidence, and --photo the full-resolution photo itself through the network's photo branch, so the
judge can tell materials apart and see detail finer than its grid. Whole set groups of the
training split are held out for validation (recon.depth.multiview.da3_distill.set_split).
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from functools import lru_cache

from recon.common.progress import bar, done
from recon.depth.judge.appearance import APPEARANCE, photo_files
from recon.depth.judge.evidence import CROSS_FEATURES, IMAGE_FEATURES, OWN_FEATURES, SELF_FEATURES, combine_pairs
from recon.depth.judge.network import PHOTO_SCALE, JudgeNet
from recon.depth.multiview.da3_distill import set_split

# Own features rescaled to about unit range (roughness is capped at 0.05, slope at 0.2).
FEATURE_SCALE = torch.tensor([20. if name == 'roughness' else 5. if name == 'slope' else 1. for name in OWN_FEATURES])
SELF_SCALE = torch.tensor([1., 1., 10., 5.])
CANDIDATE_FEATURES = len(OWN_FEATURES) + len(CROSS_FEATURES) + len(SELF_FEATURES)
PHOTO_FEATURES = 32


def judge_inputs(own: torch.Tensor, pairs: torch.Tensor, depth: torch.Tensor, report: torch.Tensor | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Network input (B, C, CANDIDATE_FEATURES, h, w) and presence (B, C, h, w) for a chosen candidate set.

    report holds the self-reports (B, C, SELF_FEATURES, h, w); None builds the input of a judge trained
    before self-reports (runs/models/judge/v1.pt)."""
    present = depth > 0
    own = own * FEATURE_SCALE.to(own.device)[None, None, :, None, None]
    parts = [own, combine_pairs(depth, pairs)]
    if report is not None:
        parts.append((report * SELF_SCALE.to(own.device)[None, None, :, None, None]).clamp(max=3))
    return torch.cat(parts, 2) * present[:, :, None], present


@lru_cache(maxsize=None)
def folder_photos(folder: Path) -> dict[int, Path]:
    return photo_files(folder / 'photos')


def appearance_grid(features: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Appearance features (APPEARANCE, th, tw) resampled onto the evidence grid (APPEARANCE, h, w)."""
    return cv2.resize(features.astype(np.float32).transpose(1, 2, 0), (size[1], size[0]),
                      interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)


def photo_grid(photo: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """A BGR photo as RGB (3, PHOTO_SCALE h, PHOTO_SCALE w) uint8 for the judge's photo branch."""
    shape = (PHOTO_SCALE * size[1], PHOTO_SCALE * size[0])
    if photo.shape[1::-1] != shape:
        photo = cv2.resize(photo, shape, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(photo[..., ::-1].transpose(2, 0, 1))


def load_frame(path: Path, chosen: list[int], crop: tuple[int, int, int, int] | None = None,
               extra: tuple[str, ...] = ()) -> dict:
    """One frame's arrays for the chosen candidates, optionally cropped to (top, left, height, width).

    extra adds 'appearance' (recon.depth.judge.appearance, appended to the image evidence) and 'photo'
    (the full-resolution photo over the same window)."""
    data = np.load(path)
    window = (slice(None), slice(None)) if crop is None else (slice(crop[0], crop[0] + crop[2]),
                                                              slice(crop[1], crop[1] + crop[3]))
    own = data['own'][chosen][(..., *window)].astype(np.float32)
    pairs = data['pairs'][:, chosen][:, :, chosen][(..., *window)].astype(np.float32)
    sidecar = path.with_name(path.stem + '_self.npy')
    report = (np.load(sidecar)[chosen][(..., *window)].astype(np.float32) if sidecar.is_file()
              else np.zeros((len(chosen), len(SELF_FEATURES), *own.shape[-2:]), np.float32))
    image = data['image'].astype(np.float32)
    size = image.shape[-2:]
    if 'appearance' in extra:
        image = np.concatenate([image, appearance_grid(np.load(path.with_name(path.stem + '_appearance.npy')), size)])
    frame = {'own': own, 'pairs': pairs, 'depth': data['depth'][chosen][(..., *window)],
             'truth': data['truth'][window], 'image': image[(slice(None), *window)],
             'region': data['region'][window], 'report': report}
    if 'photo' in extra:
        photo = photo_grid(cv2.imread(str(folder_photos(path.parent)[int(path.stem)])), size)
        frame['photo'] = photo if crop is None else photo[:, PHOTO_SCALE * crop[0]:PHOTO_SCALE * (crop[0] + crop[2]),
                                                          PHOTO_SCALE * crop[1]:PHOTO_SCALE * (crop[1] + crop[3])]
    return frame


def smooth_field(shape: tuple[int, int], rng: np.random.Generator, cells: int = 6) -> np.ndarray:
    """Smooth unit-variance noise (h, w)."""
    coarse = rng.normal(size=(cells, cells)).astype(np.float32)
    return cv2.resize(coarse, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def synthetic_report(depth: np.ndarray, truth: np.ndarray, mm_per_unit: float, rng: np.random.Generator) -> np.ndarray:
    """A self-report (SELF_FEATURES, h, w) for a synthetic candidate: honest but noisy, uninformative, or none.

    Ranges follow the Gaussian pass's own (jeep: residual median 0.016, p95 0.07; normal median 0.005,
    p95 0.19; opacity about 1)."""
    out = np.zeros((len(SELF_FEATURES), *depth.shape), np.float32)
    mode = rng.choice(3)
    if mode == 2:
        return out
    if mode == 0:                                           # honest: worse where the error is larger
        error = np.where((depth > 0) & (truth > 0), np.abs(depth - truth) * mm_per_unit, 0).astype(np.float32)
        quality = np.minimum(cv2.GaussianBlur(error, (0, 0), 1.5), 10)
    else:                                                   # uninformative: a random field of the same range
        quality = np.abs(smooth_field(depth.shape, rng)) * 2
    noise = lambda spread: np.exp(spread * smooth_field(depth.shape, rng, 12))
    out[0] = 1
    out[1] = np.clip(1 - .02 * quality * noise(.5), .5, 1)
    out[2] = .008 * (1 + quality) * noise(.6)
    out[3] = .004 * (1 + quality) ** 1.8 * noise(1.)
    return np.where(depth[None] > 0, out, out * (np.arange(len(SELF_FEATURES)) == 0)[:, None, None])


def stack(frames: list[dict], device: str) -> dict:
    """Batch frames with different candidate counts by padding with absent candidates."""
    C = max(len(f['depth']) for f in frames)
    pad = lambda x, axes: np.pad(x, [(0, C - x.shape[a]) if a in axes else (0, 0) for a in range(x.ndim)])
    out = {'own': np.stack([pad(f['own'], (0,)) for f in frames]),
           'pairs': np.stack([pad(f['pairs'], (1, 2)) for f in frames]),
           'depth': np.stack([pad(f['depth'], (0,)) for f in frames]),
           'truth': np.stack([f['truth'] for f in frames]), 'image': np.stack([f['image'] for f in frames]),
           'report': np.stack([pad(f['report'], (0,)) for f in frames])}
    out = {key: torch.from_numpy(value).to(device) for key, value in out.items()}
    if 'photo' in frames[0]:
        out['photo'] = torch.from_numpy(np.stack([f['photo'] for f in frames])).to(device).float() / 255
    return out


def errors_mm(depth: torch.Tensor, truth: torch.Tensor, mm_per_unit: torch.Tensor) -> torch.Tensor:
    """Each candidate's absolute error in millimetres (B, C, h, w); inf where it has no depth."""
    error = (depth - truth[:, None]).abs() * mm_per_unit[:, None, None, None]
    return torch.where(depth > 0, error, torch.inf)


def loss_terms(logits: torch.Tensor, confidence: torch.Tensor, error: torch.Tensor, truth: torch.Tensor,
               none_mm: float) -> tuple[torch.Tensor, dict]:
    finite = torch.isfinite(error)
    labelled = (truth > 0) & finite.any(1)
    capped = error.clamp(max=10.)                                        # absent candidates count as 10 mm
    best = capped.min(1).values
    target = torch.softmax(torch.where(finite, -capped / .5, -torch.inf), 1).nan_to_num(0)
    nothing = best > none_mm
    target = torch.cat([target * ~nothing[:, None], nothing[:, None].float()], 1)
    log_p = torch.log_softmax(logits, 1)
    choice = -(target * log_p.nan_to_num(0, neginf=0)).sum(1)
    p = log_p.exp()
    expected = (p[:, :-1] * capped).sum(1) + p[:, -1] * none_mm
    picked = torch.gather(capped, 1, logits[:, :-1].argmax(1, keepdim=True))[:, 0]
    confident = F.binary_cross_entropy_with_logits(confidence, (picked < 1).float(), reduction='none')
    median = torch.nanmedian(torch.where(finite, error, torch.nan), 1).values.nan_to_num(0)
    spread = torch.where(finite, capped, 0).max(1).values - best
    weight = (1 + (spread > .5).float() + 3 * ((best < 1) & (median > 1.5)).float()) * labelled
    norm = weight.sum().clamp(min=1)
    mean = lambda x: torch.where(labelled, x * weight, 0).sum() / norm
    terms = {'choice': mean(choice), 'expected': mean(expected), 'confidence': mean(confident)}
    return terms['choice'] + .5 * terms['expected'] + .2 * terms['confidence'], terms


def smoothness(probability: torch.Tensor, present: torch.Tensor, image: torch.Tensor, labelled: torch.Tensor,
               edge: float = .05) -> torch.Tensor:
    """How much the choice (B, C + 1, h, w) changes between neighbouring pixels with the same candidates,
    as half the L1 distance of the two distributions, down-weighted across colour edges of the image."""
    total, count = probability.new_zeros(()), probability.new_zeros(())
    for axis in (2, 3):
        step = lambda x: (x.narrow(axis, 1, x.shape[axis] - 1), x.narrow(axis, 0, x.shape[axis] - 1))
        (p1, p0), (s1, s0), (i1, i0), (l1, l0) = step(probability), step(present), step(image[:, :3]), step(labelled[:, None])
        same = (s1 == s0).all(1) & l1[:, 0] & l0[:, 0]
        weight = torch.exp(-(i1 - i0).abs().mean(1) / edge) * same
        total = total + (.5 * (p1 - p0).abs().sum(1) * weight).sum()
        count = count + weight.sum()
    return total / count.clamp(min=1)


def pooled_photometric(own: torch.Tensor, window: int = 5) -> torch.Tensor:
    """The fusion's rule on the grid: mean of the best three per-view NCCs, pooled over a window."""
    first = OWN_FEATURES.index('ncc_1')
    score = own[:, :, first:first + 3].mean(2)
    present = own[:, :, 0]
    B, C, h, w = score.shape
    total = F.avg_pool2d((score * present).flatten(0, 1)[:, None], window, 1, window // 2, count_include_pad=False)
    count = F.avg_pool2d(present.flatten(0, 1)[:, None], window, 1, window // 2, count_include_pad=False)
    pooled = (total / count.clamp(min=1e-6)).reshape(B, C, h, w)
    return torch.where(present > 0, pooled, -torch.inf)


class Data:
    """The frames of a set of judge-data folders, with each folder's candidate names and scale."""

    def __init__(self, folders: list[Path], extra: tuple[str, ...] = ()):
        self.items, self.extra = [], extra
        for folder in folders:
            meta = json.loads((folder / 'meta.json').read_text())
            for frame in meta['frames']:
                self.items.append((folder / f'{frame:08d}.npz', meta))


def sample(data: Data, exclude: set[str], crop: int, size: int, rng: np.random.Generator) -> tuple[list, list]:
    """A training batch: random crops of random frames, each with a random subset of its candidates."""
    frames, scales = [], []
    while len(frames) < size:
        path, meta = data.items[rng.integers(len(data.items))]
        usable = [i for i, n in enumerate(meta['names']) if n not in exclude]
        low = 1 if rng.random() < .1 else min(2, len(usable))
        chosen = sorted(rng.choice(usable, int(rng.integers(low, len(usable) + 1)), replace=False).tolist())
        region = np.load(path)['region']
        ys, xs = np.nonzero(region)
        if not len(ys) or min(region.shape) < crop:
            continue
        centre = rng.integers(len(ys))
        top = int(np.clip(ys[centre] - crop // 2, 0, region.shape[0] - crop))
        left = int(np.clip(xs[centre] - crop // 2, 0, region.shape[1] - crop))
        frame = load_frame(path, chosen, (top, left, crop, crop), data.extra)
        for c, i in enumerate(chosen):
            if meta['kinds'][meta['names'][i]] == 'synthetic':
                frame['report'][c] = synthetic_report(frame['depth'][c], frame['truth'], meta['mm_per_unit'], rng)
            elif rng.random() < .3:
                frame['report'][c] = 0
        frames.append(frame)
        scales.append(meta['mm_per_unit'])
    return frames, scales


def evaluate(net: JudgeNet, data: Data, device: str, exclude: set[str], none_mm: float) -> dict:
    """Judge, the fusion's photometric rule and the oracle on every held-out frame, real candidates only.

    'seen' leaves the --exclude sources out (the pool the judge trained on); 'all' adds them back."""
    net.eval()
    result = {}
    for setting in ('seen', 'all') if exclude else ('all',):
        sums = {'judge': [], 'photometric': [], 'oracle': [], 'judge_right': 0, 'photometric_right': 0,
                'pixels': 0, 'abstained': 0, 'newcomer_best': 0, 'newcomer_picked_when_best': 0}
        for path, meta in data.items:
            names = [n for n in meta['names'] if meta['kinds'][n] == 'real' and (setting == 'all' or n not in exclude)]
            if len(names) < 2:
                continue
            chosen = [meta['names'].index(n) for n in names]
            batch = stack([load_frame(path, chosen, extra=data.extra)], device)
            with torch.no_grad():
                inputs, present = judge_inputs(batch['own'], batch['pairs'], batch['depth'], batch['report'])
                logits, _ = net(inputs, batch['image'], present, batch.get('photo'))
            error = errors_mm(batch['depth'], batch['truth'], torch.tensor([meta['mm_per_unit']], device=device))
            valid = (batch['truth'] > 0) & (present.sum(1) >= 2)
            if not valid.any():
                continue
            best = error.argmin(1)
            judge = logits[:, :-1].argmax(1)
            photo = pooled_photometric(batch['own']).argmax(1)
            pick = lambda choice: torch.gather(error, 1, choice[:, None])[:, 0][valid]
            sums['judge'].append(pick(judge).cpu())
            sums['photometric'].append(pick(photo).cpu())
            sums['oracle'].append(error.min(1).values[valid].cpu())
            sums['judge_right'] += int((judge == best)[valid].sum())
            sums['photometric_right'] += int((photo == best)[valid].sum())
            sums['pixels'] += int(valid.sum())
            sums['abstained'] += int((logits.argmax(1) == len(names))[valid].sum())
            newcomers = [i for i, n in enumerate(names) if n in exclude]
            for i in newcomers:
                is_best = (best == i) & valid
                sums['newcomer_best'] += int(is_best.sum())
                sums['newcomer_picked_when_best'] += int((judge == i)[is_best].sum())
        if not sums['pixels']:
            continue
        row = {}
        for key in ('judge', 'photometric', 'oracle'):
            values = torch.cat(sums[key]).clamp(max=50).numpy()
            row[key] = {'median_mm': float(np.median(values)), 'mean_mm': float(values.mean()),
                        'within_1mm_pct': float((values < 1).mean() * 100)}
        row['judge_right_pct'] = sums['judge_right'] / sums['pixels'] * 100
        row['photometric_right_pct'] = sums['photometric_right'] / sums['pixels'] * 100
        row['abstained_pct'] = sums['abstained'] / sums['pixels'] * 100
        row['pixels'] = sums['pixels']
        if sums['newcomer_best']:
            row['newcomer_picked_when_best_pct'] = sums['newcomer_picked_when_best'] / sums['newcomer_best'] * 100
        result[setting] = row
    net.train()
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('data', type=Path, help='Folder of per-scene judge data (recon.depth.judge.dataset outputs)')
    p.add_argument('--output', type=Path, required=True, help='Judge weights (.pt); a .json report goes beside it')
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--crop', type=int, default=96, help='Crop side on the evidence grid')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--none-mm', type=float, default=3.)
    p.add_argument('--exclude', default='', help='Comma-separated sources never used in training')
    p.add_argument('--val-every', type=int, default=8, help='Hold out every n-th set number')
    p.add_argument('--evaluate-every', type=int, default=2000)
    p.add_argument('--smooth-weight', type=float, default=.2, help='Penalty on the choice changing between pixels')
    p.add_argument('--appearance', action='store_true', help='Add DINOv2 appearance (recon.depth.judge.appearance)')
    p.add_argument('--projection', type=Path, default=Path('runs/models/judge/appearance.npz'),
                   help='The appearance projection the data was written with; saved into the weights')
    p.add_argument('--photo', action='store_true', help='Add the full-resolution photo branch')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    folders = sorted(path for path in a.data.iterdir() if (path / 'meta.json').is_file())
    train_names, held_names = set_split([f.name for f in folders], a.val_every)
    extra = tuple(name for name in ('appearance', 'photo') if getattr(a, name))
    train = Data([f for f in folders if f.name in train_names], extra)
    held = Data([f for f in folders if f.name in held_names], extra)
    exclude = {name for name in a.exclude.split(',') if name}
    config = {'candidate_features': CANDIDATE_FEATURES, 'extra': list(extra),
              'image_features': len(IMAGE_FEATURES) + APPEARANCE * ('appearance' in extra),
              'photo_features': PHOTO_FEATURES * ('photo' in extra)}
    projection = dict(np.load(a.projection)) if 'appearance' in extra else None
    net = JudgeNet(config['candidate_features'], config['image_features'],
                   photo_features=config['photo_features']).to(a.device)
    optimiser = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, a.lr, total_steps=a.steps, pct_start=.05)
    report = {'train_scenes': sorted(train_names), 'held_out_scenes': sorted(held_names), 'exclude': sorted(exclude),
              'steps': a.steps, 'batch': a.batch, 'crop': a.crop, 'lr': a.lr, 'none_mm': a.none_mm,
              'smooth_weight': a.smooth_weight,
              **config,
              'train_frames': len(train.items), 'held_out_frames': len(held.items), 'evaluations': []}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    started, losses = time.time(), []
    pool = ThreadPoolExecutor(8)
    queue = [pool.submit(sample, train, exclude, a.crop, a.batch, np.random.default_rng(i)) for i in range(16)]
    for step in bar(range(a.steps), desc='Training the judge', unit='step'):
        frames, scales = queue.pop(0).result()
        queue.append(pool.submit(sample, train, exclude, a.crop, a.batch, np.random.default_rng(step + 16)))
        batch = stack(frames, a.device)
        inputs, present = judge_inputs(batch['own'], batch['pairs'], batch['depth'], batch['report'])
        logits, confidence = net(inputs, batch['image'], present, batch.get('photo'))
        error = errors_mm(batch['depth'], batch['truth'], torch.tensor(scales, device=a.device))
        loss, terms = loss_terms(logits, confidence, error, batch['truth'], a.none_mm)
        if a.smooth_weight:
            labelled = (batch['truth'] > 0) & present.any(1)
            terms['smooth'] = smoothness(torch.softmax(logits, 1).nan_to_num(0), present, batch['image'], labelled)
            loss = loss + a.smooth_weight * terms['smooth']
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        optimiser.step()
        schedule.step()
        losses.append({k: float(v.detach()) for k, v in terms.items()})
        if (step + 1) % a.evaluate_every == 0 or step + 1 == a.steps:
            row = {'step': step + 1, 'seconds': time.time() - started,
                   'loss': {k: float(np.mean([l[k] for l in losses[-500:] if k in l])) for k in losses[-1]},
                   **evaluate(net, held, a.device, exclude, a.none_mm)}
            report['evaluations'].append(row)
            torch.save({'config': config, 'state': net.state_dict(), 'projection': projection}, a.output)
            a.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
            last = row['all']
            print(f"step {step + 1}: judge {last['judge']['median_mm']:.3f} mm ({last['judge_right_pct']:.1f}% right), "
                  f"photometric {last['photometric']['median_mm']:.3f} mm ({last['photometric_right_pct']:.1f}%), "
                  f"oracle {last['oracle']['median_mm']:.3f} mm")
    done(f'Judge trained on {len(train.items)} frames -> {a.output}')


if __name__ == '__main__':
    main()
