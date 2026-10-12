#!/usr/bin/env python3
"""Object masks from BiRefNet, automatically or completing supplied masks.

BiRefNet (Zheng et al., "Bilateral Reference for High-Resolution Dichotomous Image
Segmentation", CAAI AIR 2024; https://github.com/ZhengPeng7/BiRefNet, MIT) segments
the salient foreground of each frame. In an object-centred capture that is the
object, so it serves as the default mask generator when the user supplies none.
Its preprocessing (1024x1024 input, ImageNet normalisation, last decoder output
through a sigmoid) follows the model card.

With --guide, only foreground pieces that overlap the supplied mask of the same
frame are kept. That completes masks that miss parts of the object (a tracker
dropping wheels) without picking up other salient things in view. A frame with
no overlapping foreground keeps its supplied mask.

The model code is downloaded from Hugging Face and run (trust_remote_code), so
it is pinned to a reviewed revision. Reference masks, when supplied, are used
only to score the outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from recon.common.progress import bar, done

MODEL = 'ZhengPeng7/BiRefNet'
REVISION = 'e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4'
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)


def load_model(device: str, revision: str = REVISION):
    from transformers import AutoModelForImageSegmentation
    model = AutoModelForImageSegmentation.from_pretrained(MODEL, revision=revision, trust_remote_code=True)
    model = model.to(device).eval()
    return model.half() if device.startswith('cuda') else model


def foreground(model, image: np.ndarray, threshold: float = .5) -> np.ndarray:
    """Binary foreground of an RGB uint8 image at its own resolution."""
    x = (cv2.resize(image, (1024, 1024)).astype(np.float32) / 255 - MEAN) / STD
    parameter = next(model.parameters())
    x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(parameter.device, parameter.dtype)
    with torch.no_grad():
        probability = model(x)[-1].sigmoid()[0, 0].float().cpu().numpy()
    return cv2.resize(probability, (image.shape[1], image.shape[0])) > threshold


def keep_guided(full: np.ndarray, guide: np.ndarray, min_overlap: float = .3) -> np.ndarray:
    """Pieces of full that the guide mask largely covers, or that cover most of the guide."""
    count, labels = cv2.connectedComponents(full.astype(np.uint8))
    kept = np.zeros_like(full, dtype=bool)
    guide_area = guide.sum()
    for label in range(1, count):
        piece = labels == label
        overlap = (piece & guide).sum()
        if overlap > min_overlap * piece.sum() or (guide_area and overlap > .5 * guide_area):
            kept |= piece
    return kept if kept.any() else guide


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return float(np.count_nonzero(left & right) / union) if union else 1.


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('frames', type=Path, help='Folder of NNNNNNNN.png frames')
    p.add_argument('--output', type=Path, required=True, help='Folder for one binary mask per frame')
    p.add_argument('--guide', type=Path, help='Supplied masks to complete (same file names)')
    p.add_argument('--reference-masks', type=Path, help='Optional evaluation only')
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--revision', default=REVISION)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    frames = sorted(a.frames.glob('*.png'))
    if not frames:
        p.error(f'No PNG frames in {a.frames}')
    model = load_model(a.device, a.revision)
    a.output.mkdir(parents=True, exist_ok=True)
    empty, scores = 0, []
    for path in bar(frames, desc='BiRefNet masks', unit='frame'):
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        mask = foreground(model, image, a.threshold)
        if a.guide is not None:
            guide = cv2.imread(str(a.guide / path.name), cv2.IMREAD_GRAYSCALE)
            if guide is None:
                raise FileNotFoundError(a.guide / path.name)
            mask = keep_guided(mask, guide > 127)
        empty += not mask.any()
        cv2.imwrite(str(a.output / path.name), mask.astype(np.uint8) * 255)
        if a.reference_masks is not None:
            reference = cv2.imread(str(a.reference_masks / path.name), cv2.IMREAD_GRAYSCALE) > 127
            scores.append(mask_iou(mask, reference))
    report = {'model': MODEL, 'revision': a.revision, 'frames': len(frames), 'guided': a.guide is not None,
              'empty_masks': empty}
    if scores:
        report['reference_iou'] = {'mean': float(np.mean(scores)), 'min': float(np.min(scores))}
    (a.output.parent / f'{a.output.name}_birefnet.json').write_text(json.dumps(report, indent=2) + '\n')
    done(f'BiRefNet: {len(frames)} masks in {a.output}' + (f', {empty} empty' if empty else '')
         + (f", reference IoU {report['reference_iou']['mean']:.3f}" if scores else ''))


if __name__ == '__main__':
    main()
