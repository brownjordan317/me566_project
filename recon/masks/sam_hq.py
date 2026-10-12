#!/usr/bin/env python3
"""Refine tracked binary object masks with a local SAM-HQ checkpoint.

The source masks provide a box, one interior point, and a low-resolution mask
prompt. Reference masks, when supplied, are used only to score the outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return float(np.count_nonzero(left & right) / union) if union else 1.0


def prompts(mask: np.ndarray, box_pad: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError('Empty input mask')
    box = np.array([max(0, int(xs.min()) - box_pad), max(0, int(ys.min()) - box_pad),
                    min(width - 1, int(xs.max()) + box_pad),
                    min(height - 1, int(ys.max()) + box_pad)], dtype=np.float32)
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    py, px = np.unravel_index(np.argmax(distance), distance.shape)
    point = np.array([[px, py]], dtype=np.float32)
    low_res = cv2.resize(mask.astype(np.float32), (256, 256),
                         interpolation=cv2.INTER_LINEAR)
    logits = ((low_res - .5) * 6.0)[None].astype(np.float32)
    return box, point, logits


def keep_boundary_changes(original: np.ndarray, proposed: np.ndarray,
                          band: int) -> np.ndarray:
    if band <= 0:
        return proposed
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * band + 1, 2 * band + 1))
    interior = cv2.erode(original.astype(np.uint8), kernel) > 0
    exterior_limit = cv2.dilate(original.astype(np.uint8), kernel) > 0
    return interior | (proposed & exterior_limit)


def summarize(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {'mean': float(np.mean(data)), 'median': float(np.median(data)),
            'p10': float(np.percentile(data, 10)), 'min': float(np.min(data))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sequence', type=Path, help='Sequence containing frames/ PNGs')
    p.add_argument('--input-masks', type=Path, required=True)
    p.add_argument('--output-masks', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--reference-masks', type=Path, help='Optional evaluation only')
    p.add_argument('--boundary-band', type=int, default=20,
                   help='Maximum change distance in pixels from source boundary; 0 disables')
    p.add_argument('--box-pad', type=int, default=12)
    p.add_argument('--min-input-iou', type=float, default=.85,
                   help='Fallback to source mask if prediction differs too much')
    p.add_argument('--device', choices=('cuda', 'cpu'),
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--limit', type=int, help='Process first N masks for a smoke run')
    args = p.parse_args()
    if not args.checkpoint.is_file():
        p.error(f'Missing checkpoint: {args.checkpoint}')
    if args.boundary_band < 0 or not 0 <= args.min_input_iou <= 1:
        p.error('Invalid boundary band or minimum input IoU')
    from segment_anything_hq import SamPredictor, sam_model_registry
    model = sam_model_registry['vit_b'](checkpoint=str(args.checkpoint))
    predictor = SamPredictor(model.to(device=args.device).eval())
    files = sorted(args.input_masks.glob('*.png'))
    if args.limit is not None:
        files = files[:args.limit]
    if not files:
        p.error(f'No PNG masks in {args.input_masks}')
    args.output_masks.mkdir(parents=True, exist_ok=True)
    records = []
    for path in files:
        frame = cv2.imread(str(args.sequence / 'frames' / path.name), cv2.IMREAD_COLOR)
        source = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if frame is None or source is None or frame.shape[:2] != source.shape:
            raise ValueError(f'Missing or mismatched frame/mask: {path.name}')
        original = source > 127
        fallback = None
        if not original.any():
            refined = original
            fallback = 'empty_source'
        else:
            predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            box, point, logits = prompts(original, args.box_pad)
            with torch.inference_mode():
                predicted, _, _ = predictor.predict(
                    point_coords=point, point_labels=np.array([1]), box=box,
                    mask_input=logits, multimask_output=False,
                    hq_token_only=False)
            refined = keep_boundary_changes(original, predicted[0].astype(bool),
                                            args.boundary_band)
            if mask_iou(refined, original) < args.min_input_iou:
                refined = original
                fallback = 'low_source_overlap'
        destination = args.output_masks / path.name
        if not cv2.imwrite(str(destination), refined.astype(np.uint8) * 255):
            raise OSError(destination)
        record = {'frame': path.stem, 'source_iou': mask_iou(refined, original),
                  'source_pixels': int(original.sum()),
                  'refined_pixels': int(refined.sum()), 'fallback': fallback}
        if args.reference_masks:
            reference = cv2.imread(str(args.reference_masks / path.name),
                                   cv2.IMREAD_GRAYSCALE)
            if reference is None or reference.shape != original.shape:
                raise ValueError(f'Missing or mismatched reference mask: {path.name}')
            reference = reference > 127
            record['input_reference_iou'] = mask_iou(original, reference)
            record['refined_reference_iou'] = mask_iou(refined, reference)
        records.append(record)
        print(f"{path.stem}: source overlap {record['source_iou']:.4f}" +
              (f", reference {record['input_reference_iou']:.4f} -> "
               f"{record['refined_reference_iou']:.4f}" if args.reference_masks else ''),
              flush=True)
    summary = {'model': 'SAM-HQ ViT-B', 'frames': len(records),
               'device': args.device, 'boundary_band_px': args.boundary_band,
               'fallbacks': sum(x['fallback'] is not None for x in records),
               'source_overlap': summarize([x['source_iou'] for x in records]),
               'per_frame': records}
    if args.reference_masks:
        summary['input_reference_iou'] = summarize(
            [x['input_reference_iou'] for x in records])
        summary['refined_reference_iou'] = summarize(
            [x['refined_reference_iou'] for x in records])
        summary['improved_frames'] = sum(
            x['refined_reference_iou'] > x['input_reference_iou'] for x in records)
    report = args.output_masks.parent / (args.output_masks.name + '_report.json')
    report.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_frame'}, indent=2))
    print(f'Report: {report}')


if __name__ == '__main__':
    main()
