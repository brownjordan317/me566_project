#!/usr/bin/env python3
"""Keep full-scene depth while using refined object depth inside tracked masks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from recon.common.progress import bar, done


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene-depth-dir', type=Path, required=True)
    p.add_argument('--object-depth-dir', type=Path, required=True)
    p.add_argument('--masks-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--object-depth-kind', default='refined')
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for mask_path in bar(sorted(a.masks_dir.glob('*.png')), desc='Combining depth', unit='frame'):
        frame = mask_path.stem
        scene_path = a.scene_depth_dir / f'{frame}_grounded.npy'
        object_path = a.object_depth_dir / f'{frame}_{a.object_depth_kind}.npy'
        if not scene_path.is_file() or not object_path.is_file():
            continue  # unregistered, or rejected by the depth stage
        scene = np.load(scene_path, allow_pickle=False).astype('float32')
        obj = np.load(object_path, allow_pickle=False).astype('float32')
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or scene.shape != obj.shape or mask.shape != scene.shape:
            raise ValueError(f'Depth/mask mismatch: {frame}')
        selected = (mask > 127) & np.isfinite(obj) & (obj > 0)
        combined = np.where(selected, obj, scene).astype('float32')
        np.save(a.output_dir / f'{frame}_grounded.npy', combined)
        records.append({'frame': frame, 'object_pixels': int(selected.sum()),
                        'median_object_depth_change': float(np.median(
                            np.abs(obj[selected] - scene[selected])) if selected.any() else 0)})
    if not records:
        p.error('No mask PNGs found')
    summary = {'frames': len(records), 'scene_depth_dir': str(a.scene_depth_dir),
               'object_depth_dir': str(a.object_depth_dir),
               'masks_dir': str(a.masks_dir),
               'median_object_depth_change_map_units': float(np.median(
                   [x['median_object_depth_change'] for x in records])),
               'per_frame': records}
    (a.output_dir / 'hybrid_depth.json').write_text(json.dumps(summary, indent=2) + '\n')
    done(f"Combined depth: {summary['frames']} frames")


if __name__ == '__main__':
    main()
