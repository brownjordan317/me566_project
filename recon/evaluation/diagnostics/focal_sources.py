#!/usr/bin/env python3
"""Which library estimates the focal length best? (component diagnostic)

Every library that predicts a camera's field of view is asked for the focal
length of the same frames (up to --frames, spread evenly over the sequence),
and the median is compared with the dataset's calibration scaled to the working
frames. A camera solve (--colmap, a sparse_txt model) is scored the same way.

  vggt    VGGT-1B camera head, frames as one batch (load_and_preprocess_images mode='pad')
  omega   VGGT-Omega camera head, one batch ('balanced' resize)
  da3     Depth Anything 3 Giant without poses (its camera decoder), one batch at 504 px
  moge    MoGe-3 per frame, field of view recovered from its point map

Focal lengths are compared in working pixels along x; error is in percent,
positive when the estimate is too long (the object comes out too large).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

from recon.common.camera import load_dataset_intrinsics
from recon.common.colmap import parse_model
from recon.common.progress import done, spinner

SOURCES = ('vggt', 'omega', 'da3', 'moge')


def vggt_focal(paths: list[Path], width: int) -> np.ndarray:
    from huggingface_hub import snapshot_download
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    warnings.filterwarnings('ignore', category=FutureWarning, module='vggt')
    model = VGGT.from_pretrained(snapshot_download('facebook/VGGT-1B', local_files_only=True)).to('cuda').eval()
    images = load_and_preprocess_images([str(path) for path in paths], mode='pad').to('cuda')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        tokens, index = model.aggregator(images[None])
        encoding = model.camera_head(tokens)[-1].float()
    _, K = pose_encoding_to_extri_intri(encoding, images.shape[-2:])
    # mode='pad': the longest side (here the width) is resized to 518 px, so x scales by width / 518.
    return K[0, :, 0, 0].cpu().numpy() * width / images.shape[-1]


def omega_focal(paths: list[Path], width: int) -> np.ndarray:
    from huggingface_hub import hf_hub_download
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera
    model = VGGTOmega().to('cuda').eval()
    model.load_state_dict(torch.load(hf_hub_download('facebook/VGGT-Omega', 'vggt_omega_1b_512.pt'),
                                     map_location='cpu', weights_only=True))
    images = load_and_preprocess_images([str(path) for path in paths], mode='balanced',
                                        image_resolution=512).to('cuda')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        encoding = model(images)['pose_enc'].float()
    _, K = encoding_to_camera(encoding, images.shape[-2:])
    return K[0, :, 0, 0].cpu().numpy() * width / images.shape[-1]


def da3_focal(paths: list[Path], width: int) -> np.ndarray:
    from depth_anything_3.api import DepthAnything3
    with contextlib.redirect_stdout(io.StringIO()):
        model = DepthAnything3.from_pretrained('depth-anything/DA3-GIANT').to('cuda').eval()
        images = [cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB) for path in paths]
        with torch.inference_mode():
            prediction = model.inference(images, process_res=504)
    return prediction.intrinsics[:, 0, 0] * width / prediction.processed_images.shape[2]


def moge_focal(paths: list[Path], width: int) -> np.ndarray:
    from huggingface_hub import hf_hub_download
    from moge.model.v3 import MoGeModel
    model = MoGeModel.from_pretrained(hf_hub_download('Ruicheng/moge-3-vitl', 'model.pt',
                                                      local_files_only=True)).to('cuda').eval()
    focal = []
    with torch.inference_mode():
        for path in paths:
            rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).to('cuda', dtype=torch.float32).permute(2, 0, 1) / 255.
            focal.append(float(model.infer(tensor)['intrinsics'][0, 0]) * width)   # normalised by width
    return np.array(focal)


ESTIMATORS = {'vggt': vggt_focal, 'omega': omega_focal, 'da3': da3_focal, 'moge': moge_focal}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('sequence', type=Path, help='Prepared sequence (frames/, calibration.json)')
    p.add_argument('--intrinsics', type=Path, required=True, help="Dataset's per-frame matrices (NNNNNN.txt)")
    p.add_argument('--colmap', type=Path, help='A camera solve to score too (sparse_txt)')
    p.add_argument('--sources', nargs='+', choices=SOURCES, default=list(SOURCES))
    p.add_argument('--frames', type=int, default=32)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    paths = sorted((a.sequence / 'frames').glob('*.png'))
    paths = [paths[i] for i in np.unique(np.linspace(0, len(paths) - 1, min(a.frames, len(paths))).round().astype(int))]
    height, width = cv2.imread(str(paths[0])).shape[:2]
    calibration = json.loads((a.sequence / 'calibration.json').read_text())
    truth = load_dataset_intrinsics(a.intrinsics, [int(path.stem) for path in paths], (width, height),
                                    tuple(calibration['image_size']))
    true_focal = float(np.median([K[0, 0] for K in truth.values()]))
    report = {'sequence': str(a.sequence), 'frames': len(paths), 'true_focal_px': true_focal, 'sources': {}}

    def score(name: str, focal: np.ndarray) -> None:
        median = float(np.median(focal))
        report['sources'][name] = {'median_focal_px': median, 'error_pct': (median / true_focal - 1) * 100,
                                   'spread_pct': float((np.quantile(focal, .9) / np.quantile(focal, .1) - 1) * 100)}

    if a.colmap is not None:
        intrinsics, _, frames = parse_model(a.colmap)
        score('colmap', np.array([intrinsics[number][0, 0] for number, *_ in frames]))
    for name in a.sources:
        with spinner(f'{name} focal'):
            score(name, ESTIMATORS[name](paths, width))
        torch.cuda.empty_cache()
    text = json.dumps(report, indent=2) + '\n'
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(text)
    done('Focal error: ' + ', '.join(f"{name} {row['error_pct']:+.2f}%" for name, row in report['sources'].items()))


if __name__ == '__main__':
    main()
