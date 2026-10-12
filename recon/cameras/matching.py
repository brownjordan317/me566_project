"""Keypoints and frame-pair matching (SuperPoint + LightGlue).

sequential_pairs chooses which frames to match; extract_multiscale detects
keypoints at several scales and merges them in full-resolution pixels.
"""
from __future__ import annotations

import torch


def sequential_pairs(names: list[str], overlap: int = 10, quadratic: bool = True,
                     loop: bool = False) -> list[tuple[str, str]]:
    """COLMAP's sequential matcher: nearby frames plus power-of-two jumps.

    Reimplements the quadratic-overlap pair selection of COLMAP's
    SequentialPairGenerator (src/colmap/controllers/pairing.cc,
    https://github.com/colmap/colmap; BSD-3-Clause).

    With loop=True the sequence is treated as circular, so the last frames are
    also matched to the first ones. An orbit that ends where it started then
    gets constraints closing the loop; without them the start and end are tied
    together only through every frame in between, and drift accumulates.
    """
    count = len(names)
    offsets = set(range(1, overlap + 1))
    if quadratic:
        offsets |= {2 ** step for step in range(overlap)}
    pairs = set()
    for index in range(count):
        for offset in offsets:
            other = index + offset
            if other >= count:
                if not loop:
                    continue
                other %= count
            if other != index:
                pairs.add((names[min(index, other)], names[max(index, other)]))
    return sorted(pairs)


def extract_multiscale(extractor, image: torch.Tensor, scales: list[float]) -> dict:
    """Keypoints from several image scales, merged in full-resolution pixels.

    Fine scales add precision where detail is distinctive; coarse scales stay
    matchable where fine detail repeats (patterned rugs, fabric, JPEG noise).
    """
    merged = None
    for scale in scales:
        scaled = image if scale == 1 else torch.nn.functional.interpolate(
            image[None], scale_factor=scale, mode='bilinear', antialias=True,
            align_corners=False)[0]
        result = extractor.extract(scaled, resize=None)
        result['keypoints'] = (result['keypoints'] + .5) / scale - .5
        if merged is None:
            merged = result
            continue
        for key in ('keypoints', 'keypoint_scores', 'descriptors'):
            merged[key] = torch.cat((merged[key], result[key]), dim=1)
    merged['image_size'] = torch.tensor(image.shape[-2:][::-1], device=image.device,
                                        dtype=merged['image_size'].dtype)[None]
    return merged
