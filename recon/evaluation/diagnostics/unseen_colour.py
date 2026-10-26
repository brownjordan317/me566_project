#!/usr/bin/env python3
"""Score an asset's inferred colour against photographs it never saw.

About a sixth of a closed asset's texture is not observed: no view photographed that surface, so the
colour comes from rendering the fitted Gaussians or from a mirrored counterpart (see the provenance map
`recon.surface.texture --provenance-map` writes). "It looks plausible" is not a measurement, and the
benchmark cannot help - its official score excludes exactly the regions in question.

So hold views back from the texturing and judge the fill against them. The asset is textured from a
subset of the photos; surface that only the held-out views ever saw is then inferred, and those views are
real photographs of it. Each held-out camera is ray-cast against the mesh, the hit's barycentric
coordinates give its texel, and the asset's colour there is compared with the photograph. Splitting the
comparison by provenance separates two different questions: whether the texture is right where it was
observed (a sanity check), and whether the fill is right where it was not (the point).

Any later generative filling can be scored the same way, against the same held-out photographs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from recon.common.camera import frame_intrinsics
from recon.common.progress import done, spinner
from recon.common.trajectory import load_poses


def load_asset(path: Path) -> tuple[o3d.geometry.TriangleMesh, np.ndarray, np.ndarray]:
    """The asset's mesh, its per-triangle-corner UVs and its albedo."""
    mesh = o3d.io.read_triangle_mesh(str(path), True)
    if not mesh.has_triangle_uvs():
        raise ValueError(f'No texture coordinates in {path}')
    uvs = np.asarray(mesh.triangle_uvs).reshape(-1, 3, 2)
    textures = mesh.textures
    if not textures:
        raise ValueError(f'No texture image in {path}')
    albedo = np.asarray(textures[-1])
    return mesh, uvs, albedo


def sample_atlas(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Nearest-texel lookup for UV coordinates.

    v is *not* flipped: Open3D hands back the glTF texture already in the row order its UVs index, and
    flipping it here cost more than twice the colour error (48.7 against 21.7 on views the texture was
    built from), which is how the convention was settled.
    """
    height, width = image.shape[:2]
    x = np.clip((uv[:, 0] * (width - 1)).round().astype(int), 0, width - 1)
    y = np.clip((uv[:, 1] * (height - 1)).round().astype(int), 0, height - 1)
    return image[y, x]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('asset', type=Path, help='Textured .glb from recon.surface.texture')
    p.add_argument('--provenance', type=Path, help='<asset>_source.png; defaults beside the asset')
    p.add_argument('--model', type=Path, required=True, help='COLMAP text model (working-frame intrinsics)')
    p.add_argument('--trajectory', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True, help='Numbered full-resolution photographs')
    p.add_argument('--sequence', type=Path, required=True, help='Prepared sequence; sets the working width')
    p.add_argument('--held-out', type=int, nargs='+', required=True,
                   help='Frames kept out of the texturing, which are the evidence here')
    p.add_argument('--output', type=Path)
    p.add_argument('--stride', type=int, default=2, help='Score every STRIDE-th pixel')
    a = p.parse_args()
    mesh, uvs, albedo = load_asset(a.asset)
    source_path = a.provenance or a.asset.with_name(a.asset.stem + '_source.png')
    source = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE) if source_path.is_file() else None
    if source is None:
        p.error(f'No provenance map at {source_path}; texture with --provenance-map')
    raycast = o3d.t.geometry.RaycastingScene()
    raycast.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    intrinsics, poses = frame_intrinsics(a.model), load_poses(a.trajectory)
    working = cv2.imread(str(next(iter(sorted((a.sequence / 'frames').glob('*.png')))))).shape[1]
    errors = {1: [], 2: [], 3: []}
    counts = {1: 0, 2: 0, 3: 0}
    with spinner(f'Comparing {len(a.held_out)} held-out photographs'):
        # Frame numbers are matched by the file stem's value, not its width: the datasets number their
        # photographs with six digits where the prepared sequences use eight.
        available = {int(path.stem): path for path in a.images.iterdir()
                     if path.suffix.lower() in ('.png', '.jpg', '.jpeg') and path.stem.isdigit()}
        for frame in a.held_out:
            path = available.get(frame)
            if path is None or frame not in intrinsics or frame not in poses:
                continue
            photo = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            scale = photo.shape[1] / working
            K = intrinsics[frame].copy() * np.array([[scale], [scale], [1.]])
            rotation, centre = poses[frame]
            height, width = photo.shape[:2]
            v, u = np.mgrid[:height:a.stride, :width:a.stride].astype(np.float32)
            rays = np.stack(((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)), -1)
            rays = rays @ rotation.T
            rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
            origins = np.broadcast_to(centre, rays.shape)
            hits = raycast.cast_rays(o3d.core.Tensor(
                np.concatenate([origins, rays], -1).reshape(-1, 6).astype(np.float32)))
            primitive = hits['primitive_ids'].numpy()
            bary = hits['primitive_uvs'].numpy()
            valid = primitive != o3d.t.geometry.RaycastingScene.INVALID_ID
            if not valid.any():
                continue
            corner = uvs[primitive[valid]]
            weights = np.stack([1 - bary[valid, 0] - bary[valid, 1], bary[valid, 0], bary[valid, 1]], 1)
            uv = (corner * weights[:, :, None]).sum(1)
            predicted = sample_atlas(albedo, uv).astype(np.float32)[:, :3]
            observed = photo[v.astype(int).ravel()[valid], u.astype(int).ravel()[valid]].astype(np.float32)
            provenance = sample_atlas(source, uv)
            difference = np.abs(predicted - observed).mean(1)
            for kind in errors:
                picked = provenance == kind
                if picked.any():
                    errors[kind].append(difference[picked])
                    counts[kind] += int(picked.sum())
    report = {'asset': str(a.asset), 'held_out_frames': a.held_out, 'pixels': counts}
    for kind, name in ((1, 'photographed'), (2, 'mirrored'), (3, 'rendered')):
        if errors[kind]:
            joined = np.concatenate(errors[kind])
            report[name] = {'mean_abs_colour_error': float(joined.mean()),
                            'median': float(np.median(joined)),
                            'p90': float(np.percentile(joined, 90))}
    print(json.dumps(report, indent=2))
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + '\n')
    done('Unseen-colour check')


if __name__ == '__main__':
    main()
