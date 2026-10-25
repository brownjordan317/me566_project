#!/usr/bin/env python3
"""Fuse full-image depth maps into a sparse static-scene TSDF.

Volumetric TSDF fusion (Curless and Levoy, SIGGRAPH 1996) on Open3D's
VoxelBlockGrid spatial hash (Dong et al., "ASH", TPAMI 2023). The
compute_unique_block_coordinates / integrate / extract_triangle_mesh sequence
follows Open3D's tensor reconstruction example
(examples/python/t_reconstruction_system/integrate.py,
https://github.com/isl-org/Open3D; MIT). Capacity sizing, prefetching and the
discontinuity check are this project's.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from recon.common.camera import frame_intrinsics
from recon.common.trajectory import load_poses
from recon.common.progress import bar, done, spinner


def depth_validity(depth: np.ndarray, min_depth: float, max_depth: float,
                   discontinuity_tau: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
    safe = np.where(valid, depth, 0).astype('float32')
    # A missing neighbour is not a zero-depth surface. Normalize the local
    # stencil by its valid samples before measuring depth discontinuities.
    weight = cv2.GaussianBlur(valid.astype('float32'), (3, 3), 0,
                              borderType=cv2.BORDER_REPLICATE)
    average = cv2.GaussianBlur(safe, (3, 3), 0,
                               borderType=cv2.BORDER_REPLICATE) / np.maximum(weight, 1e-6)
    gx = cv2.Sobel(average, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(average, cv2.CV_32F, 0, 1, ksize=3) / 8
    relative_jump = np.hypot(gx, gy) / np.maximum(safe, 1e-6)
    return valid & (relative_jump <= discontinuity_tau)


BLOCK_RESOLUTION = 16
# tsdf + weight + RGB color, float32, per voxel.
BYTES_PER_BLOCK = BLOCK_RESOLUTION ** 3 * (1 + 1 + 3) * 4


def frame_blocks(depth: np.ndarray, valid: np.ndarray, K: np.ndarray,
                 rotation: np.ndarray, center: np.ndarray, block_size: float,
                 truncation: float, stride: int = 4, bounds: np.ndarray | None = None) -> np.ndarray:
    """Block keys within the truncation band of one frame's surface samples."""
    v, u = np.nonzero(valid[::stride, ::stride])
    z = depth[::stride, ::stride][v, u].astype('float64')
    u = u * stride
    v = v * stride
    camera = np.column_stack(((u - K[0, 2]) * z / K[0, 0],
                              (v - K[1, 2]) * z / K[1, 1], z))
    world = camera @ rotation.T + center
    if bounds is not None:
        world = world[np.all((world >= bounds[0] - truncation) & (world <= bounds[1] + truncation), axis=1)]
    keys = []
    # Corners of the truncation cube cover every block a sample can touch.
    for corner in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3):
        index = np.floor((world + corner * truncation) / block_size).astype(np.int64)
        index += 1 << 20
        keys.append((index[:, 0] << 42) | (index[:, 1] << 21) | index[:, 2])
    return np.unique(np.concatenate(keys))


def frame_extent(depth: np.ndarray, K: np.ndarray, rotation: np.ndarray, center: np.ndarray,
                 stride: int = 8) -> np.ndarray | None:
    """World-frame box enclosing one frame's surface samples, or None if it has none."""
    v, u = np.nonzero(depth[::stride, ::stride] > 0)
    if not len(u):
        return None
    z = depth[::stride, ::stride][v, u].astype('float64')
    u, v = u * stride, v * stride
    camera = np.column_stack(((u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z))
    world = camera @ rotation.T + center
    return np.stack([world.min(0), world.max(0)])


def tile_boxes(bounds: np.ndarray, splits: int) -> list[np.ndarray]:
    """`splits` cuts per axis over a world-frame box."""
    edges = [np.linspace(bounds[0][axis], bounds[1][axis], splits + 1) for axis in range(3)]
    boxes = []
    for i in range(splits):
        for j in range(splits):
            for k in range(splits):
                boxes.append(np.array([[edges[0][i], edges[1][j], edges[2][k]],
                                       [edges[0][i + 1], edges[1][j + 1], edges[2][k + 1]]]))
    return boxes


def estimate_block_count(frames, load_depth, intrinsics, poses, voxel, truncation, bounds=None) -> int:
    def keys(frame_id):
        depth = load_depth(frame_id)
        if depth is None:
            return np.zeros(0, np.int64)
        rotation, center = poses[frame_id]
        return frame_blocks(depth, depth > 0, intrinsics[frame_id], rotation, center,
                            voxel * BLOCK_RESOLUTION, truncation, bounds=bounds)
    blocks = np.zeros(0, np.int64)
    with ThreadPoolExecutor(8) as pool:
        for frame_keys in pool.map(keys, frames):
            blocks = np.union1d(blocks, frame_keys)
    return len(blocks)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sequence', type=Path)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--trajectory', type=Path, required=True)
    p.add_argument('--depth-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--voxel', type=float, default=.02)
    p.add_argument('--sdf-trunc', type=float, default=.12)
    p.add_argument('--tiles', type=int, default=0,
                   help='With --bounds, mesh in this many cuts per axis and merge, so GPU memory is per tile '
                        'and the voxel is no longer memory-bound; 0 picks the fewest cuts that fit')
    p.add_argument('--bounds', type=float, nargs=6, metavar=('X0', 'Y0', 'Z0', 'X1', 'Y1', 'Z1'),
                   help='Integrate only inside this world-frame box. A large capture spends its whole voxel '
                        'budget on the full scene; bounding it to the region of interest buys a much finer '
                        'voxel from the same memory (the grid is sparse, so memory grows as 1 / voxel^2)')
    p.add_argument('--block-count', type=int, default=0,
                   help='Voxel-block capacity; 0 sizes it from the depth maps')
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--discontinuity-tau', type=float, default=.03)
    p.add_argument('--min-depth', type=float, default=.5)
    p.add_argument('--max-depth', type=float, default=30.)
    p.add_argument('--max-frames', type=int)
    p.add_argument('--min-views', type=float, default=3.,
                   help='Keep surface only where at least this many frames observed it; '
                        'surface seen by one or two frames is mostly floaters and '
                        'ray-aligned shards at the edge of the capture')
    a = p.parse_args()
    if (a.voxel <= 0 or a.sdf_trunc <= 0 or a.block_count < 0
            or a.discontinuity_tau <= 0 or a.min_depth <= 0
            or a.max_depth <= a.min_depth):
        p.error('Invalid voxel, depth, or discontinuity settings')
    if a.device == 'cuda' and not o3d.core.cuda.is_available():
        p.error('Open3D CUDA is unavailable; use --device cpu')
    poses = load_poses(a.trajectory)
    intrinsics = frame_intrinsics(a.model)
    device = o3d.core.Device('CUDA:0' if a.device == 'cuda' else 'CPU:0')
    truncation_voxels = a.sdf_trunc / a.voxel

    def load_depth(frame_id):
        """Validated depth with rejected pixels zeroed, or None if the frame has none."""
        depth_path = a.depth_dir / f'{frame_id:08d}_grounded.npy'
        if not depth_path.is_file():
            return None  # depth stage rejected this frame
        depth = np.load(depth_path, allow_pickle=False).astype('float32')
        valid = depth_validity(depth, a.min_depth, a.max_depth,
                               a.discontinuity_tau)
        return np.where(valid, depth, 0).astype('float32')

    def load(frame_id):
        metric = load_depth(frame_id)
        if metric is None:
            return None
        image = cv2.imread(str(a.sequence / 'frames' / f'{frame_id:08d}.png'),
                           cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != metric.shape:
            raise ValueError(f'Frame/depth mismatch {frame_id}')
        color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32') / 255.
        return metric, color, int(np.count_nonzero(metric))

    frame_ids = sorted(poses)[:a.max_frames]
    bounds = np.array(a.bounds, dtype=np.float64).reshape(2, 3) if a.bounds else None
    if bounds is not None:
        bounds = np.stack([bounds.min(0), bounds.max(0)])
    block_count = a.block_count
    if not block_count:
        # Open3D's CUDA hash map silently drops inserts and corrupts the volume
        # well before it is full (audi failed at 51% load), so keep load <= ~33%.
        # The old fixed 100000 stays the floor; small voxels need far more.
        with spinner('Sizing the TSDF volume'):
            needed = estimate_block_count(frame_ids, load_depth, intrinsics, poses,
                                          a.voxel, a.sdf_trunc, bounds)
        block_count = max(100000, 3 * needed)
    budget = None
    if a.device == 'cuda':
        import torch
        free, _ = torch.cuda.mem_get_info()
        budget = .8 * free
        if block_count * BYTES_PER_BLOCK > budget and a.tiles == 1:
            p.error(f'TSDF needs about {block_count * BYTES_PER_BLOCK / 2**30:.1f} GiB but '
                    f'{free / 2**30:.1f} GiB GPU memory is free; use a larger --voxel or allow tiling')
    # Tiling: memory grows with the surface area in the grid, so cutting the box into t^3 tiles cuts the
    # requirement by about t^2 and the voxel stops being memory-bound. Only the tile changes; every frame
    # that can reach it is still integrated, so a tile's surface is the same as the untiled one.
    splits = a.tiles
    if not splits and budget is not None:
        splits = max(1, int(np.ceil(np.sqrt(block_count * BYTES_PER_BLOCK / budget))))
    splits = max(1, splits)
    if splits > 1 and bounds is None:
        # Tiling needs a box to cut, so take the one the depth maps occupy; the grid is sized from the
        # depth anyway, and this keeps tiling available on a plain large scene with no region of interest.
        with spinner('Bounding the depth for tiling'):
            extent = [frame_extent(depth, intrinsics[f], *poses[f])
                      for f, depth in ((f, load_depth(f)) for f in frame_ids) if depth is not None]
            extent = [box for box in extent if box is not None]
        if extent:
            bounds = np.stack([np.min([b[0] for b in extent], 0), np.max([b[1] for b in extent], 0)])
        else:
            splits = 1
    boxes = tile_boxes(bounds, splits) if (bounds is not None and splits > 1) else [bounds]
    tile_blocks = max(100000, int(np.ceil(block_count / max(1, splits ** 2))))
    if len(boxes) > 1:
        print(f'TSDF in {len(boxes)} tiles ({splits} per axis), '
              f'{tile_blocks * BYTES_PER_BLOCK / 2**30:.1f} GiB each')

    cache: dict[int, tuple] = {}
    extents: dict[int, np.ndarray] = {}

    def frame(frame_id):
        """Depth, colour and world extent, loaded once and kept for every tile."""
        if frame_id not in cache:
            loaded = load(frame_id)
            cache[frame_id] = loaded
            if loaded is not None:
                rotation, center = poses[frame_id]
                extents[frame_id] = frame_extent(loaded[0], intrinsics[frame_id], rotation, center)
        return cache[frame_id]

    counts = []
    skipped = 0
    blocks_used = 0
    meshes = []
    reader = ThreadPoolExecutor(4)
    for tile_index, box in enumerate(boxes):
        volume = o3d.t.geometry.VoxelBlockGrid(
            ['tsdf', 'weight', 'color'], [o3d.core.Dtype.Float32] * 3,
            [[1], [1], [3]], voxel_size=a.voxel, block_resolution=BLOCK_RESOLUTION,
            block_count=tile_blocks if len(boxes) > 1 else block_count, device=device)
        if len(boxes) > 1:                      # only the frames whose surface can touch this tile
            reach = [f for f in frame_ids if frame(f) is not None and extents.get(f) is not None
                     and np.all(extents[f][1] + a.sdf_trunc >= box[0]) and np.all(extents[f][0] - a.sdf_trunc <= box[1])]
        else:
            reach = frame_ids
        integrated = False
        label = f'TSDF fusion ({tile_index + 1}/{len(boxes)})' if len(boxes) > 1 else 'TSDF fusion'
        if not reach:
            continue
        # CPU loading overlaps GPU integration; frames still integrate in order.
        prefetch = 8
        loads = [reader.submit(frame, frame_id) for frame_id in reach[:prefetch]]
        for index, frame_id in enumerate(bar(reach, desc=label, unit='frame')):
            if index + prefetch < len(reach):
                loads.append(reader.submit(frame, reach[index + prefetch]))
            loaded = loads[index].result()
            loads[index] = None
            if loaded is None:
                skipped += 1
                continue
            metric, color, valid_count = loaded
            K = intrinsics[frame_id]
            rotation, center = poses[frame_id]
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = rotation.T
            extrinsic[:3, 3] = -rotation.T @ center
            depth_image = o3d.t.geometry.Image(o3d.core.Tensor(metric, device=device))
            color_image = o3d.t.geometry.Image(o3d.core.Tensor(color, device=device))
            K_tensor = o3d.core.Tensor(K, device=device)
            E_tensor = o3d.core.Tensor(extrinsic, device=device)
            blocks = volume.compute_unique_block_coordinates(
                depth_image, K_tensor, E_tensor, depth_scale=1.,
                depth_max=a.max_depth, trunc_voxel_multiplier=truncation_voxels)
            if not len(blocks):                              # nothing of this frame lands in the grid
                continue
            if box is not None:
                keys = blocks.cpu().numpy()                  # block coordinates live on the GPU
                corner = keys * (a.voxel * BLOCK_RESOLUTION)
                inside = np.all((corner + a.voxel * BLOCK_RESOLUTION >= box[0]) & (corner <= box[1]), axis=1)
                if not inside.any():
                    continue
                blocks = o3d.core.Tensor(keys[inside], device=blocks.device)
            integrated = True
            volume.integrate(blocks, depth_image, color_image,
                             K_tensor, K_tensor, E_tensor, depth_scale=1.,
                             depth_max=a.max_depth,
                             trunc_voxel_multiplier=truncation_voxels)
        if len(boxes) > 1:
            if not integrated:          # a tile no frame reaches holds no blocks, and extraction rejects that
                del volume
                o3d.core.cuda.release_cache()
                continue
            blocks_used += int(volume.hashmap().size())
            piece = volume.extract_triangle_mesh(weight_threshold=a.min_views).to_legacy()
            if len(piece.triangles):
                # Blocks straddle a tile edge, so clip each piece to its own box; neighbours meet there.
                piece = piece.crop(o3d.geometry.AxisAlignedBoundingBox(box[0], box[1]))
                if len(piece.triangles):
                    meshes.append(piece)
            del volume
            o3d.core.cuda.release_cache()
    reader.shutdown()
    # Valid-pixel counts belong to the frames, not to any one tile: the first tile may reach only a few.
    counts = [loaded[2] for loaded in cache.values() if loaded is not None]
    if len(counts) < 2:
        raise ValueError('Need at least two full-scene frames')
    if meshes:
        mesh = meshes[0]
        for piece in meshes[1:]:
            mesh += piece
        mesh = mesh.merge_close_vertices(a.voxel / 4)
    else:
        with spinner('Extracting the TSDF surface'):
            blocks_used = int(volume.hashmap().size())
            mesh = volume.extract_triangle_mesh(weight_threshold=a.min_views).to_legacy()
    if not len(mesh.triangles):
        raise ValueError('Full-scene TSDF produced no mesh')
    mesh.compute_vertex_normals()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(a.output), mesh):
        raise OSError(a.output)
    report = {'method': 'full-image sparse voxel-block TSDF', 'frames': len(counts),
              'frames_without_depth': skipped,
              'valid_depth_pixels': sum(counts),
              'median_valid_pixels_per_frame': float(np.median(counts)),
              'voxel_map_units': a.voxel, 'sdf_trunc_map_units': a.sdf_trunc,
              'min_depth_map_units': a.min_depth,
              'max_depth_map_units': a.max_depth,
              'block_count_capacity': block_count,
              'blocks_used': blocks_used, 'tiles': len(boxes), 'device': a.device,
              'discontinuity_tau_relative_per_pixel': a.discontinuity_tau,
              'vertices': len(mesh.vertices), 'faces': len(mesh.triangles)}
    a.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    done(f"TSDF fusion: {report['frames']} frames, {report['faces']:,} faces")


if __name__ == '__main__':
    main()
