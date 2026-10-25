#!/usr/bin/env python3
"""Same-camera baseline and warm-start test: PGSR trained on a finished run's cameras.

PGSR (Chen et al., "PGSR: Planar-based Gaussian Splatting for Efficient and
High-Fidelity Surface Reconstruction", TVCG 2024; https://github.com/zju3dv/PGSR,
licence: research and non-profit use only) is run unmodified from third_party/PGSR in
the 2DGS environment (--python), into which its diff-plane-rasterization extension
is built with the same flags as 2DGS's. PGSR imports one function from pytorch3d
(quaternion_to_matrix); that environment carries a one-function stand-in instead of
the full package. Academic comparison only: nothing from PGSR is part of the pipeline.

The workspace is the 2DGS wrapper's (recon.evaluation.baselines.twodgs.write_workspace):
working frames shifted so the principal point is at the centre (PGSR, like 2DGS,
assumes it) and cropped, the model normalised to a unit sphere around the object,
and the start points either our run's sparse points or, with --init-points, our
dense point clouds (a warm start). Training follows PGSR's DTU recipe
(scripts/run_dtu.py: --ncc_scale 0.5; our frames are already at its -r 2 size);
--train-args overrides its schedule, e.g. a short warm-start run that switches the
single- and multi-view geometric losses on early. The TSDF mesh (render.py,
voxel 0.002 in normalised units, largest cluster) is moved back to map units and
culled with the masks as in the 2DGS wrapper, so the standard scorers apply.

--depth-only skips PGSR's meshing (Open3D's CPU TSDF at a fine voxel: 89 s of a
3k-iteration pass on scan24 with the GPU idle) and instead saves the trained model's
depth for every frame (pgsr_depth.py, 7 s), mapped back to our working frames (crop and
principal-point shift undone, normalised units scaled back to map units), as
OUTPUT/depth/<frame>_grounded.npy with a hybrid_depth.json, so our own cross-view
check and GPU object TSDF build the mesh (make ... OBJECT_DEPTH_INPUT=OUTPUT/depth).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from recon.common.camera import frame_intrinsics
from recon.evaluation.baselines.twodgs import cull, write_workspace


def write_priors(a: argparse.Namespace, workspace: Path, radius: float) -> None:
    """Our depth in PGSR's frame per training view: object pixels, normalised, centred and cropped like the images."""
    intrinsics = frame_intrinsics(a.source / 'poses/sparse_txt')
    cropped = next(line.split() for line in (workspace / 'sparse/0/cameras.txt').read_text().splitlines() if line.strip())
    out = workspace / 'prior'
    out.mkdir(exist_ok=True)
    for image in sorted((workspace / 'images').iterdir()):
        frame = int(image.stem)
        path = a.depth_prior / f'{frame:08d}_grounded.npy'
        if not path.is_file() or frame not in intrinsics:
            continue
        depth = np.load(path).astype(np.float32) / radius
        mask = cv2.imread(str((a.masks or a.sequence / 'masks') / f'{frame:08d}.png'), cv2.IMREAD_GRAYSCALE)
        depth = np.where(mask > 127, depth, 0)
        height, width = depth.shape
        margin = (width - int(cropped[2])) // 2
        K = intrinsics[frame]
        o = a.principal_offset
        moved = cv2.warpAffine(depth, np.float32([[1, 0, width / 2 - o - K[0, 2]], [0, 1, height / 2 - o - K[1, 2]]]),
                               (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
        np.save(out / f'{image.stem}.npy', moved[margin:height - margin, margin:width - margin].astype(np.float32))


def export_depth(a: argparse.Namespace, model: Path, workspace: Path, radius: float) -> None:
    """PGSR's per-view depth, back on our working frames and in map units."""
    raw = a.output / 'raw_depth'
    subprocess.run([a.python, str(Path(__file__).with_name('pgsr_depth.py').resolve()), '-m', str(model),
                    '--iteration', str(a.iterations), '--output', str(raw.resolve())], cwd=a.repo.resolve(), check=True,
                   stdout=subprocess.DEVNULL)
    intrinsics = frame_intrinsics(a.source / 'poses/sparse_txt')
    out = a.output / 'depth'
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(raw.glob('*.npy')):
        frame = int(path.stem)
        height, width = cv2.imread(str(a.sequence / 'frames' / f'{frame:08d}.png'), cv2.IMREAD_GRAYSCALE).shape
        depth = np.load(path).astype(np.float32) * radius
        margin = (width - depth.shape[1]) // 2
        canvas = np.zeros((height, width), np.float32)
        canvas[margin:margin + depth.shape[0], margin:margin + depth.shape[1]] = depth
        K = intrinsics[frame]
        # The workspace moved each frame by (w/2 - offset - cx, h/2 - offset - cy) before cropping; move back.
        o = a.principal_offset
        back = np.float32([[1, 0, K[0, 2] + o - width / 2], [0, 1, K[1, 2] + o - height / 2]])
        canvas = cv2.warpAffine(canvas, back, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
        np.save(out / f'{frame:08d}_grounded.npy', canvas)
    (out / 'hybrid_depth.json').write_text(json.dumps({'method': 'PGSR rendered plane depth', 'model': str(model),
                                                       'frames': len(list(raw.glob('*.npy')))}, indent=2) + '\n')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('source', type=Path, help='Finished run of ours (poses/sparse_txt, export, export_object)')
    p.add_argument('--sequence', type=Path, required=True)
    p.add_argument('--masks', type=Path, help='Default: sequence/masks')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--python', default=str(Path.home() / '.venvs/me566-2dgs/bin/python'))
    p.add_argument('--repo', type=Path, default=Path('third_party/PGSR'))
    p.add_argument('--iterations', type=int, default=30000)
    p.add_argument('--principal-offset', type=float, default=0.,
                   help='Land each frame\'s principal point on w/2 - offset (0.5: where PGSR\'s rasteriser projects; '
                        'its own ray maths uses w/2)')
    p.add_argument('--init-points', type=Path, nargs='+', default=[],
                   help='Start from these point clouds (e.g. RUN/points/{object,scene}.ply) instead of the SfM points')
    p.add_argument('--max-points', type=int, default=300_000, help='Most --init-points kept (random subset)')
    p.add_argument('--train-args', default='', help="Extra train.py options as one string")
    p.add_argument('--depth-prior', type=Path,
                   help='Our depth (<frame>_grounded.npy on the working grid) kept as a loss during training '
                        '(object pixels only; pgsr_train.py)')
    p.add_argument('--depth-weight', type=float, default=1.0, help='Weight of the depth-prior term')
    p.add_argument('--depth-clip', type=float, default=0., help='Cap on the per-pixel relative difference (0 = none)')
    p.add_argument('--depth-decay', type=int, default=0, help='Fade the prior to zero over this many iterations (0 = constant)')
    p.add_argument('--voxel', type=float, default=.002, help='TSDF voxel in normalised units (PGSR DTU: 0.002)')
    p.add_argument('--original-sampling', action='store_true',
                   help="Run train.py directly, without pgsr_train.py's GPU pixel sampling")
    p.add_argument('--depth-only', action='store_true',
                   help='Save per-frame depth on our working grid instead of meshing with PGSR')
    a = p.parse_args()
    if a.depth_prior is not None and a.original_sampling:
        p.error('--depth-prior needs the pgsr_train.py launcher (drop --original-sampling)')
    workspace = a.output / 'pgsr'
    center, radius, depth_trunc = write_workspace(a.source, a.sequence, workspace, a.init_points, a.max_points,
                                                   a.principal_offset)
    for name in ('cameras.txt', 'images.txt', 'points3D.txt'):          # PGSR reads source/sparse/
        shutil.copy(workspace / 'sparse/0' / name, workspace / 'sparse' / name)
    repo, model = a.repo.resolve(), (workspace / 'model').resolve()
    prior_args = []
    if a.depth_prior is not None:
        write_priors(a, workspace, radius)
        prior_args = ['--depth-prior', str((workspace / 'prior').resolve()), '--depth-weight', str(a.depth_weight),
                      '--depth-clip', str(a.depth_clip), '--depth-decay', str(a.depth_decay)]
    started = time.time()
    launcher = 'train.py' if a.original_sampling else str(Path(__file__).with_name('pgsr_train.py').resolve())
    subprocess.run([a.python, launcher, '-s', str(workspace.resolve()), '-m', str(model), '--quiet', '-r', '1',
                    '--ncc_scale', '0.5', '--iterations', str(a.iterations), '--position_lr_max_steps', str(a.iterations),
                    '--test_iterations', '-1', '--save_iterations', str(a.iterations), *a.train_args.split(), *prior_args],
                   cwd=repo, check=True)
    trained = time.time()
    if a.depth_only:
        export_depth(a, model, workspace, radius)
        report = {'source': str(a.source), 'init_points': [str(path) for path in a.init_points], 'iterations': a.iterations,
                  'train_args': a.train_args, 'gpu_sampling': not a.original_sampling,
                  'depth_prior': str(a.depth_prior) if a.depth_prior else None, 'depth_weight': a.depth_weight, 'depth_clip': a.depth_clip, 'depth_decay': a.depth_decay,
                  'train_s': trained - started, 'depth_s': time.time() - trained,
                  'normalisation_radius': radius}
        (a.output / 'pgsr.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        return
    subprocess.run([a.python, 'render.py', '-m', str(model), '--iteration', str(a.iterations), '--quiet', '--skip_test',
                    '--num_cluster', '1', '--voxel_size', str(a.voxel), '--max_depth', f'{depth_trunc:.3f}'],
                   cwd=repo, check=True)
    meshed = time.time()
    mesh = o3d.io.read_triangle_mesh(str(model / 'mesh/tsdf_fusion_post.ply'))
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * radius + center)
    mesh = cull(mesh, a.source / 'poses/sparse_txt', a.source / 'poses/export/trajectory.csv',
                a.masks or a.sequence / 'masks')
    (a.output / 'mesh').mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(a.output / 'mesh/object.ply'), mesh)
    shutil.copytree(a.source / 'poses/export', a.output / 'poses/export', dirs_exist_ok=True)
    report = {'source': str(a.source), 'init_points': [str(path) for path in a.init_points], 'iterations': a.iterations,
              'train_args': a.train_args, 'train_s': trained - started, 'mesh_s': meshed - trained,
              'normalisation_radius': radius, 'depth_trunc': depth_trunc, 'faces': len(mesh.triangles)}
    (a.output / 'pgsr.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
