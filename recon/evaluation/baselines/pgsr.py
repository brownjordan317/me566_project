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
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from recon.evaluation.baselines.twodgs import cull, write_workspace


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('source', type=Path, help='Finished run of ours (poses/sparse_txt, export, export_object)')
    p.add_argument('--sequence', type=Path, required=True)
    p.add_argument('--masks', type=Path, help='Default: sequence/masks')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--python', default=str(Path.home() / '.venvs/me566-2dgs/bin/python'))
    p.add_argument('--repo', type=Path, default=Path('third_party/PGSR'))
    p.add_argument('--iterations', type=int, default=30000)
    p.add_argument('--init-points', type=Path, nargs='+', default=[],
                   help='Start from these point clouds (e.g. RUN/points/{object,scene}.ply) instead of the SfM points')
    p.add_argument('--train-args', default='', help="Extra train.py options as one string")
    p.add_argument('--voxel', type=float, default=.002, help='TSDF voxel in normalised units (PGSR DTU: 0.002)')
    a = p.parse_args()
    workspace = a.output / 'pgsr'
    center, radius, depth_trunc = write_workspace(a.source, a.sequence, workspace, a.init_points)
    for name in ('cameras.txt', 'images.txt', 'points3D.txt'):          # PGSR reads source/sparse/
        shutil.copy(workspace / 'sparse/0' / name, workspace / 'sparse' / name)
    repo, model = a.repo.resolve(), (workspace / 'model').resolve()
    started = time.time()
    subprocess.run([a.python, 'train.py', '-s', str(workspace.resolve()), '-m', str(model), '--quiet', '-r', '1',
                    '--ncc_scale', '0.5', '--iterations', str(a.iterations), '--position_lr_max_steps', str(a.iterations),
                    '--test_iterations', '-1', '--save_iterations', str(a.iterations), *a.train_args.split()],
                   cwd=repo, check=True)
    trained = time.time()
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
