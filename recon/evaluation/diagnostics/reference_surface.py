#!/usr/bin/env python3
"""Build a reference surface mesh from a DTU laser-scan point cloud (diagnostic).

DTU's reference is a point cloud (stl_total.ply, millimetres). Rendering
reference depth needs a surface, so the points are thinned to 0.3 mm, given
normals oriented towards the mean camera centre (DTU's cameras all face the
object from one side), meshed with screened Poisson reconstruction (Kazhdan
and Hoppe, "Screened Poisson Surface Reconstruction", TOG 2013, via Open3D),
and trimmed to triangles within 0.6 mm of a scanned point. The mesh is written
in metres, matching the DTU poses produced by recon.inputs.dtu.prepare.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from recon.common.ply import write_mesh
from recon.common.progress import done, spinner


def build(reference: Path, pose_dir: Path, output: Path, voxel: float = .3,
          depth: int = 11, trim: float = .6) -> dict:
    cloud = o3d.io.read_point_cloud(str(reference / 'stl_total.ply')).voxel_down_sample(voxel)
    centers = np.array([np.loadtxt(path)[:3, 3] * 1000 for path in sorted(pose_dir.glob('*.txt'))])
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel, max_nn=30))
    cloud.orient_normals_towards_camera_location(centers.mean(axis=0))
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=depth)
    points = np.asarray(cloud.points)
    distance = cKDTree(points).query(np.asarray(mesh.vertices), workers=-1)[0]
    mesh.remove_vertices_by_mask(distance > trim)
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) / 1000.)
    mesh.vertex_colors = o3d.utility.Vector3dVector()
    mesh.compute_vertex_normals()
    write_mesh(output, mesh)
    samples = np.asarray(mesh.sample_points_uniformly(200000).points) * 1000.
    error = cKDTree(points).query(samples, workers=-1)[0]
    report = {'source': str(reference / 'stl_total.ply'), 'scan_points': len(points),
              'faces': len(mesh.triangles), 'surface_to_scan_median_mm': float(np.median(error)),
              'surface_to_scan_p90_mm': float(np.percentile(error, 90))}
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True, help='DTU reference/ folder')
    p.add_argument('--pose-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    with spinner(f'Poisson reference surface ({a.output.stem})'):
        report = build(a.reference, a.pose_dir, a.output)
    done(f"Reference surface: {report['faces']:,} faces, "
         f"{report['surface_to_scan_median_mm']:.2f} mm median from the scan")


if __name__ == '__main__':
    main()
