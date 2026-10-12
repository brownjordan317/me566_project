#!/usr/bin/env python3
"""Remove floating TSDF fragments, fill small tears, and optionally smooth.

The result is written with 32-bit float positions and normals plus uchar
vertex colours (see recon/common/ply.py).

Hole filling is Open3D's TriangleMesh.fill_holes; smoothing is Taubin's
lambda/mu filter (Taubin, SIGGRAPH 1995) as implemented by Open3D.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import open3d as o3d

from recon.surface.components import triangle_components
from recon.common.ply import write_mesh
from recon.common.progress import done

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

def clean(source:Path,output:Path,min_component_triangles:int=1000,smooth_iterations:int=5,
          min_fraction:float=0.,hole_size:float=0.):
    mesh=o3d.io.read_triangle_mesh(str(source))
    if not len(mesh.triangles):raise ValueError(f'Empty mesh: {source}')
    ids,sizes=triangle_components(np.asarray(mesh.triangles))
    if not len(sizes):raise ValueError('No connected components')
    # Keeping tiny isolated sheets produces floating debris in Blender. Scene
    # meshes also drop pieces small relative to the main surface: those are
    # depth outliers that a few views agreed on, floating off the surface.
    cutoff=min(max(min_component_triangles,int(min_fraction*sizes.max())),int(sizes.max()))
    mesh.remove_triangles_by_mask(sizes[ids]<cutoff)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles();mesh.remove_duplicated_triangles()
    if hole_size>0:
        # Keep the tensor mesh alive: fill_holes' result shares its buffers.
        tensor_mesh=o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        mesh=tensor_mesh.fill_holes(hole_size=hole_size).to_legacy()
    if smooth_iterations:
        mesh=mesh.filter_smooth_taubin(number_of_iterations=smooth_iterations,
                                        lambda_filter=.5,mu=-.53)
    mesh.compute_vertex_normals()
    write_mesh(output,mesh)
    summary={'source_components':len(sizes),'kept_components':int(np.sum(sizes>=cutoff)),
             'source_faces':int(sizes.sum()),'faces':len(mesh.triangles),
             'vertices':len(mesh.vertices),'smoothing_iterations':smooth_iterations,
             'hole_size':hole_size}
    output.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--min-component-triangles',type=int,default=1000)
    p.add_argument('--min-fraction',type=float,default=0.,
                   help='Also drop pieces smaller than this fraction of the largest')
    p.add_argument('--hole-size',type=float,default=0.,
                   help='Fill holes up to this size in map units (0 = off)')
    p.add_argument('--smooth-iterations',type=int,default=5)
    a=p.parse_args()
    summary=clean(a.source,a.output,a.min_component_triangles,a.smooth_iterations,
                  a.min_fraction,a.hole_size)
    done(f"Mesh cleanup: kept {summary['kept_components']} of {summary['source_components']} pieces, {summary['faces']:,} faces")
if __name__=='__main__':main()
