"""Mesh cleanup should remove debris without deleting the main surface."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import open3d as o3d

from recon.surface.clean import clean

class CleanMeshTests(unittest.TestCase):
    def test_keeps_main_component_and_removes_isolated_triangle(self):
        box=o3d.geometry.TriangleMesh.create_box()
        vertices=np.vstack((np.asarray(box.vertices),[[5,0,0],[6,0,0],[5,1,0]]))
        faces=np.vstack((np.asarray(box.triangles),[[8,9,10]]))
        mesh=o3d.geometry.TriangleMesh()
        mesh.vertices=o3d.utility.Vector3dVector(vertices)
        mesh.triangles=o3d.utility.Vector3iVector(faces)
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.ply';output=Path(tmp)/'clean.ply'
            self.assertTrue(o3d.io.write_triangle_mesh(str(source),mesh))
            result=clean(source,output,min_component_triangles=5,smooth_iterations=0)
            self.assertEqual(result['kept_components'],1)
            self.assertEqual(len(o3d.io.read_triangle_mesh(str(output)).triangles),12)

if __name__=='__main__':unittest.main()


def test_triangle_components_match_open3d_edge_connectivity():
    import open3d as o3d
    from recon.surface.components import triangle_components
    # Two triangles sharing an edge, one touching them only at a vertex, one isolated.
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0],
                         [2, 2, 0], [5, 5, 0], [6, 5, 0], [5, 6, 0]], float)
    triangles = np.array([[0, 1, 2], [1, 3, 2], [3, 4, 5], [6, 7, 8]])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                     o3d.utility.Vector3iVector(triangles))
    expected_ids, expected_sizes, _ = mesh.cluster_connected_triangles()
    labels, sizes = triangle_components(triangles)
    np.testing.assert_array_equal(sizes[labels],
                                  np.asarray(expected_sizes)[np.asarray(expected_ids)])


def test_min_fraction_drops_pieces_small_next_to_the_main_surface(tmp_path: Path):
    big = o3d.geometry.TriangleMesh.create_sphere(resolution=20)
    small = o3d.geometry.TriangleMesh.create_box().translate([5, 0, 0])
    source, output = tmp_path / 'source.ply', tmp_path / 'clean.ply'
    o3d.io.write_triangle_mesh(str(source), big + small)
    kept = clean(source, output, min_component_triangles=1, smooth_iterations=0)
    assert kept['kept_components'] == 2
    dropped = clean(source, output, min_component_triangles=1, smooth_iterations=0,
                    min_fraction=.05)
    assert dropped['kept_components'] == 1
    assert len(o3d.io.read_triangle_mesh(str(output)).triangles) == len(big.triangles)


def test_hole_filling_keeps_vertex_positions(tmp_path: Path):
    sphere = o3d.geometry.TriangleMesh.create_sphere(resolution=20)
    sphere.remove_triangles_by_index([0, 1])
    source, output = tmp_path / 'source.ply', tmp_path / 'clean.ply'
    o3d.io.write_triangle_mesh(str(source), sphere)
    clean(source, output, min_component_triangles=1, smooth_iterations=0, hole_size=1.)
    result = o3d.io.read_triangle_mesh(str(output))
    assert len(result.triangles) > len(sphere.triangles)
    assert np.allclose(np.linalg.norm(np.asarray(result.vertices), axis=1), 1, atol=1e-5)
