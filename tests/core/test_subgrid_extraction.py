import unittest
import numpy as np

from core.grids import structured, simplex
from core.grids.extract_subgrid import extract_subgrid

def compare_grid_geometries(g, h, sub_c, sub_f, sub_n):
    assert np.array_equal(g.nodes[:, sub_n], h.nodes)
    assert np.array_equal(g.face_areas[sub_f], h.face_areas)
    assert np.array_equal(g.face_normals[:, sub_f], h.face_normals)
    assert np.array_equal(g.face_centers[:, sub_f], h.face_centers)
    assert np.array_equal(g.cell_centers[:, sub_c], h.cell_centers)
    assert np.array_equal(g.cell_volumes[sub_c], h.cell_volumes)


class TestGrids(unittest.TestCase):

    def test_cart_2d(self):
        g = structured.CartGrid([3, 2])
        g.nodes = g.nodes + 0.2 * np.random.random((g.dim, g.nodes.shape[1]))
        g.compute_geometry()

        c = np.array([0, 1, 3])

        h, sub_f, sub_n = extract_subgrid(g, c)

        true_nodes = np.array([0, 1, 2, 4, 5, 6, 8, 9])
        true_faces = np.array([0, 1, 2, 4, 5, 8, 9, 11, 12, 14])

        assert np.array_equal(true_nodes, sub_n)
        assert np.array_equal(true_faces, sub_f)

        compare_grid_geometries(g, h, c, true_faces, true_nodes)

    def test_cart_3d(self):
        g = structured.CartGrid([4, 3, 3])
        g.nodes = g.nodes + 0.2 * np.random.random((g.dim, g.nodes.shape[1]))
        g.compute_geometry()

        # Pick out cells (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 2), (1, 0, 2)
        c = np.array([0, 1, 4, 12, 24, 25])

        h, sub_f, sub_n = extract_subgrid(g, c)

        # There are 20 nodes in each layer
        nodes_0 = np.array([0, 1, 2, 5, 6, 7, 10, 11])
        nodes_1 = nodes_0 + 20
        nodes_2 = np.array([0, 1, 2, 5, 6, 7]) + 40
        nodes_3 = nodes_2 + 20
        true_nodes = np.hstack((nodes_0, nodes_1, nodes_2, nodes_3))


        faces_x = np.array([0, 1, 2, 5, 6, 15, 16, 30, 31, 32])
        # y-faces start on 45
        faces_y = np.array([45, 46, 49, 50, 53, 61, 65, 77, 78, 81, 82])
        # z-faces start on 93
        faces_z = np.array([93, 94, 97, 105, 106, 109, 117, 118, 129, 130])
        true_faces = np.hstack((faces_x, faces_y, faces_z))

        assert np.array_equal(true_nodes, sub_n)
        assert np.array_equal(true_faces, sub_f)

        compare_grid_geometries(g, h, c, true_faces, true_nodes)

    def test_simplex_2d(self):
        g = simplex.StructuredTriangleGrid(np.array([3, 2]))
        g.compute_geometry()
        c = np.array([0, 1, 3])
        true_nodes = np.array([0, 1, 4, 5, 6])
        true_faces = np.array([0, 1, 2, 4, 5, 10, 13])

        h, sub_f, sub_n = extract_subgrid(g, c)

        assert np.array_equal(true_nodes, sub_n)
        assert np.array_equal(true_faces, sub_f)

        compare_grid_geometries(g, h, c, true_faces, true_nodes)

    if __name__ == '__main__':
        unittest.main()
