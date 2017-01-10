import unittest
import numpy as np

from core.grids import structured, simplex
from core.grids import partition

class TestCartGrids(unittest.TestCase):

    def test_2d_coarse_dims_specified(self):
        g = structured.CartGrid([4, 10])
        coarse_dims = np.array([2, 3])
        p = partition.partition_structured(g, coarse_dims)

        p_known = np.array([[ 0.,  0.,  1.,  1.],
                            [ 0.,  0.,  1.,  1.],
                            [ 0.,  0.,  1.,  1.],
                            [ 2.,  2.,  3.,  3.],
                            [ 2.,  2.,  3.,  3.],
                            [ 2.,  2.,  3.,  3.],
                            [ 4.,  4.,  5.,  5.],
                            [ 4.,  4.,  5.,  5.],
                            [ 4.,  4.,  5.,  5.],
                            [ 4.,  4.,  5.,  5.]], dtype='int').ravel('C')
        assert np.allclose(p, p_known)

    def test_3d_coarse_dims_specified(self):
        g = structured.CartGrid([4, 4, 4])
        coarse_dims = np.array([2, 2, 2])

        p = partition.partition_structured(g, coarse_dims)
        p_known = np.array([[0, 0, 1, 1],
                            [0, 0, 1, 1],
                            [2, 2, 3, 3],
                            [2, 2, 3, 3],
                            [0, 0, 1, 1],
                            [0, 0, 1, 1],
                            [2, 2, 3, 3],
                            [2, 2, 3, 3],
                            [4, 4, 5, 5],
                            [4, 4, 5, 5],
                            [6, 6, 7, 7],
                            [6, 6, 7, 7],
                            [4, 4, 5, 5],
                            [4, 4, 5, 5],
                            [6, 6, 7, 7],
                            [6, 6, 7, 7]]).ravel('C')
        assert np.allclose(p, p_known)

    def test_3d_coarse_dims_specified_unequal_size(self):
        g = structured.CartGrid(np.array([6, 5, 4]))
        coarse_dims = np.array([3, 2, 2])

        p = partition.partition_structured(g, coarse_dims)
        # This just happens to be correct
        p_known = np.array([0,  0,  1,  1,  2,  2,  0,  0,  1, 1,  2,
                            2,  3,  3,  4,  4,  5,  5,  3, 3,  4,  4,
                            5,  5,  3,  3,  4,  4,  5,  5, 0,  0,  1,
                            1,  2,  2,  0,  0,  1,  1,  2, 2,  3,  3,
                            4,  4,  5,  5,  3,  3,  4,  4, 5,  5,  3,
                            3,  4,  4,  5,  5,  6,  6,  7, 7,  8,  8,
                            6,  6,  7,  7,  8,  8,  9,  9, 10, 10, 11,
                            11,  9,  9, 10, 10, 11, 11,  9, 9, 10, 10,
                            11, 11,  6,  6,  7,  7,  8,  8, 6,  6,  7,
                            7,  8,  8,  9,  9, 10, 10, 11, 11,  9,  9,
                            10, 10, 11, 11,  9,  9, 10, 10, 11, 11])

        assert np.allclose(p, p_known)


    if __name__ == '__main__':
        unittest.main()


class TestCoarseDimensionDeterminer(unittest.TestCase):

    def test_coarse_dimensions_all_fine(self):
        nx = np.array([5, 5, 5])
        target = nx.prod()
        coarse = partition.determine_coarse_dimensions(target, nx)
        assert np.array_equal(nx, coarse)

    def test_coarse_dimensions_single_coarse(self):
        nx = np.array([5, 5, 5])
        target = 1
        coarse = partition.determine_coarse_dimensions(target, nx)
        assert np.array_equal(coarse, np.ones_like(nx))

    def test_anisotropic_2d(self):
        nx = np.array([10, 4])
        target = 4
        coarse = partition.determine_coarse_dimensions(target, nx)
        assert np.array_equal(coarse, np.array([2, 2]))

    def test_round_down(self):
        nx = np.array([10, 10])
        target = 17
        coarse = partition.determine_coarse_dimensions(target, nx)
        assert np.array_equal(coarse, np.array([4, 4]))

    def test_round_up_and_down(self):
        nx = np.array([10, 10])
        target = 19
        coarse = partition.determine_coarse_dimensions(target, nx)
        assert np.array_equal(np.sort(coarse), np.array([4, 5]))

    def test_bounded_single(self):
        # The will ideally require a 5x4 grid, but the limit on two cells in
        # the  y-direction should redirect to a 10x2.
        nx = np.array([100, 2])
        target = 20
        coarse = partition.determine_coarse_dimensions(target, nx)
        assert np.array_equal(np.sort(coarse), np.array([2, 10]))

    def test_two_bounds(self):
        # This will seemingly require 900^1/3 ~ 10 cells in each direction, but
        # the z-dimension has only 1, thus the two other have to do 30 each. y
        # can only do 15, so the loop has to be reset once more to get the
        # final 60.
        nx = np.array([2000, 15, 1])
        target = 900
        coarse = partition.determine_coarse_dimensions(target, nx)
        assert np.array_equal(np.sort(coarse), np.array([1, 15, 60]))

    if __name__ == '__main__':
        unittest.main()




class TestGrids(unittest.TestCase):

    def compare_grid_geometries(self, g, h, sub_c, sub_f, sub_n):
        assert np.array_equal(g.nodes[:, sub_n], h.nodes)
        assert np.array_equal(g.face_areas[sub_f], h.face_areas)
        assert np.array_equal(g.face_normals[:, sub_f], h.face_normals)
        assert np.array_equal(g.face_centers[:, sub_f], h.face_centers)
        assert np.array_equal(g.cell_centers[:, sub_c], h.cell_centers)
        assert np.array_equal(g.cell_volumes[sub_c], h.cell_volumes)

    def test_cart_2d(self):
        g = structured.CartGrid([3, 2])
        g.nodes = g.nodes + 0.2 * np.random.random(g.nodes.shape)
        g.compute_geometry()

        c = np.array([0, 1, 3])

        h, sub_f, sub_n = partition.extract_single(g, c)

        true_nodes = np.array([0, 1, 2, 4, 5, 6, 8, 9])
        true_faces = np.array([0, 1, 2, 4, 5, 8, 9, 11, 12, 14])

        assert np.array_equal(true_nodes, sub_n)
        assert np.array_equal(true_faces, sub_f)

        self.compare_grid_geometries(g, h, c, true_faces, true_nodes)

    def test_cart_3d(self):
        g = structured.CartGrid([4, 3, 3])
        g.nodes = g.nodes + 0.2 * np.random.random((g.dim, g.nodes.shape[1]))
        g.compute_geometry()

        # Pick out cells (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 2), (1, 0, 2)
        c = np.array([0, 1, 4, 12, 24, 25])

        h, sub_f, sub_n = partition.extract_single(g, c)

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

        self.compare_grid_geometries(g, h, c, true_faces, true_nodes)

    def test_simplex_2d(self):
        g = simplex.StructuredTriangleGrid(np.array([3, 2]))
        g.compute_geometry()
        c = np.array([0, 1, 3])
        true_nodes = np.array([0, 1, 4, 5, 6])
        true_faces = np.array([0, 1, 2, 4, 5, 10, 13])

        h, sub_f, sub_n = partition.extract_single(g, c)

        assert np.array_equal(true_nodes, sub_n)
        assert np.array_equal(true_faces, sub_f)

        self.compare_grid_geometries(g, h, c, true_faces, true_nodes)

    if __name__ == '__main__':
        unittest.main()
