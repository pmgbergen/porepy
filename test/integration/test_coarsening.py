import unittest
import sys

import numpy as np
import scipy.sparse as sps

from porepy.grids import structured, simplex
from porepy.grids import coarsening as co
from porepy.fracs import meshing

# ------------------------------------------------------------------------------#


class BasicsTest(unittest.TestCase):

    # ------------------------------------------------------------------------------#

    def test_coarse_grid_2d(self):
        g = structured.CartGrid([3, 2])
        g.compute_geometry()
        co.generate_coarse_grid(g, [5, 2, 2, 5, 2, 2])

        assert g.num_cells == 2
        assert g.num_faces == 12
        assert g.num_nodes == 11

        pt = np.tile(np.array([2, 1, 0]), (g.nodes.shape[1], 1)).T
        find = np.isclose(pt, g.nodes).all(axis=0)
        assert find.any() == False

        faces_cell0, _, orient_cell0 = sps.find(g.cell_faces[:, 0])
        assert np.array_equal(faces_cell0, [1, 2, 4, 5, 7, 8, 10, 11])
        assert np.array_equal(orient_cell0, [-1, 1, -1, 1, -1, -1, 1, 1])

        faces_cell1, _, orient_cell1 = sps.find(g.cell_faces[:, 1])
        assert np.array_equal(faces_cell1, [0, 1, 3, 4, 6, 9])
        assert np.array_equal(orient_cell1, [-1, 1, -1, 1, -1, 1])

        known = np.array(
            [
                [0, 4],
                [1, 5],
                [3, 6],
                [4, 7],
                [5, 8],
                [6, 10],
                [0, 1],
                [1, 2],
                [2, 3],
                [7, 8],
                [8, 9],
                [9, 10],
            ]
        )

        for f in np.arange(g.num_faces):
            assert np.array_equal(sps.find(g.face_nodes[:, f])[0], known[f, :])

    # ------------------------------------------------------------------------------#

    def test_coarse_grid_3d(self):
        g = structured.CartGrid([2, 2, 2])
        g.compute_geometry()
        co.generate_coarse_grid(g, [0, 0, 0, 0, 1, 1, 2, 2])

        assert g.num_cells == 3
        assert g.num_faces == 30
        assert g.num_nodes == 27

        faces_cell0, _, orient_cell0 = sps.find(g.cell_faces[:, 0])
        known = [0, 1, 2, 3, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25]
        assert np.array_equal(faces_cell0, known)
        known = [-1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1]
        assert np.array_equal(orient_cell0, known)

        faces_cell1, _, orient_cell1 = sps.find(g.cell_faces[:, 1])
        known = [4, 5, 12, 13, 14, 15, 22, 23, 26, 27]
        assert np.array_equal(faces_cell1, known)
        known = [-1, 1, -1, -1, 1, 1, -1, -1, 1, 1]
        assert np.array_equal(orient_cell1, known)

        faces_cell2, _, orient_cell2 = sps.find(g.cell_faces[:, 2])
        known = [6, 7, 14, 15, 16, 17, 24, 25, 28, 29]
        assert np.array_equal(faces_cell2, known)
        known = [-1, 1, -1, -1, 1, 1, -1, -1, 1, 1]
        assert np.array_equal(orient_cell2, known)

        known = np.array(
            [
                [0, 3, 9, 12],
                [2, 5, 11, 14],
                [3, 6, 12, 15],
                [5, 8, 14, 17],
                [9, 12, 18, 21],
                [11, 14, 20, 23],
                [12, 15, 21, 24],
                [14, 17, 23, 26],
                [0, 1, 9, 10],
                [1, 2, 10, 11],
                [6, 7, 15, 16],
                [7, 8, 16, 17],
                [9, 10, 18, 19],
                [10, 11, 19, 20],
                [12, 13, 21, 22],
                [13, 14, 22, 23],
                [15, 16, 24, 25],
                [16, 17, 25, 26],
                [0, 1, 3, 4],
                [1, 2, 4, 5],
                [3, 4, 6, 7],
                [4, 5, 7, 8],
                [9, 10, 12, 13],
                [10, 11, 13, 14],
                [12, 13, 15, 16],
                [13, 14, 16, 17],
                [18, 19, 21, 22],
                [19, 20, 22, 23],
                [21, 22, 24, 25],
                [22, 23, 25, 26],
            ]
        )

        for f in np.arange(g.num_faces):
            assert np.array_equal(sps.find(g.face_nodes[:, f])[0], known[f, :])

    # ------------------------------------------------------------------------------#

    def test_coarse_grid_2d_1d(self):
        part = np.array([0, 0, 1, 1, 2, 0, 3, 1])
        f = np.array([[2, 2], [0, 2]])

        gb = meshing.cart_grid([f], [4, 2])
        gb.compute_geometry()
        co.generate_coarse_grid(gb, part)

        # Test
        known = np.array([1, 5, 18, 19])

        for _, d in gb.edges():
            faces = sps.find(d["face_cells"])[1]
            assert np.array_equal(faces, known)

    # ------------------------------------------------------------------------------#

    def test_coarse_grid_2d_1d_cross(self):
        # NOTE: Since for python 2.7 and 3.5 the meshes in gridbucket may have
        # non-fixed order, we need to exclude this test.
        if sys.version_info >= (3, 6):
            part = np.zeros(36)
            part[[0, 1, 2, 6, 7]] = 1
            part[[8, 14, 13]] = 2
            part[[12, 18, 19]] = 3
            part[[24, 30, 31, 32]] = 4
            part[[21, 22, 23, 27, 28, 29, 33, 34, 35]] = 5
            part[[9]] = 6
            part[[15, 16, 17]] = 7
            part[[9, 10]] = 8
            part[[20, 26, 25]] = 9
            part[[3, 4, 5, 11]] = 10
            f1 = np.array([[3., 3.], [1., 5.]])
            f2 = np.array([[1., 5.], [3., 3.]])

            gb = meshing.cart_grid([f1, f2], [6, 6])
            gb.compute_geometry()

            cell_centers_1 = np.array(
                [
                    [3.00000000e+00, 3.00000000e+00, 3.00000000e+00, 3.00000000e+00],
                    [4.50000000e+00, 3.50000000e+00, 2.50000000e+00, 1.50000000e+00],
                    [-1.66533454e-16, -5.55111512e-17, 5.55111512e-17, 1.66533454e-16],
                ]
            )
            cell_centers_2 = np.array(
                [
                    [4.50000000e+00, 3.50000000e+00, 2.50000000e+00, 1.50000000e+00],
                    [3.00000000e+00, 3.00000000e+00, 3.00000000e+00, 3.00000000e+00],
                    [-1.66533454e-16, -5.55111512e-17, 5.55111512e-17, 1.66533454e-16],
                ]
            )

            co.generate_coarse_grid(gb, part)

            # Test
            for e_d in gb.edges():
                faces = sps.find(e_d[1]["face_cells"])[1]

                if (e_d[0][0].dim == 0 and e_d[0][1].dim == 1) or (
                    e_d[0][0].dim == 1 and e_d[0][1].dim == 0
                ):
                    known = [2, 5]

                if (e_d[0][0].dim == 1 and e_d[0][1].dim == 2) or (
                    e_d[0][0].dim == 2 and e_d[0][1].dim == 1
                ):

                    g = e_d[0][0] if e_d[0][0].dim == 1 else e_d[0][1]

                    if np.allclose(g.cell_centers, cell_centers_1):
                        known = [5, 10, 14, 18, 52, 53, 54, 55]
                    elif np.allclose(g.cell_centers, cell_centers_2):
                        known = [37, 38, 39, 40, 56, 57, 58, 59]
                    else:
                        raise ValueError("Grid not found")

                assert np.array_equal(faces, known)

    # ------------------------------------------------------------------------------#

    def test_coarse_grid_3d_2d(self):
        f = np.array([[2., 2., 2., 2.], [0., 2., 2., 0.], [0., 0., 2., 2.]])
        gb = meshing.cart_grid([f], [4, 2, 2])
        gb.compute_geometry()

        g = gb.get_grids(lambda g: g.dim == gb.dim_max())[0]
        part = np.zeros(g.num_cells)
        part[g.cell_centers[0, :] < 2.] = 1
        co.generate_coarse_grid(gb, part)

        # Test
        known_indices = np.array([1, 3, 0, 2, 1, 3, 0, 2])
        known = np.array([1, 4, 7, 10, 44, 45, 46, 47])

        for _, d in gb.edges():
            indices, faces, _ = sps.find(d["face_cells"])
            assert np.array_equal(indices, known_indices)
            assert np.array_equal(faces, known)

    # ------------------------------------------------------------------------------#

    def test_coarse_grid_3d_2d_cross(self):
        # NOTE: Since for python 2.7 and 3.5 the meshes in gridbucket may have
        # non-fixed order, we need to exclude this test.
        if sys.version_info >= (3, 6):
            f1 = np.array([[3., 3., 3., 3.], [1., 5., 5., 1.], [1., 1., 5., 5.]])
            f2 = np.array([[1., 5., 5., 1.], [1., 1., 5., 5.], [3., 3., 3., 3.]])
            gb = meshing.cart_grid([f1, f2], [6, 6, 6])
            gb.compute_geometry()

            g = gb.get_grids(lambda g: g.dim == gb.dim_max())[0]
            part = np.zeros(g.num_cells)
            p1, p2 = g.cell_centers[0, :] < 3., g.cell_centers[2, :] < 3.
            part[np.logical_and(p1, p2)] = 1
            part[np.logical_and(p1, np.logical_not(p2))] = 2
            part[np.logical_and(np.logical_not(p1), p2)] = 3
            part[np.logical_and(np.logical_not(p1), np.logical_not(p2))] = 4

            co.generate_coarse_grid(gb, part)

            cell_centers_1 = np.array(
                [
                    [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
                    [
                        1.5,
                        1.5,
                        1.5,
                        1.5,
                        2.5,
                        2.5,
                        2.5,
                        2.5,
                        3.5,
                        3.5,
                        3.5,
                        3.5,
                        4.5,
                        4.5,
                        4.5,
                        4.5,
                    ],
                    [
                        4.5,
                        3.5,
                        2.5,
                        1.5,
                        4.5,
                        3.5,
                        2.5,
                        1.5,
                        4.5,
                        3.5,
                        2.5,
                        1.5,
                        4.5,
                        3.5,
                        2.5,
                        1.5,
                    ],
                ]
            )
            cell_centers_2 = np.array(
                [
                    [
                        1.5,
                        2.5,
                        3.5,
                        4.5,
                        1.5,
                        2.5,
                        3.5,
                        4.5,
                        1.5,
                        2.5,
                        3.5,
                        4.5,
                        1.5,
                        2.5,
                        3.5,
                        4.5,
                    ],
                    [
                        1.5,
                        1.5,
                        1.5,
                        1.5,
                        2.5,
                        2.5,
                        2.5,
                        2.5,
                        3.5,
                        3.5,
                        3.5,
                        3.5,
                        4.5,
                        4.5,
                        4.5,
                        4.5,
                    ],
                    [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
                ]
            )

            # Test
            for e_d in gb.edges():
                indices, faces, _ = sps.find(e_d[1]["face_cells"])

                if (e_d[0][0].dim == 1 and e_d[0][1].dim == 2) or (
                    e_d[0][0].dim == 2 and e_d[0][1].dim == 1
                ):
                    known_indices = [3, 2, 1, 0, 3, 2, 1, 0]
                    known = [2, 7, 12, 17, 40, 41, 42, 43]

                if (e_d[0][0].dim == 2 and e_d[0][1].dim == 3) or (
                    e_d[0][0].dim == 3 and e_d[0][1].dim == 2
                ):

                    g = e_d[0][0] if e_d[0][0].dim == 2 else e_d[0][1]

                    if np.allclose(g.cell_centers, cell_centers_1):
                        known_indices = [
                            3,
                            7,
                            11,
                            15,
                            2,
                            6,
                            10,
                            14,
                            1,
                            5,
                            9,
                            13,
                            0,
                            4,
                            8,
                            12,
                            3,
                            7,
                            11,
                            15,
                            2,
                            6,
                            10,
                            14,
                            1,
                            5,
                            9,
                            13,
                            0,
                            4,
                            8,
                            12,
                        ]
                        known = [
                            22,
                            25,
                            28,
                            31,
                            40,
                            43,
                            46,
                            49,
                            58,
                            61,
                            64,
                            67,
                            76,
                            79,
                            82,
                            85,
                            288,
                            289,
                            290,
                            291,
                            292,
                            293,
                            294,
                            295,
                            296,
                            297,
                            298,
                            299,
                            300,
                            301,
                            302,
                            303,
                        ]
                    elif np.allclose(g.cell_centers, cell_centers_2):
                        known_indices = [
                            0,
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                            9,
                            10,
                            11,
                            12,
                            13,
                            14,
                            15,
                            0,
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                            9,
                            10,
                            11,
                            12,
                            13,
                            14,
                            15,
                        ]
                        known = [
                            223,
                            224,
                            225,
                            226,
                            229,
                            230,
                            231,
                            232,
                            235,
                            236,
                            237,
                            238,
                            241,
                            242,
                            243,
                            244,
                            304,
                            305,
                            306,
                            307,
                            308,
                            309,
                            310,
                            311,
                            312,
                            313,
                            314,
                            315,
                            316,
                            317,
                            318,
                            319,
                        ]
                    else:
                        raise ValueError("Grid not found")

                assert np.array_equal(indices, np.array(known_indices))
                assert np.array_equal(faces, np.array(known))

    # ------------------------------------------------------------------------------#

    def test_create_partition_2d_cart(self):
        g = structured.CartGrid([5, 5])
        g.compute_geometry()
        part = co.create_partition(co.tpfa_matrix(g))
        known = np.array(
            [0, 0, 0, 1, 1, 0, 0, 2, 1, 1, 3, 2, 2, 2, 1, 3, 3, 2, 4, 4, 3, 3, 4, 4, 4]
        )
        assert np.array_equal(part, known)

    # ------------------------------------------------------------------------------#

    def test_create_partition_2d_tri(self):
        g = simplex.StructuredTriangleGrid([3, 2])
        g.compute_geometry()
        part = co.create_partition(co.tpfa_matrix(g))
        known = np.array([1, 1, 1, 0, 0, 1, 0, 2, 2, 0, 2, 2])
        known_map = np.array([4, 3, 7, 5, 11, 8, 1, 2, 10, 6, 12, 9]) - 1
        assert np.array_equal(part, known[known_map])

    # ------------------------------------------------------------------------------#

    def test_create_partition_2d_cart_cdepth4(self):
        g = structured.CartGrid([10, 10])
        g.compute_geometry()
        part = co.create_partition(co.tpfa_matrix(g), cdepth=4)
        known = (
            np.array(
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    3,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    3,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    2,
                    2,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    2,
                    2,
                    2,
                    2,
                    2,
                ]
            )
            - 1
        )
        assert np.array_equal(part, known)

    # ------------------------------------------------------------------------------#

    def test_create_partition_3d_cart(self):
        g = structured.CartGrid([4, 4, 4])
        g.compute_geometry()
        part = co.create_partition(co.tpfa_matrix(g))
        known = (
            np.array(
                [
                    1,
                    1,
                    1,
                    1,
                    2,
                    4,
                    1,
                    3,
                    2,
                    2,
                    3,
                    3,
                    2,
                    2,
                    3,
                    3,
                    5,
                    4,
                    1,
                    6,
                    4,
                    4,
                    4,
                    3,
                    2,
                    4,
                    7,
                    3,
                    8,
                    8,
                    3,
                    3,
                    5,
                    5,
                    6,
                    6,
                    5,
                    4,
                    7,
                    6,
                    8,
                    7,
                    7,
                    7,
                    8,
                    8,
                    7,
                    9,
                    5,
                    5,
                    6,
                    6,
                    5,
                    5,
                    6,
                    6,
                    8,
                    8,
                    7,
                    9,
                    8,
                    8,
                    9,
                    9,
                ]
            )
            - 1
        )
        assert np.array_equal(part, known)

    # ------------------------------------------------------------------------------#

    def test_create_partition_2d_1d_test0(self):
        f = np.array([[1., 1.], [0., 2.]])
        gb = meshing.cart_grid([f], [2, 2])
        gb.compute_geometry()

        part = co.create_partition(co.tpfa_matrix(gb))
        co.generate_coarse_grid(gb, part)

        # Test
        known_indices = np.array([1, 0, 1, 0])
        known = np.array([1, 4, 10, 11])

        for _, d in gb.edges():
            indices, faces, _ = sps.find(d["face_cells"])
            assert np.array_equal(faces, known)
            assert np.array_equal(indices, known_indices)

    # ------------------------------------------------------------------------------#

    def test_create_partition_2d_1d_test1(self):
        f = np.array([[1., 1.], [0., 1.]])
        gb = meshing.cart_grid([f], [2, 2])
        gb.compute_geometry()

        part = co.create_partition(co.tpfa_matrix(gb))
        co.generate_coarse_grid(gb, part)

        # Test
        known_indices = np.array([0, 0])
        known = np.array([1, 9])

        for _, d in gb.edges():
            indices, faces, _ = sps.find(d["face_cells"])
            assert np.array_equal(faces, known)
            assert np.array_equal(indices, known_indices)

    # ------------------------------------------------------------------------------#

    def test_create_partition_2d_1d_test2(self):
        f = np.array([[1., 1.], [0., 1.]])
        gb = meshing.cart_grid([f], [2, 2])
        gb.compute_geometry()

        seeds = co.generate_seeds(gb)
        known_seeds = np.array([0, 1])
        assert np.array_equal(seeds, known_seeds)

        part = co.create_partition(co.tpfa_matrix(gb), seeds=seeds)
        co.generate_coarse_grid(gb, part)

        # Test
        known_indices = np.array([0, 0])
        known = np.array([1, 10])

        for _, d in gb.edges():
            indices, faces, _ = sps.find(d["face_cells"])
            assert np.array_equal(faces, known)
            assert np.array_equal(indices, known_indices)

    # ------------------------------------------------------------------------------#

    def test_create_partition_2d_1d_test3(self):
        f = np.array([[1., 1.], [1., 2.]])
        gb = meshing.cart_grid([f], [2, 2])
        gb.compute_geometry()

        part = co.create_partition(co.tpfa_matrix(gb))
        co.generate_coarse_grid(gb, part)

        # Test
        known_indices = np.array([0, 0])
        known = np.array([3, 9])

        for _, d in gb.edges():
            indices, faces, _ = sps.find(d["face_cells"])
            assert np.array_equal(faces, known)
            assert np.array_equal(indices, known_indices)

    # ------------------------------------------------------------------------------#

    def test_create_partition_2d_1d_test4(self):
        f = np.array([[1., 1.], [1., 2.]])
        gb = meshing.cart_grid([f], [2, 2])
        gb.compute_geometry()

        seeds = co.generate_seeds(gb)
        known_seeds = np.array([2, 3])
        assert np.array_equal(seeds, known_seeds)

        part = co.create_partition(co.tpfa_matrix(gb), seeds=seeds)
        co.generate_coarse_grid(gb, part)

        # Test
        known_indices = np.array([0, 0])
        known = np.array([4, 10])

        for _, d in gb.edges():
            indices, faces, _ = sps.find(d["face_cells"])
            assert np.array_equal(faces, known)
            assert np.array_equal(indices, known_indices)

    # ------------------------------------------------------------------------------#

    def test_create_partition_2d_1d_cross_test5(self):
        # NOTE: Since for python 2.7 and 3.5 the meshes in gridbucket may have
        # non-fixed order, we need to exclude this test.
        if sys.version_info >= (3, 6):
            f1 = np.array([[3., 3.], [1., 5.]])
            f2 = np.array([[1., 5.], [3., 3.]])
            gb = meshing.cart_grid([f1, f2], [6, 6])
            gb.compute_geometry()

            part = co.create_partition(co.tpfa_matrix(gb), cdepth=3)
            co.generate_coarse_grid(gb, part)

            cell_centers_1 = np.array(
                [
                    [3.00000000e+00, 3.00000000e+00, 3.00000000e+00, 3.00000000e+00],
                    [4.50000000e+00, 3.50000000e+00, 2.50000000e+00, 1.50000000e+00],
                    [-1.66533454e-16, -5.55111512e-17, 5.55111512e-17, 1.66533454e-16],
                ]
            )
            cell_centers_2 = np.array(
                [
                    [4.50000000e+00, 3.50000000e+00, 2.50000000e+00, 1.50000000e+00],
                    [3.00000000e+00, 3.00000000e+00, 3.00000000e+00, 3.00000000e+00],
                    [-1.66533454e-16, -5.55111512e-17, 5.55111512e-17, 1.66533454e-16],
                ]
            )

            # Test
            for e_d in gb.edges():
                indices, faces, _ = sps.find(e_d[1]["face_cells"])

                if (e_d[0][0].dim == 0 and e_d[0][1].dim == 1) or (
                    e_d[0][0].dim == 1 and e_d[0][1].dim == 0
                ):
                    known = [2, 5]
                    known_indices = [0, 0]

                if (e_d[0][0].dim == 1 and e_d[0][1].dim == 2) or (
                    e_d[0][0].dim == 2 and e_d[0][1].dim == 1
                ):

                    g = e_d[0][0] if e_d[0][0].dim == 1 else e_d[0][1]

                    if np.allclose(g.cell_centers, cell_centers_1):
                        known = [4, 9, 12, 16, 44, 45, 46, 47]
                        known_indices = [3, 2, 1, 0, 3, 2, 1, 0]
                    elif np.allclose(g.cell_centers, cell_centers_2):
                        known = [31, 32, 33, 34, 48, 49, 50, 51]
                        known_indices = [3, 2, 1, 0, 3, 2, 1, 0]
                    else:
                        raise ValueError("Grid not found")

                assert np.array_equal(faces, np.array(known))
                assert np.array_equal(indices, np.array(known_indices))

    # ------------------------------------------------------------------------------#

    def test_create_partition_2d_1d_cross_test6(self):
        # NOTE: Since for python 2.7 and 3.5 the meshes in gridbucket may have
        # non-fixed order, we need to exclude this test.
        if sys.version_info >= (3, 6):
            f1 = np.array([[3., 3.], [1., 5.]])
            f2 = np.array([[1., 5.], [3., 3.]])
            gb = meshing.cart_grid([f1, f2], [6, 6])
            gb.compute_geometry()

            seeds = co.generate_seeds(gb)
            known_seeds = np.array([8, 9, 26, 27, 13, 16, 19, 22])
            assert np.array_equal(np.sort(seeds), np.sort(known_seeds))

            part = co.create_partition(co.tpfa_matrix(gb), cdepth=3, seeds=seeds)
            co.generate_coarse_grid(gb, part)

            cell_centers_1 = np.array(
                [
                    [3.00000000e+00, 3.00000000e+00, 3.00000000e+00, 3.00000000e+00],
                    [4.50000000e+00, 3.50000000e+00, 2.50000000e+00, 1.50000000e+00],
                    [-1.66533454e-16, -5.55111512e-17, 5.55111512e-17, 1.66533454e-16],
                ]
            )
            cell_centers_2 = np.array(
                [
                    [4.50000000e+00, 3.50000000e+00, 2.50000000e+00, 1.50000000e+00],
                    [3.00000000e+00, 3.00000000e+00, 3.00000000e+00, 3.00000000e+00],
                    [-1.66533454e-16, -5.55111512e-17, 5.55111512e-17, 1.66533454e-16],
                ]
            )

            # Test
            for e_d in gb.edges():
                indices, faces, _ = sps.find(e_d[1]["face_cells"])

                if (e_d[0][0].dim == 0 and e_d[0][1].dim == 1) or (
                    e_d[0][0].dim == 1 and e_d[0][1].dim == 0
                ):
                    known = [2, 5]
                    known_indices = [0, 0]

                if (e_d[0][0].dim == 1 and e_d[0][1].dim == 2) or (
                    e_d[0][0].dim == 2 and e_d[0][1].dim == 1
                ):

                    g = e_d[0][0] if e_d[0][0].dim == 1 else e_d[0][1]

                    if np.allclose(g.cell_centers, cell_centers_1):
                        known = [5, 10, 14, 18, 52, 53, 54, 55]
                        known_indices = [3, 2, 1, 0, 3, 2, 1, 0]
                    elif np.allclose(g.cell_centers, cell_centers_2):
                        known = [37, 38, 39, 40, 56, 57, 58, 59]
                        known_indices = [3, 2, 1, 0, 3, 2, 1, 0]
                    else:
                        raise ValueError("Grid not found")

                assert np.array_equal(faces, np.array(known))
                assert np.array_equal(indices, np.array(known_indices))

    # ------------------------------------------------------------------------------#

    def test_create_partition_2d_1d_cross_test7(self):
        # NOTE: Since for python 2.7 and 3.5 the meshes in gridbucket may have
        # non-fixed order, we need to exclude this test.
        if sys.version_info >= (3, 6):
            N = 20
            f1 = np.array([[N / 2., N / 2.], [1., N - 1.]])
            f2 = np.array([[1., N - 1.], [N / 2., N / 2.]])
            gb = meshing.cart_grid([f1, f2], [N, N])
            gb.compute_geometry()

            seeds = co.generate_seeds(gb)
            known_seeds = np.array([29, 30, 369, 370, 181, 198, 201, 218])
            assert np.array_equal(np.sort(seeds), np.sort(known_seeds))

            part = co.create_partition(co.tpfa_matrix(gb), cdepth=3, seeds=seeds)
            co.generate_coarse_grid(gb, part)

            cell_centers_1 = np.array(
                [
                    [
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                    ],
                    [
                        1.85000000e+01,
                        1.75000000e+01,
                        1.65000000e+01,
                        1.55000000e+01,
                        1.45000000e+01,
                        1.35000000e+01,
                        1.25000000e+01,
                        1.15000000e+01,
                        1.05000000e+01,
                        9.50000000e+00,
                        8.50000000e+00,
                        7.50000000e+00,
                        6.50000000e+00,
                        5.50000000e+00,
                        4.50000000e+00,
                        3.50000000e+00,
                        2.50000000e+00,
                        1.50000000e+00,
                    ],
                    [
                        -9.43689571e-16,
                        -8.32667268e-16,
                        -7.21644966e-16,
                        -6.10622664e-16,
                        -4.99600361e-16,
                        -3.88578059e-16,
                        -2.77555756e-16,
                        -1.66533454e-16,
                        -5.55111512e-17,
                        5.55111512e-17,
                        1.66533454e-16,
                        2.77555756e-16,
                        3.88578059e-16,
                        4.99600361e-16,
                        6.10622664e-16,
                        7.21644966e-16,
                        8.32667268e-16,
                        9.43689571e-16,
                    ],
                ]
            )
            cell_centers_2 = np.array(
                [
                    [
                        1.85000000e+01,
                        1.75000000e+01,
                        1.65000000e+01,
                        1.55000000e+01,
                        1.45000000e+01,
                        1.35000000e+01,
                        1.25000000e+01,
                        1.15000000e+01,
                        1.05000000e+01,
                        9.50000000e+00,
                        8.50000000e+00,
                        7.50000000e+00,
                        6.50000000e+00,
                        5.50000000e+00,
                        4.50000000e+00,
                        3.50000000e+00,
                        2.50000000e+00,
                        1.50000000e+00,
                    ],
                    [
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                        1.00000000e+01,
                    ],
                    [
                        -9.43689571e-16,
                        -8.32667268e-16,
                        -7.21644966e-16,
                        -6.10622664e-16,
                        -4.99600361e-16,
                        -3.88578059e-16,
                        -2.77555756e-16,
                        -1.66533454e-16,
                        -5.55111512e-17,
                        5.55111512e-17,
                        1.66533454e-16,
                        2.77555756e-16,
                        3.88578059e-16,
                        4.99600361e-16,
                        6.10622664e-16,
                        7.21644966e-16,
                        8.32667268e-16,
                        9.43689571e-16,
                    ],
                ]
            )

            # Test
            for e_d in gb.edges():
                indices, faces, _ = sps.find(e_d[1]["face_cells"])

                if (e_d[0][0].dim == 0 and e_d[0][1].dim == 1) or (
                    e_d[0][0].dim == 1 and e_d[0][1].dim == 0
                ):
                    known = [9, 19]
                    known_indices = [0, 0]

                if (e_d[0][0].dim == 1 and e_d[0][1].dim == 2) or (
                    e_d[0][0].dim == 2 and e_d[0][1].dim == 1
                ):

                    g = e_d[0][0] if e_d[0][0].dim == 1 else e_d[0][1]

                    if np.allclose(g.cell_centers, cell_centers_1):
                        known = [
                            10,
                            18,
                            28,
                            37,
                            46,
                            54,
                            62,
                            71,
                            77,
                            84,
                            91,
                            99,
                            108,
                            116,
                            124,
                            134,
                            143,
                            151,
                            328,
                            329,
                            330,
                            331,
                            332,
                            333,
                            334,
                            335,
                            336,
                            337,
                            338,
                            339,
                            340,
                            341,
                            342,
                            343,
                            344,
                            345,
                        ]
                        known_indices = [
                            17,
                            16,
                            15,
                            14,
                            13,
                            12,
                            11,
                            10,
                            9,
                            8,
                            7,
                            6,
                            5,
                            4,
                            3,
                            2,
                            1,
                            0,
                            17,
                            16,
                            15,
                            14,
                            13,
                            12,
                            11,
                            10,
                            9,
                            8,
                            7,
                            6,
                            5,
                            4,
                            3,
                            2,
                            1,
                            0,
                        ]
                    elif np.allclose(g.cell_centers, cell_centers_2):
                        known = [
                            236,
                            237,
                            238,
                            239,
                            240,
                            241,
                            242,
                            243,
                            244,
                            245,
                            246,
                            247,
                            248,
                            249,
                            250,
                            251,
                            252,
                            253,
                            346,
                            347,
                            348,
                            349,
                            350,
                            351,
                            352,
                            353,
                            354,
                            355,
                            356,
                            357,
                            358,
                            359,
                            360,
                            361,
                            362,
                            363,
                        ]
                        known_indices = [
                            17,
                            16,
                            15,
                            14,
                            13,
                            12,
                            11,
                            10,
                            9,
                            8,
                            7,
                            6,
                            5,
                            4,
                            3,
                            2,
                            1,
                            0,
                            17,
                            16,
                            15,
                            14,
                            13,
                            12,
                            11,
                            10,
                            9,
                            8,
                            7,
                            6,
                            5,
                            4,
                            3,
                            2,
                            1,
                            0,
                        ]
                    else:
                        raise ValueError("Grid not found")

                assert np.array_equal(faces, np.array(known))
                assert np.array_equal(indices, np.array(known_indices))


# ------------------------------------------------------------------------------#
