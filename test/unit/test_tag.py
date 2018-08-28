import numpy as np
import unittest

from porepy.grids import structured, simplex
from porepy.fracs import meshing

# ------------------------------------------------------------------------------#


class BasicsTest(unittest.TestCase):

    # ------------------------------------------------------------------------------#

    def test_tag_1d(self):
        g = structured.CartGrid(3, 1)

        assert np.array_equal(g.tags["fracture_faces"], [False] * g.num_faces)
        assert np.array_equal(g.tags["fracture_nodes"], [False] * g.num_nodes)
        assert np.array_equal(g.tags["tip_faces"], [False] * g.num_faces)
        assert np.array_equal(g.tags["tip_nodes"], [False] * g.num_nodes)
        known = [True, False, False, True]
        assert np.array_equal(g.tags["domain_boundary_faces"], known)
        assert np.array_equal(g.tags["domain_boundary_nodes"], known)

    # ------------------------------------------------------------------------------#

    def test_tag_2d_simplex(self):
        g = simplex.StructuredTriangleGrid([3] * 2, [1] * 2)

        assert np.array_equal(g.tags["fracture_faces"], [False] * g.num_faces)
        assert np.array_equal(g.tags["fracture_nodes"], [False] * g.num_nodes)
        assert np.array_equal(g.tags["tip_faces"], [False] * g.num_faces)
        assert np.array_equal(g.tags["tip_nodes"], [False] * g.num_nodes)
        known = np.array(
            [
                True,
                True,
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
            ],
            dtype=bool,
        )
        assert np.array_equal(g.tags["domain_boundary_faces"], known)
        known = np.array(
            [
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                True,
                True,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
            ],
            dtype=bool,
        )
        assert np.array_equal(g.tags["domain_boundary_nodes"], known)

    # ------------------------------------------------------------------------------#

    def test_tag_2d_cart(self):
        g = structured.CartGrid([4] * 2, [1] * 2)

        assert np.array_equal(g.tags["fracture_faces"], [False] * g.num_faces)
        assert np.array_equal(g.tags["fracture_nodes"], [False] * g.num_nodes)
        assert np.array_equal(g.tags["tip_faces"], [False] * g.num_faces)
        assert np.array_equal(g.tags["tip_nodes"], [False] * g.num_nodes)
        known = np.array(
            [
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
            ],
            dtype=bool,
        )
        assert np.array_equal(g.tags["domain_boundary_faces"], known)
        known = np.array(
            [
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            dtype=bool,
        )

        assert np.array_equal(g.tags["domain_boundary_nodes"], known)

    # ------------------------------------------------------------------------------#

    def test_tag_3d_simplex(self):
        g = simplex.StructuredTetrahedralGrid([2] * 3, [1] * 3)

        assert np.array_equal(g.tags["fracture_faces"], [False] * g.num_faces)
        assert np.array_equal(g.tags["fracture_nodes"], [False] * g.num_nodes)
        assert np.array_equal(g.tags["tip_faces"], [False] * g.num_faces)
        assert np.array_equal(g.tags["tip_nodes"], [False] * g.num_nodes)
        known = np.array(
            [
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                True,
                False,
                False,
                True,
                False,
                True,
                True,
                True,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                True,
                True,
                False,
                False,
                True,
                False,
                True,
                True,
                False,
                True,
                True,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                True,
                False,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                True,
                True,
                False,
                False,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            dtype=bool,
        )
        assert np.array_equal(g.tags["domain_boundary_faces"], known)
        known = np.array(
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            dtype=bool,
        )
        assert np.array_equal(g.tags["domain_boundary_nodes"], known)

    # ------------------------------------------------------------------------------#

    def test_tag_3d_cart(self):
        g = structured.CartGrid([4] * 3, [1] * 3)

        assert np.array_equal(g.tags["fracture_faces"], [False] * g.num_faces)
        assert np.array_equal(g.tags["fracture_nodes"], [False] * g.num_nodes)
        assert np.array_equal(g.tags["tip_faces"], [False] * g.num_faces)
        assert np.array_equal(g.tags["tip_nodes"], [False] * g.num_nodes)
        known = np.array(
            [
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            dtype=bool,
        )
        assert np.array_equal(g.tags["domain_boundary_faces"], known)
        known = np.array(
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            dtype=bool,
        )
        assert np.array_equal(g.tags["domain_boundary_nodes"], known)

    # ------------------------------------------------------------------------------#

    def test_tag_2d_1d_cart(self):
        f1 = np.array([[0, 1], [.5, .5]])
        gb = meshing.cart_grid([f1], [4, 4], **{"physdims": [1, 1]})

        for g, _ in gb:

            if g.dim == 1:
                assert np.array_equal(g.tags["fracture_faces"], [False] * g.num_faces)
                assert np.array_equal(g.tags["fracture_nodes"], [False] * g.num_nodes)
                assert np.array_equal(g.tags["tip_faces"], [False] * g.num_faces)
                assert np.array_equal(g.tags["tip_nodes"], [False] * g.num_nodes)
                known = [0, 4]
                computed = np.where(g.tags["domain_boundary_faces"])[0]
                assert np.array_equal(computed, known)
                known = [0, 4]
                computed = np.where(g.tags["domain_boundary_nodes"])[0]
                assert np.array_equal(computed, known)

            if g.dim == 2:
                known = [28, 29, 30, 31, 40, 41, 42, 43]
                computed = np.where(g.tags["fracture_faces"])[0]
                assert np.array_equal(computed, known)
                known = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
                computed = np.where(g.tags["fracture_nodes"])[0]
                print(computed, known)
                assert np.array_equal(computed, known)
                assert np.array_equal(g.tags["tip_faces"], [False] * g.num_faces)
                assert np.array_equal(g.tags["tip_nodes"], [False] * g.num_nodes)
                known = [0, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 36, 37, 38, 39]
                computed = np.where(g.tags["domain_boundary_faces"])[0]
                assert np.array_equal(computed, known)
                known = [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    9,
                    10,
                    11,
                    18,
                    19,
                    20,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                ]
                computed = np.where(g.tags["domain_boundary_nodes"])[0]
                assert np.array_equal(computed, known)

    # ------------------------------------------------------------------------------#

    def test_tag_2d_1d_cart_complex(self):
        f1 = np.array([[0, 1], [.5, .5]])
        f2 = np.array([[.5, .5], [.25, .75]])
        gb = meshing.cart_grid([f1, f2], [4, 4], **{"physdims": [1, 1]})

        for g, _ in gb:

            if g.dim == 0:
                assert np.sum(g.tags["fracture_faces"]) == 0
                assert np.sum(g.tags["fracture_nodes"]) == 0
                assert np.sum(g.tags["tip_faces"]) == 0
                assert np.sum(g.tags["tip_nodes"]) == 0
                assert np.sum(g.tags["domain_boundary_faces"]) == 0
                assert np.sum(g.tags["domain_boundary_nodes"]) == 0

            if g.dim == 1 and g.nodes[1, 0] == 0.5:
                known = [2, 5]
                computed = np.where(g.tags["fracture_faces"])[0]
                assert np.array_equal(known, computed)
                known = [2, 3]
                computed = np.where(g.tags["fracture_nodes"])[0]
                assert np.array_equal(known, computed)
                assert np.array_equal(g.tags["tip_faces"], [False] * g.num_faces)
                assert np.array_equal(g.tags["tip_nodes"], [False] * g.num_nodes)
                known = [0, 4]
                computed = np.where(g.tags["domain_boundary_faces"])[0]
                assert np.array_equal(computed, known)
                known = [0, 5]
                computed = np.where(g.tags["domain_boundary_nodes"])[0]
                assert np.array_equal(computed, known)

            if g.dim == 1 and g.nodes[0, 0] == 0.5:
                known = [1, 3]
                computed = np.where(g.tags["fracture_faces"])[0]
                assert np.array_equal(known, computed)
                known = [1, 2]
                computed = np.where(g.tags["fracture_nodes"])[0]
                assert np.array_equal(known, computed)
                known = [0, 2]
                computed = np.where(g.tags["tip_faces"])[0]
                assert np.array_equal(known, computed)
                known = [0, 3]
                computed = np.where(g.tags["tip_nodes"])[0]
                assert np.array_equal(known, computed)
                assert np.sum(g.tags["domain_boundary_faces"]) == 0
                assert np.sum(g.tags["domain_boundary_nodes"]) == 0

            if g.dim == 2:
                known = [7, 12, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45]
                computed = np.where(g.tags["fracture_faces"])[0]
                assert np.array_equal(computed, known)
                known = [7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24]
                computed = np.where(g.tags["fracture_nodes"])[0]
                assert np.array_equal(computed, known)
                assert np.array_equal(g.tags["tip_faces"], [False] * g.num_faces)
                assert np.array_equal(g.tags["tip_nodes"], [False] * g.num_nodes)
                known = [0, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 36, 37, 38, 39]
                computed = np.where(g.tags["domain_boundary_faces"])[0]
                assert np.array_equal(computed, known)
                known = [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    9,
                    10,
                    11,
                    20,
                    21,
                    22,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                ]
                computed = np.where(g.tags["domain_boundary_nodes"])[0]
                assert np.array_equal(computed, known)


# ------------------------------------------------------------------------------#

BasicsTest().test_tag_2d_1d_cart_complex()
