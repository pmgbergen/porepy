""" Tests of tagging of mesh quantities.

The tests have been developed gradually, and are not coherent, but should give reasonable
coverage.
"""
import unittest

import numpy as np

import porepy as pp


def _compare_tip_nodes(g, known_tips):
    # Compare nodes tagged as being on a fracture boundary (seen from the
    # grid in the host medium) and known coordinates of such nodes.

    tip_ind = []

    # For each known coordinate find its closest representation in the grid.
    # This should be more or less identical, or something is wrong
    for i in range(known_tips.shape[1]):
        p = known_tips[:, i].reshape((-1, 1))
        tip_ind.append(np.argmin(np.sum(np.power(p - g.nodes, 2), axis=0)))

    tip_ind = np.array(tip_ind)

    # Check that the nodes with the known coordinates are tagged as tips
    is_tip = g.tags["node_is_fracture_tip"]
    assert np.all(is_tip[tip_ind])

    # Check that all other nodes are tagged as non-tips
    other_ind = np.setdiff1d(np.arange(g.num_nodes), tip_ind)
    assert np.all(np.logical_not(is_tip[other_ind]))


def test_node_is_fracture_tip_2d():
    # Test that nodes in the highest dimensional grids are correctly labeled as tip nodes

    f1 = np.array([[1, 3], [2, 2]])
    f2 = np.array([[1, 3], [3, 3]])
    f3 = np.array([[3, 3], [1, 2]])
    f4 = np.array([[2, 2], [2, 4]])

    # T-intersection between 1 and 4.
    # L-intersection between 1 and 3
    # X-intersection between 2 and 4
    # 2 has two endings in the domain 1 and 3 have one ending in the domain.
    # 4 has one node at the domain boundary - should not be a tip node

    fracs = [f1, f2, f3, f4]
    gb = pp.meshing.cart_grid(fracs, nx=np.array([4, 4]))

    g = gb.grids_of_dimension(2)[0]

    # Base comparison on coordinates (safe on Cartesian grids), then we don't have to deal
    # with changing node indices
    known_tips = np.array([[1, 1, 3, 3], [2, 3, 3, 1], [0, 0, 0, 0]])

    _compare_tip_nodes(g, known_tips)

def test_node_is_fracture_tip_3d():

    dims = np.array([6, 5, 5])

    # f1 is isolated, all tip nodes should be marked in gh
    f1 = np.array([[4, 4, 4, 4], [2, 3, 3, 2], [1, 1, 2, 2]])

    # f2 has several intersections, see below for description
    f2 = np.array([[2, 2, 2, 2], [1, 4, 4, 1], [1, 1, 4, 4]])

    # Nodes 1 and 2 of f3 are tips, 0 and 3 ends in a T-intersection with f2
    f3 = np.array([[2, 3, 3, 2], [2, 2, 2, 2], [2, 2, 3, 3]])

    # f4 has an X-intersection with f2, and L-intersection with f5
    f4 = np.array([[1, 3, 3, 1], [3, 3, 3, 3], [1, 1, 3, 3]])

    # f5 has an L-intersection with f4, but since f5 is taller than f4, the
    # node at (1, 3, 4) is a tip. f5 extends to the domain boundary, so the
    # nodes at z=1 and z=4 are tips, but not z=2, z=3.
    f5 = np.array([[1, 1, 1, 1], [3, 3, 5, 5], [1, 4, 4, 1]])

    gb = pp.meshing.cart_grid([f1, f2, f3, f4, f5], dims)
    g = gb.grids_of_dimension(3)[0]

    # Gather the tip nodes from one fracture at a time
    known_tips_1 = f1
    known_tips_2 = np.array([[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                             [1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 2],
                             [1, 2, 3, 4, 4, 4, 4, 3, 2, 1, 1]])
    known_tips_3 = np.array([[3, 3], [2, 2], [2, 3]])
    known_tips_4 = np.array([[3, 3, 3], [3, 3, 3], [1, 2, 3]])
    known_tips_5 = np.array([[1, 1, 1, 1, 1], [3, 4, 5, 5, 4], [4, 4, 4, 1, 1]])

    known_tips = np.hstack((known_tips_1, known_tips_2, known_tips_3, known_tips_4, known_tips_5))
    _compare_tip_nodes(g, known_tips)


class BasicsTest(unittest.TestCase):
    def test_tag_1d(self):
        g = pp.CartGrid(3, 1)

        self.assertTrue(np.array_equal(g.tags["fracture_faces"], [False] * g.num_faces))
        self.assertTrue(np.array_equal(g.tags["fracture_nodes"], [False] * g.num_nodes))
        self.assertTrue(np.array_equal(g.tags["tip_faces"], [False] * g.num_faces))
        self.assertTrue(np.array_equal(g.tags["tip_nodes"], [False] * g.num_nodes))
        known = [True, False, False, True]
        self.assertTrue(np.array_equal(g.tags["domain_boundary_faces"], known))
        self.assertTrue(np.array_equal(g.tags["domain_boundary_nodes"], known))

    def test_tag_2d_simplex(self):
        g = pp.StructuredTriangleGrid([3] * 2, [1] * 2)

        self.assertTrue(np.array_equal(g.tags["fracture_faces"], [False] * g.num_faces))
        self.assertTrue(np.array_equal(g.tags["fracture_nodes"], [False] * g.num_nodes))
        self.assertTrue(np.array_equal(g.tags["tip_faces"], [False] * g.num_faces))
        self.assertTrue(np.array_equal(g.tags["tip_nodes"], [False] * g.num_nodes))
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
        self.assertTrue(np.array_equal(g.tags["domain_boundary_faces"], known))
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
        self.assertTrue(np.array_equal(g.tags["domain_boundary_nodes"], known))

    def test_tag_2d_cart(self):
        g = pp.CartGrid([4] * 2, [1] * 2)

        self.assertTrue(np.array_equal(g.tags["fracture_faces"], [False] * g.num_faces))
        self.assertTrue(np.array_equal(g.tags["fracture_nodes"], [False] * g.num_nodes))
        self.assertTrue(np.array_equal(g.tags["tip_faces"], [False] * g.num_faces))
        self.assertTrue(np.array_equal(g.tags["tip_nodes"], [False] * g.num_nodes))
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
        self.assertTrue(np.array_equal(g.tags["domain_boundary_faces"], known))
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

        self.assertTrue(np.array_equal(g.tags["domain_boundary_nodes"], known))

    def test_tag_3d_simplex(self):
        g = pp.StructuredTetrahedralGrid([2] * 3, [1] * 3)

        self.assertTrue(np.array_equal(g.tags["fracture_faces"], [False] * g.num_faces))
        self.assertTrue(np.array_equal(g.tags["fracture_nodes"], [False] * g.num_nodes))
        self.assertTrue(np.array_equal(g.tags["tip_faces"], [False] * g.num_faces))
        self.assertTrue(np.array_equal(g.tags["tip_nodes"], [False] * g.num_nodes))
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
        self.assertTrue(np.array_equal(g.tags["domain_boundary_faces"], known))
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
        self.assertTrue(np.array_equal(g.tags["domain_boundary_nodes"], known))

    # ------------------------------------------------------------------------------#

    def test_tag_3d_cart(self):
        g = pp.CartGrid([4] * 3, [1] * 3)

        self.assertTrue(np.array_equal(g.tags["fracture_faces"], [False] * g.num_faces))
        self.assertTrue(np.array_equal(g.tags["fracture_nodes"], [False] * g.num_nodes))
        self.assertTrue(np.array_equal(g.tags["tip_faces"], [False] * g.num_faces))
        self.assertTrue(np.array_equal(g.tags["tip_nodes"], [False] * g.num_nodes))
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
        self.assertTrue(np.array_equal(g.tags["domain_boundary_faces"], known))
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
        self.assertTrue(np.array_equal(g.tags["domain_boundary_nodes"], known))

    def test_tag_2d_1d_cart(self):
        gb, _ = pp.grid_buckets_2d.single_horizontal([4, 4], simplex=False)

        for g, _ in gb:

            if g.dim == 1:
                self.assertTrue(
                    np.array_equal(g.tags["fracture_faces"], [False] * g.num_faces)
                )
                self.assertTrue(
                    np.array_equal(g.tags["fracture_nodes"], [False] * g.num_nodes)
                )
                self.assertTrue(
                    np.array_equal(g.tags["tip_faces"], [False] * g.num_faces)
                )
                self.assertTrue(
                    np.array_equal(g.tags["tip_nodes"], [False] * g.num_nodes)
                )
                known = [0, 4]
                computed = np.where(g.tags["domain_boundary_faces"])[0]
                self.assertTrue(np.array_equal(computed, known))
                known = [0, 4]
                computed = np.where(g.tags["domain_boundary_nodes"])[0]
                self.assertTrue(np.array_equal(computed, known))

            if g.dim == 2:
                known = [28, 29, 30, 31, 40, 41, 42, 43]
                computed = np.where(g.tags["fracture_faces"])[0]
                self.assertTrue(np.array_equal(computed, known))
                known = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
                computed = np.where(g.tags["fracture_nodes"])[0]
                print(computed, known)
                self.assertTrue(np.array_equal(computed, known))
                self.assertTrue(
                    np.array_equal(g.tags["tip_faces"], [False] * g.num_faces)
                )
                self.assertTrue(
                    np.array_equal(g.tags["tip_nodes"], [False] * g.num_nodes)
                )
                known = [0, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 36, 37, 38, 39]
                computed = np.where(g.tags["domain_boundary_faces"])[0]
                self.assertTrue(np.array_equal(computed, known))
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
                self.assertTrue(np.array_equal(computed, known))

    def test_tag_2d_1d_cart_complex(self):
        gb, _ = pp.grid_buckets_2d.two_intersecting(
            [4, 4], y_endpoints=[0.25, 0.75], simplex=False
        )

        for g, _ in gb:

            if g.dim == 0:
                self.assertTrue(np.sum(g.tags["fracture_faces"]) == 0)
                self.assertTrue(np.sum(g.tags["fracture_nodes"]) == 0)
                self.assertTrue(np.sum(g.tags["tip_faces"]) == 0)
                self.assertTrue(np.sum(g.tags["tip_nodes"]) == 0)
                self.assertTrue(np.sum(g.tags["domain_boundary_faces"]) == 0)
                self.assertTrue(np.sum(g.tags["domain_boundary_nodes"]) == 0)

            if g.dim == 1 and g.nodes[1, 0] == 0.5:
                known = [2, 5]
                computed = np.where(g.tags["fracture_faces"])[0]
                self.assertTrue(np.array_equal(known, computed))
                known = [2, 3]
                computed = np.where(g.tags["fracture_nodes"])[0]
                self.assertTrue(np.array_equal(known, computed))
                self.assertTrue(
                    np.array_equal(g.tags["tip_faces"], [False] * g.num_faces)
                )
                self.assertTrue(
                    np.array_equal(g.tags["tip_nodes"], [False] * g.num_nodes)
                )
                known = [0, 4]
                computed = np.where(g.tags["domain_boundary_faces"])[0]
                self.assertTrue(np.array_equal(computed, known))
                known = [0, 5]
                computed = np.where(g.tags["domain_boundary_nodes"])[0]
                self.assertTrue(np.array_equal(computed, known))

            if g.dim == 1 and g.nodes[0, 0] == 0.5:
                known = [1, 3]
                computed = np.where(g.tags["fracture_faces"])[0]
                self.assertTrue(np.array_equal(known, computed))
                known = [1, 2]
                computed = np.where(g.tags["fracture_nodes"])[0]
                self.assertTrue(np.array_equal(known, computed))
                known = [0, 2]
                computed = np.where(g.tags["tip_faces"])[0]
                self.assertTrue(np.array_equal(known, computed))
                known = [0, 3]
                computed = np.where(g.tags["tip_nodes"])[0]
                self.assertTrue(np.array_equal(known, computed))
                self.assertTrue(np.sum(g.tags["domain_boundary_faces"]) == 0)
                self.assertTrue(np.sum(g.tags["domain_boundary_nodes"]) == 0)

            if g.dim == 2:
                known = [7, 12, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45]
                computed = np.where(g.tags["fracture_faces"])[0]
                self.assertTrue(np.array_equal(computed, known))
                known = [7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24]
                computed = np.where(g.tags["fracture_nodes"])[0]
                self.assertTrue(np.array_equal(computed, known))
                self.assertTrue(
                    np.array_equal(g.tags["tip_faces"], [False] * g.num_faces)
                )
                self.assertTrue(
                    np.array_equal(g.tags["tip_nodes"], [False] * g.num_nodes)
                )
                known = [0, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 36, 37, 38, 39]
                computed = np.where(g.tags["domain_boundary_faces"])[0]
                self.assertTrue(np.array_equal(computed, known))
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
                self.assertTrue(np.array_equal(computed, known))


if __name__ == "__main__":
    unittest.main()
