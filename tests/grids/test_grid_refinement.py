"""
Created on Sat Nov 11 17:25:01 2017

@author: Eirik Keilegavlens
"""
from __future__ import division

import unittest

import numpy as np
import scipy.sparse as sps

from porepy.fracs import meshing
from porepy.grids import refinement
from porepy.grids.simplex import TriangleGrid
from porepy.grids.structured import TensorGrid
from tests import test_utils


class TestGridPerturbation(unittest.TestCase):
    def test_grid_perturbation_1d_bound_nodes_fixed(self):
        g = TensorGrid(np.array([0, 1, 2]))
        h = refinement.distort_grid_1d(g)
        self.assertTrue(np.allclose(g.nodes[:, [0, 2]], h.nodes[:, [0, 2]]))

    def test_grid_perturbation_1d_internal_and_bound_nodes_fixed(self):
        g = TensorGrid(np.arange(4))
        h = refinement.distort_grid_1d(g, fixed_nodes=[0, 1, 3])
        self.assertTrue(np.allclose(g.nodes[:, [0, 1, 3]], h.nodes[:, [0, 1, 3]]))

    def test_grid_perturbation_1d_internal_nodes_fixed(self):
        g = TensorGrid(np.arange(4))
        h = refinement.distort_grid_1d(g, fixed_nodes=[1])
        self.assertTrue(np.allclose(g.nodes[:, [0, 1, 3]], h.nodes[:, [0, 1, 3]]))


class TestGridRefinement1d(unittest.TestCase):
    def test_refinement_grid_1d_uniform(self):
        x = np.array([0, 2, 4])
        g = TensorGrid(x)
        h = refinement.refine_grid_1d(g, ratio=2)
        self.assertTrue(np.allclose(h.nodes[0], np.arange(5)))

    def test_refinement_grid_1d_non_uniform(self):
        x = np.array([0, 2, 6])
        g = TensorGrid(x)
        h = refinement.refine_grid_1d(g, ratio=2)
        self.assertTrue(np.allclose(h.nodes[0], np.array([0, 1, 2, 4, 6])))

    def test_refinement_grid_1d_general_orientation(self):
        x = np.array([0, 2, 6]) * np.ones((3, 1))
        g = TensorGrid(x[0])
        g.nodes = x
        h = refinement.refine_grid_1d(g, ratio=2)
        self.assertTrue(
            np.allclose(
                h.nodes, np.array([[0, 1, 2, 4, 6], [0, 1, 2, 4, 6], [0, 1, 2, 4, 6]])
            )
        )


class TestGridRefinement2dSimplex(unittest.TestCase):
    class OneCellGrid(TriangleGrid):
        def __init__(self):
            pts = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
            tri = np.array([[0], [1], [2]])

            super().__init__(pts, tri, "GridWithOneCell")
            self.history = ["OneGrid"]

    class TwoCellsGrid(TriangleGrid):
        def __init__(self):
            pts = np.array([[0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]])
            tri = np.array([[0, 1], [1, 3], [2, 2]])

            super().__init__(pts, tri, "GridWithTwoCells")
            self.history = ["TwoGrid"]

    def test_refinement_single_cell(self):
        g = self.OneCellGrid()
        h, parent = refinement.refine_triangle_grid(g)
        h.compute_geometry()

        self.assertTrue(h.num_cells == 4)
        self.assertTrue(h.num_faces == 9)
        self.assertTrue(h.num_nodes == 6)
        self.assertTrue(h.cell_volumes.sum() == 0.5)
        self.assertTrue(np.allclose(h.cell_volumes, 1 / 8))

        known_nodes = np.array([[0, 0.5, 1, 0.5, 0, 0], [0, 0, 0, 0.5, 1, 0.5]])
        test_utils.compare_arrays(h.nodes[:2], known_nodes)
        self.assertTrue(np.all(parent == 0))

    def test_refinement_two_cells(self):
        g = self.TwoCellsGrid()
        h, parent = refinement.refine_triangle_grid(g)
        h.compute_geometry()

        self.assertTrue(h.num_cells == 8)
        self.assertTrue(h.num_faces == 16)
        self.assertTrue(h.num_nodes == 9)
        self.assertTrue(h.cell_volumes.sum() == 1)
        self.assertTrue(np.allclose(h.cell_volumes, 1 / 8))

        known_nodes = np.array(
            [[0, 0.5, 1, 0.5, 0, 0, 1, 1, 0.5], [0, 0, 0, 0.5, 1, 0.5, 0.5, 1, 1]]
        )
        test_utils.compare_arrays(h.nodes[:2], known_nodes)
        self.assertTrue(np.sum(parent == 0) == 4)
        self.assertTrue(np.sum(parent == 1) == 4)
        self.assertTrue(np.allclose(np.bincount(parent, h.cell_volumes), 0.5))


class TestRefinementMortarGrid(unittest.TestCase):
    def test_mortar_grid_1d(self):
        f1 = np.array([[0, 1], [0.5, 0.5]])

        mdg = meshing.cart_grid([f1], [2, 2], **{"physdims": [1, 1]})

        for intf in mdg.interfaces():
            high_to_mortar_known = np.matrix(
                [
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                    ],
                ]
            )
            low_to_mortar_known = np.matrix(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
            )

            self.assertTrue(
                np.allclose(
                    high_to_mortar_known, intf.primary_to_mortar_int().todense()
                )
            )
            self.assertTrue(
                np.allclose(
                    low_to_mortar_known, intf.secondary_to_mortar_int().todense()
                )
            )

    def test_mortar_grid_1d_equally_refine_mortar_grids(self):
        f1 = np.array([[0, 1], [0.5, 0.5]])

        mdg = meshing.cart_grid([f1], [2, 2], **{"physdims": [1, 1]})
        mdg.compute_geometry()

        for intf, data in mdg.interfaces(return_data=True):
            new_side_grids = {
                s: refinement.remesh_1d(g, num_nodes=4)
                for s, g in intf.side_grids.items()
            }

            intf.update_mortar(new_side_grids, 1e-4)

            high_to_mortar_known = (
                1.0
                / 3.0
                * np.matrix(
                    [
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            2.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            2.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            2.0,
                            0.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            1.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            2.0,
                        ],
                    ]
                )
            )

            low_to_mortar_known = (
                1.0
                / 3.0
                * np.matrix(
                    [
                        [0.0, 2.0],
                        [1.0, 1.0],
                        [2.0, 0.0],
                        [0.0, 2.0],
                        [1.0, 1.0],
                        [2.0, 0.0],
                    ]
                )
            )

            self.assertTrue(
                np.allclose(
                    high_to_mortar_known, intf.primary_to_mortar_int().todense()
                )
            )
            self.assertTrue(
                np.allclose(
                    low_to_mortar_known, intf.secondary_to_mortar_int().todense()
                )
            )

    # ------------------------------------------------------------------------------#

    def test_mortar_grid_1d_unequally_refine_mortar_grids(self):
        f1 = np.array([[0, 1], [0.5, 0.5]])

        mdg = meshing.cart_grid([f1], [2, 2], **{"physdims": [1, 1]})

        for intf, data in mdg.interfaces(return_data=True):
            new_side_grids = {
                s: refinement.remesh_1d(g, num_nodes=s.value + 3)
                for s, g in intf.side_grids.items()
            }

            intf.update_mortar(new_side_grids, 1e-4)

            high_to_mortar_known = np.matrix(
                [
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.66666667,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.33333333,
                        0.33333333,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.66666667,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.5,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.5,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.5,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.5,
                    ],
                ]
            )
            low_to_mortar_known = np.matrix(
                [
                    [0.0, 0.66666667],
                    [0.33333333, 0.33333333],
                    [0.66666667, 0.0],
                    [0.0, 0.5],
                    [0.0, 0.5],
                    [0.5, 0.0],
                    [0.5, 0.0],
                ]
            )
            self.assertTrue(
                np.allclose(
                    high_to_mortar_known, intf.primary_to_mortar_int().todense()
                )
            )
            self.assertTrue(
                np.allclose(
                    low_to_mortar_known, intf.secondary_to_mortar_int().todense()
                )
            )

    # ------------------------------------------------------------------------------#

    def test_mortar_grid_1d_refine_1d_grid(self):
        """Refine the lower-dimensional grid so that it is matching with the
        higher dimensional grid.
        """

        f1 = np.array([[0, 1], [0.5, 0.5]])

        mdg = meshing.cart_grid([f1], [2, 2], **{"physdims": [1, 1]})

        for intf in mdg.interfaces():
            # refine the 1d-physical grid
            old_g = mdg.interface_to_subdomain_pair(intf)[1]
            new_g = refinement.remesh_1d(old_g, num_nodes=5)
            new_g.compute_geometry()

            mdg.replace_subdomains_and_interfaces(sd_map={old_g: new_g})
            intf.update_secondary(new_g, 1e-4)

            high_to_mortar_known = np.matrix(
                [
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                    ],
                ]
            )
            low_to_mortar_known_avg = np.matrix(
                [
                    [0.0, 0.0, 0.5, 0.5],
                    [0.5, 0.5, 0.0, 0.0],
                    [0.0, 0.0, 0.5, 0.5],
                    [0.5, 0.5, 0.0, 0.0],
                ]
            )

            self.assertTrue(
                np.allclose(
                    high_to_mortar_known, intf.primary_to_mortar_int().todense()
                )
            )

            # The ordering of the cells in the new 1d grid may be flipped on
            # some systems; therefore allow two configurations
            self.assertTrue(
                np.logical_or(
                    np.allclose(
                        low_to_mortar_known_avg, intf.secondary_to_mortar_avg().A
                    ),
                    np.allclose(
                        low_to_mortar_known_avg,
                        intf.secondary_to_mortar_avg().A[::-1],
                    ),
                )
            )

    # ------------------------------------------------------------------------------#

    def test_mortar_grid_1d_refine_1d_grid_2(self):
        """Refine the 1D grid so that it no longer matches the 2D grid."""

        f1 = np.array([[0, 1], [0.5, 0.5]])

        mdg = meshing.cart_grid([f1], [2, 2], **{"physdims": [1, 1]})

        for intf in mdg.interfaces():
            # refine the 1d-physical grid
            old_g = mdg.interface_to_subdomain_pair(intf)[1]
            new_g = refinement.remesh_1d(old_g, num_nodes=4)
            new_g.compute_geometry()

            mdg.replace_subdomains_and_interfaces(sd_map={old_g: new_g})
            intf.update_secondary(new_g, 1e-4)

            high_to_mortar_known = np.matrix(
                [
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                    ],
                ]
            )
            low_to_mortar_known_avg = (
                1.0
                / 3.0
                * np.matrix(
                    [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]
                )
            )
            low_to_mortar_known_int = np.matrix(
                [[0.0, 0.5, 1.0], [1.0, 0.5, 0.0], [0.0, 0.5, 1.0], [1.0, 0.5, 0.0]]
            )

            self.assertTrue(
                np.allclose(
                    high_to_mortar_known, intf.primary_to_mortar_int().todense()
                )
            )
            # The ordering of the cells in the new 1d grid may be flipped on
            # some systems; therefore allow two configurations
            self.assertTrue(
                np.logical_or(
                    np.allclose(
                        low_to_mortar_known_avg, intf.secondary_to_mortar_avg().A
                    ),
                    np.allclose(
                        low_to_mortar_known_avg,
                        intf.secondary_to_mortar_avg().A[::-1],
                    ),
                )
            )
            self.assertTrue(
                np.logical_or(
                    np.allclose(
                        low_to_mortar_known_int, intf.secondary_to_mortar_int().A
                    ),
                    np.allclose(
                        low_to_mortar_known_int,
                        intf.secondary_to_mortar_int().A[::-1],
                    ),
                )
            )

    # ------------------------------------------------------------------------------#

    def test_mortar_grid_2d(self):
        f = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]])
        mdg = meshing.cart_grid([f], [2] * 3, **{"physdims": [1] * 3})
        mdg.compute_geometry()
        # meshing.create_mortar_grids(mdg)

        # mdg.assign_node_ordering()

        for intf, data in mdg.interfaces(return_data=True):
            indices_known = np.array([28, 29, 30, 31, 36, 37, 38, 39])
            self.assertTrue(
                np.array_equal(intf.primary_to_mortar_int().indices, indices_known)
            )

            indptr_known = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
            self.assertTrue(
                np.array_equal(intf.primary_to_mortar_int().indptr, indptr_known)
            )

            data_known = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            self.assertTrue(
                np.array_equal(intf.primary_to_mortar_int().data, data_known)
            )

            indices_known = np.array([0, 4, 1, 5, 2, 6, 3, 7])
            self.assertTrue(
                np.array_equal(intf.secondary_to_mortar_int().indices, indices_known)
            )

            indptr_known = np.array([0, 2, 4, 6, 8])
            self.assertTrue(
                np.array_equal(intf.secondary_to_mortar_int().indptr, indptr_known)
            )

            data_known = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            self.assertTrue(
                np.array_equal(intf.secondary_to_mortar_int().data, data_known)
            )


if __name__ == "__main__":
    unittest.main()
