#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 17:25:01 2017

@author: Eirik Keilegavlens
"""
from __future__ import division

import unittest
from test import test_utils

import numpy as np
import scipy.sparse as sps

from porepy.fracs import meshing
from porepy.grids import refinement
from porepy.grids.structured import TensorGrid


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
    class OneCellGrid:
        def __init__(self):
            self.nodes = np.array([[0, 1, 0], [0, 0, 1]])
            self.cell_faces = sps.csc_matrix(np.array([[1], [1], [1]]))
            self.face_nodes = sps.csc_matrix(
                np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
            )
            self.face_centers = np.array([[0.5, 0.5, 0], [0, 0.5, 0.5]])
            self.num_faces = 3
            self.num_cells = 1
            self.num_nodes = 3
            self.dim = 2
            self.name = ["OneGrid"]

    class TwoCellsGrid:
        def __init__(self):
            self.nodes = np.array([[0, 1, 1, 0], [0, 0, 1, 1]])
            self.cell_faces = sps.csc_matrix(
                np.array([[1, 0, 0, 1, 1], [0, 1, 1, 0, 1]]).T
            )
            self.face_nodes = sps.csc_matrix(
                np.array(
                    [
                        [1, 1, 0, 0],
                        [0, 1, 1, 0],
                        [0, 0, 1, 1],
                        [1, 0, 0, 1],
                        [0, 1, 0, 1],
                    ]
                ).T
            )
            self.face_centers = np.array([[0.5, 1, 0.5, 0, 0.5], [0, 0.5, 1, 0.5, 0.5]])
            self.num_faces = 5
            self.num_cells = 2
            self.num_nodes = 4
            self.dim = 2
            self.name = ["TwoGrid"]

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



""" EK: I can no longer recall the intention behind these tests - they seem to
be related to an early implementation of mortar functionality. The tests are
disabled for now, but should be brought back to life at some point.

class TestRefinementGridBucket(unittest.TestCase):
    def test_refinement_grid_1d_in_gb_uniform_ratio_2(self):
        f1 = np.array([[0, 1], [.5, .5]])
        ratio = 2

        gb = meshing.cart_grid([f1], [2, 2], **{'physdims': [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        gs_1d = np.array(gb.grids_of_dimension(1))
        hs_1d = np.array([refinement.remesh_1d(g, ratio) for g in gs_1d])

        gb.update_nodes(gs_1d, hs_1d)

        known_face_cells = \
                np.array([[ 0.  ,  0.,  0.,  0.  ,  0.  ,  0.,  0.,  0.,  0.  ,
                            0.5,  0.,  0.,  0.  ,  0.5],
                          [ 0.  ,  0.,  0.,  0.  ,  0.  ,  0.,  0.,  0.,  0.  ,
                            0.5,  0.,  0.,  0.  ,  0.5],
                          [ 0.  ,  0.,  0.,  0.  ,  0.  ,  0.,  0.,  0.,  0.5,
                            0.  ,  0.,  0.,  0.5,  0.  ],
                          [ 0.  ,  0.,  0.,  0.  ,  0.  ,  0.,  0.,  0.,  0.5,
                            0.  ,  0.,  0.,  0.5,  0.  ]])

        for _, d in gb.edges():
            self.assertTrue(np.allclose(d["face_cells"].todense(), known_face_cells))

#------------------------------------------------------------------------------#

    def test_refinement_grid_1d_in_gb_uniform_ratio_3(self):
        f1 = np.array([[0, 1], [.5, .5]])
        ratio = 3

        gb = meshing.cart_grid([f1], [2, 2], **{'physdims': [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        gs_1d = np.array(gb.grids_of_dimension(1))
        hs_1d = np.array([refinement.remesh_1d(g, ratio) for g in gs_1d])

        gb.update_nodes(gs_1d, hs_1d)

        known_face_cells = \
                    np.array([[ 0.,  0.,  0.        ,  0.        ,  0.        ,
                                0.,  0.,  0.        ,  0.        ,  0.33333333,
                                0.,  0.,  0.        ,  0.33333333],
                              [ 0.,  0.,  0.        ,  0.        ,  0.        ,
                                0.,  0.,  0.        ,  0.        ,  0.33333333,
                                0.,  0.,  0.        ,  0.33333333],
                              [ 0.,  0.,  0.        ,  0.        ,  0.        ,
                                0.,  0.,  0.        ,  0.        ,  0.33333333,
                                0.,  0.,  0.        ,  0.33333333],
                              [ 0.,  0.,  0.        ,  0.        ,  0.        ,
                                0.,  0.,  0.        ,  0.33333333,  0.        ,
                                0.,  0.,  0.33333333,  0.        ],
                              [ 0.,  0.,  0.        ,  0.        ,  0.        ,
                                0.,  0.,  0.        ,  0.33333333,  0.        ,
                                0.,  0.,  0.33333333,  0.        ],
                              [ 0.,  0.,  0.        ,  0.        ,  0.        ,
                                0.,  0.,  0.        ,  0.33333333,  0.        ,
                                0.,  0.,  0.33333333,  0.        ]])

        for _, d in gb.edges():
            self.assertTrue(np.allclose(d["face_cells"].todense(), known_face_cells))

#------------------------------------------------------------------------------#

    def test_refinement_grid_1d_in_gb(self):
        f1 = np.array([[0, 1], [.5, .5]])
        num_nodes = 4

        gb = meshing.cart_grid([f1], [2, 2], **{'physdims': [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        gs_1d = np.array(gb.grids_of_dimension(1))
        hs_1d = np.array([refinement.remesh_1d(g, num_nodes) for g in gs_1d])

        gb.update_nodes(gs_1d, hs_1d)

        known_face_cells = \
                    np.array([[ 0.,  0.,  0.        ,  0.        ,  0.        ,
                                0.,  0.,  0.        ,  0.66666667,  0.        ,
                                0.,  0.,  0.66666667,  0.        ],
                              [ 0.,  0.,  0.        ,  0.        ,  0.        ,
                                0.,  0.,  0.        ,  0.33333333,  0.33333333,
                                0.,  0.,  0.33333333,  0.33333333],
                              [ 0.,  0.,  0.        ,  0.        ,  0.        ,
                                0.,  0.,  0.        ,  0.        ,  0.66666667,
                                0.,  0.,  0.        ,  0.66666667]])

        for _, d in gb.edges():
            self.assertTrue(np.allclose(d["face_cells"].todense(), known_face_cells))

#------------------------------------------------------------------------------#

    def test_coarse_grid_1d_in_gb(self):
        f1 = np.array([[0, 1], [.5, .5]])
        num_nodes = 4

        gb = meshing.cart_grid([f1], [4, 2], **{'physdims': [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        gs_1d = np.array(gb.grids_of_dimension(1))
        hs_1d = np.array([refinement.remesh_1d(g, num_nodes) for g in gs_1d])

        gb.update_nodes(gs_1d, hs_1d)

        known_face_cells = np.array([\
               [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                 0.        ,  0.        ,  0.        ,  0.        ,  1.        ,
                 0.33333333,  0.        ,  0.        ,  0.        ,  0.        ,
                 0.        ,  0.        ,  1.        ,  0.33333333,  0.        ,
                 0.        ],
               [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                 0.66666667,  0.66666667,  0.        ,  0.        ,  0.        ,
                 0.        ,  0.        ,  0.        ,  0.66666667,  0.66666667,
                 0.        ],
               [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                 0.        ,  0.33333333,  1.        ,  0.        ,  0.        ,
                 0.        ,  0.        ,  0.        ,  0.        ,  0.33333333,
                 1.        ]])

        for _, d in gb.edges():
            self.assertTrue(np.allclose(d["face_cells"].todense(), known_face_cells))
"""



class TestRefinementMortarGrid(unittest.TestCase):
    def test_mortar_grid_1d(self):

        f1 = np.array([[0, 1], [0.5, 0.5]])

        gb = meshing.cart_grid([f1], [2, 2], **{"physdims": [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        for e, d in gb.edges():

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

            mg = d["mortar_grid"]
            self.assertTrue(
                np.allclose(high_to_mortar_known, mg.primary_to_mortar_int().todense())
            )
            self.assertTrue(
                np.allclose(low_to_mortar_known, mg.secondary_to_mortar_int().todense())
            )

    def test_mortar_grid_1d_equally_refine_mortar_grids(self):

        f1 = np.array([[0, 1], [0.5, 0.5]])

        gb = meshing.cart_grid([f1], [2, 2], **{"physdims": [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        for e, d in gb.edges():

            mg = d["mortar_grid"]
            new_side_grids = {
                s: refinement.remesh_1d(g, num_nodes=4)
                for s, g in mg.side_grids.items()
            }

            mg.update_mortar(new_side_grids, 1e-4)

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
                np.allclose(high_to_mortar_known, mg.primary_to_mortar_int().todense())
            )
            self.assertTrue(
                np.allclose(low_to_mortar_known, mg.secondary_to_mortar_int().todense())
            )

    # ------------------------------------------------------------------------------#

    def test_mortar_grid_1d_unequally_refine_mortar_grids(self):

        f1 = np.array([[0, 1], [0.5, 0.5]])

        gb = meshing.cart_grid([f1], [2, 2], **{"physdims": [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        for e, d in gb.edges():

            mg = d["mortar_grid"]
            new_side_grids = {
                s: refinement.remesh_1d(g, num_nodes=s.value + 3)
                for s, g in mg.side_grids.items()
            }

            mg.update_mortar(new_side_grids, 1e-4)

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
                np.allclose(high_to_mortar_known, mg.primary_to_mortar_int().todense())
            )
            self.assertTrue(
                np.allclose(low_to_mortar_known, mg.secondary_to_mortar_int().todense())
            )

    # ------------------------------------------------------------------------------#

    def test_mortar_grid_1d_refine_1d_grid(self):
        """Refine the lower-dimensional grid so that it is matching with the
        higher dimensional grid.
        """

        f1 = np.array([[0, 1], [0.5, 0.5]])

        gb = meshing.cart_grid([f1], [2, 2], **{"physdims": [1, 1]})
        gb.compute_geometry()
        meshing.create_mortar_grids(gb)
        gb.assign_node_ordering()

        for e, d in gb.edges():

            # refine the 1d-physical grid
            old_g = gb.nodes_of_edge(e)[0]
            new_g = refinement.remesh_1d(old_g, num_nodes=5)
            new_g.compute_geometry()

            gb.update_nodes({old_g: new_g})
            mg = d["mortar_grid"]
            mg.update_secondary(new_g, 1e-4)

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
                [
                    [0.0, 0.0, 0.5, 0.5],
                    [0.5, 0.5, 0.0, 0.0],
                    [0.0, 0.0, 0.5, 0.5],
                    [0.5, 0.5, 0.0, 0.0],
                ]
            )

            self.assertTrue(
                np.allclose(high_to_mortar_known, mg.primary_to_mortar_int().todense())
            )

            # The ordering of the cells in the new 1d grid may be flipped on
            # some systems; therefore allow two configurations
            self.assertTrue(
                np.logical_or(
                    np.allclose(
                        low_to_mortar_known, mg.secondary_to_mortar_int().todense()
                    ),
                    np.allclose(
                        low_to_mortar_known,
                        mg.secondary_to_mortar_int().todense()[::-1],
                    ),
                )
            )

    # ------------------------------------------------------------------------------#

    def test_mortar_grid_1d_refine_1d_grid_2(self):
        """Refine the 1D grid so that it is no longer matching the 2D grid."""

        f1 = np.array([[0, 1], [0.5, 0.5]])

        gb = meshing.cart_grid([f1], [2, 2], **{"physdims": [1, 1]})
        gb.compute_geometry()
        meshing.create_mortar_grids(gb)
        gb.assign_node_ordering()

        for e, d in gb.edges():

            # refine the 1d-physical grid
            old_g = gb.nodes_of_edge(e)[0]
            new_g = refinement.remesh_1d(old_g, num_nodes=4)
            new_g.compute_geometry()

            gb.update_nodes({old_g: new_g})
            mg = d["mortar_grid"]
            mg.update_secondary(new_g, 1e-4)

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
            low_to_mortar_known = (
                1.0
                / 3.0
                * np.matrix(
                    [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]
                )
            )

            self.assertTrue(
                np.allclose(high_to_mortar_known, mg.primary_to_mortar_int().todense())
            )
            # The ordering of the cells in the new 1d grid may be flipped on
            # some systems; therefore allow two configurations
            self.assertTrue(
                np.logical_or(
                    np.allclose(
                        low_to_mortar_known, mg.secondary_to_mortar_int().todense()
                    ),
                    np.allclose(
                        low_to_mortar_known,
                        mg.secondary_to_mortar_int().todense()[::-1],
                    ),
                )
            )

    # ------------------------------------------------------------------------------#

    def test_mortar_grid_2d(self):

        f = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]])
        gb = meshing.cart_grid([f], [2] * 3, **{"physdims": [1] * 3})
        gb.compute_geometry()
        meshing.create_mortar_grids(gb)

        gb.assign_node_ordering()

        for e, d in gb.edges():

            mg = d["mortar_grid"]
            indices_known = np.array([0, 1, 2, 3, 4, 5, 6, 7])
            self.assertTrue(
                np.array_equal(mg.primary_to_mortar_int().indices, indices_known)
            )

            indptr_known = np.array(
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    2,
                    3,
                    4,
                    4,
                    4,
                    4,
                    4,
                    5,
                    6,
                    7,
                    8,
                ]
            )
            self.assertTrue(
                np.array_equal(mg.primary_to_mortar_int().indptr, indptr_known)
            )

            data_known = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            self.assertTrue(np.array_equal(mg.primary_to_mortar_int().data, data_known))

            indices_known = np.array([0, 4, 1, 5, 2, 6, 3, 7])
            self.assertTrue(
                np.array_equal(mg.secondary_to_mortar_int().indices, indices_known)
            )

            indptr_known = np.array([0, 2, 4, 6, 8])
            self.assertTrue(
                np.array_equal(mg.secondary_to_mortar_int().indptr, indptr_known)
            )

            data_known = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            self.assertTrue(
                np.array_equal(mg.secondary_to_mortar_int().data, data_known)
            )


if __name__ == "__main__":
    unittest.main()
