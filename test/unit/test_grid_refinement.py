#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 17:25:01 2017

@author: Eirik Keilegavlens
"""
import numpy as np
import unittest

from porepy.grids.structured import TensorGrid
from porepy.grids import refinement
from porepy.fracs import meshing, mortars

class TestGridRefinement1d(unittest.TestCase):

    def test_refinement_grid_1d_uniform(self):
        x = np.array([0, 2, 4])
        g = TensorGrid(x)
        h = refinement.refine_grid_1d(g, ratio=2)
        assert np.allclose(h.nodes[0], np.arange(5))

    def test_refinement_grid_1d_non_uniform(self):
        x = np.array([0, 2, 6])
        g = TensorGrid(x)
        h = refinement.refine_grid_1d(g, ratio=2)
        assert np.allclose(h.nodes[0], np.array([0, 1, 2, 4, 6]))

    def test_refinement_grid_1d_general_orientation(self):
        x = np.array([0, 2, 6]) * np.ones((3, 1))
        g = TensorGrid(x[0])
        g.nodes = x
        h = refinement.refine_grid_1d(g, ratio=2)
        assert np.allclose(h.nodes, np.array([[0, 1, 2, 4, 6],
                                              [0, 1, 2, 4, 6],
                                              [0, 1, 2, 4, 6]]))

#------------------------------------------------------------------------------#

    def test_refinement_grid_1d_in_gb_uniform(self):
        f1 = np.array([[0, 1], [.5, .5]])
        ratio = 2

        gb = meshing.cart_grid([f1], [2, 2], **{'physdims': [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        gs_1d = np.array(gb.grids_of_dimension(1))
        hs_1d = np.array([refinement.refine_grid_1d(g, ratio) for g in gs_1d])

        mortars.update_gb_1d(gb, gs_1d, hs_1d)

        known_face_cells = \
                np.array([[ 0.  ,  0.,  0.,  0.  ,  0.  ,  0.,  0.,  0.,  0.  ,
                            0.25,  0.,  0.,  0.  ,  0.25],
                          [ 0.  ,  0.,  0.,  0.  ,  0.  ,  0.,  0.,  0.,  0.  ,
                            0.25,  0.,  0.,  0.  ,  0.25],
                          [ 0.  ,  0.,  0.,  0.  ,  0.  ,  0.,  0.,  0.,  0.25,
                            0.  ,  0.,  0.,  0.25,  0.  ],
                          [ 0.  ,  0.,  0.,  0.  ,  0.  ,  0.,  0.,  0.,  0.25,
                            0.  ,  0.,  0.,  0.25,  0.  ]])

        for _, d in gb.edges_props():
            assert np.allclose(d["face_cells"].todense(), known_face_cells)

#------------------------------------------------------------------------------#

    def test_refinement_grid_1d_in_gb(self):
        f1 = np.array([[0, 1], [.5, .5]])
        num_nodes = 4

        gb = meshing.cart_grid([f1], [2, 2], **{'physdims': [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        gs_1d = np.array(gb.grids_of_dimension(1))
        hs_1d = np.array([refinement.new_grid_1d(g, num_nodes) for g in gs_1d])

        mortars.update_gb_1d(gb, gs_1d, hs_1d)

        known_face_cells = \
                    np.array([[ 0.,  0.,  0.        ,  0.        ,  0.        ,
                                0.,  0.,  0.        ,  0.33333333,  0.        ,
                                0.,  0.,  0.33333333,  0.        ],
                              [ 0.,  0.,  0.        ,  0.        ,  0.        ,
                                0.,  0.,  0.        ,  0.16666667,  0.16666667,
                                0.,  0.,  0.16666667,  0.16666667],
                              [ 0.,  0.,  0.        ,  0.        ,  0.        ,
                                0.,  0.,  0.        ,  0.        ,  0.33333333,
                                0.,  0.,  0.        ,  0.33333333]])

        for _, d in gb.edges_props():
            assert np.allclose(d["face_cells"].todense(), known_face_cells)

#------------------------------------------------------------------------------#
