#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 17:25:01 2017

@author: Eirik Keilegavlens
"""
import numpy as np
import scipy.sparse as sps
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

        gb = meshing.cart_grid([f1], [2, 2], **{'physdims': [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        gs_1d = np.array(gb.grids_of_dimension(1))
        hs_1d = np.empty(gs_1d.size, dtype=np.object)

        for pos, g in enumerate(gs_1d):
            h = refinement.refine_grid_1d(g, ratio=2)
            weights, new_cells, old_cells = mortars.match_grids_1d(h, g)
            split_matrix = sps.csr_matrix((weights, (new_cells, old_cells)))

            for g_2d in gb.node_neighbors(g, lambda _g: _g.dim > g.dim):
                face_cells = gb.edge_prop((g_2d, g), "face_cells")[0]
                face_cells = split_matrix * face_cells
                gb.add_edge_prop("face_cells", (g_2d, g), face_cells)

            hs_1d[pos] = h

        gb.update_nodes(gs_1d, hs_1d)

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

        gb = meshing.cart_grid([f1], [2, 2], **{'physdims': [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        gs_1d = np.array(gb.grids_of_dimension(1))
        hs_1d = np.empty(gs_1d.size, dtype=np.object)

        for pos, g in enumerate(gs_1d):
            h = refinement.new_grid_1d(g, num_nodes=4)
            weights, new_cells, old_cells = mortars.match_grids_1d(h, g)
            split_matrix = sps.csr_matrix((weights, (new_cells, old_cells)))

            for g_2d in gb.node_neighbors(g, lambda _g: _g.dim > g.dim):
                face_cells = gb.edge_prop((g_2d, g), "face_cells")[0]
                face_cells = split_matrix * face_cells
                gb.add_edge_prop("face_cells", (g_2d, g), face_cells)

            hs_1d[pos] = h

        gb.update_nodes(gs_1d, hs_1d)

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
