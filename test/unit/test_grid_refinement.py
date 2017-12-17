#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 17:25:01 2017

@author: Eirik Keilegavlens
"""
import numpy as np
import unittest

from porepy.grids.structured import TensorGrid
from porepy.grids import refinement, mortar_grid
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

    def test_refinement_grid_1d_in_gb_uniform_ratio_2(self):
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
                            0.5,  0.,  0.,  0.  ,  0.5],
                          [ 0.  ,  0.,  0.,  0.  ,  0.  ,  0.,  0.,  0.,  0.  ,
                            0.5,  0.,  0.,  0.  ,  0.5],
                          [ 0.  ,  0.,  0.,  0.  ,  0.  ,  0.,  0.,  0.,  0.5,
                            0.  ,  0.,  0.,  0.5,  0.  ],
                          [ 0.  ,  0.,  0.,  0.  ,  0.  ,  0.,  0.,  0.,  0.5,
                            0.  ,  0.,  0.,  0.5,  0.  ]])

        for _, d in gb.edges_props():
            assert np.allclose(d["face_cells"].todense(), known_face_cells)

#------------------------------------------------------------------------------#

    def test_refinement_grid_1d_in_gb_uniform_ratio_3(self):
        f1 = np.array([[0, 1], [.5, .5]])
        ratio = 3

        gb = meshing.cart_grid([f1], [2, 2], **{'physdims': [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        gs_1d = np.array(gb.grids_of_dimension(1))
        hs_1d = np.array([refinement.refine_grid_1d(g, ratio) for g in gs_1d])

        mortars.update_gb_1d(gb, gs_1d, hs_1d)

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
                                0.,  0.,  0.        ,  0.66666667,  0.        ,
                                0.,  0.,  0.66666667,  0.        ],
                              [ 0.,  0.,  0.        ,  0.        ,  0.        ,
                                0.,  0.,  0.        ,  0.33333333,  0.33333333,
                                0.,  0.,  0.33333333,  0.33333333],
                              [ 0.,  0.,  0.        ,  0.        ,  0.        ,
                                0.,  0.,  0.        ,  0.        ,  0.66666667,
                                0.,  0.,  0.        ,  0.66666667]])

        for _, d in gb.edges_props():
            assert np.allclose(d["face_cells"].todense(), known_face_cells)

#------------------------------------------------------------------------------#

    def test_coarse_grid_1d_in_gb(self):
        f1 = np.array([[0, 1], [.5, .5]])
        num_nodes = 4

        gb = meshing.cart_grid([f1], [4, 2], **{'physdims': [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        gs_1d = np.array(gb.grids_of_dimension(1))
        hs_1d = np.array([refinement.new_grid_1d(g, num_nodes) for g in gs_1d])

        mortars.update_gb_1d(gb, gs_1d, hs_1d)

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

        for _, d in gb.edges_props():
            assert np.allclose(d["face_cells"].todense(), known_face_cells)

#------------------------------------------------------------------------------#

    def test_mortar_grid_1d(self):
        from porepy.viz import plot_grid

        f1 = np.array([[0, 1], [.5, .5]])

        gb = meshing.cart_grid([f1], [2, 2], **{'physdims': [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        #plot_grid.plot_grid(gb, alpha=0, info='f')

        for e, d in gb.edges_props():

            mg = d['mortar']
            # devo farlo per ogni side
            new_side_grids = {s: refinement.new_grid_1d(g, num_nodes=4) \
                              for s, g in mg.side_grids.items()}

            mortars.refine_mortar(mg, new_side_grids)
            mg.compute_geometry()

            # refine the 1d-physical grid
            old_g = gb.sorted_nodes_of_edge(e)[0]
            new_g = refinement.new_grid_1d(old_g, num_nodes=5)
            new_g.compute_geometry()

            gb.update_nodes(old_g, new_g)
            mortars.refine_co_dimensional_grid(mg, new_g)

        #plot_grid.plot_grid(gb, alpha=0, info='c')

        from porepy.params.data import Parameters
        from porepy.params.bc import BoundaryCondition
        from porepy.grids.grid import FaceTag

        internal_flag = FaceTag.FRACTURE
        [g.remove_face_tag_if_tag(FaceTag.BOUNDARY, internal_flag) for g, _ in gb]

        gb.add_node_props(['param'])
        for g, d in gb:
            param = Parameters(g)
            bound_faces = g.get_domain_boundary_faces()
            labels = np.array(['dir'] * bound_faces.size)
            param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))
            d['param'] = param

        gb.add_edge_prop('kn')
        for e, d in gb.edges_props():
            gn = gb.sorted_nodes_of_edge(e)
            d['kn'] = np.ones(gn[0].num_cells)

        from porepy.numerics.vem import vem_dual, vem_source
        # Choose and define the solvers and coupler
        solver_flow = vem_dual.DualVEMMixedDim('flow')
        A_flow, b_flow = solver_flow.matrix_rhs(gb)




TestGridRefinement1d().test_mortar_grid_1d()
