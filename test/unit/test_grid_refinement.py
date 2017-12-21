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


class TestGridPerturbation(unittest.TestCase):

    def test_grid_perturbation_1d_bound_nodes_fixed(self):
        g = TensorGrid(np.array([0, 1, 2]))
        h = refinement.distort_grid_1d(g)
        assert np.allclose(g.nodes[:, [0, 2]], h.nodes[:, [0, 2]])

    def test_grid_perturbation_1d_internal_and_bound_nodes_fixed(self):
        g = TensorGrid(np.arange(4))
        h = refinement.distort_grid_1d(g, fixed_nodes=[0, 1, 3])
        assert np.allclose(g.nodes[:, [0, 1, 3]], h.nodes[:, [0, 1, 3]])

    def test_grid_perturbation_1d_internal_nodes_fixed(self):
        g = TensorGrid(np.arange(4))
        h = refinement.distort_grid_1d(g, fixed_nodes=[1])
        assert np.allclose(g.nodes[:, [0, 1, 3]], h.nodes[:, [0, 1, 3]])

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

class TestRefinementGridBucket(unittest.TestCase):
    def test_refinement_grid_1d_in_gb_uniform_ratio_2(self):
        f1 = np.array([[0, 1], [.5, .5]])
        ratio = 2

        gb = meshing.cart_grid([f1], [2, 2], **{'physdims': [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        gs_1d = np.array(gb.grids_of_dimension(1))
        hs_1d = np.array([refinement.refine_grid_1d(g, ratio) for g in gs_1d])

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

        for _, d in gb.edges_props():
            assert np.allclose(d["face_cells"].todense(), known_face_cells)

#------------------------------------------------------------------------------#

class TestRefinementMortarGrid(unittest.TestCase):
    def test_mortar_grid_1d(self):

        f1 = np.array([[0, 1], [.5, .5]])

        gb = meshing.cart_grid([f1], [2, 2], **{'physdims': [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        for e, d in gb.edges_props():

            high_to_mortar_known = np.matrix([[ 0.,  0.,  0.,  0.,
                                                0.,  0.,  0.,  0.,
                                                0.,  1.,  0.,  0.,
                                                0.,  0.],
                                              [ 0.,  0.,  0.,  0.,
                                                0.,  0.,  0.,  0.,
                                                1.,  0.,  0.,  0.,
                                                0.,  0.],
                                              [ 0.,  0.,  0.,  0.,
                                                0.,  0.,  0.,  0.,
                                                0.,  0.,  0.,  0.,
                                                0.,  1.],
                                              [ 0.,  0.,  0.,  0.,
                                                0.,  0.,  0.,  0.,
                                                0.,  0.,  0.,  0.,
                                                1.,  0.]])
            mortar_to_low_known = np.matrix([[ 1.,  0.],
                                             [ 0.,  1.],
                                             [ 1.,  0.],
                                             [ 0.,  1.]])

            mg = d['mortar']
            assert np.allclose(high_to_mortar_known, mg.high_to_mortar.todense())
            assert np.allclose(mortar_to_low_known, mg.mortar_to_low.todense())

#------------------------------------------------------------------------------#

    def test_mortar_grid_1d_equally_refine_mortar_grids(self):

        f1 = np.array([[0, 1], [.5, .5]])

        gb = meshing.cart_grid([f1], [2, 2], **{'physdims': [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        for e, d in gb.edges_props():

            mg = d['mortar']
            new_side_grids = {s: refinement.new_grid_1d(g, num_nodes=4) \
                              for s, g in mg.side_grids.items()}

            mortars.refine_mortar(mg, new_side_grids)

            high_to_mortar_known = 1./3.*np.matrix([
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  2.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  2.]])

            mortar_to_low_known = 1./3.*np.matrix([[ 0.,  2.],
                                                   [ 1.,  1.],
                                                   [ 2.,  0.],
                                                   [ 0.,  2.],
                                                   [ 1.,  1.],
                                                   [ 2.,  0.]])

            assert np.allclose(high_to_mortar_known, mg.high_to_mortar.todense())
            assert np.allclose(mortar_to_low_known, mg.mortar_to_low.todense())

#------------------------------------------------------------------------------#

    def test_mortar_grid_1d_unequally_refine_mortar_grids(self):

        f1 = np.array([[0, 1], [.5, .5]])

        gb = meshing.cart_grid([f1], [2, 2], **{'physdims': [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        for e, d in gb.edges_props():

            mg = d['mortar']
            new_side_grids = {s: refinement.new_grid_1d(g, num_nodes=int(s)+3) \
                              for s, g in mg.side_grids.items()}

            mortars.refine_mortar(mg, new_side_grids)

            high_to_mortar_known = np.matrix(
                                        [[ 0.        ,  0.        ,  0.        ,
                                           0.        ,  0.        ,  0.        ,
                                           0.        ,  0.        ,  0.66666667,
                                           0.        ,  0.        ,  0.        ,
                                           0.        ,  0.        ],
                                         [ 0.        ,  0.        ,  0.        ,
                                           0.        ,  0.        ,  0.        ,
                                           0.        ,  0.        ,  0.33333333,
                                           0.33333333,  0.        ,  0.        ,
                                           0.        ,  0.        ],
                                         [ 0.        ,  0.        ,  0.        ,
                                           0.        ,  0.        ,  0.        ,
                                           0.        ,  0.        ,  0.        ,
                                           0.66666667,  0.        ,  0.        ,
                                           0.        ,  0.        ],
                                         [ 0.        ,  0.        ,  0.        ,
                                           0.        ,  0.        ,  0.        ,
                                           0.        ,  0.        ,  0.        ,
                                           0.        ,  0.        ,  0.        ,
                                           0.5       ,  0.        ],
                                         [ 0.        ,  0.        ,  0.        ,
                                           0.        ,  0.        ,  0.        ,
                                           0.        ,  0.        ,  0.        ,
                                           0.        ,  0.        ,  0.        ,
                                           0.5       ,  0.        ],
                                         [ 0.        ,  0.        ,  0.        ,
                                           0.        ,  0.        ,  0.        ,
                                           0.        ,  0.        ,  0.        ,
                                           0.        ,  0.        ,  0.        ,
                                           0.        ,  0.5       ],
                                         [ 0.        ,  0.        ,  0.        ,
                                           0.        ,  0.        ,  0.        ,
                                           0.        ,  0.        ,  0.        ,
                                           0.        ,  0.        ,  0.        ,
                                           0.        ,  0.5       ]])
            mortar_to_low_known = np.matrix([[ 0.        ,  0.66666667],
                                             [ 0.33333333,  0.33333333],
                                             [ 0.66666667,  0.        ],
                                             [ 0.        ,  0.5       ],
                                             [ 0.        ,  0.5       ],
                                             [ 0.5       ,  0.        ],
                                             [ 0.5       ,  0.        ]])

            assert np.allclose(high_to_mortar_known, mg.high_to_mortar.todense())
            assert np.allclose(mortar_to_low_known, mg.mortar_to_low.todense())

#------------------------------------------------------------------------------#

    def test_mortar_grid_1d_refine_1d_grid(self):

        f1 = np.array([[0, 1], [.5, .5]])

        gb = meshing.cart_grid([f1], [2, 2], **{'physdims': [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        for e, d in gb.edges_props():

            # refine the 1d-physical grid
            old_g = gb.sorted_nodes_of_edge(e)[0]
            new_g = refinement.new_grid_1d(old_g, num_nodes=5)
            new_g.compute_geometry()

            gb.update_nodes(old_g, new_g)
            mg = d['mortar']
            mortars.refine_co_dimensional_grid(mg, new_g)

            high_to_mortar_known = np.matrix([[ 0.,  0.,  0.,  0.,  0.,  0.,
                                                0.,  0.,  0.,  1.,  0.,  0.,
                                                0.,  0.],
                                              [ 0.,  0.,  0.,  0.,  0.,  0.,
                                                0.,  0.,  1.,  0.,  0.,  0.,
                                                0.,  0.],
                                              [ 0.,  0.,  0.,  0.,  0.,  0.,
                                                0.,  0.,  0.,  0.,  0.,  0.,
                                                0.,  1.],
                                              [ 0.,  0.,  0.,  0.,  0.,  0.,
                                                0.,  0.,  0.,  0.,  0.,  0.,
                                                1.,  0.]])
            mortar_to_low_known = np.matrix([[ 0. ,  0. ,  0.5,  0.5],
                                             [ 0.5,  0.5,  0. ,  0. ],
                                             [ 0. ,  0. ,  0.5,  0.5],
                                             [ 0.5,  0.5,  0. ,  0. ]])

            assert np.allclose(high_to_mortar_known, mg.high_to_mortar.todense())
            assert np.allclose(mortar_to_low_known, mg.mortar_to_low.todense())

#------------------------------------------------------------------------------#

    def test_mortar_grid_1d_refine_1d_grid_2(self):

        f1 = np.array([[0, 1], [.5, .5]])

        gb = meshing.cart_grid([f1], [2, 2], **{'physdims': [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        for e, d in gb.edges_props():

            # refine the 1d-physical grid
            old_g = gb.sorted_nodes_of_edge(e)[0]
            new_g = refinement.new_grid_1d(old_g, num_nodes=4)
            new_g.compute_geometry()

            gb.update_nodes(old_g, new_g)
            mg = d['mortar']
            mortars.refine_co_dimensional_grid(mg, new_g)

            high_to_mortar_known = np.matrix([[ 0.,  0.,  0.,  0.,  0.,  0.,
                                                0.,  0.,  0.,  1.,  0.,  0.,
                                                0.,  0.],
                                              [ 0.,  0.,  0.,  0.,  0.,  0.,
                                                0.,  0.,  1.,  0.,  0.,  0.,
                                                0.,  0.],
                                              [ 0.,  0.,  0.,  0.,  0.,  0.,
                                                0.,  0.,  0.,  0.,  0.,  0.,
                                                0.,  1.],
                                              [ 0.,  0.,  0.,  0.,  0.,  0.,
                                                0.,  0.,  0.,  0.,  0.,  0.,
                                                1.,  0.]])
            mortar_to_low_known = 1./3.*np.matrix([[ 0.,  1.,  2.],
                                                   [ 2.,  1.,  0.],
                                                   [ 0.,  1.,  2.],
                                                   [ 2.,  1.,  0.]])

            assert np.allclose(high_to_mortar_known, mg.high_to_mortar.todense())
            assert np.allclose(mortar_to_low_known, mg.mortar_to_low.todense())

#------------------------------------------------------------------------------#

    def test_mortar_grid_2d(self):

        f = np.array([[ 0,  1,  1,  0],
                      [ 0,  0,  1,  1],
                      [.5, .5, .5, .5]])
        gb = meshing.cart_grid([f], [2]*3, **{'physdims': [1]*3})
        gb.compute_geometry()
        gb.assign_node_ordering()

        for e, d in gb.edges_props():

            mg = d['mortar']
            indices_known = np.array([0, 1, 2, 3, 4, 5, 6, 7])
            assert np.array_equal(mg.high_to_mortar.indices, indices_known)

            indptr_known = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 6, 7, 8])
            assert np.array_equal(mg.high_to_mortar.indptr, indptr_known)

            data_known = np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
            assert np.array_equal(mg.high_to_mortar.data, data_known)

            indices_known = np.array([0, 4, 1, 5, 2, 6, 3, 7])
            assert np.array_equal(mg.mortar_to_low.indices, indices_known)

            indptr_known = np.array([0, 2, 4, 6, 8])
            assert np.array_equal(mg.mortar_to_low.indptr, indptr_known)

            data_known = np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
            assert np.array_equal(mg.mortar_to_low.data, data_known)

    if __name__ == '__main__':
        unittest.main()
