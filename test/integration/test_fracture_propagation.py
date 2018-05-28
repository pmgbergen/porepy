#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of the propagation of fractures. This involves extension of the fracture
grids and splitting of faces in the matrix grid. The post-propagation buckets
are compared to equivalent buckets where the final fracture geometry is applied
at construction.
"""

import numpy as np
import unittest
import porepy as pp
from test.integration.fracture_propagation_utils import \
    check_equivalent_buckets
from test.integration import setup_mixed_dimensional_grids as setup_gb


class BasicsTest(unittest.TestCase):

    def test_propagation_visulization_2d(self):
        """
        Setup intended for visual inspection before and after fracture
        propagation of a single 1d fracture in 2d. No actual testing performed.
        """
        f = np.array([[0, 1],
                      [.5, .5]])
        gb = pp.meshing.cart_grid([f], [4, 2], physdims=[2, 1])
#        for g, d in gb:
#            d['cell_tags'] = np.ones(g.num_cells) * g.dim
#        pp.plot_grid(gb, 'cell_tags', info='f')

        face_2d = [np.array([16])]
        pp.propagate_fracture.propagate_fractures(gb, face_2d)
#        for g, d in gb:
#            d['cell_tags'] = np.ones(g.num_cells) * g.dim
#        pp.plot_grid(gb, 'cell_tags', info='f')

    def test_propagation_visualization_3d(self):
        """
        Setup intended for visual inspection before and after fracture
        propagation of a single 2d fracture in 3d. No actual testing performed.
        """
        f = np.array([[0., 1, 1, 0],
                      [.25, .25, .75, .75],
                      [.5, .5, .5, .5]])
        gb = pp.meshing.cart_grid([f], [4, 4, 2], physdims=[2, 1, 1])

#        e = pp.Exporter(gb, 'grid_before', 'test_propagation_3d')
#        e.write_vtk()
        faces_3d = [np.array([102, 106, 110])]
#        pp.propagate_fracture.propagate_fracture(gb, gh, gl, faces_h=face_3d)
        pp.propagate_fracture.propagate_fractures(gb, faces_3d)
#        e_after = pp.Exporter(gb, 'grid_after', 'test_propagation_3d')
#        e_after.write_vtk()

    def test_equivalence_2d(self):
        """
        This central test checks that certain geometry, connectivity and tags
        are independent of whether the fracture is
        1 premade in its final form
        2-3 grown in a single step (from two different inital fractures)
        4 grown in two steps
        """

        gb_1 = setup_gb.grid_2d_1d([4, 2], 0, .75)
        gb_2 = setup_gb.grid_2d_1d([4, 2], 0, .25)
        gb_3 = setup_gb.grid_2d_1d([4, 2], 0, .50)
        gb_4 = setup_gb.grid_2d_1d([4, 2], 0, .25)
        # Split
        pp.propagate_fracture.propagate_fractures(gb_2, [[15, 16]])
        pp.propagate_fracture.propagate_fractures(gb_3, [[16]])
        pp.propagate_fracture.propagate_fractures(gb_4, [[15]])
        pp.propagate_fracture.propagate_fractures(gb_4, [[16]])
        # Check that the four grid buckets are equivalent
        check_equivalent_buckets([gb_1, gb_2, gb_3, gb_4])

    def test_equivalence_3d(self):
        """
        3d version of previous test.
        This central test checks that certain geometry, connectivity and tags
        are independent of whether the fracture is
        1 premade in its final form
        2-3 grown in a single step (from two different inital fractures)
        4 grown in two steps
        """
        # Make buckets
        gb_1 = setup_gb.grid_3d_2d([4, 2, 2], 0, .75)
        gb_2 = setup_gb.grid_3d_2d([4, 2, 2], 0, .25)
        gb_3 = setup_gb.grid_3d_2d([4, 2, 2], 0, .50)
        gb_4 = setup_gb.grid_3d_2d([4, 2, 2], 0, .25)
        # Split
        pp.propagate_fracture.propagate_fractures(gb_2, [[53, 54]])
        pp.propagate_fracture.propagate_fractures(gb_3, [[54]])
        pp.propagate_fracture.propagate_fractures(gb_4, [[53]])
        pp.propagate_fracture.propagate_fractures(gb_4, [[54]])
        # Check that the four grid buckets are equivalent
        check_equivalent_buckets([gb_1, gb_2, gb_3, gb_4])

    def test_two_fractures_2d(self):
        """
        Two fractures growing towards each other, but not meeting.
        Note that the bucket equivalence test is only applied to the first
        grid in each dimension!
        Tests simultanous growth of two fractures in multiple steps, and growth
        of one fracture in the presence of an inactive one.
        """
        f_1 = np.array([[0, .25], [.5, .5]])
        f_2 = np.array([[1.75, 2], [.5, .5]])
        gb = pp.meshing.cart_grid([f_1, f_2], [8, 2], physdims=[2, 1])

        f_1 = np.array([[0, .5], [.5, .5]])
        f_2 = np.array([[1.5, 2], [.5, .5]])
        gb_1 = pp.meshing.cart_grid([f_1, f_2], [8, 2], physdims=[2, 1])

        f_1 = np.array([[0, .75], [.5, .5]])
        f_2 = np.array([[1.25, 2], [.5, .5]])
        gb_2 = pp.meshing.cart_grid([f_1, f_2], [8, 2], physdims=[2, 1])

        f_1 = np.array([[0, 1.0], [.5, .5]])
        gb_3 = pp.meshing.cart_grid([f_1, f_2], [8, 2], physdims=[2, 1])

        faces = [np.array([27]), np.array([32])]
        pp.propagate_fracture.propagate_fractures(gb, faces)
        check_equivalent_buckets([gb, gb_1])

        faces = [np.array([28]), np.array([31])]
        pp.propagate_fracture.propagate_fractures(gb, faces)
        check_equivalent_buckets([gb, gb_2])

        faces = [np.array([29]), np.empty(0, dtype=int)]
        pp.propagate_fracture.propagate_fractures(gb, faces)
        check_equivalent_buckets([gb, gb_3])


#    def test_two_fractures_3d(self):
#        f_1 = np.aarray([[0.  , 1.  , 1.  , 0.  ],
#                         [0.  , 0.  , 0.25, 0.25],
#                         [0.5 , 0.5 , 0.5 , 0.5 ]])
#        f_2 = np.aarray([[0.  , 1.  , 1.  , 0.  ],
#                         [0.75, 0.75, 0.  , 0.  ],
#                         [0.5 , 0.5 , 0.5 , 0.5 ]])
#        gb = pp.meshing.cart_grid([f_1, f_2], [8, 4, 2], physdims=[2, 1, 1])

if __name__ == '__main__':
    unittest.main()
