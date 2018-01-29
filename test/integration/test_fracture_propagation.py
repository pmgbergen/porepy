#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:54:32 2018

@author: ivar
"""

import numpy as np
import unittest
from porepy.fracs import meshing, propagate_fracture
from porepy.viz import plot_grid, exporter
from test.integration.fracture_propagation_utils import \
    check_equivalent_buckets, propagate_simple
from test.integration import setup_mixed_dimensional_grids as setup_gb

#------------------------------------------------------------------------------#


class BasicsTest(unittest.TestCase):

#------------------------------------------------------------------------------#

    def test_propagation_visulization_2d(self):
        """
        Setup intended for visual inspection before and after fracture
        propagation of a single 1d fracture in 2d. No actual testing performed.
        """
        f = np.array([[0, 1],
                      [.5, .5]])
        gb = meshing.cart_grid([f], [4, 2], physdims=[2, 1])
#        for g, d in gb:
#            d['cell_tags'] = np.ones(g.num_cells) * g.dim
#        plot_grid.plot_grid(gb, 'cell_tags', info='f')

        face_2d = np.array([16])
        gh = gb.grids_of_dimension(2)[0]
        gl = gb.grids_of_dimension(1)[0]
        propagate_fracture.propagate_fracture(gb, gh, gl, faces_h=face_2d)
#        for g, d in gb:
#            d['cell_tags'] = np.ones(g.num_cells) * g.dim
#        plot_grid.plot_grid(gb, 'cell_tags', info='f')

    def test_propagation_visualization_3d(self):
        """
        Setup intended for visual inspection before and after fracture
        propagation of a single 2d fracture in 3d. No actual testing performed.
        """
        f = np.array([[0., 1, 1, 0],
                      [.25, .25, .75, .75],
                      [.5, .5, .5, .5]])
        gb = meshing.cart_grid([f], [4, 4, 2], physdims=[2, 1, 1])

#        e = exporter.Exporter(gb, 'grid_before', 'test_propagation_3d')
#        e.write_vtk()
        face_3d = np.array([102, 106, 110])
        gh = gb.grids_of_dimension(3)[0]
        gl = gb.grids_of_dimension(2)[0]
        propagate_fracture.propagate_fracture(gb, gh, gl, faces_h=face_3d)
#        e_after = exporter.Exporter(gb, 'grid_after', 'test_propagation_3d')
#        e_after.write_vtk()

    def test_equivalence_2d(self):
        """
        This central test checks that certain geometry, connectivity and tags
        are independent of whether the fracture is
        1 premade in its final form
        2-3 grown in a single step (from two different inital fractures)
        4 grown in two steps
        """

        gb_1 = setup_gb.grid_2d_1d([4, 2], .75)
        gb_2 = setup_gb.grid_2d_1d([4, 2], .25)
        gb_3 = setup_gb.grid_2d_1d([4, 2], .50)
        gb_4 = setup_gb.grid_2d_1d([4, 2], .25)
        # Split
        propagate_simple(gb_2, [15, 16])
        propagate_simple(gb_3, [16])
        propagate_simple(gb_4, [15])
        propagate_simple(gb_4, [16])
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
        gb_1 = setup_gb.grid_3d_2d([4, 2, 2], .75)
        gb_2 = setup_gb.grid_3d_2d([4, 2, 2], .25)
        gb_3 = setup_gb.grid_3d_2d([4, 2, 2], .50)
        gb_4 = setup_gb.grid_3d_2d([4, 2, 2], .25)
        # Split
        propagate_simple(gb_2, [53, 54])
        propagate_simple(gb_3, [54])
        propagate_simple(gb_4, [53])
        propagate_simple(gb_4, [54])
        # Check that the four grid buckets are equivalent
        check_equivalent_buckets([gb_1, gb_2, gb_3, gb_4])


if __name__ == '__main__':
    BasicsTest().test_equivalence_2d()
    BasicsTest().test_equivalence_3d()
