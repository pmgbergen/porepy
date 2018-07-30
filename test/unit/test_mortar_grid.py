#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 18:23:11 2017

@author: Eirik Keilegavlen
"""

import numpy as np
import unittest
import scipy.sparse as sps

import porepy as pp


class TestGridMappings1d(unittest.TestCase):
    def test_merge_single_grid(self):
        """
        Test coupling from one grid to itself. An example setting:
                        |--|--|--| ( grid )
                        0  1  2  3
         (left_coupling) \      / (right coupling)
                          \    /
                            * (mortar grid)

        with a coupling from the left face (0) to the right face (1).
        The mortar grid will just be a point
        """

        face_faces = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
        face_faces = sps.csc_matrix(face_faces)
        left_side = pp.PointGrid(np.array([0, 0, 0]).T)
        left_side.compute_geometry()

        side_grids = {pp.grids.mortar_grid.LEFT_SIDE: left_side}
        mg = pp.grids.mortar_grid.BoundaryMortar(0, side_grids, face_faces)

        assert mg.num_cells == 1
        assert mg.num_sides() == 1
        assert np.all(mg.left_to_mortar_avg().A == [1, 0, 0])
        assert np.all(mg.left_to_mortar_int.A == [1, 0, 0])
        assert np.all(mg.right_to_mortar_avg().A == [0, 0, 1])
        assert np.all(mg.right_to_mortar_int.A == [0, 0, 1])

    def test_merge_two_grid(self):
        """
        Test coupling from one grid of three faces to grid of two faces.
        An example setting:
                        0  1  2
                        |--|--| ( left grid)
                           |    (left coupling)
                           *    (mortar_grid
                            \   (right coupling)
                          |--|  (right grid)
                          0  1
        """
        face_faces = np.array([[0, 0, 0], [0, 1, 0]])
        face_faces = sps.csc_matrix(face_faces)
        left_side = pp.PointGrid(np.array([2, 0, 0]).T)
        left_side.compute_geometry()
        side_grids = {pp.grids.mortar_grid.LEFT_SIDE: left_side}

        mg = pp.grids.mortar_grid.BoundaryMortar(0, side_grids, face_faces)

        assert mg.num_cells == 1
        assert mg.num_sides() == 1
        assert np.all(mg.left_to_mortar_avg().A == [0, 1, 0])
        assert np.all(mg.left_to_mortar_int.A == [0, 1, 0])
        assert np.all(mg.right_to_mortar_avg().A == [0, 1])
        assert np.all(mg.right_to_mortar_int.A == [0, 1])


if __name__ == "__main__":
    unittest.main()
