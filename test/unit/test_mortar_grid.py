#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 18:23:11 2017

@author: Eirik Keilegavlen
"""

import numpy as np
import unittest
import scipy.sparse as sps

from porepy.grids.structured import TensorGrid
from porepy.grids import mortar_grid


class TestGridMappings1d(unittest.TestCase):
    def test_merge_single_grid(self):
        g = TensorGrid(np.arange(3))
        g.compute_geometry()
        face_faces = np.array([[0, 0, 1],
                               [0, 0, 0],
                               [0, 0, 0]])
        face_faces = sps.csc_matrix(face_faces)

        side_grids = {mortar_grid.LEFT_SIDE: g}
        mg = mortar_grid.BoundaryMortar(0, side_grids, face_faces)

        assert mg.num_cells == 1
        assert mg.num_sides() == 1
        assert np.all(mg.left_to_mortar_avg().A == [1, 0, 0])
        assert np.all(mg.left_to_mortar_int.A == [1, 0, 0])
        assert np.all(mg.right_to_mortar_avg().A == [0, 0, 1])
        assert np.all(mg.right_to_mortar_int.A == [0, 0, 1])

    def test_merge_two_grid(self):
        g = TensorGrid(np.arange(3))
        h = TensorGrid(np.arange(2))
        g.compute_geometry()
        h.compute_geometry()
        face_faces = np.array([[0, 0, 0],
                               [0, 1, 0]])
        face_faces = sps.csc_matrix(face_faces)

        side_grids = {mortar_grid.LEFT_SIDE: h,
                      mortar_grid.RIGHT_SIDE: g}
        mg = mortar_grid.BoundaryMortar(0, side_grids, face_faces)

        assert mg.num_cells == 1
        assert mg.num_sides() == 2
        assert np.all(mg.left_to_mortar_avg().A == [0, 1])
        assert np.all(mg.left_to_mortar_int.A == [0, 1])
        assert np.all(mg.right_to_mortar_avg().A == [0, 1, 0])
        assert np.all(mg.right_to_mortar_int.A == [0, 1, 0])

if __name__ == '__main__':
    unittest.main()