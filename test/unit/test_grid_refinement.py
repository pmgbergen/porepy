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


     if __name__ == '__main__':
          unittest.main()
