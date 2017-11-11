#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 18:23:11 2017

@author: Eirik Keilegavlen
"""

import numpy as np
import unittest

from porepy.grids.structured import TensorGrid
from porepy.fracs import mortars


class TestGridMappings1d(unittest.TestCase):

     def test_merge_grids_all_common(self):
          g = TensorGrid(np.arange(3))
          weights, new, old = mortars.match_grids_1d(g, g)

          assert np.allclose(weights, np.ones(2))
          assert np.allclose(old, np.arange(2))
          assert np.allclose(new, np.arange(2))

     def test_merge_grids_non_matching(self):
          g = TensorGrid(np.arange(3))
          h = TensorGrid(np.arange(3))
          h.nodes[0, 1] = 0.5
          weights, new, old = mortars.match_grids_1d(g, h)

          assert np.allclose(weights, np.array([0.5, 0.5, 1]))
          assert np.allclose(new, np.array([0, 0, 1]))
          assert np.allclose(old, np.array([0, 1, 1]))

     def test_merge_grids_reverse_order(self):
          g = TensorGrid(np.arange(3))
          h = TensorGrid(np.arange(3))
          h.nodes = h.nodes[:, ::-1]
          weights, new, old = mortars.match_grids_1d(g, h)

          assert np.allclose(weights, np.array([1, 1]))
          # In this case, we don't know which ordering the combined grid uses
          # Instead, make sure that the two mappings are ordered in separate
          # directions
          assert np.allclose(new[::-1], old)



     if __name__ == '__main__':
          unittest.main()