#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:12:47 2018

@author: eke001
"""

import numpy as np
import unittest

import porepy as pp


class TestFractureLength(unittest.TestCase):

    def test_single_fracture(self):
        p = np.array([[0, 1], [0, 0]])
        e = np.array([[0], [1]])
        fl = pp.frac_utils.fracture_length_2d(p, e)

        assert fl == 1

    def test_single_fracture_not_aligned(self):
        p = np.array([[0, 1], [0, 1]])
        e = np.array([[0], [1]])
        fl = pp.frac_utils.fracture_length_2d(p, e)

        assert fl == np.sqrt(2)

    def test_two_fractures_separate_points(self):
        p = np.array([[0, 1, 0, 0], [0, 1, 0, 1]])
        e = np.array([[0, 2], [1, 3]])
        fl = pp.frac_utils.fracture_length_2d(p, e)

        fl_known = np.array([np.sqrt(2), 1])
        assert np.allclose(fl, fl_known)

    def test_common_points_reverse_order(self):
        p = np.array([[0, 1, 0], [0, 1, 1]])
        e = np.array([[1, 0], [0, 2]])
        fl = pp.frac_utils.fracture_length_2d(p, e)

        fl_known = np.array([np.sqrt(2), 1])
        assert np.allclose(fl, fl_known)


if __name__ == '__main__':
    unittest.main()