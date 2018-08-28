#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 13:01:04 2017

@author: eke001
"""
import numpy as np
import unittest

import porepy.utils.comp_geom as cg


class TestTriangulationMatching(unittest.TestCase):
    def test_identical_triangles(self):
        p = np.array([[0, 1, 0], [0, 0, 1]])
        t = np.array([[0, 1, 2]]).T

        l = cg.intersect_triangulations(p, p, t, t)
        assert len(l) == 1
        assert l[0][0] == 0
        assert l[0][1] == 0
        assert l[0][2] == 0.5

    def test_two_and_one(self):
        p1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1]])
        t1 = np.array([[0, 1, 3], [1, 2, 3]]).T

        p2 = np.array([[0, 1, 0], [0, 1, 1]])
        t2 = np.array([[0, 1, 2]]).T

        l = cg.intersect_triangulations(p1, p2, t1, t2)
        assert len(l) == 2
        assert l[0][0] == 0
        assert l[0][1] == 0
        assert l[0][2] == 0.25
        assert l[1][0] == 1
        assert l[1][1] == 0
        assert l[1][2] == 0.25

    def test_one_and_two(self):
        p1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1]])
        t1 = np.array([[0, 1, 3], [1, 2, 3]]).T

        p2 = np.array([[0, 1, 0], [0, 1, 1]])
        t2 = np.array([[0, 1, 2]]).T

        l = cg.intersect_triangulations(p2, p1, t2, t1)
        assert len(l) == 2
        assert l[0][0] == 0
        assert l[0][1] == 0
        assert l[0][2] == 0.25
        assert l[1][1] == 1
        assert l[1][0] == 0
        assert l[1][2] == 0.25

    if __name__ == "__main__":
        unittest.main()
