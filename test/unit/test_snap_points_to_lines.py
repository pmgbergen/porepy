#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 09:13:33 2017

@author: Eirik Keilegavlens
"""

import numpy as np
import unittest

from porepy.utils import comp_geom as cg


class TestPointSnapping(unittest.TestCase):
    def test_single_point_snap(self):
        p = np.array([[0, 1], [0, 0]])
        e = np.array([[0], [1]])
        tol = 1e-4
        p_snap = np.array([[0.5], [1e-5]])

        p_new = cg.snap_points_to_segments(p, e, tol, p_snap)
        p_known = np.array([[0.5], [0]])
        assert np.allclose(p_known, p_new, rtol=1e-8)

    def test_single_point_no_snap(self):
        p = np.array([[0, 1], [0, 0]])
        e = np.array([[0], [1]])
        tol = 1e-4
        p_snap = np.array([[0.5], [1e-3]])

        p_new = cg.snap_points_to_segments(p, e, tol, p_snap)
        assert np.allclose(p_snap, p_new)

    def test_snap_two_lines(self):
        p = np.array([[0, 1, 0.5, 0.5], [0, 0, 1e-3, 1]])
        e = np.array([[0, 2], [1, 3]])
        tol = 1e-2

        p_new = cg.snap_points_to_segments(p, e, tol)
        p_known = np.array([[0, 1, 0.5, 0.5], [0, 0, 0, 1]])
        assert np.allclose(p_new, p_known)

    def test_two_lines_no_snap(self):
        p = np.array([[0, 1, 0.5, 0.5], [0, 0, 1e-3, 1]])
        e = np.array([[0, 2], [1, 3]])
        tol = 1e-4

        p_new = cg.snap_points_to_segments(p, e, tol)
        assert np.allclose(p_new, p)

    def test_vertex_snaps(self):
        p = np.array([[0, 1, 0., 0.], [0, 0, 1e-3, 1]])
        e = np.array([[0, 2], [1, 3]])
        tol = 1e-2

        p_new = cg.snap_points_to_segments(p, e, tol)
        p_known = np.array([[0, 1, 0., 0.], [0, 0, 0, 1]])
        assert np.allclose(p_new, p_known)

    def test_snapping_3d(self):
        p = np.array([[0, 1, 0.5, 0.5], [0, 0, 1e-3, 1], [0, 1, 0.5, 1]])
        e = np.array([[0, 2], [1, 3]])
        tol = 1e-2

        p_new = cg.snap_points_to_segments(p, e, tol)
        p_known = np.array([[0, 1, 0.5, 0.5], [0, 0, 0, 1], [0, 1, 0.5, 1]])
        assert np.allclose(p_new, p_known)

    if __name__ == "__main__":
        unittest.main()
