#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:12:47 2018

@author: eke001
"""

import numpy as np
import unittest

import porepy as pp


def arrays_equal(a, b, tol=1e-5):
    # Utility function
    def nrm(x, y):
        if x.ndim == 1:
            x = np.reshape(x, (-1, 1))
        return np.sqrt(np.sum(np.power(x - y, 2), axis=0))

    for i in range(a.shape[1]):
        assert np.min(nrm(a[:, i], b)) < tol

    for i in range(b.shape[1]):
        assert np.min(nrm(b[:, i], a)) < tol


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


class TestUniquifyPoints(unittest.TestCase):
    def test_no_change(self):
        p = np.array([[0, 1], [0, 0]])
        e = np.array([[0], [1]])

        up, ue, deleted = pp.frac_utils.uniquify_points(p, e, tol=1e-4)
        assert np.allclose(up, p)
        assert np.allclose(ue, e)
        assert deleted.size == 0

    def test_merge_one_point(self):
        p = np.array([[0, 1, 0, 0], [0, 1, 0, 1]])
        e = np.array([[0, 2], [1, 3]])
        up, ue, deleted = pp.frac_utils.uniquify_points(p, e, tol=1e-4)

        p_known = np.array([[0, 1, 0], [0, 1, 1]])
        e_known = np.array([[0, 0], [1, 2]])
        arrays_equal(p_known, up)
        arrays_equal(e_known, ue)
        assert deleted.size == 0

    def test_merge_one_point_variable_tolerance(self):
        # Check that a point is merged or not, depending on tolerance
        p = np.array([[0, 1, 0, 0], [0, 1, 1e-3, 1]])
        e = np.array([[0, 2], [1, 3]])

        # We should have a merge
        up, ue, deleted = pp.frac_utils.uniquify_points(p, e, tol=1e-2)
        p_known = np.array([[0, 1, 0], [0, 1, 1]])
        e_known = np.array([[0, 0], [1, 2]])
        arrays_equal(p_known, up)
        arrays_equal(e_known, ue)
        assert deleted.size == 0

        # There should be no merge
        up, ue, deleted = pp.frac_utils.uniquify_points(p, e, tol=1e-4)
        arrays_equal(p, up)
        arrays_equal(e, ue)
        assert deleted.size == 0

    def test_delete_point_edge(self):
        p = np.array([[0, 1, 1, 2], [0, 0, 0, 0]])
        # Edge with tags
        e = np.array([[0, 1, 2], [1, 2, 3], [0, 1, 2]])

        up, ue, deleted = pp.frac_utils.uniquify_points(p, e, tol=1e-2)

        p_known = np.array([[0, 1, 2], [0, 0, 0]])
        # Edge with tags
        e_known = np.array([[0, 1], [1, 2], [0, 2]])
        arrays_equal(p_known, up)
        arrays_equal(e_known, ue)
        assert deleted.size == 1
        assert deleted[0] == 1


class TestFractureSnapping(unittest.TestCase):
    def test_no_snapping(self):
        p = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
        e = np.array([[0, 2], [1, 3]])

        pn, conv = pp.frac_utils.snap_fracture_set_2d(p, e, snap_tol=1e-3)
        assert np.allclose(p, pn)
        assert conv

    def test_snap_to_vertex(self):
        p = np.array([[0, 1, 0, 1], [0, 0, 1e-4, 1]])
        e = np.array([[0, 2], [1, 3]])

        pn, conv = pp.frac_utils.snap_fracture_set_2d(p, e, snap_tol=1e-3)
        p_known = np.array([[0, 1, 0, 1], [0, 0, 0, 1]])
        assert np.allclose(p_known, pn)
        assert conv

    def test_no_snap_to_vertex_small_tol(self):
        # No snapping because the snapping tolerance is small
        p = np.array([[0, 1, 0, 1], [0, 0, 1e-4, 1]])
        e = np.array([[0, 2], [1, 3]])

        pn, conv = pp.frac_utils.snap_fracture_set_2d(p, e, snap_tol=1e-5)
        assert np.allclose(p, pn)
        assert conv

    def test_snap_to_segment(self):
        p = np.array([[0, 1, 0.5, 1], [0, 0, 1e-4, 1]])
        e = np.array([[0, 2], [1, 3]])

        pn, conv = pp.frac_utils.snap_fracture_set_2d(p, e, snap_tol=1e-3)
        p_known = np.array([[0, 1, 0.5, 1], [0, 0, 0, 1]])
        assert np.allclose(p_known, pn)
        assert conv


if __name__ == "__main__":
    unittest.main()
