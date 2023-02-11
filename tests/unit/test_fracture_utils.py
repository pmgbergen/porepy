#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:12:47 2018

@author: eke001
"""

import unittest

import numpy as np
import pytest

import porepy as pp
from porepy.fracs.utils import linefractures_to_pts_edges, pts_edges_to_linefractures


def arrays_equal(a, b, tol=1e-5):
    # Utility function
    def nrm(x, y):
        if x.ndim == 1:
            x = np.reshape(x, (-1, 1))
        return np.sqrt(np.sum(np.power(x - y, 2), axis=0))

    is_true = True
    for i in range(a.shape[1]):
        is_true *= np.min(nrm(a[:, i], b)) < tol

    for i in range(b.shape[1]):
        is_true *= np.min(nrm(b[:, i], a)) < tol
    return is_true


class TestFractureLength(unittest.TestCase):
    def test_single_fracture(self):
        p = np.array([[0, 1], [0, 0]])
        e = np.array([[0], [1]])
        fl = pp.frac_utils.fracture_length_2d(p, e)

        self.assertTrue(fl == 1)

    def test_single_fracture_not_aligned(self):
        p = np.array([[0, 1], [0, 1]])
        e = np.array([[0], [1]])
        fl = pp.frac_utils.fracture_length_2d(p, e)

        self.assertTrue(fl == np.sqrt(2))

    def test_two_fractures_separate_points(self):
        p = np.array([[0, 1, 0, 0], [0, 1, 0, 1]])
        e = np.array([[0, 2], [1, 3]])
        fl = pp.frac_utils.fracture_length_2d(p, e)

        fl_known = np.array([np.sqrt(2), 1])
        self.assertTrue(np.allclose(fl, fl_known))

    def test_common_points_reverse_order(self):
        p = np.array([[0, 1, 0], [0, 1, 1]])
        e = np.array([[1, 0], [0, 2]])
        fl = pp.frac_utils.fracture_length_2d(p, e)

        fl_known = np.array([np.sqrt(2), 1])
        self.assertTrue(np.allclose(fl, fl_known))


class TestUniquifyPoints(unittest.TestCase):
    def test_no_change(self):
        p = np.array([[0, 1], [0, 0]])
        e = np.array([[0], [1]])

        up, ue, deleted = pp.frac_utils.uniquify_points(p, e, tol=1e-4)
        self.assertTrue(np.allclose(up, p))
        self.assertTrue(np.allclose(ue, e))
        self.assertTrue(deleted.size == 0)

    def test_merge_one_point(self):
        p = np.array([[0, 1, 0, 0], [0, 1, 0, 1]])
        e = np.array([[0, 2], [1, 3]])
        up, ue, deleted = pp.frac_utils.uniquify_points(p, e, tol=1e-4)

        p_known = np.array([[0, 1, 0], [0, 1, 1]])
        e_known = np.array([[0, 0], [1, 2]])
        self.assertTrue(arrays_equal(p_known, up))
        self.assertTrue(arrays_equal(e_known, ue))
        self.assertTrue(deleted.size == 0)

    def test_merge_one_point_variable_tolerance(self):
        # Check that a point is merged or not, depending on tolerance
        p = np.array([[0, 1, 0, 0], [0, 1, 1e-3, 1]])
        e = np.array([[0, 2], [1, 3]])

        # We should have a merge
        up, ue, deleted = pp.frac_utils.uniquify_points(p, e, tol=1e-2)
        p_known = np.array([[0, 1, 0], [0, 1, 1]])
        e_known = np.array([[0, 0], [1, 2]])
        self.assertTrue(arrays_equal(p_known, up))
        self.assertTrue(arrays_equal(e_known, ue))
        self.assertTrue(deleted.size == 0)

        # There should be no merge
        up, ue, deleted = pp.frac_utils.uniquify_points(p, e, tol=1e-4)
        self.assertTrue(arrays_equal(p, up))
        self.assertTrue(arrays_equal(e, ue))
        self.assertTrue(deleted.size == 0)

    def test_delete_point_edge(self):
        p = np.array([[0, 1, 1, 2], [0, 0, 0, 0]])
        # Edge with tags
        e = np.array([[0, 1, 2], [1, 2, 3], [0, 1, 2]])

        up, ue, deleted = pp.frac_utils.uniquify_points(p, e, tol=1e-2)

        p_known = np.array([[0, 1, 2], [0, 0, 0]])
        # Edge with tags
        e_known = np.array([[0, 1], [1, 2], [0, 2]])
        self.assertTrue(arrays_equal(p_known, up))
        self.assertTrue(arrays_equal(e_known, ue))
        self.assertTrue(deleted.size == 1)
        self.assertTrue(deleted[0] == 1)


class TestConversionBetweenLineFracturesAndPointsEdges:
    """Test conversion between pp.LineFractures and pts/edges and vice versa.

    This is done via the functions:
        - pts_edges_to_linefractures(), and
        - linefractures_to_pts_edges().

    """

    def test_linefractures_to_pts_edges(self):
        frac1 = pp.LineFracture([[0, 2], [1, 3]])
        frac2 = pp.LineFracture([[2, 4], [3, 5]])
        frac3 = pp.LineFracture([[0, 4], [1, 5]])
        pts = np.array([[0, 2, 4], [1, 3, 5]])
        edges = np.array([[0, 1, 0], [1, 2, 2]], dtype=int)
        converted_pts, converted_edges = linefractures_to_pts_edges([frac1, frac2, frac3])
        assert np.allclose(converted_pts, pts)
        assert np.allclose(converted_edges, edges)
        # TODO: Add a test to check that the tolerance works correctly.

    def test_linefractures_to_pts_edges_with_tags(self):
        frac1 = pp.LineFracture([[0, 2], [1, 3]], tags=[-1, -1, 2])
        frac2 = pp.LineFracture([[2, 4], [3, 5]])
        frac3 = pp.LineFracture([[0, 4], [1, 5]], tags=[1, 1])
        pts = np.array([[0, 2, 4], [1, 3, 5]])
        # All edges will have the maximal number of tags (3 in this example).
        edges = np.array(
            [[0, 1, 0], [1, 2, 2], [-1, -1, 1], [-1, -1, 1], [2, -1, -1]], dtype=int
        )
        converted_pts, converted_edges = linefractures_to_pts_edges([frac1, frac2, frac3])
        assert np.allclose(converted_pts, pts)
        assert np.allclose(converted_edges, edges)

    def test_pts_edges_to_linefractures(self):
        frac1 = pp.LineFracture([[0, 2], [1, 3]])
        frac2 = pp.LineFracture([[2, 4], [3, 5]])
        frac3 = pp.LineFracture([[0, 4], [1, 5]])
        pts = np.array([[0, 2, 4], [1, 3, 5]])
        edges = np.array([[0, 1, 0], [1, 2, 2]], dtype=int)
        converted_fracs = pts_edges_to_linefractures(pts, edges)
        for converted_pt, pt in zip(converted_fracs[0].points(), frac1.points()):
            assert np.allclose(converted_pt, pt)
        for converted_pt, pt in zip(converted_fracs[1].points(), frac2.points()):
            assert np.allclose(converted_pt, pt)
        for converted_pt, pt in zip(converted_fracs[2].points(), frac3.points()):
            assert np.allclose(converted_pt, pt)

    def test_pts_edges_to_linefractures_with_tags(self):
        frac1 = pp.LineFracture([[0, 2], [1, 3]], tags=[-1, 2, -1])
        frac2 = pp.LineFracture([[2, 4], [3, 5]], tags=[1])
        frac3 = pp.LineFracture([[0, 4], [1, 5]])
        pts = np.array([[0, 2, 4], [1, 3, 5]])
        edges = np.array(
            [[0, 1, 0], [1, 2, 2], [-1, 1, -1], [2, -1, -1], [-1, -1, -1]], dtype=int
        )
        converted_fracs = pts_edges_to_linefractures(pts, edges)
        for converted_pt, pt in zip(converted_fracs[0].points(), frac1.points()):
            assert np.allclose(converted_pt, pt)
        for converted_tag, tag in zip(converted_fracs[0].tags, frac1.tags):
            assert np.all(converted_tag == tag)
        for converted_pt, pt in zip(converted_fracs[1].points(), frac2.points()):
            assert np.allclose(converted_pt, pt)
        for converted_tag, tag in zip(converted_fracs[1].tags, frac2.tags):
            assert np.all(converted_tag == tag)
        for converted_pt, pt in zip(converted_fracs[2].points(), frac3.points()):
            assert np.allclose(converted_pt, pt)
        for converted_tag, tag in zip(converted_fracs[2].tags, frac3.tags):
            assert np.all(converted_tag == tag)


if __name__ == "__main__":
    unittest.main()
