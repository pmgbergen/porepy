"""This file contains testing of funcionality found within
porepy.geometry.constrain_geometry.py"""

import numpy as np

from porepy import constrain_geometry
from tests.test_utils import compare_arrays


class TestConstrainGeometry:
    """
    Various tests of intersect_polygon_lines.

    """

    def test_constrain_lines_by_polygon(self):
        # convex polygon
        polygon = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
        pts = np.array(
            [
                [0.0, 1.0, 0.0, 2.0, 0.5, 0.0, -0.5, 0.3],
                [0.0, 1.0, 0.0, 2.0, 1.0, 0.0, -0.5, 0.6],
            ]
        )
        lines = np.array([[0, 2, 4, 6], [1, 3, 5, 7]])

        new_pts, new_lines, lines_kept = constrain_geometry.lines_by_polygon(
            polygon, pts, lines
        )

        pts_known = np.array(
            [
                [0.0, 1.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.3],
                [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 3 / 16, 0.6],
            ]
        )
        lines_known = np.array([[0, 2, 4, 6], [1, 3, 5, 7]])
        kept_known = np.arange(lines_known.shape[1])

        assert compare_arrays(new_pts, pts_known)
        assert compare_arrays(new_lines, lines_known)

        assert compare_arrays(lines_kept, kept_known)

    def test_convex_polygon_line_outside(self):
        # Test intersection between a line outside a convex polygon and the convex
        # polygon.
        polygon = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])

        # The last line is completely outside, and will be kicked out
        pts = np.array(
            [
                [0.0, 1.0, 0.0, 2.0, 0.5, 0.0, -0.5, 0.3, -1, 0.0],
                [0.0, 1.0, 0.0, 2.0, 1.0, 0.0, -0.5, 0.6, -1, 0.0],
            ]
        )
        lines = np.array([[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]])

        new_pts, new_lines, lines_kept = constrain_geometry.lines_by_polygon(
            polygon, pts, lines
        )

        pts_known = np.array(
            [
                [0.0, 1.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.3],
                [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 3 / 16, 0.6],
            ]
        )
        lines_known = np.array([[0, 2, 4, 6], [1, 3, 5, 7]])
        kept_known = np.array([0, 1, 2, 3])

        assert compare_arrays(new_pts, pts_known)
        assert compare_arrays(new_lines, lines_known)
        assert compare_arrays(lines_kept, kept_known)

    def test_non_convex_polygon(self):
        # Test intersections between lines and a non-convex polygon.
        polygon = np.array(
            [[0.0, 0.5, 0.75, 1.0, 1.5, 1.5, 0], [0.0, 0.0, 0.25, 0.0, 0, 1, 1]]
        )
        pts = np.array(
            [
                [0.0, 1.0, 0.0, 2.0, 0.5, 0.0, -0.5, 0.3, -1, 0.0, 0.0, 2.0, -0.1, 1.1],
                [0.0, 1.0, 0.0, 2.0, 1.0, 0.0, -0.5, 0.6, -1, 0.0, 0.2, 0.2, 0.0, 0.0],
            ]
        )
        lines = np.array([[0, 2, 4, 6, 8, 10, 12], [1, 3, 5, 7, 9, 11, 13]])

        new_pts, new_lines, lines_kept = constrain_geometry.lines_by_polygon(
            polygon, pts, lines
        )
        pts_known = np.array(
            [
                [0.0, 1.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.3, 0.0, 0.7, 0.8, 1.5],
                [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 3 / 16, 0.6, 0.2, 0.2, 0.2, 0.2],
            ]
        )
        lines_known = np.array([[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]])

        kept_known = np.array([0, 1, 2, 3, 5, 5])

        assert compare_arrays(new_pts, pts_known)
        assert compare_arrays(new_lines, lines_known)
        assert compare_arrays(lines_kept, kept_known)


# ---------- Testing snap_points_to_segments ----------


def test_single_point_snap():
    p = np.array([[0, 1], [0, 0]])
    e = np.array([[0], [1]])
    tol = 1e-4
    p_snap = np.array([[0.5], [1e-5]])

    p_new = constrain_geometry.snap_points_to_segments(p, e, tol, p_snap)
    p_known = np.array([[0.5], [0]])
    assert np.allclose(p_known, p_new, rtol=1e-8)


def test_single_point_no_snap():
    p = np.array([[0, 1], [0, 0]])
    e = np.array([[0], [1]])
    tol = 1e-4
    p_snap = np.array([[0.5], [1e-3]])

    p_new = constrain_geometry.snap_points_to_segments(p, e, tol, p_snap)
    assert np.allclose(p_snap, p_new)


def test_snap_two_lines():
    p = np.array([[0, 1, 0.5, 0.5], [0, 0, 1e-3, 1]])
    e = np.array([[0, 2], [1, 3]])
    tol = 1e-2

    p_new = constrain_geometry.snap_points_to_segments(p, e, tol)
    p_known = np.array([[0, 1, 0.5, 0.5], [0, 0, 0, 1]])
    assert np.allclose(p_new, p_known)


def test_two_lines_no_snap():
    p = np.array([[0, 1, 0.5, 0.5], [0, 0, 1e-3, 1]])
    e = np.array([[0, 2], [1, 3]])
    tol = 1e-4

    p_new = constrain_geometry.snap_points_to_segments(p, e, tol)
    assert np.allclose(p_new, p)


def test_vertex_snaps():
    p = np.array([[0, 1, 0.0, 0.0], [0, 0, 1e-3, 1]])
    e = np.array([[0, 2], [1, 3]])
    tol = 1e-2

    p_new = constrain_geometry.snap_points_to_segments(p, e, tol)
    p_known = np.array([[0, 1, 0.0, 0.0], [0, 0, 0, 1]])
    assert np.allclose(p_new, p_known)


def test_snapping_3d():
    p = np.array([[0, 1, 0.5, 0.5], [0, 0, 1e-3, 1], [0, 1, 0.5, 1]])
    e = np.array([[0, 2], [1, 3]])
    tol = 1e-2

    p_new = constrain_geometry.snap_points_to_segments(p, e, tol)
    p_known = np.array([[0, 1, 0.5, 0.5], [0, 0, 0, 1], [0, 1, 0.5, 1]])
    assert np.allclose(p_new, p_known)
