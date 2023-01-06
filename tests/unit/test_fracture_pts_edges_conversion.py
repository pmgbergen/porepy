import numpy as np
import pytest

import porepy as pp
from porepy.fracs.utils import pts_edges_to_linefractures, linefractures_to_pts_edges


def test_pts_edges_to_linefractures():
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


def test_linefractures_to_pts_edges():
    frac1 = pp.LineFracture([[0, 2], [1, 3]])
    frac2 = pp.LineFracture([[2, 4], [3, 5]])
    frac3 = pp.LineFracture([[0, 4], [1, 5]])
    pts = np.array([[0, 2, 4], [1, 3, 5]])
    edges = np.array([[0, 1, 0], [1, 2, 2]], dtype=int)
    converted_pts, converted_edges = linefractures_to_pts_edges([frac1, frac2, frac3])
    assert np.allclose(converted_pts, pts)
    assert np.allclose(converted_edges, edges)


test_linefractures_to_pts_edges()
test_pts_edges_to_linefractures()
