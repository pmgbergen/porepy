"""Test triangulation matching in intersections module
"""
import numpy as np

import porepy as pp


def test_identical_triangles():
    p = np.array([[0, 1, 0], [0, 0, 1]])
    t = np.array([[0, 1, 2]]).T

    triangulation = pp.intersections.triangulations(p, p, t, t)
    assert len(triangulation) == 1
    assert triangulation[0][0] == 0
    assert triangulation[0][1] == 0
    assert triangulation[0][2] == 0.5


def test_two_and_one():
    p1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1]])
    t1 = np.array([[0, 1, 3], [1, 2, 3]]).T

    p2 = np.array([[0, 1, 0], [0, 1, 1]])
    t2 = np.array([[0, 1, 2]]).T

    triangulation = pp.intersections.triangulations(p1, p2, t1, t2)
    assert len(triangulation) == 2
    assert triangulation[0][0] == 0
    assert triangulation[0][1] == 0
    assert triangulation[0][2] == 0.25
    assert triangulation[1][0] == 1
    assert triangulation[1][1] == 0
    assert triangulation[1][2] == 0.25


def test_one_and_two():
    p1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1]])
    t1 = np.array([[0, 1, 3], [1, 2, 3]]).T

    p2 = np.array([[0, 1, 0], [0, 1, 1]])
    t2 = np.array([[0, 1, 2]]).T

    triangulation = pp.intersections.triangulations(p2, p1, t2, t1)
    assert len(triangulation) == 2
    assert triangulation[0][0] == 0
    assert triangulation[0][1] == 0
    assert triangulation[0][2] == 0.25
    assert triangulation[1][1] == 1
    assert triangulation[1][0] == 0
    assert triangulation[1][2] == 0.25
