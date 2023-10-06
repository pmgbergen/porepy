"""Test functionality related to line_fracture module.

Most of the common fracture functionality is covered in test_plane_fracture.py.

"""
import numpy as np

from porepy.fracs import line_fracture


def make_line_fracture(index: int | None = None):
    return line_fracture.LineFracture(np.array([[0, 2], [0, 2]]), index=index)


def test_frac_2d():
    # Simple line, known center.
    f_1 = make_line_fracture()
    c_known = np.array([1, 1]).reshape((2, 1))
    n_known = np.array([1, -1]).reshape((1, 2))
    assert np.allclose(c_known, f_1.center)
    assert np.allclose(np.cross(n_known, f_1.normal.T), 0)


def test_frac_index_2d():
    f1 = make_line_fracture(index=1)
    f2 = make_line_fracture(index=1)
    f3 = make_line_fracture(index=4)
    assert f1 == f2
    assert f1 != f3
