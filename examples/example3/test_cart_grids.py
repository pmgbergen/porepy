"""
Various method for creating grids for relatively simple fracture networks.

The module doubles as a test framework (though not unittest), and will report
on any problems if ran as a main method.

"""
import getopt
import numpy as np

from porepy.fracs import meshing


def test_x_intersection_2d(**kwargs):
    f_1 = np.array([[2, 6], [5, 5]])
    f_2 = np.array([[4, 4], [2, 7]])
    f = [f_1, f_2]

    gb = meshing.cart_grid(f, [10, 10], physdims=[10, 10])
    if kwargs.get("return_expected", False):
        expected = [0, 1, 2, 1]
        return gb, expected
    else:
        return gb


def test_T_intersection_2d(**kwargs):
    f_1 = np.array([[2, 6], [5, 5]])
    f_2 = np.array([[4, 4], [2, 5]])
    f = [f_1, f_2]

    gb = meshing.cart_grid(f, [10, 10], physdims=[10, 10])
    if kwargs.get("return_expected", False):
        expected = [0, 1, 2, 1]
        return gb, expected
    else:
        return gb


def test_L_intersection_2d(**kwargs):
    f_1 = np.array([[2, 6], [5, 5]])
    f_2 = np.array([[6, 6], [2, 5]])
    f = [f_1, f_2]

    gb = meshing.cart_grid(f, [10, 10], physdims=[10, 10])
    if kwargs.get("return_expected", False):
        expected = [0, 1, 2, 1]
        return gb, expected
    else:
        return gb


def test_x_intersection_3d(**kwargs):
    f_1 = np.array([[2, 5, 5, 2], [2, 2, 5, 5], [5, 5, 5, 5]])
    f_2 = np.array([[2, 2, 5, 5], [5, 5, 5, 5], [2, 5, 5, 2]])
    f = [f_1, f_2]
    gb = meshing.cart_grid(f, np.array([10, 10, 10]))
    if kwargs.get("return_expected", False):
        expected = [1, 2, 1, 0]
        return gb, expected
    else:
        return gb


def test_several_intersections_3d(**kwargs):
    f_1 = np.array([[2, 5, 5, 2], [2, 2, 5, 5], [5, 5, 5, 5]])
    f_2 = np.array([[2, 2, 5, 5], [5, 5, 5, 5], [2, 5, 5, 2]])
    f_3 = np.array([[4, 4, 4, 4], [1, 1, 8, 8], [1, 8, 8, 1]])
    f_4 = np.array([[3, 3, 6, 6], [3, 3, 3, 3], [3, 7, 7, 3]])
    f = [f_1, f_2, f_3, f_4]
    gb = meshing.cart_grid(f, np.array([8, 8, 8]))
    if kwargs.get("return_expected", False):
        expected = [1, 4, 9, 2]
        return gb, expected
    else:
        return gb
