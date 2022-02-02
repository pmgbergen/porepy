# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:36:14 2016

"""
from __future__ import division

import random

import numpy as np
import pytest

import porepy as pp


def setup_cart_2d(nx, physdims):

    g = pp.CartGrid(nx, physdims)
    g.compute_geometry()
    kxx = 0.5 * np.ones(g.num_cells)
    return g, kxx


def setup_periodic_pressure_field(g, kxx):
    xf = g.face_centers
    xc = g.cell_centers

    pr_bound = np.sin(2 * np.pi * xf[0]) + np.cos(2 * np.pi * xf[1])
    pr_cell = np.sin(2 * np.pi * xc[0]) + np.cos(2 * np.pi * xc[1])

    src = kxx * (2 * np.pi) ** 2 * pr_cell
    return pr_bound, pr_cell, src


@pytest.mark.parametrize("method", ["tpfa", "mpfa"])
def test_periodic_pressure_field_2d(method):
    """
    Test that TPFA approximate an analytical periodic solution by imposing
    periodic boundary conditions to the bottom and top faces of the unit square.
    """
    # Structured Cartesian grid
    g, kxx = setup_cart_2d(np.array([10, 10]), [1, 1])

    bot_faces = np.argwhere(g.face_centers[1] < 1e-5).ravel()
    top_faces = np.argwhere(g.face_centers[1] > 1 - 1e-5).ravel()

    left_faces = np.argwhere(g.face_centers[0] < 1e-5).ravel()
    right_faces = np.argwhere(g.face_centers[0] > 1 - 1e-5).ravel()

    dir_faces = np.hstack((left_faces, right_faces))

    g.set_periodic_map(np.vstack((bot_faces, top_faces)))

    bound = pp.BoundaryCondition(g, dir_faces, "dir")

    # Solve
    key = "flow"
    d = pp.initialize_default_data(
        g, {}, key, {"second_order_tensor": pp.SecondOrderTensor(kxx), "bc": bound}
    )
    if method == "tpfa":
        discr = pp.Tpfa(key)
    elif method == "mpfa":
        discr = pp.Mpfa(key)
    else:
        assert False
    discr.discretize(g, d)
    matrix_dictionary = d[pp.DISCRETIZATION_MATRICES][key]
    flux, bound_flux = matrix_dictionary["flux"], matrix_dictionary["bound_flux"]

    div = g.cell_faces.T

    a = div * flux

    pr_bound, pr_cell, src = setup_periodic_pressure_field(g, kxx)

    rhs = -div * bound_flux * pr_bound + src * g.cell_volumes
    pr = np.linalg.solve(a.todense(), rhs)

    p_diff = pr - pr_cell

    assert np.max(np.abs(p_diff)) < 0.06


@pytest.mark.parametrize("method", ["tpfa", "mpfa"])
def test_symmetry_periodic_pressure_field_2d(method):
    """
    Test that we obtain a symmetric solution accross the periodic boundary.
    The test consider the unit square with periodic boundary conditions
    on the top and bottom boundary. A source is added to the bottom row of
    cells and we test that the solution is periodic.
    Setup, with x denoting the source:
           --------
          |       |
    p = 0 |       | p = 0
          |   x   |
           -------
    """
    # Structured Cartesian grid
    g, kxx = setup_cart_2d(np.array([5, 5]), [1, 1])

    bot_faces = np.argwhere(g.face_centers[1] < 1e-5).ravel()
    top_faces = np.argwhere(g.face_centers[1] > 1 - 1e-5).ravel()

    left_faces = np.argwhere(g.face_centers[0] < 1e-5).ravel()
    right_faces = np.argwhere(g.face_centers[0] > 1 - 1e-5).ravel()

    dir_faces = np.hstack((left_faces, right_faces))

    g.set_periodic_map(np.vstack((bot_faces, top_faces)))

    bound = pp.BoundaryCondition(g, dir_faces, "dir")

    # Solve
    key = "flow"
    d = pp.initialize_default_data(
        g, {}, key, {"second_order_tensor": pp.SecondOrderTensor(kxx), "bc": bound}
    )
    if method == "tpfa":
        discr = pp.Tpfa(key)
    elif method == "mpfa":
        discr = pp.Mpfa(key)
    else:
        assert False

    discr.discretize(g, d)
    matrix_dictionary = d[pp.DISCRETIZATION_MATRICES][key]
    flux, bound_flux = matrix_dictionary["flux"], matrix_dictionary["bound_flux"]

    div = g.cell_faces.T

    a = div * flux

    pr_bound = np.zeros(g.num_faces)
    src = np.zeros(g.num_cells)
    src[2] = 1

    rhs = -div * bound_flux * pr_bound + src
    pr = np.linalg.solve(a.todense(), rhs)

    p_diff = pr[5:15] - np.hstack((pr[-5:], pr[-10:-5]))
    assert np.max(np.abs(p_diff)) < 1e-10


if __name__ == "__main__":
    unittest.main()
