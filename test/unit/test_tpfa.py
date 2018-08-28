# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 20:55:56 2016

@author: keile
"""

import numpy as np

from porepy.grids import structured
from porepy.params import tensor, bc, data
from porepy.numerics.fv import tpfa


def _assign_params(g, perm, bound):
    params = data.Parameters(g)
    params.set_tensor("Flow", perm)
    params.set_bc("Flow", bound)
    d = {"param": params}
    return d


def test_tpfa_cart_2d():
    """ Apply TPFA on Cartesian grid, should obtain Laplacian stencil. """

    # Set up 3 X 3 Cartesian grid
    nx = np.array([3, 3])
    g = structured.CartGrid(nx)
    g.compute_geometry()

    kxx = np.ones(g.num_cells)
    perm = tensor.SecondOrderTensor(g.dim, kxx)

    bound_faces = np.array([0, 3, 12])
    bound = bc.BoundaryCondition(g, bound_faces, ["dir"] * bound_faces.size)

    discr = tpfa.Tpfa()
    d = _assign_params(g, perm, bound)
    discr.discretize(g, d)
    trm, bound_flux = d["flux"], d["bound_flux"]
    div = g.cell_faces.T
    a = div * trm
    b = -(div * bound_flux).A

    # Checks on interior cell
    mid = 4
    assert a[mid, mid] == 4
    assert a[mid - 1, mid] == -1
    assert a[mid + 1, mid] == -1
    assert a[mid - 3, mid] == -1
    assert a[mid + 3, mid] == -1

    assert np.all(b[mid, :] == 0)

    # The first cell should have two Dirichlet bnds
    assert a[0, 0] == 6
    assert a[0, 1] == -1
    assert a[0, 3] == -1

    assert b[0, 0] == 2
    assert b[0, 12] == 2

    # Cell 3 has one Dirichlet, one Neumann face
    assert a[2, 2] == 4
    assert a[2, 1] == -1
    assert a[2, 5] == -1

    assert b[2, 3] == 2
    assert b[2, 14] == -1
    # Cell 2 has one Neumann face
    assert a[1, 1] == 3
    assert a[1, 0] == -1
    assert a[1, 2] == -1
    assert a[1, 4] == -1

    assert b[1, 13] == -1

    return a


def test_uniform_flow_cart_2d():
    nx = np.array([13, 13])
    g = structured.CartGrid(nx)
    g.compute_geometry()

    kxx = np.ones(g.num_cells)
    perm = tensor.SecondOrderTensor(g.dim, kxx)
    bound_faces = np.argwhere(np.abs(g.cell_faces).sum(axis=1).A.ravel("F") == 1)
    bound = bc.BoundaryCondition(g, bound_faces, ["dir"] * bound_faces.size)

    discr = tpfa.Tpfa()
    d = _assign_params(g, perm, bound)
    discr.discretize(g, d)
    flux, bound_flux = d["flux"], d["bound_flux"]


if __name__ == "__main__":
    test_tpfa_cart_2d()
