# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 20:55:56 2016

@author: keile
"""

import numpy as np

from porepy.grids import structured
from porepy.params import tensor, bc
from porepy.numerics.fv import tpfa


def test_tpfa_cart_2d():
    """ Apply TPFA on Cartesian grid, should obtain Laplacian stencil. """

    # Set up 3 X 3 Cartesian grid
    nx = np.array([3, 3])
    g = structured.CartGrid(nx)
    g.compute_geometry()

    kxx = np.ones(g.num_cells)
    perm = tensor.SecondOrder(g.dim, kxx)

    bound_faces = np.array([0, 3, 12])
    bound = bc.BoundaryCondition(g, bound_faces, ['dir'] * bound_faces.size)

    trm, bound_flux = tpfa.tpfa(g, perm, bound)
    div = g.cell_faces.T
    a = div * trm
    b = (div * bound_flux).A
    print(b)
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
    assert b[2, 14] == 1
    # Cell 2 has one Neumann face
    assert a[1, 1] == 3
    assert a[1, 0] == -1
    assert a[1, 2] == -1
    assert a[1, 4] == -1

    assert b[1, 13] == 1

    return a


def test_uniform_flow_cart_2d():
    nx = np.array([13, 13])
    g = structured.CartGrid(nx)
    g.compute_geometry()

    kxx = np.ones(g.num_cells)
    perm = tensor.SecondOrder(g.dim, kxx)
    bound_faces = np.argwhere(np.abs(g.cell_faces).sum(axis=1).A.ravel('F') == 1)
    bound = bc.BoundaryCondition(g, bound_faces, ['dir'] * bound_faces.size)

    flux = tpfa.tpfa(g, perm, bound)


if __name__ == '__main__':
    test_tpfa_cart_2d()
