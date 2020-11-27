# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 20:55:56 2016

@author: keile
"""

import numpy as np
import pytest

import porepy as pp


def _assign_params(g, perm, bc):
    data = pp.initialize_parameters(
        {}, g, "flow", {"bc": bc, "second_order_tensor": perm}
    )
    return data


@pytest.mark.parametrize("method", ["tpfa", "mpfa"])
def test_fv_cart_2d(method):
    """ Apply TPFA and MPFA on Cartesian grid, should obtain Laplacian stencil."""

    # Set up 3 X 3 Cartesian grid
    nx = np.array([3, 3])
    g = pp.CartGrid(nx)
    g.compute_geometry()

    kxx = np.ones(g.num_cells)
    perm = pp.SecondOrderTensor(kxx)

    bound_faces = np.array([0, 3, 12])
    bound = pp.BoundaryCondition(g, bound_faces, ["dir"] * bound_faces.size)

    key = "flow"
    d = pp.initialize_default_data(
        g, {}, key, {"second_order_tensor": perm, "bc": bound}
    )
    if method == "tpfa":
        discr = pp.Tpfa(key)
    elif method == "mpfa":
        discr = pp.Mpfa(key)
    else:
        assert False

    discr.discretize(g, d)
    matrix_dictionary = d[pp.DISCRETIZATION_MATRICES][key]
    trm, bound_flux = matrix_dictionary["flux"], matrix_dictionary["bound_flux"]
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


@pytest.mark.parametrize("method", ["tpfa", "mpfa"])
def test_fv_cart_2d_periodic(method):
    """ Apply TPFA and MPFA on a periodic Cartesian grid, should obtain Laplacian stencil."""

    # Set up 3 X 3 Cartesian grid
    nx = np.array([3, 3])
    g = pp.CartGrid(nx)
    g.compute_geometry()

    kxx = np.ones(g.num_cells)
    perm = pp.SecondOrderTensor(kxx)

    left_faces = [0, 4, 8, 12, 13, 14]
    right_faces = [3, 7, 11, 21, 22, 23]
    periodic_face_map = np.vstack((left_faces, right_faces))
    g.set_periodic_map(periodic_face_map)

    bound = pp.BoundaryCondition(g)

    key = "flow"
    d = pp.initialize_default_data(
        g, {}, key, {"second_order_tensor": perm, "bc": bound}
    )

    if method == "tpfa":
        discr = pp.Tpfa(key)
    elif method == "mpfa":
        discr = pp.Mpfa(key)
    else:
        assert False

    discr.discretize(g, d)
    matrix_dictionary = d[pp.DISCRETIZATION_MATRICES][key]
    trm, bound_flux = matrix_dictionary["flux"], matrix_dictionary["bound_flux"]
    div = g.cell_faces.T
    a = div * trm
    b = -(div * bound_flux).A

    # Create laplace matrix
    A_lap = np.array(
        [
            [4.0, -1.0, -1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
            [-1.0, 4.0, -1.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0],
            [-1.0, -1.0, 4.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0, 4.0, -1.0, -1.0, -1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0, -1.0, -1.0, 4.0, 0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 4.0, -1.0, -1.0],
            [0.0, -1.0, 0.0, 0.0, -1.0, 0.0, -1.0, 4.0, -1.0],
            [0.0, 0.0, -1.0, 0.0, 0.0, -1.0, -1.0, -1.0, 4.0],
        ]
    )

    assert np.allclose(a.A, A_lap)
    assert np.allclose(b, 0)
    return a


if __name__ == "__main__":
    unittest.main()
