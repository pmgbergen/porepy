# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 20:55:56 2016

@author: keile
"""

import numpy as np
import unittest
import porepy as pp


def _assign_params(g, perm, bc):
    data = pp.initialize_parameters({}, g, "flow", {"bc": bc, "permeability": perm})
    return data


class TestTPFA(unittest.TestCase):
    def test_tpfa_cart_2d(self):
        """ Apply TPFA on Cartesian grid, should obtain Laplacian stencil. """

        # Set up 3 X 3 Cartesian grid
        nx = np.array([3, 3])
        g = pp.CartGrid(nx)
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(g.dim, kxx)

        bound_faces = np.array([0, 3, 12])
        bound = pp.BoundaryCondition(g, bound_faces, ["dir"] * bound_faces.size)

        key = "flow"
        d = pp.initialize_data({}, g, key, {"permeability": perm, "bc": bound})
        discr = pp.Tpfa(key)

        discr.discretize(g, d)
        matrix_dictionary = d[pp.keywords.DISCRETIZATION_MATRICES][key]
        trm, bound_flux = matrix_dictionary["flux"], matrix_dictionary["bound_flux"]
        div = g.cell_faces.T
        a = div * trm
        b = -(div * bound_flux).A

        # Checks on interior cell
        mid = 4
        self.assertTrue(a[mid, mid] == 4)
        self.assertTrue(a[mid - 1, mid] == -1)
        self.assertTrue(a[mid + 1, mid] == -1)
        self.assertTrue(a[mid - 3, mid] == -1)
        self.assertTrue(a[mid + 3, mid] == -1)

        self.assertTrue(np.all(b[mid, :] == 0))

        # The first cell should have two Dirichlet bnds
        self.assertTrue(a[0, 0] == 6)
        self.assertTrue(a[0, 1] == -1)
        self.assertTrue(a[0, 3] == -1)

        self.assertTrue(b[0, 0] == 2)
        self.assertTrue(b[0, 12] == 2)

        # Cell 3 has one Dirichlet, one Neumann face
        self.assertTrue(a[2, 2] == 4)
        self.assertTrue(a[2, 1] == -1)
        self.assertTrue(a[2, 5] == -1)

        self.assertTrue(b[2, 3] == 2)
        self.assertTrue(b[2, 14] == -1)
        # Cell 2 has one Neumann face
        self.assertTrue(a[1, 1] == 3)
        self.assertTrue(a[1, 0] == -1)
        self.assertTrue(a[1, 2] == -1)
        self.assertTrue(a[1, 4] == -1)

        self.assertTrue(b[1, 13] == -1)

        return a


if __name__ == "__main__":
    unittest.main()
