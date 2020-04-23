# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 20:55:56 2016

@author: keile
"""

import numpy as np
import unittest
import porepy as pp


def _assign_params(g, perm, bc):
    data = pp.initialize_parameters(
        {}, g, "flow", {"bc": bc, "second_order_tensor": perm}
    )
    return data


class TestTPFA(unittest.TestCase):
    def test_tpfa_cart_2d(self):
        """ Apply TPFA on Cartesian grid, should obtain Laplacian stencil. """

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
        discr = pp.Tpfa(key)

        discr.discretize(g, d)
        matrix_dictionary = d[pp.DISCRETIZATION_MATRICES][key]
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

    def test_tpfa_cart_2d_periodic(self):
        """ Apply TPFA on a periodic Cartesian grid, should obtain Laplacian stencil. """

        # Set up 3 X 3 Cartesian grid
        nx = np.array([3, 3])
        g = pp.CartGrid(nx)
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(kxx)

        bound_faces = np.array([0, 3, 4, 7, 8, 11, 12, 13, 14, 21, 22, 23])
        bound = pp.BoundaryCondition(g, bound_faces, "per")
        left_faces = [0, 4, 8, 12, 13, 14]
        right_faces = [3, 7, 11, 21, 22, 23]
        per_map = np.vstack((left_faces, right_faces))
        bound.set_periodic_map(per_map)

        key = "flow"
        d = pp.initialize_default_data(
            g, {}, key, {"second_order_tensor": perm, "bc": bound}
        )
        discr = pp.Tpfa(key)

        discr.discretize(g, d)
        matrix_dictionary = d[pp.DISCRETIZATION_MATRICES][key]
        trm, bound_flux = matrix_dictionary["flux"], matrix_dictionary["bound_flux"]
        div = g.cell_faces.T
        a = div * trm
        b = -(div * bound_flux).A

        # Create laplace matrix
        A_lap = np.array([
            [ 4., -1., -1., -1.,  0.,  0., -1.,  0.,  0.],
            [-1.,  4., -1.,  0., -1.,  0.,  0., -1.,  0.],
            [-1., -1.,  4.,  0.,  0., -1.,  0.,  0., -1.],
            [-1.,  0.,  0.,  4., -1., -1., -1.,  0.,  0.],
            [ 0., -1.,  0., -1.,  4., -1.,  0., -1.,  0.],
            [ 0.,  0., -1., -1., -1.,  4.,  0.,  0., -1.],
            [-1.,  0.,  0., -1.,  0.,  0.,  4., -1., -1.],
            [ 0., -1.,  0.,  0., -1.,  0., -1.,  4., -1.],
            [ 0.,  0., -1.,  0.,  0., -1., -1., -1.,  4.],
        ])

        self.assertTrue(np.allclose(a.A, A_lap))
        self.assertTrue(np.allclose(b, 0))
        return a
    

if __name__ == "__main__":
    unittest.main()
