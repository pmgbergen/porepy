# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:36:14 2016

@author: eke001
"""
from __future__ import division

import random
import unittest

import numpy as np

import porepy as pp


def setup_cart_2d(nx):

    g = pp.CartGrid(nx)
    g.compute_geometry()

    kxx = np.ones(g.num_cells)
    perm = pp.SecondOrderTensor(kxx)

    return g, perm


def setup_random_pressure_field(g):
    gx = random.random()
    gy = random.random()
    xf = g.face_centers
    xc = g.cell_centers

    pr_bound = gx * xf[0] + gy * xf[1]
    pr_cell = gx * xc[0] + gy * xc[1]
    return pr_bound, pr_cell, gx, gy


class TestMpfa(unittest.TestCase):
    def test_laplacian_stencil_cart_2d(self):
        """ Apply MPFA on Cartesian grid, should obtain Laplacian stencil. """

        # Set up 3 X 3 Cartesian grid
        g, perm = setup_cart_2d(np.array([3, 3]))

        bnd_faces = np.array([0, 3, 12])
        bound = pp.BoundaryCondition(g, bnd_faces, ["dir"] * bnd_faces.size)

        # Python inverter is most efficient for small problems
        flux, bound_flux, *_ = pp.Mpfa("flow")._flux_discretization(
            g, perm, bound, inverter="python"
        )
        div = g.cell_faces.T
        A = div * flux

        # Checks on interior cell
        mid = 4
        self.assertTrue(A[mid, mid] == 4)
        self.assertTrue(A[mid - 1, mid] == -1)
        self.assertTrue(A[mid + 1, mid] == -1)
        self.assertTrue(A[mid - 3, mid] == -1)
        self.assertTrue(A[mid + 3, mid] == -1)

        # The first cell should have two Dirichlet bnds
        self.assertTrue(A[0, 0] == 6)
        self.assertTrue(A[0, 1] == -1)
        self.assertTrue(A[0, 3] == -1)

        # Cell 3 has one Dirichlet, one Neumann face
        self.assertTrue(A[2, 2] == 4)
        self.assertTrue(A[2, 1] == -1)
        self.assertTrue(A[2, 5] == -1)

        # Cell 2 has one Neumann face
        self.assertTrue(A[1, 1] == 3)
        self.assertTrue(A[1, 0] == -1)
        self.assertTrue(A[1, 2] == -1)
        self.assertTrue(A[1, 4] == -1)

        return A

    def test_uniform_flow_cart_2d(self):
        # Structured Cartesian grid
        g, perm = setup_cart_2d(np.array([10, 10]))
        bound_faces = np.argwhere(np.abs(g.cell_faces).sum(axis=1).A.ravel("F") == 1)
        bound = pp.BoundaryCondition(
            g, bound_faces.ravel("F"), ["dir"] * bound_faces.size
        )

        # Python inverter is most efficient for small problems
        flux, bound_flux, *_ = pp.Mpfa("flow")._flux_discretization(
            g, perm, bound, inverter="python"
        )
        div = g.cell_faces.T

        a = div * flux

        pr_bound, pr_cell, gx, gy = setup_random_pressure_field(g)

        rhs = div * bound_flux * pr_bound
        pr = np.linalg.solve(a.todense(), -rhs)

        p_diff = pr - pr_cell
        self.assertTrue(np.max(np.abs(p_diff)) < 1e-8)

    def test_uniform_flow_cart_2d_structured_pert(self):
        g, perm = setup_cart_2d(np.array([2, 2]))
        g.nodes[0, 4] = 1.5
        g.compute_geometry()

        bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        bound = pp.BoundaryCondition(
            g, bound_faces.ravel("F"), ["dir"] * bound_faces.size
        )

        # Python inverter is most efficient for small problems
        flux, bound_flux, *_ = pp.Mpfa("flow")._flux_discretization(
            g, perm, bound, inverter="python"
        )
        div = g.cell_faces.T

        a = div * flux

        xf = np.zeros_like(g.face_centers)
        xf[:, bound_faces.ravel()] = g.face_centers[:, bound_faces.ravel()]
        xc = g.cell_centers
        pr_bound = xf.sum(axis=0)
        pr_cell = xc.sum(axis=0)

        rhs = div * bound_flux * pr_bound
        pr = np.linalg.solve(a.todense(), -rhs)

        p_diff = pr - pr_cell
        self.assertTrue(np.max(np.abs(p_diff)) < 1e-8)

    def test_uniform_flow_cart_2d_pert(self):
        # Randomly perturbed grid, with random linear pressure field
        g, perm = setup_cart_2d(np.array([10, 10]))
        dx = 1
        pert = 0.4
        g.nodes = g.nodes + dx * pert * (
            0.5 - np.random.rand(g.nodes.shape[0], g.num_nodes)
        )
        # Cancel perturbations in z-coordinate.
        g.nodes[2, :] = 0
        g.compute_geometry()

        bound_faces = np.argwhere(np.abs(g.cell_faces).sum(axis=1).A.ravel("F") == 1)
        bound = pp.BoundaryCondition(
            g, bound_faces.ravel("F"), ["dir"] * bound_faces.size
        )

        # Python inverter is most efficient for small problems
        flux, bound_flux, *_ = pp.Mpfa("flow")._flux_discretization(
            g, perm, bound, inverter="python"
        )
        div = g.cell_faces.T

        a = div * flux

        pr_bound, pr_cell, gx, gy = setup_random_pressure_field(g)

        rhs = div * bound_flux * pr_bound
        pr = np.linalg.solve(a.todense(), -rhs)

        p_diff = pr - pr_cell
        self.assertTrue(np.max(np.abs(p_diff)) < 1e-8)


if __name__ == "__main__":
    unittest.main()
