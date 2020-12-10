#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for convergence of the mixed vem. Compare against known rates.
"""
import unittest

import numpy as np
import scipy.sparse as sps

import porepy as pp


class TestVEMVaryingPerm(unittest.TestCase):
    """Test known convergence rate for 2d domain with varying permeability"""

    def rhs(self, x, y, z):
        return (
            8.0
            * np.pi ** 2
            * np.sin(2.0 * np.pi * x)
            * np.sin(2.0 * np.pi * y)
            * self.permeability(x, y, z)
            - 400.0 * np.pi * y * np.cos(2.0 * np.pi * y) * np.sin(2.0 * np.pi * x)
            - 400.0 * np.pi * x * np.cos(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
        )

    def solution(self, x, y, z):
        return np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)

    def permeability(self, x, y, z):
        return 1 + 100.0 * x ** 2 + 100.0 * y ** 2

    def add_data(self, g):
        """
        Define the permeability, apertures, boundary conditions
        """
        # Permeability
        kxx = np.array([self.permeability(*pt) for pt in g.cell_centers.T])
        perm = pp.SecondOrderTensor(kxx)

        # Source term
        source = g.cell_volumes * np.array([self.rhs(*pt) for pt in g.cell_centers.T])

        # Boundaries
        bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        bound_face_centers = g.face_centers[:, bound_faces]

        labels = np.array(["dir"] * bound_faces.size)

        bc_val = np.zeros(g.num_faces)
        bc_val[bound_faces] = np.array(
            [self.solution(*pt) for pt in bound_face_centers.T]
        )

        bound = pp.BoundaryCondition(g, bound_faces, labels)
        specified_parameters = {
            "second_order_tensor": perm,
            "source": source,
            "bc": bound,
            "bc_values": bc_val,
        }
        return pp.initialize_default_data(g, {}, "flow", specified_parameters)

    def error_p(self, g, p):

        sol = np.array([self.solution(*pt) for pt in g.cell_centers.T])
        return np.sqrt(np.sum(np.power(np.abs(p - sol), 2) * g.cell_volumes))

    def main(self, N):
        Nx = Ny = N

        # g = structured.CartGrid([Nx, Ny], [2, 2])
        g = pp.StructuredTriangleGrid([Nx, Ny], [1, 1])
        g.compute_geometry()
        # co.coarsen(g, 'by_volume')

        # Assign parameters
        data = self.add_data(g)

        # Choose and define the solvers
        solver_flow = pp.MVEM("flow")
        solver_flow.discretize(g, data)
        A_flow, b_flow = solver_flow.assemble_matrix_rhs(g, data)

        solver_source = pp.DualScalarSource("flow")
        solver_source.discretize(g, data)
        A_source, b_source = solver_source.assemble_matrix_rhs(g, data)

        up = sps.linalg.spsolve(A_flow + A_source, b_flow + b_source)

        u = solver_flow.extract_flux(g, up, data)
        p = solver_flow.extract_pressure(g, up, data)
        #    P0u = solver_flow.project_flux(g, u, data, keyword="flow")

        diam = np.amax(g.cell_diameters())
        return diam, self.error_p(g, p)

    def test_vem_varying_k(self):
        diam_10, error_10 = self.main(10)
        diam_20, error_20 = self.main(20)

        known_order = 2.00266229752
        order = np.log(error_10 / error_20) / np.log(diam_10 / diam_20)
        assert np.isclose(order, known_order)


class TestVEMVaryingPermTiltedGrid(unittest.TestCase):
    def rhs(self, x, y, z):
        return (
            7.0 * z * (x ** 2 + y ** 2 + 1.0)
            - y * (x ** 2 - 9.0 * z ** 2)
            - 4.0 * x ** 2 * z
            - (
                8.0 * np.sin(np.pi * y)
                - 4.0 * np.pi ** 2 * y ** 2 * np.sin(np.pi * y)
                + 16.0 * np.pi * y * np.cos(np.pi * y)
            )
            * (x ** 2 / 2.0 + y ** 2 / 2.0 + 1.0 / 2.0)
            - 4.0 * y ** 2 * (2.0 * np.sin(np.pi * y) + np.pi * y * np.cos(np.pi * y))
        )

    def solution(self, x, y, z):
        return x ** 2 * z + 4.0 * y ** 2 * np.sin(np.pi * y) - 3.0 * z ** 3

    def permeability(self, x, y, z):
        return 1.0 + x ** 2 + y ** 2

    def add_data(self, g):
        """
        Define the permeability, apertures, boundary conditions
        """
        # Permeability
        kxx = np.array([self.permeability(*pt) for pt in g.cell_centers.T])
        perm = pp.SecondOrderTensor(kxx)

        # Source term
        source = g.cell_volumes * np.array([self.rhs(*pt) for pt in g.cell_centers.T])

        # Boundariesy
        bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        bound_face_centers = g.face_centers[:, bound_faces]

        labels = np.array(["dir"] * bound_faces.size)

        bc_val = np.zeros(g.num_faces)
        bc_val[bound_faces] = np.array(
            [self.solution(*pt) for pt in bound_face_centers.T]
        )

        bound = pp.BoundaryCondition(g, bound_faces, labels)
        specified_parameters = {
            "second_order_tensor": perm,
            "source": source,
            "bc": bound,
            "bc_values": bc_val,
        }
        return pp.initialize_default_data(g, {}, "flow", specified_parameters)

    def error_p(self, g, p):

        sol = np.array([self.solution(*pt) for pt in g.cell_centers.T])
        return np.sqrt(np.sum(np.power(np.abs(p - sol), 2) * g.cell_volumes))

    def main(self, N):
        Nx = Ny = N
        # g = structured.CartGrid([Nx, Ny], [1, 1])
        g = pp.StructuredTriangleGrid([Nx, Ny], [1, 1])
        R = pp.map_geometry.rotation_matrix(np.pi / 4.0, [1, 0, 0])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()
        # co.coarsen(g, 'by_volume')

        # Assign parameters
        data = self.add_data(g)

        # Choose and define the solvers
        solver_flow = pp.MVEM("flow")
        solver_flow.discretize(g, data)
        A_flow, b_flow = solver_flow.assemble_matrix_rhs(g, data)

        solver_source = pp.DualScalarSource("flow")
        solver_source.discretize(g, data)
        A_source, b_source = solver_source.assemble_matrix_rhs(g, data)

        up = sps.linalg.spsolve(A_flow + A_source, b_flow + b_source)

        u = solver_flow.extract_flux(g, up, data)
        p = solver_flow.extract_pressure(g, up, data)
        #    P0u = solver_flow.project_flux(g, u, data)

        diam = np.amax(g.cell_diameters())
        return diam, self.error_p(g, p)

    def test_vem_varing_k_surface(self):
        diam_10, error_10 = self.main(10)
        diam_20, error_20 = self.main(20)

        known_order = 1.97928213116
        order = np.log(error_10 / error_20) / np.log(diam_10 / diam_20)
        assert np.isclose(order, known_order)


class TestVEMVaryingPermSurface_2(unittest.TestCase):
    def rhs(self, x, y, z):
        return 8.0 * z * (125.0 * x ** 2 + 200.0 * y ** 2 + 425.0 * z ** 2 + 2.0)

    def solution(self, x, y, z):
        return x ** 2 * z + 4.0 * y ** 2 * np.sin(np.pi * y) - 3.0 * z ** 3

    def permeability(self, x, y, z):
        return 1.0 + 100.0 * (x ** 2 + y ** 2 + z ** 2.0)

    def add_data(self, g):
        """
        Define the permeability, apertures, boundary conditions
        """
        # Permeability
        kxx = np.array([self.permeability(*pt) for pt in g.cell_centers.T])
        perm = pp.SecondOrderTensor(kxx)

        # Source term
        source = g.cell_volumes * np.array([self.rhs(*pt) for pt in g.cell_centers.T])

        # Boundaries
        bound_faces = g.get_all_boundary_faces()
        bound_face_centers = g.face_centers[:, bound_faces]

        labels = np.array(["dir"] * bound_faces.size)

        bc_val = np.zeros(g.num_faces)
        bc_val[bound_faces] = np.array(
            [self.solution(*pt) for pt in bound_face_centers.T]
        )

        bound = pp.BoundaryCondition(g, bound_faces, labels)
        specified_parameters = {
            "second_order_tensor": perm,
            "source": source,
            "bc": bound,
            "bc_values": bc_val,
        }
        return pp.initialize_default_data(g, {}, "flow", specified_parameters)

    def error_p(self, g, p):

        sol = np.array([self.solution(*pt) for pt in g.cell_centers.T])
        return np.sqrt(np.sum(np.power(np.abs(p - sol), 2) * g.cell_volumes))

    def main(self, N):
        Nx = Ny = N
        # g = structured.CartGrid([Nx, Ny], [1, 1])
        g = pp.StructuredTriangleGrid([Nx, Ny], [1, 1])
        R = pp.map_geometry.rotation_matrix(np.pi / 2.0, [1, 0, 0])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()
        # co.coarsen(g, 'by_volume')

        # Assign parameters
        data = self.add_data(g)

        # Choose and define the solvers
        solver_flow = pp.MVEM("flow")
        solver_flow.discretize(g, data)
        A_flow, b_flow = solver_flow.assemble_matrix_rhs(g, data)

        solver_source = pp.DualScalarSource("flow")
        solver_source.discretize(g, data)
        A_source, b_source = solver_source.assemble_matrix_rhs(g, data)

        up = sps.linalg.spsolve(A_flow + A_source, b_flow + b_source)

        u = solver_flow.extract_flux(g, up, data)
        p = solver_flow.extract_pressure(g, up, data)
        P0u = solver_flow.project_flux(g, u, data)

        diam = np.amax(g.cell_diameters())
        return diam, self.error_p(g, p)

    def test_vem_varing_k_surface(self):
        diam_10, error_10 = self.main(10)
        diam_20, error_20 = self.main(20)

        known_order = 1.9890160655
        order = np.log(error_10 / error_20) / np.log(diam_10 / diam_20)
        assert np.isclose(order, known_order)


if __name__ == "__main__":
    unittest.main()
