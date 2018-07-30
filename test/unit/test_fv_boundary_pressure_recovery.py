#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:22:41 2017

@author: Eirik Keilegavlens
"""

import numpy as np
import unittest
import scipy.sparse.linalg as spl
import scipy.sparse as sps

import porepy as pp

from porepy.grids.structured import CartGrid
from porepy.grids.simplex import StructuredTriangleGrid
from porepy.grids.grid import Grid
from porepy.params import bc, tensor
from porepy.params.data import Parameters
from porepy.numerics.fv.tpfa import Tpfa
from porepy.numerics.fv.mpfa import Mpfa
from porepy.numerics.fv import fvutils


class TestTpfaBoundaryPressure(unittest.TestCase):
    def grid(self, nx=[2, 2], physdims=None):
        if physdims is None:
            physdims = nx
        g = CartGrid(nx, physdims)
        g.compute_geometry()
        return g

    def simplex_grid(self, nx=[2, 2]):
        g = StructuredTriangleGrid(nx)
        g.compute_geometry()
        return g

    def test_zero_pressure(self):
        g = self.grid()
        param = Parameters(g)

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["dir"]
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Tpfa()
        param.set_bc(fd, bound)
        param.set_bc_val(fd, np.zeros(g.num_faces))
        data = {"param": param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        assert np.allclose(p, np.zeros_like(p))

        bound_p = data["bound_pressure_cell"] * p + data[
            "bound_pressure_face"
        ] * np.zeros(g.num_faces)
        assert np.allclose(bound_p, np.zeros_like(bound_p))

    def test_constant_pressure(self):
        g = self.grid()
        param = Parameters(g)

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["dir"]
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Tpfa()
        param.set_bc(fd, bound)

        bc_val = np.ones(g.num_faces)
        param.set_bc_val(fd, bc_val)
        data = {"param": param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        assert np.allclose(p, np.ones_like(p))

        bound_p = data["bound_pressure_cell"] * p + data["bound_pressure_face"] * bc_val
        assert np.allclose(bound_p[bf], bc_val[bf])

    def test_constant_pressure_simplex_grid(self):
        g = self.simplex_grid()
        param = Parameters(g)

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["dir"]
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Tpfa()
        param.set_bc(fd, bound)

        bc_val = np.ones(g.num_faces)
        param.set_bc_val(fd, bc_val)
        data = {"param": param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        assert np.allclose(p, np.ones_like(p))

        bound_p = data["bound_pressure_cell"] * p + data["bound_pressure_face"] * bc_val
        assert np.allclose(bound_p[bf], bc_val[bf])

    def test_linear_pressure_dirichlet_conditions(self):
        g = self.grid()
        param = Parameters(g)

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["dir"]
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Tpfa()
        param.set_bc(fd, bound)

        bc_val = 1 * g.face_centers[0] + 2 * g.face_centers[1]
        param.set_bc_val(fd, bc_val)
        data = {"param": param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        bound_p = data["bound_pressure_cell"] * p + data["bound_pressure_face"] * bc_val
        assert np.allclose(bound_p[bf], bc_val[bf])

    def test_linear_pressure_part_neumann_conditions(self):
        g = self.grid()
        param = Parameters(g)

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["neu"]
        bc_type[0] = "dir"
        bc_type[2] = "dir"
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Tpfa()
        param.set_bc(fd, bound)

        bc_val = np.zeros(g.num_faces)
        # Set up unit pressure gradient in x-direction
        bc_val[[2, 5]] = 1
        param.set_bc_val(fd, bc_val)
        data = {"param": param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        bound_p = data["bound_pressure_cell"] * p + data["bound_pressure_face"] * bc_val
        assert np.allclose(bound_p[bf], -g.face_centers[0, bf])

    def test_linear_pressure_part_neumann_conditions_small_domain(self):
        g = self.grid(physdims=[1, 1])
        param = Parameters(g)

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["neu"]
        bc_type[0] = "dir"
        bc_type[2] = "dir"
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Tpfa()
        param.set_bc(fd, bound)

        bc_val = np.zeros(g.num_faces)
        # Set up unit pressure gradient in x-direction
        bc_val[[2, 5]] = 1
        param.set_bc_val(fd, bc_val)
        data = {"param": param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        bound_p = data["bound_pressure_cell"] * p + data["bound_pressure_face"] * bc_val
        assert np.allclose(bound_p[bf], -2 * g.face_centers[0, bf])

    def test_linear_pressure_part_neumann_conditions_reverse_sign(self):
        g = self.grid()
        param = Parameters(g)

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["neu"]
        bc_type[0] = "dir"
        bc_type[2] = "dir"
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Tpfa()
        param.set_bc(fd, bound)

        bc_val = np.zeros(g.num_faces)
        # Set up pressure gradient in x-direction, with value -1
        bc_val[[2, 5]] = -1
        param.set_bc_val(fd, bc_val)
        data = {"param": param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        bound_p = data["bound_pressure_cell"] * p + data["bound_pressure_face"] * bc_val
        assert np.allclose(bound_p[bf], g.face_centers[0, bf])

    def test_linear_pressure_part_neumann_conditions_smaller_domain(self):
        # Smaller domain, check that the smaller pressure gradient is captured
        g = CartGrid([2, 2], physdims=[1, 2])
        g.compute_geometry()
        param = Parameters(g)

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["neu"]
        bc_type[0] = "dir"
        bc_type[2] = "dir"
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Tpfa()
        param.set_bc(fd, bound)

        bc_val = np.zeros(g.num_faces)
        # Set up unit pressure gradient in x-direction
        bc_val[[2, 5]] = 1
        param.set_bc_val(fd, bc_val)
        data = {"param": param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        bound_p = data["bound_pressure_cell"] * p + data["bound_pressure_face"] * bc_val
        assert np.allclose(bound_p[bf], -g.face_centers[0, bf])

    def test_sign_trouble_two_neumann_sides(self):
        g = CartGrid(np.array([2, 2]), physdims=[2, 2])
        g.compute_geometry()
        d = {"param": Parameters(g)}
        bv = np.zeros(g.num_faces)
        bv[[0, 3]] = 1
        bv[[2, 5]] = -1
        d["param"].set_bc_val("flow", bv)
        t = Tpfa("flow")
        A, b = t.matrix_rhs(g, d)
        # The problem is singular, and spsolve does not work well on all systems.
        # Instead, set a consistent solution, and check that the boundary
        # pressure is recovered.
        x = g.cell_centers[0]

        bound_p = d["bound_pressure_cell"] * x + d["bound_pressure_face"] * bv
        assert bound_p[0] == x[0] - 0.5
        assert bound_p[2] == x[1] + 0.5


class TestMpfaBoundaryPressure(unittest.TestCase):
    def grid(self, nx=[2, 2]):
        g = CartGrid(nx)
        g.compute_geometry()
        return g

    def simplex_grid(self, nx=[2, 2]):
        g = StructuredTriangleGrid(nx)
        g.compute_geometry()
        return g

    def test_zero_pressure(self):
        g = self.grid()
        param = Parameters(g)

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["dir"]
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Mpfa()
        param.set_bc(fd, bound)
        param.set_bc_val(fd, np.zeros(g.num_faces))
        data = {"param": param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        assert np.allclose(p, np.zeros_like(p))

        bound_p = data["bound_pressure_cell"] * p + data[
            "bound_pressure_face"
        ] * np.zeros(g.num_faces)
        assert np.allclose(bound_p, np.zeros_like(bound_p))

    def test_constant_pressure(self):
        g = self.grid()
        param = Parameters(g)

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["dir"]
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Mpfa()
        param.set_bc(fd, bound)

        bc_val = np.ones(g.num_faces)
        param.set_bc_val(fd, bc_val)
        data = {"param": param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        assert np.allclose(p, np.ones_like(p))

        bound_p = data["bound_pressure_cell"] * p + data["bound_pressure_face"] * bc_val
        assert np.allclose(bound_p[bf], bc_val[bf])

    def test_constant_pressure_simplex_grid(self):
        g = self.simplex_grid()
        param = Parameters(g)

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["dir"]
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Mpfa()
        param.set_bc(fd, bound)

        bc_val = np.ones(g.num_faces)
        param.set_bc_val(fd, bc_val)
        data = {"param": param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        assert np.allclose(p, np.ones_like(p))

        bound_p = data["bound_pressure_cell"] * p + data["bound_pressure_face"] * bc_val
        assert np.allclose(bound_p[bf], bc_val[bf])

    def test_linear_pressure_dirichlet_conditions(self):
        g = self.grid()
        param = Parameters(g)

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["dir"]
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Mpfa()
        param.set_bc(fd, bound)

        bc_val = 1 * g.face_centers[0] + 2 * g.face_centers[1]
        param.set_bc_val(fd, bc_val)
        data = {"param": param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        bound_p = data["bound_pressure_cell"] * p + data["bound_pressure_face"] * bc_val
        assert np.allclose(bound_p[bf], bc_val[bf])

    def test_linear_pressure_part_neumann_conditions(self):
        g = self.grid()
        param = Parameters(g)

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["neu"]
        # Set Dirichlet conditions. Note that indices are relative to bf,
        # that is, counting only boundary faces.
        bc_type[0] = "dir"
        bc_type[2] = "dir"  # Not [3]
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Tpfa()
        param.set_bc(fd, bound)

        bc_val = np.zeros(g.num_faces)
        # Set up unit flow in x-direction, thus pressure gradient the other way
        bc_val[[2, 5]] = 1
        param.set_bc_val(fd, bc_val)
        data = {"param": param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        bound_p = data["bound_pressure_cell"] * p + data["bound_pressure_face"] * bc_val
        assert np.allclose(bound_p[bf], -g.face_centers[0, bf])

    def test_linear_pressure_part_neumann_conditions_reverse_sign(self):
        g = self.grid()
        param = Parameters(g)

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["neu"]
        bc_type[0] = "dir"
        bc_type[2] = "dir"
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Tpfa()
        param.set_bc(fd, bound)

        bc_val = np.zeros(g.num_faces)
        # Set up flow in negative x-direction, thus positive gradient of value 1
        bc_val[[2, 5]] = -1
        param.set_bc_val(fd, bc_val)
        data = {"param": param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        bound_p = data["bound_pressure_cell"] * p + data["bound_pressure_face"] * bc_val
        assert np.allclose(bound_p[bf], g.face_centers[0, bf])

    def test_linear_pressure_part_neumann_conditions_smaller_domain(self):
        # Smaller domain, check that the smaller pressure gradient is captured
        g = CartGrid([2, 2], physdims=[1, 2])
        g.compute_geometry()
        param = Parameters(g)

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["neu"]
        bc_type[0] = "dir"
        bc_type[2] = "dir"
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Mpfa()
        param.set_bc(fd, bound)

        bc_val = np.zeros(g.num_faces)
        # Set up unit pressure gradient in x-direction
        bc_val[[2, 5]] = 1
        param.set_bc_val(fd, bc_val)
        data = {"param": param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        bound_p = data["bound_pressure_cell"] * p + data["bound_pressure_face"] * bc_val
        assert np.allclose(bound_p[bf], -g.face_centers[0, bf])

    def test_linear_pressure_dirichlet_conditions_perturbed_grid(self):
        g = self.grid()
        g.nodes[:2] = g.nodes[:2] + np.random.random((2, g.num_nodes))
        g.compute_geometry()
        param = Parameters(g)

        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = bf.size * ["dir"]
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Mpfa()
        param.set_bc(fd, bound)

        bc_val = 1 * g.face_centers[0] + 2 * g.face_centers[1]
        param.set_bc_val(fd, bc_val)
        data = {"param": param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        bound_p = data["bound_pressure_cell"] * p + data["bound_pressure_face"] * bc_val
        assert np.allclose(bound_p[bf], bc_val[bf])

    def test_sign_trouble_two_neumann_sides(self):
        g = CartGrid(np.array([2, 2]), physdims=[2, 2])
        g.compute_geometry()
        d = {"param": Parameters(g)}
        bv = np.zeros(g.num_faces)
        # Flow from right to left
        bv[[0, 3]] = 1
        bv[[2, 5]] = -1
        d["param"].set_bc_val("flow", bv)
        t = Mpfa("flow")
        A, b = t.matrix_rhs(g, d)
        # The problem is singular, and spsolve does not work well on all systems.
        # Instead, set a consistent solution, and check that the boundary
        # pressure is recovered.
        x = g.cell_centers[0]

        bound_p = d["bound_pressure_cell"] * x + d["bound_pressure_face"] * bv
        assert bound_p[0] == x[0] - 0.5
        assert bound_p[2] == x[1] + 0.5

    def test_structured_simplex_linear_flow(self):
        g = self.simplex_grid()
        g.compute_geometry()
        param = Parameters(g)
        bv = np.zeros(g.num_faces)
        # Flow from right to left
        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = np.asarray(bf.size * ["neu"])

        xf = g.face_centers[:, bf]
        xleft = np.where(xf[0] < 1e-3 + xf[0].min())[0]
        xright = np.where(xf[0] > xf[0].max() - 1e-3)[0]
        bc_type[xleft] = "dir"

        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Mpfa()
        param.set_bc(fd, bound)

        bv = np.zeros(g.num_faces)
        bv[bf[xright]] = 1

        param.set_bc_val("flow", bv)

        data = {"param": param}
        t = Mpfa("flow")

        A, b = t.matrix_rhs(g, data)
        x = spl.spsolve(A, b)

        bound_p = data["bound_pressure_cell"] * x + data["bound_pressure_face"] * bv
        assert np.allclose(bound_p[bf], -g.face_centers[0, bf])

    def test_structured_simplex_linear_flow_reverse_sign(self):
        g = self.simplex_grid()
        g.compute_geometry()
        param = Parameters(g)
        bv = np.zeros(g.num_faces)
        # Flow from right to left
        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = np.asarray(bf.size * ["neu"])

        xf = g.face_centers[:, bf]
        xleft = np.where(xf[0] < 1e-3 + xf[0].min())[0]
        xright = np.where(xf[0] > xf[0].max() - 1e-3)[0]
        bc_type[xleft] = "dir"

        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Mpfa()
        param.set_bc(fd, bound)

        bv = np.zeros(g.num_faces)
        bv[bf[xright]] = -1

        param.set_bc_val("flow", bv)

        data = {"param": param}
        t = Mpfa("flow")

        A, b = t.matrix_rhs(g, data)
        x = spl.spsolve(A, b)

        bound_p = data["bound_pressure_cell"] * x + data["bound_pressure_face"] * bv
        assert np.allclose(bound_p[bf], g.face_centers[0, bf])


class TestMpfaSimplexGrid:
    def grid(self):
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

        p = np.zeros((2, 0))
        mesh_size = {"value": 0.3, "bound_value": 0.3}
        gb = pp.meshing.simplex_grid(
            fracs=[], domain=domain, mesh_size=mesh_size, verbose=0
        )
        return gb.grids_of_dimension(2)[0]

    def test_linear_flow(self):
        g = self.grid()
        g.compute_geometry()
        param = Parameters(g)
        bv = np.zeros(g.num_faces)
        # Flow from right to left
        bf = np.where(g.tags["domain_boundary_faces"].ravel())[0]
        bc_type = np.asarray(bf.size * ["neu"])

        xf = g.face_centers[:, bf]
        xleft = np.where(xf[0] < 1e-3 + xf[0].min())[0]
        xright = np.where(xf[0] > xf[0].max() - 1e-3)[0]
        bc_type[xleft] = "dir"

        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Mpfa()
        param.set_bc(fd, bound)

        bv = np.zeros(g.num_faces)
        bv[bf[xright]] = 1 * g.face_areas[bf[xright]]

        param.set_bc_val("flow", bv)

        data = {"param": param}
        t = Mpfa("flow")

        A, b = t.matrix_rhs(g, data)
        x = spl.spsolve(A, b)

        bound_p = data["bound_pressure_cell"] * x + data["bound_pressure_face"] * bv
        assert np.allclose(bound_p[bf], -g.face_centers[0, bf])


if __name__ == "__main__":
    unittest.main()
