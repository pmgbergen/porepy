#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:22:41 2017

@author: Eirik Keilegavlens
"""

import numpy as np
import unittest
import scipy.sparse.linalg as spl

from porepy.grids.structured import CartGrid
from porepy.grids.simplex import StructuredTriangleGrid
from porepy.params import bc
from porepy.params.data import Parameters
from porepy.numerics.fv.tpfa import Tpfa
from porepy.numerics.fv.mpfa import Mpfa


class TestTpfaBoundaryPressure(unittest.TestCase):

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

        bf = g.get_boundary_faces()
        bc_type = bf.size * ['dir']
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Tpfa()
        param.set_bc(fd, bound)
        param.set_bc_val(fd, np.zeros(g.num_faces))
        data = {'param': param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        assert np.allclose(p, np.zeros_like(p))

        bound_p = data['bound_pressure_cell'] * p\
                + data['bound_pressure_face'] * np.zeros(g.num_faces)
        assert np.allclose(bound_p, np.zeros_like(bound_p))

    def test_constant_pressure(self):
        g = self.grid()
        param = Parameters(g)

        bf = g.get_boundary_faces()
        bc_type = bf.size * ['dir']
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Tpfa()
        param.set_bc(fd, bound)

        bc_val = np.ones(g.num_faces)
        param.set_bc_val(fd, bc_val)
        data = {'param': param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        assert np.allclose(p, np.ones_like(p))

        bound_p = data['bound_pressure_cell'] * p\
                + data['bound_pressure_face'] * bc_val
        assert np.allclose(bound_p[g.get_boundary_faces()],
                           bc_val[g.get_boundary_faces()])

    def test_constant_pressure_simplex_grid(self):
        g = self.simplex_grid()
        param = Parameters(g)

        bf = g.get_boundary_faces()
        bc_type = bf.size * ['dir']
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Tpfa()
        param.set_bc(fd, bound)

        bc_val = np.ones(g.num_faces)
        param.set_bc_val(fd, bc_val)
        data = {'param': param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        assert np.allclose(p, np.ones_like(p))

        bound_p = data['bound_pressure_cell'] * p\
                + data['bound_pressure_face'] * bc_val
        assert np.allclose(bound_p[g.get_boundary_faces()],
                           bc_val[g.get_boundary_faces()])

    def test_linear_pressure_dirichlet_conditions(self):
        g = self.grid()
        param = Parameters(g)

        bf = g.get_boundary_faces()
        bc_type = bf.size * ['dir']
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Tpfa()
        param.set_bc(fd, bound)

        bc_val = 1 * g.face_centers[0] + 2 * g.face_centers[1]
        param.set_bc_val(fd, bc_val)
        data = {'param': param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        bound_p = data['bound_pressure_cell'] * p\
                + data['bound_pressure_face'] * bc_val
        assert np.allclose(bound_p[g.get_boundary_faces()],
                           bc_val[g.get_boundary_faces()])

    def test_linear_pressure_part_neumann_conditions(self):
        g = self.grid()
        param = Parameters(g)

        bf = g.get_boundary_faces()
        bc_type = bf.size * ['neu']
        bc_type[0] = 'dir'
        bc_type[2] = 'dir'
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Tpfa()
        param.set_bc(fd, bound)

        bc_val = np.zeros(g.num_faces)
        # Set up unit pressure gradient in x-direction
        bc_val[[2, 5]] = 1
        param.set_bc_val(fd, bc_val)
        data = {'param': param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        bound_p = data['bound_pressure_cell'] * p\
                + data['bound_pressure_face'] * bc_val
        assert np.allclose(bound_p[g.get_boundary_faces()],
                           -g.face_centers[0, g.get_boundary_faces()])

    def test_linear_pressure_part_neumann_conditions_reverse_sign(self):
        g = self.grid()
        param = Parameters(g)

        bf = g.get_boundary_faces()
        bc_type = bf.size * ['neu']
        bc_type[0] = 'dir'
        bc_type[2] = 'dir'
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Tpfa()
        param.set_bc(fd, bound)

        bc_val = np.zeros(g.num_faces)
        # Set up pressure gradient in x-direction, with value -1
        bc_val[[2, 5]] = -1
        param.set_bc_val(fd, bc_val)
        data = {'param': param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        bound_p = data['bound_pressure_cell'] * p\
                + data['bound_pressure_face'] * bc_val
        assert np.allclose(bound_p[g.get_boundary_faces()],
                           g.face_centers[0, g.get_boundary_faces()])

    def test_linear_pressure_part_neumann_conditions_smaller_domain(self):
        # Smaller domain, check that the smaller pressure gradient is captured
        g = CartGrid([2, 2], physdims=[1, 2])
        g.compute_geometry()
        param = Parameters(g)

        bf = g.get_boundary_faces()
        bc_type = bf.size * ['neu']
        bc_type[0] = 'dir'
        bc_type[2] = 'dir'
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Tpfa()
        param.set_bc(fd, bound)

        bc_val = np.zeros(g.num_faces)
        # Set up unit pressure gradient in x-direction
        bc_val[[2, 5]] = 1
        param.set_bc_val(fd, bc_val)
        data = {'param': param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        bound_p = data['bound_pressure_cell'] * p\
                + data['bound_pressure_face'] * bc_val
        assert np.allclose(bound_p[g.get_boundary_faces()],
                           -g.face_centers[0, g.get_boundary_faces()])


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

        bf = g.get_boundary_faces()
        bc_type = bf.size * ['dir']
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Mpfa()
        param.set_bc(fd, bound)
        param.set_bc_val(fd, np.zeros(g.num_faces))
        data = {'param': param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        assert np.allclose(p, np.zeros_like(p))

        bound_p = data['bound_pressure_cell'] * p\
                + data['bound_pressure_face'] * np.zeros(g.num_faces)
        assert np.allclose(bound_p, np.zeros_like(bound_p))

    def test_constant_pressure(self):
        g = self.grid()
        param = Parameters(g)

        bf = g.get_boundary_faces()
        bc_type = bf.size * ['dir']
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Mpfa()
        param.set_bc(fd, bound)

        bc_val = np.ones(g.num_faces)
        param.set_bc_val(fd, bc_val)
        data = {'param': param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        assert np.allclose(p, np.ones_like(p))

        bound_p = data['bound_pressure_cell'] * p\
                + data['bound_pressure_face'] * bc_val
        assert np.allclose(bound_p[g.get_boundary_faces()],
                           bc_val[g.get_boundary_faces()])

    def test_constant_pressure_simplex_grid(self):
        g = self.simplex_grid()
        param = Parameters(g)

        bf = g.get_boundary_faces()
        bc_type = bf.size * ['dir']
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Mpfa()
        param.set_bc(fd, bound)

        bc_val = np.ones(g.num_faces)
        param.set_bc_val(fd, bc_val)
        data = {'param': param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        assert np.allclose(p, np.ones_like(p))

        bound_p = data['bound_pressure_cell'] * p\
                + data['bound_pressure_face'] * bc_val
        assert np.allclose(bound_p[g.get_boundary_faces()],
                           bc_val[g.get_boundary_faces()])

    def test_linear_pressure_dirichlet_conditions(self):
        g = self.grid()
        param = Parameters(g)

        bf = g.get_boundary_faces()
        bc_type = bf.size * ['dir']
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Mpfa()
        param.set_bc(fd, bound)

        bc_val = 1 * g.face_centers[0] + 2 * g.face_centers[1]
        param.set_bc_val(fd, bc_val)
        data = {'param': param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        bound_p = data['bound_pressure_cell'] * p\
                + data['bound_pressure_face'] * bc_val
        assert np.allclose(bound_p[g.get_boundary_faces()],
                           bc_val[g.get_boundary_faces()])

    def test_linear_pressure_part_neumann_conditions(self):
        g = self.grid()
        param = Parameters(g)

        bf = g.get_boundary_faces()
        bc_type = bf.size * ['neu']
        bc_type[0] = 'dir'
        bc_type[2] = 'dir'
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Tpfa()
        param.set_bc(fd, bound)

        bc_val = np.zeros(g.num_faces)
        # Set up unit pressure gradient in x-direction
        bc_val[[2, 5]] = 1
        param.set_bc_val(fd, bc_val)
        data = {'param': param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        bound_p = data['bound_pressure_cell'] * p\
                + data['bound_pressure_face'] * bc_val
        assert np.allclose(bound_p[g.get_boundary_faces()],
                           -g.face_centers[0, g.get_boundary_faces()])

    def test_linear_pressure_part_neumann_conditions_reverse_sign(self):
        g = self.grid()
        param = Parameters(g)

        bf = g.get_boundary_faces()
        bc_type = bf.size * ['neu']
        bc_type[0] = 'dir'
        bc_type[2] = 'dir'
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Tpfa()
        param.set_bc(fd, bound)

        bc_val = np.zeros(g.num_faces)
        # Set up pressure gradient in x-direction, with value -1
        bc_val[[2, 5]] = -1
        param.set_bc_val(fd, bc_val)
        data = {'param': param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        bound_p = data['bound_pressure_cell'] * p\
                + data['bound_pressure_face'] * bc_val
        assert np.allclose(bound_p[g.get_boundary_faces()],
                           g.face_centers[0, g.get_boundary_faces()])

    def test_linear_pressure_part_neumann_conditions_smaller_domain(self):
        # Smaller domain, check that the smaller pressure gradient is captured
        g = CartGrid([2, 2], physdims=[1, 2])
        g.compute_geometry()
        param = Parameters(g)

        bf = g.get_boundary_faces()
        bc_type = bf.size * ['neu']
        bc_type[0] = 'dir'
        bc_type[2] = 'dir'
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Mpfa()
        param.set_bc(fd, bound)

        bc_val = np.zeros(g.num_faces)
        # Set up unit pressure gradient in x-direction
        bc_val[[2, 5]] = 1
        param.set_bc_val(fd, bc_val)
        data = {'param': param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        bound_p = data['bound_pressure_cell'] * p\
                + data['bound_pressure_face'] * bc_val
        assert np.allclose(bound_p[g.get_boundary_faces()],
                           -g.face_centers[0, g.get_boundary_faces()])

    def test_linear_pressure_dirichlet_conditions_perturbed_grid(self):
        g = self.grid()
        g.nodes[:2] = g.nodes[:2] + np.random.random((2, g.num_nodes))
        g.compute_geometry()
        param = Parameters(g)

        bf = g.get_boundary_faces()
        bc_type = bf.size * ['dir']
        bound = bc.BoundaryCondition(g, bf, bc_type)

        fd = Mpfa()
        param.set_bc(fd, bound)

        bc_val = 1 * g.face_centers[0] + 2 * g.face_centers[1]
        param.set_bc_val(fd, bc_val)
        data = {'param': param}

        A, b = fd.matrix_rhs(g, data)
        p = spl.spsolve(A, b)

        bound_p = data['bound_pressure_cell'] * p\
                + data['bound_pressure_face'] * bc_val
        assert np.allclose(bound_p[g.get_boundary_faces()],
                           bc_val[g.get_boundary_faces()])


    if __name__ == '__main__':
        unittest.main()
