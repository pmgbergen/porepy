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

        bf = np.where(g.tags['domain_boundary_faces'].ravel())[0]
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

        bf = np.where(g.tags['domain_boundary_faces'].ravel())[0]
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
        assert np.allclose(bound_p[bf], bc_val[bf])

    def test_constant_pressure_simplex_grid(self):
        g = self.simplex_grid()
        param = Parameters(g)

        bf = np.where(g.tags['domain_boundary_faces'].ravel())[0]
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
        assert np.allclose(bound_p[bf], bc_val[bf])

    def test_linear_pressure_dirichlet_conditions(self):
        g = self.grid()
        param = Parameters(g)

        bf = np.where(g.tags['domain_boundary_faces'].ravel())[0]
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
        assert np.allclose(bound_p[bf], bc_val[bf])

    def test_linear_pressure_part_neumann_conditions(self):
        g = self.grid()
        param = Parameters(g)

        bf = np.where(g.tags['domain_boundary_faces'].ravel())[0]
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
        assert np.allclose(bound_p[bf], -g.face_centers[0, bf])

    def test_linear_pressure_part_neumann_conditions_small_domain(self):
        g = self.grid(physdims=[1, 1])
        param = Parameters(g)

        bf = np.where(g.tags['domain_boundary_faces'].ravel())[0]
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
        assert np.allclose(bound_p[bf], -2*g.face_centers[0, bf])

    def test_linear_pressure_part_neumann_conditions_reverse_sign(self):
        g = self.grid()
        param = Parameters(g)

        bf = np.where(g.tags['domain_boundary_faces'].ravel())[0]
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
        assert np.allclose(bound_p[bf], g.face_centers[0, bf])

    def test_linear_pressure_part_neumann_conditions_smaller_domain(self):
        # Smaller domain, check that the smaller pressure gradient is captured
        g = CartGrid([2, 2], physdims=[1, 2])
        g.compute_geometry()
        param = Parameters(g)

        bf = np.where(g.tags['domain_boundary_faces'].ravel())[0]
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
        assert np.allclose(bound_p[bf], -g.face_centers[0, bf])

    def test_sign_trouble_two_neumann_sides(self):
        g = CartGrid(np.array([2, 2]), physdims=[2, 2])
        g.compute_geometry()
        d = {'param': Parameters(g)}
        bv = np.zeros(g.num_faces)
        bv[[0, 3]] = 1
        bv[[2, 5]] = -1
        d['param'].set_bc_val('flow', bv)
        t = Tpfa('flow')
        A, b = t.matrix_rhs(g, d)
        x = spl.spsolve(A, b)

        bound_p = d['bound_pressure_cell'] * x\
                + d['bound_pressure_face'] * bv
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

        bf = np.where(g.tags['domain_boundary_faces'].ravel())[0]
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

        bf = np.where(g.tags['domain_boundary_faces'].ravel())[0]
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
        assert np.allclose(bound_p[bf], bc_val[bf])

    def test_constant_pressure_simplex_grid(self):
        g = self.simplex_grid()
        param = Parameters(g)

        bf = np.where(g.tags['domain_boundary_faces'].ravel())[0]
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
        assert np.allclose(bound_p[bf], bc_val[bf])

    def test_linear_pressure_dirichlet_conditions(self):
        g = self.grid()
        param = Parameters(g)

        bf = np.where(g.tags['domain_boundary_faces'].ravel())[0]
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
        assert np.allclose(bound_p[bf], bc_val[bf])

    def test_linear_pressure_part_neumann_conditions(self):
        g = self.grid()
        param = Parameters(g)

        bf = np.where(g.tags['domain_boundary_faces'].ravel())[0]
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
        assert np.allclose(bound_p[bf], -g.face_centers[0, bf])

    def test_linear_pressure_part_neumann_conditions_reverse_sign(self):
        g = self.grid()
        param = Parameters(g)

        bf = np.where(g.tags['domain_boundary_faces'].ravel())[0]
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
        assert np.allclose(bound_p[bf], g.face_centers[0, bf])

    def test_linear_pressure_part_neumann_conditions_smaller_domain(self):
        # Smaller domain, check that the smaller pressure gradient is captured
        g = CartGrid([2, 2], physdims=[1, 2])
        g.compute_geometry()
        param = Parameters(g)

        bf = np.where(g.tags['domain_boundary_faces'].ravel())[0]
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
        assert np.allclose(bound_p[bf], -g.face_centers[0, bf])

    def test_linear_pressure_dirichlet_conditions_perturbed_grid(self):
        g = self.grid()
        g.nodes[:2] = g.nodes[:2] + np.random.random((2, g.num_nodes))
        g.compute_geometry()
        param = Parameters(g)

        bf = np.where(g.tags['domain_boundary_faces'].ravel())[0]
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
        assert np.allclose(bound_p[bf], bc_val[bf])

    def test_sign_trouble_two_neumann_sides(self):
        g = CartGrid(np.array([2, 2]), physdims=[2, 2])
        g.compute_geometry()
        d = {'param': Parameters(g)}
        bv = np.zeros(g.num_faces)
        bv[[0, 3]] = 1
        bv[[2, 5]] = -1
        d['param'].set_bc_val('flow', bv)
        t = Mpfa('flow')
        A, b = t.matrix_rhs(g, d)
        x = spl.spsolve(A, b)

        bound_p = d['bound_pressure_cell'] * x\
                + d['bound_pressure_face'] * bv
        assert bound_p[0] == x[0] - 0.5
        assert bound_p[2] == x[1] + 0.5

class TestMPFASimplexGrid():

    def grid_2d(self):
        nodes = np.array([[0, 0, 0], [1, 0, 0], [1, 0.5, 0], [0.5, 0.5, 0],
                          [0, 0.5, 0]]).T

        fn = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 3], [3, 1]]).T
        cf = np.array([[3, 4, 5], [0, 6, 5], [1, 2, 6]]).T
        cols = np.tile(np.arange(fn.shape[1]), (fn.shape[0], 1)).ravel('F')
        face_nodes = sps.csc_matrix((np.ones_like(cols),
                                     (fn.ravel('F'), cols)))

        cols = np.tile(np.arange(cf.shape[1]), (cf.shape[0], 1)).ravel('F')
        cell_faces = sps.csc_matrix((np.array([1, 1, 1, 1, 1, -1, 1, 1, -1]),
                                     (cf.ravel('F'), cols)))

        g = Grid(2, nodes, face_nodes, cell_faces, 'TriangleGrid')
        g.compute_geometry()
        #g.face_normals[1, [2, 3]] = -0.5
        #g.face_normals[1, [7, 8]] = 0.5
        return g

    def set_params(self, g):


        param = Parameters(g)
        d = {'param': param}
        perm = pp.SecondOrderTensor(g.dim, kxx=np.ones(g.num_cells))
        param.set_tensor("flow", perm)

        bound_faces = np.array([0])
        #bound_faces = g.get_boundary_faces()
        labels = np.array(['dir'] * bound_faces.size)
        param.set_bc("flow", bc.BoundaryCondition(g, bound_faces, labels))


        d['param'] = param
        return d


    def test_mpfa(self):
        g = self.grid_2d()
        d = self.set_params(g)
        bv = np.zeros(g.num_faces)
        bv[0] = 0
        bv[[2, 3]] = -0.5
        #bv = np.sum(g.face_centers, axis=0)
        d['param'].set_bc_val('flow', bv)
        t = Mpfa('flow')
        A, b = t.matrix_rhs(g, d)

        x = sps.linalg.spsolve(A, b)
        div = fvutils.scalar_divergence(g)
        print(x - g.cell_centers[1])
        print(x)
        print(g.face_centers)





if __name__ == '__main__':
    unittest.main()

#TestTpfaBoundaryPressure().test_constant_pressure()
