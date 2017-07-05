from __future__ import division
import numpy as np
import unittest

from porepy.grids import structured, simplex
import porepy.utils.comp_geom as cg
from porepy.params import bc
from porepy.numerics.fv.transport import upwind

#------------------------------------------------------------------------------#

class BasicsTest( unittest.TestCase ):

#------------------------------------------------------------------------------#

    def test_upwind_1d_beta_positive(self):
        g = structured.CartGrid(3, 1)
        g.compute_geometry()

        solver = upwind.Upwind()
        data = {'discharge': solver.discharge(g, [2, 0, 0])}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[ 2, 0, 0],
                            [-2, 2, 0],
                            [ 0,-2, 0]])
        deltaT_known = 1/12

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_1d_dischargeegative(self):
        g = structured.CartGrid(3, 1)
        g.compute_geometry()

        solver = upwind.Upwind()
        data = {'discharge': solver.discharge(g, [-2, 0, 0])}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[0,-2, 0],
                            [0, 2,-2],
                            [0, 0, 2]])
        deltaT_known = 1/12

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_2d_cart_beta_positive(self):
        g = structured.CartGrid([3, 2], [1, 1])
        g.compute_geometry()

        solver = upwind.Upwind()
        data = {'discharge': solver.discharge(g, [2, 0, 0])}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[ 1, 0, 0, 0, 0, 0],
                            [-1, 1, 0, 0, 0, 0],
                            [ 0,-1, 0, 0, 0, 0],
                            [ 0, 0, 0, 1, 0, 0],
                            [ 0, 0, 0,-1, 1, 0],
                            [ 0, 0, 0, 0,-1, 0]])

        deltaT_known = 1/12

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_2d_cart_dischargeegative(self):
        g = structured.CartGrid([3, 2], [1, 1])
        g.compute_geometry()

        solver = upwind.Upwind()
        data = {'discharge': solver.discharge(g, [-2, 0, 0])}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[ 0,-1, 0, 0, 0, 0],
                            [ 0, 1,-1, 0, 0, 0],
                            [ 0, 0, 1, 0, 0, 0],
                            [ 0, 0, 0, 0,-1, 0],
                            [ 0, 0, 0, 0, 1,-1],
                            [ 0, 0, 0, 0, 0, 1]])
        deltaT_known = 1/12

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_2d_simplex_beta_positive(self):
        g = simplex.StructuredTriangleGrid([2, 1], [1, 1])
        g.compute_geometry()

        solver = upwind.Upwind()
        data = {'discharge': solver.discharge(g, [1, 0, 0])}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[ 1,-1, 0, 0],
                            [ 0, 1, 0, 0],
                            [ 0, 0, 0,-1],
                            [-1, 0, 0, 1]])
        deltaT_known = 1/6

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_2d_simplex_dischargeegative(self):
        g = simplex.StructuredTriangleGrid([2, 1], [1, 1])
        g.compute_geometry()

        solver = upwind.Upwind()
        data = {'discharge': solver.discharge(g, [-1, 0, 0])}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[ 1, 0, 0,-1],
                            [-1, 0, 0, 0],
                            [ 0, 0, 1, 0],
                            [ 0, 0,-1, 1]])
        deltaT_known = 1/6

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_3d_cart_dischargeegative(self):
        g = structured.CartGrid([2, 2, 2], [1, 1, 1])
        g.compute_geometry()

        solver = upwind.Upwind()
        data = {'discharge': solver.discharge(g, [-1, 0, 0])}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = 0.25 * np.array([[ 0,-1, 0, 0, 0, 0, 0, 0],
                                   [ 0, 1, 0, 0, 0, 0, 0, 0],
                                   [ 0, 0, 0,-1, 0, 0, 0, 0],
                                   [ 0, 0, 0, 1, 0, 0, 0, 0],
                                   [ 0, 0, 0, 0, 0,-1, 0, 0],
                                   [ 0, 0, 0, 0, 0, 1, 0, 0],
                                   [ 0, 0, 0, 0, 0, 0, 0,-1],
                                   [ 0, 0, 0, 0, 0, 0, 0, 1]])

        deltaT_known = 1/4

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_3d_cart_beta_positive(self):
        g = structured.CartGrid([2, 2, 2], [1, 1, 1])
        g.compute_geometry()

        solver = upwind.Upwind()
        data = {'discharge': solver.discharge(g, [1, 0, 0])}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = 0.25 * np.array([[ 1, 0, 0, 0, 0, 0, 0, 0],
                                   [-1, 0, 0, 0, 0, 0, 0, 0],
                                   [ 0, 0, 1, 0, 0, 0, 0, 0],
                                   [ 0, 0,-1, 0, 0, 0, 0, 0],
                                   [ 0, 0, 0, 0, 1, 0, 0, 0],
                                   [ 0, 0, 0, 0,-1, 0, 0, 0],
                                   [ 0, 0, 0, 0, 0, 0, 1, 0],
                                   [ 0, 0, 0, 0, 0, 0,-1, 0]])

        deltaT_known = 1/4

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_1d_surf_beta_positive(self):
        g = structured.CartGrid(3, 1)
        R = cg.rot(-np.pi/5., [0,1,-1])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry(is_embedded=True)

        solver = upwind.Upwind()
        data = {'discharge': solver.discharge(g, np.dot(R, [1, 0, 0]))}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[ 1, 0, 0],
                            [-1, 1, 0],
                            [ 0,-1, 0]])
        deltaT_known = 1/6

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_1d_surf_dischargeegative(self):
        g = structured.CartGrid(3, 1)
        R = cg.rot(-np.pi/8., [-1,1,-1])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry(is_embedded=True)

        solver = upwind.Upwind()
        data = {'discharge': solver.discharge(g, np.dot(R, [-1, 0, 0]))}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[0,-1, 0],
                            [0, 1,-1],
                            [0, 0, 1]])
        deltaT_known = 1/6

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_2d_cart_surf_beta_positive(self):
        g = structured.CartGrid([3, 2], [1, 1])
        R = cg.rot(np.pi/4., [0,1,0])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry(is_embedded=True)

        solver = upwind.Upwind()
        data = {'discharge': solver.discharge(g, np.dot(R, [1, 0, 0]))}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = 0.5 * np.array([[ 1, 0, 0, 0, 0, 0],
                                  [-1, 1, 0, 0, 0, 0],
                                  [ 0,-1, 0, 0, 0, 0],
                                  [ 0, 0, 0, 1, 0, 0],
                                  [ 0, 0, 0,-1, 1, 0],
                                  [ 0, 0, 0, 0,-1, 0]])

        deltaT_known = 1/6

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_2d_cart_surf_dischargeegative(self):
        g = structured.CartGrid([3, 2], [1, 1])
        R = cg.rot(np.pi/6., [1,1,0])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry(is_embedded=True)

        solver = upwind.Upwind()
        data = {'discharge': solver.discharge(g, np.dot(R, [-1, 0, 0]))}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = 0.5 * np.array([[ 0,-1, 0, 0, 0, 0],
                                  [ 0, 1,-1, 0, 0, 0],
                                  [ 0, 0, 1, 0, 0, 0],
                                  [ 0, 0, 0, 0,-1, 0],
                                  [ 0, 0, 0, 0, 1,-1],
                                  [ 0, 0, 0, 0, 0, 1]])

        deltaT_known = 1/6

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_2d_simplex_surf_beta_positive(self):
        g = simplex.StructuredTriangleGrid([2, 1], [1, 1])
        R = cg.rot(np.pi/2., [1,1,0])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry(is_embedded=True)

        solver = upwind.Upwind()
        data = {'discharge': solver.discharge(g, np.dot(R, [1, 0, 0]))}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[ 1,-1, 0, 0],
                            [ 0, 1, 0, 0],
                            [ 0, 0, 0,-1],
                            [-1, 0, 0, 1]])
        deltaT_known = 1/6

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_2d_simplex_surf_dischargeegative(self):
        g = simplex.StructuredTriangleGrid([2, 1], [1, 1])
        R = cg.rot(-np.pi/5., [1,1,-1])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry(is_embedded=True)

        solver = upwind.Upwind()
        data = {'discharge': solver.discharge(g, np.dot(R, [-1, 0, 0]))}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[ 1, 0, 0,-1],
                            [-1, 0, 0, 0],
                            [ 0, 0, 1, 0],
                            [ 0, 0,-1, 1]])
        deltaT_known = 1/6

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_1d_dischargeegative_bc_dir(self):
        g = structured.CartGrid(3, 1)
        g.compute_geometry()

        solver = upwind.Upwind()
        data = {'discharge': solver.discharge(g, [-2, 0, 0]),
                'bc': bc.BoundaryCondition(g, g.get_boundary_faces(), ['dir']*2),
                'bc_val': 3*np.ones(g.num_faces).ravel('F')}

        M, rhs = solver.matrix_rhs(g, data)
        deltaT = solver.cfl(g, data)

        M_known = np.array([[2,-2, 0],
                            [0, 2,-2],
                            [0, 0, 2]])
        rhs_known = np.array([0, 0, 6])
        deltaT_known = 1/12

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M.todense(), M_known, rtol, atol)
        assert np.allclose(rhs, rhs_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_1d_dischargeegative_bc_neu(self):
        g = structured.CartGrid(3, 1)
        g.compute_geometry()

        solver = upwind.Upwind()
        bc_val = np.array([2, 0, 0, -2])
        data = {'discharge': solver.discharge(g, [-2, 0, 0]),
                'bc': bc.BoundaryCondition(g, g.get_boundary_faces(), ['neu']*2),
                'bc_val': bc_val.ravel('F')}

        M, rhs = solver.matrix_rhs(g, data)
        deltaT = solver.cfl(g, data)

        M_known = np.array([[0,-2, 0],
                            [0, 2,-2],
                            [0, 0, 2]])
        rhs_known = np.array([-2, 0, 2])
        deltaT_known = 1/12

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M.todense(), M_known, rtol, atol)
        assert np.allclose(rhs, rhs_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#
