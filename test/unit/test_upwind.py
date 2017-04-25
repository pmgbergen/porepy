import numpy as np
import unittest

from porepy.grids import structured, simplex
import porepy.utils.comp_geom as cg
from porepy.numerics.fv.transport import upwind

#------------------------------------------------------------------------------#

class BasicsTest( unittest.TestCase ):

#------------------------------------------------------------------------------#

    def test_upwind_1d_beta_positive(self):
        g = structured.CartGrid(3, 1)
        g.compute_geometry()

        solver = upwind.Upwind()
        data = {'beta_n': solver.beta_n(g, [1, 0, 0])}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[ 0, 0, 0],
                            [-1, 1, 0],
                            [ 0,-1, 1]])
        deltaT_known = 1/3

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_1d_beta_negative(self):
        g = structured.CartGrid(3, 1)
        g.compute_geometry()

        solver = upwind.Upwind()
        data = {'beta_n': solver.beta_n(g, [-1, 0, 0])}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[1,-1, 0],
                            [0, 1,-1],
                            [0, 0, 0]])
        deltaT_known = 1/3

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_2d_cart_beta_positive(self):
        g = structured.CartGrid([3, 2], [1, 1])
        g.compute_geometry()

        solver = upwind.Upwind()
        data = {'beta_n': solver.beta_n(g, [1, 0, 0])}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = 0.5 * np.array([[ 0, 0, 0, 0, 0, 0],
                                  [-1, 1, 0, 0, 0, 0],
                                  [ 0,-1, 1, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0,-1, 1, 0],
                                  [ 0, 0, 0, 0,-1, 1]])
        deltaT_known = 1/6

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_2d_cart_beta_negative(self):
        g = structured.CartGrid([3, 2], [1, 1])
        g.compute_geometry()

        solver = upwind.Upwind()
        data = {'beta_n': solver.beta_n(g, [-1, 0, 0])}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = 0.5 * np.array([[ 1,-1, 0, 0, 0, 0],
                                  [ 0, 1,-1, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 1,-1, 0],
                                  [ 0, 0, 0, 0, 1,-1],
                                  [ 0, 0, 0, 0, 0, 0]])
        deltaT_known = 1/6

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_2d_simplex_beta_positive(self):
        g = simplex.StructuredTriangleGrid([2, 1], [1, 1])
        g.compute_geometry()

        solver = upwind.Upwind()
        data = {'beta_n': solver.beta_n(g, [1, 0, 0])}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[ 1,-1, 0, 0],
                            [ 0, 0, 0, 0],
                            [ 0, 0, 1,-1],
                            [-1, 0, 0, 1]])
        deltaT_known = 1/8

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_2d_simplex_beta_negative(self):
        g = simplex.StructuredTriangleGrid([2, 1], [1, 1])
        g.compute_geometry()

        solver = upwind.Upwind()
        data = {'beta_n': solver.beta_n(g, [-1, 0, 0])}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[ 1, 0, 0,-1],
                            [-1, 1, 0, 0],
                            [ 0, 0, 0, 0],
                            [ 0, 0,-1, 1]])
        deltaT_known = 1/8

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_3d_cart_beta_negative(self):
        g = structured.CartGrid([2, 2, 2], [1, 1, 1])
        g.compute_geometry()

        solver = upwind.Upwind()
        data = {'beta_n': solver.beta_n(g, [-1, 0, 0])}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = 0.25 * np.array([[ 1,-1, 0, 0, 0, 0, 0, 0],
                                   [ 0, 0, 0, 0, 0, 0, 0, 0],
                                   [ 0, 0, 1,-1, 0, 0, 0, 0],
                                   [ 0, 0, 0, 0, 0, 0, 0, 0],
                                   [ 0, 0, 0, 0, 1,-1, 0, 0],
                                   [ 0, 0, 0, 0, 0, 0, 0, 0],
                                   [ 0, 0, 0, 0, 0, 0, 1,-1],
                                   [ 0, 0, 0, 0, 0, 0, 0, 0]])


        deltaT_known = 1/6

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_3d_cart_beta_positive(self):
        g = structured.CartGrid([2, 2, 2], [1, 1, 1])
        g.compute_geometry()

        solver = upwind.Upwind()
        data = {'beta_n': solver.beta_n(g, [1, 0, 0])}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = 0.25 * np.array([[ 0, 0, 0, 0, 0, 0, 0, 0],
                                   [-1, 1, 0, 0, 0, 0, 0, 0],
                                   [ 0, 0, 0, 0, 0, 0, 0, 0],
                                   [ 0, 0,-1, 1, 0, 0, 0, 0],
                                   [ 0, 0, 0, 0, 0, 0, 0, 0],
                                   [ 0, 0, 0, 0,-1, 1, 0, 0],
                                   [ 0, 0, 0, 0, 0, 0, 0, 0],
                                   [ 0, 0, 0, 0, 0, 0,-1, 1]])


        deltaT_known = 1/6

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
        data = {'beta_n': solver.beta_n(g, np.dot(R, [1, 0, 0]))}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[ 0, 0, 0],
                            [-1, 1, 0],
                            [ 0,-1, 1]])
        deltaT_known = 1/3

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_1d_surf_beta_negative(self):
        g = structured.CartGrid(3, 1)
        R = cg.rot(-np.pi/8., [-1,1,-1])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry(is_embedded=True)

        solver = upwind.Upwind()
        data = {'beta_n': solver.beta_n(g, np.dot(R, [-1, 0, 0]))}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[1,-1, 0],
                            [0, 1,-1],
                            [0, 0, 0]])
        deltaT_known = 1/3

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
        data = {'beta_n': solver.beta_n(g, np.dot(R, [1, 0, 0]))}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = 0.5 * np.array([[ 0, 0, 0, 0, 0, 0],
                                  [-1, 1, 0, 0, 0, 0],
                                  [ 0,-1, 1, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0,-1, 1, 0],
                                  [ 0, 0, 0, 0,-1, 1]])

        deltaT_known = 1/6

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_2d_cart_surf_beta_negative(self):
        g = structured.CartGrid([3, 2], [1, 1])
        R = cg.rot(np.pi/6., [1,1,0])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry(is_embedded=True)

        solver = upwind.Upwind()
        data = {'beta_n': solver.beta_n(g, np.dot(R, [-1, 0, 0]))}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = 0.5 * np.array([[ 1,-1, 0, 0, 0, 0],
                                  [ 0, 1,-1, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 1,-1, 0],
                                  [ 0, 0, 0, 0, 1,-1],
                                  [ 0, 0, 0, 0, 0, 0]])

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
        data = {'beta_n': solver.beta_n(g, np.dot(R, [1, 0, 0]))}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[ 1,-1, 0, 0],
                            [ 0, 0, 0, 0],
                            [ 0, 0, 1,-1],
                            [-1, 0, 0, 1]])
        deltaT_known = 1/8

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_2d_simplex_surf_beta_negative(self):
        g = simplex.StructuredTriangleGrid([2, 1], [1, 1])
        R = cg.rot(-np.pi/5., [1,1,-1])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry(is_embedded=True)

        solver = upwind.Upwind()
        data = {'beta_n': solver.beta_n(g, np.dot(R, [-1, 0, 0]))}
        M = solver.matrix_rhs(g, data)[0].todense()
        deltaT = solver.cfl(g, data)

        M_known = np.array([[ 1, 0, 0,-1],
                            [-1, 1, 0, 0],
                            [ 0, 0, 0, 0],
                            [ 0, 0,-1, 1]])
        deltaT_known = 1/8

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#
