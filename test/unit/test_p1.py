import numpy as np
import scipy.sparse as sps
import unittest

from porepy.grids import structured, simplex
from porepy.params import tensor
from porepy.params.bc import BoundaryCondition
from porepy.params.data import Parameters
from porepy.numerics.fem import p1, source
import porepy.utils.comp_geom as cg

#------------------------------------------------------------------------------#

class BasicsTest( unittest.TestCase ):

#------------------------------------------------------------------------------#

    def test_p1_1d_iso(self):
        g = structured.CartGrid(3, 1)
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = tensor.SecondOrder(3, kxx, kyy=1, kzz=1)
        bf = g.get_boundary_faces()
        bc = BoundaryCondition(g, bf, bf.size*['dir'])

        solver = p1.P1(physics='flow')

        param = Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {'param': param}).todense()

        M_known = np.matrix([[ 1.,  0.,  0.,  0.],
                             [-3.,  6., -3.,  0.],
                             [ 0., -3.,  6., -3.],
                             [ 0.,  0.,  0.,  1.]])

        assert np.allclose(M, M_known)

#------------------------------------------------------------------------------#

    def test_p1_1d_ani(self):
        g = structured.CartGrid(3, 1)
        g.compute_geometry()

        kxx = np.sin(g.cell_centers[0,:])+1
        perm = tensor.SecondOrder(3, kxx, kyy=1, kzz=1)
        bf = g.get_boundary_faces()
        bc = BoundaryCondition(g, bf, bf.size*['neu'])
        solver = p1.P1(physics='flow')

        param = Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {'param': param}).todense()

        M_known =\
                np.matrix([[ 3.4976884 , -3.4976884 ,  0.        ,  0.        ],
                           [-3.4976884 ,  7.93596501, -4.43827662,  0.        ],
                           [ 0.        , -4.43827662,  9.65880718, -5.22053056],
                           [ 0.        ,  0.        , -5.22053056,  5.22053056]])

        assert np.allclose(M, M.T)
        assert np.allclose(M, M_known)

#------------------------------------------------------------------------------#

    def test_p1_2d_iso_simplex(self):
        g = simplex.StructuredTriangleGrid([1, 1], [1, 1])
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = tensor.SecondOrder(3, kxx=kxx, kyy=kxx, kzz=1)

        bf = g.get_boundary_faces()
        bc = BoundaryCondition(g, bf, bf.size*['neu'])
        solver = p1.P1(physics='flow')

        param = Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {'param': param}).todense()

        # Matrix computed with an already validated code
        M_known = np.matrix([[ 1. , -0.5, -0.5,  0. ],
                             [-0.5,  1. ,  0. , -0.5],
                             [-0.5,  0. ,  1. , -0.5],
                             [ 0. , -0.5, -0.5,  1. ]])

        assert np.allclose(M, M.T)
        assert np.allclose(M, M_known)

#------------------------------------------------------------------------------#

    def test_p1_2d_ani_simplex(self):
        g = simplex.StructuredTriangleGrid([1, 1], [1, 1])
        g.compute_geometry()

        kxx = np.square(g.cell_centers[1,:])+1
        kyy = np.square(g.cell_centers[0,:])+1
        kxy =-np.multiply(g.cell_centers[0,:], g.cell_centers[1,:])

        perm = tensor.SecondOrder(3, kxx=kxx, kyy=kyy, kxy=kxy, kzz=1)

        bf = g.get_boundary_faces()
        bc = BoundaryCondition(g, bf, bf.size*['neu'])
        solver = p1.P1(physics='flow')

        param = Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {'param': param}).todense()

        # Matrix computed with an already validated code
        M_known =\
                np.matrix([[ 1.11111111, -0.66666667, -0.66666667,  0.22222222],
                           [-0.66666667,  1.5       ,  0.        , -0.83333333],
                           [-0.66666667,  0.        ,  1.5       , -0.83333333],
                           [ 0.22222222, -0.83333333, -0.83333333,  1.44444444]])

        assert np.allclose(M, M.T)
        assert np.allclose(M, M_known)

#------------------------------------------------------------------------------#

    def test_p1_3d(self):

        g = simplex.StructuredTetrahedralGrid([1, 1, 1], [1, 1, 1])
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = tensor.SecondOrder(3, kxx=kxx, kyy=kxx, kzz=kxx)

        bf = g.get_boundary_faces()
        bc = BoundaryCondition(g, bf, bf.size*['neu'])
        solver = p1.P1(physics='flow')

        param = Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {'param': param}).todense()

        M_known = matrix_for_test_p1_3d()

        assert np.allclose(M, M.T)
        assert np.allclose(M, M_known)

#------------------------------------------------------------------------------#

    def test_dual_p1_1d_iso_line(self):
        g = structured.CartGrid(3, 1)
        R = cg.rot(np.pi/6., [0,0,1])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = tensor.SecondOrder(3, kxx, kyy=1, kzz=1)
        perm.rotate(R)

        bf = g.get_boundary_faces()
        bc = BoundaryCondition(g, bf, bf.size*['dir'])
        solver = p1.P1(physics='flow')

        param = Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {'param': param}).todense()

        # Matrix computed with an already validated code
        M_known = np.matrix([[ 1.,  0.,  0.,  0.],
                             [-3.,  6., -3.,  0.],
                             [ 0., -3.,  6., -3.],
                             [ 0.,  0.,  0.,  1.]])

        assert np.allclose(M, M_known)

#------------------------------------------------------------------------------#

    def test_rt0_2d_iso_simplex_surf(self):
        g = simplex.StructuredTriangleGrid([1, 1], [1, 1])
        R = cg.rot(-np.pi/4., [1,1,-1])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = tensor.SecondOrder(3, kxx=kxx, kyy=kxx, kzz=1)
        perm.rotate(R)

        bf = g.get_boundary_faces()
        bc = BoundaryCondition(g, bf, bf.size*['neu'])
        solver = p1.P1(physics='flow')

        param = Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {'param': param}).todense()

        # Matrix computed with an already validated code
        M_known = np.matrix([[ 1. , -0.5, -0.5,  0. ],
                             [-0.5,  1. ,  0. , -0.5],
                             [-0.5,  0. ,  1. , -0.5],
                             [ 0. , -0.5, -0.5,  1. ]])

        assert np.allclose(M, M.T)
        assert np.allclose(M, M_known)

#------------------------------------------------------------------------------#

    def test_p1_2d_ani_simplex_surf(self):
        g = simplex.StructuredTriangleGrid([1, 1], [1, 1])
        g.compute_geometry()

        kxx = np.square(g.cell_centers[1,:])+1
        kyy = np.square(g.cell_centers[0,:])+1
        kxy =-np.multiply(g.cell_centers[0,:], g.cell_centers[1,:])
        perm = tensor.SecondOrder(3, kxx=kxx, kyy=kyy, kxy=kxy, kzz=1)

        R = cg.rot(np.pi/3., [1,1,0])
        perm.rotate(R)
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        bf = g.get_boundary_faces()
        bc = BoundaryCondition(g, bf, bf.size*['neu'])
        solver = p1.P1(physics='flow')

        param = Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {'param': param}).todense()

        # Matrix computed with an already validated code
        M_known =\
                np.matrix([[ 1.11111111, -0.66666667, -0.66666667,  0.22222222],
                           [-0.66666667,  1.5       ,  0.        , -0.83333333],
                           [-0.66666667,  0.        ,  1.5       , -0.83333333],
                           [ 0.22222222, -0.83333333, -0.83333333,  1.44444444]])

        assert np.allclose(M, M.T)
        assert np.allclose(M, M_known)

#------------------------------------------------------------------------------#

    def test_convergence_1d_exact(self):

        p_ex = lambda pt: 2*pt[0, :]-3

        for i in np.arange(5):
            g = structured.CartGrid(3+i, 1)
            g.compute_geometry()

            kxx = np.ones(g.num_cells)
            perm = tensor.SecondOrder(3, kxx, kyy=1, kzz=1)
            bf = g.get_boundary_faces()
            bc = BoundaryCondition(g, bf, bf.size*['dir'])
            bc_val = np.zeros(g.num_faces)
            bc_val[bf] = p_ex(g.face_centers[:, bf])

            solver = p1.P1(physics='flow')

            param = Parameters(g)
            param.set_tensor(solver, perm)
            param.set_bc(solver, bc)
            param.set_bc_val(solver, bc_val)
            M, rhs = solver.matrix_rhs(g, {'param': param})

            p = sps.linalg.spsolve(M, rhs)
            err = np.sum(np.abs(p - p_ex(g.nodes)))

            assert np.isclose(err, 0)

#------------------------------------------------------------------------------#

    def test_convergence_1d_not_exact(self):

        p_ex = lambda pt: np.sin(2*np.pi*pt[0, :])
        source_ex = lambda pt: 4*np.pi**2*np.sin(2*np.pi*pt[0, :])

        for i in np.arange(5):
            g = structured.CartGrid(3+i, 1)
            g.compute_geometry()

            kxx = np.ones(g.num_cells)
            perm = tensor.SecondOrder(3, kxx, kyy=1, kzz=1)
            bf = g.get_boundary_faces()
            bc = BoundaryCondition(g, bf, bf.size*['dir'])
            source_val = source_ex(g.cell_centers)*g.cell_volumes

            solver = p1.P1(physics='flow')
            integral = source.Integral(physics='flow')

            param = Parameters(g)
            param.set_tensor(solver, perm)
            param.set_bc(solver, bc)
            param.set_source(solver, source_val)
            M, _ = solver.matrix_rhs(g, {'param': param})
            _, rhs = integral.matrix_rhs(g, {'param': param})

            p = sps.linalg.spsolve(M, rhs)
            #print(p)
            #print(p_ex(g.nodes))
            err = np.sum(np.abs(p - p_ex(g.nodes)))
            print(err)
#            assert np.isclose(err, 0)

#------------------------------------------------------------------------------#

    def test_convergence_2d_exact(self):

        p_ex = lambda pt: 2*pt[0, :]-3*pt[1, :]-9

        for i in np.arange(5):
            g = simplex.StructuredTriangleGrid([3+i]*2, [1, 1])
            #from porepy.viz.plot_grid import plot_grid
            g.compute_geometry()
            #plot_grid(g, info="all", alpha=0)

            kxx = np.ones(g.num_cells)
            perm = tensor.SecondOrder(3, kxx=kxx, kyy=kxx, kzz=1)
            bf = g.get_boundary_faces()
            bc = BoundaryCondition(g, bf, bf.size*['dir'])
            bc_val = np.zeros(g.num_faces)
            bc_val[bf] = p_ex(g.face_centers[:, bf])

            solver = p1.P1(physics='flow')

            param = Parameters(g)
            param.set_tensor(solver, perm)
            param.set_bc(solver, bc)
            param.set_bc_val(solver, bc_val)
            M, rhs = solver.matrix_rhs(g, {'param': param})

            p = sps.linalg.spsolve(M, rhs)
            err = np.sum(np.abs(p - p_ex(g.nodes)))
            print(p)
            print(p_ex(g.nodes))

            assert np.isclose(err, 0)

#------------------------------------------------------------------------------#


def matrix_for_test_p1_3d():
    return np.matrix(\
        [[ 0.5       , -0.16666667, -0.16666667,  0.        , -0.16666667,
          0.        ,  0.        ,  0.        ],
        [-0.16666667,  1.16666667, -0.16666667, -0.5       , -0.16666667,
         -0.5       ,  0.33333333,  0.        ],
        [-0.16666667, -0.16666667,  0.83333333, -0.16666667,  0.16666667,
          0.        , -0.5       ,  0.        ],
        [ 0.        , -0.5       , -0.16666667,  0.83333333,  0.        ,
          0.16666667, -0.16666667, -0.16666667],
        [-0.16666667, -0.16666667,  0.16666667,  0.        ,  0.83333333,
         -0.16666667, -0.5       ,  0.        ],
        [ 0.        , -0.5       ,  0.        ,  0.16666667, -0.16666667,
          0.83333333, -0.16666667, -0.16666667],
        [ 0.        ,  0.33333333, -0.5       , -0.16666667, -0.5       ,
         -0.16666667,  1.16666667, -0.16666667],
        [ 0.        ,  0.        ,  0.        , -0.16666667,  0.        ,
         -0.16666667, -0.16666667,  0.5       ]])

#------------------------------------------------------------------------------#

BasicsTest().test_convergence_2d_exact()
BasicsTest().test_convergence_1d_not_exact()
