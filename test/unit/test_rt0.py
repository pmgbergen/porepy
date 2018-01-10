import numpy as np
import unittest

from porepy.grids import structured, simplex
from porepy.params import tensor
from porepy.params.bc import BoundaryCondition
from porepy.params.data import Parameters
from porepy.numerics.vem import rt0
import porepy.utils.comp_geom as cg

#------------------------------------------------------------------------------#

class BasicsTest( unittest.TestCase ):

#------------------------------------------------------------------------------#

    def test_rt0_1d_iso(self):
        g = structured.CartGrid(3, 1)
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = tensor.SecondOrder(3, kxx, kyy=1, kzz=1)
        bf = g.get_boundary_faces()
        bc = BoundaryCondition(g, bf, bf.size*['dir'])

        solver = rt0.RT0(physics='flow')

        param = Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {'param': param}).todense()

        M_known = \
            np.matrix([[ 0.11111111,  0.05555556,  0.        ,  0.        ,  1.,
                         0.        ,  0.        ],
                       [ 0.05555556,  0.22222222,  0.05555556,  0.        , -1.,
                         1.        ,  0.        ],
                       [ 0.        ,  0.05555556,  0.22222222,  0.05555556,  0.,
                        -1.        ,  1.        ],
                       [ 0.        ,  0.        ,  0.05555556,  0.11111111,  0.,
                         0.        , -1.        ],
                       [ 1.        , -1.        ,  0.        ,  0.        ,  0.,
                         0.        ,  0.        ],
                       [ 0.        ,  1.        , -1.        ,  0.        ,  0.,
                         0.        ,  0.        ],
                       [ 0.        ,  0.        ,  1.        , -1.        ,  0.,
                         0.        ,  0.        ]])

        assert np.allclose(M, M.T)
        assert np.allclose(M, M_known)

#------------------------------------------------------------------------------#

    def test_rt0_1d_ani(self):
        g = structured.CartGrid(3, 1)
        g.compute_geometry()

        kxx = np.sin(g.cell_centers[0,:])+1
        perm = tensor.SecondOrder(3, kxx, kyy=1, kzz=1)
        bf = g.get_boundary_faces()
        bc = BoundaryCondition(g, bf, bf.size*['dir'])
        solver = rt0.RT0(physics='flow')

        param = Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {'param': param}).todense()

        M_known = \
            np.matrix([[ 0.12954401,  0.06477201,  0.        ,  0.        ,  1.,
                         0.        ,  0.        ],
                       [ 0.06477201,  0.29392463,  0.08219031,  0.        , -1.,
                         1.        ,  0.        ],
                       [ 0.        ,  0.08219031,  0.3577336 ,  0.09667649,  0.,
                        -1.        ,  1.        ],
                       [ 0.        ,  0.        ,  0.09667649,  0.19335298,  0.,
                         0.        , -1.        ],
                       [ 1.        , -1.        ,  0.        ,  0.        ,  0.,
                         0.        ,  0.        ],
                       [ 0.        ,  1.        , -1.        ,  0.        ,  0.,
                         0.        ,  0.        ],
                       [ 0.        ,  0.        ,  1.        , -1.        ,  0.,
                         0.        ,  0.        ]])

        assert np.allclose(M, M.T)
        assert np.allclose(M, M_known)

#------------------------------------------------------------------------------#

    def test_rt0_2d_iso_simplex(self):
        g = simplex.StructuredTriangleGrid([1, 1], [1, 1])
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = tensor.SecondOrder(3, kxx=kxx, kyy=kxx, kzz=1)

        bf = g.get_boundary_faces()
        bc = BoundaryCondition(g, bf, bf.size*['dir'])
        solver = rt0.RT0(physics='flow')

        param = Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {'param': param}).todense()

        # Matrix computed with an already validated code
        M_known = \
    np.matrix([[ 0.33333333,  0.        ,  0.        , -0.16666667,  0.        ,
                -1.        ,  0.        ],
               [ 0.        ,  0.33333333,  0.        ,  0.        , -0.16666667,
                 0.        , -1.        ],
               [ 0.        ,  0.        ,  0.33333333,  0.        ,  0.        ,
                -1.        ,  1.        ],
               [-0.16666667,  0.        ,  0.        ,  0.33333333,  0.        ,
                -1.        ,  0.        ],
               [ 0.        , -0.16666667,  0.        ,  0.        ,  0.33333333,
                 0.        , -1.        ],
               [-1.        ,  0.        , -1.        , -1.        ,  0.        ,
                 0.        ,  0.        ],
               [ 0.        , -1.        ,  1.        ,  0.        , -1.        ,
                 0.        ,  0.        ]])

        assert np.allclose(M, M.T)
        # We test only the mass-Hdiv part
        assert np.allclose(M, M_known)

#------------------------------------------------------------------------------#

    def test_rt0_2d_ani_simplex(self): #####
        g = simplex.StructuredTriangleGrid([1, 1], [1, 1])
        g.compute_geometry()

        kxx = np.square(g.cell_centers[1,:])+1
        kyy = np.square(g.cell_centers[0,:])+1
        kxy =-np.multiply(g.cell_centers[0,:], g.cell_centers[1,:])
        perm = tensor.SecondOrder(3, kxx=kxx, kyy=kyy, kxy=kxy, kzz=1)

        bf = g.get_boundary_faces()
        bc = BoundaryCondition(g, bf, bf.size * ['dir'])
        solver = rt0.RT0(physics='flow')

        param = Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {'param': param}).todense()

        # Matrix computed with an already validated code
        faces = np.arange(5)
        map_faces = np.array([1, 4, 0, 2, 3])
        M_known = np.array([[0.865079365079365, 0.337301587301587,
                             0.301587301587302, -0.301587301587302,
                             -0.337301587301587, -1, 1],
                            [0.337301587301587, 0.599206349206349,
                             0.134920634920635, 0, 0, -1, 0],
                            [0.301587301587302, 0.134920634920635,
                             0.634920634920635, 0, 0, -1, 0],
                            [-0.301587301587302, 0, 0, 0.634920634920635,
                             0.134920634920635, 0, -1],
                            [-0.337301587301587, 0, 0, 0.134920634920635,
                             0.599206349206349, 0, -1],
                            [-1, -1, -1, 0, 0, 0, 0],
                            [1, 0, 0, -1, -1, 0, 0]])

        assert np.allclose(M, M.T)
        # We test only the mass-Hdiv part
        assert np.allclose(M[np.ix_(faces, faces)],
                           M_known[np.ix_(map_faces, map_faces)])

#------------------------------------------------------------------------------#

    def test_rt0_3d_iso(self): ###

        g = simplex.StructuredTetrahedralGrid([1, 1, 1], [1, 1, 1])
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = tensor.SecondOrder(3, kxx=kxx, kyy=kxx, kzz=kxx)

        from porepy.viz import plot_grid

        plot_grid.plot_grid(g, alpha=0, info='all')

        bf = g.get_boundary_faces()
        bc = BoundaryCondition(g, bf, bf.size * ['dir'])
        solver = rt0.RT0(physics='flow')

        param = Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {'param': param}).todense()
        #np.savetxt('matrix.txt', M, delimiter=',', newline='],\n[')
        M_known = matrix_for_test_dual_vem_3d_iso_cart()

        assert np.allclose(M, M.T)
        assert np.allclose(M, M_known)

#------------------------------------------------------------------------------#

    def test_dual_rt0_1d_iso_line(self):
        g = structured.CartGrid(3, 1)
        R = cg.rot(np.pi/6., [0,0,1])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = tensor.SecondOrder(3, kxx, kyy=1, kzz=1)
        perm.rotate(R)

        bf = g.get_boundary_faces()
        bc = BoundaryCondition(g, bf, bf.size * ['dir'])
        solver = rt0.RT0(physics='flow')

        param = Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {'param': param}).todense()

        # Matrix computed with an already validated code
        M_known = \
            np.matrix([[ 0.11111111,  0.05555556,  0.        ,  0.        ,  1.,
                         0.        ,  0.        ],
                       [ 0.05555556,  0.22222222,  0.05555556,  0.        , -1.,
                         1.        ,  0.        ],
                       [ 0.        ,  0.05555556,  0.22222222,  0.05555556,  0.,
                        -1.        ,  1.        ],
                       [ 0.        ,  0.        ,  0.05555556,  0.11111111,  0.,
                         0.        , -1.        ],
                       [ 1.        , -1.        ,  0.        ,  0.        ,  0.,
                         0.        ,  0.        ],
                       [ 0.        ,  1.        , -1.        ,  0.        ,  0.,
                         0.        ,  0.        ],
                       [ 0.        ,  0.        ,  1.        , -1.        ,  0.,
                         0.        ,  0.        ]])

        assert np.allclose(M, M.T)
        assert np.allclose(M, M_known)

#------------------------------------------------------------------------------#

    def test_rt0_2d_iso_simplex_surf(self):
        g = simplex.StructuredTriangleGrid([1, 1], [1, 1])
        R = cg.rot(-np.pi/4., [1,1,-1])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry(is_embedded=True)

        kxx = np.ones(g.num_cells)
        perm = tensor.SecondOrder(3, kxx=kxx, kyy=kxx, kzz=1)
        perm.rotate(R)

        bf = g.get_boundary_faces()
        bc = BoundaryCondition(g, bf, bf.size * ['dir'])
        solver = rt0.RT0(physics='flow')

        param = Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {'param': param}).todense()

        # Matrix computed with an already validated code
        M_known = \
    np.matrix([[ 0.33333333,  0.        ,  0.        , -0.16666667,  0.        ,
                -1.        ,  0.        ],
               [ 0.        ,  0.33333333,  0.        ,  0.        , -0.16666667,
                 0.        , -1.        ],
               [ 0.        ,  0.        ,  0.33333333,  0.        ,  0.        ,
                -1.        ,  1.        ],
               [-0.16666667,  0.        ,  0.        ,  0.33333333,  0.        ,
                -1.        ,  0.        ],
               [ 0.        , -0.16666667,  0.        ,  0.        ,  0.33333333,
                 0.        , -1.        ],
               [-1.        ,  0.        , -1.        , -1.        ,  0.        ,
                 0.        ,  0.        ],
               [ 0.        , -1.        ,  1.        ,  0.        , -1.        ,
                 0.        ,  0.        ]])

        assert np.allclose(M, M.T)
        # We test only the mass-Hdiv part
        assert np.allclose(M, M_known)

#------------------------------------------------------------------------------#

    def test_rt0_2d_ani_simplex_surf(self):
        g = simplex.StructuredTriangleGrid([1, 1], [1, 1])
        g.compute_geometry()

        kxx = np.square(g.cell_centers[1,:])+1
        kyy = np.square(g.cell_centers[0,:])+1
        kxy =-np.multiply(g.cell_centers[0,:], g.cell_centers[1,:])
        perm = tensor.SecondOrder(3, kxx=kxx, kyy=kyy, kxy=kxy, kzz=1)

        R = cg.rot(np.pi/3., [1,1,0])
        perm.rotate(R)
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry(is_embedded=True)

        bf = g.get_boundary_faces()
        bc = BoundaryCondition(g, bf, bf.size*['dir'])
        solver = rt0.RT0(physics='flow')

        param = Parameters(g)
        param.set_tensor(solver, perm)
        param.set_bc(solver, bc)
        M = solver.matrix(g, {'param': param}).todense()

        # Matrix computed with an already validated code
        faces = np.arange(5)
        map_faces = np.array([1, 4, 0, 2, 3])
        M_known = np.array([[0.865079365079365, 0.337301587301587,
                             0.301587301587302, -0.301587301587302,
                             -0.337301587301587, -1, 1],
                            [0.337301587301587, 0.599206349206349,
                             0.134920634920635, 0, 0, -1, 0],
                            [0.301587301587302, 0.134920634920635,
                             0.634920634920635, 0, 0, -1, 0],
                            [-0.301587301587302, 0, 0, 0.634920634920635,
                             0.134920634920635, 0, -1],
                            [-0.337301587301587, 0, 0, 0.134920634920635,
                             0.599206349206349, 0, -1],
                            [-1, -1, -1, 0, 0, 0, 0],
                            [1, 0, 0, -1, -1, 0, 0]])

        assert np.allclose(M, M.T)
        # We test only the mass-Hdiv part
        assert np.allclose(M[np.ix_(faces, faces)],
                           M_known[np.ix_(map_faces, map_faces)])

#------------------------------------------------------------------------------#

#BasicsTest().test_rt0_2d_iso_simplex()
BasicsTest().test_rt0_2d_iso_simplex()
