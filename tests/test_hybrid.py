import numpy as np
import unittest

from core.grids import structured, simplex
from core.constit import second_order_tensor as sot
import compgeom.basics as cg

from vem import hybrid

#------------------------------------------------------------------------------#

class BasicsTest( unittest.TestCase ):

#------------------------------------------------------------------------------#

    def test_dual_hybrid_vem_2d_iso_cart(self):
        g = structured.CartGrid([2, 1], [1, 1])
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = sot.SecondOrderTensor(g.dim, kxx)

        solver = hybrid.HybridDualVEM()
        data = {'k': perm, 'f': np.zeros(g.num_cells)}
        M = solver.matrix_rhs(g, data)[0].todense()

        M_known = np.array([[-2.25,  1.75,  0.  ,  0.25,  0.  ,  0.25,  0.  ],
                            [ 1.75, -4.5 ,  1.75,  0.25,  0.25,  0.25,  0.25],
                            [ 0.  ,  1.75, -2.25,  0.  ,  0.25,  0.  ,  0.25],
                            [ 0.25,  0.25,  0.  , -0.75,  0.  ,  0.25,  0.  ],
                            [ 0.  ,  0.25,  0.25,  0.  , -0.75,  0.  ,  0.25],
                            [ 0.25,  0.25,  0.  ,  0.25,  0.  , -0.75,  0.  ],
                            [ 0.  ,  0.25,  0.25,  0.  ,  0.25,  0.  , -0.75]])

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M.T, rtol, atol)
        assert np.allclose(M, M_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_dual_hybrid_vem_2d_ani_cart(self):
        g = structured.CartGrid([2, 1], [1, 1])
        g.compute_geometry()

        kxx = np.square(g.cell_centers[1,:])+1
        kyy = np.square(g.cell_centers[0,:])+1
        kxy =-np.multiply(g.cell_centers[0,:], g.cell_centers[1,:])
        perm = sot.SecondOrderTensor(g.dim, kxx=kxx, kyy=kyy, kxy=kxy)

        solver = hybrid.HybridDualVEM()
        data = {'k': perm, 'f': np.zeros(g.num_cells)}
        M = solver.matrix_rhs(g, data)[0].todense()

        M_known = np.array(\
                [[-2.7386363636363646,  2.2613636363636376,  0.                ,
                   0.3636363636363635,  0.                ,  0.1136363636363635,
                   0.                ],
                 [ 2.2613636363636371, -5.472507331378301 ,  2.2661290322580658,
                   0.1136363636363638,  0.6088709677419353,
                   0.3636363636363638, -0.1411290322580648],
                 [ 0.                ,  2.2661290322580654,
                  -2.7338709677419368,  0.                , -0.1411290322580644,
                   0.                ,  0.6088709677419357],
                 [ 0.3636363636363636,  0.1136363636363638,  0.                ,
                  -0.7698863636363639,  0.                ,  0.2926136363636364,
                   0.                ],
                 [ 0.                ,  0.6088709677419354,
                  -0.1411290322580645,  0.                , -1.0151209677419353,
                   0.                ,  0.5473790322580644],
                 [ 0.1136363636363636,  0.3636363636363638,  0.                ,
                   0.2926136363636364,  0.                , -0.7698863636363638,
                   0.                ],
                 [ 0.                , -0.1411290322580647,
                   0.6088709677419355,  0.                ,  0.5473790322580643,
                   0.                , -1.0151209677419353]])

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M.T, rtol, atol)
        assert np.allclose(M, M_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_dual_hybrid_vem_2d_iso_simplex(self):
        g = simplex.StructuredTriangleGrid([1, 1], [1, 1])
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = sot.SecondOrderTensor(g.dim, kxx)

        solver = hybrid.HybridDualVEM()
        data = {'k': perm, 'f': np.zeros(g.num_cells)}
        M = solver.matrix_rhs(g, data)[0].todense()

        M_known = np.array([[ -2.0000000000000000e+00,   0.0000000000000000e+00,
                               2.0000000000000000e+00,   0.0000000000000000e+00,
                               0.0000000000000000e+00],
                            [  0.0000000000000000e+00,  -1.9999999999999996e+00,
                               1.9999999999999991e+00,   0.0000000000000000e+00,
                               4.9960036108132044e-16],
                            [  2.0000000000000000e+00,  1.9999999999999993e+00,
                              -7.9999999999999982e+00,   2.0000000000000000e+00,
                               1.9999999999999993e+00],
                            [ -1.1102230246251565e-16,   0.0000000000000000e+00,
                               2.0000000000000004e+00,  -2.0000000000000000e+00,
                               0.0000000000000000e+00],
                            [  0.0000000000000000e+00,   2.2204460492503131e-16,
                               1.9999999999999996e+00,   0.0000000000000000e+00,
                              -1.9999999999999998e+00]])

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M.T, rtol, atol)
        # We test only the mass-Hdiv part
        assert np.allclose(M, M_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_dual_hybrid_vem_2d_iso_simplex(self):
        g = simplex.StructuredTriangleGrid([1, 1], [1, 1])
        g.compute_geometry()

        kxx = np.square(g.cell_centers[1,:])+1
        kyy = np.square(g.cell_centers[0,:])+1
        kxy =-np.multiply(g.cell_centers[0,:], g.cell_centers[1,:])
        perm = sot.SecondOrderTensor(g.dim, kxx=kxx, kyy=kyy, kxy=kxy)

        solver = hybrid.HybridDualVEM()
        data = {'k': perm, 'f': np.zeros(g.num_cells)}
        M = solver.matrix_rhs(g, data)[0].todense()

        M_known = np.array(\
             [[ -2.888888888888888 ,   0.               ,    3.3333333333333326,
                -0.4444444444444444,   0.                ],
              [  0.                ,  -2.888888888888888,
                 3.3333333333333321,   0.                ,  -0.4444444444444441],
              [  3.3333333333333326,   3.3333333333333321,
               -11.9999999999999964,   2.666666666666667 ,
                 2.6666666666666656],
              [ -0.4444444444444447,   0.                ,   2.666666666666667,
                -2.2222222222222223,   0.                ],
              [  0.                ,  -0.4444444444444439,
                 2.6666666666666656,   0.                ,  -2.2222222222222214]])

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M.T, rtol, atol)
        # We test only the mass-Hdiv part
        assert np.allclose(M, M_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_dual_vem_2d_iso_cart_surf(self):
        g = structured.CartGrid([2, 1], [1, 1])
        R = cg.rot(np.pi/4., [0,1,0])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry(is_embedded=True)

        kxx = np.ones(g.num_cells)
        perm = sot.SecondOrderTensor(g.dim, kxx)

        solver = hybrid.HybridDualVEM()
        data = {'k': perm, 'f': np.zeros(g.num_cells)}
        M = solver.matrix_rhs(g, data)[0].todense()

        M_known = np.array([[-2.25,  1.75,  0.  ,  0.25,  0.  ,  0.25,  0.  ],
                            [ 1.75, -4.5 ,  1.75,  0.25,  0.25,  0.25,  0.25],
                            [ 0.  ,  1.75, -2.25,  0.  ,  0.25,  0.  ,  0.25],
                            [ 0.25,  0.25,  0.  , -0.75,  0.  ,  0.25,  0.  ],
                            [ 0.  ,  0.25,  0.25,  0.  , -0.75,  0.  ,  0.25],
                            [ 0.25,  0.25,  0.  ,  0.25,  0.  , -0.75,  0.  ],
                            [ 0.  ,  0.25,  0.25,  0.  ,  0.25,  0.  , -0.75]])

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M.T, rtol, atol)
        assert np.allclose(M, M_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_dual_vem_1d_iso_line(self):
        g = structured.CartGrid(3, 1)
        R = cg.rot(np.pi/6., [0,0,1])
        g.nodes = np.dot(R, g.nodes)
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = sot.SecondOrderTensor(g.dim, kxx)

        solver = hybrid.HybridDualVEM()
        data = {'k': perm, 'f': np.zeros(g.num_cells)}
        M = solver.matrix_rhs(g, data)[0].todense()

        # Matrix computed with an already validated code (MRST)
        M_known = np.array([[-3.,  3.,  0.,  0.],
                            [ 3., -6.,  3.,  0.],
                            [ 0.,  3., -6.,  3.],
                            [ 0.,  0.,  3., -3.]])

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M.T, rtol, atol)
        assert np.allclose(M, M_known, rtol, atol)

#------------------------------------------------------------------------------#
