import numpy as np
import unittest

from core.grids import structured, simplex
from core.constit import second_order_tensor as sot

from vem import dual

#------------------------------------------------------------------------------#

class BasicsTest( unittest.TestCase ):

#------------------------------------------------------------------------------#

    def test_dual_vem_2d_iso_cart(self):
        g = structured.CartGrid([2, 1], [1, 1])
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = sot.SecondOrderTensor(g.dim, kxx)
        M = dual.matrix(g, perm).todense()

        # Matrix computed with an already validated code (MRST)
        M_known = np.array( [[ 0.625, -0.375,      0, 0, 0,  0,  0,  1,  0],
                             [-0.375,   1.25, -0.375, 0, 0,  0,  0, -1,  1],
                             [     0, -0.375,  0.625, 0, 0,  0,  0,  0, -1],
                             [     0,      0,      0, 1, 0,  0,  0,  1,  0],
                             [     0,      0,      0, 0, 1,  0,  0,  0,  1],
                             [     0,      0,      0, 0, 0,  1,  0, -1,  0],
                             [     0,      0,      0, 0, 0,  0,  1,  0, -1],
                             [     1,     -1,      0, 1, 0, -1,  0,  0,  0],
                             [     0,      1,     -1, 0, 1,  0, -1,  0,  0]] )

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M.T, rtol, atol)
        assert np.allclose(M, M_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_dual_vem_2d_ani_cart(self):
        g = structured.CartGrid([2, 1], [1, 1])
        g.compute_geometry()

        kxx = np.square(g.cell_centers[1,:])+1
        kyy = np.square(g.cell_centers[0,:])+1
        kxy =-np.multiply(g.cell_centers[0,:], g.cell_centers[1,:])
        perm = sot.SecondOrderTensor(g.dim, kxx=kxx, kyy=kyy, kxy=kxy)
        M = dual.matrix(g, perm).todense()

        # Matrix computed with an already validated code (MRST)
        M_known = np.array([[0.625000000000000, -0.422619047619048, 0,
                             0.023809523809524, 0, 0.023809523809524, 0, 1, 0],
                            [-0.422619047619048, 1.267241379310345,
                             -0.426724137931035, 0.023809523809524,
                             0.051724137931034, 0.023809523809524,
                             0.051724137931034, -1, 1],
                            [0, -0.426724137931035, 0.642241379310345, 0,
                             0.051724137931034, 0, 0.051724137931034, 0, -1],
                            [0.023809523809524, 0.023809523809524, 0, 1, 0,
                             -0.047619047619048, 0, 1, 0],
                            [0, 0.051724137931034, 0.051724137931034, 0,
                             0.879310344827586, 0, -0.189655172413793, 0, 1],
                            [0.023809523809524, 0.023809523809524, 0,
                             -0.047619047619048, 0, 1, 0, -1, 0],
                            [0, 0.051724137931034, 0.051724137931034, 0,
                             -0.189655172413793, 0, 0.879310344827586, 0, -1],
                            [1, -1, 0, 1, 0, -1, 0, 0, 0],
                            [0, 1, -1, 0, 1, 0, -1, 0, 0]])

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M.T, rtol, atol)
        assert np.allclose(M, M_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_dual_vem_2d_iso_simplex(self):
        g = simplex.StructuredTriangleGrid([1, 1], [1, 1])
        g.compute_geometry()

        kxx = np.ones(g.num_cells)
        perm = sot.SecondOrderTensor(g.dim, kxx)
        M = dual.matrix(g, perm).todense()

        # Matrix computed with an already validated code (MRST)
        faces = np.arange(5)
        map_faces = np.array([1, 4, 0, 2, 3])
        M_known = np.array([[0.888888888888889, 0.277777777777778,
                             0.277777777777778, -0.277777777777778,
                             -0.277777777777778, -1.000000000000000, 1],
                            [0.277777777777778, 0.611111111111111,
                             0.111111111111111, 0, 0, -1, 0],
                            [0.277777777777778, 0.111111111111111,
                             0.611111111111111, 0, 0, -1, 0],
                            [-0.277777777777778, 0, 0, 0.611111111111111,
                             0.111111111111111, 0, -1],
                            [-0.277777777777778, 0, 0, 0.111111111111111,
                             0.611111111111111, 0, 1],
                            [-1, -1, -1, 0, 0, 0, 0],
                            [1, 0, 0, -1, -1, 0, 0]])

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M.T, rtol, atol)
        # We test only the mass-Hdiv part
        assert np.allclose(M[np.ix_(faces, faces)],
                           M_known[np.ix_(map_faces, map_faces)], rtol, atol)

#------------------------------------------------------------------------------#

    def test_dual_vem_2d_iso_simplex(self):
        g = simplex.StructuredTriangleGrid([1, 1], [1, 1])
        g.compute_geometry()

        kxx = np.square(g.cell_centers[1,:])+1
        kyy = np.square(g.cell_centers[0,:])+1
        kxy =-np.multiply(g.cell_centers[0,:], g.cell_centers[1,:])
        perm = sot.SecondOrderTensor(g.dim, kxx=kxx, kyy=kyy, kxy=kxy)
        M = dual.matrix(g, perm).todense()

        # Matrix computed with an already validated code (MRST)
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

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M.T, rtol, atol)
        # We test only the mass-Hdiv part
        assert np.allclose(M[np.ix_(faces, faces)],
                           M_known[np.ix_(map_faces, map_faces)], rtol, atol)

#------------------------------------------------------------------------------#
