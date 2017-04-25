import unittest
import numpy as np

from porepy.utils import half_space

#------------------------------------------------------------------------------#

class BasicsTest( unittest.TestCase ):

#------------------------------------------------------------------------------#

    def test_one_half_space(self):

        n  = np.array([[-1],[0],[0]])
        x0 = np.array([[0],[0],[0]])
        pts = np.array([[1,-1],[0,0],[0,0]])
        out = half_space.half_space_int(n,x0,pts)
        assert np.all(out == np.array([True,False]))

#------------------------------------------------------------------------------#

    def test_two_half_spaces(self):
        n  = np.array([[-1,0],[0,-1],[0,0]])
        x0 = np.array([[0,0],[0,1],[0,0]])
        pts = np.array([[1,-1,1,0],[2,0,2,0],[0,0,0,0]])
        out = half_space.half_space_int(n,x0,pts)
        assert np.all(out == np.array([True,False,True,False]))

#------------------------------------------------------------------------------#

    def test_half_space_pt_convex_2d(self):
        n = np.array([[0,0,0,0,1,-1],[1,1,-1,-1,0,0],[0,0,0,0,0,0]])
        x0 = np.array([[0,0,0,0,0,2/3],[0,0,1/3,1/3,0,0],[0,0,0,0,0,0]])
        pts = np.array([[0,2/3],[0,1/3],[0,0]])
        pt = half_space.half_space_pt(n, x0, pts)
        assert np.allclose(pt,[1/6,1/6,0])

#------------------------------------------------------------------------------#

    def test_half_space_pt_star_shaped_2d(self):
        n = np.array([[0,0,1,0,-1,0,1],[1,1,0,1,0,-1,0],[0,0,0,0,0,0,0]])
        x0 = np.array([[0,0,2/3,0,1,0,0],[1/3,1/3,0,0,0,2/3,0],[0,0,0,0,0,0,0]])
        pts = np.array([[0,1],[0,2/3],[0,0]])
        pt = half_space.half_space_pt(n, x0, pts)
        assert np.allclose(pt,[5/6,1/2,0])

#------------------------------------------------------------------------------#

    def test_half_space_pt_convex_3d(self):
        n = np.array([[1,-1,0,0,0,0],[0,0,1,-1,0,0],[0,0,0,0,1,-1]])
        x0 = np.array([[0,1,0,0,0,0],[0,0,0,1,0,0],[0,0,0,0,0,1]])
        pts = np.array([[1,0,0],[0,1,0],[0,0,1]])
        pt = half_space.half_space_pt(n, x0, pts)
        assert np.allclose(pt,[1/2,1/2,1/2])

#------------------------------------------------------------------------------#

    def test_half_space_pt_star_shaped_3d(self):
        n = np.array([[0,0,1,0,-1,0,1,0,0],[1,1,0,1,0,-1,0,0,0],
                      [0,0,0,0,0,0,0,1,-1]])
        x0 = np.array([[0,0,2/3,0,1,0,0,0,0],[1/3,1/3,0,0,0,2/3,0,0,0],
                       [0,0,0,0,0,0,0,0,1]])
        pts = np.array([[0,1],[0,2/3],[0,1]])
        pt = half_space.half_space_pt(n, x0, pts)
        assert np.allclose(pt,[5/6,1/2,1/6])

#------------------------------------------------------------------------------#
