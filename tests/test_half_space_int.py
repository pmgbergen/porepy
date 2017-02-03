import unittest
import numpy as np

from utils import half_space_int

#------------------------------------------------------------------------------#

class BasicsTest( unittest.TestCase ):

#------------------------------------------------------------------------------#

    def test_one_half_space(self):

        n  = np.array([[-1],[0],[0]])
        x0 = np.array([[0],[0],[0]])
        pts = np.array([[1,-1],[0,0],[0,0]])
        out = half_space_int.half_space_int(n,x0,pts)
        assert np.all(out == np.array([True,False]))

#------------------------------------------------------------------------------#

    def test_two_half_spaces(self):
        n  = np.array([[-1,0],[0,-1],[0,0]])
        x0 = np.array([[0,0],[0,1],[0,0]])
        pts = np.array([[1,-1,1,0],[2,0,2,0],[0,0,0,0]])
        out = half_space_int.half_space_int(n,x0,pts)
        assert np.all(out == np.array([True,False,True,False]))

#------------------------------------------------------------------------------#
