import unittest
import numpy as np

from gridding.fractured.fractures import Fracture

class TestFractureCenters(unittest.TestCase):

    def test_frac_1(self):
        f_1 = Fracture(np.array([[0, 2, 2, 0],
                                 [0, 2, 2, 0],
                                 [-1, -1, 1, 1]]))
        c_known = np.array([1, 1, 0])
        assert np.allclose(c_known, f_1.center)

    def test_frac_2(self):
        f_1 = Fracture(np.array([[0, 2, 2, 0],
                                 [0, 2, 2, 0],
                                 [0, 0, 1, 1]]))
        c_known = np.array([1, 1, 0.5])
        assert np.allclose(c_known, f_1.center)

    def test_frac_3(self):
        # Fracture plane defined by x + y + z = 1
        f_1 = Fracture(np.array([[0, 1, 1, 0],
                                 [0, 0, 1, 1],
                                 [1, 0, -1, 0]]))
        c_known = np.array([0.5, 0.5, 0])
        assert np.allclose(c_known, f_1.center)

    def test_frac_4(self):
        # Fracture plane defined by x + y + z = 4
        f_1 = Fracture(np.array([[0, 1, 1, 0],
                                 [0, 0, 1, 1],
                                 [4, 3, 2, 3]]))
        c_known = np.array([0.5, 0.5, 3])
        assert np.allclose(c_known, f_1.center)

    def test_frac_rand(self):
        r = np.random.rand(4)
        x = np.array([0, 1, 1, 0])
        y = np.array([0, 0, 1, 1])
        z = (r[0] - r[1] * x - r[2] * y) / r[3]
        f = Fracture(np.vstack((x, y, z)))
        z_cc = (r[0] - 0.5 * r[1] - 0.5 * r[2]) / r[3]
        c_known = np.array([0.5, 0.5, z_cc])
        assert np.allclose(c_known, f.center)

