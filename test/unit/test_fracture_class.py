import unittest
import numpy as np

import porepy as pp


class TestFractureCenters(unittest.TestCase):
    """ Test computation of fracture centroids.
    """

    def test_frac_1(self):
        # Simple plane, known center.
        f_1 = pp.Fracture(
            np.array([[0, 2, 2, 0], [0, 2, 2, 0], [-1, -1, 1, 1]]),
            check_convexity=False,
        )
        c_known = np.array([1, 1, 0]).reshape((3, 1))
        assert np.allclose(c_known, f_1.center)

    def test_frac_2(self):
        # Center away from xy-plane
        f_1 = pp.Fracture(
            np.array([[0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 1, 1]]), check_convexity=False
        )
        c_known = np.array([1, 1, 0.5]).reshape((3, 1))
        assert np.allclose(c_known, f_1.center)

    def test_frac_3(self):
        # Fracture plane defined by x + y + z = 1
        f_1 = pp.Fracture(
            np.array([[0, 1, 1, 0], [0, 0, 1, 1], [1, 0, -1, 0]]), check_convexity=False
        )
        c_known = np.array([0.5, 0.5, 0]).reshape((3, 1))
        assert np.allclose(c_known, f_1.center)

    def test_frac_4(self):
        # Fracture plane defined by x + y + z = 4
        f_1 = pp.Fracture(
            np.array([[0, 1, 1, 0], [0, 0, 1, 1], [4, 3, 2, 3]]), check_convexity=False
        )
        c_known = np.array([0.5, 0.5, 3]).reshape((3, 1))
        assert np.allclose(c_known, f_1.center)

    def test_frac_rand(self):
        # Random normal vector.
        r = np.random.rand(4)
        x = np.array([0, 1, 1, 0])
        y = np.array([0, 0, 1, 1])
        z = (r[0] - r[1] * x - r[2] * y) / r[3]
        f = pp.Fracture(np.vstack((x, y, z)), check_convexity=False)
        z_cc = (r[0] - 0.5 * r[1] - 0.5 * r[2]) / r[3]
        c_known = np.array([0.5, 0.5, z_cc]).reshape((3, 1))
        assert np.allclose(c_known, f.center)


class TestFractureIndex(unittest.TestCase):
    """ Test index attribute of Fracture.

    The attribute may need to change later, these tests should ensure that we
    don't rush it.
    """

    def array(self):
        return np.array([[0, 1, 0], [0, 0, 0], [0, 0, 1]])

    def test_equal_index(self):
        f1 = pp.Fracture(self.array(), index=1, check_convexity=False)
        f2 = pp.Fracture(self.array(), index=1, check_convexity=False)
        assert f1 == f2

    def test_unequal_index(self):
        f1 = pp.Fracture(self.array(), index=1, check_convexity=False)
        f2 = pp.Fracture(self.array(), index=4, check_convexity=False)
        assert f1 != f2

    def test_no_index(self):
        # One fracture has no index set. Should not be equal
        f1 = pp.Fracture(self.array(), index=1, check_convexity=False)
        f2 = pp.Fracture(self.array(), check_convexity=False)
        assert f1 != f2


class TestFractureIsVertex(unittest.TestCase):
    """ Test of is_vertex function in Fractures.
    """

    def frac(self):
        return pp.Fracture(
            np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]), check_convexity=False
        )

    def test_not_vertex(self):
        p = np.array([0.5, 0.5, 0])
        f = self.frac()
        is_vert, ind = f.is_vertex(p)
        assert is_vert == False
        assert ind is None

    def test_is_vertex(self):
        p = np.array([0., 0., 0])
        f = self.frac()
        is_vert, ind = f.is_vertex(p)
        assert is_vert == True
        assert ind == 0

    def test_tolerance_sensitivity(self):
        # Borderline, is a vertex depending on the tolerance
        p = np.array([1e-4, 0, 0])
        f = self.frac()
        is_vert, ind = f.is_vertex(p, tol=1e-5)
        assert is_vert == False
        assert ind is None

        is_vert, ind = f.is_vertex(p, tol=1e-3)
        assert is_vert == True
        assert ind == 0


class TestCopyFracture(unittest.TestCase):
    def frac(self):
        return pp.Fracture(
            np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]), check_convexity=False
        )

    def test_copy(self):
        f1 = self.frac()
        f2 = f1.copy()
        assert id(f1) != id(f2)

    def test_deep_copy(self):
        f1 = self.frac()
        f2 = f1.copy()

        # Points should be identical
        assert np.allclose(f1.p, f2.p)

        f2.p[0, 0] = 7
        assert not np.allclose(f1.p, f2.p)


class TestReprStr(unittest.TestCase):
    def frac(self):
        return pp.Fracture(
            np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]), check_convexity=False
        )

    def test_str(self):
        f = self.frac()
        f.__str__()

    def test_repr(self):
        f = self.frac()
        f.__repr__()


if __name__ == "__main__":
    unittest.main()
