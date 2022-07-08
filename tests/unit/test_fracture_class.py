import unittest

import numpy as np

import porepy as pp


class TestFractureGeometry(unittest.TestCase):
    """Test computation of fracture centroids."""

    def test_frac_3d_1(self):
        # Simple plane, known center.
        f_1 = pp.PlaneFracture(
            np.array([[0, 2, 2, 0], [0, 2, 2, 0], [-1, -1, 1, 1]]),
            check_convexity=False,
        )
        c_known = np.array([1, 1, 0]).reshape((3, 1))
        n_known = np.array([1, -1, 0]).reshape((1, 3))
        self.assertTrue(np.allclose(c_known, f_1.center))
        self.assertTrue(np.allclose(np.cross(n_known, f_1.normal.T), 0))

    def test_frac_3d_2(self):
        # Center away from xy-plane
        f_1 = pp.PlaneFracture(
            np.array([[0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 1, 1]]), check_convexity=False
        )
        c_known = np.array([1, 1, 0.5]).reshape((3, 1))
        self.assertTrue(np.allclose(c_known, f_1.center))

    def test_frac_3d_3(self):
        # Fracture plane defined by x + y + z = 1
        f_1 = pp.PlaneFracture(
            np.array([[0, 1, 1, 0], [0, 0, 1, 1], [1, 0, -1, 0]]), check_convexity=False
        )
        c_known = np.array([0.5, 0.5, 0]).reshape((3, 1))
        self.assertTrue(np.allclose(c_known, f_1.center))

    def test_frac_3d_4(self):
        # Fracture plane defined by x + y + z = 4
        f_1 = pp.PlaneFracture(
            np.array([[0, 1, 1, 0], [0, 0, 1, 1], [4, 3, 2, 3]]), check_convexity=False
        )
        c_known = np.array([0.5, 0.5, 3]).reshape((3, 1))
        self.assertTrue(np.allclose(c_known, f_1.center))

    def test_frac_3d_rand(self):
        # Random normal vector.
        r = np.random.rand(4)
        x = np.array([0, 1, 1, 0])
        y = np.array([0, 0, 1, 1])
        z = (r[0] - r[1] * x - r[2] * y) / r[3]
        f = pp.PlaneFracture(np.vstack((x, y, z)), check_convexity=False)
        z_cc = (r[0] - 0.5 * r[1] - 0.5 * r[2]) / r[3]
        c_known = np.array([0.5, 0.5, z_cc]).reshape((3, 1))
        self.assertTrue(np.allclose(c_known, f.center))

    def test_frac_2d(self):
        # Simple line, known center.
        f_1 = pp.LineFracture(
            np.array([[0, 2], [0, 2]]),
        )
        c_known = np.array([1, 1]).reshape((2, 1))
        n_known = np.array([1, -1]).reshape((1, 2))
        self.assertTrue(np.allclose(c_known, f_1.center))
        self.assertTrue(np.allclose(np.cross(n_known, f_1.normal.T), 0))


class TestFractureIndex(unittest.TestCase):
    """Test index attribute of Fracture.

    The attribute may need to change later, these tests should ensure that we
    don't rush it.
    """

    def array(self):
        return np.array([[0, 1, 0], [0, 0, 0], [0, 0, 1]])

    def test_equal_index(self):
        f1 = pp.PlaneFracture(self.array(), index=1, check_convexity=False)
        f2 = pp.PlaneFracture(self.array(), index=1, check_convexity=False)
        self.assertTrue(f1 == f2)

    def test_unequal_index(self):
        f1 = pp.PlaneFracture(self.array(), index=1, check_convexity=False)
        f2 = pp.PlaneFracture(self.array(), index=4, check_convexity=False)
        self.assertTrue(f1 != f2)

    def test_no_index(self):
        # One fracture has no index set. Should not be equal
        f1 = pp.PlaneFracture(self.array(), index=1, check_convexity=False)
        f2 = pp.PlaneFracture(self.array(), check_convexity=False)
        self.assertTrue(f1 != f2)

    def test_2d(self):
        f1 = pp.LineFracture(self.array()[:2, :2], index=1)
        f2 = pp.LineFracture(self.array()[:2, :2], index=1)
        f3 = pp.LineFracture(self.array()[:2, :2], index=4)
        self.assertTrue(f1 == f2)
        self.assertTrue(f1 != f3)


class TestFractureIsVertex(unittest.TestCase):
    """Test of is_vertex function in Fractures."""

    def frac(self):
        return pp.PlaneFracture(
            np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]), check_convexity=False
        )

    def test_not_vertex(self):
        p = np.array([0.5, 0.5, 0])
        f = self.frac()
        is_vert, ind = f.is_vertex(p)
        self.assertTrue(is_vert == False)
        self.assertTrue(ind is None)

    def test_is_vertex(self):
        p = np.array([0.0, 0.0, 0])
        f = self.frac()
        is_vert, ind = f.is_vertex(p)
        self.assertTrue(is_vert == True)
        self.assertTrue(ind == 0)

    def test_tolerance_sensitivity(self):
        # Borderline, is a vertex depending on the tolerance
        p = np.array([1e-4, 0, 0])
        f = self.frac()
        is_vert, ind = f.is_vertex(p, tol=1e-5)
        self.assertTrue(is_vert == False)
        self.assertTrue(ind is None)

        is_vert, ind = f.is_vertex(p, tol=1e-3)
        self.assertTrue(is_vert == True)
        self.assertTrue(ind == 0)


class TestCopyFracture(unittest.TestCase):
    def frac(self):
        return pp.PlaneFracture(
            np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]), check_convexity=False
        )

    def test_copy(self):
        f1 = self.frac()
        f2 = f1.copy()
        self.assertTrue(id(f1) != id(f2))

    def test_deep_copy(self):
        f1 = self.frac()
        f2 = f1.copy()

        # Points should be identical
        self.assertTrue(np.allclose(f1.pts, f2.pts))

        f2.pts[0, 0] = 7
        self.assertTrue(not np.allclose(f1.pts, f2.pts))


class TestReprStr(unittest.TestCase):
    def frac(self):
        return pp.PlaneFracture(
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
