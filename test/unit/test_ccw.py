import unittest

import numpy as np

import porepy as pp


class TestCCW(unittest.TestCase):
    def setup(self):
        p1 = np.array([0, 0])
        p2 = np.array([1, 0])
        p3 = np.array([1, 1])
        return p1, p2, p3

    def setup_close(self, y):
        p1 = np.array([0, 0])
        p2 = np.array([2, 0])
        p3 = np.array([1, y])
        return p1, p2, p3

    def test_is_ccw(self):
        p1, p2, p3 = self.setup()
        self.assertTrue(pp.geometry_property_checks.is_ccw_polyline(p1, p2, p3))

    def test_not_ccw(self):
        p1, p2, p3 = self.setup()
        self.assertTrue(not pp.geometry_property_checks.is_ccw_polyline(p1, p3, p2))

    def test_on_boundary(self):
        p1, p2, p3 = self.setup_close(0)
        self.assertTrue(
            pp.geometry_property_checks.is_ccw_polyline(p1, p2, p3, default=True)
        )
        self.assertTrue(
            not pp.geometry_property_checks.is_ccw_polyline(p1, p2, p3, default=False)
        )

    def test_tolerance(self):
        p1, p2, p3 = self.setup_close(1e-6)

        # Safety margin saves ut
        self.assertTrue(
            pp.geometry_property_checks.is_ccw_polyline(
                p1, p2, p3, tol=1e-4, default=True
            )
        )

        # Default kills us, even though we're inside safety margin
        self.assertTrue(
            not pp.geometry_property_checks.is_ccw_polyline(
                p1, p2, p3, tol=1e-4, default=False
            )
        )

        # Outside safety margin, and on the ccw side
        self.assertTrue(
            pp.geometry_property_checks.is_ccw_polyline(
                p1, p2, p3, tol=1e-8, default=False
            )
        )

    def test_tolerance_outside(self):
        p1, p2, p3 = self.setup_close(-1e-6)

        # Safety margin saves ut
        self.assertTrue(
            pp.geometry_property_checks.is_ccw_polyline(
                p1, p2, p3, tol=1e-4, default=True
            )
        )

        # Default kills us, even though we're inside safety margin
        self.assertTrue(
            not pp.geometry_property_checks.is_ccw_polyline(
                p1, p2, p3, tol=1e-4, default=False
            )
        )

        # Outside safety margin, and not on the ccw side
        self.assertTrue(
            not pp.geometry_property_checks.is_ccw_polyline(
                p1, p2, p3, tol=1e-8, default=False
            )
        )

    def test_several_points(self):
        p1, p2, _ = self.setup()

        p_test = np.array([[0.5, 0.5], [1, -1]])
        known = np.array([1, 0], dtype=np.bool)
        self.assertTrue(
            np.allclose(
                known,
                pp.geometry_property_checks.is_ccw_polyline(
                    p1, p2, p_test, tol=1e-8, default=False
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
