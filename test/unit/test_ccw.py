import numpy as np
import unittest

from porepy.utils import comp_geom as cg


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
        assert cg.is_ccw_polyline(p1, p2, p3)

    def test_not_ccw(self):
        p1, p2, p3 = self.setup()
        assert not cg.is_ccw_polyline(p1, p3, p2)

    def test_tolerance(self):
        p1, p2, p3 = self.setup_close(1e-6)

        # Safety margin saves ut
        assert cg.is_ccw_polyline(p1, p2, p3, tol=1e-4, default=True)

        # Default kills us, even though we're inside safety margin
        assert not cg.is_ccw_polyline(p1, p2, p3, tol=1e-4, default=False)

        # Outside safety margin, and on the ccw side
        assert cg.is_ccw_polyline(p1, p2, p3, tol=1e-8, default=False)

    def test_tolerance_outside(self):
        p1, p2, p3 = self.setup_close(-1e-6)

        # Safety margin saves ut
        assert cg.is_ccw_polyline(p1, p2, p3, tol=1e-4, default=True)

        # Default kills us, even though we're inside safety margin
        assert not cg.is_ccw_polyline(p1, p2, p3, tol=1e-4, default=False)

        # Outside safety margin, and not on the ccw side
        assert not cg.is_ccw_polyline(p1, p2, p3, tol=1e-8, default=False)

    if __name__ == "__main__":
        unittest.main()
