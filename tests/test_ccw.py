import numpy as np
import unittest

from compgeom import basics as cg


class TestCCW(unittest.TestCase):
    
    def setup(self):
        p1 = np.array([0, 0])
        p2 = np.array([1, 0])
        p3 = np.array([1, 1])
        return p1, p2, p3

    def test_is_ccw(self):
        p1, p2, p3 = self.setup()
        assert cg.is_ccw(p1, p2, p3)

    def test_not_ccw(self):
        p1, p2, p3 = self.setup()
        assert not cg.is_ccw(p1, p3, p2)

    if __name__ == '__main__':
        unittest.main()

