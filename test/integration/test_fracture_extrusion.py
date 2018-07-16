import unittest
import numpy as np

from porepy.fracs import extrusion


class TestFractureExtrusion(unittest.TestCase):
    def test_cut_off(self):
        # Two fractures, the second abutting. Check that the second fracture is
        # cut correctly.
        tol = 1e-4
        p = np.array([[0, 0], [2, 0], [1, 0], [1, 10]]).T
        e = np.array([[0, 1], [2, 3]]).T

        fractures = extrusion.fractures_from_outcrop(p, e, tol=tol)
        assert np.all(fractures[1].p[1] > -tol)

    def test_not_aligned_outcrop(self):
        # The fractures are not aligned with the mesh
        p = np.array([[0, 0], [2, 2], [1, 1], [1, 3]]).T
        e = np.array([[0, 1], [2, 3]]).T

        tol = 1e-4

        fractures = extrusion.fractures_from_outcrop(p, e, tol=tol)
        #
        assert np.all(fractures[1].p[1] >= fractures[1].p[0] - tol)

    def test_no_family_given(self):
        # Assign fracture family automatically
        tol = 1e-2
        p = np.array([[0, 0], [2, 0], [1, 0], [1, 3]]).T
        e = np.array([[0, 1], [2, 3]]).T

        fractures = extrusion.fractures_from_outcrop(p, e, tol=tol)
        assert np.all(fractures[1].p[1] >= -tol)

    if __name__ == "__main__":
        unittest.main()
