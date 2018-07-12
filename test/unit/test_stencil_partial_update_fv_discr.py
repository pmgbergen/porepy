import unittest
import numpy as np

from porepy.grids.structured import CartGrid
from porepy.numerics.fv import fvutils


class TestCellIndForPartialUpdate(unittest.TestCase):
    def setUp(self):
        self.g_2d = CartGrid([5, 5])
        self.g_3d = CartGrid([3, 3, 3])

    def test_node_based_ind_2d(self):
        # Nodes of cell 12 (middle one) - from counting
        n = np.array([14, 15, 20, 21])

        known_cells = np.array([6, 7, 8, 11, 12, 13, 16, 17, 18])
        known_faces = np.array([14, 15, 42, 47])

        cell_ind, face_ind = fvutils.cell_ind_for_partial_update(self.g_2d, nodes=n)
        assert np.allclose(known_cells, cell_ind)
        assert np.allclose(known_faces, face_ind)

    def test_node_based_ind_2d_bound(self):
        # Nodes of cell 1
        n = np.array([1, 2, 7, 8])
        known_cells = np.array([0, 1, 2, 5, 6, 7])
        known_faces = np.array([1, 2, 31, 36])

        cell_ind, face_ind = fvutils.cell_ind_for_partial_update(self.g_2d, nodes=n)

        assert np.alltrue(known_cells == cell_ind)
        assert np.alltrue(known_faces == face_ind)

    def test_node_based_ind_3d(self):
        # Nodes of cell 13 (middle one) - from counting
        n = np.array([21, 22, 25, 26, 37, 38, 41, 42])

        known_cells = np.arange(27)
        known_faces = np.array([17, 18, 52, 55, 85, 94])

        cell_ind, face_ind = fvutils.cell_ind_for_partial_update(self.g_3d, nodes=n)

        assert np.alltrue(known_cells == cell_ind)
        assert np.alltrue(known_faces == face_ind)

    def test_node_based_ind_3d_bound(self):
        # Nodes of cell 1
        n = np.array([1, 2, 5, 6, 17, 18, 21, 22])
        known_cells = np.array([0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14])
        known_faces = np.array([1, 2, 37, 40, 73, 82])

        cell_ind, face_ind = fvutils.cell_ind_for_partial_update(self.g_3d, nodes=n)

        assert np.alltrue(known_cells == cell_ind)
        assert np.alltrue(known_faces == face_ind)

        cell_ind, face_ind = fvutils.cell_ind_for_partial_update(self.g_3d, nodes=n)

        assert np.alltrue(known_cells == cell_ind)
        assert np.alltrue(known_faces == face_ind)

    def test_cell_based_ind_2d(self):

        c = np.array([12])
        known_cells = np.setdiff1d(np.arange(25), np.array([0, 4, 20, 24]))
        known_faces = np.array([8, 9, 14, 15, 20, 21, 41, 42, 43, 46, 47, 48])

        cell_ind, face_ind = fvutils.cell_ind_for_partial_update(self.g_2d, cells=c)

        assert np.alltrue(known_cells == cell_ind)
        assert np.alltrue(known_faces == face_ind)

    def test_cell_based_ind_3d(self):
        # Use cell 13 (middle one)
        c = np.array([13])
        known_cells = np.arange(27)
        fx = np.hstack(
            (
                np.array([1, 2, 5, 6, 9, 10]),
                np.array([1, 2, 5, 6, 9, 10]) + 12,
                np.array([1, 2, 5, 6, 9, 10]) + 24,
            )
        )
        fy = 36 + np.hstack(
            (
                np.array([3, 4, 5, 6, 7, 8]),
                np.array([3, 4, 5, 6, 7, 8]) + 12,
                np.array([3, 4, 5, 6, 7, 8]) + 24,
            )
        )
        fz = 72 + np.hstack((np.arange(9) + 9, np.arange(9) + 18))
        known_faces = np.hstack((fx, fy, fz))

        cell_ind, face_ind = fvutils.cell_ind_for_partial_update(self.g_3d, cells=c)

        assert np.alltrue(known_cells == cell_ind)
        assert np.alltrue(known_faces == face_ind)

    def test_cell_based_ind_bound_3d(self):
        c = np.array([1])
        known_cells = np.arange(27)
        fx = np.array([1, 2, 5, 6, 13, 14, 17, 18])
        fy = 36 + np.array([0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17])
        fz = 72 + np.array([0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14])
        known_faces = np.hstack((fx, fy, fz))
        cell_ind, face_ind = fvutils.cell_ind_for_partial_update(self.g_3d, cells=c)

        assert np.alltrue(known_cells == cell_ind)
        assert np.alltrue(known_faces == face_ind)

    def test_face_based_ind_2d(self):

        # Use face between cells 11 and 12
        f = np.array([14])

        known_cells = np.array(
            [1, 2, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 21, 22]
        )
        known_faces = np.array([8, 14, 20, 41, 42, 46, 47])
        cell_ind, face_ind = fvutils.cell_ind_for_partial_update(self.g_2d, faces=f)

        assert np.alltrue(known_cells == cell_ind)
        assert np.alltrue(known_faces == face_ind)

    def test_face_based_ind_2d_bound(self):
        f = np.array([2])
        known_cells = np.array([0, 1, 2, 3, 5, 6, 7, 8, 11, 12])
        known_faces = np.array([2, 8, 31, 32, 36, 37])
        cell_ind, face_ind = fvutils.cell_ind_for_partial_update(self.g_2d, faces=f)

        assert np.alltrue(known_cells == cell_ind)
        assert np.alltrue(known_faces == face_ind)

    if __name__ == "__main__":
        unittest.main()
