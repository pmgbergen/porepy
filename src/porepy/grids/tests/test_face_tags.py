import unittest
import numpy as np

from gridding.fractured import structured
from gridding.fractured import meshing
from core.grids.grid import FaceTag


class TestFaceTags(unittest.TestCase):
    def test_x_intersection_2d(self):
        """ Check that the faces has correct tags for a 2D grid.
        """

        f_1 = np.array([[0, 2], [1, 1]])
        f_2 = np.array([[1, 1], [0, 2]])

        f_set = [f_1, f_2]
        nx = [3, 3]

        grids = meshing.cart_grid(f_set, nx, physdims=nx)

        # 2D grid:
        g_2d = grids.grids_of_dimension(2)[0]

        f_tags_2d = np.array([False, True, False, False,   # first row
                              False, True, False, False,   # Second row
                              False, False, False, False,  # third row
                              False, False, False,         # Bottom column
                              True, True, False,           # Second column
                              False, False, False,         # Third column
                              False, False, False,         # Top column
                              True, True, True, True])     # Added faces

        d_tags_2d = np.array([True, False, False, True,    # first row
                              True, False, False, True,    # Second row
                              True, False, False, True,    # third row
                              True, True, True,            # Bottom column
                              False, False, False,         # Second column
                              False, False, False,         # Third column
                              True, True, True,            # Top column
                              False, False, False, False])  # Added Faces
        t_tags_2d = np.zeros(f_tags_2d.size, dtype=bool)
        b_tags_1d = np.sum(f_tags_2d + d_tags_2d +
                           t_tags_2d, axis=0).astype(bool)
        b_tags_2d = (f_tags_2d + d_tags_2d + t_tags_2d).astype(bool)
        assert np.all(g_2d.has_face_tag(FaceTag.TIP) == t_tags_2d)
        assert np.all(g_2d.has_face_tag(FaceTag.FRACTURE) == f_tags_2d)
        assert np.all(g_2d.has_face_tag(FaceTag.DOMAIN_BOUNDARY) == d_tags_2d)
        assert np.all(g_2d.has_face_tag(FaceTag.BOUNDARY) == b_tags_2d)

        # 1D grids:
        for g_1d in grids.grids_of_dimension(1):
            f_tags_1d = np.array([False, True, False, True])
            if g_1d.face_centers[0, 0] > 0.1:
                t_tags_1d = np.array([True, False, False, False])
                d_tags_1d = np.array([False, False, True, False])
            else:
                t_tags_1d = np.array([False, False, True, False])
                d_tags_1d = np.array([True, False, False, False])

            b_tags_1d = np.sum(f_tags_1d + d_tags_1d +
                               t_tags_1d, axis=0).astype(bool)

            assert np.all(g_1d.has_face_tag(FaceTag.TIP) == t_tags_1d)
            assert np.all(g_1d.has_face_tag(FaceTag.FRACTURE) == f_tags_1d)
            assert np.all(g_1d.has_face_tag(
                FaceTag.DOMAIN_BOUNDARY) == d_tags_1d)
            assert np.all(g_1d.has_face_tag(FaceTag.BOUNDARY) == b_tags_1d)
