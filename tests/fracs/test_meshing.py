"""Tests for the fracs.meshing module.

So far, only a test of the function _tag_faces() is implemented.

"""
import unittest

import numpy as np

from porepy.fracs import meshing


def test_fracture_and_boundary_face_tags_2d_domain_x_intersection():
    """For a 2d Cartesian grid with two intersecting fractures, verify that the
    boundary, fracture, and fracture tip face tags are correctly set.

    The test is based on comparing the constructed and known tags.

    """

    # Define fractures, domain boundaries, and grid
    f_1 = np.array([[0, 2], [1, 1]])
    f_2 = np.array([[1, 1], [0, 2]])

    f_set = [f_1, f_2]
    nx = [3, 3]
    # Generate grid. This will also tag the faces.
    grids = meshing.cart_grid(f_set, nx, physdims=nx)

    # 2D grid:
    g_2d = grids.subdomains(dim=2)[0]

    f_tags_2d = np.array(
        [
            False,
            True,
            False,
            False,  # first row
            False,
            True,
            False,
            False,  # Second row
            False,
            False,
            False,
            False,  # third row
            False,
            False,
            False,  # Bottom column
            True,
            True,
            False,  # Second column
            False,
            False,
            False,  # Third column
            False,
            False,
            False,  # Top column
            True,
            True,
            True,
            True,
        ]
    )  # Added faces

    d_tags_2d = np.array(
        [
            True,
            False,
            False,
            True,  # first row
            True,
            False,
            False,
            True,  # Second row
            True,
            False,
            False,
            True,  # third row
            True,
            True,
            True,  # Bottom column
            False,
            False,
            False,  # Second column
            False,
            False,
            False,  # Third column
            True,
            True,
            True,  # Top column
            False,
            False,
            False,
            False,
        ]
    )  # Added Faces

    # No fracture tips for the highest-dimensional grid
    t_tags_2d = np.zeros(f_tags_2d.size, dtype=bool)

    assert np.all(g_2d.tags["tip_faces"] == t_tags_2d)
    assert np.all(g_2d.tags["fracture_faces"] == f_tags_2d)
    assert np.all(g_2d.tags["domain_boundary_faces"] == d_tags_2d)

    # 1D grids:
    for g_1d in grids.subdomains(dim=1):
        f_tags_1d = np.array([False, True, False, True])
        # Use the coordinate of the first face to determine which of the fractures
        # this is.
        if g_1d.face_centers[0, 0] > 0.1:
            t_tags_1d = np.array([True, False, False, False])
            d_tags_1d = np.array([False, False, True, False])
        else:
            t_tags_1d = np.array([False, False, True, False])
            d_tags_1d = np.array([True, False, False, False])

        assert np.all(g_1d.tags["tip_faces"] == t_tags_1d)
        assert np.all(g_1d.tags["fracture_faces"] == f_tags_1d)
        assert np.all(g_1d.tags["domain_boundary_faces"] == d_tags_1d)

    # The 0D grid in the intersection point has no faces, thus no tags.
