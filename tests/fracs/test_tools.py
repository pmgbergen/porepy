"""Tests of mesh size determination for 2d simplex grids."""
import numpy as np

import porepy as pp


def test_2d_domain_one_immersed_fracture():
    """Test mesh size determination for a 2d domain with one immersed fracture.

    This function generates a fixed set of points and lines, and verifies that the mesh
    size is correctly determined by comparing against hard-coded values.

    """
    # Define points, lines (both domain and fracture)
    pts = np.array([[0.0, 5.0, 5.0, 0.0, 0.5, 4.0], [0.0, 0.0, 5.0, 5.0, 1.0, 1.0]])
    on_boundary = np.array([True, True, True, True, False, False])
    # First two rows refer to point indices. The two last rows refer to the tagging
    # system used for 2d meshing in fractured domains, see FractureNetwork2d for
    # details.
    lines = np.array(
        [[0, 3, 1, 2, 4], [1, 0, 2, 3, 5], [1, 1, 1, 1, 3], [0, 3, 1, 2, 4]]
    )

    # Known mesh sizes.
    mesh_sizes_known = np.array(
        [1.11803399, 1.41421356, 2.0, 2.0, 0.5, 1.0, 0.5, 1.0, 1.80277564, 1.0, 1.0]
    )
    # Known point coordinates after inserting additional points to enforce the mesh
    # sizes.
    pts_split_known = np.array(
        [
            [0.0, 5.0, 5.0, 0.0, 0.5, 4.0, 0.0, 2.5, 5.0, 0.0, 2.25],
            [0.0, 0.0, 5.0, 5.0, 1.0, 1.0, 1.0, 0.0, 2.5, 3.0, 1.0],
        ]
    )
    # Call helper function to generate and verify the mesh sizes.
    _generate_and_verify_2d_grids(
        pts, lines, on_boundary, mesh_sizes_known, pts_split_known
    )


def test_2d_domain_one_fracture_reaches_boundary():
    """Test mesh size determination for a 2d domain with one fracture that extends to
    the domain boundary.

    This function generates a fixed set of points and lines, and verifies that the mesh
    size is correctly determined by comparing against hard-coded values.

    """
    # See function test_2d_domain_one_immersed_fracture() for detailed comments.
    pts = np.array([[0.0, 5.0, 5.0, 0.0, 0.5, 5.0], [0.0, 0.0, 5.0, 5.0, 1.5, 1.5]])
    on_boundary = np.array([True, True, True, True, False, False])
    lines = np.array(
        [
            [0, 3, 1, 5, 2, 4],
            [1, 0, 5, 2, 3, 5],
            [1, 1, 1, 1, 1, 3],
            [0, 3, 1, 1, 2, 4],
        ]
    )

    mesh_sizes_known = np.array(
        [1.58113883, 1.5, 2.0, 2.0, 0.5, 1.0, 0.5, 1.5, 0.75, 1.0, 1.0]
    )
    pts_split_known = np.array(
        [
            [0.0, 5.0, 5.0, 0.0, 0.5, 5.0, 0.0, 2.5, 0.0, 0.0, 2.75],
            [0.0, 0.0, 5.0, 5.0, 1.5, 1.5, 1.5, 0.0, 0.75, 3.25, 1.5],
        ]
    )
    _generate_and_verify_2d_grids(
        pts, lines, on_boundary, mesh_sizes_known, pts_split_known
    )


def _generate_and_verify_2d_grids(
    pts, lines, on_boundary, mesh_sizes_known, pts_split_known
):
    """Helper function to generate and verify 2d grids"""
    mesh_size_min = 0.2
    mesh_size_frac = 1
    mesh_size_bound = 2
    mesh_sizes, pts_split, _ = pp.fracs.tools.determine_mesh_size(
        pts,
        lines,
        on_boundary,
        mesh_size_frac=mesh_size_frac,
        mesh_size_min=mesh_size_min,
        mesh_size_bound=mesh_size_bound,
    )

    assert np.all(np.isclose(mesh_sizes, mesh_sizes_known))
    assert np.all(np.isclose(pts_split, pts_split_known))
