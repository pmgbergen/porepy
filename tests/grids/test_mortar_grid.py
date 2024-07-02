"""Tests of the mortar grids. Mainly focuses on mappings between mortar grids and
surrounding grids.

The module contains the following groups of tests:
    - One set of tests for 2d domains, where the 2d grid is replaced.
    - One set of tests for 3d domains, where the 2d and 1d grids are replaced.
    - test_pickle_mortar_grid: Individual test to verify that MortarGrids can be
      pickled.

A further description is given for each of the groups of tests.

"""
import os
import pickle

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp
from porepy.fracs import meshing

"""Simple testing of 1d mortar grid mapping"""


def test_1d_mortar_grid_mappings():
    f1 = np.array([[0, 1], [0.5, 0.5]])
    mdg = meshing.cart_grid([f1], [2, 2], **{"physdims": [1, 1]})

    for intf in mdg.interfaces():
        high_to_mortar_known = np.matrix(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            ]
        )
        low_to_mortar_known = np.matrix([[1, 0], [0, 1], [1, 0], [0, 1]])

        assert np.allclose(high_to_mortar_known, intf.primary_to_mortar_int().todense())
        assert np.allclose(
            low_to_mortar_known, intf.secondary_to_mortar_int().todense()
        )


"""Tests of replacement of the 2d subdomain in a 2d domain with a single fracture.

Various perturbations of the grid are tested. The tests are based on the following
logic:
    1. A 2d grid is created, with a single fracture.
    2. A new 2d grid is created, and possibly perturbed.
    3. The new grid is inserted into the old md-grid.
    4. The projections between the old fracture and the new 2d grid are fetched, and
       some simple sanity checks are done (common for all tests).
    5. Specific checks of the projections are done for each test, based on knowledge of
       how the new grid was perturbed and what the expected result is. This involves
       checking the projections against hard coded values.

The tests are made up of the following functions:
    - A set of functions to test indivdiual replacement operations.
    - A helper function _create_2d_mdg, which creates a 2d domain with a single fracture.
    - A helper function _replace_2d_grid_fetch_projections, which replaces the 2d grid
        in the domain and fetches the projections between the old and the new grid.

"""


def test_2d_domain_replace_2d_grid_by_identical_copy():
    """Copy the higher dimensional grid and replace. The mapping should stay the same."""
    # Create a first md grid
    mdg, sd_old, old_projection = _create_2d_mdg([1, 2])
    # Copy the highest-dimensional grid
    sd_new = sd_old.copy()

    # Tracking the boundary.
    bg_old = mdg.subdomain_to_boundary_grid(sd_old)

    # Replace the 2d grid with a finer one and get the new projections. This function
    # will also verify that the projections have the right size, and that the new
    # average projection has rows summing to unity.
    new_projection_avg, new_projection_int = _replace_2d_grid_fetch_projections(
        mdg, sd_old, sd_new, old_projection
    )
    # The new and the old projections should be identical.
    pp.test_utils.arrays.compare_matrices(new_projection_avg, old_projection)
    pp.test_utils.arrays.compare_matrices(new_projection_int, old_projection)

    # Check that old grids are removed properly.
    assert sd_old not in mdg
    assert bg_old not in mdg

    # Check that the new grid and its boundary appeared properly.
    assert sd_new in mdg
    bg_new = mdg.subdomain_to_boundary_grid(sd_new)
    assert bg_new is not None


def test_2d_domain_replace_2d_grid_with_refined_grid():
    """Replace the 2d grid with a finer one."""

    # Create a first md grid
    mdg, sd_old, old_projection = _create_2d_mdg([1, 2])

    # Create a new grid. We will take the higher dimensional grid from this and insert
    # it into the old md grid.
    _, sd_new, _ = _create_2d_mdg([2, 2])

    # Replace the 2d grid with a finer one and get the new projections. This function
    # will also verify that the projections have the right size, and that the new
    # average projection has rows summing to unity.
    new_projection_avg, new_projection_int = _replace_2d_grid_fetch_projections(
        mdg, sd_old, sd_new, old_projection
    )
    # The new and the old projections should be identical.
    pp.test_utils.arrays.compare_matrices(new_projection_avg, old_projection)
    pp.test_utils.arrays.compare_matrices(new_projection_int, old_projection)

    fi = np.where(sd_new.face_centers[1] == 0.5)[0]
    assert fi.size == 4
    # Hard coded test (based on knowledge of how the grids and pp.meshing is
    # implemented). Faces to the uppermost cell are always kept in place, the lowermost
    # are duplicated towards the end of the face definition.
    assert np.all(new_projection_avg[0, fi[:2]].toarray() == 0.5)
    assert np.all(new_projection_avg[1, fi[2:]].toarray() == 0.5)


def test_2d_domain_replace_2d_grid_with_coarse_grid():
    """Replace the 2d grid with a coarser one."""

    # Create a first md grid
    mdg, sd_old, old_projection = _create_2d_mdg([2, 2])

    # Create a new grid. We will take the higher dimensional grid from this and insert
    # it into the old md grid.
    _, sd_new, _ = _create_2d_mdg([1, 2])

    # Replace the 2d grid with a finer one and get the new projections. This function
    # will also verify that the projections have the right size, and that the new
    # average projection has rows summing to unity.
    new_projection_avg, new_projection_int = _replace_2d_grid_fetch_projections(
        mdg, sd_old, sd_new, old_projection
    )

    # Columns in integrated projection sum to either 0 or 1
    assert np.all(
        np.logical_or(
            new_projection_int.toarray().sum(axis=0) == 1,
            new_projection_int.toarray().sum(axis=0) == 0,
        ),
    )

    fi = np.where(sd_new.face_centers[1] == 0.5)[0]
    assert fi.size == 2
    # Hard coded test (based on knowledge of how the grids and pp.meshing is
    # implemented). Faces to the uppermost cell are always kept in place, the lowermost
    # are duplicated towards the end of the face definition.
    assert np.all(new_projection_avg[0, fi[0]] == 1)
    assert np.all(new_projection_avg[1, fi[0]] == 1)
    assert np.all(new_projection_avg[2, fi[1]] == 1)
    assert np.all(new_projection_avg[3, fi[1]] == 1)


def test_2d_domain_replace_2d_grid_with_fine_perturbed_grid():
    """Replace the 2d grid with a finer one, and move the nodes along the interface so
    that areas along the interface are no longer equal.
    """

    # Create a first md grid
    mdg, sd_old, old_projection = _create_2d_mdg([1, 2])

    # Create a new grid. We will take the higher dimensional grid from this and insert
    # it into the old md grid.
    _, sd_new, _ = _create_2d_mdg([2, 2])

    # By construction of the split grid, we know that the nodes at (0.5, 0.5) are no 5
    # and 6, and that no 5 is associated with the face belonging to the lower cells.
    # Move node belonging to the lower face
    sd_new.nodes[0, [5, 6]] = [0.2, 0.7]
    sd_new.compute_geometry()

    # Replace the 2d grid with a finer one and get the new projections. This function
    # will also verify that the projections have the right size, and that the new
    # average projection has rows summing to unity.
    new_projection_avg, new_projection_int = _replace_2d_grid_fetch_projections(
        mdg, sd_old, sd_new, old_projection
    )

    # Columns in integrated projection sum to either 0 or 1.
    assert np.all(
        np.logical_or(
            new_projection_int.toarray().sum(axis=0) == 1,
            new_projection_int.toarray().sum(axis=0) == 0,
        ),
    )

    fi = np.where(sd_new.face_centers[1] == 0.5)[0]
    assert fi.size == 4
    # Hard coded test (based on knowledge of how the grids and pp.meshing is
    # implemented). Faces to the uppermost cell are always kept in place, the lowermost
    # are duplicated towards the end of the face definition.
    assert np.abs(new_projection_avg[0, 8] - 0.7 < 1e-6)
    assert np.abs(new_projection_avg[0, 9] - 0.3 < 1e-6)
    assert np.abs(new_projection_avg[1, 12] - 0.2 < 1e-6)
    assert np.abs(new_projection_avg[1, 13] - 0.8 < 1e-6)


def test_2d_domain_replace_2d_grid_with_perturbed_grid():
    """Replace the 2d grid with a finer one, and move the nodes along the interface so
    that areas along the interface are no longer equal.
    """

    # Create a first md grid
    mdg, sd_old, old_projection = _create_2d_mdg([2, 2])

    # Create a new grid. We will take the higher dimensional grid from this and insert
    # it into the old md grid.
    _, sd_new, _ = _create_2d_mdg([2, 2])

    # By construction of the split grid, we know that the nodes at (0.5, 0.5) are no 5
    # and 6, and that no 5 is associated with the face belonging to the lower cells.
    # Move node belonging to the lower face
    sd_new.nodes[0, [5, 6]] = [0.2, 0.7]
    sd_new.compute_geometry()

    # Replace the 2d grid with a finer one and get the new projections. This function
    # will also verify that the projections have the right size, and that the new
    # average projection has rows summing to unity.
    new_projection_avg, new_projection_int = _replace_2d_grid_fetch_projections(
        mdg, sd_old, sd_new, old_projection
    )

    # Columns in integrated projection sum to either 0 or 1.
    assert np.all(
        np.logical_or(
            new_projection_int.toarray().sum(axis=0) == 1,
            new_projection_int.toarray().sum(axis=0) == 0,
        ),
    )

    fi = np.where(sd_new.face_centers[1] == 0.5)[0]
    assert fi.size == 4
    # Hard coded test (based on knowledge of how the grids and pp.meshing is
    # implemented). Faces to the uppermost cell are always kept in place, the lowermost
    # are duplicated towards the end of the face definition.

    # It seems the mortar grid is designed so that the first cell is associated with
    # face 9 in the old grid. This is split into 2/5 face 8 and 3/5 face 9.

    assert np.abs(new_projection_avg[0, 8] - 0.4 < 1e-6)
    assert np.abs(new_projection_avg[0, 9] - 0.6 < 1e-6)
    # The second cell in mortar grid is still fully connected to face 9
    assert np.abs(new_projection_avg[1, 9] - 1 < 1e-6)
    assert np.abs(new_projection_avg[2, 13] - 1 < 1e-6)
    assert np.abs(new_projection_avg[3, 12] - 0.4 < 1e-6)
    assert np.abs(new_projection_avg[3, 13] - 0.6 < 1e-6)


def test_2d_domain_replace_2d_grid_with_permuted_nodes():
    """Replace higher dimensional grid with an identical one, except the node indices
    are perturbed. This will test sorting of nodes along 1d lines.
    """

    # Create a first md grid
    mdg, sd_old, old_projection = _create_2d_mdg([2, 2])

    # Create a new grid. We will take the higher dimensional grid from this and insert
    # it into the old md grid.
    _, sd_new, _ = _create_2d_mdg([2, 2])

    # By construction of the split grid, we know that the nodes at (0.5, 0.5) are no 5
    # and 6, and that no 5 is associated with the face belonging to the lower cells.
    # Move node belonging to the lower face
    #     g_new.nodes[0, 5] = 0.2
    #     g_new.nodes[0, 6] = 0.7

    # Replacements: along lower segment (3, 5, 7) -> (7, 5, 3)
    # On upper segment: (4, 6, 8) -> (8, 4, 6)
    sd_new.nodes[0, [3, 4, 5, 6, 7, 8]] = [1, 0.5, 0.5, 1, 0, 0]

    fn = sd_new.face_nodes.indices.reshape((2, sd_new.num_faces), order="F")
    fn[:, 8] = np.array([4, 8])
    fn[:, 9] = np.array([4, 6])
    fn[:, 12] = np.array([7, 5])
    fn[:, 13] = np.array([5, 3])

    # Replace the 2d grid with a finer one and get the new projections. This function
    # will also verify that the projections have the right size, and that the new
    # average projection has rows summing to unity.
    new_projection_avg, new_projection_int = _replace_2d_grid_fetch_projections(
        mdg, sd_old, sd_new, old_projection
    )

    # Columns in integrated projection sum to either 0 or 1.
    assert np.all(
        np.logical_or(
            new_projection_int.toarray().sum(axis=0) == 1,
            new_projection_int.toarray().sum(axis=0) == 0,
        ),
    )
    fi = np.where(sd_new.face_centers[1] == 0.5)[0]
    assert fi.size == 4
    # Hard coded test (based on knowledge of how the grids and pp.meshing is
    # implemented). Faces to the uppermost cell are always kept in place, the lowermost
    # are duplicated towards the end of the face definition.
    assert (old_projection != new_projection_avg).nnz == 0


def test_2d_domain_replace_2d_grid_with_permuted_and_perturbed_nodes():
    """Replace higher dimensional grid with an identical one, except the node indices
    are perturbed. This will test sorting of nodes along 1d lines. Also perturb nodes
    along the segment.
    """

    # Create a first md grid
    mdg, sd_old, old_projection = _create_2d_mdg([2, 2])

    # Create a new grid. We will take the higher dimensional grid from this and insert
    # it into the old md grid.
    _, sd_new, _ = _create_2d_mdg([2, 2])

    # By construction of the split grid, we know that the nodes at (0.5, 0.5) are no 5
    # and 6, and that no 5 is associated with the face belonging to the lower cells.
    # Replacements: along lower segment (3, 5, 7) -> (7, 5, 3)
    # On upper segment: (4, 6, 8) -> (8, 4, 6)
    sd_new.nodes[0, [3, 4, 5, 6, 7, 8]] = [1, 0.7, 0.2, 1, 0, 0]

    fn = sd_new.face_nodes.indices.reshape((2, sd_new.num_faces), order="F")
    fn[:, 8] = np.array([4, 8])
    fn[:, 9] = np.array([4, 6])
    fn[:, 12] = np.array([7, 5])
    fn[:, 13] = np.array([5, 3])

    # Replace the 2d grid with a finer one and get the new projections. This function
    # will also verify that the projections have the right size, and that the new
    # average projection has rows summing to unity.
    new_projection_avg, new_projection_int = _replace_2d_grid_fetch_projections(
        mdg, sd_old, sd_new, old_projection
    )

    # Columns in integrated projection sum to either 0 or 1.
    assert np.all(
        np.logical_or(
            new_projection_int.toarray().sum(axis=0) == 1,
            new_projection_int.toarray().sum(axis=0) == 0,
        ),
    )

    fi = np.where(sd_new.face_centers[1] == 0.5)[0]
    assert fi.size == 4
    # Hard coded test (based on knowledge of how the grids and pp.meshing is
    # implemented). Faces to the uppermost cell are always kept in place, the lowermost
    # are duplicated towards the end of the face definition.
    # It seems the mortar grid is designed so that the first cell is associated with
    # face 9 in the old grid. This is split into 2/5 face 8 and 3/5 face 9.

    assert np.abs(new_projection_avg[0, 8] - 0.4 < 1e-6)
    assert np.abs(new_projection_avg[0, 9] - 0.6 < 1e-6)
    # The second cell in mortar grid is still fully connected to face 9
    assert np.abs(new_projection_avg[1, 9] - 1 < 1e-6)
    assert np.abs(new_projection_avg[2, 13] - 1 < 1e-6)
    assert np.abs(new_projection_avg[3, 12] - 0.4 < 1e-6)
    assert np.abs(new_projection_avg[3, 13] - 0.6 < 1e-6)


def _create_2d_mdg(
    size: list[int, int]
) -> tuple[pp.MixedDimensionalGrid, pp.Grid, sps.spmatrix]:
    """Helper function to create a 2d md grid with a single interface.

    Parameters:
        size: Number of cells in the x and y direction.

    Returns:
        mdg: The mixed-dimensional grid.
        sd: The subdomain grid.
        projection: The projection from the subdomain to the mortar grid.

    """
    # Create the new grid
    mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
        "cartesian",
        meshing_args={"cell_size_x": 1 / size[0], "cell_size_y": 1 / size[1]},
        fracture_indices=[1],
    )
    # Fetch the interface and a projection matrix. The grid is matching, so we
    # arbitrarily use the integration projection.
    intf = mdg.interfaces()[0]
    projection = intf.primary_to_mortar_int().copy()
    sd = mdg.subdomains(dim=2)[0]
    return mdg, sd, projection


def _replace_2d_grid_fetch_projections(mdg, sd_old, sd_new, old_projection):
    """Helper function to replace a 2d grid in a md grid, and fetch the new projections.

    The function also does a small sanity check on the size of the projections.

    Parameters:
        mdg: The mixed-dimensional grid.
        sd_old: The subdomain grid to be replaced.
        sd_new: The new subdomain grid.
        old_projection: The projection from the old subdomain grid to the mortar grid.

    Returns:
        new_projection_avg: The new projection from the new subdomain grid to the mortar
            grid, for quantities which should be averaged.
        new_projection_int: The new projection from the new subdomain grid to the mortar
            grid, for quantities which should be summed.

    """
    # Do the replacement
    mdg.replace_subdomains_and_interfaces({sd_old: sd_new})
    # Get mortar grid
    intf_new = mdg.interfaces()[0]

    # Fetch the new projections
    new_projection_avg = intf_new.primary_to_mortar_avg()
    new_projection_int = intf_new.primary_to_mortar_int()

    # Sanity check: The mortar grid is not changed, so the number of rows in the
    # projection should not change. The number of columns should be the number of faces
    # in the new subdomain grid.
    assert new_projection_avg.shape[0] == old_projection.shape[0]
    assert new_projection_avg.shape[1] == sd_new.num_faces

    assert np.all(new_projection_avg.toarray().sum(axis=1) == 1)

    return new_projection_avg, new_projection_int


"""Various tests for replacing subdomain and interface grids in a 3d mixed-
dimensional grid.

While the method called in all tests is replace_subdomains_and_interfaces() in the
MixedDimensionalGrid class, the tests are in effect integration tests for replacement of
grids in the MortarGrid class, hence the placemnet in this test module.


The geometries considered are designed so that grids can be replaced and that the
mappings between mortar and primary/secondary grids are updated correctly. A description
of the geometry, with available options, is given below.

The tests consists of the following functions:
    - Individual test functions which set up a tailored mixed-dimensional grid, and does
        a replacement operation.
    - A helper function to set up a MixedDimensional grid, according to specifications
        given in the main test function, by combining, 3d, 2d and (optionally) 1d grids.
    - A helper function to fetch mortar projection matrices from a MixedDimensionalGrid.
    - A helper function to compare two sets of mortar projection matrices.
    - A helper function to create a three dimensional grid.
    - A helper function to create a two dimensional grid consisting of two cells.
    - A helper function to create a two dimensional grid consisting of four cells.
    - A helper function to create a one dimensional grid.

The grid consists of the following components:
    A 3d grid, which is intersected by a fracture at y=0. On each side of this plane the
    grid has 5 nodes (four of them along y=0, the fifth is at y=+-1) and two cells. Most
    importantly, each side (above and below y=0) has two faces at y=0, one with node
    coordinates {(0, 0), (1, 0), (1, 1)}, the other with {(0, 0), (1, 1), (0, 1)}. The
    node at (1, 1) will in some cases be perturbed to (1, 2) (if pert=True), besides
    this, the 3d grid is not changed in the tests, and really not that important.

    A 2d grid, which is located at y=0. Two versions of this grid can be generated:
        1) One with four nodes, at {(0, 0), (1, 0), (1, 1), (0, 1)} - the third node
            is moved to (1, 2) if pert=True - and two cells formed by the sets of nodes
            {(0, 0), (1, 0), (1, 1)} and {(0, 0), (1, 1), (0, 1)}.
        2) One with five nodes, at {(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)}, and
            four cells formed by the midle node and pairs of neighboring nodes on the
            sides. Again, pert=True will move the node (1, 1), but it this case, it will
            also move the midle node to (0.5, 1) to ensure it stays on the line between
            (0, 0) and now (1, 2) - or else the 1d grid defined below will not conform
            to the 2d faces.
    Several of the tests below consists of replacing the 2d grid with two cells with
    that with four cells, and ensure all mappings are correct.

    A 1d grid that extends from (0, 0) to (1, 1) (or (1, 2) if pert=True).

    At the 3d-2d and 2d-1d interfaces, there are of course mortar grids that will have
    their mappings updated as the adjacent subdomain grids are replaced.

IMPLEMENTATION NOTE:

    When perturbing the grid (moving (1, 1) -> (1, 2)), several updates of the grid
    geometry are hardcoded, like cell centers, face normals etc. This is messy, and a
    more transparent approach would have been preferrable, but it will have to do for
    now.

    While the tests all have some grids in common, the geometries are individual tests
    are modified one way or another. It was therefore decided not to use pytest
    parametrization, but rather use helper methods to set up the grids.

"""


def test_3d_domain_replace_1d_grid_with_identity():
    """Generate the full md_grid, and replace the 1d grid with a copy of itself.

    This should not change the mortar mappings. An error here indicates something is
    fundamentally wrong with the implementation of the method
    replace_subdomains_and_interfaces, or with the functions to match 1d grids (used
    when updating the grid).

    """
    mdg = _create_3d_mdg(pert=False)

    # Fetch the mortar mappings
    old_proj_1_h, old_proj_1_l, old_proj_2_h, old_proj_2_l = _get_3d_mortar_projections(
        mdg
    )

    gn = _grid_1d(2)
    go = mdg.subdomains(dim=1)[0]
    mdg.replace_subdomains_and_interfaces({go: gn})

    # Fetch the new mortar mappings.
    new_proj_1_h, new_proj_1_l, new_proj_2_h, new_proj_2_l = _get_3d_mortar_projections(
        mdg
    )

    # There should be no changes to the mortar mappings, we can compare them directly
    _compare_3d_mortar_projections(
        [
            (old_proj_1_h, new_proj_1_h),
            (old_proj_1_l, new_proj_1_l),
            (old_proj_2_h, new_proj_2_h),
            (old_proj_2_l, new_proj_2_l),
        ]
    )


def test_3d_domain_without_1d_grid_replace_2d_grid_with_identity():
    """Generate the an md grid of a 3d and a 2d grid,  replace the 2d grid with a copy
    of itself.

    This should not change the mortar mappings. An error here indicates something is
    fundamentally wrong with the implementation of the method
    replace_subdomains_and_interfaces, or with the functions to match 2d grids (used
    when updating the grid).

    """

    mdg = _create_3d_mdg(pert=False, include_1d=False)

    # Fetch the mortar mappings. There is no 1d grid, thus the mappings to 1d are None
    _, _, old_proj_2_h, old_proj_2_l = _get_3d_mortar_projections(mdg)

    gn = _grid_2d_two_cells(pert=False, include_1d=False)
    go = mdg.subdomains(dim=2)[0]
    mdg.replace_subdomains_and_interfaces(sd_map={go: gn})

    # Fetch the new mortar mappings.
    _, _, new_proj_2_h, new_proj_2_l = _get_3d_mortar_projections(mdg)
    _compare_3d_mortar_projections(
        [
            (old_proj_2_h, new_proj_2_h),
            (old_proj_2_l, new_proj_2_l),
        ]
    )


def test_3d_domain_without_1d_grid_replace_2d_grid_with_finer():
    """Generate the an md grid of a 3d and a 2d grid, replace the 2d grid with a refined
    2d grid.

    This will change the mappings between the mortar and the 2d grid. An error most
    likely points to a problem with match_2d_grids.

    """

    mdg = _create_3d_mdg(pert=False, include_1d=False)

    # Fetch the mortar mappings. There is no 1d grid, thus the mappings to 1d are None
    _, _, old_proj_2_h, _ = _get_3d_mortar_projections(mdg)

    gn = _grid_2d_four_cells(include_1d=False, pert=False, move_interior_point=False)
    go = mdg.subdomains(dim=2)[0]
    mdg.replace_subdomains_and_interfaces({go: gn})

    # Fetch the new mortar mappings.
    _, _, new_proj_2_h, new_proj_2_l = _get_3d_mortar_projections(mdg)

    # There should be no changes in the mapping between mortar and primary
    # The known projection matrix, from secondary to one of the mortar grids.
    known_p_2_l = np.tile(np.array([[1, 1, 0, 0], [0, 0, 1, 1]]), (2, 1))

    _compare_3d_mortar_projections(
        [
            (old_proj_2_h, new_proj_2_h),
            (known_p_2_l, new_proj_2_l),
        ]
    )


def test_3d_domain_without_1d_grid_replace_2d_grid_with_finer_perturbed_grid():
    """Generate the an md grid of a 3d and a 2d grid, replace the 2d grid with a refined
    and perturbed 2d grid.

    This will change the mappings between the mortar and the 2d grid. An error most
    likely points to a problem with the method match_grids.match_2d().

    """

    # Create the md_grid, single 2d grid, no splitting.
    mdg = _create_3d_mdg(pert=True, include_1d=False)

    # Fetch the mortar mappings. There is no 1d grid, thus the mappings to 1d are None
    _, _, old_proj_2_h, _ = _get_3d_mortar_projections(mdg)

    gn = _grid_2d_four_cells(pert=True, include_1d=False, move_interior_point=False)
    go = mdg.subdomains(dim=2)[0]
    mdg.replace_subdomains_and_interfaces({go: gn})

    # Fetch the mortar mappings again. There is no 1d grid, thus the mappings to 1d are None
    _, _, new_proj_2_h, new_proj_2_l = _get_3d_mortar_projections(mdg)

    # The known projection matrix, from secondary to one of the mortar grids.
    known_2_l = np.tile(np.array([[1, 1, 1 / 3, 1 / 3], [0, 0, 2 / 3, 2 / 3]]), (2, 1))
    # There should be no changes in the mapping between mortar and primary
    _compare_3d_mortar_projections(
        [
            (old_proj_2_h, new_proj_2_h),
            (known_2_l, new_proj_2_l),
        ]
    )


def test_3d_domain_replace_2d_grid_with_identity():
    """Generate the an md grid of a 3d, 2d and 1d grid, replace the 2d grid with a copy
    of itself.

    This should not change the mortar mappings. An error here indicates something is
    fundamentally wrong with the implementation of the method
    replace_subdomains_and_interfaces, or with the functions to match 2d grids (used
    when updating the grid).

    """

    mdg = _create_3d_mdg(pert=False, include_1d=True)
    # Fetch the mortar mappings.
    old_proj_1_h, old_proj_1_l, old_proj_2_h, old_proj_2_l = _get_3d_mortar_projections(
        mdg
    )

    # Do a formal replacement
    gn = _grid_2d_two_cells(pert=False, include_1d=True)
    go = mdg.subdomains(dim=2)[0]
    mdg.replace_subdomains_and_interfaces({go: gn})

    # Fetch the mortar mappings again
    new_proj_1_h, new_proj_1_l, new_proj_2_h, new_proj_2_l = _get_3d_mortar_projections(
        mdg
    )

    # None of the mappings should have changed
    _compare_3d_mortar_projections(
        [
            (old_proj_2_h, new_proj_2_h),
            (old_proj_2_l, new_proj_2_l),
            (old_proj_1_h, new_proj_1_h),
            (old_proj_1_l, new_proj_1_l),
        ]
    )


def test_3d_domain_replace_2d_grid_with_finer_perturbed_grid():
    """Generate the an md grid of a 3d, 2d and 1d grid, replace the 2d grid with a finer
    and perturbed 2d grid.

    An error here will most likely indicate a problem with the method
    match_grids.match_2d.

    """
    mdg = _create_3d_mdg(pert=True, include_1d=True)

    # Fetch the mortar mappings.
    old_proj_1_h, old_proj_1_l, old_proj_2_h, old_proj_2_l = _get_3d_mortar_projections(
        mdg
    )

    # Change the 2d grid, obtain new mortar mappings
    gn = _grid_2d_four_cells(include_1d=True, pert=True, move_interior_point=True)
    go = mdg.subdomains(dim=2)[0]
    mdg.replace_subdomains_and_interfaces({go: gn})
    new_proj_1_h, new_proj_1_l, new_proj_2_h, new_proj_2_l = _get_3d_mortar_projections(
        mdg
    )

    # The mappings between mortars and the 2d grids will change, between mortar and
    # 3d/1d will stay the same.
    known_proj_2_l = np.tile(np.array([[1, 1, 0, 0], [0, 0, 1, 1]]), (2, 1))
    known_proj_1_h = np.zeros((2, 10))
    known_proj_1_h[0, [2, 4]] = 1
    known_proj_1_h[1, [7, 8]] = 1

    # Verify that the mappings are as expected
    _compare_3d_mortar_projections(
        [
            (old_proj_2_h, new_proj_2_h),
            (known_proj_2_l, new_proj_2_l),
            (known_proj_1_h, new_proj_1_h),
            (old_proj_1_l, new_proj_1_l),
        ]
    )


def _compare_3d_mortar_projections(
    projections: list[tuple[np.ndarray | sps.spmatrix, np.ndarray | sps.spmatrix]]
):
    """Helper method to compare two sets of mortar projections."""
    # Loop over projections
    for proj in projections:
        # If the projection is a sparse matrix, convert it to dense
        if isinstance(proj[0], sps.spmatrix):
            p0 = proj[0].toarray()
        else:
            p0 = proj[0]
        if isinstance(proj[1], sps.spmatrix):
            p1 = proj[1].toarray()
        else:
            p1 = proj[1]

        assert np.allclose(p0, p1)


def _get_3d_mortar_projections(mdg: pp.MixedDimensionalGrid):
    """Fetch the mortar projections from the md_grid.

    Parameters:
        mdg (pp.MixedDimensionalGrid): The mixed-dimensional grid.

    Returns:
        proj_1_h (sps.spmatrix): Projection from 2d grid to 1d mortar grid.
        proj_1_l (sps.spmatrix): Projection from 1d grid to 1d mortar grid.
        proj_2_h (sps.spmatrix): Projection from 3d grid to 2d mortar grid.
        proj_2_l (sps.spmatrix): Projection from 2d grid to 2d mortar grid.

        If the mortar grids do not exist, the corresponding projections are None.

    """
    # By default we assume that the mortar grids are None.
    mg1 = None  # Mortar grid of dimension 1 (e.g., 2d-1d coupling)
    mg2 = None  # Mortar grid of dimension 2 (e.g., 3d-2d coupling)

    # Fetch the mortar grids if they exist
    for intf in mdg.interfaces():
        # Fetch the lower-dimensional subdomain neighbor of this interface
        _, gl = mdg.interface_to_subdomain_pair(intf)
        if gl.dim == 1:
            mg1 = intf
        else:
            mg2 = intf

    # Fetch the projections if they exist.
    if mg1 is not None:
        proj_1_h = mg1.primary_to_mortar_int().copy()
        proj_1_l = mg1.secondary_to_mortar_int().copy()
    else:
        proj_1_h = None
        proj_1_l = None
    if mg2 is not None:
        proj_2_h = mg2.primary_to_mortar_int().copy()
        proj_2_l = mg2.secondary_to_mortar_int().copy()
    else:
        proj_2_h = None
        proj_2_l = None

    return proj_1_h, proj_1_l, proj_2_h, proj_2_l


def _create_3d_mdg(pert: bool = False, include_1d: bool = True):
    """Set up a mixed-dimensional grid based on parameters given in the test function."""
    if include_1d:
        sd_3 = _grid_3d(include_1d=include_1d, pert=pert)
        sd_2 = _grid_2d_two_cells(include_1d=include_1d, pert=pert)
        sd_1 = _grid_1d()

        mdg, _ = pp.meshing._assemble_mdg([[sd_3], [sd_2], [sd_1]])
        map = sps.csc_matrix(np.array([[0, 0, 1, 1, 0, 0]]))
        pp.meshing.create_interfaces(mdg, {(sd_2, sd_1): map})

        a = np.zeros((16, 2))
        a[3, 0] = 1
        a[7, 1] = 1
        a[11, 0] = 1
        a[15, 1] = 1
        map = sps.csc_matrix(a.T)
        pp.meshing.create_interfaces(mdg, {(sd_3, sd_2): map})

    else:
        sd_3 = _grid_3d(include_1d=include_1d, pert=pert)
        sd_2 = _grid_2d_two_cells(include_1d=include_1d, pert=pert)
        mdg, _ = pp.meshing._assemble_mdg([[sd_3], [sd_2]])
        a = np.zeros((16, 2))
        a[3, 0] = 1
        a[7, 1] = 1
        a[11, 0] = 1
        a[15, 1] = 1
        map = sps.csc_matrix(a.T)

        pp.meshing.create_interfaces(mdg, {(sd_3, sd_2): map})
    return mdg


def _grid_3d(include_1d: bool, pert: bool) -> pp.Grid:
    """Create a 3d grid. See docstring above for details."""
    if include_1d:
        # Grid consisting of 3d, 2d and 1d grid. Nodes along the main
        # surface are split into two or four.
        # The first two cells are below the plane y=0, the last two are above the plane.
        n = np.array(
            [
                [0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
            ]
        )
        fn = np.array(
            [
                [0, 0, 0, 1, 0, 0, 0, 4, 13, 13, 13, 7, 13, 13, 13, 10],
                [1, 2, 3, 2, 4, 5, 6, 5, 7, 8, 9, 8, 10, 11, 12, 11],
                [2, 3, 1, 3, 5, 6, 4, 6, 8, 9, 7, 9, 11, 12, 10, 12],
            ]
        )
        cf = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]).T

    else:
        n = np.array(
            [
                [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 1, 0, 0, 1, 1, 0],
            ]
        )

        fn = np.array(
            [
                [0, 0, 0, 0, 0, 1, 1, 9, 9, 9, 9, 9, 5, 5],
                [1, 2, 3, 4, 1, 2, 3, 5, 6, 7, 8, 5, 6, 7],
                [2, 3, 4, 1, 3, 3, 4, 6, 7, 8, 5, 7, 7, 8],
            ]
        )
        cf = np.array([[0, 1, 4, 5], [2, 3, 4, 6], [7, 8, 11, 12], [9, 10, 11, 13]]).T

    cols = np.tile(np.arange(fn.shape[1]), (fn.shape[0], 1)).ravel("F")
    face_nodes = sps.csc_matrix((np.ones_like(cols), (fn.ravel("F"), cols)))

    cols = np.tile(np.arange(cf.shape[1]), (cf.shape[0], 1)).ravel("F")
    cell_faces = sps.csc_matrix((np.ones_like(cols), (cf.ravel("F"), cols)))

    cell_centers = np.array(
        [
            [-0.25, 0.75, 0.25],
            [-0.25, 0.5, 0.5],
            [-0.25, 0.75, 0.25],
            [0.25, 0.5, 0.5],
        ]
    ).T
    face_normals = np.zeros((3, face_nodes.shape[1]))

    if include_1d:
        # We will only use face normals for faces on the interface
        face_normals[1, [3, 7, 11, 15]] = 1
        if pert:
            # Move the node at  (1, 0, 1) to (1, 0, 2)
            # This will invalidate the assigned geometry, but it should not matter
            # since we do not use the face_area for anything (we never replace the
            # highest-dimensional, 3d grid)
            n[2, [3, 9]] = 2
    else:
        face_normals[2, [5, 6, 12, 13]] = 1
        if pert:
            # This will invalidate the assigned geometry, but it should not matter
            n[2, 4] = 2
            n[2, 9] = 2

    cell_volumes = 1 / 6 * np.ones(cell_centers.shape[1])

    # Create a grid
    g = pp.Grid(
        nodes=n,
        face_nodes=face_nodes,
        cell_faces=cell_faces,
        dim=3,
        name="TetrahedralGrid",
    )
    # Assign additional fields to ensure we are in full control of the grid geometry
    g.face_normals = face_normals
    g.cell_centers = cell_centers
    g.cell_volumes = cell_volumes
    g.global_point_ind = np.arange(n.shape[1])
    if include_1d:
        g.tags["fracture_faces"][[3, 7, 11, 15]] = True
    else:
        g.tags["fracture_faces"][[5, 6, 12, 13]] = True
    return g


def _grid_2d_two_cells(include_1d: bool, pert: bool) -> pp.Grid:
    """Create a 2d grid consisting of two cells. See docstring above for details."""
    if include_1d:
        n = np.array([[0, 1, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 1]])
        if pert:
            n[2, 2] = 2
            n[2, 4] = 2

        fn = np.array([[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]]).T
        cf = np.array([[0, 1, 2], [3, 4, 5]]).T

        face_normals = np.array([[0, -1], [1, 0], [-1, 1], [-1, 1], [0, 1], [-1, 0]]).T
        face_normals = np.vstack(
            (face_normals[0], np.zeros_like(face_normals[0]), face_normals[1])
        )
        cell_centers = np.array([[2 / 3, 0, 1 / 3], [1 / 3, 0, 2 / 3]]).T
        cell_volumes = 1 / 2 * np.ones(cell_centers.shape[1])

        if pert:
            cell_volumes[0] = 1
            cell_volumes[1] = 0.5

            face_normals[0, 2] = -2
            face_normals[0, 3] = -2
            cell_centers[2, 1] = 1

    else:
        n = np.array([[0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 1, 1]])
        if pert:
            n[2, 2] = 2

        fn = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]]).T

        cf = np.array([[0, 1, 4], [4, 2, 3]]).T
        face_normals = np.array([[0, -1], [1, 0], [0, 1], [-1, 0], [-1, 1]]).T
        face_normals = np.vstack(
            (face_normals[0], np.zeros_like(face_normals[0]), face_normals[1])
        )
        cell_centers = np.array([[2 / 3, 0, 1 / 3], [1 / 3, 0, 2 / 3]]).T
        cell_volumes = 1 / 2 * np.ones(cell_centers.shape[1])

        if pert:
            cell_volumes[0] = 1
            cell_volumes[1] = 0.5

    cols = np.tile(np.arange(fn.shape[1]), (fn.shape[0], 1)).ravel("F")
    face_nodes = sps.csc_matrix((np.ones_like(cols), (fn.ravel("F"), cols)))
    cols = np.tile(np.arange(cf.shape[1]), (cf.shape[0], 1)).ravel("F")
    cell_faces = sps.csc_matrix((np.ones_like(cols), (cf.ravel("F"), cols)))

    # Create a grid
    g = pp.Grid(
        nodes=n,
        face_nodes=face_nodes,
        cell_faces=cell_faces,
        dim=2,
        name="TriangleGrid",
    )
    # Assign additional fields to ensure we are in full control of the grid geometry
    g.face_normals = face_normals
    g.cell_centers = cell_centers
    g.cell_volumes = cell_volumes
    g.global_point_ind = 1 + np.arange(n.shape[1])
    if include_1d:
        g.tags["fracture_faces"][[2, 3]] = True
    return g


def _grid_2d_four_cells(
    include_1d: bool, pert: bool, move_interior_point: bool
) -> pp.Grid:
    """Create a 2d grid consisting of four cells. See docstring above for details."""
    if include_1d:
        n = np.array(
            [
                [0, 1, 1, 0.5, 0, 0, 1, 0.5],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0.5, 0, 1, 1, 0.5],
            ]
        )
        if pert:
            # Move the point x=1, z=1 to x=1, z = 2
            n[2, 2] = 2
            n[2, 6] = 2
            if move_interior_point:
                # To make the midpoint (x, z) = (0.5, 0.5) stay on the line between
                # (0, 0) and [the newly moved to] (1, 2), we need to update the coordinates
                # of points 3 and 7.
                n[2, 3] = 1
                n[2, 7] = 1
        fn = np.array([[0, 1, 3, 1, 2, 4, 5, 7, 7, 6], [1, 3, 0, 2, 3, 5, 7, 4, 6, 5]])
        cf = np.array([[0, 1, 2], [3, 4, 1], [5, 6, 7], [8, 9, 6]]).T
        face_normals = np.array(
            [[0, 1, -1, 1, -1, -1, 1, 1, 1, 0], [-1, 1, 1, 0, 1, 0, 1, -1, -1, 1]]
        )

        face_normals = np.vstack(
            (face_normals[0], np.zeros_like(face_normals[0]), face_normals[1])
        )

        cell_volumes = 1 / 4 * np.ones(4)

    else:  # Do not inculde 1d
        n = np.array(
            [
                [0, 1, 1, 0, 0.5],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0.5],
            ]
        )

        if pert:
            n[2, 2] = 2

        fn = np.array([[0, 1, 2, 3, 0, 1, 2, 3], [1, 2, 3, 0, 4, 4, 4, 4]])
        cf = np.array([[0, 4, 5], [1, 5, 6], [2, 7, 6], [3, 4, 7]]).T

        face_normals = np.array(
            [[0, -1], [1, 0], [0, 1], [-1, 0], [-1, 1], [1, 1], [1, -1], [1, 1]]
        ).T
        face_normals = np.vstack(
            (face_normals[0], np.zeros_like(face_normals[0]), face_normals[1])
        )

    cell_centers = np.array(
        [[0.5, 0, 1 / 6], [5 / 6, 0, 0.5], [0.5, 0, 5 / 6], [1 / 6, 0, 0.5]]
    ).T
    cell_volumes = 1 / 4 * np.ones(cell_centers.shape[1])

    if include_1d:
        if pert:
            cell_volumes[1] = 1
            cell_volumes[2] = 0.5

            cell_centers[2, 1] = 1
            cell_centers[2, 2] = 4 / 3

            if move_interior_point:
                cell_volumes[0] = 0.5
                cell_volumes[1] = 0.5
                cell_volumes[2] = 0.25

                face_normals[0, 2] = -2
                face_normals[0, 4] = -2
                face_normals[0, 7] = 2
                face_normals[0, 8] = 2
    else:
        if pert:
            cell_volumes[1] = 0.5
            cell_volumes[2] = 0.5

    cols = np.tile(np.arange(fn.shape[1]), (fn.shape[0], 1)).ravel("F")
    face_nodes = sps.csc_matrix((np.ones_like(cols), (fn.ravel("F"), cols)))

    cols = np.tile(np.arange(cf.shape[1]), (cf.shape[0], 1)).ravel("F")
    cell_faces = sps.csc_matrix((np.ones_like(cols), (cf.ravel("F"), cols)))

    g = pp.Grid(
        nodes=n,
        face_nodes=face_nodes,
        cell_faces=cell_faces,
        dim=2,
        name="TriangleGrid",
    )
    # Assign additional fields to ensure we are in full control of the grid geometry
    g.face_normals = face_normals
    g.cell_centers = cell_centers
    g.cell_volumes = cell_volumes
    g.global_point_ind = 10 + np.arange(n.shape[1])

    if include_1d:
        g.tags["fracture_faces"][[2, 4, 7, 8]] = True

    return g


def _grid_1d(n_nodes: int = 2) -> pp.Grid:
    """Create a 1d grid. See docstring above for details."""
    x = np.linspace(0, 1, n_nodes)
    g = pp.TensorGrid(x)
    g.nodes = np.tile(x, (3, 1))
    g.compute_geometry()
    g.global_point_ind = 1 + np.arange(n_nodes)
    return g


@pytest.mark.parametrize(
    "g",
    [
        pp.PointGrid([0, 0, 0]),
        pp.CartGrid([2]),
        pp.CartGrid([2, 2]),
        pp.StructuredTriangleGrid([2, 2]),
    ],
)
def test_pickle_mortar_grid(g):
    """Test that a mortar grid can be pickled."""

    # Create a mortar grid from the passed grid.
    g.compute_geometry()
    mg = pp.MortarGrid(g.dim, {0: g, 1: g})

    # Dump the grid to file using pickle.
    file_name = "tmp.grid"
    pickle.dump(mg, open(file_name, "wb"))
    # Read back
    mg_read = pickle.load(open(file_name, "rb"))
    # Compare the grids
    pp.test_utils.grids.compare_mortar_grids(mg, mg_read)

    # Do the same operation with the one-sided grid.
    mg_one_sided = pp.MortarGrid(g.dim, {0: g})
    pickle.dump(mg, open(file_name, "wb"))
    mg_read = pickle.load(open(file_name, "rb"))
    pp.test_utils.grids.compare_mortar_grids(mg_one_sided, mg_read)

    # Delete the file
    os.unlink(file_name)
