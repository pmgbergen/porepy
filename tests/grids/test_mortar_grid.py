"""Tests of the mortar grids. Mainly focuses on mappings between mortar grids and
surrounding grids.

The module contains the following groups of tests:

    TestReplaceHigherDimensionalGridInMixedDimensionalGrid: Test the method
        replace_subdomains_and_interfaces in a mixed-dimensional grid applied to
        the highest-dimensional grid. In practice, this only applies to 2d domains,
        since replacement of 3d grids is not supported.

    TestReplace1dand2dGridsIn3dDomain: For a 3d domain with a 2d fracture, and optionally
        a 1d intersection (geometry is hard coded), replace the 2d and/or the 1d grids.
        This verifies that updates of the projections between mortar and primary/secondary
        are correctly updated for 1d and 2d grids.

    test_pickle_mortar_grid: Method to verify that MortarGrids can be pickled.

"""
import pickle
import unittest
from itertools import count
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp


class TestReplaceHigherDimensionalGridInMixedDimensionalGrid(unittest.TestCase):
    """Test of functionality to replace the higher-dimensional grid in a MixedDimensionalGrid.

    Since we do not support replacement of 3d grids, this test considers only a 2d domain
    with a single fracture, and replace the 2d grid with various perturbations etc. of
    the grid.

    The critical point is to check that the projection from the primary grid to the
    mortar grid is updated correctly.

    Replacement of the highest-dimensional grid in a 1d domain is not checked, but that
    seems to be a less relevant case.
    """

    def test_replace_by_same(self):
        """1x2 grid. Copy the higher dimensional grid and replace. The mapping should be
        the same.

        We also check that the boundary grid is updated properly.

        """
        mdg, _ = pp.md_grids_2d.single_horizontal([1, 2], simplex=False)

        intf_old = mdg.interfaces()[0]

        old_projection = intf_old.primary_to_mortar_int().copy()

        sd_old = mdg.subdomains(dim=2)[0]
        sd_new = sd_old.copy()

        # Tracking the boundary.
        bg_old = mdg.subdomain_to_boundary_grid(sd_old)

        mdg.replace_subdomains_and_interfaces({sd_old: sd_new})

        # Get mortar grid again
        intf_new = mdg.interfaces()[0]

        new_projection = intf_new.primary_to_mortar_int()
        assert (
            old_projection != new_projection
        ).nnz == 0, "The projections should be identical."

        # Check that old grids are removed properly.
        assert sd_old not in mdg
        assert bg_old not in mdg

        # Check that the new grid and its boundary appeared properly.
        assert sd_new in mdg
        bg_new = mdg.subdomain_to_boundary_grid(sd_new)
        assert bg_new is not None

    def test_refine_high_dim(self):
        # Replace the 2d grid with a finer one

        mdg, _ = pp.md_grids_2d.single_horizontal([1, 2], simplex=False)

        intf_old = mdg.interfaces()[0]

        old_projection = intf_old.primary_to_mortar_int().copy()
        sd_old = mdg.subdomains(dim=2)[0]

        # Create a new, finer 2d grid. This is the simplest
        # way to put the fracture in the right place is to create a new
        # bucket, and pick out the higher dimensional grid
        mdg_new, _ = pp.md_grids_2d.single_horizontal([2, 2], simplex=False)
        mdg_new.compute_geometry()

        sd_new = mdg_new.subdomains(dim=2)[0]

        mdg.replace_subdomains_and_interfaces({sd_old: sd_new})

        # Get mortar grid again
        intf_new = mdg.interfaces()[0]

        new_projection = intf_new.primary_to_mortar_avg()

        # Check shape

        self.assertTrue(new_projection.shape[0] == old_projection.shape[0])
        self.assertTrue(new_projection.shape[1] == sd_new.num_faces)
        # Projection sums to unity.
        self.assertTrue(np.all(new_projection.toarray().sum(axis=1) == 1))

        fi = np.where(sd_new.face_centers[1] == 0.5)[0]
        self.assertTrue(fi.size == 4)
        # Hard coded test (based on knowledge of how the grids and pp.meshing
        # is implemented). Faces to the uppermost cell are always kept in
        # place, the lowermost are duplicated towards the end of the face
        # definition.
        self.assertTrue(np.all(new_projection[0, fi[:2]].toarray() == 0.5))
        self.assertTrue(np.all(new_projection[1, fi[2:]].toarray() == 0.5))

    def test_coarsen_high_dim(self):
        # Replace the 2d grid with a coarser one

        mdg, _ = pp.md_grids_2d.single_horizontal([2, 2], simplex=False)

        # Pick out mortar grid by a loop, there is only one edge in the bucket
        intf_old = mdg.interfaces()[0]
        old_projection = intf_old.primary_to_mortar_int().copy()
        sd_old = mdg.subdomains(dim=2)[0]

        # Create a new, coarser 2d grid. This is the simplest
        # way to put the fracture in the right place is to create a new
        # bucket, and pick out the higher dimensional grid
        mdg_new, _ = pp.md_grids_2d.single_horizontal([1, 2], simplex=False)

        sd_new = mdg_new.subdomains(dim=2)[0]

        mdg.replace_subdomains_and_interfaces({sd_old: sd_new})

        # Get mortar grid again
        intf_new = mdg.interfaces()[0]

        new_projection_avg = intf_new.primary_to_mortar_avg()
        new_projection_int = intf_new.primary_to_mortar_int()

        # Check shape
        self.assertTrue(new_projection_avg.shape[0] == old_projection.shape[0])
        self.assertTrue(new_projection_avg.shape[1] == sd_new.num_faces)

        # Projection of averages sum to unity in the rows.
        self.assertTrue(np.all(new_projection_avg.toarray().sum(axis=1) == 1))
        # Columns in integrated projection sum to either 0 or 1
        self.assertTrue(
            np.all(
                np.logical_or(
                    new_projection_int.A.sum(axis=0) == 1,
                    new_projection_int.A.sum(axis=0) == 0,
                ),
            )
        )

        fi = np.where(sd_new.face_centers[1] == 0.5)[0]
        self.assertTrue(fi.size == 2)
        # Hard coded test (based on knowledge of how the grids and pp.meshing
        # is implemented). Faces to the uppermost cell are always kept in
        # place, the lowermost are duplicated towards the end of the face
        # definition.
        self.assertTrue(np.all(new_projection_avg[0, fi[0]] == 1))
        self.assertTrue(np.all(new_projection_avg[1, fi[0]] == 1))
        self.assertTrue(np.all(new_projection_avg[2, fi[1]] == 1))
        self.assertTrue(np.all(new_projection_avg[3, fi[1]] == 1))

    def test_refine_distort_high_dim(self):
        # Replace the 2d grid with a finer one, and move the nodes along the
        # interface so that areas along the interface are no longer equal.

        mdg, _ = pp.md_grids_2d.single_horizontal([1, 2], simplex=False)

        intf_old = mdg.interfaces()[0]

        old_projection = intf_old.primary_to_mortar_int().copy()
        sd_old = mdg.subdomains(dim=2)[0]

        # Create a new, finer 2d grid. This is the simplest
        # way to put the fracture in the right place is to create a new
        # bucket, and pick out the higher dimensional grid
        mdg_new, _ = pp.md_grids_2d.single_horizontal([2, 2], simplex=False)

        sd_new = mdg_new.subdomains(dim=2)[0]

        # By construction of the split grid, we know that the nodes at
        # (0.5, 0.5) are no 5 and 6, and that no 5 is associated with the
        # face belonging to the lower cells.
        # Move node belonging to the lower face
        sd_new.nodes[0, 5] = 0.2
        sd_new.nodes[0, 6] = 0.7
        sd_new.compute_geometry()

        mdg.replace_subdomains_and_interfaces({sd_old: sd_new})

        # Get mortar grid again
        intf_new = mdg.interfaces()[0]

        new_projection_avg = intf_new.primary_to_mortar_avg()
        new_projection_int = intf_new.primary_to_mortar_int()

        # Check shape

        self.assertTrue(new_projection_avg.shape[0] == old_projection.shape[0])
        self.assertTrue(new_projection_avg.shape[1] == sd_new.num_faces)

        # Projection of averages sum to unity in the rows
        self.assertTrue(np.all(new_projection_avg.toarray().sum(axis=1) == 1))
        # Columns in integrated projection sum to either 0 or 1.
        self.assertTrue(
            np.all(
                np.logical_or(
                    new_projection_int.A.sum(axis=0) == 1,
                    new_projection_int.A.sum(axis=0) == 0,
                ),
            )
        )

        fi = np.where(sd_new.face_centers[1] == 0.5)[0]
        self.assertTrue(fi.size == 4)
        # Hard coded test (based on knowledge of how the grids and pp.meshing
        # is implemented). Faces to the uppermost cell are always kept in
        # place, the lowermost are duplicated towards the end of the face
        # definition.
        self.assertTrue(np.abs(new_projection_avg[0, 8] - 0.7 < 1e-6))
        self.assertTrue(np.abs(new_projection_avg[0, 9] - 0.3 < 1e-6))
        self.assertTrue(np.abs(new_projection_avg[1, 12] - 0.2 < 1e-6))
        self.assertTrue(np.abs(new_projection_avg[1, 13] - 0.8 < 1e-6))

    def test_distort_high_dim(self):
        # Replace the 2d grid with a finer one, and move the nodes along the
        # interface so that areas along the interface are no longer equal.

        mdg, _ = pp.md_grids_2d.single_horizontal([2, 2], simplex=False)

        intf_old = mdg.interfaces()[0]

        old_projection = intf_old.primary_to_mortar_int().copy()
        sd_old = mdg.subdomains(dim=2)[0]

        # Create a new, finer 2d grid. This is the simplest
        # way to put the fracture in the right place is to create a new
        # bucket, and pick out the higher dimensional grid
        mdg_new, _ = pp.md_grids_2d.single_horizontal([2, 2], simplex=False)

        sd_new = mdg_new.subdomains(dim=2)[0]

        # By construction of the split grid, we know that the nodes at
        # (0.5, 0.5) are no 5 and 6, and that no 5 is associated with the
        # face belonging to the lower cells.
        # Move node belonging to the lower face
        sd_new.nodes[0, 5] = 0.2
        sd_new.nodes[0, 6] = 0.7
        sd_new.compute_geometry()

        mdg.replace_subdomains_and_interfaces({sd_old: sd_new})

        # Get mortar grid again
        intf_new = mdg.interfaces()[0]

        new_projection_avg = intf_new.primary_to_mortar_avg()
        new_projection_int = intf_new.primary_to_mortar_int()

        # Check shape
        self.assertTrue(new_projection_avg.shape[0] == old_projection.shape[0])
        self.assertTrue(new_projection_avg.shape[1] == sd_new.num_faces)
        # Projection of averages sums to unity in the rows
        self.assertTrue(np.all(new_projection_avg.toarray().sum(axis=1) == 1))
        # Columns in integrated projection sum to either 0 or 1.
        self.assertTrue(
            np.all(
                np.logical_or(
                    new_projection_int.A.sum(axis=0) == 1,
                    new_projection_int.A.sum(axis=0) == 0,
                ),
            )
        )

        fi = np.where(sd_new.face_centers[1] == 0.5)[0]
        self.assertTrue(fi.size == 4)
        # Hard coded test (based on knowledge of how the grids and pp.meshing
        # is implemented). Faces to the uppermost cell are always kept in
        # place, the lowermost are duplicated towards the end of the face
        # definition.

        # It seems the mortar grid is designed so that the first cell is
        # associated with face 9 in the old grid. This is split into 2/5 face
        # 8 and 3/5 face 9.

        self.assertTrue(np.abs(new_projection_avg[0, 8] - 0.4 < 1e-6))
        self.assertTrue(np.abs(new_projection_avg[0, 9] - 0.6 < 1e-6))
        # The second cell in mortar grid is still fully connected to face 9
        self.assertTrue(np.abs(new_projection_avg[1, 9] - 1 < 1e-6))
        self.assertTrue(np.abs(new_projection_avg[2, 13] - 1 < 1e-6))
        self.assertTrue(np.abs(new_projection_avg[3, 12] - 0.4 < 1e-6))
        self.assertTrue(np.abs(new_projection_avg[3, 13] - 0.6 < 1e-6))

    def test_permute_nodes_in_replacement_grid(self):
        # Replace higher dimensional grid with an identical one, except the
        # node indices are perturbed. This will test sorting of nodes along
        # 1d lines
        mdg, _ = pp.md_grids_2d.single_horizontal([2, 2], simplex=False)

        intf_old = mdg.interfaces()[0]

        old_projection = intf_old.primary_to_mortar_int().copy()
        sd_old = mdg.subdomains(dim=2)[0]

        # Create a new, finer 2d grid. This is the simplest
        # way to put the fracture in the right place is to create a new
        # bucket, and pick out the higher dimensional grid
        mdg_new, _ = pp.md_grids_2d.single_horizontal([2, 2], simplex=False)

        sd_new = mdg_new.subdomains(dim=2)[0]

        # By construction of the split grid, we know that the nodes at
        # (0.5, 0.5) are no 5 and 6, and that no 5 is associated with the
        # face belonging to the lower cells.
        # Move node belonging to the lower face
        #     g_new.nodes[0, 5] = 0.2
        #     g_new.nodes[0, 6] = 0.7

        # Replacements: along lower segment (3, 5, 7) -> (7, 5, 3)
        # On upper segment: (4, 6, 8) -> (8, 4, 6)
        sd_new.nodes[0, 3] = 1
        sd_new.nodes[0, 4] = 0.5
        sd_new.nodes[0, 5] = 0.5
        sd_new.nodes[0, 6] = 1
        sd_new.nodes[0, 7] = 0
        sd_new.nodes[0, 8] = 0

        fn = sd_new.face_nodes.indices.reshape((2, sd_new.num_faces), order="F")
        fn[:, 8] = np.array([4, 8])
        fn[:, 9] = np.array([4, 6])
        fn[:, 12] = np.array([7, 5])
        fn[:, 13] = np.array([5, 3])

        mdg.replace_subdomains_and_interfaces({sd_old: sd_new})

        # Get mortar grid again
        intf_new = mdg.interfaces()[0]

        new_projection_avg = intf_new.primary_to_mortar_avg()
        new_projection_int = intf_new.primary_to_mortar_int()

        # Check shape
        self.assertTrue(new_projection_avg.shape[0] == old_projection.shape[0])
        self.assertTrue(new_projection_avg.shape[1] == sd_new.num_faces)

        # Projection of averages sum to unity in the rows
        self.assertTrue(np.all(new_projection_avg.toarray().sum(axis=1) == 1))
        # Columns in integrated projection sum to either 0 or 1.
        self.assertTrue(
            np.all(
                np.logical_or(
                    new_projection_int.A.sum(axis=0) == 1,
                    new_projection_int.A.sum(axis=0) == 0,
                ),
            )
        )
        fi = np.where(sd_new.face_centers[1] == 0.5)[0]
        self.assertTrue(fi.size == 4)
        # Hard coded test (based on knowledge of how the grids and pp.meshing
        # is implemented). Faces to the uppermost cell are always kept in
        # place, the lowermost are duplicated towards the end of the face
        # definition.
        self.assertTrue((old_projection != new_projection_avg).nnz == 0)

    def test_permute_perturb_nodes_in_replacement_grid(self):
        # Replace higher dimensional grid with an identical one, except the
        # node indices are perturbed. This will test sorting of nodes along
        # 1d lines. Also perturb nodes along the segment.
        mdg, _ = pp.md_grids_2d.single_horizontal([2, 2], simplex=False)

        intf_old = mdg.interfaces()[0]

        old_projection = intf_old.primary_to_mortar_int().copy()
        sd_old = mdg.subdomains(dim=2)[0]

        # Create a new, finer 2d grid. This is the simplest
        # way to put the fracture in the right place is to create a new
        # bucket, and pick out the higher dimensional grid
        mdg_new, _ = pp.md_grids_2d.single_horizontal([2, 2], simplex=False)

        sd_new = mdg_new.subdomains(dim=2)[0]
        # By construction of the split grid, we know that the nodes at
        # (0.5, 0.5) are no 5 and 6, and that no 5 is associated with the
        # face belonging to the lower cells.
        # Replacements: along lower segment (3, 5, 7) -> (7, 5, 3)
        # On upper segment: (4, 6, 8) -> (8, 4, 6)
        sd_new.nodes[0, 3] = 1
        sd_new.nodes[0, 4] = 0.7
        sd_new.nodes[0, 5] = 0.2
        sd_new.nodes[0, 6] = 1
        sd_new.nodes[0, 7] = 0
        sd_new.nodes[0, 8] = 0

        fn = sd_new.face_nodes.indices.reshape((2, sd_new.num_faces), order="F")
        fn[:, 8] = np.array([4, 8])
        fn[:, 9] = np.array([4, 6])
        fn[:, 12] = np.array([7, 5])
        fn[:, 13] = np.array([5, 3])

        mdg.replace_subdomains_and_interfaces({sd_old: sd_new})
        # Get mortar grid again
        intf_new = mdg.interfaces()[0]

        new_projection_avg = intf_new.primary_to_mortar_avg()
        new_projection_int = intf_new.primary_to_mortar_int()

        # Check shape
        self.assertTrue(new_projection_avg.shape[0] == old_projection.shape[0])
        self.assertTrue(new_projection_avg.shape[1] == sd_new.num_faces)
        # Projection of averages sum to unity in the rows
        self.assertTrue(np.all(new_projection_avg.toarray().sum(axis=1) == 1))
        # Columns in integrated projection sum to either 0 or 1.
        self.assertTrue(
            np.all(
                np.logical_or(
                    new_projection_int.A.sum(axis=0) == 1,
                    new_projection_int.A.sum(axis=0) == 0,
                ),
            )
        )

        fi = np.where(sd_new.face_centers[1] == 0.5)[0]
        self.assertTrue(fi.size == 4)
        # Hard coded test (based on knowledge of how the grids and pp.meshing
        # is implemented). Faces to the uppermost cell are always kept in
        # place, the lowermost are duplicated towards the end of the face
        # definition.
        # It seems the mortar grid is designed so that the first cell is
        # associated with face 9 in the old grid. This is split into 2/5 face
        # 8 and 3/5 face 9.

        self.assertTrue(np.abs(new_projection_avg[0, 8] - 0.4 < 1e-6))
        self.assertTrue(np.abs(new_projection_avg[0, 9] - 0.6 < 1e-6))
        # The second cell in mortar grid is still fully connected to face 9
        self.assertTrue(np.abs(new_projection_avg[1, 9] - 1 < 1e-6))
        self.assertTrue(np.abs(new_projection_avg[2, 13] - 1 < 1e-6))
        self.assertTrue(np.abs(new_projection_avg[3, 12] - 0.4 < 1e-6))
        self.assertTrue(np.abs(new_projection_avg[3, 13] - 0.6 < 1e-6))


"""Various tests for replacing subdomain and interface grids in a 3d mixed-
dimensional grid.

This can be considered integration tests for the method
replace_subdomains_and_interfaces in the MixedDimensionalGrid class.

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


def test_replace_1d_with_identity():
    """Generate the full md_grid, and replace the 1d grid with a copy of itself.

    This should not change the mortar mappings. An error here indicates something is
    fundamentally wrong with the implementation of the method
    replace_subdomains_and_interfaces, or with the functions to match 1d grids (used
    when updating the grid).

    """
    mdg = _setup_mdg(pert=False)

    # Fetch the mortar mappings
    old_proj_1_h, old_proj_1_l, old_proj_2_h, old_proj_2_l = _mortar_projections(mdg)

    gn = _grid_1d(2)
    go = mdg.subdomains(dim=1)[0]
    mdg.replace_subdomains_and_interfaces({go: gn})

    # Fetch the new mortar mappings.
    new_proj_1_h, new_proj_1_l, new_proj_2_h, new_proj_2_l = _mortar_projections(mdg)

    # There should be no changes to the mortar mappings, we can compare them directly
    _compare_mortar_projections(
        [
            (old_proj_1_h, new_proj_1_h),
            (old_proj_1_l, new_proj_1_l),
            (old_proj_2_h, new_proj_2_h),
            (old_proj_2_l, new_proj_2_l),
        ]
    )


def test_replace_2d_with_identity_no_1d():
    """Generate the an md grid of a 3d and a 2d grid,  replace the 2d grid with a copy
    of itself.

    This should not change the mortar mappings. An error here indicates something is
    fundamentally wrong with the implementation of the method
    replace_subdomains_and_interfaces, or with the functions to match 2d grids (used
    when updating the grid).

    """

    mdg = _setup_mdg(pert=False, include_1d=False)

    # Fetch the mortar mappings. There is no 1d grid, thus the mappings to 1d are None
    _, _, old_proj_2_h, old_proj_2_l = _mortar_projections(mdg)

    gn = _grid_2d_two_cells(pert=False, include_1d=False)
    go = mdg.subdomains(dim=2)[0]
    mdg.replace_subdomains_and_interfaces(sd_map={go: gn})

    # Fetch the new mortar mappings.
    _, _, new_proj_2_h, new_proj_2_l = _mortar_projections(mdg)
    _compare_mortar_projections(
        [
            (old_proj_2_h, new_proj_2_h),
            (old_proj_2_l, new_proj_2_l),
        ]
    )


def test_replace_2d_with_finer_no_1d():
    """Generate the an md grid of a 3d and a 2d grid, replace the 2d grid with a refined
    2d grid.

    This will change the mappings between the mortar and the 2d grid. An error most
    likely points to a problem with match_2d_grids.

    """

    mdg = _setup_mdg(pert=False, include_1d=False)

    # Fetch the mortar mappings. There is no 1d grid, thus the mappings to 1d are None
    _, _, old_proj_2_h, _ = _mortar_projections(mdg)

    gn = _grid_2d_four_cells(include_1d=False, pert=False, move_interior_point=False)
    go = mdg.subdomains(dim=2)[0]
    mdg.replace_subdomains_and_interfaces({go: gn})

    # Fetch the new mortar mappings.
    _, _, new_proj_2_h, new_proj_2_l = _mortar_projections(mdg)

    # There should be no changes in the mapping between mortar and primary
    # The known projection matrix, from secondary to one of the mortar grids.
    known_p_2_l = np.tile(np.array([[1, 1, 0, 0], [0, 0, 1, 1]]), (2, 1))

    _compare_mortar_projections(
        [
            (old_proj_2_h, new_proj_2_h),
            (known_p_2_l, new_proj_2_l),
        ]
    )


def test_replace_2d_with_finer_no_1d_pert():
    """Generate the an md grid of a 3d and a 2d grid, replace the 2d grid with a refined
    and perturbed 2d grid.

    This will change the mappings between the mortar and the 2d grid. An error most
    likely points to a problem with the method match_grids.match_2d().

    """

    # Create the md_grid, single 2d grid, no splitting.
    mdg = _setup_mdg(pert=True, include_1d=False)

    # Fetch the mortar mappings. There is no 1d grid, thus the mappings to 1d are None
    _, _, old_proj_2_h, _ = _mortar_projections(mdg)

    gn = _grid_2d_four_cells(pert=True, include_1d=False, move_interior_point=False)
    go = mdg.subdomains(dim=2)[0]
    mdg.replace_subdomains_and_interfaces({go: gn})

    # Fetch the mortar mappings again. There is no 1d grid, thus the mappings to 1d are None
    _, _, new_proj_2_h, new_proj_2_l = _mortar_projections(mdg)

    # The known projection matrix, from secondary to one of the mortar grids.
    known_2_l = np.tile(np.array([[1, 1, 1 / 3, 1 / 3], [0, 0, 2 / 3, 2 / 3]]), (2, 1))
    # There should be no changes in the mapping between mortar and primary
    _compare_mortar_projections(
        [
            (old_proj_2_h, new_proj_2_h),
            (known_2_l, new_proj_2_l),
        ]
    )


def test_replace_2d_with_identity():
    """Generate the an md grid of a 3d, 2d and 1d grid, replace the 2d grid with a copy
    of itself.

    This should not change the mortar mappings. An error here indicates something is
    fundamentally wrong with the implementation of the method
    replace_subdomains_and_interfaces, or with the functions to match 2d grids (used
    when updating the grid).

    """

    mdg = _setup_mdg(pert=False, include_1d=True)
    # Fetch the mortar mappings.
    old_proj_1_h, old_proj_1_l, old_proj_2_h, old_proj_2_l = _mortar_projections(mdg)

    # Do a formal replacement
    gn = _grid_2d_two_cells(pert=False, include_1d=True)
    go = mdg.subdomains(dim=2)[0]
    mdg.replace_subdomains_and_interfaces({go: gn})

    # Fetch the mortar mappings again
    new_proj_1_h, new_proj_1_l, new_proj_2_h, new_proj_2_l = _mortar_projections(mdg)

    # None of the mappings should have changed
    _compare_mortar_projections(
        [
            (old_proj_2_h, new_proj_2_h),
            (old_proj_2_l, new_proj_2_l),
            (old_proj_1_h, new_proj_1_h),
            (old_proj_1_l, new_proj_1_l),
        ]
    )


def test_replace_2d_with_finer_pert():
    """Generate the an md grid of a 3d, 2d and 1d grid, replace the 2d grid with a finer
    and perturbed 2d grid.

    An error here will most likely indicate a problem with the method
    match_grids.match_2d.

    """
    mdg = _setup_mdg(pert=True, include_1d=True)

    # Fetch the mortar mappings.
    old_proj_1_h, old_proj_1_l, old_proj_2_h, old_proj_2_l = _mortar_projections(mdg)

    # Change the 2d grid, obtain new mortar mappings
    gn = _grid_2d_four_cells(include_1d=True, pert=True, move_interior_point=True)
    go = mdg.subdomains(dim=2)[0]
    mdg.replace_subdomains_and_interfaces({go: gn})
    new_proj_1_h, new_proj_1_l, new_proj_2_h, new_proj_2_l = _mortar_projections(mdg)

    # The mappings between mortars and the 2d grids will change, between mortar and
    # 3d/1d will stay the same.
    known_proj_2_l = np.tile(np.array([[1, 1, 0, 0], [0, 0, 1, 1]]), (2, 1))
    known_proj_1_h = np.zeros((2, 10))
    known_proj_1_h[0, [2, 4]] = 1
    known_proj_1_h[1, [7, 8]] = 1

    # Verify that the mappings are as expected
    _compare_mortar_projections(
        [
            (old_proj_2_h, new_proj_2_h),
            (known_proj_2_l, new_proj_2_l),
            (known_proj_1_h, new_proj_1_h),
            (old_proj_1_l, new_proj_1_l),
        ]
    )


def _compare_mortar_projections(
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


def _mortar_projections(mdg: pp.MixedDimensionalGrid):
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


def _setup_mdg(pert: bool = False, include_1d: bool = True):
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
        cell_centers = np.array([[2.0 / 3, 0, 1.0 / 3], [1.0 / 3, 0, 2.0 / 3]]).T
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
        cell_centers = np.array([[2.0 / 3, 0, 1.0 / 3], [1.0 / 3, 0, 2.0 / 3]]).T
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
                [0.0, 1.0, 1.0, 0.5, 0.0, 0.0, 1.0, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.5, 0.0, 1.0, 1.0, 0.5],
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
                [0.0, 1.0, 1.0, 0.0, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.5],
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
        [[0.5, 0, 1.0 / 6], [5.0 / 6, 0, 0.5], [0.5, 0, 5.0 / 6], [1.0 / 6, 0, 0.5]]
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
    fn = "tmp.grid"
    g.compute_geometry()
    mg = pp.MortarGrid(g.dim, {0: g, 1: g})

    pickle.dump(mg, open(fn, "wb"))
    mg_read = pickle.load(open(fn, "rb"))

    pp.test_utils.grids.compare_mortar_grids(mg, mg_read)

    mg_one_sided = pp.MortarGrid(g.dim, {0: g})

    pickle.dump(mg, open(fn, "wb"))
    mg_read = pickle.load(open(fn, "rb"))

    pp.test_utils.grids.compare_mortar_grids(mg_one_sided, mg_read)

    Path.unlink(fn)
