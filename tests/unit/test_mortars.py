"""Tests of the mortar grids. Mainly focuses on mappings between mortar grids and
surrounding grids. 

The module contains the following groups of tests:

    TestGridMappings1d: In practice tests of pp.match_grids.match_1d().

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
import pytest
import unittest

import numpy as np
import scipy.sparse as sps

from tests import test_utils
import porepy as pp


class TestGridMappings1d(unittest.TestCase):
    """Tests of matching of 1d grids.

    This is in practice a test of pp.match_grids.match_1d()
    """

    def test_merge_grids_all_common(self):
        """ "Replace a grid by itself. The mappings should be identical."""
        g = pp.TensorGrid(np.arange(3))
        g.compute_geometry()
        mat = pp.match_grids.match_1d(g, g, tol=1e-4)
        mat.eliminate_zeros()
        mat = mat.tocoo()

        self.assertTrue(np.allclose(mat.data, np.ones(2)))
        self.assertTrue(np.allclose(mat.row, np.arange(2)))
        self.assertTrue(np.allclose(mat.col, np.arange(2)))

    def test_merge_grids_non_matching(self):
        """ "Perturb one node in the new grid."""
        g = pp.TensorGrid(np.arange(3))
        h = pp.TensorGrid(np.arange(3))
        g.nodes[0, 1] = 0.5
        g.compute_geometry()
        h.compute_geometry()
        mat = pp.match_grids.match_1d(g, h, tol=1e-4, scaling="averaged")
        mat.eliminate_zeros()
        mat = mat.tocoo()

        # Weights give mappings from h to g. The first cell in h is
        # fully within the first cell in g. The second in h is split 1/3
        # in first of g, 2/3 in second.
        self.assertTrue(np.allclose(mat.data, np.array([1, 1.0 / 3, 2.0 / 3])))
        self.assertTrue(np.allclose(mat.row, np.array([0, 1, 1])))
        self.assertTrue(np.allclose(mat.col, np.array([0, 0, 1])))

    def test_merge_grids_reverse_order(self):
        g = pp.TensorGrid(np.arange(3))
        h = pp.TensorGrid(np.arange(3))
        h.nodes = h.nodes[:, ::-1]
        g.compute_geometry()
        h.compute_geometry()
        mat = pp.match_grids.match_1d(g, h, tol=1e-4, scaling="averaged")
        mat.eliminate_zeros()
        mat = mat.tocoo()

        self.assertTrue(np.allclose(mat.data, np.array([1, 1])))
        # In this case, we don't know which ordering the combined grid uses
        # Instead, make sure that the two mappings are ordered in separate
        # directions
        self.assertTrue(np.allclose(mat.row[::-1], mat.col))

    def test_merge_grids_split(self):
        g1 = pp.TensorGrid(np.linspace(0, 2, 2))
        g2 = pp.TensorGrid(np.linspace(2, 4, 2))
        g_nodes = np.hstack((g1.nodes, g2.nodes))
        g_face_nodes = sps.block_diag((g1.face_nodes, g2.face_nodes), "csc")
        g_cell_faces = sps.block_diag((g1.cell_faces, g2.cell_faces), "csc")
        g = pp.Grid(1, g_nodes, g_face_nodes, g_cell_faces, "pp.TensorGrid")

        h1 = pp.TensorGrid(np.linspace(0, 2, 3))
        h2 = pp.TensorGrid(np.linspace(2, 4, 3))
        h_nodes = np.hstack((h1.nodes, h2.nodes))
        h_face_nodes = sps.block_diag((h1.face_nodes, h2.face_nodes), "csc")
        h_cell_faces = sps.block_diag((h1.cell_faces, h2.cell_faces), "csc")
        h = pp.Grid(1, h_nodes, h_face_nodes, h_cell_faces, "pp.TensorGrid")

        g.compute_geometry()
        h.compute_geometry()
        # Construct a map from g to h
        mat_g_2_h = pp.match_grids.match_1d(h, g, tol=1e-4, scaling="averaged")
        mat_g_2_h.eliminate_zeros()
        mat_g_2_h = mat_g_2_h.tocoo()

        # Weights give mappings from g to h.
        self.assertTrue(np.allclose(mat_g_2_h.data, np.array([1.0, 1.0, 1.0, 1.0])))
        self.assertTrue(np.allclose(mat_g_2_h.row, np.array([0, 1, 2, 3])))
        self.assertTrue(np.allclose(mat_g_2_h.col, np.array([0, 0, 1, 1])))

        # Next, make a map from h to g. In this case, the cells in h are split in two
        # thus the weight is 0.5.
        mat_h_2_g = pp.match_grids.match_1d(g, h, tol=1e-4, scaling="averaged")
        mat_h_2_g.eliminate_zeros()
        mat_h_2_g = mat_h_2_g.tocoo()

        self.assertTrue(np.allclose(mat_h_2_g.data, np.array([0.5, 0.5, 0.5, 0.5])))
        self.assertTrue(np.allclose(mat_h_2_g.row, np.array([0, 0, 1, 1])))
        self.assertTrue(np.allclose(mat_h_2_g.col, np.array([0, 1, 2, 3])))


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
        # 1x2 grid.
        # Copy the higher dimensional grid and replace. The mapping should be
        # the same.
        mdg, _ = pp.grid_buckets_2d.single_horizontal([1, 2], simplex=False)

        intf_old = list(mdg.interfaces())[0]

        old_projection = intf_old.primary_to_mortar_int().copy()

        sd_old = list(mdg.subdomains(dim=2))[0]
        sd_new = sd_old.copy()

        mdg.replace_subdomains_and_interfaces({sd_old: sd_new})

        # Get mortar grid again
        intf_new = list(mdg.interfaces())[0]

        new_projection = intf_new.primary_to_mortar_int()

        # The projections should be identical
        self.assertTrue((old_projection != new_projection).nnz == 0)

    def test_refine_high_dim(self):
        # Replace the 2d grid with a finer one

        mdg, _ = pp.grid_buckets_2d.single_horizontal([1, 2], simplex=False)

        intf_old = list(mdg.interfaces())[0]

        old_projection = intf_old.primary_to_mortar_int().copy()
        sd_old = list(mdg.subdomains(dim=2))[0]

        # Create a new, finer 2d grid. This is the simplest
        # way to put the fracture in the right place is to create a new
        # bucket, and pick out the higher dimensional grid
        mdg_new, _ = pp.grid_buckets_2d.single_horizontal([2, 2], simplex=False)
        mdg_new.compute_geometry()

        sd_new = list(mdg_new.subdomains(dim=2))[0]

        mdg.replace_subdomains_and_interfaces({sd_old: sd_new})

        # Get mortar grid again
        intf_new = list(mdg.interfaces())[0]

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

        mdg, _ = pp.grid_buckets_2d.single_horizontal([2, 2], simplex=False)

        # Pick out mortar grid by a loop, there is only one edge in the bucket
        intf_old = list(mdg.interfaces())[0]
        old_projection = intf_old.primary_to_mortar_int().copy()
        sd_old = list(mdg.subdomains(dim=2))[0]

        # Create a new, coarser 2d grid. This is the simplest
        # way to put the fracture in the right place is to create a new
        # bucket, and pick out the higher dimensional grid
        mdg_new, _ = pp.grid_buckets_2d.single_horizontal([1, 2], simplex=False)

        sd_new = list(mdg_new.subdomains(dim=2))[0]

        mdg.replace_subdomains_and_interfaces({sd_old: sd_new})

        # Get mortar grid again
        intf_new = list(mdg.interfaces())[0]

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

        mdg, _ = pp.grid_buckets_2d.single_horizontal([1, 2], simplex=False)

        intf_old = list(mdg.interfaces())[0]

        old_projection = intf_old.primary_to_mortar_int().copy()
        sd_old = list(mdg.subdomains(dim=2))[0]

        # Create a new, finer 2d grid. This is the simplest
        # way to put the fracture in the right place is to create a new
        # bucket, and pick out the higher dimensional grid
        mdg_new, _ = pp.grid_buckets_2d.single_horizontal([2, 2], simplex=False)

        sd_new = list(mdg_new.subdomains(dim=2))[0]

        # By construction of the split grid, we know that the nodes at
        # (0.5, 0.5) are no 5 and 6, and that no 5 is associated with the
        # face belonging to the lower cells.
        # Move node belonging to the lower face
        sd_new.nodes[0, 5] = 0.2
        sd_new.nodes[0, 6] = 0.7
        sd_new.compute_geometry()

        mdg.replace_subdomains_and_interfaces({sd_old: sd_new})

        # Get mortar grid again
        intf_new = list(mdg.interfaces())[0]

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

        mdg, _ = pp.grid_buckets_2d.single_horizontal([2, 2], simplex=False)

        intf_old = list(mdg.interfaces())[0]

        old_projection = intf_old.primary_to_mortar_int().copy()
        sd_old = list(mdg.subdomains(dim=2))[0]

        # Create a new, finer 2d grid. This is the simplest
        # way to put the fracture in the right place is to create a new
        # bucket, and pick out the higher dimensional grid
        mdg_new, _ = pp.grid_buckets_2d.single_horizontal([2, 2], simplex=False)

        sd_new = list(mdg_new.subdomains(dim=2))[0]

        # By construction of the split grid, we know that the nodes at
        # (0.5, 0.5) are no 5 and 6, and that no 5 is associated with the
        # face belonging to the lower cells.
        # Move node belonging to the lower face
        sd_new.nodes[0, 5] = 0.2
        sd_new.nodes[0, 6] = 0.7
        sd_new.compute_geometry()

        mdg.replace_subdomains_and_interfaces({sd_old: sd_new})

        # Get mortar grid again
        intf_new = list(mdg.interfaces())[0]

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
        mdg, _ = pp.grid_buckets_2d.single_horizontal([2, 2], simplex=False)

        intf_old = list(mdg.interfaces())[0]

        old_projection = intf_old.primary_to_mortar_int().copy()
        sd_old = list(mdg.subdomains(dim=2))[0]

        # Create a new, finer 2d grid. This is the simplest
        # way to put the fracture in the right place is to create a new
        # bucket, and pick out the higher dimensional grid
        mdg_new, _ = pp.grid_buckets_2d.single_horizontal([2, 2], simplex=False)

        sd_new = list(mdg_new.subdomains(dim=2))[0]

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
        intf_new = list(mdg.interfaces())[0]

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
        mdg, _ = pp.grid_buckets_2d.single_horizontal([2, 2], simplex=False)

        intf_old = list(mdg.interfaces())[0]

        old_projection = intf_old.primary_to_mortar_int().copy()
        sd_old = list(mdg.subdomains(dim=2))[0]

        # Create a new, finer 2d grid. This is the simplest
        # way to put the fracture in the right place is to create a new
        # bucket, and pick out the higher dimensional grid
        mdg_new, _ = pp.grid_buckets_2d.single_horizontal([2, 2], simplex=False)

        sd_new = list(mdg_new.subdomains(dim=2))[0]
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
        intf_new = list(mdg.interfaces())[0]

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


class MockGrid:
    """Data structure for a mock grid. Used to mimic the full PorePy grid structure,
    while still having full control of the underlying data.
    """

    def __init__(self, nodes, fn, cf, cc, n, cv, dim, frac_face=None, glob_pi=None):
        self.nodes = nodes
        self.face_nodes = fn
        self.cell_faces = cf
        self.cell_centers = cc
        self.face_normals = n
        self.cell_volumes = cv

        self.num_nodes = self.nodes.shape[1]
        self.num_faces = self.face_nodes.shape[1]
        self.num_cells = self.cell_faces.shape[1]

        self.dim = dim
        if self.dim == 1:
            self.name = ["pp.TensorGrid"]
        elif self.dim == 2:
            self.name = ["TriangleGrid"]
        elif self.dim == 3:
            self.name = ["TetrahedralGrid"]

        ff = np.zeros(self.num_faces, dtype=bool)
        if frac_face is not None:
            ff[frac_face] = 1
        self.tags = {"fracture_faces": ff}

        if glob_pi is not None:
            self.global_point_ind = glob_pi

        self.history = []

    def cell_nodes(self):
        return self.face_nodes * self.cell_faces

    def copy(self):
        g = MockGrid(
            self.nodes.copy(),
            self.face_nodes.copy(),
            self.cell_faces.copy(),
            self.cell_centers.copy(),
            self.face_normals.copy(),
            self.cell_volumes.copy(),
            self.dim,
        )
        return g

    def get_boundary_faces(self):
        return np.arange(self.num_faces)


class TestReplace1dand2dGridsIn3dDomain(unittest.TestCase):
    """Various tests for replacing subdomain and interface grids in a 3d mixed-dimensional
    grid.

    The grid consists of the following components:
        A 3d grid, which is intersected by a fracture at y=0. On each side of this plane
        the grid has 5 nodes (four of them along y=0, the fifth is at y=+-1) and two cells.
        Most importantly, each side (above and below y=0) has two faces at y=0, one
        with node coordinates {(0, 0), (1, 0), (1, 1)}, the other with {(0, 0), (1, 1), (0, 1)}.
        The node at (1, 1) will in some cases be perturbed to (1, 2) (if pert=True),
        besides this, the 3d grid is not changed in the tests, and really not that
        important.

        A 2d grid, which is located at y=0. Two versions of this grid can be generated:
            1) One with four nodes, at {(0, 0), (1, 0), (1, 1), (0, 1)} - the third node
               is moved to (1, 2) if pert=True - and two cells formed by the sets of nodes
               {(0, 0), (1, 0), (1, 1)} and {(0, 0), (1, 1), (0, 1)}.
            2) One with five nodes, at {(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)}, and
               four cells formed by the midle node and pairs of neighboring nodes on the sides.
               Again, pert=True will move the node (1, 1), but it this case, it will also
               move the midle node to (0.5, 1) to ensure it stays on the line between
               (0, 0) and now (1, 2) - or else the 1d grid defined below will not conform
               to the 2d faces.
        Several of the tests below consists of replacing the 2d grid with two cells with
        that with four cells, and ensure all mappings are correct.

        A 1d grid that extends from (0, 0) to (1, 1) (or (1, 2) if pert=True).

        At the 3d-2d and 2d-1d interfaces, there are of course mortar grids that will have
        their mappings updated as the adjacent subdomain grids are replaced.

    IMPLEMENTATION NOTE: When perturbing the grid (moving (1, 1) -> (1, 2)), several updates
    of the grid geometry are hardcoded, like cell centers, face normals etc. This is messy,
    and a more transparent approach would have been preferrable, but it will have to do
    for now.

    """

    def grid_3d(self, pert=False):
        # Grid consisting of 3d, 2d and 1d grid. Nodes along the main
        # surface are split into two or four.
        # The first two cells are below the plane y=0, the last two are above the plane.
        n = np.array(
            [
                [0, -1, 0],
                [0, 0, 0],
                [1, 0, 0],
                [1, 0, 1],
                [0, 0, 0],
                [1, 0, 1],
                [0, 0, 1],
                [0, 0, 0],
                [1, 0, 0],
                [1, 0, 1],
                [0, 0, 0],
                [1, 0, 1],
                [0, 0, 1],
                [0, 1, 0],
            ]
        ).T

        fn = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 1],
                [1, 2, 3],
                [0, 4, 5],
                [0, 5, 6],
                [0, 6, 4],
                [4, 5, 6],
                [13, 7, 8],
                [13, 8, 9],
                [13, 9, 7],
                [7, 8, 9],
                [13, 10, 11],
                [13, 11, 12],
                [13, 12, 10],
                [10, 11, 12],
            ]
        ).T
        cols = np.tile(np.arange(fn.shape[1]), (fn.shape[0], 1)).ravel("F")
        face_nodes = sps.csc_matrix((np.ones_like(cols), (fn.ravel("F"), cols)))

        cf = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]).T
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
        # We will only use face normals for faces on the interface
        face_normals[1, [3, 7, 11, 15]] = 1

        cell_volumes = 1 / 6 * np.ones(cell_centers.shape[1])

        if pert:
            # Move the node at  (1, 0, 1) to (1, 0, 2)
            # This will invalidate the assigned geometry, but it should not matter
            # since we do not use the face_area for anything (we never replace the
            # highest-dimensional, 3d grid)
            n[2, [3, 9]] = 2

        g = MockGrid(
            n,
            face_nodes,
            cell_faces,
            cell_centers,
            face_normals,
            cell_volumes,
            3,
            frac_face=[3, 7, 11, 15],
        )
        g.global_point_ind = np.arange(n.shape[1])
        return g

    def grid_3d_no_1d(self, pert=False):
        # Domain consisting of only 2d and 3d grid. The nodes along the 2d grid
        # are split in two
        n = np.array(
            [
                [0, -1, 0],
                [0, 0, 0],
                [1, 0, 0],
                [1, 0, 1],
                [0, 0, 1],
                [0, 0, 0],
                [1, 0, 0],
                [1, 0, 1],
                [0, 0, 1],
                [0, 1, 0],
            ]
        ).T

        fn = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 4],
                [0, 4, 1],
                [0, 1, 3],
                [1, 2, 3],
                [1, 3, 4],  # Up to now, the lower half. Then the upper
                [9, 5, 6],
                [9, 6, 7],
                [9, 7, 8],
                [9, 8, 5],
                [9, 5, 7],
                [5, 6, 7],
                [5, 7, 8],
            ]
        ).T
        cols = np.tile(np.arange(fn.shape[1]), (fn.shape[0], 1)).ravel("F")
        face_nodes = sps.csc_matrix((np.ones_like(cols), (fn.ravel("F"), cols)))

        cf = np.array([[0, 1, 4, 5], [2, 3, 4, 6], [7, 8, 11, 12], [9, 10, 11, 13]]).T
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
        # We will only use face normals for faces on the interface
        face_normals[2, [5, 6, 12, 13]] = 1

        cell_volumes = 1 / 6 * np.ones(cell_centers.shape[1])

        if pert:
            # This will invalidate the assigned geometry, but it should not matter
            n[2, 4] = 2
            n[2, 9] = 2

        g = MockGrid(
            n,
            face_nodes,
            cell_faces,
            cell_centers,
            face_normals,
            cell_volumes,
            3,
            frac_face=[5, 6, 12, 13],
        )
        g.global_point_ind = np.arange(n.shape[1])
        return g

    def grid_2d_two_cells(self, pert=False):

        n = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 0], [1, 0, 1], [0, 0, 1]]
        ).T
        if pert:
            n[2, 2] = 2
            n[2, 4] = 2

        fn = np.array([[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]]).T
        cols = np.tile(np.arange(fn.shape[1]), (fn.shape[0], 1)).ravel("F")
        face_nodes = sps.csc_matrix((np.ones_like(cols), (fn.ravel("F"), cols)))

        cf = np.array([[0, 1, 2], [3, 4, 5]]).T
        cols = np.tile(np.arange(cf.shape[1]), (cf.shape[0], 1)).ravel("F")
        cell_faces = sps.csc_matrix((np.ones_like(cols), (cf.ravel("F"), cols)))

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

        g = MockGrid(
            n,
            face_nodes,
            cell_faces,
            cell_centers,
            face_normals,
            cell_volumes,
            2,
            frac_face=[2, 3],
        )
        g.global_point_ind = 1 + np.arange(n.shape[1])
        return g

    def grid_2d_two_cells_no_1d(self, pert=False):

        n = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]).T
        if pert:
            n[2, 2] = 2

        fn = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]]).T
        cols = np.tile(np.arange(fn.shape[1]), (fn.shape[0], 1)).ravel("F")
        face_nodes = sps.csc_matrix((np.ones_like(cols), (fn.ravel("F"), cols)))

        cf = np.array([[0, 1, 4], [4, 2, 3]]).T
        cols = np.tile(np.arange(cf.shape[1]), (cf.shape[0], 1)).ravel("F")
        cell_faces = sps.csc_matrix((np.ones_like(cols), (cf.ravel("F"), cols)))

        face_normals = np.array([[0, -1], [1, 0], [0, 1], [-1, 0], [-1, 1]]).T
        face_normals = np.vstack(
            (face_normals[0], np.zeros_like(face_normals[0]), face_normals[1])
        )
        cell_centers = np.array([[2.0 / 3, 0, 1.0 / 3], [1.0 / 3, 0, 2.0 / 3]]).T
        cell_volumes = 1 / 2 * np.ones(cell_centers.shape[1])

        if pert:
            cell_volumes[0] = 1
            cell_volumes[1] = 0.5

        g = MockGrid(
            n, face_nodes, cell_faces, cell_centers, face_normals, cell_volumes, 2
        )
        g.global_point_ind = 1 + np.arange(n.shape[1])
        return g

    def grid_2d_four_cells(self, pert=False, move_interior_point=False):
        # pert: Move the point (1, 0 1) -> (1, 0, 2)
        # move_interior_point: Move the point (0.5, 0, 0.5) -> (0.5, 0, 1). Should only
        #    be used together with pert.

        n = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 0, 1],
                [0.5, 0, 0.5],
                [0, 0, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0.5, 0, 0.5],
            ]
        ).T
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

        fn = np.array(
            [
                [0, 1],
                [1, 3],
                [3, 0],
                [1, 2],
                [2, 3],
                [4, 5],
                [5, 7],
                [7, 4],
                [7, 6],
                [6, 5],
            ]
        ).T
        cols = np.tile(np.arange(fn.shape[1]), (fn.shape[0], 1)).ravel("F")
        face_nodes = sps.csc_matrix((np.ones_like(cols), (fn.ravel("F"), cols)))

        cf = np.array([[0, 1, 2], [3, 4, 1], [5, 6, 7], [8, 9, 6]]).T
        cols = np.tile(np.arange(cf.shape[1]), (cf.shape[0], 1)).ravel("F")
        cell_faces = sps.csc_matrix((np.ones_like(cols), (cf.ravel("F"), cols)))

        face_normals = np.array(
            [
                [0, -1],
                [1, 1],
                [-1, 1],
                [1, 0],
                [-1, 1],
                [-1, 0],
                [1, 1],
                [1, -1],
                [1, -1],
                [0, 1],
            ]
        ).T

        face_normals = np.vstack(
            (face_normals[0], np.zeros_like(face_normals[0]), face_normals[1])
        )
        cell_centers = np.array(
            [[0.5, 0, 1.0 / 6], [5.0 / 6, 0, 0.5], [0.5, 0, 5.0 / 6], [1.0 / 6, 0, 0.5]]
        ).T
        cell_volumes = 1 / 4 * np.ones(cell_centers.shape[1])
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

        g = MockGrid(
            n,
            face_nodes,
            cell_faces,
            cell_centers,
            face_normals,
            cell_volumes,
            2,
            frac_face=[2, 4, 7, 8],
        )
        g.global_point_ind = 0 + np.arange(n.shape[1])
        return g

    def grid_2d_four_cells_no_1d(self, pert=False):

        n = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0.5, 0, 0.5]]).T
        if pert:
            n[2, 2] = 2

        fn = np.array(
            [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 4], [2, 4], [3, 4]]
        ).T
        cols = np.tile(np.arange(fn.shape[1]), (fn.shape[0], 1)).ravel("F")
        face_nodes = sps.csc_matrix((np.ones_like(cols), (fn.ravel("F"), cols)))

        cf = np.array([[0, 4, 5], [1, 5, 6], [2, 7, 6], [3, 4, 7]]).T
        cols = np.tile(np.arange(cf.shape[1]), (cf.shape[0], 1)).ravel("F")
        cell_faces = sps.csc_matrix((np.ones_like(cols), (cf.ravel("F"), cols)))

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
        if pert:
            cell_volumes[1] = 0.5
            cell_volumes[2] = 0.5

        g = MockGrid(
            n, face_nodes, cell_faces, cell_centers, face_normals, cell_volumes, 2
        )
        g.global_point_ind = 10 + np.arange(n.shape[1])
        return g

    def grid_1d(self, n_nodes=2):
        x = np.linspace(0, 1, n_nodes)
        g = pp.TensorGrid(x)
        g.nodes = np.tile(x, (3, 1))
        g.compute_geometry()
        g.global_point_ind = 1 + np.arange(n_nodes)
        return g

    def setup_mdg(self, pert=False, include_1d=True):
        # Mainly test of setup of
        if include_1d:
            sd_3 = self.grid_3d(pert)
            sd_2 = self.grid_2d_two_cells(pert)
            sd_1 = self.grid_1d()

            mdg, _ = pp.meshing._assemble_mdg(
                [[sd_3], [sd_2], [sd_1]], ensure_matching_face_cell=False
            )
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
            sd_3 = self.grid_3d_no_1d(pert)
            sd_2 = self.grid_2d_two_cells_no_1d(pert)
            mdg, _ = pp.meshing._assemble_mdg(
                [[sd_3], [sd_2]], ensure_matching_face_cell=False
            )
            a = np.zeros((16, 2))
            a[3, 0] = 1
            a[7, 1] = 1
            a[11, 0] = 1
            a[15, 1] = 1
            map = sps.csc_matrix(a.T)

            pp.meshing.create_interfaces(mdg, {(sd_3, sd_2): map})
        return mdg

    def _mortar_grids(self, mdg):
        mg1 = None
        mg2 = None
        
        for intf in mdg.interfaces():
            _, gl = mdg.interface_to_subdomain_pair(intf)
            if gl.dim == 1:
                mg1 = intf
            else:
                mg2 = intf
        return mg1, mg2

    def test_replace_1d_with_identity(self):
        # Replace the 1d grid with a copy of itself. The mappings should stay the same.
        mdg = self.setup_mdg(pert=False)
        mg1, mg2 = self._mortar_grids(mdg)

        proj_1_h = mg1.primary_to_mortar_int().copy()
        proj_1_l = mg1.secondary_to_mortar_int().copy()

        gn = self.grid_1d(2)
        go = list(mdg.subdomains(dim=1))[0]
        mdg.replace_subdomains_and_interfaces({go: gn})

        mg1, mg2 = self._mortar_grids(mdg)
        p1h = mg1.primary_to_mortar_int().copy()
        p1l = mg1.secondary_to_mortar_int().copy()

        # No changes to the mappings
        self.assertTrue((proj_1_h != p1h).nnz == 0)
        self.assertTrue((proj_1_l != p1l).nnz == 0)

    def test_replace_2d_with_identity_no_1d(self):
        # Replace the 2d grid with a copy of itself. The mappings should stay the same.

        mdg = self.setup_mdg(pert=False, include_1d=False)
        mg1, mg2 = self._mortar_grids(mdg)

        proj_2_h = mg2.primary_to_mortar_int().copy()
        proj_2_l = mg2.secondary_to_mortar_int().copy()

        gn = self.grid_2d_two_cells()
        go = list(mdg.subdomains(dim=2))[0]
        mdg.replace_subdomains_and_interfaces(sd_map={go: gn})

        mg1, mg2 = self._mortar_grids(mdg)
        p2h = mg2.primary_to_mortar_int().copy()
        p2l = mg2.secondary_to_mortar_int().copy()

        # No changes to the mappings
        self.assertTrue((proj_2_h != p2h).nnz == 0)
        self.assertTrue((proj_2_l != p2l).nnz == 0)

    def test_replace_2d_with_finer_no_1d(self):
        # Replace the fracture grid on the unperturbed geometry

        mdg = self.setup_mdg(pert=False, include_1d=False)
        mg1, mg2 = self._mortar_grids(mdg)
        proj_2_h = mg2.primary_to_mortar_int().copy()

        gn = self.grid_2d_four_cells_no_1d()
        go = list(mdg.subdomains(dim=2))[0]
        mdg.replace_subdomains_and_interfaces({go: gn})

        mg1, mg2 = self._mortar_grids(mdg)
        p2h = mg2.primary_to_mortar_int().copy()
        p2l = mg2.secondary_to_mortar_int().copy()

        # The known projection matrix, from secondary to one of the mortar grids.
        # Needs to be duplicated along the first axis before comparison.
        known_p2l = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])

        # There should be no changes in the mapping between mortar and primary
        self.assertTrue((proj_2_h != p2h).nnz == 0)
        self.assertTrue(np.allclose(np.tile(known_p2l, (2, 1)), p2l.A))

    def test_replace_2d_with_finer_no_1d_pert(self):
        # Replace the fracture grid on the perturbed geometry.

        # Create the md_grid, single 2d grid, no splitting.
        mdg = self.setup_mdg(pert=True, include_1d=False)
        _, mg2 = self._mortar_grids(mdg)
        proj_2_h = mg2.primary_to_mortar_int().copy()

        gn = self.grid_2d_four_cells_no_1d(pert=True)
        go = list(mdg.subdomains(dim=2))[0]
        mdg.replace_subdomains_and_interfaces({go: gn})

        _, mg2 = self._mortar_grids(mdg)
        p2h = mg2.primary_to_mortar_int().copy()
        p2l = mg2.secondary_to_mortar_int().copy()

        # The known projection matrix, from secondary to one of the mortar grids.
        # Needs to be duplicated along the first axis before comparison.
        known_p2l = np.array([[1, 1, 1 / 3, 1 / 3], [0, 0, 2 / 3, 2 / 3]])
        # There should be no changes in the mapping between mortar and primary
        self.assertTrue((proj_2_h != p2h).nnz == 0)
        self.assertTrue(np.allclose(np.tile(known_p2l, (2, 1)), p2l.A))

    def test_replace_2d_with_identity(self):
        # Replace the fracture grid with a copy of itself,
        # with a 1d grid included.

        mdg = self.setup_mdg(pert=False, include_1d=True)
        mg1, mg2 = self._mortar_grids(mdg)

        proj_1_h = mg1.primary_to_mortar_int().copy()
        proj_1_l = mg1.secondary_to_mortar_int().copy()
        proj_2_h = mg2.primary_to_mortar_int().copy()
        proj_2_l = mg2.secondary_to_mortar_int().copy()

        gn = self.grid_2d_two_cells()
        go = list(mdg.subdomains(dim=2))[0]
        mdg.replace_subdomains_and_interfaces({go: gn})

        mg1, mg2 = self._mortar_grids(mdg)
        p1h = mg1.primary_to_mortar_int().copy()
        p1l = mg1.secondary_to_mortar_int().copy()
        p2h = mg2.primary_to_mortar_int().copy()
        p2l = mg2.secondary_to_mortar_int().copy()

        # The mappings should stay the same
        self.assertTrue((proj_1_h != p1h).nnz == 0)
        self.assertTrue((proj_1_l != p1l).nnz == 0)
        self.assertTrue((proj_2_h != p2h).nnz == 0)
        self.assertTrue((proj_2_l != p2l).nnz == 0)

    def test_replace_2d_with_finer_pert(self):
        # Replace the fracture grid with a refined grid. 1d grid is included
        mdg = self.setup_mdg(pert=True, include_1d=True)
        mg1, mg2 = self._mortar_grids(mdg)
        proj_2_h = mg2.primary_to_mortar_int().copy()
        proj_1_l = mg1.secondary_to_mortar_int().copy()

        gn = self.grid_2d_four_cells(pert=True, move_interior_point=True)
        go = list(mdg.subdomains(dim=2))[0]
        mdg.replace_subdomains_and_interfaces({go: gn})

        mg1, mg2 = self._mortar_grids(mdg)
        p1h = mg1.primary_to_mortar_avg().copy()
        p1l = mg1.secondary_to_mortar_int().copy()
        p2h = mg2.primary_to_mortar_int().copy()
        p2l = mg2.secondary_to_mortar_int().copy()

        # Projection from 2d mortar to the 3d domain stays the same
        self.assertTrue((proj_2_h != p2h).nnz == 0)
        # Projection from 1d mortar to the 1d domain stays the same.
        self.assertTrue((proj_1_l != p1l).nnz == 0)

        # Check mapping from 2d fracture to 2d mortar
        # The known projection matrix, from 2d fracture grid to one of the 2d mortar grids.
        # Needs to be duplicated along the first axis before comparison.
        known_p2l = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
        # Compare new mapping
        self.assertTrue(np.allclose(np.tile(known_p2l, (2, 1)), p2l.A))

        # Check mapping from 1d mortar to 2d fracture
        known_p1h = np.zeros((2, 10))
        known_p1h[0, [2, 4]] = 0.5
        known_p1h[1, [7, 8]] = 0.5
        self.assertTrue(np.allclose(known_p1h, p1h.A))


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

    test_utils.compare_mortar_grids(mg, mg_read)

    mg_one_sided = pp.MortarGrid(g.dim, {0: g})

    pickle.dump(mg, open(fn, "wb"))
    mg_read = pickle.load(open(fn, "rb"))

    test_utils.compare_mortar_grids(mg_one_sided, mg_read)

    test_utils.delete_file(fn)


if __name__ == "__main__":
    unittest.main()
