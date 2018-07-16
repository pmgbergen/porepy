#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 18:23:11 2017

@author: Eirik Keilegavlen
"""

import numpy as np
import unittest
import scipy.sparse as sps

from porepy.grids.structured import TensorGrid
from porepy.grids.simplex import StructuredTriangleGrid
from porepy.fracs import mortars, meshing


class TestGridMappings1d(unittest.TestCase):
    def test_merge_grids_all_common(self):
        g = TensorGrid(np.arange(3))
        weights, new, old = mortars.match_grids_1d(g, g, tol=1e-4)

        assert np.allclose(weights, np.ones(2))
        assert np.allclose(old, np.arange(2))
        assert np.allclose(new, np.arange(2))

    def test_merge_grids_non_matching(self):
        g = TensorGrid(np.arange(3))
        h = TensorGrid(np.arange(3))
        h.nodes[0, 1] = 0.5
        weights, new, old = mortars.match_grids_1d(g, h, tol=1e-4)

        # Weights give mappings from h to g. The first cell in h is
        # fully within the first cell in g. The second in h is split 1/3
        # in first of g, 2/3 in second.
        assert np.allclose(weights, np.array([1, 1. / 3, 2. / 3]))
        assert np.allclose(new, np.array([0, 0, 1]))
        assert np.allclose(old, np.array([0, 1, 1]))

    def test_merge_grids_reverse_order(self):
        g = TensorGrid(np.arange(3))
        h = TensorGrid(np.arange(3))
        h.nodes = h.nodes[:, ::-1]
        weights, new, old = mortars.match_grids_1d(g, h, tol=1e-4)

        assert np.allclose(weights, np.array([1, 1]))
        # In this case, we don't know which ordering the combined grid uses
        # Instead, make sure that the two mappings are ordered in separate
        # directions
        assert np.allclose(new[::-1], old)


class TestReplaceHigherDimensionalGrid(unittest.TestCase):
    # Test functionality for replacing the higher dimensional grid in a bucket.
    # The critical point is to check updates of the projection from high
    # dimensional grid to mortar grid.

    def test_replace_by_same(self):
        # 1x2 grid.
        # Copy the higher dimensional grid and replace. The mapping should be
        # the same.

        f1 = np.array([[0, 1], [.5, .5]])
        N = [1, 2]
        gb = meshing.cart_grid([f1], N, **{"physdims": [1, 1]})
        gb.compute_geometry()

        # Pick out mortar grid by a loop, there is only one edge in the bucket
        for e, d in gb.edges():
            mg = d["mortar_grid"]

        old_projection = mg.high_to_mortar_int.copy()

        g_old = gb.grids_of_dimension(2)[0]
        g_new = g_old.copy()

        mortars.replace_grids_in_bucket(gb, {g_old: g_new})

        # Get mortar grid again
        for e, d in gb.edges():
            mg = d["mortar_grid"]

        new_projection = mg.high_to_mortar_int

        # The projections should be identical
        assert (old_projection != new_projection).nnz == 0

    def test_refine_high_dim(self):
        # Replace the 2d grid with a finer one

        f1 = np.array([[0, 1], [.5, .5]])
        N = [1, 2]
        gb = meshing.cart_grid([f1], N, **{"physdims": [1, 1]})
        gb.compute_geometry()

        # Pick out mortar grid by a loop, there is only one edge in the bucket
        for e, d in gb.edges():
            mg = d["mortar_grid"]

        old_projection = mg.high_to_mortar_int.copy()
        g_old = gb.grids_of_dimension(2)[0]

        # Create a new, finer 2d grid. This is the simplest
        # way to put the fracture in the right place is to create a new
        # bucket, and pick out the higher dimensional grid
        gb_new = meshing.cart_grid([f1], [2, 2], **{"physdims": [1, 1]})
        gb_new.compute_geometry()

        g_new = gb_new.grids_of_dimension(2)[0]

        mortars.replace_grids_in_bucket(gb, {g_old: g_new})

        # Get mortar grid again
        for e, d in gb.edges():
            mg = d["mortar_grid"]

        new_projection = mg.high_to_mortar_int

        # Check shape
        assert new_projection.shape[0] == old_projection.shape[0]
        assert new_projection.shape[1] == g_new.num_faces
        # Projection sums to unity.
        assert np.all(new_projection.toarray().sum(axis=1) == 1)

        fi = np.where(g_new.face_centers[1] == 0.5)[0]
        assert fi.size == 4
        # Hard coded test (based on knowledge of how the grids and meshing
        # is implemented). Faces to the uppermost cell are always kept in
        # place, the lowermost are duplicated towards the end of the face
        # definition.
        assert np.all(new_projection[0, fi[:2]].toarray() == 0.5)
        assert np.all(new_projection[1, fi[2:]].toarray() == 0.5)

    def test_coarsen_high_dim(self):
        # Replace the 2d grid with a coarser one

        f1 = np.array([[0, 1], [.5, .5]])
        N = [2, 2]
        gb = meshing.cart_grid([f1], N, **{"physdims": [1, 1]})
        gb.compute_geometry()

        # Pick out mortar grid by a loop, there is only one edge in the bucket
        for e, d in gb.edges():
            mg = d["mortar_grid"]

        old_projection = mg.high_to_mortar_int.copy()
        g_old = gb.grids_of_dimension(2)[0]

        # Create a new, coarser 2d grid. This is the simplest
        # way to put the fracture in the right place is to create a new
        # bucket, and pick out the higher dimensional grid
        gb_new = meshing.cart_grid([f1], [1, 2], **{"physdims": [1, 1]})
        gb_new.compute_geometry()

        g_new = gb_new.grids_of_dimension(2)[0]

        mortars.replace_grids_in_bucket(gb, {g_old: g_new})

        # Get mortar grid again
        for e, d in gb.edges():
            mg = d["mortar_grid"]

        new_projection = mg.high_to_mortar_int

        # Check shape
        assert new_projection.shape[0] == old_projection.shape[0]
        assert new_projection.shape[1] == g_new.num_faces
        # Projection sums to unity.
        assert np.all(new_projection.toarray().sum(axis=1) == 1)

        fi = np.where(g_new.face_centers[1] == 0.5)[0]
        assert fi.size == 2
        # Hard coded test (based on knowledge of how the grids and meshing
        # is implemented). Faces to the uppermost cell are always kept in
        # place, the lowermost are duplicated towards the end of the face
        # definition.
        assert np.all(new_projection[0, fi[0]] == 1)
        assert np.all(new_projection[1, fi[0]] == 1)
        assert np.all(new_projection[2, fi[1]] == 1)
        assert np.all(new_projection[3, fi[1]] == 1)

    def test_refine_distort_high_dim(self):
        # Replace the 2d grid with a finer one, and move the nodes along the
        # interface so that areas along the interface are no longer equal.

        f1 = np.array([[0, 1], [.5, .5]])
        N = [1, 2]
        gb = meshing.cart_grid([f1], N, **{"physdims": [1, 1]})
        gb.compute_geometry()

        # Pick out mortar grid by a loop, there is only one edge in the bucket
        for e, d in gb.edges():
            mg = d["mortar_grid"]

        old_projection = mg.high_to_mortar_int.copy()
        g_old = gb.grids_of_dimension(2)[0]

        # Create a new, finer 2d grid. This is the simplest
        # way to put the fracture in the right place is to create a new
        # bucket, and pick out the higher dimensional grid
        gb_new = meshing.cart_grid([f1], [2, 2], **{"physdims": [1, 1]})
        gb_new.compute_geometry()

        g_new = gb_new.grids_of_dimension(2)[0]

        # By construction of the split grid, we know that the nodes at
        # (0.5, 0.5) are no 5 and 6, and that no 5 is associated with the
        # face belonging to the lower cells.
        # Move node belonging to the lower face
        g_new.nodes[0, 5] = 0.2
        g_new.nodes[0, 6] = 0.7
        g_new.compute_geometry()

        mortars.replace_grids_in_bucket(gb, {g_old: g_new})

        # Get mortar grid again
        for e, d in gb.edges():
            mg = d["mortar_grid"]

        new_projection = mg.high_to_mortar_int

        # Check shape
        assert new_projection.shape[0] == old_projection.shape[0]
        assert new_projection.shape[1] == g_new.num_faces
        # Projection sums to unity.
        assert np.all(new_projection.toarray().sum(axis=1) == 1)

        fi = np.where(g_new.face_centers[1] == 0.5)[0]
        assert fi.size == 4
        # Hard coded test (based on knowledge of how the grids and meshing
        # is implemented). Faces to the uppermost cell are always kept in
        # place, the lowermost are duplicated towards the end of the face
        # definition.
        assert np.abs(new_projection[0, 8] - 0.7 < 1e-6)
        assert np.abs(new_projection[0, 9] - 0.3 < 1e-6)
        assert np.abs(new_projection[1, 12] - 0.2 < 1e-6)
        assert np.abs(new_projection[1, 13] - 0.8 < 1e-6)

    def test_distort_high_dim(self):
        # Replace the 2d grid with a finer one, and move the nodes along the
        # interface so that areas along the interface are no longer equal.

        f1 = np.array([[0, 1], [.5, .5]])
        N = [2, 2]
        gb = meshing.cart_grid([f1], N, **{"physdims": [1, 1]})
        gb.compute_geometry()

        # Pick out mortar grid by a loop, there is only one edge in the bucket
        for e, d in gb.edges():
            mg = d["mortar_grid"]

        old_projection = mg.high_to_mortar_int.copy()
        g_old = gb.grids_of_dimension(2)[0]

        # Create a new, finer 2d grid. This is the simplest
        # way to put the fracture in the right place is to create a new
        # bucket, and pick out the higher dimensional grid
        gb_new = meshing.cart_grid([f1], [2, 2], **{"physdims": [1, 1]})
        gb_new.compute_geometry()

        g_new = gb_new.grids_of_dimension(2)[0]

        # By construction of the split grid, we know that the nodes at
        # (0.5, 0.5) are no 5 and 6, and that no 5 is associated with the
        # face belonging to the lower cells.
        # Move node belonging to the lower face
        g_new.nodes[0, 5] = 0.2
        g_new.nodes[0, 6] = 0.7
        g_new.compute_geometry()

        mortars.replace_grids_in_bucket(gb, {g_old: g_new})

        # Get mortar grid again
        for e, d in gb.edges():
            mg = d["mortar_grid"]

        new_projection = mg.high_to_mortar_int

        # Check shape
        assert new_projection.shape[0] == old_projection.shape[0]
        assert new_projection.shape[1] == g_new.num_faces
        # Projection sums to unity.
        assert np.all(new_projection.toarray().sum(axis=1) == 1)

        fi = np.where(g_new.face_centers[1] == 0.5)[0]
        assert fi.size == 4
        # Hard coded test (based on knowledge of how the grids and meshing
        # is implemented). Faces to the uppermost cell are always kept in
        # place, the lowermost are duplicated towards the end of the face
        # definition.

        # It seems the mortar grid is designed so that the first cell is
        # associated with face 9 in the old grid. This is split into 2/5 face
        # 8 and 3/5 face 9.
        assert np.abs(new_projection[0, 8] - 0.4 < 1e-6)
        assert np.abs(new_projection[0, 9] - 0.6 < 1e-6)
        # The second cell in mortar grid is still fully connected to face 9
        assert np.abs(new_projection[1, 9] - 1 < 1e-6)
        assert np.abs(new_projection[2, 13] - 1 < 1e-6)
        assert np.abs(new_projection[3, 12] - 0.4 < 1e-6)
        assert np.abs(new_projection[3, 13] - 0.6 < 1e-6)

    def test_permute_nodes_in_replacement_grid(self):
        # Replace higher dimensional grid with an identical one, except the
        # node indices are perturbed. This will test sorting of nodes along
        # 1d lines
        f1 = np.array([[0, 1], [.5, .5]])
        N = [2, 2]
        gb = meshing.cart_grid([f1], N, **{"physdims": [1, 1]})
        gb.compute_geometry()

        # Pick out mortar grid by a loop, there is only one edge in the bucket
        for e, d in gb.edges():
            mg = d["mortar_grid"]

        old_projection = mg.high_to_mortar_int.copy()
        g_old = gb.grids_of_dimension(2)[0]

        # Create a new, finer 2d grid. This is the simplest
        # way to put the fracture in the right place is to create a new
        # bucket, and pick out the higher dimensional grid
        gb_new = meshing.cart_grid([f1], [2, 2], **{"physdims": [1, 1]})
        gb_new.compute_geometry()

        g_new = gb_new.grids_of_dimension(2)[0]

        # By construction of the split grid, we know that the nodes at
        # (0.5, 0.5) are no 5 and 6, and that no 5 is associated with the
        # face belonging to the lower cells.
        # Move node belonging to the lower face
        #     g_new.nodes[0, 5] = 0.2
        #     g_new.nodes[0, 6] = 0.7

        # Replacements: along lower segment (3, 5, 7) -> (7, 5, 3)
        # On upper segment: (4, 6, 8) -> (8, 4, 6)
        g_new.nodes[0, 3] = 1
        g_new.nodes[0, 4] = 0.5
        g_new.nodes[0, 5] = 0.5
        g_new.nodes[0, 6] = 1
        g_new.nodes[0, 7] = 0
        g_new.nodes[0, 8] = 0

        fn = g_new.face_nodes.indices.reshape((2, g_new.num_faces), order="F")
        fn[:, 8] = np.array([4, 8])
        fn[:, 9] = np.array([4, 6])
        fn[:, 12] = np.array([7, 5])
        fn[:, 13] = np.array([5, 3])

        mortars.replace_grids_in_bucket(gb, {g_old: g_new})

        # Get mortar grid again
        for e, d in gb.edges():
            mg = d["mortar_grid"]

        new_projection = mg.high_to_mortar_int

        # Check shape
        assert new_projection.shape[0] == old_projection.shape[0]
        assert new_projection.shape[1] == g_new.num_faces
        # Projection sums to unity.
        assert np.all(new_projection.toarray().sum(axis=1) == 1)

        fi = np.where(g_new.face_centers[1] == 0.5)[0]
        assert fi.size == 4
        # Hard coded test (based on knowledge of how the grids and meshing
        # is implemented). Faces to the uppermost cell are always kept in
        # place, the lowermost are duplicated towards the end of the face
        # definition.
        assert (old_projection != new_projection).nnz == 0

    #        assert np.abs(new_projection[0, 8] - 0.7 < 1e-6)
    #        assert np.abs(new_projection[0, 9] - 0.3 < 1e-6)
    #        assert np.abs(new_projection[1, 12] - 0.2 < 1e-6)
    #        assert np.abs(new_projection[1, 13] - 0.8 < 1e-6)

    def test_permute_perturb_nodes_in_replacement_grid(self):
        # Replace higher dimensional grid with an identical one, except the
        # node indices are perturbed. This will test sorting of nodes along
        # 1d lines. Also perturb nodes along the segment.
        f1 = np.array([[0, 1], [.5, .5]])
        N = [2, 2]
        gb = meshing.cart_grid([f1], N, **{"physdims": [1, 1]})
        gb.compute_geometry()

        # Pick out mortar grid by a loop, there is only one edge in the bucket
        for e, d in gb.edges():
            mg = d["mortar_grid"]

        old_projection = mg.high_to_mortar_int.copy()
        g_old = gb.grids_of_dimension(2)[0]

        # Create a new, finer 2d grid. This is the simplest
        # way to put the fracture in the right place is to create a new
        # bucket, and pick out the higher dimensional grid
        gb_new = meshing.cart_grid([f1], [2, 2], **{"physdims": [1, 1]})
        gb_new.compute_geometry()

        g_new = gb_new.grids_of_dimension(2)[0]

        # By construction of the split grid, we know that the nodes at
        # (0.5, 0.5) are no 5 and 6, and that no 5 is associated with the
        # face belonging to the lower cells.
        # Replacements: along lower segment (3, 5, 7) -> (7, 5, 3)
        # On upper segment: (4, 6, 8) -> (8, 4, 6)
        g_new.nodes[0, 3] = 1
        g_new.nodes[0, 4] = 0.7
        g_new.nodes[0, 5] = 0.2
        g_new.nodes[0, 6] = 1
        g_new.nodes[0, 7] = 0
        g_new.nodes[0, 8] = 0

        fn = g_new.face_nodes.indices.reshape((2, g_new.num_faces), order="F")
        fn[:, 8] = np.array([4, 8])
        fn[:, 9] = np.array([4, 6])
        fn[:, 12] = np.array([7, 5])
        fn[:, 13] = np.array([5, 3])

        mortars.replace_grids_in_bucket(gb, {g_old: g_new})

        # Get mortar grid again
        for e, d in gb.edges():
            mg = d["mortar_grid"]

        new_projection = mg.high_to_mortar_int

        # Check shape
        assert new_projection.shape[0] == old_projection.shape[0]
        assert new_projection.shape[1] == g_new.num_faces
        # Projection sums to unity.
        assert np.all(new_projection.toarray().sum(axis=1) == 1)

        fi = np.where(g_new.face_centers[1] == 0.5)[0]
        assert fi.size == 4
        # Hard coded test (based on knowledge of how the grids and meshing
        # is implemented). Faces to the uppermost cell are always kept in
        # place, the lowermost are duplicated towards the end of the face
        # definition.
        # It seems the mortar grid is designed so that the first cell is
        # associated with face 9 in the old grid. This is split into 2/5 face
        # 8 and 3/5 face 9.
        assert np.abs(new_projection[0, 8] - 0.4 < 1e-6)
        assert np.abs(new_projection[0, 9] - 0.6 < 1e-6)
        # The second cell in mortar grid is still fully connected to face 9
        assert np.abs(new_projection[1, 9] - 1 < 1e-6)
        assert np.abs(new_projection[2, 13] - 1 < 1e-6)
        assert np.abs(new_projection[3, 12] - 0.4 < 1e-6)
        assert np.abs(new_projection[3, 13] - 0.6 < 1e-6)


class MockGrid(object):
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
            self.name = ["TensorGrid"]
        elif self.dim == 2:
            self.name = ["TriangleGrid"]
        elif self.dim == 3:
            self.name = ["TetrahedralGrid"]

        ff = np.zeros(self.num_faces, dtype=np.bool)
        if frac_face is not None:
            ff[frac_face] = 1
        self.tags = {"fracture_faces": ff}

        if glob_pi is not None:
            self.global_point_ind = glob_pi

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


class TestMeshReplacement3d(unittest.TestCase):
    def grid_3d(self, pert=False):
        # Grid consisting of 3d, 2d and 1d grid. Nodes along the main
        # surface are split into two or four.
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
            # This will invalidate the assigned geometry, but it should not matter
            n[2, [3, 6, 9, 12]] = 2

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
            n[2, 5] = 2

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
        cell_centers = np.array([[2. / 3, 0, 1. / 3], [1. / 3, 0, 2. / 3]]).T
        cell_volumes = 1 / 2 * np.ones(cell_centers.shape[1])
        if pert:
            cell_volumes[1] = 1
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
            n[2, 3] = 2

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
        cell_centers = np.array([[2. / 3, 0, 1. / 3], [1. / 3, 0, 2. / 3]]).T
        cell_volumes = 1 / 2 * np.ones(cell_centers.shape[1])
        if pert:
            cell_volumes[1] = 1
        g = MockGrid(
            n, face_nodes, cell_faces, cell_centers, face_normals, cell_volumes, 2
        )
        g.global_point_ind = 1 + np.arange(n.shape[1])
        return g

    def grid_2d_four_cells(self, pert=False):

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
            n[2, 5] = 2

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
                [0, 1],
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
            [[0.5, 0, 1. / 6], [5. / 6, 0, 0.5], [0.5, 0, 5. / 6], [1. / 6, 0, 0.5]]
        ).T
        cell_volumes = 1 / 4 * np.ones(cell_centers.shape[1])
        if pert:
            cell_volumes[2:] = 0.5
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
            n[2, 3] = 2

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
            [[0.5, 0, 1. / 6], [5. / 6, 0, 0.5], [0.5, 0, 5. / 6], [1. / 6, 0, 0.5]]
        ).T
        cell_volumes = 1 / 4 * np.ones(cell_centers.shape[1])
        if pert:
            cell_volumes[2:] = 0.5

        g = MockGrid(
            n, face_nodes, cell_faces, cell_centers, face_normals, cell_volumes, 2
        )
        g.global_point_ind = 10 + np.arange(n.shape[1])
        return g

    def grid_1d(self, n_nodes=2):
        x = np.linspace(0, 1, n_nodes)
        g = TensorGrid(x)
        g.nodes = np.tile(x, (3, 1))
        g.compute_geometry()
        g.global_point_ind = 1 + np.arange(n_nodes)
        return g

    def setup_bucket(self, pert=False, include_1d=True):
        # Mainly test of setup of
        if include_1d:
            g3 = self.grid_3d(pert)
            g2 = self.grid_2d_two_cells(pert)
            g1 = self.grid_1d()

            gb = meshing._assemble_in_bucket(
                [[g3], [g2], [g1]], ensure_matching_face_cell=False
            )

            gb.add_edge_props("face_cells")
            for e, d in gb.edges():
                gl = gb.nodes_of_edge(e)[0]
                if gl.dim == 1:
                    m = sps.csc_matrix(np.array([[0, 0, 1, 1, 0, 0]]))
                    d["face_cells"] = m
                else:
                    a = np.zeros((16, 2))
                    a[3, 0] = 1
                    a[7, 1] = 1
                    a[11, 0] = 1
                    a[15, 1] = 1
                    d["face_cells"] = sps.csc_matrix(a.T)

        else:
            g3 = self.grid_3d_no_1d(pert)
            g2 = self.grid_2d_two_cells_no_1d(pert)
            gb = meshing._assemble_in_bucket(
                [[g3], [g2]], ensure_matching_face_cell=False
            )
            for e, d in gb.edges():
                a = np.zeros((16, 2))
                a[3, 0] = 1
                a[7, 1] = 1
                a[11, 0] = 1
                a[15, 1] = 1
                d["face_cells"] = sps.csc_matrix(a.T)

        meshing.create_mortar_grids(gb)
        return gb

    def _mortar_grids(self, gb):
        mg1 = None
        mg2 = None
        for e, d in gb.edges():
            gh = gb.nodes_of_edge(e)[0]
            if gh.dim == 1:
                mg1 = d["mortar_grid"]
            else:
                mg2 = d["mortar_grid"]
        return mg1, mg2

    def test_replace_1d_with_identity(self):
        gb = self.setup_bucket(pert=False)
        mg1, mg2 = self._mortar_grids(gb)

        proj_1_h = mg1.high_to_mortar_int.copy()
        proj_1_l = mg1.low_to_mortar_int.copy()

        gn = self.grid_1d(2)
        go = gb.grids_of_dimension(1)[0]
        mortars.replace_grids_in_bucket(gb, {go: gn})

        mg1, mg2 = self._mortar_grids(gb)
        p1h = mg1.high_to_mortar_int.copy()
        p1l = mg1.low_to_mortar_int.copy()

        assert (proj_1_h != p1h).nnz == 0
        assert (proj_1_l != p1l).nnz == 0

    def test_replace_2d_with_identity_no_1d(self):
        gb = self.setup_bucket(pert=False, include_1d=False)
        mg1, mg2 = self._mortar_grids(gb)

        proj_2_h = mg2.high_to_mortar_int.copy()
        proj_2_l = mg2.low_to_mortar_int.copy()

        gn = self.grid_2d_two_cells()
        go = gb.grids_of_dimension(2)[0]
        mortars.replace_grids_in_bucket(gb, {go: gn})

        mg1, mg2 = self._mortar_grids(gb)
        p2h = mg2.high_to_mortar_int.copy()
        p2l = mg2.low_to_mortar_int.copy()

        assert (proj_2_h != p2h).nnz == 0
        assert (proj_2_l != p2l).nnz == 0

    def test_replace_2d_with_finer_no_1d(self):
        gb = self.setup_bucket(pert=False, include_1d=False)
        mg1, mg2 = self._mortar_grids(gb)
        proj_2_h = mg2.high_to_mortar_int.copy()

        gn = self.grid_2d_four_cells_no_1d()
        go = gb.grids_of_dimension(2)[0]
        mortars.replace_grids_in_bucket(gb, {go: gn})

        mg1, mg2 = self._mortar_grids(gb)
        p2h = mg2.high_to_mortar_int.copy()
        p2l = mg2.low_to_mortar_int.copy()

        assert (proj_2_h != p2h).nnz == 0
        assert np.abs(p2l[0, 0] - 0.5) < 1e-6
        assert np.abs(p2l[0, 1] - 0.5) < 1e-6
        assert np.abs(p2l[1, 2] - 0.5) < 1e-6
        assert np.abs(p2l[1, 3] - 0.5) < 1e-6

    def test_replace_2d_with_finer_no_1d_pert(self):
        gb = self.setup_bucket(pert=True, include_1d=False)
        mg1, mg2 = self._mortar_grids(gb)
        proj_2_h = mg2.high_to_mortar_int.copy()

        gn = self.grid_2d_four_cells_no_1d(pert=True)
        go = gb.grids_of_dimension(2)[0]
        mortars.replace_grids_in_bucket(gb, {go: gn})

        mg1, mg2 = self._mortar_grids(gb)
        p2h = mg2.high_to_mortar_int.copy()
        p2l = mg2.low_to_mortar_int.copy()

        assert (proj_2_h != p2h).nnz == 0
        assert np.abs(p2l[0, 0] - 0.5) < 1e-6
        assert np.abs(p2l[0, 1] - 0.5) < 1e-6
        assert np.abs(p2l[1, 2] - 0.5) < 1e-6
        assert np.abs(p2l[1, 3] - 0.5) < 1e-6
        assert np.abs(p2l[2, 0] - 0.5) < 1e-6
        assert np.abs(p2l[2, 1] - 0.5) < 1e-6
        assert np.abs(p2l[3, 2] - 0.5) < 1e-6
        assert np.abs(p2l[3, 3] - 0.5) < 1e-6

    def test_replace_2d_with_identity(self):
        gb = self.setup_bucket(pert=False, include_1d=True)
        mg1, mg2 = self._mortar_grids(gb)

        proj_1_h = mg1.high_to_mortar_int.copy()
        proj_1_l = mg1.low_to_mortar_int.copy()
        proj_2_h = mg2.high_to_mortar_int.copy()
        proj_2_l = mg2.low_to_mortar_int.copy()

        gn = self.grid_2d_two_cells()
        go = gb.grids_of_dimension(2)[0]
        mortars.replace_grids_in_bucket(gb, {go: gn})

        mg1, mg2 = self._mortar_grids(gb)
        p1h = mg1.high_to_mortar_int.copy()
        p1l = mg1.low_to_mortar_int.copy()
        p2h = mg2.high_to_mortar_int.copy()
        p2l = mg2.low_to_mortar_int.copy()

        assert (proj_1_h != p1h).nnz == 0
        assert (proj_1_l != p1l).nnz == 0
        assert (proj_2_h != p2h).nnz == 0
        assert (proj_2_l != p2l).nnz == 0

    def test_replace_2d_with_finer_pert(self):
        gb = self.setup_bucket(pert=True, include_1d=True)
        mg1, mg2 = self._mortar_grids(gb)
        proj_1_h = mg1.high_to_mortar_int.copy()
        proj_2_h = mg2.high_to_mortar_int.copy()
        proj_1_l = mg1.low_to_mortar_int.copy()
        proj_2_l = mg2.low_to_mortar_int.copy()

        gn = self.grid_2d_four_cells(pert=True)
        go = gb.grids_of_dimension(2)[0]
        mortars.replace_grids_in_bucket(gb, {go: gn})

        mg1, mg2 = self._mortar_grids(gb)
        p1h = mg1.high_to_mortar_int.copy()
        p1l = mg1.low_to_mortar_int.copy()
        p2h = mg2.high_to_mortar_int.copy()
        p2l = mg2.low_to_mortar_int.copy()

        assert (proj_1_l != p1l).nnz == 0
        assert (proj_2_h != p2h).nnz == 0

        assert np.abs(p2l[0, 0] - 0.5) < 1e-6
        assert np.abs(p2l[0, 1] - 0.5) < 1e-6
        assert np.abs(p2l[1, 2] - 0.5) < 1e-6
        assert np.abs(p2l[1, 3] - 0.5) < 1e-6
        assert np.abs(p2l[2, 0] - 0.5) < 1e-6
        assert np.abs(p2l[2, 1] - 0.5) < 1e-6
        assert np.abs(p2l[3, 2] - 0.5) < 1e-6
        assert np.abs(p2l[3, 3] - 0.5) < 1e-6

        assert np.abs(p1h[0, 2] - 0.5) < 1e-6
        assert np.abs(p1h[0, 4] - 0.5) < 1e-6
        assert np.abs(p1h[1, 7] - 0.5) < 1e-6
        assert np.abs(p1h[1, 8] - 0.5) < 1e-6


if __name__ == "__main__":
    unittest.main()
