#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 18:23:11 2017

@author: Eirik Keilegavlen
"""

import numpy as np
import unittest

from porepy.grids.structured import TensorGrid
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
          assert np.allclose(weights, np.array([1, 1./3, 2./3]))
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
        gb = meshing.cart_grid([f1], N, **{'physdims': [1, 1]})
        gb.compute_geometry()

        # Pick out mortar grid by a loop, there is only one edge in the bucket
        for e, d in gb.edges_props():
            mg = d['mortar_grid']

        old_projection = mg.high_to_mortar_int.copy()

        g_old = gb.grids_of_dimension(2)[0]
        g_new = g_old.copy()

        mortars.replace_grids_in_bucket(gb, {g_old: g_new})

        # Get mortar grid again
        for e, d in gb.edges_props():
            mg = d['mortar_grid']

        new_projection = mg.high_to_mortar_int

        # The projections should be identical
        assert (old_projection != new_projection).nnz == 0

    def test_refine_high_dim(self):
        # Replace the 2d grid with a finer one

        f1 = np.array([[0, 1], [.5, .5]])
        N = [1, 2]
        gb = meshing.cart_grid([f1], N, **{'physdims': [1, 1]})
        gb.compute_geometry()

        # Pick out mortar grid by a loop, there is only one edge in the bucket
        for e, d in gb.edges_props():
            mg = d['mortar_grid']

        old_projection = mg.high_to_mortar_int.copy()
        g_old = gb.grids_of_dimension(2)[0]

        # Create a new, finer 2d grid. This is the simplest
        # way to put the fracture in the right place is to create a new
        # bucket, and pick out the higher dimensional grid
        gb_new = meshing.cart_grid([f1], [2, 2], **{'physdims': [1, 1]})
        gb_new.compute_geometry()

        g_new = gb_new.grids_of_dimension(2)[0]

        mortars.replace_grids_in_bucket(gb, {g_old: g_new})

        # Get mortar grid again
        for e, d in gb.edges_props():
            mg = d['mortar_grid']

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
        gb = meshing.cart_grid([f1], N, **{'physdims': [1, 1]})
        gb.compute_geometry()

        # Pick out mortar grid by a loop, there is only one edge in the bucket
        for e, d in gb.edges_props():
            mg = d['mortar_grid']

        old_projection = mg.high_to_mortar_int.copy()
        g_old = gb.grids_of_dimension(2)[0]

        # Create a new, coarser 2d grid. This is the simplest
        # way to put the fracture in the right place is to create a new
        # bucket, and pick out the higher dimensional grid
        gb_new = meshing.cart_grid([f1], [1, 2], **{'physdims': [1, 1]})
        gb_new.compute_geometry()

        g_new = gb_new.grids_of_dimension(2)[0]

        mortars.replace_grids_in_bucket(gb, {g_old: g_new})

        # Get mortar grid again
        for e, d in gb.edges_props():
            mg = d['mortar_grid']

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
        gb = meshing.cart_grid([f1], N, **{'physdims': [1, 1]})
        gb.compute_geometry()

        # Pick out mortar grid by a loop, there is only one edge in the bucket
        for e, d in gb.edges_props():
            mg = d['mortar_grid']

        old_projection = mg.high_to_mortar_int.copy()
        g_old = gb.grids_of_dimension(2)[0]

        # Create a new, finer 2d grid. This is the simplest
        # way to put the fracture in the right place is to create a new
        # bucket, and pick out the higher dimensional grid
        gb_new = meshing.cart_grid([f1], [2, 2], **{'physdims': [1, 1]})
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
        for e, d in gb.edges_props():
            mg = d['mortar_grid']

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
        gb = meshing.cart_grid([f1], N, **{'physdims': [1, 1]})
        gb.compute_geometry()

        # Pick out mortar grid by a loop, there is only one edge in the bucket
        for e, d in gb.edges_props():
            mg = d['mortar_grid']

        old_projection = mg.high_to_mortar_int.copy()
        g_old = gb.grids_of_dimension(2)[0]

        # Create a new, finer 2d grid. This is the simplest
        # way to put the fracture in the right place is to create a new
        # bucket, and pick out the higher dimensional grid
        gb_new = meshing.cart_grid([f1], [2, 2], **{'physdims': [1, 1]})
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
        for e, d in gb.edges_props():
            mg = d['mortar_grid']

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
        gb = meshing.cart_grid([f1], N, **{'physdims': [1, 1]})
        gb.compute_geometry()

        # Pick out mortar grid by a loop, there is only one edge in the bucket
        for e, d in gb.edges_props():
            mg = d['mortar_grid']

        old_projection = mg.high_to_mortar_int.copy()
        g_old = gb.grids_of_dimension(2)[0]

        # Create a new, finer 2d grid. This is the simplest
        # way to put the fracture in the right place is to create a new
        # bucket, and pick out the higher dimensional grid
        gb_new = meshing.cart_grid([f1], [2, 2], **{'physdims': [1, 1]})
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

        fn = g_new.face_nodes.indices.reshape((2, g_new.num_faces), order='F')
        fn[:, 8] = np.array([4, 8])
        fn[:, 9] = np.array([4, 6])
        fn[:, 12] = np.array([7, 5])
        fn[:, 13] = np.array([5, 3])

        mortars.replace_grids_in_bucket(gb, {g_old: g_new})

        # Get mortar grid again
        for e, d in gb.edges_props():
            mg = d['mortar_grid']

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
        gb = meshing.cart_grid([f1], N, **{'physdims': [1, 1]})
        gb.compute_geometry()

        # Pick out mortar grid by a loop, there is only one edge in the bucket
        for e, d in gb.edges_props():
            mg = d['mortar_grid']

        old_projection = mg.high_to_mortar_int.copy()
        g_old = gb.grids_of_dimension(2)[0]

        # Create a new, finer 2d grid. This is the simplest
        # way to put the fracture in the right place is to create a new
        # bucket, and pick out the higher dimensional grid
        gb_new = meshing.cart_grid([f1], [2, 2], **{'physdims': [1, 1]})
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

        fn = g_new.face_nodes.indices.reshape((2, g_new.num_faces), order='F')
        fn[:, 8] = np.array([4, 8])
        fn[:, 9] = np.array([4, 6])
        fn[:, 12] = np.array([7, 5])
        fn[:, 13] = np.array([5, 3])

        mortars.replace_grids_in_bucket(gb, {g_old: g_new})

        # Get mortar grid again
        for e, d in gb.edges_props():
            mg = d['mortar_grid']

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



if __name__ == '__main__':
    unittest.main()
