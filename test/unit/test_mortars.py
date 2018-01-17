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
          weights, new, old = mortars.match_grids_1d(g, g)

          assert np.allclose(weights, np.ones(2))
          assert np.allclose(old, np.arange(2))
          assert np.allclose(new, np.arange(2))

     def test_merge_grids_non_matching(self):
          g = TensorGrid(np.arange(3))
          h = TensorGrid(np.arange(3))
          h.nodes[0, 1] = 0.5
          weights, new, old = mortars.match_grids_1d(g, h)

          assert np.allclose(weights, np.array([0.5, 0.5, 1]))
          assert np.allclose(new, np.array([0, 0, 1]))
          assert np.allclose(old, np.array([0, 1, 1]))

     def test_merge_grids_reverse_order(self):
          g = TensorGrid(np.arange(3))
          h = TensorGrid(np.arange(3))
          h.nodes = h.nodes[:, ::-1]
          weights, new, old = mortars.match_grids_1d(g, h)

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

        old_projection = mg.high_to_mortar.copy()

        g_old = gb.grids_of_dimension(2)[0]
        g_new = g_old.copy()

        mortars.replace_grids_in_bucket(gb, {g_old: g_new})

        # Get mortar grid again
        for e, d in gb.edges_props():
            mg = d['mortar_grid']

        new_projection = mg.high_to_mortar

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

        old_projection = mg.high_to_mortar.copy()
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

        new_projection = mg.high_to_mortar

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

        old_projection = mg.high_to_mortar.copy()
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

        new_projection = mg.high_to_mortar

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

        old_projection = mg.high_to_mortar.copy()
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

        mortars.replace_grids_in_bucket(gb, {g_old: g_new})

        # Get mortar grid again
        for e, d in gb.edges_props():
            mg = d['mortar_grid']

        new_projection = mg.high_to_mortar

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

    if __name__ == '__main__':
        unittest.main()

a = TestReplaceHigherDimensionalGrid()
a.test_coarsen_high_dima()