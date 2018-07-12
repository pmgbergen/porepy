#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 13:37:55 2017

@author: eke001
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 11:19:02 2017

@author: eke001
"""
import unittest
import numpy as np
import scipy.sparse as sps

from porepy.grids.structured import TensorGrid
from porepy.fracs import non_conforming
from porepy.utils import tags
from porepy.utils.setmembership import ismember_rows


class TestMeshMerging(unittest.TestCase):
    def test_merge_1d_grids_equal_nodes(self):
        g = TensorGrid(np.array([0, 1, 2]))
        g.compute_geometry()
        h, offset, g_in_comb, g_in_comb, g_sort, _ = non_conforming.merge_1d_grids(
            g, g, global_ind_offset=0, tol=1e-4
        )

        assert np.allclose(h.nodes[:, g_in_comb], g.nodes[:, g_sort])

    def test_merge_1d_grids_partly_equal_nodes(self):
        g = TensorGrid(np.array([0, 1, 2]))
        h = TensorGrid(np.array([0, 0.5, 1, 2]))
        g.compute_geometry()
        h.compute_geometry()
        gh, offset, g_in_comb, h_in_comb, g_sort, h_sort = non_conforming.merge_1d_grids(
            g, h, global_ind_offset=0, tol=1e-4
        )

        assert np.allclose(gh.nodes[:, g_in_comb], g.nodes[:, g_sort])
        assert np.allclose(gh.nodes[:, h_in_comb], h.nodes[:, h_sort])

    def test_merge_1d_grids_unequal_nodes(self):
        # Unequal nodes along the x-axis
        g = TensorGrid(np.array([0, 1, 2]))
        h = TensorGrid(np.array([0, 0.5, 2]))
        g.compute_geometry()
        h.compute_geometry()
        gh, offset, g_in_comb, h_in_comb, g_sort, h_sort = non_conforming.merge_1d_grids(
            g, h, global_ind_offset=0, tol=1e-4
        )

        assert np.allclose(gh.nodes[:, g_in_comb], g.nodes[:, g_sort])
        assert np.allclose(gh.nodes[:, h_in_comb], h.nodes[:, h_sort])

    def test_merge_1d_grids_rotation(self):
        # 1d grids rotated
        g = TensorGrid(np.array([0, 1, 2]))
        g.nodes = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]).T
        g.compute_geometry()
        h = TensorGrid(np.array([0, 1, 2]))
        h.nodes = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [2, 2, 2]]).T
        h.compute_geometry()

        gh, offset, g_in_comb, h_in_comb, g_sort, h_sort = non_conforming.merge_1d_grids(
            g, h, global_ind_offset=0
        )
        assert np.allclose(gh.nodes[:, g_in_comb], g.nodes[:, g_sort])
        assert np.allclose(gh.nodes[:, h_in_comb], h.nodes[:, h_sort])

    def test_merge_1d_permuted_nodes(self):
        g = TensorGrid(np.array([0, 1, 2]))
        g.nodes = np.array([[1, -1, 0], [0, 0, 0], [0, 0, 0]])
        g.global_point_ind = np.array([2, 0, 1])
        g.face_nodes.indices = np.array([1, 2, 0])

        h = TensorGrid(np.array([-1, 0, 1]))
        h.global_point_ind = np.array([0, 1, 2])
        g.compute_geometry()
        h.compute_geometry()
        c, _, _, _, _, _ = non_conforming.merge_1d_grids(g, h)

        ismem, maps = ismember_rows(c.global_point_ind, g.global_point_ind)
        assert ismem.sum() == c.num_nodes
        assert np.allclose(g.nodes[:, maps], c.nodes)
        ismem, maps = ismember_rows(c.global_point_ind, h.global_point_ind)
        assert ismem.sum() == c.num_nodes
        assert np.allclose(h.nodes[:, maps], c.nodes)

    def test_update_face_nodes_equal_2d(self):
        data = np.ones(4)
        rows = np.array([0, 1, 2, 3])
        cols = np.array([0, 0, 1, 1])
        fn = sps.coo_matrix((data, (rows, cols)), shape=(4, 2))
        g = MockGrid(dim=2, num_faces=2, face_nodes=fn)

        delete_faces = np.array([0])
        new_face_ind = non_conforming.update_face_nodes(g, delete_faces, 1, 2)
        assert new_face_ind.size == 1
        assert new_face_ind[0] == 1
        fn_known = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.bool)

        assert np.allclose(fn_known, g.face_nodes.A)

    def test_update_face_nodes_equal_3d(self):
        data = np.ones(6)
        rows = np.array([0, 1, 2, 3, 1, 2])
        cols = np.array([0, 0, 0, 1, 1, 1])
        fn = sps.coo_matrix((data, (rows, cols)), shape=(4, 2))
        g = MockGrid(dim=3, num_faces=2, face_nodes=fn)

        delete_faces = np.array([0])
        new_face_ind = non_conforming.update_face_nodes(g, delete_faces, 1, 0)
        assert new_face_ind.size == 1
        assert new_face_ind[0] == 1
        fn_known = np.array([[0, 1], [1, 1], [1, 1], [1, 0]], dtype=np.bool)

        assert np.allclose(fn_known, g.face_nodes.A)

    def test_update_face_nodes_add_none(self):
        # Only delete cells
        data = np.ones(4)
        rows = np.array([0, 1, 2, 3])
        cols = np.array([0, 0, 1, 1])
        fn = sps.coo_matrix((data, (rows, cols)), shape=(4, 2))
        g = MockGrid(dim=2, num_faces=2, face_nodes=fn)

        delete_faces = np.array([0])
        new_face_ind = non_conforming.update_face_nodes(
            g, delete_faces, num_new_faces=0, new_node_offset=2
        )
        assert new_face_ind.size == 0
        fn_known = np.array([[0], [0], [1], [1]], dtype=np.bool)

        assert np.allclose(fn_known, g.face_nodes.A)

    def test_update_face_nodes_remove_all(self):
        # only add cells
        data = np.ones(4)
        rows = np.array([0, 1, 2, 3])
        cols = np.array([0, 0, 1, 1])
        fn = sps.coo_matrix((data, (rows, cols)), shape=(4, 2))
        g = MockGrid(dim=2, num_faces=2, face_nodes=fn)

        delete_faces = np.array([0, 1])
        new_face_ind = non_conforming.update_face_nodes(g, delete_faces, 1, 2)
        assert new_face_ind.size == 1
        assert new_face_ind[0] == 0
        fn_known = np.array([[0], [0], [1], [1]], dtype=np.bool)

        assert np.allclose(fn_known, g.face_nodes.A)

    def test_update_cell_faces_no_update(self):
        # Same number of delete and new faces
        # cell-face
        data = np.ones(3)
        rows = np.array([0, 1, 2])
        cols = np.array([0, 0, 0])
        cf = sps.coo_matrix((data, (rows, cols)), shape=(3, 1))
        # face-nodes
        fn_orig = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])

        data = np.ones(6)
        rows = np.array([1, 2, 2, 3, 0, 1])
        cols = np.array([0, 0, 1, 1, 2, 2])
        fn = sps.coo_matrix((data, (rows, cols)), shape=(4, 3))

        nodes = np.array([[0, 1, 2, 3], [0, 0, 0, 0], [0, 0, 0, 0]])
        nodes_orig = nodes
        g = MockGrid(
            dim=2, num_faces=3, face_nodes=fn, num_cells=1, cell_faces=cf, nodes=nodes
        )
        delete_faces = np.array([0])
        new_faces = np.array([2])
        in_combined = np.array([0, 1])
        non_conforming.update_cell_faces(
            g, delete_faces, new_faces, in_combined, fn_orig, nodes_orig
        )

        cf_expected = np.array([0, 1, 2])
        assert np.allclose(np.sort(g.cell_faces.indices), cf_expected)

    def test_update_cell_faces_one_by_two(self):
        # cell-face
        data = np.ones(3)
        rows = np.array([0, 1, 2])
        cols = np.array([0, 0, 0])
        cf = sps.coo_matrix((data, (rows, cols)), shape=(3, 1))
        # face-nodes
        fn_orig = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])

        data = np.ones(8)
        rows = np.array([2, 3, 3, 4, 0, 1, 1, 2])
        cols = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        fn = sps.coo_matrix((data, (rows, cols)), shape=(5, 4))

        nodes = np.array([[0, 0.5, 1, 2, 3], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        nodes_orig = nodes[:, [0, 2, 3, 4]]
        g = MockGrid(
            dim=2, num_faces=4, face_nodes=fn, num_cells=1, cell_faces=cf, nodes=nodes
        )
        delete_faces = np.array([0])
        new_faces = np.array([2, 3])
        in_combined = np.array([0, 2])
        non_conforming.update_cell_faces(
            g, delete_faces, new_faces, in_combined, fn_orig, nodes_orig
        )

        cf_expected = np.array([0, 1, 2, 3])
        assert np.allclose(np.sort(g.cell_faces.indices), cf_expected)

    def test_update_cell_faces_one_by_two_reverse_order(self):
        # cell-face
        data = np.ones(3)
        rows = np.array([0, 1, 2])
        cols = np.array([0, 0, 0])
        cf = sps.coo_matrix((data, (rows, cols)), shape=(3, 1))
        # face-nodes
        fn_orig = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])

        data = np.ones(8)
        rows = np.array([2, 3, 3, 4, 0, 1, 1, 2])
        cols = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        fn = sps.coo_matrix((data, (rows, cols)), shape=(5, 4))

        nodes = np.array([[0, 0.5, 1, 2, 3], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        nodes_orig = nodes[:, [0, 2, 3, 4]]
        g = MockGrid(
            dim=2, num_faces=4, face_nodes=fn, num_cells=1, cell_faces=cf, nodes=nodes
        )
        delete_faces = np.array([0])
        new_faces = np.array([3, 2])
        in_combined = np.array([0, 2])
        non_conforming.update_cell_faces(
            g, delete_faces, new_faces, in_combined, fn_orig, nodes_orig
        )

        cf_expected = np.array([0, 1, 2, 3])
        assert np.allclose(np.sort(g.cell_faces.indices), cf_expected)

    def test_update_cell_faces_delete_shared(self):
        # Two cells sharing a face
        # cell-face
        data = np.ones(5)
        rows = np.array([0, 1, 2, 2, 3])
        cols = np.array([0, 0, 0, 1, 1])
        cf = sps.coo_matrix((data, (rows, cols)), shape=(4, 2))
        # face-nodes
        fn_orig = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])

        data = np.ones(10)
        rows = np.array([0, 1, 1, 2, 4, 5, 2, 3, 3, 4])
        cols = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        fn = sps.coo_matrix((data, (rows, cols)), shape=(6, 5))

        nodes = np.array([[0, 1, 2, 2.5, 3, 4], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        nodes_orig = nodes[:, [0, 1, 2, 4, 5]]
        g = MockGrid(
            dim=2, num_faces=5, face_nodes=fn, num_cells=2, cell_faces=cf, nodes=nodes
        )
        delete_faces = np.array([2])
        new_faces = np.array([3, 4])
        in_combined = np.array([0, 2])
        non_conforming.update_cell_faces(
            g, delete_faces, new_faces, in_combined, fn_orig, nodes_orig
        )

        cf_expected = np.array([[1, 1, 0, 1, 1], [0, 0, 1, 1, 1]], dtype=np.bool).T
        assert np.allclose(np.abs(g.cell_faces.toarray()), cf_expected)

    def test_update_cell_faces_delete_shared_reversed(self):
        # cell-face
        data = np.ones(5)
        rows = np.array([0, 1, 2, 2, 3])
        cols = np.array([0, 0, 0, 1, 1])
        cf = sps.coo_matrix((data, (rows, cols)), shape=(4, 2))
        # face-nodes
        fn_orig = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])

        data = np.ones(10)
        rows = np.array([0, 1, 1, 2, 4, 5, 2, 3, 3, 4])
        cols = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        fn = sps.coo_matrix((data, (rows, cols)), shape=(6, 5))

        nodes = np.array([[0, 1, 2, 2.5, 3, 4], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        nodes_orig = nodes[:, [0, 1, 2, 4, 5]]
        g = MockGrid(
            dim=2, num_faces=5, face_nodes=fn, num_cells=2, cell_faces=cf, nodes=nodes
        )
        delete_faces = np.array([2])
        new_faces = np.array([4, 3])
        in_combined = np.array([0, 2])
        non_conforming.update_cell_faces(
            g, delete_faces, new_faces, in_combined, fn_orig, nodes_orig
        )

        cf_expected = np.array([[1, 1, 0, 1, 1], [0, 0, 1, 1, 1]], dtype=np.bool).T
        assert np.allclose(np.abs(g.cell_faces.toarray()), cf_expected)

    def test_update_cell_faces_change_all(self):
        data = np.ones(2)
        rows = np.array([0, 1])
        cols = np.array([0, 1])
        cf = sps.coo_matrix((data, (rows, cols)), shape=(2, 2))
        # face-nodes
        fn_orig = np.array([[0, 1], [1, 2]])

        data = np.ones(10)
        rows = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5])
        cols = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        fn = sps.coo_matrix((data, (rows, cols)), shape=(6, 5))

        nodes = np.array([[0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        nodes_orig = nodes[:, [0, 2, 5]]
        g = MockGrid(
            dim=2, num_faces=5, face_nodes=fn, num_cells=2, cell_faces=cf, nodes=nodes
        )
        delete_faces = np.array([0, 1])
        new_faces = np.array([0, 1, 2, 3, 4, 5])
        in_combined = np.array([0, 2, 5])
        non_conforming.update_cell_faces(
            g, delete_faces, new_faces, in_combined, fn_orig, nodes_orig
        )

        cf_expected = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 1, 1]], dtype=np.bool).T
        assert np.allclose(np.abs(g.cell_faces.toarray()), cf_expected)

    def test_update_tag_simple(self):
        n_tags = 3
        g = TagClass(n_tags)
        g.tags["tip_faces"] = np.array([True, False, False])
        delete_face = [0]
        new_face = [[2]]
        non_conforming.update_face_tags(g, delete_face, new_face)

        known_tag = np.array([False, False, True])  # np.array([1, 2, 0])
        assert np.allclose(known_tag, g.tags["tip_faces"])

    def test_update_tag_one_to_many(self):
        n_tags = 3
        g = TagClass(n_tags)
        g.tags["tip_faces"] = np.array([True, False, False])
        delete_face = [0]
        new_face = [[2, 3]]
        non_conforming.update_face_tags(g, delete_face, new_face)

        known_tag = np.array([False, False, True, True])  # np.array([1, 2, 0, 0])
        assert np.allclose(known_tag, g.tags["tip_faces"])

    def test_update_tag_two_to_many(self):
        n_tags = 3
        g = TagClass(n_tags)
        g.tags["tip_faces"] = np.array([True, False, False])
        delete_face = [0, 2]
        new_face = [[2, 3], [4]]
        non_conforming.update_face_tags(g, delete_face, new_face)

        known_tag = np.array([False, True, True, False])  # np.array([1, 0, 0, 2])
        assert np.allclose(known_tag, g.tags["tip_faces"])

    def test_update_tag_pure_deletion(self):
        n_tags = 3
        g = TagClass(n_tags)
        g.tags["tip_faces"] = np.array([False, True, False])
        delete_face = [0]
        new_face = [[]]
        non_conforming.update_face_tags(g, delete_face, new_face)

        known_tag = np.array([True, False])  # np.array([1, 2])
        assert np.allclose(known_tag, g.tags["tip_faces"])

    def test_global_ind_assignment(self):
        data = np.ones(3)
        rows = np.array([0, 1, 2])
        cols = np.array([0, 0, 0])
        cf = sps.coo_matrix((data, (rows, cols)))

        data = np.ones(6)
        rows = np.array([0, 1, 1, 2, 2, 0])
        cols = np.array([0, 0, 1, 1, 2, 2])
        fn = sps.coo_matrix((data, (rows, cols)))
        g1 = MockGrid(2, num_faces=3, face_nodes=fn, cell_faces=cf, num_cells=1)
        g2 = MockGrid(2, num_faces=3, face_nodes=fn, cell_faces=cf, num_cells=1)
        gl = [[[g1]], [[g2]]]

        list_of_grids, glob_ind = non_conforming.init_global_ind(gl)

        assert list_of_grids[0].frac_num == 0
        assert list_of_grids[1].frac_num == 1

        assert np.allclose(list_of_grids[0].global_point_ind, np.arange(3))
        assert np.allclose(list_of_grids[1].global_point_ind, 3 + np.arange(3))

    def test_merge_two_grids(self):
        # Merge two grids that have a common face. Check that global indices are
        # updated to match, and that they point to the same point coordinates
        data = np.ones(3)
        rows = np.array([0, 1, 2])
        cols = np.array([0, 0, 0])
        cf = sps.coo_matrix((data, (rows, cols)))

        data = np.ones(6)
        rows = np.array([0, 1, 1, 2, 2, 0])
        cols = np.array([0, 0, 1, 1, 2, 2])
        fn = sps.coo_matrix((data, (rows, cols)))
        nodes_1 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        nodes_2 = np.array([[0, 1, 0], [0, 0, -1], [0, 0, 0]])
        g1 = MockGrid(
            2, num_faces=3, face_nodes=fn, cell_faces=cf, num_cells=1, nodes=nodes_1
        )

        g2 = MockGrid(
            2, num_faces=3, face_nodes=fn, cell_faces=cf, num_cells=1, nodes=nodes_2
        )
        g_11 = TensorGrid(np.array([0, 1]))
        g_11.global_point_ind = np.arange(2)
        g_22 = TensorGrid(np.array([0, 1]))
        g_22.global_point_ind = np.arange(2)

        gl = [[[g1], [g_11]], [[g2], [g_22]]]
        intersections = [np.array([1]), np.array([0])]

        list_of_grids, glob_ind = non_conforming.init_global_ind(gl)
        grid_list_1d = non_conforming.process_intersections(
            gl, intersections, glob_ind, list_of_grids, tol=1e-4
        )

        g_1d = grid_list_1d[0]
        ismem, maps = ismember_rows(g_1d.global_point_ind, g1.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(g1.nodes[:, maps], g_1d.nodes)

        ismem, maps = ismember_rows(g_1d.global_point_ind, g2.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(g2.nodes[:, maps], g_1d.nodes)

    def test_merge_three_grids_no_common_point(self):
        # Merge three grids: One in the mid
        data = np.ones(3)
        rows = np.array([0, 1, 2])
        cols = np.array([0, 0, 0])
        cf_1 = sps.coo_matrix((data, (rows, cols)))

        data = np.ones(6)
        rows = np.array([0, 1, 2, 1, 3, 4])
        cols = np.array([0, 0, 0, 1, 1, 1])
        cf_2 = sps.coo_matrix((data, (rows, cols)))

        data = np.ones(6)
        rows = np.array([0, 1, 1, 2, 2, 0])
        cols = np.array([0, 0, 1, 1, 2, 2])
        fn_1 = sps.coo_matrix((data, (rows, cols)))

        data = np.ones(10)
        rows = np.array([0, 1, 1, 3, 3, 0, 1, 2, 2, 3])
        cols = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        fn_2 = sps.coo_matrix((data, (rows, cols)))

        nodes_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        nodes_2 = np.array([[0, 1, 0], [0, 0, -1], [0, 0, 0]])
        nodes_3 = np.array([[0, 1, 0], [2, 1, 1], [0, 0, 0]])

        # Middle grid, unit square divided into two. Will have neighbors on top and
        # bottom.
        g1 = MockGrid(
            2, num_faces=5, face_nodes=fn_2, cell_faces=cf_2, num_cells=2, nodes=nodes_1
        )
        # Neighbor on bottom
        g2 = MockGrid(
            2, num_faces=3, face_nodes=fn_1, cell_faces=cf_1, num_cells=1, nodes=nodes_2
        )
        # Neighbor on top.
        g3 = MockGrid(
            2, num_faces=3, face_nodes=fn_1, cell_faces=cf_1, num_cells=1, nodes=nodes_3
        )

        # Bottom 1d grid, as seen from g1
        g_11 = TensorGrid(np.array([0, 1]))
        g_11.global_point_ind = np.arange(2)
        # Top 1d grid, as seen from g1
        g_13 = TensorGrid(np.array([0, 1]))
        g_13.nodes = np.array([[0, 1], [1, 1], [0, 0]])
        # Note global point indices here, in accordance with the ordering in
        # nodes_1
        g_13.global_point_ind = np.array([2, 3])

        # Bottom 1d grid, as seen from g2
        g_22 = TensorGrid(np.array([0, 1]))
        g_22.global_point_ind = np.arange(2)
        # Top 1d grid, as seen from g3
        g_33 = TensorGrid(np.array([1, 2]))
        g_33.nodes = np.array([[0, 1], [1, 1], [0, 0]])
        # Global point indices, as ordered in nodes_3
        g_33.global_point_ind = np.array([1, 2])

        gl = [[[g1], [g_11, g_13]], [[g2], [g_22]], [[g3], [g_33]]]
        intersections = [np.array([1, 2]), np.array([0]), np.array([0])]

        list_of_grids, glob_ind = non_conforming.init_global_ind(gl)
        grid_list_1d = non_conforming.process_intersections(
            gl, intersections, glob_ind, list_of_grids, tol=1e-4
        )
        assert len(grid_list_1d) == 2

        g_1d = grid_list_1d[0]
        ismem, maps = ismember_rows(g_1d.global_point_ind, g1.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(g1.nodes[:, maps], g_1d.nodes)
        ismem, maps = ismember_rows(g_1d.global_point_ind, g2.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(g2.nodes[:, maps], g_1d.nodes)

        g_1d = grid_list_1d[1]
        ismem, maps = ismember_rows(g_1d.global_point_ind, g1.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(g1.nodes[:, maps], g_1d.nodes)
        ismem, maps = ismember_rows(g_1d.global_point_ind, g3.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(g3.nodes[:, maps], g_1d.nodes)

    def test_merge_three_grids_common_point(self):
        # Merge three grids, where a central cell share one face each with the two
        # other. Importantly, one node will be involved in both shared faces.
        data = np.ones(3)
        rows = np.array([0, 1, 2])
        cols = np.array([0, 0, 0])
        cf = sps.coo_matrix((data, (rows, cols)))

        data = np.ones(6)
        rows = np.array([0, 1, 1, 2, 2, 0])
        cols = np.array([0, 0, 1, 1, 2, 2])
        fn = sps.coo_matrix((data, (rows, cols)))
        nodes_1 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        nodes_2 = np.array([[0, 1, 0], [0, 0, -1], [0, 0, 0]])
        nodes_3 = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]])

        # Central grid
        g1 = MockGrid(
            2, num_faces=3, face_nodes=fn, cell_faces=cf, num_cells=1, nodes=nodes_1
        )
        # First neighboring grid
        g2 = MockGrid(
            2, num_faces=3, face_nodes=fn, cell_faces=cf, num_cells=1, nodes=nodes_2
        )
        # Second neighboring grid
        g3 = MockGrid(
            2, num_faces=3, face_nodes=fn, cell_faces=cf, num_cells=1, nodes=nodes_3
        )
        # First 1d grid, as seen from g1
        g_11 = TensorGrid(np.array([0, 1]))
        g_11.global_point_ind = np.arange(2)
        # Second 1d grid, as seen from g1
        g_13 = TensorGrid(np.array([0, 1]))
        g_13.nodes = np.array([[0, 0], [0, 1], [0, 0]])
        # Point indices adjusted according to ordering in nodes_1
        g_13.global_point_ind = np.array([0, 2])

        # First 1d grid, as seen from g2
        g_22 = TensorGrid(np.array([0, 1]))
        g_22.global_point_ind = np.arange(2)
        # Second 1d grid, as seen from g3
        g_33 = TensorGrid(np.array([0, 1]))
        g_33.nodes = np.array([[0, 0], [0, 1], [0, 0]])
        g_33.global_point_ind = np.arange(2)

        gl = [[[g1], [g_11, g_13]], [[g2], [g_22]], [[g3], [g_33]]]
        intersections = [np.array([1, 2]), np.array([0]), np.array([0])]

        list_of_grids, glob_ind = non_conforming.init_global_ind(gl)
        grid_list_1d = non_conforming.process_intersections(
            gl, intersections, glob_ind, list_of_grids, tol=1e-4
        )
        assert len(grid_list_1d) == 2

        g_1d = grid_list_1d[0]
        ismem, maps = ismember_rows(g_1d.global_point_ind, g1.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(g1.nodes[:, maps], g_1d.nodes)
        ismem, maps = ismember_rows(g_1d.global_point_ind, g2.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(g2.nodes[:, maps], g_1d.nodes)

        g_1d = grid_list_1d[1]
        ismem, maps = ismember_rows(g_1d.global_point_ind, g1.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(g1.nodes[:, maps], g_1d.nodes)
        ismem, maps = ismember_rows(g_1d.global_point_ind, g3.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(g3.nodes[:, maps], g_1d.nodes)

    def test_merge_two_grids_hanging_node(self):
        # Merge three grids, where a central cell share one face each with the two
        # other. Importantly, one node will be involved in both shared faces.
        data = np.ones(10)
        rows = np.array([0, 1, 1, 2, 2, 3, 3, 0, 3, 1])
        cols = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        fn_1 = sps.coo_matrix((data, (rows, cols)))

        data = np.ones(6)
        rows = np.array([0, 1, 1, 2, 2, 0])
        cols = np.array([0, 0, 1, 1, 2, 2])
        fn_2 = sps.coo_matrix((data, (rows, cols)))

        data = np.ones(6)
        rows = np.array([0, 4, 3, 1, 2, 4])
        cols = np.array([0, 0, 0, 1, 1, 1])
        cf_1 = sps.coo_matrix((data, (rows, cols)))

        data = np.ones(3)
        rows = np.array([0, 1, 2])
        cols = np.array([0, 0, 0])
        cf_2 = sps.coo_matrix((data, (rows, cols)))

        nodes_1 = np.array([[0, 1, 2, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
        nodes_2 = np.array([[0, 2, 1], [0, 0, -1], [0, 0, 0]])

        # Central grid
        g1 = MockGrid(
            2, num_faces=5, face_nodes=fn_1, cell_faces=cf_1, num_cells=2, nodes=nodes_1
        )
        # First neighboring grid
        g2 = MockGrid(
            2, num_faces=3, face_nodes=fn_2, cell_faces=cf_2, num_cells=1, nodes=nodes_2
        )
        # First 1d grid, as seen from g1
        g_11 = TensorGrid(np.array([0, 1, 2]))
        g_11.global_point_ind = np.arange(3)
        # Second 1d grid, as seen from g1

        # First 1d grid, as seen from g2
        g_22 = TensorGrid(np.array([0, 2]))
        g_22.global_point_ind = np.arange(2)
        # Second 1d grid, as seen from g3

        gl = [[[g1], [g_11]], [[g2], [g_22]]]
        intersections = [np.array([1]), np.array([0])]

        list_of_grids, glob_ind = non_conforming.init_global_ind(gl)
        grid_list_1d = non_conforming.process_intersections(
            gl, intersections, glob_ind, list_of_grids, tol=1e-4
        )
        assert len(grid_list_1d) == 1

        g_1d = grid_list_1d[0]
        ismem, maps = ismember_rows(g_1d.global_point_ind, g1.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(g1.nodes[:, maps], g_1d.nodes)
        ismem, maps = ismember_rows(g_1d.global_point_ind, g2.global_point_ind)

        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(g2.nodes[:, maps], g_1d.nodes)

    def test_merge_three_grids_hanging_node_shared_node(self):
        # Merge three grids, where a central cell share one face each with the two
        # other. Importantly, one node will be involved in both shared faces.
        data = np.ones(10)
        rows = np.array([0, 1, 1, 2, 2, 3, 3, 0, 3, 1])
        cols = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        fn_1 = sps.coo_matrix((data, (rows, cols)))

        data = np.ones(6)
        rows = np.array([0, 1, 1, 2, 2, 0])
        cols = np.array([0, 0, 1, 1, 2, 2])
        fn_2 = sps.coo_matrix((data, (rows, cols)))

        data = np.ones(6)
        rows = np.array([0, 4, 3, 1, 2, 4])
        cols = np.array([0, 0, 0, 1, 1, 1])
        cf_1 = sps.coo_matrix((data, (rows, cols)))

        data = np.ones(3)
        rows = np.array([0, 1, 2])
        cols = np.array([0, 0, 0])
        cf_2 = sps.coo_matrix((data, (rows, cols)))

        nodes_1 = np.array([[0, 1, 2, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
        nodes_2 = np.array([[0, 2, 1], [0, 0, -1], [0, 0, 0]])
        nodes_3 = np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]])
        # Central grid
        g1 = MockGrid(
            2, num_faces=5, face_nodes=fn_1, cell_faces=cf_1, num_cells=2, nodes=nodes_1
        )
        # First neighboring grid
        g2 = MockGrid(
            2, num_faces=3, face_nodes=fn_2, cell_faces=cf_2, num_cells=1, nodes=nodes_2
        )
        g3 = MockGrid(
            2, num_faces=3, face_nodes=fn_2, cell_faces=cf_2, num_cells=1, nodes=nodes_3
        )

        # First 1d grid, as seen from g1
        g_11 = TensorGrid(np.array([0, 1, 2]))
        g_11.global_point_ind = np.arange(3)
        # Second 1d grid, as seen from g1
        g_13 = TensorGrid(np.array([0, 1]))
        g_13.nodes = np.array([[0, 1], [0, 1], [0, 0]])
        # Point indices adjusted according to ordering in nodes_1
        g_13.global_point_ind = np.array([0, 3])

        # First 1d grid, as seen from g2
        g_22 = TensorGrid(np.array([0, 2]))
        g_22.global_point_ind = np.arange(2)
        # Second 1d grid, as seen from g3
        g_33 = TensorGrid(np.array([0, 1]))
        g_33.nodes = np.array([[0, 1], [0, 1], [0, 0]])
        g_33.global_point_ind = np.arange(2)

        gl = [[[g1], [g_11, g_13]], [[g2], [g_22]], [[g3], [g_33]]]
        intersections = [np.array([1, 2]), np.array([0]), np.array([0])]

        list_of_grids, glob_ind = non_conforming.init_global_ind(gl)
        grid_list_1d = non_conforming.process_intersections(
            gl, intersections, glob_ind, list_of_grids, tol=1e-4
        )
        assert len(grid_list_1d) == 2

        g_1d = grid_list_1d[0]
        ismem, maps = ismember_rows(g_1d.global_point_ind, g1.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(g1.nodes[:, maps], g_1d.nodes)
        ismem, maps = ismember_rows(g_1d.global_point_ind, g2.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(g2.nodes[:, maps], g_1d.nodes)

        g_1d = grid_list_1d[1]
        ismem, maps = ismember_rows(g_1d.global_point_ind, g1.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(g1.nodes[:, maps], g_1d.nodes)
        ismem, maps = ismember_rows(g_1d.global_point_ind, g3.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(g3.nodes[:, maps], g_1d.nodes)

    def test_merge_three_grids_internal_intersection_no_hanging_node(self):
        # Merge three grids, where a central cell share one face each with the two
        # other. Importantly, one node will be involved in both shared faces.
        data = np.ones(16)
        rows = np.array([0, 1, 1, 2, 0, 3, 0, 4, 1, 3, 1, 4, 2, 3, 2, 4])
        cols = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])
        fn = sps.coo_matrix((data, (rows, cols)))

        data = np.ones(12)
        rows = np.array([0, 5, 3, 1, 7, 5, 0, 2, 4, 1, 4, 6])
        cols = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        cf = sps.coo_matrix((data, (rows, cols)))

        nodes_1 = np.array([[-1, 0, 1, 0, 0], [0, 0, 0, -1, 1], [0, 0, 0, 0, 0]])
        nodes_2 = np.array([[-1, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, -1, 1]])
        nodes_3 = np.array([[0, 0, 0, 0, 0], [-1, 0, 1, 0, 0], [0, 0, 0, -1, 1]])
        # Central grid
        gxy = MockGrid(
            2, num_faces=8, face_nodes=fn, cell_faces=cf, num_cells=4, nodes=nodes_1
        )
        # First neighboring grid
        gxz = MockGrid(
            2, num_faces=8, face_nodes=fn, cell_faces=cf, num_cells=4, nodes=nodes_2
        )
        gyz = MockGrid(
            2, num_faces=8, face_nodes=fn, cell_faces=cf, num_cells=4, nodes=nodes_3
        )

        # First 1d grid, as seen from g1
        g_1x = TensorGrid(np.array([0, 1, 2]))
        g_1x.nodes = np.array([[-1, 0, 1], [0, 0, 0], [0, 0, 0]])
        g_1x.global_point_ind = np.array([0, 1, 2])
        # Third 1d grid, as seen from g1
        g_1y = TensorGrid(np.array([0, 1, 2]))
        g_1y.nodes = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
        # Point indices adjusted according to ordering in nodes_1
        g_1y.global_point_ind = np.array([3, 1, 4])

        # First 1d grid, as seen from g2
        g_2x = TensorGrid(np.array([0, 1, 2]))
        g_2x.nodes = np.array([[-1, 0, 1], [0, 0, 0], [0, 0, 0]])
        g_2x.global_point_ind = np.arange(3)
        # Third 1d grid, as seen from g2
        g_2z = TensorGrid(np.array([0, 1, 2]))
        g_2z.nodes = np.array([[0, 0, 0], [0, 0, 0], [-1, 0, 1]])
        # Point indices adjusted according to ordering in nodes_1
        g_2z.global_point_ind = np.array([3, 1, 4])

        g_3y = TensorGrid(np.array([0, 1, 2]))
        g_3y.nodes = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
        g_3y.global_point_ind = np.arange(3)
        # Second 1d grid, as seen from g1
        g_3z = TensorGrid(np.array([0, 1, 2]))
        g_3z.nodes = np.array([[0, 0, 0], [0, 0, 0], [-1, 0, 1]])
        # Point indices adjusted according to ordering in nodes_1
        g_3z.global_point_ind = np.array([3, 1, 4])

        gl = [[[gxy], [g_1x, g_1y]], [[gxz], [g_2x, g_2z]], [[gyz], [g_3y, g_3z]]]
        intersections = [np.array([1, 2]), np.array([0, 2]), np.array([0, 1])]

        list_of_grids, glob_ind = non_conforming.init_global_ind(gl)
        grid_list_1d = non_conforming.process_intersections(
            gl, intersections, glob_ind, list_of_grids, tol=1e-4
        )
        assert len(grid_list_1d) == 3

        g_1d = grid_list_1d[0]
        ismem, maps = ismember_rows(g_1d.global_point_ind, gxy.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(gxy.nodes[:, maps], g_1d.nodes)
        ismem, maps = ismember_rows(g_1d.global_point_ind, gxz.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(gxz.nodes[:, maps], g_1d.nodes)

        g_1d = grid_list_1d[1]
        ismem, maps = ismember_rows(g_1d.global_point_ind, gxy.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(gxy.nodes[:, maps], g_1d.nodes)
        ismem, maps = ismember_rows(g_1d.global_point_ind, gyz.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(gyz.nodes[:, maps], g_1d.nodes)

        g_1d = grid_list_1d[2]
        ismem, maps = ismember_rows(g_1d.global_point_ind, gxz.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(gxz.nodes[:, maps], g_1d.nodes)

        ismem, maps = ismember_rows(g_1d.global_point_ind, gyz.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(gyz.nodes[:, maps], g_1d.nodes)

    def test_merge_three_grids_internal_intersection_no_hanging_node_reverse_order(
        self
    ):
        # Merge three grids, where a central cell share one face each with the two
        # other. Importantly, one node will be involved in both shared faces.
        data = np.ones(16)
        rows = np.array([0, 1, 1, 2, 0, 3, 0, 4, 1, 3, 1, 4, 2, 3, 2, 4])
        cols = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])
        fn = sps.coo_matrix((data, (rows, cols)))

        data = np.ones(12)
        rows = np.array([0, 5, 3, 1, 7, 5, 0, 2, 4, 1, 4, 6])
        cols = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        cf = sps.coo_matrix((data, (rows, cols)))

        nodes_1 = np.array([[-1, 0, 1, 0, 0], [0, 0, 0, -1, 1], [0, 0, 0, 0, 0]])
        nodes_2 = np.array([[-1, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, -1, 1]])
        nodes_3 = np.array([[0, 0, 0, 0, 0], [-1, 0, 1, 0, 0], [0, 0, 0, -1, 1]])
        # Central grid
        gxy = MockGrid(
            2, num_faces=8, face_nodes=fn, cell_faces=cf, num_cells=4, nodes=nodes_1
        )
        # First neighboring grid
        gxz = MockGrid(
            2, num_faces=8, face_nodes=fn, cell_faces=cf, num_cells=4, nodes=nodes_2
        )
        gyz = MockGrid(
            2, num_faces=8, face_nodes=fn, cell_faces=cf, num_cells=4, nodes=nodes_3
        )

        # First 1d grid, as seen from g1
        g_1x = TensorGrid(np.array([0, 1, 2]))
        g_1x.nodes = np.array([[1, 0, -1], [0, 0, 0], [0, 0, 0]])
        g_1x.global_point_ind = np.array([2, 1, 0])
        # g_1x.face_nodes.indices = np.array([1, 2, 0])
        # Third 1d grid, as seen from g1
        g_1y = TensorGrid(np.array([0, 1, 2]))
        g_1y.nodes = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
        # Point indices adjusted according to ordering in nodes_1
        g_1y.global_point_ind = np.array([3, 1, 4])

        # First 1d grid, as seen from g2
        g_2x = TensorGrid(np.array([0, 1, 2]))
        g_2x.nodes = np.array([[-1, 0, 1], [0, 0, 0], [0, 0, 0]])
        g_2x.global_point_ind = np.arange(3)
        # Third 1d grid, as seen from g2
        g_2z = TensorGrid(np.array([0, 1, 2]))
        g_2z.nodes = np.array([[0, 0, 0], [0, 0, 0], [-1, 0, 1]])
        # Point indices adjusted according to ordering in nodes_1
        g_2z.global_point_ind = np.array([3, 1, 4])

        g_3y = TensorGrid(np.array([0, 1, 2]))
        g_3y.nodes = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
        g_3y.global_point_ind = np.arange(3)
        # Second 1d grid, as seen from g1
        g_3z = TensorGrid(np.array([0, 1, 2]))
        g_3z.nodes = np.array([[0, 0, 0], [0, 0, 0], [-1, 0, 1]])
        # Point indices adjusted according to ordering in nodes_1
        g_3z.global_point_ind = np.array([3, 1, 4])

        gl = [[[gxy], [g_1x, g_1y]], [[gxz], [g_2x, g_2z]], [[gyz], [g_3y, g_3z]]]
        intersections = [np.array([1, 2]), np.array([0, 2]), np.array([0, 1])]

        list_of_grids, glob_ind = non_conforming.init_global_ind(gl)
        grid_list_1d = non_conforming.process_intersections(
            gl, intersections, glob_ind, list_of_grids, tol=1e-4
        )
        assert len(grid_list_1d) == 3

        g_1d = grid_list_1d[0]
        ismem, maps = ismember_rows(g_1d.global_point_ind, gxy.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(gxy.nodes[:, maps], g_1d.nodes)
        ismem, maps = ismember_rows(g_1d.global_point_ind, gxz.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(gxz.nodes[:, maps], g_1d.nodes)

        g_1d = grid_list_1d[1]
        ismem, maps = ismember_rows(g_1d.global_point_ind, gxy.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(gxy.nodes[:, maps], g_1d.nodes)
        ismem, maps = ismember_rows(g_1d.global_point_ind, gyz.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(gyz.nodes[:, maps], g_1d.nodes)

        g_1d = grid_list_1d[2]
        ismem, maps = ismember_rows(g_1d.global_point_ind, gxz.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(gxz.nodes[:, maps], g_1d.nodes)

        ismem, maps = ismember_rows(g_1d.global_point_ind, gyz.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(gyz.nodes[:, maps], g_1d.nodes)

    def test_merge_three_grids_internal_intersection_hanging_node(self):
        # Merge three grids, where a central cell share one face each with the two
        # other. Importantly, one node will be involved in both shared faces.
        data = np.ones(20)
        rows = np.array([0, 1, 1, 2, 2, 3, 0, 4, 0, 5, 1, 4, 2, 4, 2, 5, 3, 4, 3, 5])
        cols = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9])
        fn_1 = sps.coo_matrix((data, (rows, cols)))

        data = np.ones(15)
        rows = np.array([0, 5, 3, 1, 5, 6, 0, 4, 7, 2, 6, 8, 2, 7, 9])
        cols = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        cf_1 = sps.coo_matrix((data, (rows, cols)))

        data = np.ones(16)
        rows = np.array([0, 1, 1, 2, 0, 3, 0, 4, 1, 3, 1, 4, 2, 3, 2, 4])
        cols = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])
        fn_2 = sps.coo_matrix((data, (rows, cols)))

        data = np.ones(12)
        rows = np.array([0, 5, 3, 1, 7, 5, 0, 2, 4, 1, 4, 6])
        cols = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        cf_2 = sps.coo_matrix((data, (rows, cols)))

        nodes_1 = np.array(
            [[-1, -0.5, 0, 1, 0, 0], [0, 0, 0, 0, -1, 1], [0, 0, 0, 0, 0, 0]]
        )
        nodes_2 = np.array([[-1, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, -1, 1]])
        nodes_3 = np.array([[0, 0, 0, 0, 0], [-1, 0, 1, 0, 0], [0, 0, 0, -1, 1]])
        # Central grid
        gxy = MockGrid(
            2,
            num_faces=10,
            face_nodes=fn_1,
            cell_faces=cf_1,
            num_cells=5,
            nodes=nodes_1,
        )
        # First neighboring grid
        gxz = MockGrid(
            2, num_faces=8, face_nodes=fn_2, cell_faces=cf_2, num_cells=4, nodes=nodes_2
        )
        gyz = MockGrid(
            2, num_faces=8, face_nodes=fn_2, cell_faces=cf_2, num_cells=4, nodes=nodes_3
        )

        # First 1d grid, as seen from g1
        g_1x = TensorGrid(np.array([0, 1, 2, 3]))
        g_1x.nodes = np.array([[-1, -0.5, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        g_1x.global_point_ind = np.arange(4)
        # Third 1d grid, as seen from g1
        g_1y = TensorGrid(np.array([0, 1, 2]))
        g_1y.nodes = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
        # Point indices adjusted according to ordering in nodes_1
        g_1y.global_point_ind = np.array([4, 2, 5])

        # First 1d grid, as seen from g2
        g_2x = TensorGrid(np.array([0, 1, 2]))
        g_2x.nodes = np.array([[-1, 0, 1], [0, 0, 0], [0, 0, 0]])
        g_2x.global_point_ind = np.arange(3)
        # Third 1d grid, as seen from g2
        g_2z = TensorGrid(np.array([0, 1, 2]))
        g_2z.nodes = np.array([[0, 0, 0], [0, 0, 0], [-1, 0, 1]])
        # Point indices adjusted according to ordering in nodes_1
        g_2z.global_point_ind = np.array([3, 1, 4])

        g_3y = TensorGrid(np.array([0, 1, 2]))
        g_3y.nodes = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
        g_3y.global_point_ind = np.arange(3)
        # Second 1d grid, as seen from g1
        g_3z = TensorGrid(np.array([0, 1, 2]))
        g_3z.nodes = np.array([[0, 0, 0], [0, 0, 0], [-1, 0, 1]])
        # Point indices adjusted according to ordering in nodes_1
        g_3z.global_point_ind = np.array([3, 1, 4])

        gl = [[[gxz], [g_2z, g_2x]], [[gyz], [g_3z, g_3y]], [[gxy], [g_1x, g_1y]]]
        intersections = [np.array([1, 2]), np.array([0, 2]), np.array([0, 1])]

        list_of_grids, glob_ind = non_conforming.init_global_ind(gl)
        grid_list_1d = non_conforming.process_intersections(
            gl, intersections, glob_ind, list_of_grids, tol=1e-4
        )
        assert len(grid_list_1d) == 3

        g_1d = grid_list_1d[0]
        ismem, maps = ismember_rows(g_1d.global_point_ind, gxz.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(gxz.nodes[:, maps], g_1d.nodes)
        ismem, maps = ismember_rows(g_1d.global_point_ind, gyz.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(gyz.nodes[:, maps], g_1d.nodes)

        g_1d = grid_list_1d[1]
        ismem, maps = ismember_rows(g_1d.global_point_ind, gxy.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(gxy.nodes[:, maps], g_1d.nodes)
        ismem, maps = ismember_rows(g_1d.global_point_ind, gxz.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(gxz.nodes[:, maps], g_1d.nodes)

        g_1d = grid_list_1d[2]
        ismem, maps = ismember_rows(g_1d.global_point_ind, gxy.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(gxy.nodes[:, maps], g_1d.nodes)

        ismem, maps = ismember_rows(g_1d.global_point_ind, gyz.global_point_ind)
        assert ismem.sum() == g_1d.num_nodes
        assert np.allclose(gyz.nodes[:, maps], g_1d.nodes)

    if __name__ == "__main__":
        unittest.main()


class MockGrid:
    """ Class with attributes similar to (some of) those in a real grid. Used
    for testing purposes
    """

    def __init__(
        self,
        dim,
        num_faces=None,
        face_nodes=None,
        nodes=None,
        cell_faces=None,
        num_cells=None,
    ):

        self.dim = dim
        self.face_nodes = face_nodes.tocsc()
        self.num_faces = num_faces
        if cell_faces is not None:
            self.cell_faces = cell_faces.tocsc()
        self.num_cells = num_cells

        self.num_nodes = self.face_nodes.shape[0]
        if nodes is None:
            self.nodes = np.zeros((3, self.num_nodes))
        else:
            self.nodes = nodes
        self.global_point_ind = np.arange(self.num_nodes)


class TagClass:
    def __init__(self, n_tags):
        keys = tags.standard_face_tags()
        values = [np.zeros(n_tags, dtype=bool) for _ in range(len(keys))]
        tags.add_tags(self, dict(zip(keys, values)))
