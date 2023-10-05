import unittest
import numpy as np
import scipy.sparse as sps
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
