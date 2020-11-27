import unittest

import numpy as np

import porepy as pp


class TestBasisTransformation(unittest.TestCase):
    def test_default(self):
        g = pp.CartGrid([1, 1])
        g.compute_geometry()
        subcell_topology = pp.fvutils.SubcellTopology(g)
        g_sub = pp.FvSubGrid(g, 0)
        bc = pp.BoundaryConditionVectorial(g_sub)

        subcell_topology = pp.fvutils.SubcellTopology(g)
        BM = pp.fvutils.ExcludeBoundaries(subcell_topology, bc, g.dim).basis_matrix
        bm_ref = np.eye(2 * 2 * 4)
        self.assertTrue(np.allclose(BM.A, bm_ref))

    def test_rotation(self):
        # we do a 45 degree rotation
        g = pp.CartGrid([1, 1])
        g.compute_geometry()
        subcell_topology = pp.fvutils.SubcellTopology(g)
        g_sub = pp.FvSubGrid(g, 0)
        bc = pp.BoundaryConditionVectorial(g_sub)

        bc.basis = np.array([[[1] * 8, [1] * 8], [[1] * 8, [-1] * 8]])

        BM = pp.fvutils.ExcludeBoundaries(subcell_topology, bc, g.dim).basis_matrix

        bm_ref = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1],
            ]
        )
        self.assertTrue(np.allclose(BM.A, bm_ref))
