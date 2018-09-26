import numpy as np
import unittest

import porepy as pp


class TestBasisTransformation(unittest.TestCase):
    def test_default(self):
        g = pp.CartGrid([1, 1])
        bc = pp.BoundaryConditionVectorial(g)
        subcell_topology = pp.fvutils.SubcellTopology(g)
        BM = pp.fvutils.ExcludeBoundaries(subcell_topology, bc, g.dim).basis_matrix
        bm_ref = np.eye(2 * 4 * 2)
        self.assertTrue(np.allclose(BM.A, bm_ref))

    def test_rotation(self):
        # we do a 45 degree rotation
        g = pp.CartGrid([1, 1])
        bc = pp.BoundaryConditionVectorial(g)
        subcell_topology = pp.fvutils.SubcellTopology(g)
        bc.basis = np.array(
            [[[1, 1], [1, 1], [1, 1], [1, 1]], [[1, -1], [1, -1], [1, -1], [1, -1]]]
        )
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
