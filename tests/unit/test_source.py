""" Unit tests for source discretizations.
"""
import unittest

import numpy as np
import scipy.sparse as sps

import porepy as pp


class TestSource(unittest.TestCase):
    def test_integral(self):
        g, d = setup_3d_grid()

        src_disc = pp.ScalarSource("flow")
        src_disc.discretize(g, d)
        lhs, rhs = src_disc.assemble_matrix_rhs(g, d)

        rhs_t = np.array([0, 0, 0, 0, 1, 0, 0, 0])

        self.assertTrue(src_disc.ndof(g) == g.num_cells)
        self.assertTrue(np.all(rhs == rhs_t))
        self.assertTrue(lhs.shape == (8, 8))
        self.assertTrue(lhs.nnz == 0)


def setup_3d_grid():
    g = pp.CartGrid([2, 2, 2], physdims=[1, 1, 1])
    g.compute_geometry()
    src = np.zeros(g.num_cells)
    src[4] = 1
    data = pp.initialize_default_data(g, {}, "flow", {"source": src})
    return g, data


if __name__ == "__main__":
    unittest.main()
