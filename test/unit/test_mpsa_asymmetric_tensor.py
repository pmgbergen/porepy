import numpy as np
import scipy.sparse as sps
import unittest

import porepy as pp


class TestAsymmetricNeumann(unittest.TestCase):
    def test_cart_2d(self):
        g = pp.CartGrid([2, 1], physdims=(1, 1))
        g.compute_geometry()
        right = g.face_centers[0] > 1 - 1e-10
        top = g.face_centers[1] > 1 - 1e-10

        bc = pp.BoundaryConditionVectorial(g)
        bc.is_dir[:, top] = True
        bc.is_dir[0, right] = True

        bc.is_neu[bc.is_dir] = False

        g = true_2d(g)

        subcell_topology = pp.fvutils.SubcellTopology(g)
        bound_exclusion = pp.fvutils.ExcludeBoundaries(subcell_topology, bc, g.dim)
        cell_node_blocks = np.array(
            [[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 3, 4, 1, 2, 4, 5]]
        )
        ncasym_np = np.arange(32 * 32).reshape((32, 32))
        ncasym = sps.csr_matrix(ncasym_np)
        pp.numerics.fv.mpsa._eliminate_ncasym_neumann(
            ncasym, subcell_topology, bound_exclusion, cell_node_blocks, 2
        )
        eliminate_ind = np.array([0, 1, 16, 17, 26, 27])
        ncasym_np[eliminate_ind] = 0
        self.assertTrue(np.allclose(ncasym.A, ncasym_np))


def true_2d(g, constit=None):
    if g.dim == 2:
        g = g.copy()
        g.cell_centers = np.delete(g.cell_centers, (2), axis=0)
        g.face_centers = np.delete(g.face_centers, (2), axis=0)
        g.face_normals = np.delete(g.face_normals, (2), axis=0)
        g.nodes = np.delete(g.nodes, (2), axis=0)

    if constit is None:
        return g
    constit = constit.copy()
    constit.c = np.delete(constit.c, (2, 5, 6, 7, 8), axis=0)
    constit.c = np.delete(constit.c, (2, 5, 6, 7, 8), axis=1)
    return g, constit


if __name__ == "__main__":
    unittest.main()
