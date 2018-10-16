import numpy as np
import scipy.sparse as sps
import unittest

import porepy as pp


class TestAsymmetricNeumann(unittest.TestCase):
    def test_cart_2d(self):
        g = pp.CartGrid([1, 1], physdims=(1, 1))
        g.compute_geometry()
        right = g.face_centers[0] > 1 - 1e-10
        top = g.face_centers[1] > 1 - 1e-10

        bc = pp.BoundaryConditionVectorial(g)
        bc.is_dir[:, top] = True
        bc.is_dir[0, right] = True

        bc.is_neu[bc.is_dir] = False

        k = pp.FourthOrderTensor(g.dim, np.ones(g.num_cells), np.ones(g.num_cells))
        g, k = true_2d(g, k)

        subcell_topology = pp.fvutils.SubcellTopology(g)
        bound_exclusion = pp.fvutils.ExcludeBoundaries(subcell_topology, bc, g.dim)
        _, igrad, _, _, _ = pp.numerics.fv.mpsa.mpsa_elasticity(
            g, k, subcell_topology, bound_exclusion, 0, "python"
        )
        data = np.array(
            [
                -0.75,
                0.25,
                -2.,
                -2.,
                0.25,
                -0.75,
                2.,
                -2.,
                -2.,
                2.,
                -2 / 3,
                -2 / 3,
                -2 / 3,
                -2 / 3,
                2.,
                -2.,
                -2.,
                2.,
                2.,
                2.,
                2.,
                -2.,
                2.,
            ]
        )
        indices = np.array(
            [
                0,
                8,
                2,
                4,
                0,
                8,
                10,
                3,
                6,
                6,
                9,
                10,
                1,
                14,
                12,
                5,
                12,
                14,
                11,
                13,
                7,
                13,
                15,
            ]
        )
        indptr = np.array([0, 2, 3, 4, 6, 7, 9, 10, 12, 14, 15, 17, 18, 19, 20, 22, 23])
        igrad_known = sps.csr_matrix((data, indices, indptr))
        self.assertTrue(np.all(np.abs(igrad - igrad_known).A < 1e-12))

    def test_cart_3d(self):
        g = pp.CartGrid([1, 1, 1], physdims=(1, 1, 1))
        g.compute_geometry()

        west = g.face_centers[0] < 1e-10
        east = g.face_centers[0] > 1 - 1e-10
        south = g.face_centers[1] < 1e-10
        north = g.face_centers[1] > 1 - 1e-10

        bc = pp.BoundaryConditionVectorial(g)
        bc.is_dir[:, west + east + south] = True
        bc.is_neu[bc.is_dir] = False

        k = pp.FourthOrderTensor(g.dim, np.ones(g.num_cells), np.ones(g.num_cells))

        subcell_topology = pp.fvutils.SubcellTopology(g)
        bound_exclusion = pp.fvutils.ExcludeBoundaries(subcell_topology, bc, g.dim)

        _, igrad, _, _, _ = pp.numerics.fv.mpsa.mpsa_elasticity(
            g, k, subcell_topology, bound_exclusion, 0, "python"
        )
        import pdb

        pdb.set_trace()


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
