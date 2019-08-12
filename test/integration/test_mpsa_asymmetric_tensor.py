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

        k = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))
        g, k = true_2d(g, k)

        subcell_topology = pp.fvutils.SubcellTopology(g)
        bc = pp.fvutils.boundary_to_sub_boundary(bc, subcell_topology)
        bound_exclusion = pp.fvutils.ExcludeBoundaries(subcell_topology, bc, g.dim)
        _, igrad, _, _ = pp.numerics.fv.mpsa.mpsa_elasticity(
            g, k, subcell_topology, bound_exclusion, 0, "python"
        )
        data = np.array(
            [
                -0.75,
                0.25,
                -2.0,
                -2.0,
                0.25,
                -0.75,
                2.0,
                -2.0,
                -2.0,
                2.0,
                -2 / 3,
                -2 / 3,
                -2 / 3,
                -2 / 3,
                2.0,
                -2.0,
                -2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                -2.0,
                2.0,
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

        bc = pp.BoundaryConditionVectorial(g)
        bc.is_dir[:, west + east + south] = True
        bc.is_neu[bc.is_dir] = False

        k = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))

        subcell_topology = pp.fvutils.SubcellTopology(g)
        bc = pp.fvutils.boundary_to_sub_boundary(bc, subcell_topology)
        bound_exclusion = pp.fvutils.ExcludeBoundaries(subcell_topology, bc, g.dim)

        _, igrad, _, _ = pp.numerics.fv.mpsa.mpsa_elasticity(
            g, k, subcell_topology, bound_exclusion, 0, "python"
        )
        data = np.array(
            [
                2.0,
                2.0,
                -4.0,
                -2.0,
                2.0,
                2.0,
                -4.0,
                -2.0,
                2.0,
                2.0,
                -4 / 3,
                -2 / 3,
                -2 / 3,
                2.0,
                2.0,
                -4.0,
                -2.0,
                2.0,
                2.0,
                -4.0,
                -2.0,
                2.0,
                2.0,
                -4 / 3,
                -2 / 3,
                -2 / 3,
                2.0,
                4.0,
                -4.0,
                2.0,
                1.5,
                0.5,
                -0.5,
                -4.0,
                2.0,
                4.0,
                -0.5,
                -1.5,
                -0.5,
                2.0,
                4.0,
                -4.0,
                2.0,
                1.5,
                0.5,
                -0.5,
                -4.0,
                2.0,
                4.0,
                -0.5,
                -1.5,
                -0.5,
                2.0,
                2.0,
                4.0,
                -2.0,
                2.0,
                2.0,
                4.0,
                -2.0,
                2.0,
                2.0,
                4 / 3,
                -2 / 3,
                -2 / 3,
                2.0,
                2.0,
                4.0,
                -2.0,
                2.0,
                2.0,
                4.0,
                -2.0,
                2.0,
                2.0,
                4 / 3,
                -2 / 3,
                -2 / 3,
                2.0,
                4.0,
                4.0,
                2.0,
                1.5,
                -0.5,
                -0.5,
                4.0,
                2.0,
                4.0,
                -0.5,
                1.5,
                -0.5,
                2.0,
                4.0,
                4.0,
                2.0,
                1.5,
                -0.5,
                -0.5,
                4.0,
                2.0,
                4.0,
                -0.5,
                1.5,
                -0.5,
            ]
        )
        indices = np.array(
            [
                36,
                44,
                4,
                60,
                48,
                56,
                16,
                68,
                60,
                68,
                28,
                36,
                56,
                40,
                47,
                5,
                64,
                52,
                59,
                17,
                71,
                64,
                71,
                29,
                40,
                59,
                37,
                0,
                7,
                49,
                12,
                31,
                37,
                19,
                61,
                24,
                12,
                31,
                37,
                41,
                3,
                6,
                53,
                15,
                30,
                41,
                18,
                65,
                27,
                15,
                30,
                41,
                39,
                45,
                8,
                63,
                51,
                57,
                20,
                69,
                63,
                69,
                32,
                39,
                57,
                43,
                46,
                9,
                67,
                55,
                58,
                21,
                70,
                67,
                70,
                33,
                43,
                58,
                38,
                1,
                11,
                50,
                13,
                35,
                38,
                23,
                62,
                25,
                13,
                35,
                38,
                42,
                2,
                10,
                54,
                14,
                34,
                42,
                22,
                66,
                26,
                14,
                34,
                42,
            ]
        )
        indptr = np.array(
            [
                0,
                1,
                2,
                4,
                5,
                6,
                8,
                9,
                10,
                13,
                14,
                15,
                17,
                18,
                19,
                21,
                22,
                23,
                26,
                27,
                28,
                29,
                30,
                33,
                34,
                35,
                36,
                39,
                40,
                41,
                42,
                43,
                46,
                47,
                48,
                49,
                52,
                53,
                54,
                56,
                57,
                58,
                60,
                61,
                62,
                65,
                66,
                67,
                69,
                70,
                71,
                73,
                74,
                75,
                78,
                79,
                80,
                81,
                82,
                85,
                86,
                87,
                88,
                91,
                92,
                93,
                94,
                95,
                98,
                99,
                100,
                101,
                104,
            ]
        )

        igrad_known = sps.csr_matrix((data, indices, indptr))
        self.assertTrue(np.all(np.abs(igrad - igrad_known).A < 1e-12))


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
    constit.values = np.delete(constit.values, (2, 5, 6, 7, 8), axis=0)
    constit.values = np.delete(constit.values, (2, 5, 6, 7, 8), axis=1)
    return g, constit


if __name__ == "__main__":
    unittest.main()
