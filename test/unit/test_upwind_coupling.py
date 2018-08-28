"""
Tests of class UpwindCoupling in module porepy.numerics.fv.transport.upwind
"""
import unittest
import numpy as np
import scipy.sparse as sps

import porepy as pp


class TestUpwindCoupling(unittest.TestCase):
    def generate_grid(self):
        # Generate cartesian grid with one fracture:
        # ---------
        # |   |   |
        # --------- horizontal fracture
        # |   |   |
        # --------
        f = np.array([[0, 2], [1, 1]])
        return pp.meshing.cart_grid([f], [2, 2])

    def block_matrix(self, gs):
        def ndof(g):
            return g.num_cells

        coupler = pp.numerics.mixed_dim.abstract_coupling.AbstractCoupling(
            discr_ndof=ndof
        )
        return coupler.create_block_matrix(gs)

    def test_upwind_2d_1d_positive_flux(self):
        # test coupling between 2D grid and 1D grid with a fluid flux going from
        # 2D grid to 1D grid. The upwind weighting should in this case choose the
        # 2D cell variables as weights

        gb = self.generate_grid()
        g2 = gb.grids_of_dimension(2)[0]
        g1 = gb.grids_of_dimension(1)[0]

        d2 = gb.node_props(g2)
        d1 = gb.node_props(g1)
        de = gb.edge_props((g1, g2))

        _, zero_mat = self.block_matrix([g2, g1, de["mortar_grid"]])

        lam = np.arange(de["mortar_grid"].num_cells)
        de["mortar_solution"] = lam
        upwind = pp.Upwind()
        upwind_coupler = pp.numerics.fv.transport.upwind.UpwindCoupling(upwind)

        matrix = upwind_coupler.matrix_rhs(zero_mat, g2, g1, d2, d1, de)

        matrix_2 = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            ]
        )
        matrix_1 = np.array(
            [[0, 0, 0, 0, 0, 0, -1, 0, -1, 0], [0, 0, 0, 0, 0, 0, 0, -1, 0, -1]]
        )
        matrix_l = np.array(
            [
                [0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, -1, 0, 0],
                [0, 2, 0, 0, 0, 0, 0, 0, -1, 0],
                [3, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            ]
        )
        assert np.allclose(sps.hstack(matrix[0, :]).A, matrix_2)
        assert np.allclose(sps.hstack(matrix[1, :]).A, matrix_1)
        assert np.allclose(sps.hstack(matrix[2, :]).A, matrix_l)

    def test_upwind_2d_1d_negative_flux(self):
        # test coupling between 2D grid and 1D grid with a fluid flux going from
        # 1D grid to 2D grid. The upwind weighting should in this case choose the
        # 1D cell variables as weights

        gb = self.generate_grid()
        g2 = gb.grids_of_dimension(2)[0]
        g1 = gb.grids_of_dimension(1)[0]

        d2 = gb.node_props(g2)
        d1 = gb.node_props(g1)
        de = gb.edge_props((g1, g2))

        _, zero_mat = self.block_matrix([g2, g1, de["mortar_grid"]])

        lam = np.arange(de["mortar_grid"].num_cells)
        de["mortar_solution"] = -lam
        upwind = pp.Upwind()
        upwind_coupler = pp.numerics.fv.transport.upwind.UpwindCoupling(upwind)

        matrix = upwind_coupler.matrix_rhs(zero_mat, g2, g1, d2, d1, de)

        matrix_2 = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            ]
        )
        matrix_1 = np.array(
            [[0, 0, 0, 0, 0, 0, -1, 0, -1, 0], [0, 0, 0, 0, 0, 0, 0, -1, 0, -1]]
        )
        matrix_l = np.array(
            [
                [0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
                [0, 0, 0, 0, 0, -1, 0, -1, 0, 0],
                [0, 0, 0, 0, -2, 0, 0, 0, -1, 0],
                [0, 0, 0, 0, 0, -3, 0, 0, 0, -1],
            ]
        )

        assert np.allclose(sps.hstack(matrix[0, :]).A, matrix_2)
        assert np.allclose(sps.hstack(matrix[1, :]).A, matrix_1)
        assert np.allclose(sps.hstack(matrix[2, :]).A, matrix_l)


if __name__ == "__main__":
    unittest.main()
