""" Various tests of the derived implicit euler discretizations.

TODO: Expand!
"""
import unittest

import numpy as np

import porepy as pp


class TestImplicitUpwind(unittest.TestCase):
    def test_1d_heterogeneous_weights(self):
        """Check that the weights are upwinded."""
        g = pp.CartGrid(3, 1)
        g.compute_geometry()

        solver = pp.utils.derived_discretizations.implicit_euler.ImplicitUpwind()
        dis = solver.darcy_flux(g, [2, 0, 0])

        bf = g.tags["domain_boundary_faces"].nonzero()[0]
        bc = pp.BoundaryCondition(g, bf, bf.size * ["dir"])
        bc_val = np.array([2, 0, 0, 2]).ravel("F")
        specified_parameters = {
            "bc": bc,
            "bc_values": bc_val,
            "darcy_flux": dis,
            "time_step": 1,
            "advection_weight": np.arange(2, 5),
        }
        data = pp.initialize_default_data(g, {}, "transport", specified_parameters)

        solver.discretize(g, data)

        M, rhs = solver.assemble_matrix_rhs(g, data)
        # First row:    flux x a_w[0] = 2 x 2, which flows out of the cell (+)
        # Second row:   flux x a_w[0] = 2 x 2, which flows into the cell (-)
        #               flux x a_w[1] = 2 x 3, which flows out of the cell (+)
        # Last row:     flux x a_w[1] = 2 x 3,, which flows into the cell (-)
        #               flux x a_w[2] = 2 x 4, which flows out of the cell (+)
        M_known = np.array([[4, 0, 0], [-4, 6, 0], [0, -6, 8]])
        # Left boundary (first cell): Influx x bc_val x advection_weight[0]
        #                                 = 2 x 2 x 2.
        # Right boundary (last cell): outflux => rhs = 0
        rhs_known = np.array([8, 0, 0])

        rtol = 1e-15
        atol = rtol
        self.assertTrue(np.allclose(M.todense(), M_known, rtol, atol))
        self.assertTrue(np.allclose(rhs, rhs_known, rtol, atol))


if __name__ == "__main__":
    unittest.main()
