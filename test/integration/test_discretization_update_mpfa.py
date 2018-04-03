#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:25:17 2018

@author: ivar

"""


import unittest
import porepy as pp
from test.integration import setup_mixed_dimensional_grids as setup_gb
from test.integration.setup_mixed_dimensional_grids import set_bc_flow, \
    update_apertures
from test.integration.fracture_propagation_utils import propagate_and_update, \
    compare_updates
#------------------------------------------------------------------------------#


class BasicsTest(unittest.TestCase):

#------------------------------------------------------------------------------#


    def test_discretization_and_propagation_2d(self):
        """
        Discretize and solve flow problem, propagate fracture, update
        discretization and solve anew. Buckets:
        1 premade in its final form
        2-3 grown in a single step (from two different inital fractures)
        4 grown in two steps
        """
        physics = 'flow'
        n_cells = [4, 4]
        gb_1 = setup_gb.setup_flow(n_cells, .75)
        gb_2 = setup_gb.setup_flow(n_cells, .25)
        gb_3 = setup_gb.setup_flow(n_cells, .50)
        gb_4 = setup_gb.setup_flow(n_cells, .25)

        flux_discr = pp.MpfaMixedDim(physics)

        # Initial discretizations
        lhs_1, rhs_1 = flux_discr.matrix_rhs(gb_1)
        lhs_2, rhs_2 = flux_discr.matrix_rhs(gb_2)
        lhs_3, rhs_3 = flux_discr.matrix_rhs(gb_3)
        lhs_4, rhs_4 = flux_discr.matrix_rhs(gb_4)
#        p = sps.linalg.spsolve(lhs_4, rhs_4)
#        flux_discr.solver.split(gb_4, 'pressure_1', p)
#        plot_grid.plot_grid(gb_4, 'pressure_1')

        # Propagate and update discretizations
        lhs_2, rhs_2 = propagate_and_update(gb_2, [29, 30], flux_discr,
                                            set_bc_flow, update_apertures)
        lhs_3, rhs_3 = propagate_and_update(gb_3, [30], flux_discr,
                                            set_bc_flow, update_apertures)
        lhs_4, rhs_4 = propagate_and_update(gb_4, [29], flux_discr,
                                            set_bc_flow, update_apertures)
        lhs_4, rhs_4 = propagate_and_update(gb_4, [30], flux_discr,
                                            set_bc_flow, update_apertures)
        buckets = [gb_1, gb_2, gb_3, gb_4]
        lhs = [lhs_1, lhs_2, lhs_3, lhs_4]
        rhs = [rhs_1, rhs_2, rhs_3, rhs_4]
        compare_updates(buckets, lhs, rhs, phys=physics)

    def test_discretization_and_propagation_3d(self):
        """
        Discretize and solve flow problem, propagate fracture, update
        discretization and solve anew. Buckets:
        1 premade in its final form
        2-3 grown in a single step (from two different inital fractures)
        4 grown in two steps
        """
        physics = 'flow'
        n_cells = [4, 4, 4]
        gb_1 = setup_gb.setup_flow(n_cells, .75)
        gb_2 = setup_gb.setup_flow(n_cells, .25)
        gb_3 = setup_gb.setup_flow(n_cells, .50)
        gb_4 = setup_gb.setup_flow(n_cells, .25)

        flux_discr = pp.MpfaMixedDim(physics)

        # Initial discretizations
        lhs_1, rhs_1 = flux_discr.matrix_rhs(gb_1)
        lhs_2, rhs_2 = flux_discr.matrix_rhs(gb_2)
        lhs_3, rhs_3 = flux_discr.matrix_rhs(gb_3)
        lhs_4, rhs_4 = flux_discr.matrix_rhs(gb_4)

        # Propagate and update discretizations
        lhs_2, rhs_2 = propagate_and_update(gb_2, [193, 194, 197, 198],
                                            flux_discr, set_bc_flow,
                                            update_apertures)
        lhs_3, rhs_3 = propagate_and_update(gb_3, [194, 198],
                                            flux_discr, set_bc_flow,
                                            update_apertures)
        lhs_4, rhs_4 = propagate_and_update(gb_4, [193],
                                            flux_discr, set_bc_flow,
                                            update_apertures)
        lhs_4, rhs_4 = propagate_and_update(gb_4, [194, 197, 198],
                                            flux_discr, set_bc_flow,
                                            update_apertures)

        buckets = [gb_1, gb_2, gb_3, gb_4]
        lhs = [lhs_1, lhs_2, lhs_3, lhs_4]
        rhs = [rhs_1, rhs_2, rhs_3, rhs_4]
        compare_updates(buckets, lhs, rhs, phys=physics)


if __name__ == '__main__':
    unittest.main()
