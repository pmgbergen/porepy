#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for the partial (i.e. spatially local) update of mpsa discretizations
after grid updates due to fracture propagation.

Practical note: Faces in the fracture plane are given by
(nx + 1) * ny * nz + nx * (ny + 1) * nz + nx * ny * nz/2
+ placement in the plane itself
"""


import unittest
import porepy as pp
from test.integration import setup_mixed_dimensional_grids as setup_gb
from test.integration.setup_mixed_dimensional_grids import set_bc_mech, \
    update_apertures
from test.integration.fracture_propagation_utils import propagate_and_update, \
    compare_updates


#-----------------------------------------------------------------------------#


class BasicsTest(unittest.TestCase):

#-----------------------------------------------------------------------------#

    def test_discretization_and_propagation_2d(self):
        """
        Discretize and solve mechanical problem, propagate fracture, update
        discretization and solve anew. Buckets:
        1 premade in its final form
        2-3 grown in a single step (from two different inital fractures)
        4 grown in two steps
        """
        n_cells = [4, 4]
        gb_1 = setup_gb.setup_mech(n_cells, .75)
        gb_2 = setup_gb.setup_mech(n_cells, .25)
        gb_3 = setup_gb.setup_mech(n_cells, .50)
        gb_4 = setup_gb.setup_mech(n_cells, .25)

        discr = pp.FracturedMpsa()

        # Initial discretizations
        g_1 = gb_1.grids_of_dimension(2)[0]
        d_1 = gb_1.node_props(g_1)
        lhs_1, rhs_1 = discr.matrix_rhs(g_1, d_1)
        g_2 = gb_2.grids_of_dimension(2)[0]
        d_2 = gb_2.node_props(g_2)
        lhs_2, rhs_2 = discr.matrix_rhs(g_2, d_2)

        g_3 = gb_3.grids_of_dimension(2)[0]
        d_3 = gb_3.node_props(g_3)
        lhs_3, rhs_3 = discr.matrix_rhs(g_3, d_3)
        g_4 = gb_4.grids_of_dimension(2)[0]
        d_4 = gb_4.node_props(g_4)
        lhs_4, rhs_4 = discr.matrix_rhs(g_4, d_4)

        # Propagate and update discretizations
        lhs_2, rhs_2 = propagate_and_update(gb_2, [29, 30], discr, set_bc_mech,
                                            update_apertures)
        lhs_3, rhs_3 = propagate_and_update(gb_3, [30], discr, set_bc_mech,
                                            update_apertures)
        lhs_4, rhs_4 = propagate_and_update(gb_4, [29], discr, set_bc_mech,
                                            update_apertures)
        lhs_4, rhs_4 = propagate_and_update(gb_4, [30], discr, set_bc_mech,
                                            update_apertures)

        buckets = [gb_1, gb_2, gb_3, gb_4]
        lhs = [lhs_1, lhs_2, lhs_3, lhs_4]
        rhs = [rhs_1, rhs_2, rhs_3, rhs_4]
        compare_updates(buckets, lhs, rhs, fractured_mpsa=True)

    def test_discretization_and_propagation_2d_small(self):
        """
        Discretize and solve mechanical problem, propagate fracture, update
        discretization and solve anew. Buckets:
        1 premade in its final form
        2-3 grown in a single step (from two different inital fractures)
        4 grown in two steps
        """
        n_cells = [2, 2]
        gb_1 = setup_gb.setup_mech(n_cells, 1)
        gb_2 = setup_gb.setup_mech(n_cells, .5)

        discr = pp.FracturedMpsa()

        # Initial discretizations
        g_1 = gb_1.grids_of_dimension(2)[0]
        d_1 = gb_1.node_props(g_1)
        lhs_1, rhs_1 = discr.matrix_rhs(g_1, d_1)
        g_2 = gb_2.grids_of_dimension(2)[0]
        d_2 = gb_2.node_props(g_2)
        lhs_2, rhs_2 = discr.matrix_rhs(g_2, d_2)

        # Propagate and update discretizations
        lhs_2, rhs_2 = propagate_and_update(gb_2, [9], discr, set_bc_mech,
                                            update_apertures)

        buckets = [gb_1, gb_2]
        lhs = [lhs_1, lhs_2]
        rhs = [rhs_1, rhs_2]
        compare_updates(buckets, lhs, rhs, fractured_mpsa=True)

    def test_discretization_and_propagation_3d(self):
        """
        Discretize and solve mechanical problem, propagate fracture, update
        discretization and solve anew. Buckets:
        1 premade in its final form
        2-3 grown in a single step (from two different inital fractures)
        4 grown in two steps
        """
        n_cells = [4, 4, 4]
        dim_max = 3
        gb_1 = setup_gb.setup_mech(n_cells, .75)
        gb_2 = setup_gb.setup_mech(n_cells, .25)
        gb_3 = setup_gb.setup_mech(n_cells, .50)
        gb_4 = setup_gb.setup_mech(n_cells, .25)

        discr = pp.FracturedMpsa()

        # Initial discretizations
        g_1 = gb_1.grids_of_dimension(dim_max)[0]
        d_1 = gb_1.node_props(g_1)
        lhs_1, rhs_1 = discr.matrix_rhs(g_1, d_1)
        g_2 = gb_2.grids_of_dimension(dim_max)[0]
        d_2 = gb_2.node_props(g_2)
        lhs_2, rhs_2 = discr.matrix_rhs(g_2, d_2)

        g_3 = gb_3.grids_of_dimension(dim_max)[0]
        d_3 = gb_3.node_props(g_3)
        lhs_3, rhs_3 = discr.matrix_rhs(g_3, d_3)
        g_4 = gb_4.grids_of_dimension(dim_max)[0]
        d_4 = gb_4.node_props(g_4)
        lhs_4, rhs_4 = discr.matrix_rhs(g_4, d_4)

        # Propagate and update discretizations
        lhs_2, rhs_2 = propagate_and_update(gb_2, [193, 194, 197, 198],
                                            discr, set_bc_mech,
                                            update_apertures)
        lhs_3, rhs_3 = propagate_and_update(gb_3, [194, 198], discr,
                                            set_bc_mech, update_apertures)
        lhs_4, rhs_4 = propagate_and_update(gb_4, [193], discr, set_bc_mech,
                                            update_apertures)
        lhs_4, rhs_4 = propagate_and_update(gb_4, [194, 197, 198], discr,
                                            set_bc_mech, update_apertures)

        buckets = [gb_1, gb_2, gb_3, gb_4]
        lhs = [lhs_1, lhs_2, lhs_3, lhs_4]
        rhs = [rhs_1, rhs_2, rhs_3, rhs_4]
        compare_updates(buckets, lhs, rhs, fractured_mpsa=True)


if __name__ == '__main__':
    unittest.main()
