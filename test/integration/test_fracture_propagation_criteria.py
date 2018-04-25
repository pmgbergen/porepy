#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:31:05 2018

@author: ivar

Test of displacement correlation for stress intensity factor computation and
resulting fracture growth. The fractures are grown for sifs above a
(artificial) critical threshold. The code is as yet rather messy with commented
vtk export code etc. This is because the test also serves as an example and
debugging aid, where visualization may be very useful.

TODO: Decide if this is a test or an example!
"""
import scipy.sparse as sps
import numpy as np
import unittest
import porepy as pp

from test.integration import setup_mixed_dimensional_grids as setup_gb
from test.integration.setup_mixed_dimensional_grids import set_bc_mech, \
    update_apertures
from test.integration.fracture_propagation_utils import propagate_and_update, \
    check_equivalent_buckets


class BasicsTest(unittest.TestCase):

    def test_displacement_correlation_2d(self, do_save=False):
        """
        Set up a displacement field and evaluate whether to propagate. Buckets:
        1 premade in its final form
        2 grown from a different inital fracture
        """
        n_cells = [12, 12]
        critical_sifs = [.1, .1]
        gb_1 = setup_gb.setup_mech(n_cells, .5)
        gb_2 = setup_gb.setup_mech(n_cells, .25)

        discr = pp.FracturedMpsa(given_traction=True)

        # Initial discretizations

        g_h = gb_2.grids_of_dimension(2)[0]
        d_h = gb_2.node_props(g_h)

        # Initial discretization and solution
        lhs_2, rhs_2 = discr.matrix_rhs(g_h, d_h)
        u = sps.linalg.spsolve(lhs_2, rhs_2)
        save = None
        if do_save:
            save = pp.Exporter(gb_2, "solution", folder="results",
                                     fixed_grid=False)

        # Check for propagation
        faces, _ = pp.displacement_correlation.faces_to_open(gb_2, u,
                                                             critical_sifs)
        save_vtk(save, u, gb_2, 0)
        # Increase boundary displacement, rediscretize, solve and evaluate
        # propagation.

        def set_bc_mech_2(gb):
            set_bc_mech(gb, top_displacement=.3)
        lhs_2, rhs_2 = propagate_and_update(gb_2, faces, discr, set_bc_mech_2,
                                            update_apertures)
        u = sps.linalg.spsolve(lhs_2, rhs_2)
        faces, _ = pp.displacement_correlation.faces_to_open(gb_2, u,
                                                             critical_sifs)
        save_vtk(save, u, gb_2, 1)
        # Now the condition is met. Propagate and update discretizations
        lhs_2, rhs_2 = propagate_and_update(gb_2, faces, discr, set_bc_mech_2,
                                            update_apertures)
        u = sps.linalg.spsolve(lhs_2, rhs_2)
        faces, _ = pp.displacement_correlation.faces_to_open(gb_2, u,
                                                             critical_sifs)
        save_vtk(save, u, gb_2, 2)

        def set_bc_mech_3(gb):
            set_bc_mech(gb, top_displacement=.3)

        lhs_2, rhs_2 = propagate_and_update(gb_2, faces, discr, set_bc_mech_3,
                                            update_apertures)
        u = sps.linalg.spsolve(lhs_2, rhs_2)
        faces, _ = pp.displacement_correlation.faces_to_open(gb_2, u,
                                                             critical_sifs)
        save_vtk(save, u, gb_2, 3)

        lhs_2, rhs_2 = propagate_and_update(gb_2, faces, discr, set_bc_mech_3,
                                            update_apertures)
        u = sps.linalg.spsolve(lhs_2, rhs_2)
        save_vtk(save, u, gb_2, 4)
        if do_save:
            save.write_pvd(np.arange(5))

        buckets = [gb_1, gb_2]
        check_equivalent_buckets(buckets)

    def test_displacement_correlation_2d_internal_fracture(self,
                                                           do_save=False):
        """
        Set up a displacement field and evaluate whether to propagate. Buckets:
        1 premade in its final form
        2 grown from a different inital fracture
        """
        n_cells = [12, 12]
        critical_sifs = [.01, .01]
        gb_1 = setup_gb.setup_mech(n_cells, x_start=0, x_stop=.75)
        gb_2 = setup_gb.setup_mech(n_cells, x_start=.25, x_stop=.50)
#        gb_2_copy = gb_2.copy()

        discr = pp.FracturedMpsa(given_traction=True)

        g_h = gb_2.grids_of_dimension(2)[0]
        d_h = gb_2.node_props(g_h)

        # Initial discretization and solution
        lhs_2, rhs_2 = discr.matrix_rhs(g_h, d_h)
        u = sps.linalg.spsolve(lhs_2, rhs_2)
        save = None
        if do_save:
            save = pp.Exporter(gb_2, "solution", folder="results",
                               fixed_grid=False)

        # Check for propagation
        faces, _ = pp.displacement_correlation.faces_to_open(gb_2, u,
                                                             critical_sifs)
        save_vtk(save, u, gb_2, 0)
        # Increase boundary displacement, rediscretize, solve and evaluate
        # propagation.

        def set_bc_mech_2(gb):
            set_bc_mech(gb, top_displacement=.3)
        lhs_2, rhs_2 = propagate_and_update(gb_2, faces, discr, set_bc_mech_2,
                                            update_apertures)
        u = sps.linalg.spsolve(lhs_2, rhs_2)
        faces, _ = pp.displacement_correlation.faces_to_open(gb_2, u,
                                                             critical_sifs)
        save_vtk(save, u, gb_2, 1)
        # Now the condition is met. Propagate and update discretizations
        lhs_2, rhs_2 = propagate_and_update(gb_2, faces, discr, set_bc_mech_2,
                                            update_apertures)
        u = sps.linalg.spsolve(lhs_2, rhs_2)
        faces, _ = pp.displacement_correlation.faces_to_open(gb_2, u,
                                                             critical_sifs)
        save_vtk(save, u, gb_2, 2)

        def set_bc_mech_3(gb):
            set_bc_mech(gb, top_displacement=.3)

        lhs_2, rhs_2 = propagate_and_update(gb_2, faces, discr, set_bc_mech_3,
                                            update_apertures)
        u = sps.linalg.spsolve(lhs_2, rhs_2)
        faces, _ = pp.displacement_correlation.faces_to_open(gb_2, u,
                                                             critical_sifs)
        save_vtk(save, u, gb_2, 3)
#
        lhs_2, rhs_2 = propagate_and_update(gb_2, faces, discr, set_bc_mech_3,
                                            update_apertures)
        u = sps.linalg.spsolve(lhs_2, rhs_2)
        save_vtk(save, u, gb_2, 4)
        if do_save:
            save.write_pvd(np.arange(5))

        buckets = [gb_1, gb_2]
        check_equivalent_buckets(buckets)

    def test_displacement_correlation_3d_internal_fracture(self,
                                                           do_save=False):
        """
        Set up a displacement field and evaluate whether to propagate. Buckets:
        1 premade in its final form
        2 grown from a different inital fracture
        """
        dim_h = 3
        rm_f = .8
        n_cells = [10, 10, 10]
        critical_sifs = [.005, .1, .1]
        gb_2 = setup_gb.setup_mech(n_cells, x_start=.4, x_stop=.6)

        discr = pp.FracturedMpsa(given_traction=True)

        # Initial discretizations

        def set_bc_mech_2(gb):
            set_bc_mech(gb, top_displacement=.03)

        def set_bc_mech_3(gb):
            set_bc_mech(gb, top_displacement=0.05)

        g_h = gb_2.grids_of_dimension(dim_h)[0]
        d_h = gb_2.node_props(g_h)

        # Initial discretization and solution
        lhs_2, rhs_2 = discr.matrix_rhs(g_h, d_h)
        u = sps.linalg.spsolve(lhs_2, rhs_2)

        save = None
        if do_save:
            save = pp.Exporter(gb_2, "SIF_propagation_3D", folder="results",
                               fixed_grid=False)

        # Check for propagation
        faces, _ = pp.displacement_correlation.faces_to_open(gb_2, u,
                                                             critical_sifs,
                                                             rm_factor=rm_f)
        save_vtk(save, u, gb_2, 0)
        # Increase boundary displacement, rediscretize, solve and evaluate
        # propagation.
        lhs_2, rhs_2 = propagate_and_update(gb_2, faces, discr, set_bc_mech_2,
                                            update_apertures)

        u = sps.linalg.spsolve(lhs_2, rhs_2)
        faces, _ = pp.displacement_correlation.faces_to_open(gb_2, u,
                                                             critical_sifs,
                                                             rm_factor=rm_f)
        save_vtk(save, u, gb_2, 1)
        # Now the condition is met. Propagate and update discretizations
        lhs_2, rhs_2 = propagate_and_update(gb_2, faces, discr, set_bc_mech_2,
                                            update_apertures)
        u = sps.linalg.spsolve(lhs_2, rhs_2)
        faces, _ = pp.displacement_correlation.faces_to_open(gb_2, u,
                                                             critical_sifs,
                                                             rm_factor=rm_f)
        save_vtk(save, u, gb_2, 2)

        if do_save:
            save.write_pvd(np.arange(3))

        # The resulting fracture geometry is non-standard, so the check is
        # not as strict as in the other tests:
        g_l = gb_2.grids_of_dimension(2)[0]
        assert g_l.num_cells == 24
        assert g_l.num_faces == 60


def save_vtk(save, u, gb, i):
    if save is None:
        return
    g_h = gb.grids_of_dimension(gb.dim_max())[0]
    g_l = gb.grids_of_dimension(gb.dim_min())[0]
    d_h = gb.node_props(g_h)
    d_l = gb.node_props(g_l)
#    n_cells = g_h.num_cells + g_l.num_cells
    u_cells = u[:g_h.dim * g_h.num_cells]
    u_faces = u[g_h.dim * g_h.num_cells:]
#    u_extended = np.append(u_cells, np.zeros(g_l.num_cells * g_h.dim))
#        u_comp = u_cells[np.arange(0, g_h.dim * g_h.num_cells, 2)]
#        v_comp = u_cells[np.arange(1, g_h.dim * g_h.num_cells, 2)]
#    v_comp = u_extended[np.arange(1, u_extended.size, g_h.dim)]
    u_h = u_cells.reshape((g_h.num_cells, g_h.dim), order='F')
    u_l = np.mean(u_faces.reshape((2, -1)), axis=0)
    u_l = u_l.reshape((g_l.num_cells, g_h.dim), order='F')
    if g_h.dim == 2:
        u_h = np.vstack((u_h, np.zeros(g_h.num_cells)))
        u_l = np.vstack((u_l, np.zeros(g_h.num_cells)))

    d_h["displacement"] = u_h
    d_l["displacement"] = u_l
    save.write_vtk(["displacement"], i, grid=gb)


if __name__ == '__main__':
    unittest.main()
