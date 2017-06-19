from __future__ import division
import numpy as np
import unittest

from porepy.fracs import meshing
import porepy.utils.comp_geom as cg
from porepy.params import bc
from porepy.numerics.fv.transport import upwind, upwind_coupling
from porepy.numerics.mixed_dim import coupler

#------------------------------------------------------------------------------#

class BasicsTest( unittest.TestCase ):

#------------------------------------------------------------------------------#

    def test_upwind_2d_beta_positive(self):

        f = np.array([[2, 2], [0, 2]])
        gb = meshing.cart_grid([f], [4, 2])
        gb.assign_node_ordering()
        gb.compute_geometry()

        solver = upwind.Upwind()

        gb.add_node_props(['apertures', 'discharge'])
        for g, d in gb:
            d['apertures'] = 1e-1*np.ones(g.num_cells)
            d['discharge'] = solver.discharge(g, [2, 0, 0])

        gb.add_edge_prop('discharge')
        for e, d in gb.edges_props():
            gn = gb.sorted_nodes_of_edge(e)
            d['discharge'] = gb.node_props(gn[1])['discharge']

        coupling = upwind_coupling.UpwindCoupling(solver)
        solver_coupler = coupler.Coupler(solver, coupling)
        M = solver_coupler.matrix_rhs(gb)[0].todense()

        M_known = np.array([[ 2,  0,  0,  0,  0,  0,  0,  0,  0,  0.],
                            [-2,  2,  0,  0,  0,  0,  0,  0,  0,  0.],
                            [ 0,  0,  2,  0,  0,  0,  0,  0,  0, -2.],
                            [ 0,  0, -2,  0,  0,  0,  0,  0,  0,  0.],
                            [ 0,  0,  0,  0,  2,  0,  0,  0,  0,  0.],
                            [ 0,  0,  0,  0, -2,  2,  0,  0,  0,  0.],
                            [ 0,  0,  0,  0,  0,  0,  2,  0, -2,  0.],
                            [ 0,  0,  0,  0,  0,  0, -2,  0,  0,  0.],
                            [ 0,  0,  0,  0,  0, -2,  0,  0,  2,  0.],
                            [ 0, -2,  0,  0,  0,  0,  0,  0,  0,  2.]])

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_2d_full_beta_bc_dir(self):

        f = np.array([[2, 2], [0, 2]])
        gb = meshing.cart_grid([f], [4, 2])
        gb.assign_node_ordering()
        gb.compute_geometry()


        solver = upwind.Upwind()

        gb.add_node_props(['apertures', 'discharge', 'bc', 'bc_val'])
        for g, d in gb:
            d['apertures'] = np.ones(g.num_cells)*np.power(1e-1, 2-g.dim)
            d['discharge'] = solver.discharge(g, [1, 1, 0], d['apertures'])

            bound_faces = g.get_domain_boundary_faces()
            labels = np.array(['dir'] * bound_faces.size)
            d['bc'] = bc.BoundaryCondition(g, bound_faces, labels)
            d['bc_val'] = np.zeros(g.num_faces)
            d['bc_val'][bound_faces] = 3

        gb.add_edge_prop('discharge')
        for e, d in gb.edges_props():
            gn = gb.sorted_nodes_of_edge(e)
            d['discharge'] = gb.node_props(gn[1])['discharge']

        coupling = upwind_coupling.UpwindCoupling(solver)
        solver_coupler = coupler.Coupler(solver, coupling)
        M, rhs = solver_coupler.matrix_rhs(gb)

        M_known = np.array([[ 2,  0,  0,  0,  0,  0,  0,  0,  0,  0.],
                            [-1,  2,  0,  0,  0,  0,  0,  0,  0,  0.],
                            [ 0,  0,  2,  0,  0,  0,  0,  0,  0, -1.],
                            [ 0,  0, -1,  2,  0,  0,  0,  0,  0,  0.],
                            [-1,  0,  0,  0,  2,  0,  0,  0,  0,  0.],
                            [ 0, -1,  0,  0, -1,  2,  0,  0,  0,  0.],
                            [ 0,  0, -1,  0,  0,  0,  2,  0, -1,  0.],
                            [ 0,  0,  0, -1,  0,  0, -1,  2,  0,  0.],
                            [ 0,  0,  0,  0,  0, -1,  0,  0,1.1,-0.1],
                            [ 0, -1,  0,  0,  0,  0,  0,  0,  0, 1.1]])

        rhs_known = np.array([6, 3, 3, 3, 3, 0, 0, 0, 0, 0.3])

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M.todense(), M_known, rtol, atol)
        assert np.allclose(rhs, rhs_known, rtol, atol)

#------------------------------------------------------------------------------#
