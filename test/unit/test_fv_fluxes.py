"""
Tests for the FV flux back-computation for grid buckets.
Solves simple pressure problems, computes all fluxes on gb nodes and edges
and compares to known solutions.
"""
from __future__ import division
import numpy as np
import scipy.sparse as sps
import unittest

import porepy as pp
from porepy.numerics.mixed_dim import coupler, condensation

#------------------------------------------------------------------------------#


class BasicsTest(unittest.TestCase):

    #------------------------------------------------------------------------------#

    def test_tpfa_fluxes_2d_1d_left_right_dir_neu(self):
        """
        Grid: 2 x 2 cells in matrix + 2 cells in the fracture from left to right.
        Dirichlet + inflow + no-flow, conductive fracture.
        Tests pressure solution and fluxes.
        """
        f = np.array([[0, 1],
                      [.5, .5]])
        gb = pp.meshing.cart_grid([f], [2, 2], **{'physdims': [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        tol = 1e-3
        solver = pp.TpfaMixedDim(physics='flow')
        gb.add_node_props(['param'])
        a = 1e-2
        for g, d in gb:
            param = pp.Parameters(g)

            a_dim = np.power(a, gb.dim_max() - g.dim)
            aperture = np.ones(g.num_cells) * a_dim
            param.set_aperture(aperture)

            p = pp.SecondOrder(3, np.ones(
                g.num_cells) * np.power(1e-3, g.dim < gb.dim_max()))
            param.set_tensor('flow', p)
            bound_faces = g.tags['domain_boundary_faces'].nonzero()[0]
            bound_face_centers = g.face_centers[:, bound_faces]

            right = bound_face_centers[0, :] > 1 - tol
            left = bound_face_centers[0, :] < tol

            labels = np.array(['neu'] * bound_faces.size)
            labels[right] = ['dir']

            bc_val = np.zeros(g.num_faces)
            bc_dir = bound_faces[right]
            bc_neu = bound_faces[left]
            bc_val[bc_dir] = g.face_centers[0, bc_dir]
            bc_val[bc_neu] = -g.face_areas[bc_neu] * a_dim

            param.set_bc('flow', pp.BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val('flow', bc_val)

            d['param'] = param

        gb.add_edge_prop('param')
        for e, d in gb.edges_props():
            g_h = gb.nodes_of_edge(e)[1]
            d['param'] = pp.Parameters(g_h)

        A, rhs = solver.matrix_rhs(gb)
        p = sps.linalg.spsolve(A, rhs)
        solver.split(gb, "pressure", p)
        pp.fvutils.compute_discharges(gb)

        p_known = np.array([1.7574919,  1.25249747,  1.7574919,  1.25249747,
                            1.25250298,  1.80993337])

        # Known discharges
        d_0, d_1 = fluxes_2d_1d_left_right_dir_neu()

        rtol = 1e-6
        atol = rtol

        for _, d in gb:
            if d['node_number'] == 0:
                assert np.allclose(d['discharge'], d_0, rtol, atol)
            if d['node_number'] == 1:
                assert np.allclose(d['discharge'], d_1, rtol, atol)

        assert np.allclose(p, p_known, rtol, atol)

        #------------------------------------------------------------------------------#

    def test_mpfa_fluxes_2d_1d_left_right_dir_neu(self):
        """
        Grid: 2 x 2 cells in matrix + 2 cells in the fracture from left to right.
        Dirichlet + inflow + no-flow, conductive fracture.
        Tests pressure solution and fluxes.
        """
        f = np.array([[0, 1],
                      [.5, .5]])
        gb = pp.meshing.cart_grid([f], [2, 2], **{'physdims': [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        tol = 1e-3
        solver = pp.MpfaMixedDim(physics='flow')
        gb.add_node_props(['param'])
        a = 1e-2
        for g, d in gb:
            param = pp.Parameters(g)

            a_dim = np.power(a, gb.dim_max() - g.dim)
            aperture = np.ones(g.num_cells) * a_dim
            param.set_aperture(aperture)

            p = pp.SecondOrder(3, np.ones(
                g.num_cells) * np.power(1e-3, g.dim < gb.dim_max()))
            param.set_tensor('flow', p)
            bound_faces = g.tags['domain_boundary_faces'].nonzero()[0]
            bound_face_centers = g.face_centers[:, bound_faces]

            right = bound_face_centers[0, :] > 1 - tol
            left = bound_face_centers[0, :] < tol

            labels = np.array(['neu'] * bound_faces.size)
            labels[right] = ['dir']

            bc_val = np.zeros(g.num_faces)
            bc_dir = bound_faces[right]
            bc_neu = bound_faces[left]
            bc_val[bc_dir] = g.face_centers[0, bc_dir]
            bc_val[bc_neu] = -g.face_areas[bc_neu] * a_dim

            param.set_bc('flow', pp.BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val('flow', bc_val)

            d['param'] = param

        gb.add_edge_prop('param')
        for e, d in gb.edges_props():
            g_h = gb.nodes_of_edge(e)[1]
            d['param'] = pp.Parameters(g_h)

        A, rhs = solver.matrix_rhs(gb)
        p = sps.linalg.spsolve(A, rhs)
        solver.solver.split(gb, "pressure", p)
        pp.fvutils.compute_discharges(gb)

        p_known = np.array([1.7574919,  1.25249747,  1.7574919,  1.25249747,
                            1.25250298,  1.80993337])

        # Known discharges
        d_0, d_1 = fluxes_2d_1d_left_right_dir_neu()

        rtol = 1e-6
        atol = rtol

        for _, d in gb:

            if d['node_number'] == 0:
                assert np.allclose(d['discharge'], d_0, rtol, atol)
            if d['node_number'] == 1:
                assert np.allclose(d['discharge'], d_1, rtol, atol)

        assert np.allclose(p, p_known, rtol, atol)

    def test_tpfa_fluxes_2d_1d_cross_with_elimination(self):
        f1 = np.array([[0, 1],
                       [.5, .5]])
        f2 = np.array([[.5, .5],
                       [0, 1]])
        mesh_size = 0.4
        mesh_kwargs = {}
        mesh_kwargs['mesh_size'] = {'mode': 'constant',
                                    'value': mesh_size, 'bound_value': mesh_size}
        gb = pp.meshing.cart_grid([f1, f2], [2, 2], **{'physdims': [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        # Enforce node orderning because of Python 3.5 and 2.7.
        # Don't do it in general.
        cell_centers_1 = np.array([[7.50000000e-01, 2.500000000e-01],
                                   [5.00000000e-01, 5.00000000e-01],
                                   [-5.55111512e-17, 5.55111512e-17]])
        cell_centers_2 = np.array([[5.00000000e-01, 5.00000000e-01],
                                   [7.50000000e-01, 2.500000000e-01],
                                   [-5.55111512e-17, 5.55111512e-17]])

        for g, d in gb:
            if g.dim == 1:
                if np.allclose(g.cell_centers, cell_centers_1):
                    d['node_number'] = 1
                elif np.allclose(g.cell_centers, cell_centers_2):
                    d['node_number'] = 2
                else:
                    raise ValueError('Grid not found')

        tol = 1e-3
        solver = pp.TpfaMixedDim('flow')
        gb.add_node_props(['param'])
        a = 1e-2
        for g, d in gb:
            param = pp.Parameters(g)

            a_dim = np.power(a, gb.dim_max() - g.dim)
            aperture = np.ones(g.num_cells) * a_dim
            param.set_aperture(aperture)

            kxx = np.ones(g.num_cells) * np.power(1e3, g.dim < gb.dim_max())
            p = pp.SecondOrder(3, kxx, kyy=kxx, kzz=kxx)
            param.set_tensor('flow', p)

            bound_faces = g.tags['domain_boundary_faces'].nonzero()[0]
            if bound_faces.size != 0:
                bound_face_centers = g.face_centers[:, bound_faces]

                right = bound_face_centers[0, :] > 1 - tol
                left = bound_face_centers[0, :] < tol

                labels = np.array(['neu'] * bound_faces.size)
                labels[right] = ['dir']

                bc_val = np.zeros(g.num_faces)
                bc_dir = bound_faces[right]
                bc_neu = bound_faces[left]
                bc_val[bc_dir] = g.face_centers[0, bc_dir]
                bc_val[bc_neu] = -g.face_areas[bc_neu] * a_dim

                param.set_bc('flow', pp.BoundaryCondition(
                    g, bound_faces, labels))
                param.set_bc_val('flow', bc_val)
            else:
                param.set_bc("flow", pp.BoundaryCondition(
                    g, np.empty(0), np.empty(0)))
            d['param'] = param

        gb.add_edge_prop('param')
        for e, d in gb.edges_props():
            g_h = gb.nodes_of_edge(e)[1]
            d['param'] = pp.Parameters(g_h)

        A, rhs = solver.matrix_rhs(gb)
        p = sps.linalg.spsolve(A, rhs)

        p = sps.linalg.spsolve(A, rhs)
        p_cond, p_red, _, _ = condensation.solve_static_condensation(
            A, rhs, gb, dim=0)

        solver.split(gb, "p_cond", p_cond)
        solver.split(gb, "pressure", p)

        # Make a copy of the grid bucket without the 0d grid
        dim_to_remove = 0
        gb_r, elimination_data = gb.duplicate_without_dimension(dim_to_remove)
        # Compute the flux discretization on the new edges
        condensation.compute_elimination_fluxes(gb, gb_r, elimination_data)
        # Compute the discharges from the flux discretizations and computed
        # pressures
        solver.split(gb_r, "pressure", p_red)
        pp.fvutils.compute_discharges(gb)
        pp.fvutils.compute_discharges(gb_r)

        # Known discharges
        d_0, d_1, d_2 = fluxes_2d_1d_cross_with_elimination()

        # Check node fluxes, ...
        rtol = 1e-6
        atol = rtol
        for g, d in gb:
            if d['node_number'] == 0:
                assert np.allclose(d['discharge'], d_0, rtol, atol)
            if d['node_number'] == 1:
                assert np.allclose(d['discharge'], d_1, rtol, atol)
            if d['node_number'] == 2:
                assert np.allclose(d['discharge'], d_2, rtol, atol)
        for g, d in gb_r:

            if d['node_number'] == 0:
                assert np.allclose(d['discharge'], d_0, rtol, atol)
            if d['node_number'] == 1:
                assert np.allclose(d['discharge'], d_1, rtol, atol)
            if d['node_number'] == 2:
                assert np.allclose(d['discharge'], d_2, rtol, atol)

        # ... edge fluxes ...
        d_01, d_10, d_02, d_20, d_13, d_23 \
            = coupling_fluxes_2d_1d_cross_no_el()

        for e, data in gb.edges_props():
            g1, g2 = gb.nodes_of_edge(e)
            pa = data['param']
            node_numbers = gb.nodes_prop([g2, g1], 'node_number')
            if pa is not None:

                if node_numbers == (0, 1):
                    assert np.allclose(data['discharge'], d_01, rtol, atol) or \
                        np.allclose(data['discharge'], d_10, rtol, atol)
                if node_numbers == (0, 2):
                    assert np.allclose(data['discharge'], d_02, rtol, atol) or \
                        np.allclose(data['discharge'], d_20, rtol, atol)
                if node_numbers == (1, 3):
                    assert np.allclose(data['discharge'], d_13, rtol, atol)
                if node_numbers == (2, 3):
                    assert np.allclose(data['discharge'], d_23, rtol, atol)

        d_11, d_21, d_22 = coupling_fluxes_2d_1d_cross_with_el()
        for e, data in gb_r.edges_props():
            g1, g2 = gb_r.nodes_of_edge(e)
            pa = data['param']
            node_numbers = gb_r.nodes_prop([g2, g1], 'node_number')
            if pa is not None:

                if node_numbers == (0, 1):
                    assert np.allclose(data['discharge'], d_01, rtol, atol) or \
                        np.allclose(data['discharge'], d_10, rtol, atol)
                if node_numbers == (0, 2):
                    assert np.allclose(data['discharge'], d_02, rtol, atol) or \
                        np.allclose(data['discharge'], d_20, rtol, atol)
                if node_numbers == (1, 1):
                    assert np.allclose(data['discharge'], d_11, rtol, atol)
                if node_numbers == (2, 1):
                    assert np.allclose(data['discharge'], d_21, rtol, atol)
                if node_numbers == (2, 2):
                    assert np.allclose(data['discharge'], d_22, rtol, atol)
        # ... and pressures
        tol = 1e-10
        assert((np.amax(np.absolute(p - p_cond))) < tol)
        assert(np.sum(pp.error.error_L2(
            g, d['pressure'], d['p_cond']) for g, d in gb) < tol)


# #------------------------------------------------------------------------------#
def fluxes_2d_1d_left_right_dir_neu():
    d_0 = np.array([5.00000000e-01,   5.04994426e-01,   5.04994950e-01,
                    5.00000000e-01,   5.04994426e-01,   5.04994950e-01,
                    0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                    0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                    0.00000000e+00,  0.00000000e+00])
    d_1 = np.array([-1.01001192e-05,  -1.11486078e-05,  -1.00000000e-02])
    return d_0, d_1
#------------------------------------------------------------------------------#


def fluxes_2d_1d_cross_with_elimination():
    d_0 = np.array([0.5,  0.,  0.04923282,  0.5,  0.,
                    0.04923282,  0.,  0.,  0.,  0.,
                    0.,  0.,  0.,  0.,  0.,  0.])

    d_1 = np.array([-0.91153437,  0., -0.01,  0.])

    d_2 = np.array([0.,  0.,  0.,  0.])
    return d_0, d_1, d_2
#------------------------------------------------------------------------------#


def coupling_fluxes_2d_1d_cross_no_el():
    d_01 = np.array([0.,  0.,  0.,  0.,  0.,
                     0.,  0.,  0., -0.24258847, -0.00365602,
                     0.,  0.,  0.24258847,  0.00365602,  0.,
                     0.])
    d_10 = np.array([0.,  0.,  0.,  0.,  0.,
                     0.,  0.,  0., -0.24258847, -0.00365602,
                     0.,  0.,  0.,  0.,  0.24258847,
                     0.00365602])

    d_02 = np.array([0.,  0.05288884,  0.,  0.,  0.05288884,
                     0.,  0.,  0.,  0.,  0.,
                     0.,  0.,  0.,  0.,  0.25741153,
                     0.25741153])
    d_20 = np.array([0.,  0.05288884,  0.,  0.,  0.05288884,
                     0.,  0.,  0.,  0.,  0.,
                     0.,  0.,  0.25741153,  0.25741153,  0.,
                     0.])

    d_13 = np.array([0., -0.49517693,  0., -0.90422232])

    d_23 = np.array([0., -0.20452269,  0.,  0.20452269])

    return d_01, d_10, d_02, d_20, d_13, d_23
#------------------------------------------------------------------------------#


def coupling_fluxes_2d_1d_cross_with_el():
    d_11 = np.array([0.34984981])

    d_21 = np.array([0.27718625,  0.27718625, -0.07266356, -0.07266356])

    d_22 = np.array([0.00000000e+00])

    return d_11, d_21, d_22
#------------------------------------------------------------------------------#
