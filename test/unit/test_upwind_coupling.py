from __future__ import division
import numpy as np
import unittest

from porepy.fracs import meshing
from porepy.params.bc import BoundaryCondition
from porepy.params.data import Parameters

from porepy.numerics.fv.transport import upwind
from porepy.numerics.mixed_dim import coupler

#------------------------------------------------------------------------------#


class BasicsTest(unittest.TestCase):

    #------------------------------------------------------------------------------#

    def test_upwind_coupling_2d_1d_bottom_top(self):
        f = np.array([[0, 1],
                      [1, 1]])
        gb = meshing.cart_grid([f], [1, 2])
        gb.compute_geometry()
        gb.assign_node_ordering()

        tol = 1e-3
        gb.add_node_props(['param'])
        solver = upwind.UpwindMixedDim('transport')

        for g, d in gb:
            param = Parameters(g)

            aperture = np.ones(g.num_cells) * \
                np.power(1e-2, gb.dim_max() - g.dim)
            param.set_aperture(aperture)
            d['discharge'] = solver.discr.discharge(g, [0, 1, 0], aperture)

            bound_faces = g.tags['domain_boundary_faces'].nonzero()[0]
            if bound_faces.size != 0:
                bound_face_centers = g.face_centers[:, bound_faces]

                top = bound_face_centers[1, :] > 2 - tol
                bottom = bound_face_centers[1, :] < tol

                labels = np.array(['neu'] * bound_faces.size)
                labels[np.logical_or(top, bottom)] = ['dir']

                bc_val = np.zeros(g.num_faces)
                bc_dir = bound_faces[np.logical_or(top, bottom)]
                bc_val[bc_dir] = 1

                param.set_bc('transport', BoundaryCondition(
                    g, bound_faces, labels))
                param.set_bc_val('transport', bc_val)

            d['param'] = param

        # Assign coupling discharge
        gb.add_edge_prop('param')
        for e, d in gb.edges_props():
            g_h = gb.sorted_nodes_of_edge(e)[1]
            discharge = gb.node_prop(g_h, 'discharge')
            d['param'] = Parameters(g_h)
            d['discharge'] = discharge

        U, rhs = solver.matrix_rhs(gb)
        deltaT = solver.cfl(gb)

        U_known = np.array([[1, 0,  0],
                            [0, 1, -1],
                            [-1, 0, 1]])
        rhs_known = np.array([1, 0, 0])

        deltaT_known = 5 * 1e-3

        rtol = 1e-15
        atol = rtol
        assert np.allclose(U.todense(), U_known, rtol, atol)
        assert np.allclose(rhs, rhs_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_coupling_2d_1d_left_right(self):
        f = np.array([[0, 1],
                      [1, 1]])
        gb = meshing.cart_grid([f], [1, 2])
        gb.compute_geometry()
        gb.assign_node_ordering()

        tol = 1e-3
        solver = upwind.UpwindMixedDim('transport')
        gb.add_node_props(['param'])

        for g, d in gb:
            param = Parameters(g)

            aperture = np.ones(g.num_cells) * \
                np.power(1e-2, gb.dim_max() - g.dim)
            param.set_aperture(aperture)
            d['discharge'] = solver.discr.discharge(g, [1, 0, 0], aperture)

            bound_faces = g.tags['domain_boundary_faces'].nonzero()[0]
            if bound_faces.size != 0:
                bound_face_centers = g.face_centers[:, bound_faces]

                left = bound_face_centers[0, :] < tol
                right = bound_face_centers[0, :] > 1 - tol

                labels = np.array(['neu'] * bound_faces.size)
                labels[np.logical_or(left, right)] = ['dir']

                bc_val = np.zeros(g.num_faces)
                bc_dir = bound_faces[np.logical_or(left, right)]
                bc_val[bc_dir] = 1

                param.set_bc('transport', BoundaryCondition(
                    g, bound_faces, labels))
                param.set_bc_val('transport', bc_val)

            d['param'] = param

        # Assign coupling discharge
        gb.add_edge_prop('param')
        for e, d in gb.edges_props():
            g_h = gb.sorted_nodes_of_edge(e)[1]
            discharge = gb.node_prop(g_h, 'discharge')
            d['param'] = Parameters(g_h)
            d['discharge'] = discharge

        U, rhs = solver.matrix_rhs(gb)
        deltaT = solver.cfl(gb)

        U_known = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1e-2]])
        rhs_known = np.array([1, 1, 1e-2])

        deltaT_known = 5 * 1e-1

        rtol = 1e-15
        atol = rtol
        assert np.allclose(U.todense(), U_known, rtol, atol)
        assert np.allclose(rhs, rhs_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_coupling_2d_1d_left_right_cross(self):
        f1 = np.array([[0, 2],
                       [1, 1]])
        f2 = np.array([[1, 1],
                       [0, 2]])

        gb = meshing.cart_grid([f1, f2], [2, 2])
        gb.compute_geometry()
        gb.assign_node_ordering()

        # Enforce node orderning because of Python 3.5 and 2.7.
        # Don't do it in general.
        cell_centers_1 = np.array([[1.50000000e+00, 5.00000000e-01],
                                   [1.00000000e+00, 1.00000000e+00],
                                   [-5.55111512e-17, 5.55111512e-17]])
        cell_centers_2 = np.array([[1.00000000e+00, 1.00000000e+00],
                                   [1.50000000e+00, 5.00000000e-01],
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
        solver = upwind.UpwindMixedDim('transport')
        gb.add_node_props(['param'])

        for g, d in gb:
            param = Parameters(g)

            aperture = np.ones(g.num_cells) * \
                np.power(1e-2, gb.dim_max() - g.dim)
            param.set_aperture(aperture)
            d['discharge'] = solver.discr.discharge(g, [1, 0, 0], aperture)

            bound_faces = g.tags['domain_boundary_faces'].nonzero()[0]
            if bound_faces.size != 0:
                bound_face_centers = g.face_centers[:, bound_faces]

                left = bound_face_centers[0, :] < tol
                right = bound_face_centers[0, :] > 2 - tol

                labels = np.array(['neu'] * bound_faces.size)
                labels[np.logical_or(left, right)] = ['dir']

                bc_val = np.zeros(g.num_faces)
                bc_dir = bound_faces[np.logical_or(left, right)]
                bc_val[bc_dir] = 1

                param.set_bc('transport', BoundaryCondition(
                    g, bound_faces, labels))
                param.set_bc_val('transport', bc_val)
            else:
                param.set_bc("transport", BoundaryCondition(
                    g, np.empty(0), np.empty(0)))
            d['param'] = param

        # Assign coupling discharge
        gb.add_edge_prop('param')
        for e, d in gb.edges_props():
            g_h = gb.sorted_nodes_of_edge(e)[1]
            discharge = gb.node_prop(g_h, 'discharge')
            d['param'] = Parameters(g_h)
            d['discharge'] = discharge

        U, rhs = solver.matrix_rhs(gb)
        deltaT = solver.cfl(gb)

        U_known = np.array(
            [[1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
             [0.,  1.,  0.,  0.,  0.,  0.,  0., -1.,  0.],
                [0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                [0.,  0.,  0.,  1.,  0.,  0., -1.,  0.,  0.],
                [0.,  0.,  0.,  0.,  0.01,  0.,  0.,  0., -0.01],
                [0.,  0.,  0.,  0.,  0.,  0.01,  0.,  0.,  0.],
                [0.,  0., -1.,  0.,  0.,  0.,  1.,  0.,  0.],
                [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
                [0.,  0.,  0.,  0.,  0., -0.01,  0.,  0.,  0.01]])

        rhs_known = np.array([1., 0., 1., 0., 0., 0.01, 0., 0.,  0.])

        deltaT_known = 5 * 1e-3

        rtol = 1e-15
        atol = rtol
        assert np.allclose(U.todense(), U_known, rtol, atol)
        assert np.allclose(rhs, rhs_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_coupling_3d_2d_bottom_top(self):
        f = np.array([[0,  1,  1,  0],
                      [0,  0,  1,  1],
                      [.5, .5, .5, .5]])
        gb = meshing.cart_grid([f], [1, 1, 2], **{'physdims': [1, 1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        tol = 1e-3
        solver = upwind.UpwindMixedDim('transport')
        gb.add_node_props(['param'])

        for g, d in gb:
            param = Parameters(g)

            aperture = np.ones(g.num_cells) * \
                np.power(1e-2, gb.dim_max() - g.dim)
            param.set_aperture(aperture)
            d['discharge'] = solver.discr.discharge(g, [0, 0, 1], aperture)

            bound_faces = g.tags['domain_boundary_faces'].nonzero()[0]
            if bound_faces.size != 0:
                bound_face_centers = g.face_centers[:, bound_faces]

                top = bound_face_centers[2, :] > 1 - tol
                bottom = bound_face_centers[2, :] < tol

                labels = np.array(['neu'] * bound_faces.size)
                labels[np.logical_or(top, bottom)] = ['dir']

                bc_val = np.zeros(g.num_faces)
                bc_dir = bound_faces[np.logical_or(top, bottom)]
                bc_val[bc_dir] = 1

                param.set_bc('transport', BoundaryCondition(
                    g, bound_faces, labels))
                param.set_bc_val('transport', bc_val)

            d['param'] = param

        # Assign coupling discharge
        gb.add_edge_prop('param')
        for e, d in gb.edges_props():
            g_h = gb.sorted_nodes_of_edge(e)[1]
            discharge = gb.node_prop(g_h, 'discharge')
            d['param'] = Parameters(g_h)
            d['discharge'] = discharge

        U, rhs = solver.matrix_rhs(gb)
        deltaT = solver.cfl(gb)

        U_known = np.array([[1, 0, 0],
                            [0, 1, -1],
                            [-1, 0, 1]])
        rhs_known = np.array([1, 0, 0])

        deltaT_known = 5 * 1e-3

        rtol = 1e-15
        atol = rtol
        assert np.allclose(U.todense(), U_known, rtol, atol)
        assert np.allclose(rhs, rhs_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_coupling_3d_2d_left_right(self):
        f = np.array([[0,  1,  1,  0],
                      [0,  0,  1,  1],
                      [.5, .5, .5, .5]])
        gb = meshing.cart_grid([f], [1, 1, 2], **{'physdims': [1, 1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        tol = 1e-3
        solver = upwind.UpwindMixedDim('transport')
        gb.add_node_props(['param'])

        for g, d in gb:
            param = Parameters(g)

            aperture = np.ones(g.num_cells) * \
                np.power(1e-2, gb.dim_max() - g.dim)
            param.set_aperture(aperture)
            d['discharge'] = solver.discr.discharge(g, [1, 0, 0], aperture)

            bound_faces = g.tags['domain_boundary_faces'].nonzero()[0]
            bound_face_centers = g.face_centers[:, bound_faces]

            left = bound_face_centers[0, :] < tol
            right = bound_face_centers[0, :] > 1 - tol

            labels = np.array(['neu'] * bound_faces.size)
            labels[np.logical_or(left, right)] = ['dir']

            bc_val = np.zeros(g.num_faces)
            bc_dir = bound_faces[np.logical_or(left, right)]
            bc_val[bc_dir] = 1

            param.set_bc('transport', BoundaryCondition(
                g, bound_faces, labels))
            param.set_bc_val('transport', bc_val)

            d['param'] = param

        # Assign coupling discharge
        gb.add_edge_prop('param')
        for e, d in gb.edges_props():
            g_h = gb.sorted_nodes_of_edge(e)[1]
            discharge = gb.node_prop(g_h, 'discharge')
            d['param'] = Parameters(g_h)
            d['discharge'] = discharge

        U, rhs = solver.matrix_rhs(gb)
        deltaT = solver.cfl(gb)

        U_known = np.array([[.5, 0, 0],
                            [0, .5, 0],
                            [0, 0, 1e-2]])
        rhs_known = np.array([.5, .5, 1e-2])

        deltaT_known = 5 * 1e-1

        rtol = 1e-15
        atol = rtol
        assert np.allclose(U.todense(), U_known, rtol, atol)
        assert np.allclose(rhs, rhs_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_coupling_3d_2d_1d_0d(self):
        f1 = np.array([[0,  1,  1,  0],
                       [0,  0,  1,  1],
                       [.5, .5, .5, .5]])
        f2 = np.array([[.5, .5, .5, .5],
                       [0,  1,  1,  0],
                       [0,  0,  1,  1]])
        f3 = np.array([[0,  1,  1,  0],
                       [.5, .5, .5, .5],
                       [0,  0,  1,  1]])

        gb = meshing.cart_grid([f1, f2, f3], [2, 2, 2],
                               **{'physdims': [1, 1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        cell_centers1 = np.array([[0.25, 0.75, 0.25, 0.75],
                                  [0.25, 0.25, 0.75, 0.75],
                                  [0.5, 0.5, 0.5, 0.5]])
        cell_centers2 = np.array([[0.5, 0.5, 0.5, 0.5],
                                  [0.25, 0.25, 0.75, 0.75],
                                  [0.75, 0.25, 0.75, 0.25]])
        cell_centers3 = np.array([[0.25, 0.75, 0.25, 0.75],
                                  [0.5, 0.5, 0.5, 0.5],
                                  [0.25, 0.25, 0.75, 0.75]])
        cell_centers4 = np.array([[0.5], [0.25], [0.5]])
        cell_centers5 = np.array([[0.5], [0.75], [0.5]])
        cell_centers6 = np.array([[0.75], [0.5], [0.5]])
        cell_centers7 = np.array([[0.25], [0.5], [0.5]])
        cell_centers8 = np.array([[0.5], [0.5], [0.25]])
        cell_centers9 = np.array([[0.5], [0.5], [0.75]])

        for g, d in gb:
            if np.allclose(g.cell_centers[:, 0], cell_centers1[:, 0]):
                d['node_number'] = 1
            elif np.allclose(g.cell_centers[:, 0], cell_centers2[:, 0]):
                d['node_number'] = 2
            elif np.allclose(g.cell_centers[:, 0], cell_centers3[:, 0]):
                d['node_number'] = 3
            elif np.allclose(g.cell_centers[:, 0], cell_centers4[:, 0]):
                d['node_number'] = 4
            elif np.allclose(g.cell_centers[:, 0], cell_centers5[:, 0]):
                d['node_number'] = 5
            elif np.allclose(g.cell_centers[:, 0], cell_centers6[:, 0]):
                d['node_number'] = 6
            elif np.allclose(g.cell_centers[:, 0], cell_centers7[:, 0]):
                d['node_number'] = 7
            elif np.allclose(g.cell_centers[:, 0], cell_centers8[:, 0]):
                d['node_number'] = 8
            elif np.allclose(g.cell_centers[:, 0], cell_centers9[:, 0]):
                d['node_number'] = 9
            else:
                pass

        tol = 1e-3
        solver = upwind.UpwindMixedDim('transport')
        gb.add_node_props(['param'])

        for g, d in gb:
            param = Parameters(g)

            aperture = np.ones(g.num_cells) * \
                np.power(1e-2, gb.dim_max() - g.dim)
            param.set_aperture(aperture)
            d['discharge'] = solver.discr.discharge(g, [1, 0, 0], aperture)
            bound_faces = g.tags['domain_boundary_faces'].nonzero()[0]
            if bound_faces.size != 0:

                bound_face_centers = g.face_centers[:, bound_faces]

                left = bound_face_centers[0, :] < tol
                right = bound_face_centers[0, :] > 1 - tol

                labels = np.array(['neu'] * bound_faces.size)
                labels[np.logical_or(left, right)] = ['dir']

                bc_val = np.zeros(g.num_faces)
                bc_dir = bound_faces[np.logical_or(left, right)]
                bc_val[bc_dir] = 1

                param.set_bc('transport', BoundaryCondition(
                    g, bound_faces, labels))
                param.set_bc_val('transport', bc_val)

            d['param'] = param

        # Assign coupling discharge
        gb.add_edge_prop('param')
        for e, d in gb.edges_props():
            g_h = gb.sorted_nodes_of_edge(e)[1]
            discharge = gb.node_prop(g_h, 'discharge')
            d['param'] = Parameters(g_h)
            d['discharge'] = discharge

        U, rhs = solver.matrix_rhs(gb)
        deltaT = solver.cfl(gb)

        U_known, rhs_known = matrix_rhs_for_test_upwind_coupling_3d_2d_1d_0d()

        deltaT_known = 5 * 1e-3

        rtol = 1e-15
        atol = rtol
        assert np.allclose(U.todense(), U_known, rtol, atol)
        assert np.allclose(rhs, rhs_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_2d_beta_positive(self):

        f = np.array([[2, 2],
                      [0, 2]])
        gb = meshing.cart_grid([f], [4, 2])
        gb.assign_node_ordering()
        gb.compute_geometry()

        solver = upwind.UpwindMixedDim('transport')

        gb.add_node_props(['param'])

        for g, d in gb:
            param = Parameters(g)
            aperture = np.ones(g.num_cells) * \
                np.power(1e-2, gb.dim_max() - g.dim)
            param.set_aperture(aperture)
            d['discharge'] = solver.discr.discharge(g, [2, 0, 0], aperture)

            bf = g.tags['domain_boundary_faces'].nonzero()[0]
            bc = BoundaryCondition(g, bf, bf.size * ['neu'])
            param.set_bc('transport', bc)
            d['param'] = param

        # Assign coupling discharge
        gb.add_edge_prop('param')
        for e, d in gb.edges_props():
            g_h = gb.sorted_nodes_of_edge(e)[1]
            discharge = gb.node_prop(g_h, 'discharge')
            d['param'] = Parameters(g_h)
            d['discharge'] = discharge

        M = solver.matrix_rhs(gb)[0].todense()

        M_known = np.array([[2,  0,  0,  0,  0,  0,  0,  0,  0,  0.],
                            [-2,  2,  0,  0,  0,  0,  0,  0,  0,  0.],
                            [0,  0,  2,  0,  0,  0,  0,  0,  0, -2.],
                            [0,  0, -2,  0,  0,  0,  0,  0,  0,  0.],
                            [0,  0,  0,  0,  2,  0,  0,  0,  0,  0.],
                            [0,  0,  0,  0, -2,  2,  0,  0,  0,  0.],
                            [0,  0,  0,  0,  0,  0,  2,  0, -2,  0.],
                            [0,  0,  0,  0,  0,  0, -2,  0,  0,  0.],
                            [0,  0,  0,  0,  0, -2,  0,  0,  2,  0.],
                            [0, -2,  0,  0,  0,  0,  0,  0,  0,  2.]])

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M, M_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_2d_full_beta_bc_dir(self):

        f = np.array([[2, 2], [0, 2]])
        gb = meshing.cart_grid([f], [4, 2])
        gb.assign_node_ordering()
        gb.compute_geometry()

        solver = upwind.UpwindMixedDim('transport')
        gb.add_node_props(['param'])

        for g, d in gb:
            param = Parameters(g)

            aperture = np.ones(g.num_cells) * \
                np.power(1e-1, gb.dim_max() - g.dim)
            param.set_aperture(aperture)
            d['discharge'] = solver.discr.discharge(g, [1, 1, 0], aperture)

            bound_faces = g.tags['domain_boundary_faces'].nonzero()[0]
            labels = np.array(['dir'] * bound_faces.size)
            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces] = 3

            param.set_bc('transport', BoundaryCondition(
                g, bound_faces, labels))
            param.set_bc_val('transport', bc_val)

            d['param'] = param

        # Assign coupling discharge
        gb.add_edge_prop('param')
        for e, d in gb.edges_props():
            g_h = gb.sorted_nodes_of_edge(e)[1]
            discharge = gb.node_prop(g_h, 'discharge')
            d['param'] = Parameters(g_h)
            d['discharge'] = discharge

        M, rhs = solver.matrix_rhs(gb)

        M_known = np.array([[2,  0,  0,  0,  0,  0,  0,  0,  0,  0.],
                            [-1,  2,  0,  0,  0,  0,  0,  0,  0,  0.],
                            [0,  0,  2,  0,  0,  0,  0,  0,  0, -1.],
                            [0,  0, -1,  2,  0,  0,  0,  0,  0,  0.],
                            [-1,  0,  0,  0,  2,  0,  0,  0,  0,  0.],
                            [0, -1,  0,  0, -1,  2,  0,  0,  0,  0.],
                            [0,  0, -1,  0,  0,  0,  2,  0, -1,  0.],
                            [0,  0,  0, -1,  0,  0, -1,  2,  0,  0.],
                            [0,  0,  0,  0,  0, -1,  0,  0, 1.1, -0.1],
                            [0, -1,  0,  0,  0,  0,  0,  0,  0, 1.1]])

        rhs_known = np.array([6, 3, 3, 3, 3, 0, 0, 0, 0, 0.3])

        rtol = 1e-15
        atol = rtol
        assert np.allclose(M.todense(), M_known, rtol, atol)
        assert np.allclose(rhs, rhs_known, rtol, atol)

#------------------------------------------------------------------------------#


def matrix_rhs_for_test_upwind_coupling_3d_2d_1d_0d():
    U = np.array([
        [2.50e-01,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00],
        [0.00e+00,   2.50e-01,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,  -2.50e-01,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00],
        [0.00e+00,   0.00e+00,   2.50e-01,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00],
        [0.00e+00,   0.00e+00,   0.00e+00,   2.50e-01,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         -2.50e-01,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00],
        [0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   2.50e-01,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00],
        [0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         2.50e-01,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,  -2.50e-01,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00],
        [0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   2.50e-01,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00],
        [0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   2.50e-01,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,  -2.50e-01,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00],
        [0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   5.00e-03,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00],
        [0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   5.00e-03,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         -5.00e-03,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00],
        [0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         5.00e-03,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00],
        [0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   5.00e-03,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,  -5.00e-03,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00],
        [0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,  -2.50e-01,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   2.50e-01,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00],
        [-2.50e-01,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   2.50e-01,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         -5.55e-19,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00],
        [0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,  -2.50e-01,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   2.50e-01,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00],
        [0.00e+00,   0.00e+00,  -2.50e-01,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         2.50e-01,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,  -5.55e-19,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00],
        [0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   5.00e-03,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00],
        [0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   5.00e-03,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,  -5.00e-03,
         0.00e+00,   0.00e+00],
        [0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   5.00e-03,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00],
        [0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   5.00e-03,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         -5.00e-03,   0.00e+00],
        [0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,  -5.00e-03,   0.00e+00,
         0.00e+00,   0.00e+00,  -5.55e-19,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         5.00e-03,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00],
        [0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         -5.00e-03,   0.00e+00,   0.00e+00,   0.00e+00,  -5.55e-19,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   5.00e-03,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00],
        [0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   1.00e-04,   0.00e+00,   0.00e+00,
         0.00e+00,  -1.00e-04],
        [0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   1.00e-04,   0.00e+00,
         0.00e+00,   0.00e+00],
        [0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,  -5.00e-03,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   5.00e-03,
         0.00e+00,   0.00e+00],
        [0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,  -5.00e-03,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         5.00e-03,   0.00e+00],
        [0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,  -1.00e-04,   0.00e+00,
         0.00e+00,   1.00e-04]])
    rhs = np.array(
        [2.50e-01,   0.00e+00,   2.50e-01,   0.00e+00,   2.50e-01,
         0.00e+00,   2.50e-01,   0.00e+00,   5.00e-03,   0.00e+00,
         5.00e-03,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   5.00e-03,   0.00e+00,   5.00e-03,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   1.00e-04,   0.00e+00,
         0.00e+00,   0.00e+00])

    return U, rhs

#------------------------------------------------------------------------------#
