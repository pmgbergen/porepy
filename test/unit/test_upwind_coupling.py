from __future__ import division
import numpy as np
import unittest

from porepy.fracs import meshing
import porepy.utils.comp_geom as cg
from porepy.params.bc import BoundaryCondition
from porepy.params.data import Parameters

from porepy.numerics.fv.transport import upwind, upwind_coupling
from porepy.numerics.mixed_dim import coupler

#------------------------------------------------------------------------------#

class BasicsTest( unittest.TestCase ):

#------------------------------------------------------------------------------#

    def test_upwind_coupling_2d_1d_bottom_top(self):
        f = np.array([[0, 1],
                      [1, 1]])
        gb = meshing.cart_grid( [f], [1, 2])
        gb.compute_geometry()
        gb.assign_node_ordering()

        tol = 1e-3
        solver = upwind.Upwind()
        gb.add_node_props(['param'])

        for g, d in gb:
            param = Parameters(g)

            aperture = np.ones(g.num_cells)*np.power(1e-2, gb.dim_max() - g.dim)
            param.set_aperture(aperture)
            param.set_discharge(solver.discharge(g, [0, 1, 0], aperture))

            bound_faces = g.get_boundary_faces()
            bound_face_centers = g.face_centers[:, bound_faces]

            top = bound_face_centers[1, :] > 2 - tol
            bottom = bound_face_centers[1, :] < tol

            labels = np.array(['neu'] * bound_faces.size)
            labels[np.logical_or(top, bottom)] = ['dir']

            bc_val = np.zeros(g.num_faces)
            bc_dir = bound_faces[np.logical_or(top, bottom)]
            bc_val[bc_dir] = 1

            param.set_bc(solver, BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val(solver, bc_val)

            d['param'] = param

        # Assign coupling discharge
        gb.add_edge_prop('param')
        for e, d in gb.edges_props():
            g_h = gb.sorted_nodes_of_edge(e)[1]
            discharge = gb.node_prop(g_h, 'param').get_discharge()
            d['param'] = Parameters(g_h)
            d['param'].set_discharge(discharge)

        coupling_conditions = upwind_coupling.UpwindCoupling(solver)
        solver_coupler = coupler.Coupler(solver, coupling_conditions)
        U, rhs = solver_coupler.matrix_rhs(gb)
        deltaT = np.amin(gb.loop(solver.cfl, coupling_conditions.cfl).data)

        U_known = np.array([[ 1, 0,  0],
                            [ 0, 1, -1],
                            [-1, 0, 1]])
        rhs_known = np.array([1, 0, 0])

        deltaT_known = 5*1e-3

        rtol = 1e-15
        atol = rtol
        assert np.allclose(U.todense(), U_known, rtol, atol)
        assert np.allclose(rhs, rhs_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_coupling_2d_1d_left_right(self):
        f = np.array([[0, 1],
                      [1, 1]])
        gb = meshing.cart_grid( [f], [1, 2])
        gb.compute_geometry()
        gb.assign_node_ordering()

        tol = 1e-3
        solver = upwind.Upwind()
        gb.add_node_props(['param'])

        for g, d in gb:
            param = Parameters(g)

            aperture = np.ones(g.num_cells)*np.power(1e-2, gb.dim_max() - g.dim)
            param.set_aperture(aperture)
            param.set_discharge(solver.discharge(g, [1, 0, 0], aperture))

            bound_faces = g.get_boundary_faces()
            bound_face_centers = g.face_centers[:, bound_faces]

            left = bound_face_centers[0, :] < tol
            right = bound_face_centers[0, :] > 1 - tol

            labels = np.array(['neu'] * bound_faces.size)
            labels[np.logical_or(left, right)] = ['dir']

            bc_val = np.zeros(g.num_faces)
            bc_dir = bound_faces[np.logical_or(left, right)]
            bc_val[bc_dir] = 1

            param.set_bc(solver, BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val(solver, bc_val)

            d['param'] = param

        # Assign coupling discharge
        gb.add_edge_prop('param')
        for e, d in gb.edges_props():
            g_h = gb.sorted_nodes_of_edge(e)[1]
            discharge = gb.node_prop(g_h, 'param').get_discharge()
            d['param'] = Parameters(g_h)
            d['param'].set_discharge(discharge)

        coupling_conditions = upwind_coupling.UpwindCoupling(solver)
        solver_coupler = coupler.Coupler(solver, coupling_conditions)
        U, rhs = solver_coupler.matrix_rhs(gb)
        deltaT = np.amin(gb.loop(solver.cfl, coupling_conditions.cfl).data)

        U_known = np.array([[ 1, 0, 0],
                            [ 0, 1, 0],
                            [ 0, 0, 1e-2]])
        rhs_known = np.array([1, 1, 1e-2])

        deltaT_known = 5*1e-1

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

        gb = meshing.cart_grid( [f1, f2], [2, 2])
        gb.compute_geometry()
        gb.assign_node_ordering()

        tol = 1e-3
        solver = upwind.Upwind()
        gb.add_node_props(['param'])

        for g, d in gb:
            param = Parameters(g)

            aperture = np.ones(g.num_cells)*np.power(1e-2, gb.dim_max() - g.dim)
            param.set_aperture(aperture)
            param.set_discharge(solver.discharge(g, [1, 0, 0], aperture))

            bound_faces = g.get_boundary_faces()
            bound_face_centers = g.face_centers[:, bound_faces]

            left = bound_face_centers[0, :] < tol
            right = bound_face_centers[0, :] > 2 - tol

            labels = np.array(['neu'] * bound_faces.size)
            labels[np.logical_or(left, right)] = ['dir']

            bc_val = np.zeros(g.num_faces)
            bc_dir = bound_faces[np.logical_or(left, right)]
            bc_val[bc_dir] = 1

            param.set_bc(solver, BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val(solver, bc_val)

            d['param'] = param

        # Assign coupling discharge
        gb.add_edge_prop('param')
        for e, d in gb.edges_props():
            g_h = gb.sorted_nodes_of_edge(e)[1]
            discharge = gb.node_prop(g_h, 'param').get_discharge()
            d['param'] = Parameters(g_h)
            d['param'].set_discharge(discharge)

        coupling_conditions = upwind_coupling.UpwindCoupling(solver)
        solver_coupler = coupler.Coupler(solver, coupling_conditions)
        U, rhs = solver_coupler.matrix_rhs(gb)

        deltaT = np.amin(gb.loop(solver.cfl, coupling_conditions.cfl).data)

        U_known = np.array(\
        [[ 1.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
         [ 0.  ,  1.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  , -1.  ,  0.  ],
         [ 0.  ,  0.  ,  1.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
         [ 0.  ,  0.  ,  0.  ,  1.  ,  0.  ,  0.  , -1.  ,  0.  ,  0.  ],
         [ 0.  ,  0.  ,  0.  ,  0.  ,  0.01,  0.  ,  0.  ,  0.  , -0.01],
         [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.01,  0.  ,  0.  ,  0.  ],
         [ 0.  ,  0.  , -1.  ,  0.  ,  0.  ,  0.  ,  1.  ,  0.  ,  0.  ],
         [-1.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  1.  ,  0.  ],
         [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  , -0.01,  0.  ,  0.  ,  0.01]])

        rhs_known = np.array([ 1., 0., 1., 0., 0., 0.01, 0., 0.,  0.])

        deltaT_known = 5*1e-3

        rtol = 1e-15
        atol = rtol
        assert np.allclose(U.todense(), U_known, rtol, atol)
        assert np.allclose(rhs, rhs_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_coupling_3d_2d_bottom_top(self):
        f = np.array([[ 0,  1,  1,  0],
                      [ 0,  0,  1,  1],
                      [.5, .5, .5, .5]])
        gb = meshing.cart_grid( [f], [1, 1, 2], **{'physdims': [1, 1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        tol = 1e-3
        solver = upwind.Upwind()
        gb.add_node_props(['param'])

        for g, d in gb:
            param = Parameters(g)

            aperture = np.ones(g.num_cells)*np.power(1e-2, gb.dim_max() - g.dim)
            param.set_aperture(aperture)
            param.set_discharge(solver.discharge(g, [0, 0, 1], aperture))

            bound_faces = g.get_boundary_faces()
            bound_face_centers = g.face_centers[:, bound_faces]

            top = bound_face_centers[2, :] > 1 - tol
            bottom = bound_face_centers[2, :] < tol

            labels = np.array(['neu'] * bound_faces.size)
            labels[np.logical_or(top, bottom)] = ['dir']

            bc_val = np.zeros(g.num_faces)
            bc_dir = bound_faces[np.logical_or(top, bottom)]
            bc_val[bc_dir] = 1

            param.set_bc(solver, BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val(solver, bc_val)

            d['param'] = param

        # Assign coupling discharge
        gb.add_edge_prop('param')
        for e, d in gb.edges_props():
            g_h = gb.sorted_nodes_of_edge(e)[1]
            discharge = gb.node_prop(g_h, 'param').get_discharge()
            d['param'] = Parameters(g_h)
            d['param'].set_discharge(discharge)

        coupling_conditions = upwind_coupling.UpwindCoupling(solver)
        solver_coupler = coupler.Coupler(solver, coupling_conditions)
        U, rhs = solver_coupler.matrix_rhs(gb)
        deltaT = np.amin(gb.loop(solver.cfl, coupling_conditions.cfl).data)

        U_known = np.array([[ 1, 0, 0],
                            [ 0, 1,-1],
                            [-1, 0, 1]])
        rhs_known = np.array([1, 0, 0])

        deltaT_known = 5*1e-3

        rtol = 1e-15
        atol = rtol
        assert np.allclose(U.todense(), U_known, rtol, atol)
        assert np.allclose(rhs, rhs_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_coupling_3d_2d_left_right(self):
        f = np.array([[ 0,  1,  1,  0],
                      [ 0,  0,  1,  1],
                      [.5, .5, .5, .5]])
        gb = meshing.cart_grid( [f], [1, 1, 2], **{'physdims': [1, 1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        tol = 1e-3
        solver = upwind.Upwind()
        gb.add_node_props(['param'])

        for g, d in gb:
            param = Parameters(g)

            aperture = np.ones(g.num_cells)*np.power(1e-2, gb.dim_max() - g.dim)
            param.set_aperture(aperture)
            param.set_discharge(solver.discharge(g, [1, 0, 0], aperture))

            bound_faces = g.get_boundary_faces()
            bound_face_centers = g.face_centers[:, bound_faces]

            left = bound_face_centers[0, :] < tol
            right = bound_face_centers[0, :] > 1 - tol

            labels = np.array(['neu'] * bound_faces.size)
            labels[np.logical_or(left, right)] = ['dir']

            bc_val = np.zeros(g.num_faces)
            bc_dir = bound_faces[np.logical_or(left, right)]
            bc_val[bc_dir] = 1

            param.set_bc(solver, BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val(solver, bc_val)

            d['param'] = param

        # Assign coupling discharge
        gb.add_edge_prop('param')
        for e, d in gb.edges_props():
            g_h = gb.sorted_nodes_of_edge(e)[1]
            discharge = gb.node_prop(g_h, 'param').get_discharge()
            d['param'] = Parameters(g_h)
            d['param'].set_discharge(discharge)

        coupling_conditions = upwind_coupling.UpwindCoupling(solver)
        solver_coupler = coupler.Coupler(solver, coupling_conditions)
        U, rhs = solver_coupler.matrix_rhs(gb)
        deltaT = np.amin(gb.loop(solver.cfl, coupling_conditions.cfl).data)

        U_known = np.array([[.5, 0, 0],
                            [0, .5, 0],
                            [0, 0, 1e-2]])
        rhs_known = np.array([.5, .5, 1e-2])

        deltaT_known = 5*1e-1

        rtol = 1e-15
        atol = rtol
        assert np.allclose(U.todense(), U_known, rtol, atol)
        assert np.allclose(rhs, rhs_known, rtol, atol)
        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

    def test_upwind_coupling_3d_2d_1d_0d(self):
        f1 = np.array([[ 0,  1,  1,  0],
                       [ 0,  0,  1,  1],
                       [.5, .5, .5, .5]])
        f2 = np.array([[.5, .5, .5, .5],
                       [ 0,  1,  1,  0],
                       [ 0,  0,  1,  1]])
        f3 = np.array([[ 0,  1,  1,  0],
                       [.5, .5, .5, .5],
                       [ 0,  0,  1,  1]])

        gb = meshing.cart_grid([f1, f2, f3], [2, 2, 2],
                               **{'physdims': [1, 1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        tol = 1e-3
        solver = upwind.Upwind()
        gb.add_node_props(['param'])

        for g, d in gb:
            param = Parameters(g)

            aperture = np.ones(g.num_cells)*np.power(1e-2, gb.dim_max() - g.dim)
            param.set_aperture(aperture)
            param.set_discharge(solver.discharge(g, [1, 0, 0], aperture))

            bound_faces = g.get_boundary_faces()
            bound_face_centers = g.face_centers[:, bound_faces]

            left = bound_face_centers[0, :] < tol
            right = bound_face_centers[0, :] > 1 - tol

            labels = np.array(['neu'] * bound_faces.size)
            labels[np.logical_or(left, right)] = ['dir']

            bc_val = np.zeros(g.num_faces)
            bc_dir = bound_faces[np.logical_or(left, right)]
            bc_val[bc_dir] = 1

            param.set_bc(solver, BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val(solver, bc_val)

            d['param'] = param

        # Assign coupling discharge
        gb.add_edge_prop('param')
        for e, d in gb.edges_props():
            g_h = gb.sorted_nodes_of_edge(e)[1]
            discharge = gb.node_prop(g_h, 'param').get_discharge()
            d['param'] = Parameters(g_h)
            d['param'].set_discharge(discharge)

        coupling_conditions = upwind_coupling.UpwindCoupling(solver)
        solver_coupler = coupler.Coupler(solver, coupling_conditions)
        U, rhs = solver_coupler.matrix_rhs(gb)
        deltaT = np.amin(gb.loop(solver.cfl, coupling_conditions.cfl).data)

        U_known, rhs_known = matrix_rhs_for_test_upwind_coupling_3d_2d_1d_0d()

        deltaT_known = 5*1e-3

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

        solver = upwind.Upwind()

        gb.add_node_props(['param'])

        for g, d in gb:
            param = Parameters(g)
            aperture = np.ones(g.num_cells)*np.power(1e-2, gb.dim_max() - g.dim)
            param.set_aperture(aperture)
            param.set_discharge(solver.discharge(g, [2, 0, 0], aperture))

            bf = g.get_boundary_faces()
            bc = BoundaryCondition(g, bf, bf.size * ['neu'])
            param.set_bc(solver, bc)
            d['param'] = param

        # Assign coupling discharge
        gb.add_edge_prop('param')
        for e, d in gb.edges_props():
            g_h = gb.sorted_nodes_of_edge(e)[1]
            discharge = gb.node_prop(g_h, 'param').get_discharge()
            d['param'] = Parameters(g_h)
            d['param'].set_discharge(discharge)

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
        gb.add_node_props(['param'])

        for g, d in gb:
            param = Parameters(g)

            aperture = np.ones(g.num_cells)*np.power(1e-1, gb.dim_max() - g.dim)
            param.set_aperture(aperture)
            param.set_discharge(solver.discharge(g, [1, 1, 0], aperture))

            bound_faces = g.get_domain_boundary_faces()
            labels = np.array(['dir'] * bound_faces.size)
            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces] = 3

            param.set_bc(solver, BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val(solver, bc_val)

            d['param'] = param

        # Assign coupling discharge
        gb.add_edge_prop('param')
        for e, d in gb.edges_props():
            g_h = gb.sorted_nodes_of_edge(e)[1]
            discharge = gb.node_prop(g_h, 'param').get_discharge()
            d['param'] = Parameters(g_h)
            d['param'].set_discharge(discharge)

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

def matrix_rhs_for_test_upwind_coupling_3d_2d_1d_0d():
    U = np.array([\
        [  2.50e-01,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00],
        [  0.00e+00,   2.50e-01,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,  -2.50e-01,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00],
        [  0.00e+00,   0.00e+00,   2.50e-01,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00],
        [  0.00e+00,   0.00e+00,   0.00e+00,   2.50e-01,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
          -2.50e-01,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00],
        [  0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   2.50e-01,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00],
        [  0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           2.50e-01,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,  -2.50e-01,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00],
        [  0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   2.50e-01,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00],
        [  0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   2.50e-01,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,  -2.50e-01,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00],
        [  0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   5.00e-03,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00],
        [  0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   5.00e-03,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
          -5.00e-03,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00],
        [  0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           5.00e-03,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00],
        [  0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   5.00e-03,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,  -5.00e-03,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00],
        [  0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,  -2.50e-01,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   2.50e-01,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00],
        [ -2.50e-01,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   2.50e-01,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
          -5.55e-19,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00],
        [  0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,  -2.50e-01,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   2.50e-01,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00],
        [  0.00e+00,   0.00e+00,  -2.50e-01,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           2.50e-01,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,  -5.55e-19,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00],
        [  0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   5.00e-03,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00],
        [  0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   5.00e-03,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,  -5.00e-03,
           0.00e+00,   0.00e+00],
        [  0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   5.00e-03,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00],
        [  0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   5.00e-03,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
          -5.00e-03,   0.00e+00],
        [  0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,  -5.00e-03,   0.00e+00,
           0.00e+00,   0.00e+00,  -5.55e-19,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           5.00e-03,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00],
        [  0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
          -5.00e-03,   0.00e+00,   0.00e+00,   0.00e+00,  -5.55e-19,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   5.00e-03,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00],
        [  0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   1.00e-04,   0.00e+00,   0.00e+00,
           0.00e+00,  -1.00e-04],
        [  0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   1.00e-04,   0.00e+00,
           0.00e+00,   0.00e+00],
        [  0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,  -5.00e-03,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   5.00e-03,
           0.00e+00,   0.00e+00],
        [  0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,  -5.00e-03,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           5.00e-03,   0.00e+00],
        [  0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,  -1.00e-04,   0.00e+00,
           0.00e+00,   1.00e-04]])
    rhs = np.array(\
        [  2.50e-01,   0.00e+00,   2.50e-01,   0.00e+00,   2.50e-01,
         0.00e+00,   2.50e-01,   0.00e+00,   5.00e-03,   0.00e+00,
         5.00e-03,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   5.00e-03,   0.00e+00,   5.00e-03,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   1.00e-04,   0.00e+00,
         0.00e+00,   0.00e+00])

    return U, rhs

#------------------------------------------------------------------------------#
