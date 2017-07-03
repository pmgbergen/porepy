from __future__ import division
import numpy as np
import unittest

from porepy.fracs import meshing
import porepy.utils.comp_geom as cg
from porepy.params import bc
from porepy.numerics.fv.transport import upwind, upwind_coupling
from porepy.numerics.mixed_dim import coupler

from porepy.viz import plot_grid ############################

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
        gb.add_node_props(['beta_n', 'bc', 'bc_val', 'a'])

        for g, d in gb:
            d['a'] = np.ones(g.num_cells)*np.power(1e-2, gb.dim_max() - g.dim)
            d['beta_n'] = solver.beta_n(g, [0, 1, 0], d['a'])

            bound_faces = g.get_boundary_faces()
            bound_face_centers = g.face_centers[:, bound_faces]

            top = bound_face_centers[1, :] > 2 - tol
            bottom = bound_face_centers[1, :] < tol

            labels = np.array(['neu'] * bound_faces.size)
            labels[np.logical_or(top, bottom)] = ['dir']

            bc_val = np.zeros(g.num_faces)
            bc_dir = bound_faces[np.logical_or(top, bottom)]
            bc_val[bc_dir] = 1

            d['bc'] = bc.BoundaryCondition(g, bound_faces, labels)
            d['bc_val'] = bc_val

        # Assign coupling permeability
        gb.add_edge_prop('beta_n')
        for e, d in gb.edges_props():
            g_h = gb.sorted_nodes_of_edge(e)[1]
            d['beta_n'] = gb.node_prop(g_h, 'beta_n')

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
        gb.add_node_props(['beta_n', 'bc', 'bc_val', 'a'])

        for g, d in gb:
            d['a'] = np.ones(g.num_cells)*np.power(1e-2, gb.dim_max() - g.dim)
            d['beta_n'] = solver.beta_n(g, [1, 0, 0], d['a'])

            bound_faces = g.get_boundary_faces()
            bound_face_centers = g.face_centers[:, bound_faces]

            left = bound_face_centers[0, :] < tol
            right = bound_face_centers[0, :] > 1 - tol

            labels = np.array(['neu'] * bound_faces.size)
            labels[np.logical_or(left, right)] = ['dir']

            bc_val = np.zeros(g.num_faces)
            bc_dir = bound_faces[np.logical_or(left, right)]
            bc_val[bc_dir] = 1

            d['bc'] = bc.BoundaryCondition(g, bound_faces, labels)
            d['bc_val'] = bc_val

        # Assign coupling permeability
        gb.add_edge_prop('beta_n')
        for e, d in gb.edges_props():
            g_h = gb.sorted_nodes_of_edge(e)[1]
            d['beta_n'] = gb.node_prop(g_h, 'beta_n')

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
        gb.add_node_props(['beta_n', 'bc', 'bc_val', 'a'])

        for g, d in gb:
            d['a'] = np.ones(g.num_cells)*np.power(1e-2, gb.dim_max() - g.dim)
            d['beta_n'] = solver.beta_n(g, [1, 0, 0], d['a'])

            bound_faces = g.get_boundary_faces()
            bound_face_centers = g.face_centers[:, bound_faces]

            left = bound_face_centers[0, :] < tol
            right = bound_face_centers[0, :] > 2 - tol

            labels = np.array(['neu'] * bound_faces.size)
            labels[np.logical_or(left, right)] = ['dir']

            bc_val = np.zeros(g.num_faces)
            bc_dir = bound_faces[np.logical_or(left, right)]
            bc_val[bc_dir] = 1

            d['bc'] = bc.BoundaryCondition(g, bound_faces, labels)
            d['bc_val'] = bc_val

        # Assign coupling permeability
        gb.add_edge_prop('beta_n')
        for e, d in gb.edges_props():
            g_h = gb.sorted_nodes_of_edge(e)[1]
            d['beta_n'] = gb.node_prop(g_h, 'beta_n')

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
        gb.add_node_props(['beta_n', 'bc', 'bc_val', 'a'])

        for g, d in gb:
            d['a'] = np.ones(g.num_cells)*np.power(1e-2, gb.dim_max() - g.dim)
            d['beta_n'] = solver.beta_n(g, [0, 0, 1], d['a'])

            bound_faces = g.get_boundary_faces()
            bound_face_centers = g.face_centers[:, bound_faces]

            top = bound_face_centers[2, :] > 1 - tol
            bottom = bound_face_centers[2, :] < tol

            labels = np.array(['neu'] * bound_faces.size)
            labels[np.logical_or(top, bottom)] = ['dir']

            bc_val = np.zeros(g.num_faces)
            bc_dir = bound_faces[np.logical_or(top, bottom)]
            bc_val[bc_dir] = 1

            d['bc'] = bc.BoundaryCondition(g, bound_faces, labels)
            d['bc_val'] = bc_val

        # Assign coupling permeability
        gb.add_edge_prop('beta_n')
        for e, d in gb.edges_props():
            g_h = gb.sorted_nodes_of_edge(e)[1]
            d['beta_n'] = gb.node_prop(g_h, 'beta_n')

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
        gb.add_node_props(['beta_n', 'bc', 'bc_val', 'a'])

        for g, d in gb:
            d['a'] = np.ones(g.num_cells)*np.power(1e-2, gb.dim_max() - g.dim)
            d['beta_n'] = solver.beta_n(g, [1, 0, 0], d['a'])

            bound_faces = g.get_boundary_faces()
            bound_face_centers = g.face_centers[:, bound_faces]

            left = bound_face_centers[0, :] < tol
            right = bound_face_centers[0, :] > 1 - tol

            labels = np.array(['neu'] * bound_faces.size)
            labels[np.logical_or(left, right)] = ['dir']

            bc_val = np.zeros(g.num_faces)
            bc_dir = bound_faces[np.logical_or(left, right)]
            bc_val[bc_dir] = 1

            d['bc'] = bc.BoundaryCondition(g, bound_faces, labels)
            d['bc_val'] = bc_val

        # Assign coupling permeability
        gb.add_edge_prop('beta_n')
        for e, d in gb.edges_props():
            g_h = gb.sorted_nodes_of_edge(e)[1]
            d['beta_n'] = gb.node_prop(g_h, 'beta_n')

        coupling_conditions = upwind_coupling.UpwindCoupling(solver)
        solver_coupler = coupler.Coupler(solver, coupling_conditions)
        U, rhs = solver_coupler.matrix_rhs(gb)
        deltaT = np.amin(gb.loop(solver.cfl, coupling_conditions.cfl).data)

        print( repr(U.todense() ))
        print( repr(rhs) )
        print( deltaT )

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

        #plot_grid.plot_grid(gb, alpha=0, info="c")

        tol = 1e-3
        solver = upwind.Upwind()
        gb.add_node_props(['beta_n', 'bc', 'bc_val', 'a'])

        for g, d in gb:
            d['a'] = np.ones(g.num_cells)*np.power(1e-2, gb.dim_max() - g.dim)
            d['beta_n'] = solver.beta_n(g, [1, 0, 0], d['a'])

            bound_faces = g.get_boundary_faces()
            bound_face_centers = g.face_centers[:, bound_faces]

            left = bound_face_centers[0, :] < tol
            right = bound_face_centers[0, :] > 1 - tol

            labels = np.array(['neu'] * bound_faces.size)
            labels[np.logical_or(left, right)] = ['dir']

            bc_val = np.zeros(g.num_faces)
            bc_dir = bound_faces[np.logical_or(left, right)]
            bc_val[bc_dir] = 1

            d['bc'] = bc.BoundaryCondition(g, bound_faces, labels)
            d['bc_val'] = bc_val

        # Assign coupling permeability
        gb.add_edge_prop('beta_n')
        for e, d in gb.edges_props():
            g_h = gb.sorted_nodes_of_edge(e)[1]
            d['beta_n'] = gb.node_prop(g_h, 'beta_n')
            print(g_h.dim,  d['beta_n'] )

        coupling_conditions = upwind_coupling.UpwindCoupling(solver)
        solver_coupler = coupler.Coupler(solver, coupling_conditions)
        U, rhs = solver_coupler.matrix_rhs(gb)
        deltaT = np.amin(gb.loop(solver.cfl, coupling_conditions.cfl).data)

        np.set_printoptions(linewidth=75, precision=2)
        #print( repr(U.todense() ))
        #print( repr(rhs) )
        #print( deltaT )

#        U_known = np.array([[.5, 0, 0],
#                            [0, .5, 0],
#                            [0, 0, 1e-2]])
#        rhs_known = np.array([.5, .5, 1e-2])
#
#        deltaT_known = 5*1e-1
#
#        rtol = 1e-15
#        atol = rtol
#        assert np.allclose(U.todense(), U_known, rtol, atol)
#        assert np.allclose(rhs, rhs_known, rtol, atol)
#        assert np.allclose(deltaT, deltaT_known, rtol, atol)

#------------------------------------------------------------------------------#

BasicsTest().test_upwind_coupling_3d_2d_1d_0d()


def matrix_rhs_for_test_upwind_coupling_3d_2d_1d_0d():
    U = np.array(\
       [[  2.50e-01,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
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
           0.00e+00,   0.00e+00],
        [  0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
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
           0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
           0.00e+00,   0.00e+00]])
    rhs = np.array(\
        [  2.50e-01,   0.00e+00,   2.50e-01,   0.00e+00,   2.50e-01,
         0.00e+00,   2.50e-01,   0.00e+00,   5.00e-03,   0.00e+00,
         5.00e-03,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,
         0.00e+00,   5.00e-03,   0.00e+00,   5.00e-03,   0.00e+00,
         0.00e+00,   0.00e+00,   0.00e+00,   1.00e-04,   0.00e+00,
         0.00e+00,   0.00e+00])
    return U, rhs
