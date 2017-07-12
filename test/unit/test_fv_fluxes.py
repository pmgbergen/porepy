from __future__ import division
import numpy as np
import scipy.sparse as sps####
import unittest
from porepy.fracs import meshing
import porepy.utils.comp_geom as cg
from porepy.params import bc, tensor
from porepy.params.data import Parameters

from porepy.numerics.fv import mpfa, tpfa, tpfa_coupling
from porepy.numerics.mixed_dim import coupler


#------------------------------------------------------------------------------#

class BasicsTest( unittest.TestCase ):

#------------------------------------------------------------------------------#

    def test_tpfa_fluxes_2d_1d_left_right_dir_neu(self):
        """
        Grid: 2 x 2 cells in matrix + 2 cells in the fracture from left to right.
        Dirichlet + inflow + no-flow, conductive fracture.
        Tests pressure solution and fluxes.
        """
        f = np.array([[0, 1],
                      [.5, .5]])
        gb = meshing.cart_grid( [f], [2, 2], **{'physdims': [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        tol = 1e-3
        solver = tpfa.Tpfa(physics='flow')
        gb.add_node_props(['param'])
        a = 1e-2
        for g, d in gb:
            param = Parameters(g)

            a_dim = np.power(a, gb.dim_max() - g.dim)
            aperture = np.ones(g.num_cells)*a_dim
            param.set_aperture(aperture)

            p = tensor.SecondOrder(3,np.ones(g.num_cells)* np.power(1e-3, g.dim<gb.dim_max()))
            param.set_tensor('flow', p)
            bound_faces = g.get_boundary_faces()
            bound_face_centers = g.face_centers[:, bound_faces]

            right = bound_face_centers[0, :] > 1 - tol
            left = bound_face_centers[0, :] < tol

            labels = np.array(['neu'] * bound_faces.size)
            labels[right] = ['dir']

            bc_val = np.zeros(g.num_faces)
            bc_dir = bound_faces[right]
            bc_neu = bound_faces[left]
            bc_val[bc_dir] = g.face_centers[0,bc_dir]
            bc_val[bc_neu] = -g.face_areas[bc_neu]*a_dim

            param.set_bc(solver, bc.BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val(solver, bc_val)

            d['param'] = param

        coupling_conditions = tpfa_coupling.TpfaCoupling(solver)
        solver_coupler = coupler.Coupler(solver, coupling_conditions)
        A, rhs = solver_coupler.matrix_rhs(gb)
        p = sps.linalg.spsolve(A, rhs)
        solver_coupler.split(gb, "p", p)
        coupling_conditions.compute_discharges(gb)

        p_known = np.array([1.7574919 ,  1.25249747,  1.7574919 ,  1.25249747,
                            1.25250298,  1.80993337])

        # Known discharges
        d_0, d_1 = fluxes_2d_1d_left_right_dir_neu()
                
        rtol = 1e-6
        atol = rtol
        
        for _, d in gb:            
            if d['node_number'] == 0:
                assert np.allclose( d['discharge'], d_0, rtol, atol)
            if d['node_number'] == 1:
                assert np.allclose( d['discharge'], d_1, rtol, atol)
                
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
        gb = meshing.cart_grid( [f], [2, 2], **{'physdims': [1, 1]})
        gb.compute_geometry()
        gb.assign_node_ordering()

        tol = 1e-3
        solver = mpfa.Mpfa(physics='flow')
        gb.add_node_props(['param'])
        a = 1e-2
        for g, d in gb:
            param = Parameters(g)

            a_dim = np.power(a, gb.dim_max() - g.dim)
            aperture = np.ones(g.num_cells)*a_dim
            param.set_aperture(aperture)

            p = tensor.SecondOrder(3,np.ones(g.num_cells)* np.power(1e-3, g.dim<gb.dim_max()))
            param.set_tensor('flow', p)
            bound_faces = g.get_boundary_faces()
            bound_face_centers = g.face_centers[:, bound_faces]

            right = bound_face_centers[0, :] > 1 - tol
            left = bound_face_centers[0, :] < tol

            labels = np.array(['neu'] * bound_faces.size)
            labels[right] = ['dir']

            bc_val = np.zeros(g.num_faces)
            bc_dir = bound_faces[right]
            bc_neu = bound_faces[left]
            bc_val[bc_dir] = g.face_centers[0,bc_dir]
            bc_val[bc_neu] = -g.face_areas[bc_neu]*a_dim

            param.set_bc(solver, bc.BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val(solver, bc_val)

            d['param'] = param

        coupling_conditions = tpfa_coupling.TpfaCoupling(solver)
        solver_coupler = coupler.Coupler(solver, coupling_conditions)
        A, rhs = solver_coupler.matrix_rhs(gb)
        p = sps.linalg.spsolve(A, rhs)
        solver_coupler.split(gb, "p", p)
        coupling_conditions.compute_discharges(gb)

        p_known = np.array([1.7574919 ,  1.25249747,  1.7574919 ,  1.25249747,
                            1.25250298,  1.80993337])

        # Known discharges
        d_0, d_1 = fluxes_2d_1d_left_right_dir_neu()
                
        rtol = 1e-6
        atol = rtol
        
        for _, d in gb:
            print('d_{} = np.'.format(d['node_number']), repr(d['discharge']))
            if d['node_number'] == 0:
                assert np.allclose( d['discharge'], d_0, rtol, atol)
            if d['node_number'] == 1:
                assert np.allclose( d['discharge'], d_1, rtol, atol)
                
        assert np.allclose(p, p_known, rtol, atol)
        


        
#------------------------------------------------------------------------------#
# Left out due to problems with fracture face id: not the same each time the grids
# are generated.
#     def test_tpfa_coupling_3d_2d_1d_0d_dir(self):
#         f1 = np.array([[ 0,  1,  1,  0],
#                        [ 0,  0,  1,  1],
#                        [.5, .5, .5, .5]])
#         f2 = np.array([[.5, .5, .5, .5],
#                        [ 0,  1,  1,  0],
#                        [ 0,  0,  1,  1]])
#         f3 = np.array([[ 0,  1,  1,  0],
#                        [.5, .5, .5, .5],
#                        [ 0,  0,  1,  1]])

#         gb = meshing.cart_grid([f1, f2, f3], [2, 2, 2],
#                                **{'physdims': [1, 1, 1]})
#         gb.compute_geometry()
#         gb.assign_node_ordering()
#         # Remove flag for dual
#         cell_centers1 = np.array([[ 0.25 , 0.75 , 0.25 , 0.75],
#                                   [ 0.25 , 0.25 , 0.75 , 0.75],
#                                   [ 0.5  , 0.5  , 0.5  , 0.5 ]])
#         cell_centers2 = np.array([[ 0.5  , 0.5  , 0.5  , 0.5 ],
#                                   [ 0.25 , 0.25 , 0.75 , 0.75],
#                                   [ 0.75 , 0.25 , 0.75 , 0.25]])
#         cell_centers3 = np.array([[ 0.25 , 0.75 , 0.25 , 0.75],
#                                   [ 0.5  , 0.5  , 0.5  , 0.5 ],
#                                   [ 0.25 , 0.25 , 0.75 , 0.75]])
#         cell_centers4 = np.array([[ 0.5 ], [ 0.25], [ 0.5 ]])
#         cell_centers5 = np.array([[ 0.5 ], [ 0.75], [ 0.5 ]])
#         cell_centers6 = np.array([[ 0.75], [ 0.5 ], [ 0.5 ]])
#         cell_centers7 = np.array([[ 0.25], [ 0.5 ], [ 0.5 ]])
#         cell_centers8 = np.array([[ 0.5 ], [ 0.5 ], [ 0.25]])
#         cell_centers9 = np.array([[ 0.5 ], [ 0.5 ], [ 0.75]])

#         for g, d in gb:
#             if np.allclose(g.cell_centers[:, 0], cell_centers1[:, 0]):
#                 d['node_number'] = 1
#             elif np.allclose(g.cell_centers[:, 0], cell_centers2[:, 0]):
#                 d['node_number'] = 2
#             elif np.allclose(g.cell_centers[:, 0], cell_centers3[:, 0]):
#                 d['node_number'] = 3
#             elif np.allclose(g.cell_centers[:, 0], cell_centers4[:, 0]):
#                 d['node_number'] = 4
#             elif np.allclose(g.cell_centers[:, 0], cell_centers5[:, 0]):
#                 d['node_number'] = 5
#             elif np.allclose(g.cell_centers[:, 0], cell_centers6[:, 0]):
#                 d['node_number'] = 6
#             elif np.allclose(g.cell_centers[:, 0], cell_centers7[:, 0]):
#                 d['node_number'] = 7
#             elif np.allclose(g.cell_centers[:, 0], cell_centers8[:, 0]):
#                 d['node_number'] = 8
#             elif np.allclose(g.cell_centers[:, 0], cell_centers9[:, 0]):
#                 d['node_number'] = 9
#             else:
#                 pass

#         tol = 1e-3
#         solver = tpfa.Tpfa()
#         gb.add_node_props(['param'])

#         a = 1e-2
#         for g, d in gb:
#             param = Parameters(g)

#             aperture = np.ones(g.num_cells)*np.power(a, gb.dim_max() - g.dim)
#             param.set_aperture(aperture)

#             p = tensor.SecondOrder(3,np.ones(g.num_cells)* np.power(1e3, g.dim<gb.dim_max()))
#             param.set_tensor('flow', p)
#             bound_faces = g.get_boundary_faces()
#             bound_face_centers = g.face_centers[:, bound_faces]

#             left = bound_face_centers[0, :] > 1 - tol
#             right = bound_face_centers[0, :] < tol

#             labels = np.array(['neu'] * bound_faces.size)
#             labels[np.logical_or(left, right)] = ['dir']

#             bc_val = np.zeros(g.num_faces)
#             bc_dir = bound_faces[np.logical_or(left, right)]
#             bc_val[bc_dir] = g.face_centers[0,bc_dir]

#             param.set_bc(solver, bc.BoundaryCondition(g, bound_faces, labels))
#             param.set_bc_val(solver, bc_val)

#             d['param'] = param

       

#         coupling_conditions = tpfa_coupling.TpfaCoupling(solver)
#         solver_coupler = coupler.Coupler(solver, coupling_conditions)
#         A, rhs = solver_coupler.matrix_rhs(gb)
#         p = sps.linalg.spsolve(A, rhs)
#         solver_coupler.split(gb, "p", p)
#         coupling_conditions.compute_discharges(gb)

        
#         discharges_known, p_known = \
#                 discharges_pressure_for_test_tpfa_coupling_3d_2d_1d_0d()
        
#         rtol = 1e-6
#         atol = rtol
    
#         for _, d in gb:
#             n = d['node_number']
#             print('n',n)
#             print('d',d['discharge'])
#             print(discharges_known[n])
#             if discharges_known[n] is not None:
#                 assert np.allclose(d['discharge'], discharges_known[n], rtol, atol)
#         assert np.allclose(p, p_known, rtol, atol)
        
# #------------------------------------------------------------------------------#

# def discharges_pressure_for_test_tpfa_coupling_3d_2d_1d_0d():
#     d_4 = np.array([  8.32667268e-17,   0.00000000e+00])
#     d_0 = np.array([-0.24879143, -0.25120354, -0.24879143, -0.24879143, -0.25120354,
#                       -0.24879143, -0.24879143, -0.25120354, -0.24879143, -0.24879143,
#                       -0.25120354, -0.24879143,  0.        ,  0.        , -0.00120606,
#                       0.00120606,  0.        ,  0.        ,  0.        ,  0.        ,
#                       -0.00120606,  0.00120606,  0.        ,  0.        ,  0.        ,
#                       0.        ,  0.        ,  0.        , -0.00120606,  0.00120606,
#                       -0.00120606,  0.00120606,  0.        ,  0.        ,  0.        ,
#                       0.        , -0.25120354, -0.25120354, -0.25120354, -0.25120354,
#                       0.00120606, -0.00120606,  0.00120606, -0.00120606,  0.00120606,
#                       -0.00120606,  0.00120606, -0.00120606])
#     d_10 = None
#     d_3 = np.array([ -4.95170705e+00,  -4.94930682e+00,  -4.95170705e+00,
#                        -4.95170705e+00,  -4.94930682e+00,  -4.95170705e+00,
#                        0.00000000e+00,   0.00000000e+00,  -1.18811521e-05,
#                        1.18811521e-05,   0.00000000e+00,   0.00000000e+00,
#                        -4.94930682e+00,   1.18811521e-05,  -1.18811521e-05,
#                        -4.94930682e+00])
#     d_9 = np.array([  5.55111512e-17,   0.00000000e+00])
#     d_2 = np.array([  0.00000000e+00,   1.77635684e-15,   0.00000000e+00,
#                         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#                         0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#                         -1.77635684e-15,   0.00000000e+00,   0.00000000e+00,
#                         0.00000000e+00,  -3.55271368e-15,  -1.77635684e-15,
#                         -1.77635684e-15])
#     d_8 = np.array([  0.00000000e+00,  -5.55111512e-17])
#     d_5 = np.array([ 0.,  0.])
#     d_1 = np.array([ -4.95170705e+00,  -4.94930682e+00,  -4.95170705e+00,
#                        -4.95170705e+00,  -4.94930682e+00,  -4.95170705e+00,
#                        0.00000000e+00,   0.00000000e+00,  -1.18811521e-05,
#                        1.18811521e-05,   0.00000000e+00,   0.00000000e+00,
#                        -4.94930682e+00,  -4.94930682e+00,   1.18811521e-05,
#                        -1.18811521e-05])
#     d_7 = np.array([ 0.09898637,  0.0990339 ])
#     d_6 = np.array([ 0.0990339 ,  0.09898637])

#     discharges = [d_0, d_1,d_2, d_3, d_4, d_5, d_6 ,d_7, d_8,d_9, d_10]
#     pressure = np.array([\
#         0.24879143,  0.75120857,  0.24879143,  0.75120857,  0.24879143,
#         0.75120857,  0.24879143,  0.75120857,  0.24758535,  0.75241465,
#         0.24758535,  0.75241465,  0.5       ,  0.5       ,  0.5       ,
#         0.5       ,  0.24758535,  0.75241465,  0.24758535,  0.75241465,
#         0.5       ,  0.5       ,  0.75241525,  0.24758475,  0.5       ,
#         0.5       ,  0.5       ])
#     return discharges, pressure

# #------------------------------------------------------------------------------#
def fluxes_2d_1d_left_right_dir_neu():
    d_0 = np.array([  5.00000000e-01,   5.04994426e-01,   5.04994950e-01,
                          5.00000000e-01,   5.04994426e-01,   5.04994950e-01,
                          0.00000000e+00,   0.00000000e+00,   4.99442570e-03,
                          5.24244319e-07,   0.00000000e+00,   0.00000000e+00,
                          -4.99442570e-03,  -5.24244319e-07])
    d_1 = np.array([ -1.01001192e-05,  -1.11486078e-05,  -1.00000000e-02])
    return d_0, d_1

