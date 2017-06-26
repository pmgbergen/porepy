import scipy.sparse as sps
import numpy as np
import numpy.linalg
import unittest

from porepy.numerics.fv import mpfa, biot, fvutils
from porepy.params import second_order_tensor, fourth_order_tensor, bc
from test.integration import setup_grids_mpfa_mpsa_tests as setup_grids


class BiotTest(unittest.TestCase):

    def test_steady_state(self):
        # Zero boundary conditions and no right hand sid. Nothing should
        # happen.
        g_list = setup_grids.setup_2d()
        for g in g_list:
            discr = biot.Biot()

            bound_faces = g.get_boundary_faces()
            bound = bc.BoundaryCondition(g, bound_faces.ravel('F'),
                                         ['dir'] * bound_faces.size)

            mu = np.ones(g.num_cells)
            c = fourth_order_tensor.FourthOrderTensor(g.dim, mu, mu)
            k = second_order_tensor.SecondOrderTensor(g.dim, np.ones(g.num_cells))

            bound_val = np.zeros(g.num_faces)

            data = {'bound_flow': bound, 'bound_mech': bound, 'perm': k,
                    'stiffness': c, 'inverter': 'python',
                    'bc_val_mech': np.tile(bound_val, g.dim),
                    'bc_val_flow': bound_val,
                    'poro': np.ones(g.num_cells),
                    'dt': 1,
                    'biot_alpha': 1
                   }

            A, b = discr.matrix_rhs(g, data)
            sol = np.linalg.solve(A.todense(), b)

            assert np.isclose(sol, np.zeros(g.num_cells * (g.dim + 1))).all()

#    def test_uniform_displacement(self):
#        # Uniform displacement in mechanics (enforced by boundary conditions).
#        # Constant pressure boundary conditions. 
#        g_list = setup_grids.setup_2d()
#        for g in g_list:
#            bound_faces = g.get_boundary_faces()
#            bound = bc.BoundaryCondition(g, bound_faces.ravel('F'),
#                                         ['dir'] * bound_faces.size)
#            flux, bound_flux, div_flow = self.mpfa_discr(g, bound)
#
#            a_flow = div_flow * flux
#
#            stress, bound_stress, grad_p, div_d, \
#                stabilization, bound_div_d, div_mech = self.mpsa_discr(g, bound)
#
#            a_mech = div_mech * stress
#
#            a_biot = sps.bmat([[a_mech, grad_p],
#                               [div_d, a_flow + stabilization]])
#            
#            const_bound_val_mech = 1
#            bval_mech = const_bound_val_mech * np.ones(g.num_faces * g.dim)
#            bval_flow = np.ones(g.num_faces)
#            rhs = np.hstack((-div_mech * bound_stress * bval_mech,
#                             div_flow * bound_flux * bval_flow\
#                             + div_flow * bound_div_d * bval_mech))
#            sol = np.linalg.solve(a_biot.todense(), rhs)
#
#            sz_mech = g.num_cells * g.dim
#            assert np.isclose(sol[:sz_mech], 
#                              const_bound_val_mech * np.ones(sz_mech)).all()

    def test_face_vector_to_scalar(self):
        # Test of function face_vector_to_scalar
        nf = 3
        nd = 2
        rows = np.array([0, 0, 1, 1, 2, 2])
        cols = np.arange(6)
        vals = np.ones(6)
        known_matrix = sps.coo_matrix((vals, (rows, cols))).tocsr().toarray()
        a = biot.Biot()._face_vector_to_scalar(3, 2).toarray()
        assert np.allclose(known_matrix, a)


    if __name__ == '__main__':
        unittest.main()




