import scipy.sparse as sps
import numpy as np
import numpy.linalg
import unittest
import porepy as pp
from porepy.numerics.fv import mpfa, mpsa, fvutils, biot
from porepy.params import tensor, bc
from porepy.params.data import Parameters
from test.integration import setup_grids_mpfa_mpsa_tests as setup_grids


class BiotTest(unittest.TestCase):
    def test_no_dynamics_2d(self):
        g_list = setup_grids.setup_2d()
        kw_f = "flow"
        kw_m = "mechanics"
        for g in g_list:
            discr = biot.Biot()

            bound_faces = g.get_all_boundary_faces()
            bound = bc.BoundaryCondition(
                g, bound_faces.ravel("F"), ["dir"] * bound_faces.size
            )
            bound_mech = bc.BoundaryConditionVectorial(
                g, bound_faces.ravel("F"), ["dir"] * bound_faces.size
            )

            mu = np.ones(g.num_cells)
            c = tensor.FourthOrderTensor(g.dim, mu, mu)
            k = tensor.SecondOrderTensor(g.dim, np.ones(g.num_cells))

            bound_val = np.zeros(g.num_faces)
            porosity = np.ones(g.num_cells)
            aperture = np.ones(g.num_cells)

            param = Parameters(g, [kw_f, kw_m], [{}, {}])

            param[kw_f]["bc"] = bound
            param[kw_m]["bc"] = bound_mech
            param[kw_f]["aperture"] = aperture
            param[kw_m]["aperture"] = aperture
            param[kw_f]["second_order_tensor"] = k
            param[kw_m]["fourth_order_tensor"] = c
            param[kw_f]["bc_values"] = bound_val
            param[kw_m]["bc_values"] = np.tile(bound_val, g.dim)
            param[kw_f]["porosity"] = porosity
            param[kw_m]["porosity"] = porosity
            param[kw_f]["biot_alpha"] = 1
            param[kw_m]["biot_alpha"] = 1
            param[kw_f]["fluid_compressibility"] = 0
            param[kw_f]["fluid_viscosity"] = 1
            data = {"inverter": "python", "time_step": 1}
            data[pp.keywords.PARAMETERS] = param
            data[pp.keywords.DISCRETIZATION_MATRICES] = {kw_f: {}, kw_m: {}}
            A, b = discr.matrix_rhs(g, data)
            sol = np.linalg.solve(A.todense(), b)

            self.assertTrue(np.isclose(sol, np.zeros(g.num_cells * (g.dim + 1))).all())

    #    def test_uniform_displacement(self):
    #        # Uniform displacement in mechanics (enforced by boundary conditions).
    #        # Constant pressure boundary conditions.
    #        g_list = setup_grids.setup_2d()
    #        for g in g_list:
    #            bound_faces = g.get_all_boundary_faces()
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
    #            self.assertTrue(np.isclose(sol[:sz_mech],)
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
        self.assertTrue(np.allclose(known_matrix, a))


if __name__ == "__main__":
    unittest.main()
