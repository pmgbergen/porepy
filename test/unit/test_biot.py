import scipy.sparse as sps
import numpy as np
import numpy.linalg
import unittest

from porepy_new.src.porepy.numerics.fv import mpfa, mpsa, fvutils
from porepy_new.src.porepy.params import second_order_tensor, fourth_order_tensor, bc
import setup_grids_mpfa_mpsa_tests as setup_grids


class BiotTest():
    # class BiotTest(unittest.TestCase):

    def mpfa_discr(self, g, bound):
        k = second_order_tensor.SecondOrderTensor(g.dim, np.ones(g.num_cells))
        flux, bound_flux = mpfa.mpfa(g, k, bound, inverter='python')
        div_flow = fvutils.scalar_divergence(g)
        return flux, bound_flux, div_flow

    def mpsa_discr(self, g, bound):
        mu = np.ones(g.num_cells)
        c = fourth_order_tensor.FourthOrderTensor(g.dim, mu, mu)
        stress, bound_stress, \
            grad_p, div_d, stabilization = mpsa.biot(g, c, bound,
                                                     inverter='python')
        div_mech = fvutils.vector_divergence(g)
        return stress, bound_stress, grad_p, div_d, stabilization, div_mech

    def test_no_dynamics_2d(self):
        g_list = setup_grids.setup_2d()
        for g in g_list:
            bound_faces = g.get_boundary_faces()
            bound = bc.BoundaryCondition(g, bound_faces.ravel('F'),
                                         ['dir'] * bound_faces.size)
            flux, bound_flux, div_flow = self.mpfa_discr(g, bound)

            a_flow = div_flow * flux

            stress, bound_stress, grad_p, div_d, \
                stabilization, div_mech = self.mpsa_discr(g, bound)

            a_mech = div_mech * stress

            a_biot = sps.bmat([[a_mech, grad_p],
                               [div_d, a_flow + stabilization]])

            bval_mech = np.ones(g.num_faces * g.dim)
            bval_flow = np.ones(g.num_faces)
            rhs = np.hstack((div_mech * bound_stress * bval_mech,
                             div_flow * bound_flux * bval_flow))
            sol = np.linalg.solve(a_biot.todense(), rhs)

            assert np.isclose(sol, np.ones(g.num_cells * (g.dim + 1))).all()

    # if __name__ == '__main__':
    #     unittest.main()


if __name__ == '__main__':
    bt = BiotTest()
    bt.test_no_dynamics_2d()


