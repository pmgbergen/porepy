import numpy as np
import scipy.sparse as sps
import unittest
import porepy as pp


from test.integration import setup_grids_mpfa_mpsa_tests as setup_grids


class MpsaTest(unittest.TestCase):
    def test_matrix_rhs(self):
        g_list = setup_grids.setup_2d()
        kw = "mechanics"
        for g in g_list:
            solver = pp.Mpsa(kw)
            data = pp.initialize_default_data(g, {}, kw)

            solver.discretize(g, data)
            A, b = solver.assemble_matrix_rhs(g, data)
            self.assertTrue(
                np.all(A.shape == (g.dim * g.num_cells, g.dim * g.num_cells))
            )
            self.assertTrue(b.size == g.dim * g.num_cells)

    def test_matrix_rhs_no_disc(self):
        g_list = setup_grids.setup_2d()
        kw = "mechanics"
        for g in g_list:
            solver = pp.Mpsa(kw)
            data = pp.initialize_default_data(g, {}, kw)
            cell_dof = g.dim * g.num_cells
            face_dof = g.dim * g.num_faces
            stress = sps.csc_matrix((face_dof, cell_dof))
            bound_stress = sps.csc_matrix((face_dof, face_dof))
            data[pp.DISCRETIZATION_MATRICES][kw] = {
                "stress": stress,
                "bound_stress": bound_stress,
            }

            A, b = solver.assemble_matrix_rhs(g, data)
            self.assertTrue(np.sum(A != 0) == 0)
            self.assertTrue(np.all(b == 0))


if __name__ == "__main__":
    MpsaTest().test_matrix_rhs()
    unittest.main()
