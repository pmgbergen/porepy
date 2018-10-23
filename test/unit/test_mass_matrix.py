import numpy as np
import unittest


import porepy as pp

class MassMatrixTest(unittest.TestCase):
    def test_mass_matrix(self):
        g = pp.CartGrid([3, 3, 3])
        g.compute_geometry()
        phi = np.random.rand(g.num_cells)
        dt = 0.2
        param = pp.Parameters(g)
        param.set_porosity(phi)
        data = {"param": param, "deltaT": dt}
        time_discr = pp.MassMatrix()
        lhs, rhs = time_discr.assemble_matrix_rhs(g, data)
        self.assertTrue(np.allclose(rhs, 0))
        self.assertTrue(np.allclose(lhs.diagonal(), g.cell_volumes * phi / dt))
        off_diag = np.where(~np.eye(lhs.shape[0], dtype=bool))
        self.assertTrue(np.allclose(lhs.A[off_diag], 0))

    def test_inv_mass_matrix(self):
        g = pp.CartGrid([3, 3, 3])
        g.compute_geometry()
        phi = np.random.rand(g.num_cells)
        dt = 0.2
        param = pp.Parameters(g)
        param.set_porosity(phi)
        data = {"param": param, "deltaT": dt}
        time_discr = pp.InvMassMatrix()
        lhs, rhs = time_discr.assemble_matrix_rhs(g, data)
        self.assertTrue(np.allclose(rhs, 0))
        self.assertTrue(np.allclose(lhs.diagonal(), dt / (g.cell_volumes * phi)))
        off_diag = np.where(~np.eye(lhs.shape[0], dtype=bool))
        self.assertTrue(np.allclose(lhs.A[off_diag], 0))


if __name__ == '__main__':
    unittest.main()
