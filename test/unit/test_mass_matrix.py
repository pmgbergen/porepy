import numpy as np
import unittest


from porepy.numerics.fv.mass_matrix import MassMatrix, InvMassMatrix
from porepy.grids import structured
from porepy.params.data import Parameters


class MassMatrixTest(unittest.TestCase):
    def test_mass_matrix(self):
        g = structured.CartGrid([3, 3, 3])
        g.compute_geometry()
        phi = np.random.rand(g.num_cells)
        dt = 0.2
        param = Parameters(g)
        param.set_porosity(phi)
        data = {"param": param, "deltaT": dt}
        time_discr = MassMatrix()
        lhs, rhs = time_discr.matrix_rhs(g, data)
        assert np.allclose(rhs, 0)
        assert np.allclose(lhs.diagonal(), g.cell_volumes * phi / dt)
        off_diag = np.where(~np.eye(lhs.shape[0], dtype=bool))
        assert np.allclose(lhs.A[off_diag], 0)

    def test_inv_mass_matrix(self):
        g = structured.CartGrid([3, 3, 3])
        g.compute_geometry()
        phi = np.random.rand(g.num_cells)
        dt = 0.2
        param = Parameters(g)
        param.set_porosity(phi)
        data = {"param": param, "deltaT": dt}
        time_discr = InvMassMatrix()
        lhs, rhs = time_discr.matrix_rhs(g, data)
        assert np.allclose(rhs, 0)
        assert np.allclose(lhs.diagonal(), dt / (g.cell_volumes * phi))
        off_diag = np.where(~np.eye(lhs.shape[0], dtype=bool))
        assert np.allclose(lhs.A[off_diag], 0)
