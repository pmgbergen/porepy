import numpy as np
import scipy.sparse as sps
import unittest
import warnings

from porepy.numerics.fv import mpsa
from porepy.fracs import meshing
from porepy.params.data import Parameters

from test.integration import setup_grids_mpfa_mpsa_tests as setup_grids


class MpsaTest(unittest.TestCase):
    def test_matrix_rhs(self):
        g_list = setup_grids.setup_2d()

        for g in g_list:
            solver = mpsa.Mpsa()
            data = {"param": Parameters(g)}
            A, b = solver.matrix_rhs(g, data)
            assert np.all(A.shape == (g.dim * g.num_cells, g.dim * g.num_cells))
            assert b.size == g.dim * g.num_cells

    def test_matrix_rhs_no_disc(self):
        g_list = setup_grids.setup_2d()

        for g in g_list:
            solver = mpsa.Mpsa()
            data = {"param": Parameters(g)}
            cell_dof = g.dim * g.num_cells
            face_dof = g.dim * g.num_faces
            data["stress"] = sps.csc_matrix((face_dof, cell_dof))
            data["bound_stress"] = sps.csc_matrix((face_dof, face_dof))
            A, b = solver.matrix_rhs(g, data, discretize=False)
            assert np.sum(A != 0) == 0
            assert np.all(b == 0)
