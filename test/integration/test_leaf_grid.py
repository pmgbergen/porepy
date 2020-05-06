import numpy as np
import scipy.sparse as sps
import unittest
import warnings

import porepy as pp


class TestCartLeafGrid(unittest.TestCase):
    def test_refinement_one_level_tpfa(self):
        lg = pp.CartLeafGrid([2, 2], [1, 1], 2)
        lg.refine_cells(np.arange(lg.num_cells))

        g_ref = pp.CartGrid([4, 4], [1, 1])
        g_ref.compute_geometry()

        key = "flow"
        d = pp.initialize_default_data(lg, {}, key)
        d_ref = pp.initialize_default_data(g_ref, {}, key)
        discr = pp.Tpfa(key)

        discr.discretize(g_ref, d_ref)
        discr.discretize(lg, d)

        mat_dict = d[pp.DISCRETIZATION_MATRICES][key]
        mat_dict_ref = d[pp.DISCRETIZATION_MATRICES][key]

        for key in mat_dict_ref.keys():
            self.assertTrue(np.allclose(mat_dict[key].A, mat_dict_ref[key].A))

    def test_random_refinement_fv(self):
        lg = pp.CartLeafGrid([5, 5], [1, 1], 2)
        lg.refine_cells(np.random.rand(lg.num_cells) < 0.5)

        keys = ["tpfa", "mpfa"]
        disc = [pp.Tpfa, pp.Mpfa]
        error_tol = [0.06, 1e-10]  # because of consistency error of TPFA,
        # it does not reproduce linear pressure exactly

        left_faces = np.argwhere(lg.face_centers[0] < 1e-5).ravel()
        right_faces = np.argwhere(lg.face_centers[0] > 1 - 1e-5).ravel()

        dir_faces = np.hstack((left_faces, right_faces))
        bound = pp.BoundaryCondition(lg, dir_faces, "dir")

        d = {}
        for key in keys:
            pp.initialize_default_data(lg, d, "flow", {"bc": bound}, key)

        for i, key in enumerate(keys):
            discretization = disc[i](key)
            discretization.discretize(lg, d)

        p_ref = lambda x: x[0]
        p_bc = np.zeros(lg.num_faces)
        p_bc[dir_faces] = p_ref(lg.face_centers[:, dir_faces])

        for key in keys:
            mat_dict = d[pp.DISCRETIZATION_MATRICES][key]
            mat_dict_ref = d[pp.DISCRETIZATION_MATRICES][key]

            flux, bound_flux = mat_dict["flux"], mat_dict["bound_flux"]

            div = lg.cell_faces.T

            A = div * flux

            rhs = -div * bound_flux * p_bc
            p = np.linalg.solve(A.todense(), rhs)

            p_diff = p - p_ref(lg.cell_centers)

            self.assertTrue(np.max(np.abs(p_diff)) < 0.06)
