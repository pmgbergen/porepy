import numpy as np
import scipy.sparse as sps
import unittest

import porepy as pp


class MpsaReconstructDisplacement(unittest.TestCase):
    def test_cart_2d(self):
        """
        Test that mpsa gives out the correct matrices for
        reconstruction of the displacement at the faces
        """
        nx = 1
        ny = 1
        g = pp.CartGrid([nx, ny], physdims=[2, 2])
        g.compute_geometry()

        lam = np.array([1])
        mu = np.array([2])
        k = pp.FourthOrderTensor(mu, lam)

        bc = pp.BoundaryConditionVectorial(g)
        keyword = "mechanics"

        discr = pp.Mpsa(keyword)

        specified_data = {"fourth_order_tensor": k, "bc": bc, "inverter": "python"}

        data = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr.discretize(g, data)

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][keyword]

        grad_cell = matrix_dictionary[discr.bound_displacment_cell_matrix_key]
        grad_bound = matrix_dictionary[discr.bound_displacment_face_matrix_key]

        hf2f = pp.fvutils.map_hf_2_f(g=g)
        num_subfaces = hf2f.sum(axis=1).A.ravel()
        scaling = sps.dia_matrix(
            (1.0 / num_subfaces, 0), shape=(hf2f.shape[0], hf2f.shape[0])
        )

        hf2f = (scaling * hf2f).toarray()

        grad_bound_known = np.array(
            [
                [0.10416667, 0.0, 0.0, 0.0, 0.0, -0.02083333, 0.0, 0.0],
                [0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.10416667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02083333],
                [0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.10416667, 0.0, 0.0, 0.02083333, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.10416667, 0.0, 0.0, 0.0, 0.0, -0.02083333],
                [0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0],
                [-0.02083333, 0.0, 0.0, 0.0, 0.0, 0.10416667, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.02083333, 0.0, 0.0, 0.10416667, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0],
                [0.02083333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10416667],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0],
                [0.0, 0.0, -0.02083333, 0.0, 0.0, 0.0, 0.0, 0.10416667],
            ]
        )

        grad_cell_known = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        self.assertTrue(np.all(np.abs(grad_bound - hf2f.dot(grad_bound_known)) < 1e-7))
        self.assertTrue(np.all(np.abs(grad_cell - hf2f.dot(grad_cell_known)) < 1e-12))

    def test_simplex_3d_dirichlet(self):
        """
        Test that we retrieve a linear solution exactly
        """
        nx = 2
        ny = 2
        nz = 2
        g = pp.StructuredTetrahedralGrid([nx, ny, nz], physdims=[1, 1, 1])
        g.compute_geometry()

        np.random.seed(2)

        s_t = pp.fvutils.SubcellTopology(g)

        lam = np.ones(g.num_cells)
        mu = np.ones(g.num_cells)
        k = pp.FourthOrderTensor(mu, lam)

        bc = pp.BoundaryConditionVectorial(g)
        bc.is_dir[:, g.get_all_boundary_faces()] = True
        bc.is_neu[bc.is_dir] = False

        x0 = np.array([[1, 2, 3]]).T
        u_b = g.face_centers + x0

        keyword = "mechanics"

        discr = pp.Mpsa(keyword)
        bc_val = u_b.ravel("F")

        specified_data = {
            "fourth_order_tensor": k,
            "bc": bc,
            "inverter": "python",
            "bc_values": bc_val,
        }

        data = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr.discretize(g, data)
        A, b = discr.assemble_matrix_rhs(g, data)

        U = sps.linalg.spsolve(A, b)
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][keyword]

        grad_cell = matrix_dictionary[discr.bound_displacment_cell_matrix_key]
        grad_bound = matrix_dictionary[discr.bound_displacment_face_matrix_key]

        U_f = (grad_cell * U + grad_bound * u_b.ravel("F")).reshape(
            (g.dim, -1), order="F"
        )

        U = U.reshape((g.dim, -1), order="F")

        self.assertTrue(np.all(np.abs(U - g.cell_centers - x0) < 1e-10))
        self.assertTrue(np.all(np.abs(U_f - g.face_centers - x0) < 1e-10))

    def test_simplex_3d_boundary(self):
        """
        Even if we do not get exact solution at interiour we should be able to
        retrieve the boundary conditions
        """
        nx = 2
        ny = 2
        nz = 2
        g = pp.StructuredTetrahedralGrid([nx, ny, nz], physdims=[1, 1, 1])
        g.compute_geometry()

        np.random.seed(2)

        lam = 10 * np.random.rand(g.num_cells)
        mu = 10 * np.random.rand(g.num_cells)
        k = pp.FourthOrderTensor(mu, lam)

        bc = pp.BoundaryConditionVectorial(g)
        dir_ind = g.get_all_boundary_faces()[[0, 2, 5, 8, 10, 13, 15, 21]]
        bc.is_dir[:, dir_ind] = True
        bc.is_neu[bc.is_dir] = False

        u_b = np.random.randn(g.face_centers.shape[0], g.face_centers.shape[1])

        keyword = "mechanics"

        discr = pp.Mpsa(keyword)
        bc_val = u_b.ravel("F")

        specified_data = {
            "fourth_order_tensor": k,
            "bc": bc,
            "inverter": "python",
            "bc_values": bc_val,
        }

        data = pp.initialize_default_data(
            g, {}, keyword, specified_parameters=specified_data
        )

        discr.discretize(g, data)
        A, b = discr.assemble_matrix_rhs(g, data)

        U = sps.linalg.spsolve(A, b)
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][keyword]

        grad_cell = matrix_dictionary[discr.bound_displacment_cell_matrix_key]
        grad_bound = matrix_dictionary[discr.bound_displacment_face_matrix_key]

        U_f = (grad_cell * U + grad_bound * u_b.ravel("F")).reshape(
            (g.dim, -1), order="F"
        )

        # _, IA = np.unique(s_t.fno_unique, True)
        # U_f = U_hf[:, IA]

        self.assertTrue(np.all(np.abs(U_f[:, dir_ind] - u_b[:, dir_ind]) < 1e-10))


if __name__ == "__main__":
    unittest.main()
