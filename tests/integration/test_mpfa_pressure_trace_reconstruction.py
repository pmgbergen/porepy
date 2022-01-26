import unittest

import numpy as np
import scipy.sparse as sps

import porepy as pp


class MpfaReconstructPressure(unittest.TestCase):
    def test_cart_2d(self):
        """
        Test that mpfa gives out the correct matrices for
        reconstruction of the pressures at the faces
        """
        nx = 1
        ny = 1
        g = pp.CartGrid([nx, ny], physdims=[2, 2])
        g.compute_geometry()

        kxx = np.array([2])
        k = pp.SecondOrderTensor(kxx)

        bc = pp.BoundaryCondition(g)

        mpfa = pp.Mpfa("flow")
        _, _, grad_cell, grad_bound, *_ = mpfa._flux_discretization(
            g, k, bc, inverter="python"
        )

        grad_bound_known = np.array(
            [
                [-0.25, 0.0, 0.0, 0.0],
                [0.0, -0.25, 0.0, 0.0],
                [0.0, 0.0, -0.25, 0.0],
                [0.0, 0.0, 0.0, -0.25],
            ]
        )

        grad_cell_known = np.array([[1.0], [1.0], [1.0], [1.0]])

        self.assertTrue(np.all(np.abs(grad_bound - grad_bound_known) < 1e-7))
        self.assertTrue(np.all(np.abs(grad_cell - grad_cell_known) < 1e-12))

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

        kxx = np.ones(g.num_cells)
        k = pp.SecondOrderTensor(kxx)

        s_t = pp.fvutils.SubcellTopology(g)

        bc = pp.BoundaryCondition(g)
        bc.is_dir[g.get_all_boundary_faces()] = True
        bc.is_neu[bc.is_dir] = False

        p0 = 1
        p_b = np.sum(g.face_centers, axis=0) + p0

        mpfa = pp.Mpfa("flow")

        flux, bound_flux, p_t_cell, p_t_bound, *_ = mpfa._flux_discretization(
            g, k, bc, eta=0, inverter="python"
        )

        div = pp.fvutils.scalar_divergence(g)

        P = sps.linalg.spsolve(div * flux, -div * bound_flux * p_b)

        P_f = p_t_cell * P + p_t_bound * p_b

        self.assertTrue(np.all(np.abs(P - np.sum(g.cell_centers, axis=0) - p0) < 1e-10))
        self.assertTrue(
            np.all(np.abs(P_f - np.sum(g.face_centers, axis=0) - p0) < 1e-10)
        )

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

        kxx = 10 * np.random.rand(g.num_cells)
        k = pp.SecondOrderTensor(kxx)

        bc = pp.BoundaryCondition(g)
        dir_ind = g.get_all_boundary_faces()[[0, 2, 5, 8, 10, 13, 15, 21]]

        bc.is_dir[dir_ind] = True
        bc.is_neu[bc.is_dir] = False

        p_b = np.random.randn(g.face_centers.shape[1])

        mpfa = pp.Mpfa("flow")
        flux, bound_flux, p_t_cell, p_t_bound, *_ = mpfa._flux_discretization(
            g, k, bc, eta=0, inverter="python"
        )

        div = pp.fvutils.scalar_divergence(g)

        P = sps.linalg.spsolve(div * flux, -div * bound_flux * p_b.ravel("F"))

        P_f = p_t_cell * P + p_t_bound * p_b

        self.assertTrue(np.all(np.abs(P_f[dir_ind] - p_b[dir_ind]) < 1e-10))

    def test_simplex_3d_sub_face(self):
        """
        Test that we reconstruct the exact solution on subfaces
        """
        nx = 2
        ny = 2
        nz = 2
        g = pp.StructuredTetrahedralGrid([nx, ny, nz], physdims=[1, 1, 1])
        g.compute_geometry()
        s_t = pp.fvutils.SubcellTopology(g)

        kxx = 10 * np.ones(g.num_cells)
        k = pp.SecondOrderTensor(kxx)

        bc = pp.BoundaryCondition(g)
        bc.is_dir[g.get_all_boundary_faces()] = True
        bc.is_neu[bc.is_dir] = False
        bc = pp.fvutils.boundary_to_sub_boundary(bc, s_t)

        p_b = np.sum(-g.face_centers, axis=0)
        p_b = p_b[s_t.fno_unique]

        mpfa = pp.Mpfa("flow")
        flux, bound_flux, p_t_cell, p_t_bound, *_ = mpfa._flux_discretization(
            g, k, bc, eta=0, inverter="python"
        )

        div = pp.fvutils.scalar_divergence(g)

        hf2f = pp.fvutils.map_hf_2_f(nd=1, g=g)
        P = sps.linalg.spsolve(div * hf2f * flux, -div * hf2f * bound_flux * p_b)

        P_hf = p_t_cell * P + p_t_bound * p_b
        _, IA = np.unique(s_t.fno_unique, True)
        P_f = P_hf[IA]

        self.assertTrue(np.all(np.abs(P + np.sum(g.cell_centers, axis=0)) < 1e-10))
        self.assertTrue(np.all(np.abs(P_f + np.sum(g.face_centers, axis=0)) < 1e-10))


if __name__ == "__main__":
    unittest.main()
