import unittest
import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.fv import mpsa


class TestPartialMPFA(unittest.TestCase):
    def setup(self):
        g = pp.CartGrid([5, 5])
        g.compute_geometry()
        perm = pp.SecondOrderTensor(np.ones(g.num_cells))
        bnd = pp.BoundaryCondition(g)
        flux, bound_flux, _, _ = pp.Mpfa("flow")._local_discr(
            g, perm, bnd, inverter="python"
        )
        return g, perm, bnd, flux, bound_flux

    def test_inner_cell_node_keyword(self):
        # Compute update for a single cell in the interior.
        g, perm, bnd, flux, bound_flux = self.setup()

        inner_cell = 12
        nodes_of_cell = np.array([14, 15, 20, 21])
        faces_of_cell = np.array([14, 15, 42, 47])

        partial_flux, partial_bound, _, _, active_faces = pp.Mpfa("flow").partial_discr(
            g, perm, bnd, nodes=nodes_of_cell, inverter="python"
        )

        self.assertTrue(faces_of_cell.size == active_faces.size)
        self.assertTrue(np.all(np.sort(faces_of_cell) == np.sort(active_faces)))

        diff_flux = (flux - partial_flux).todense()
        diff_bound = (bound_flux - partial_bound).todense()

        self.assertTrue(np.max(np.abs(diff_flux[faces_of_cell])) == 0)
        self.assertTrue(np.max(np.abs(diff_bound[faces_of_cell])) == 0)

        # Only the faces of the central cell should be zero
        partial_flux[faces_of_cell, :] = 0
        partial_bound[faces_of_cell, :] = 0
        self.assertTrue(np.max(np.abs(partial_flux.data)) == 0)
        self.assertTrue(np.max(np.abs(partial_bound.data)) == 0)

    def test_bound_cell_node_keyword(self):
        # Compute update for a single cell on the boundary
        g, perm, bnd, flux, bound_flux = self.setup()

        inner_cell = 10
        nodes_of_cell = np.array([12, 13, 18, 19])
        faces_of_cell = np.array([12, 13, 40, 45])
        partial_flux, partial_bound, _, _, active_faces = pp.Mpfa("flow").partial_discr(
            g, perm, bnd, nodes=nodes_of_cell, inverter="python"
        )

        self.assertTrue(faces_of_cell.size == active_faces.size)
        self.assertTrue(np.all(np.sort(faces_of_cell) == np.sort(active_faces)))

        diff_flux = (flux - partial_flux).todense()
        diff_bound = (bound_flux - partial_bound).todense()

        self.assertTrue(np.max(np.abs(diff_flux[faces_of_cell])) == 0)
        self.assertTrue(np.max(np.abs(diff_bound[faces_of_cell])) == 0)

        # Only the faces of the central cell should be zero
        partial_flux[faces_of_cell, :] = 0
        partial_bound[faces_of_cell, :] = 0
        self.assertTrue(np.max(np.abs(partial_flux.data)) == 0)
        self.assertTrue(np.max(np.abs(partial_bound.data)) == 0)

    def test_one_cell_a_time_node_keyword(self):
        # Update one and one cell, and verify that the result is the same as
        # with a single computation.
        # The test is similar to what will happen with a memory-constrained
        # splitting.
        g = pp.CartGrid([3, 3])
        g.compute_geometry()

        # Assign random permeabilities, for good measure
        np.random.seed(42)
        kxx = np.random.random(g.num_cells)
        kyy = np.random.random(g.num_cells)
        # Ensure positive definiteness
        kxy = np.random.random(g.num_cells) * kxx * kyy
        perm = pp.SecondOrderTensor(kxx=kxx, kyy=kyy, kxy=kxy)

        flux = sps.csr_matrix((g.num_faces, g.num_cells))
        bound_flux = sps.csr_matrix((g.num_faces, g.num_faces))
        faces_covered = np.zeros(g.num_faces, np.bool)

        bnd = pp.BoundaryCondition(g)

        cn = g.cell_nodes()
        for ci in range(g.num_cells):
            ind = np.zeros(g.num_cells)
            ind[ci] = 1
            nodes = np.squeeze(np.where(cn * ind > 0))
            partial_flux, partial_bound, _, _, active_faces = pp.Mpfa(
                "flow"
            ).partial_discr(g, perm, bnd, nodes=nodes, inverter="python")

            if np.any(faces_covered):
                partial_flux[faces_covered, :] *= 0
                partial_bound[faces_covered, :] *= 0
            faces_covered[active_faces] = True

            flux += partial_flux
            bound_flux += partial_bound

        flux_full, bound_flux_full, _, _ = pp.Mpfa("flow")._local_discr(
            g, perm, bnd, inverter="python"
        )

        self.assertTrue((flux_full - flux).max() < 1e-8)
        self.assertTrue((flux_full - flux).min() > -1e-8)
        self.assertTrue((bound_flux - bound_flux_full).max() < 1e-8)
        self.assertTrue((bound_flux - bound_flux_full).min() > -1e-8)


class TestPartialMPSA(unittest.TestCase):
    def setup(self):
        g = pp.CartGrid([5, 5])
        g.compute_geometry()
        stiffness = pp.FourthOrderTensor(np.ones(g.num_cells), np.ones(g.num_cells))
        bnd = pp.BoundaryConditionVectorial(g)
        stress, bound_stress, _, _ = mpsa.mpsa(g, stiffness, bnd, inverter="python")

        return g, stiffness, bnd, stress, bound_stress

    def expand_indices_nd(self, ind, nd, direction="F"):
        dim_inds = np.arange(nd)
        dim_inds = dim_inds[:, np.newaxis]  # Prepare for broadcasting
        new_ind = nd * ind + dim_inds
        new_ind = new_ind.ravel(direction)
        return new_ind

    def test_inner_cell_node_keyword(self):
        # Compute update for a single cell in the interior.
        g, stiffness, bnd, stress, bound_stress = self.setup()

        inner_cell = 12
        nodes_of_cell = np.array([14, 15, 20, 21])
        faces_of_cell = np.array([14, 15, 42, 47])

        partial_stress, partial_bound, active_faces = mpsa.mpsa_partial(
            g, stiffness, bnd, nodes=nodes_of_cell, inverter="python"
        )

        self.assertTrue(faces_of_cell.size == active_faces.size)
        self.assertTrue(np.all(np.sort(faces_of_cell) == np.sort(active_faces)))

        diff_stress = (stress - partial_stress).todense()
        diff_bound = (bound_stress - partial_bound).todense()

        faces_of_cell = self.expand_indices_nd(faces_of_cell, g.dim)
        self.assertTrue(np.max(np.abs(diff_stress[faces_of_cell])) == 0)
        self.assertTrue(np.max(np.abs(diff_bound[faces_of_cell])) == 0)

        # Only the faces of the central cell should be zero
        partial_stress[faces_of_cell, :] = 0
        partial_bound[faces_of_cell, :] = 0
        self.assertTrue(np.max(np.abs(partial_stress.data)) == 0)
        self.assertTrue(np.max(np.abs(partial_bound.data)) == 0)

    def test_bound_cell_node_keyword(self):
        # Compute update for a single cell on the boundary
        g, perm, bnd, stress, bound_stress = self.setup()

        inner_cell = 10
        nodes_of_cell = np.array([12, 13, 18, 19])
        faces_of_cell = np.array([12, 13, 40, 45])
        partial_stress, partial_bound, active_faces = mpsa.mpsa_partial(
            g, perm, bnd, nodes=nodes_of_cell, inverter="python"
        )

        self.assertTrue(faces_of_cell.size == active_faces.size)
        self.assertTrue(np.all(np.sort(faces_of_cell) == np.sort(active_faces)))

        faces_of_cell = self.expand_indices_nd(faces_of_cell, g.dim)
        diff_stress = (stress - partial_stress).todense()
        diff_bound = (bound_stress - partial_bound).todense()

        self.assertTrue(np.max(np.abs(diff_stress[faces_of_cell])) == 0)
        self.assertTrue(np.max(np.abs(diff_bound[faces_of_cell])) == 0)

        # Only the faces of the central cell should be non-zero.
        # Zero out these ones, and the entire
        partial_stress[faces_of_cell, :] = 0
        partial_bound[faces_of_cell, :] = 0
        self.assertTrue(np.max(np.abs(partial_stress.data)) == 0)
        self.assertTrue(np.max(np.abs(partial_bound.data)) == 0)

    def test_one_cell_a_time_node_keyword(self):
        # Update one and one cell, and verify that the result is the same as
        # with a single computation. The test is similar to what will happen
        # with a memory-constrained splitting.
        g = pp.CartGrid([3, 3])
        g.compute_geometry()

        # Assign random permeabilities, for good measure
        np.random.seed(42)
        mu = np.random.random(g.num_cells)
        lmbda = np.random.random(g.num_cells)
        stiffness = pp.FourthOrderTensor(mu=mu, lmbda=lmbda)

        stress = sps.csr_matrix((g.num_faces * g.dim, g.num_cells * g.dim))
        bound_stress = sps.csr_matrix((g.num_faces * g.dim, g.num_faces * g.dim))
        faces_covered = np.zeros(g.num_faces, np.bool)

        bnd = pp.BoundaryConditionVectorial(g)
        stress_full, bound_stress_full, _, _ = mpsa.mpsa(
            g, stiffness, bnd, inverter="python"
        )

        cn = g.cell_nodes()
        for ci in range(g.num_cells):
            ind = np.zeros(g.num_cells)
            ind[ci] = 1
            nodes = np.squeeze(np.where(cn * ind > 0))
            partial_stress, partial_bound, active_faces = mpsa.mpsa_partial(
                g, stiffness, bnd, nodes=nodes, inverter="python"
            )

            if np.any(faces_covered):
                del_faces = self.expand_indices_nd(np.where(faces_covered)[0], g.dim)
                partial_stress[del_faces, :] *= 0
                partial_bound[del_faces, :] *= 0
            faces_covered[active_faces] = True

            stress += partial_stress
            bound_stress += partial_bound

        self.assertTrue((stress_full - stress).max() < 1e-8)
        self.assertTrue((stress_full - stress).min() > -1e-8)
        self.assertTrue((bound_stress - bound_stress_full).max() < 1e-8)
        self.assertTrue((bound_stress - bound_stress_full).min() > -1e-8)


if __name__ == "__main__":
    unittest.main()
