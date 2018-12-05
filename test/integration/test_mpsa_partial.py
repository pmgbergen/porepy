import numpy as np
import unittest

import porepy as pp


class TestDisplacementReconstruction(unittest.TestCase):
    """
    Class for testing updating the discretization, including the reconstruction
    of gradient displacements. Given a discretization we want
    to rediscretize parts of the domain. This will typically be a change of boundary
    conditions, fracture growth, or a change in aperture.
    """

    def test_no_change_input(self):
        """
        The input matrices should not be changed
        """
        g = pp.CartGrid([4, 4], physdims=(1, 1))
        g.compute_geometry()
        bc = pp.BoundaryConditionVectorial(g)
        k = pp.FourthOrderTensor(g.dim, np.ones(g.num_cells), np.ones(g.num_cells))

        stress, bound_stress, hf_cell, hf_bound = pp.numerics.fv.mpsa.mpsa(
            g, k, bc, inverter="python"
        )

        stress_old = stress.copy()
        bound_stress_old = bound_stress.copy()
        hf_cell_old = hf_cell.copy()
        hf_bound_old = hf_bound.copy()

        # Update should not change anything
        faces = np.array([0, 4, 5, 6])
        _ = pp.numerics.fv.mpsa.mpsa_update_partial(
            stress,
            bound_stress,
            hf_cell,
            hf_bound,
            g,
            k,
            bc,
            faces=faces,
            inverter="python",
        )

        self.assertTrue(np.allclose((stress - stress_old).data, 0))
        self.assertTrue(np.allclose((bound_stress - bound_stress_old).data, 0))
        self.assertTrue(np.allclose((hf_cell - hf_cell_old).data, 0))
        self.assertTrue(np.allclose((hf_bound - hf_bound_old).data, 0))

    def test_cart_2d(self):
        """
        When not changing the parameters the output should equal the input
        """
        g = pp.CartGrid([1, 1], physdims=(1, 1))
        g.compute_geometry()
        bc = pp.BoundaryConditionVectorial(g)
        k = pp.FourthOrderTensor(g.dim, np.ones(g.num_cells), np.ones(g.num_cells))

        stress_full, bound_stress_full, hf_cell_full, hf_bound_full = pp.numerics.fv.mpsa.mpsa(
            g, k, bc, inverter="python"
        )

        # Update should not change anything
        faces = np.array([0, 3])
        stress, bound_stress, hf_cell, hf_bound = pp.numerics.fv.mpsa.mpsa_update_partial(
            stress_full,
            bound_stress_full,
            hf_cell_full,
            hf_bound_full,
            g,
            k,
            bc,
            faces=faces,
            inverter="python",
        )

        self.assertTrue(np.allclose((stress - stress_full).data, 0))
        self.assertTrue(np.allclose((bound_stress - bound_stress_full).data, 0))
        self.assertTrue(np.allclose((hf_cell - hf_cell_full).data, 0))
        self.assertTrue(np.allclose((hf_bound - hf_bound_full).data, 0))

    def test_changing_bc(self):
        """
        We test that we can change the boundary condition
        """
        g = pp.StructuredTriangleGrid([1, 1], physdims=(1, 1))
        g.compute_geometry()
        bc = pp.BoundaryConditionVectorial(g)
        k = pp.FourthOrderTensor(g.dim, np.ones(g.num_cells), np.ones(g.num_cells))
        stress_neu, bound_stress_neu, hf_cell_neu, hf_bound_neu = pp.numerics.fv.mpsa.mpsa(
            g, k, bc, inverter="python"
        )
        faces = 0  # g.get_all_boundary_faces()

        bc.is_dir[:, faces] = True
        bc.is_neu[bc.is_dir] = False

        # Full discretization
        stress_dir, bound_stress_dir, hf_cell_dir, hf_bound_dir = pp.numerics.fv.mpsa.mpsa(
            g, k, bc, inverter="python"
        )

        # Partiall should give same ressult as full

        stress, bound_stress, hf_cell, hf_bound = pp.numerics.fv.mpsa.mpsa_update_partial(
            stress_neu,
            bound_stress_neu,
            hf_cell_neu,
            hf_bound_neu,
            g,
            k,
            bc,
            faces=faces,
            inverter="python",
        )

        self.assertTrue(np.allclose((stress - stress_dir).data, 0))
        self.assertTrue(np.allclose((bound_stress - bound_stress_dir).data, 0))
        self.assertTrue(np.allclose((hf_cell - hf_cell_dir).data, 0))
        self.assertTrue(np.allclose((hf_bound - hf_bound_dir).data, 0))

    def test_changing_bc_by_cells(self):
        """
        Test that we can change the boundary condition by specifying the boundary cells
        """
        g = pp.StructuredTetrahedralGrid([2, 2, 2], physdims=(1, 1, 1))
        g.compute_geometry()
        bc = pp.BoundaryConditionVectorial(g)
        k = pp.FourthOrderTensor(g.dim, np.ones(g.num_cells), np.ones(g.num_cells))
        stress_neu, bound_stress_neu, hf_cell_neu, hf_bound_neu = pp.numerics.fv.mpsa.mpsa(
            g, k, bc, inverter="python"
        )

        faces = g.face_centers[2] < 1e-10

        bc.is_rob[:, faces] = True
        bc.is_neu[bc.is_rob] = False

        # Full discretization
        stress_rob, bound_stress_rob, hf_cell_rob, hf_bound_rob = pp.numerics.fv.mpsa.mpsa(
            g, k, bc, inverter="python"
        )
        # Partiall should give same result as full
        cells = np.argwhere(g.cell_faces[faces, :])[:, 1].ravel()
        cells = np.unique(cells)
        stress, bound_stress, hf_cell, hf_bound = pp.numerics.fv.mpsa.mpsa_update_partial(
            stress_neu,
            bound_stress_neu,
            hf_cell_neu,
            hf_bound_neu,
            g,
            k,
            bc,
            cells=cells,
            inverter="python",
        )
        self.assertTrue(np.allclose((stress - stress_rob).data, 0))
        self.assertTrue(np.allclose((bound_stress - bound_stress_rob).data, 0))
        self.assertTrue(np.allclose((hf_cell - hf_cell_rob).data, 0))
        self.assertTrue(np.allclose((hf_bound - hf_bound_rob).data, 0))

    def test_mixed_bc(self):
        """
        We test that we can change the boundary condition in given direction
        """
        g = pp.StructuredTriangleGrid([2, 2], physdims=(1, 1))
        g.compute_geometry()
        bc = pp.BoundaryConditionVectorial(g)
        k = pp.FourthOrderTensor(g.dim, np.ones(g.num_cells), np.ones(g.num_cells))
        stress_neu, bound_stress_neu, hf_cell_neu, hf_bound_neu = pp.numerics.fv.mpsa.mpsa(
            g, k, bc, inverter="python"
        )

        faces = g.face_centers[0] > 1 - 1e-10

        bc.is_rob[1, faces] = True
        bc.is_neu[bc.is_rob] = False

        # Full discretization
        stress_rob, bound_stress_rob, hf_cell_rob, hf_bound_rob = pp.numerics.fv.mpsa.mpsa(
            g, k, bc, inverter="python"
        )
        # Partiall should give same ressult as full
        stress, bound_stress, hf_cell, hf_bound = pp.numerics.fv.mpsa.mpsa_update_partial(
            stress_neu,
            bound_stress_neu,
            hf_cell_neu,
            hf_bound_neu,
            g,
            k,
            bc,
            faces=faces,
            inverter="python",
        )

        self.assertTrue(np.allclose((stress - stress_rob).data, 0))
        self.assertTrue(np.allclose((bound_stress - bound_stress_rob).data, 0))
        self.assertTrue(np.allclose((hf_cell - hf_cell_rob).data, 0))
        self.assertTrue(np.allclose((hf_bound - hf_bound_rob).data, 0))


if __name__ == "__main__":
    unittest.main()
