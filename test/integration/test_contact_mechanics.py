"""
Various integration tests for contact mechanics.
"""
import numpy as np
import unittest

import porepy as pp
import test.common.contact_mechanics_examples


class TestContactMechanics(unittest.TestCase):
    def _solve(self, setup):
        pp.run_stationary_model(setup, {"convergence_tol": 1e-10})
        gb = setup.gb

        nd = gb.dim_max()

        g2 = gb.grids_of_dimension(2)[0]
        g1 = gb.grids_of_dimension(1)[0]

        d_m = gb.edge_props((g1, g2))
        d_1 = gb.node_props(g1)

        mg = d_m["mortar_grid"]

        u_mortar = d_m[pp.STATE][setup.mortar_displacement_variable]
        contact_force = d_1[pp.STATE][setup.contact_traction_variable]

        displacement_jump_global_coord = (
            mg.mortar_to_secondary_avg(nd=nd) * mg.sign_of_mortar_sides(nd=nd) * u_mortar
        )
        projection = d_m["tangential_normal_projection"]

        project_to_local = projection.project_tangential_normal(int(mg.num_cells / 2))
        u_mortar_local = project_to_local * displacement_jump_global_coord
        u_mortar_local_decomposed = u_mortar_local.reshape((2, -1), order="F")

        contact_force = contact_force.reshape((2, -1), order="F")

        return u_mortar_local_decomposed, contact_force

    def test_pull_north_positive_opening(self):

        setup = SetupContactMechanics(
            ux_south=0, uy_south=0, ux_north=0, uy_north=0.001
        )

        u_mortar, contact_force = self._solve(setup)

        # All components should be open in the normal direction
        self.assertTrue(np.all(u_mortar[1] > 0))

        # By symmetry (reasonable to expect from this grid), the jump in tangential
        # deformation should be zero.
        self.assertTrue(np.abs(np.sum(u_mortar[0])) < 1e-5)

        # The contact force in normal direction should be zero
        # NB: This assumes the contact force is expressed in local coordinates
        self.assertTrue(np.all(np.abs(contact_force) < 1e-7))

    def test_pull_south_positive_opening(self):

        setup = SetupContactMechanics(
            ux_south=0, uy_south=-0.001, ux_north=0, uy_north=0
        )

        u_mortar, contact_force = self._solve(setup)

        # All components should be open in the normal direction
        self.assertTrue(np.all(u_mortar[1] > 0))

        # By symmetry (reasonable to expect from this grid), the jump in tangential
        # deformation should be zero.
        self.assertTrue(np.abs(np.sum(u_mortar[0])) < 1e-5)

        # The contact force in normal direction should be zero

        # NB: This assumes the contact force is expressed in local coordinates
        self.assertTrue(np.all(np.abs(contact_force) < 1e-7))

    def test_push_north_zero_opening(self):

        setup = SetupContactMechanics(
            ux_south=0, uy_south=0, ux_north=0, uy_north=-0.001
        )

        u_mortar, contact_force = self._solve(setup)
        # All components should be closed in the normal direction
        self.assertTrue(np.abs(np.sum(u_mortar[1])) < 1e-5)

        # Contact force in normal direction should be negative
        self.assertTrue(np.all(contact_force[1] < 0))

    def test_push_south_zero_opening(self):

        setup = SetupContactMechanics(
            ux_south=0, uy_south=0.001, ux_north=0, uy_north=0
        )

        u_mortar, contact_force = self._solve(setup)

        # All components should be closed in the normal direction
        self.assertTrue(np.abs(np.sum(u_mortar[1])) < 1e-5)

        # Contact force in normal direction should be negative
        self.assertTrue(np.all(contact_force[1] < 0))


class SetupContactMechanics(
    test.common.contact_mechanics_examples.ContactMechanicsExample
):
    def __init__(self, ux_south, uy_south, ux_north, uy_north):
        mesh_args = {
            "mesh_size_frac": 0.5,
            "mesh_size_min": 0.023,
            "mesh_size_bound": 0.5,
        }
        super().__init__(mesh_args, folder_name="dummy", params={"max_iterations": 25})
        self.ux_south = ux_south
        self.uy_south = uy_south
        self.ux_north = ux_north
        self.uy_north = uy_north

    def create_grid(self):
        """
        Method that creates and returns the GridBucket of a 2D domain with six
        fractures. The two sides of the fractures are coupled together with a
        mortar grid.
        """
        rotate_fracture = getattr(self, "rotate_fracture", False)
        if rotate_fracture:
            self.frac_pts = np.array([[0.7, 0.3], [0.3, 0.7]])
        else:
            self.frac_pts = np.array([[0.3, 0.7], [0.5, 0.5]])
        frac_edges = np.array([[0], [1]])

        self.box = {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}

        network = pp.FractureNetwork2d(self.frac_pts, frac_edges, domain=self.box)
        # Generate the mixed-dimensional mesh
        gb = network.mesh(self.mesh_args)

        # Set projections to local coordinates for all fractures
        pp.contact_conditions.set_projections(gb)

        self.gb = gb
        self.Nd = gb.dim_max()

    def bc_values(self, g):
        _, _, _, north, south, _, _ = self.domain_boundary_sides(g)
        values = np.zeros((g.dim, g.num_faces))
        values[0, south] = self.ux_south
        values[1, south] = self.uy_south
        values[0, north] = self.ux_north
        values[1, north] = self.uy_north
        return values.ravel("F")

    def bc_type(self, g):
        _, _, _, north, south, _, _ = self.domain_boundary_sides(g)
        bc = pp.BoundaryConditionVectorial(g, north + south, "dir")
        # Default internal BC is Neumann. We change to Dirichlet for the contact
        # problem. I.e., the mortar variable represents the displacement on the
        # fracture faces.
        frac_face = g.tags["fracture_faces"]
        bc.is_neu[:, frac_face] = False
        bc.is_dir[:, frac_face] = True
        return bc

    def set_parameters(self):
        super().set_parameters()
        dilation_angle = getattr(self, "dilation_angle", 0)
        for g, d in self.gb:
            if g.dim < self.Nd:

                initial_gap = getattr(self, "initial_gap", np.zeros(g.num_cells))

                d[pp.PARAMETERS]["mechanics"].update(
                    {"initial_gap": initial_gap, "dilation_angle": dilation_angle}
                )


if __name__ == "__main__":
    unittest.main()
