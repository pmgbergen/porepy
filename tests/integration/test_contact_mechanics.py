"""
Various integration tests for contact mechanics.
"""
import unittest

import numpy as np

import porepy as pp


def test_contact_mechanics_model_no_modification():
    """Test that the raw contact mechanics model with no modifications can be run with
    no error messages. Failure of this test would signify rather fundamental problems
    in the model.
    """
    model = pp.ContactMechanics({"use_ad": True})
    pp.run_stationary_model(model, {})


class TestContactMechanics(unittest.TestCase):
    def _solve(self, setup):
        pp.run_stationary_model(setup, {"convergence_tol": 1e-10, "max_iterations": 20})
        mdg = setup.mdg

        nd = mdg.dim_max()

        sd_2 = mdg.subdomains(dim=nd)[0]
        sd_1 = mdg.subdomains(dim=nd - 1)[0]
        intf = mdg.subdomain_pair_to_interface((sd_1, sd_2))
        d_m = mdg.interface_data(intf)
        d_1 = mdg.subdomain_data(sd_1)

        u_interface = d_m[pp.STATE][setup.mortar_displacement_variable]
        contact_force = d_1[pp.STATE][setup.contact_traction_variable]

        displacement_jump_global_coord = (
            intf.mortar_to_secondary_avg(nd=nd)
            * intf.sign_of_mortar_sides(nd=nd)
            * u_interface
        )
        projection = d_1["tangential_normal_projection"]

        project_to_local = projection.project_tangential_normal(int(intf.num_cells / 2))
        u_frac_local = project_to_local * displacement_jump_global_coord
        u_frac_local_decomposed = u_frac_local.reshape((nd, -1), order="F")

        contact_force = contact_force.reshape((nd, -1), order="F")

        return u_frac_local_decomposed, contact_force

    def _compare_ad(
        self, ux_south=0, uy_south=0, ux_north=0, uy_north=0, dim=2, dilation_angle=0
    ):
        model = Model(ux_south, uy_south, ux_north, uy_north, dim, dilation_angle)
        model_ad = Model(ux_south, uy_south, ux_north, uy_north, dim, dilation_angle)
        model_ad._use_ad = True
        displacement, force = self._solve(model)
        displacement_ad, force_ad = self._solve(model_ad)
        # Use norm for comparison to allow for non-unique tangential basis
        assert np.all(
            np.isclose(
                np.linalg.norm(displacement, axis=0),
                np.linalg.norm(displacement_ad, axis=0),
                atol=1e-7,
            )
        )
        assert np.all(np.isclose(displacement[-1], displacement_ad[-1]))
        assert np.all(np.isclose(force[-1], force_ad[-1]))
        assert np.all(
            np.isclose(np.linalg.norm(force, axis=0), np.linalg.norm(force_ad, axis=0))
        )
        return displacement, force

    def test_pull_north_positive_opening(self):

        u_frac, contact_force = self._compare_ad(uy_north=0.001)

        # All components should be open in the normal direction
        self.assertTrue(np.all(u_frac[1] > 0))

        # By symmetry (reasonable to expect from this grid), the jump in tangential
        # deformation should be zero.
        self.assertTrue(np.abs(np.sum(u_frac[0])) < 1e-5)

        # The contact force in normal direction should be zero
        # NB: This assumes the contact force is expressed in local coordinates
        self.assertTrue(np.all(np.abs(contact_force) < 1e-7))

    def test_pull_south_positive_opening(self):

        u_frac, contact_force = self._compare_ad(uy_south=-0.001)

        # All components should be open in the normal direction
        self.assertTrue(np.all(u_frac[1] > 0))

        # By symmetry (reasonable to expect from this grid), the jump in tangential
        # deformation should be zero.
        self.assertTrue(np.abs(np.sum(u_frac[0])) < 1e-5)

        # The contact force in normal direction should be zero

        # NB: This assumes the contact force is expressed in local coordinates
        self.assertTrue(np.all(np.abs(contact_force) < 1e-7))

    def test_push_north_zero_opening(self):

        u_frac, contact_force = self._compare_ad(uy_north=-0.001)

        # All components should be closed in the normal direction
        self.assertTrue(np.abs(np.sum(u_frac[1])) < 1e-5)

        # Contact force in normal direction should be negative
        self.assertTrue(np.all(contact_force[1] < 0))

    def test_push_south_zero_opening(self):
        u_frac, contact_force = self._compare_ad(uy_south=0.001)

        # All components should be closed in the normal direction
        self.assertTrue(np.abs(np.sum(u_frac[1])) < 1e-5)

        # Contact force in normal direction should be negative
        self.assertTrue(np.all(contact_force[1] < 0))

    def test_closed_shear(self):

        u_frac, contact_force = self._compare_ad(ux_north=0.1, uy_north=-0.1)

        # All components should be closed in the normal direction
        self.assertTrue(np.all(u_frac[1] < 1e-10))

        # The contact force in normal direction should be nonzero
        # NB: This assumes the contact force is expressed in local coordinates
        self.assertTrue(np.all(np.abs(contact_force) > 1e-5))

    def test_closed_shear_dilation(self):

        u_frac, contact_force = self._compare_ad(
            ux_north=0.1, uy_north=-0.02, dilation_angle=0.1
        )
        # All components should be closed in the normal direction
        self.assertTrue(np.all(u_frac[1] > 1e-10))

        # The contact force in normal direction should be nonzero
        # NB: This assumes the contact force is expressed in local coordinates
        self.assertTrue(np.all(np.abs(contact_force) > 1e-5))

    def test_closed_shear_3d(self):

        u_frac, contact_force = self._compare_ad(ux_north=0.01, uy_north=-0.005, dim=3)

        # All components should be closed in the normal direction
        self.assertTrue(np.all(u_frac[-1] < 1e-10))

        # The contact force should be nonzero
        self.assertTrue(np.all(np.abs(contact_force) > 1e-10))

    def test_closed_shear_3d_dilation(self):

        u_frac, contact_force = self._compare_ad(
            ux_north=0.01, uy_north=-0.001, dim=3, dilation_angle=0.1
        )
        # All components should be closed in the normal direction
        self.assertTrue(np.all(u_frac[-1] > 1e-10))

        # The contact force should be nonzero
        self.assertTrue(np.all(np.abs(contact_force) > 1e-10))

    def test_open_shear_3d(self):

        u_frac, contact_force = self._compare_ad(ux_north=0.01, uy_north=0.01, dim=3)

        # All components should be open in the normal direction
        self.assertTrue(np.all(u_frac[-1] > 1e-5))

        # The contact force should be zero
        self.assertTrue(np.all(np.abs(contact_force) < 1e-10))


class Model(pp.ContactMechanics):
    def __init__(self, ux_south, uy_south, ux_north, uy_north, dim, dilation_angle):
        mesh_args = {
            "mesh_size_frac": 0.5,
            "mesh_size_min": 0.023,
            "mesh_size_bound": 0.5,
        }
        super().__init__(params={"mesh_args": mesh_args, "max_iterations": 25})
        self.ux_south = ux_south
        self.uy_south = uy_south
        self.ux_north = ux_north
        self.uy_north = uy_north
        self.nd = dim
        self.dilation_angle = dilation_angle

    def create_grid(self):
        """
        Method that creates and returns the MixedDimensionalGrid of a 2D domain with six
        fractures. The two sides of the fractures are coupled together with a
        mortar grid.
        """
        rotate_fracture = getattr(self, "rotate_fracture", False)
        self.box = {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}
        if self.nd == 2:
            if rotate_fracture:
                self.frac_pts = np.array([[0.7, 0.3], [0.3, 0.7]])
            else:
                self.frac_pts = np.array([[0.3, 0.7], [0.5, 0.5]])
            frac_edges = np.array([[0], [1]])
            network = pp.FractureNetwork2d(self.frac_pts, frac_edges, domain=self.box)
        else:
            self.box.update({"zmin": 0, "zmax": 1})
            pts = np.array(
                [[0.2, 0.2, 0.8, 0.8], [0.5, 0.5, 0.5, 0.5], [0.2, 0.8, 0.8, 0.2]]
            )
            if rotate_fracture:
                pts[1] = [0.2, 0.2, 0.8, 0.8]
            network = pp.FractureNetwork3d([pp.PlaneFracture(pts)], domain=self.box)

        # Generate the mixed-dimensional mesh
        mdg = network.mesh(self.params["mesh_args"])

        # Set projections to local coordinates for all fractures
        pp.contact_conditions.set_projections(mdg)

        self.mdg = mdg
        self.nd = mdg.dim_max()

    def _bc_values(self, sd):
        _, _, _, north, south, _, _ = self._domain_boundary_sides(sd)
        values = np.zeros((sd.dim, sd.num_faces))
        values[0, south] = self.ux_south
        values[1, south] = self.uy_south
        values[0, north] = self.ux_north
        values[1, north] = self.uy_north
        return values.ravel("F")

    def _bc_type(self, sd):
        _, _, _, north, south, _, _ = self._domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, north + south, "dir")
        # Default internal BC is Neumann. We change to Dirichlet for the contact
        # problem. I.e., the mortar variable represents the displacement on the
        # fracture faces.
        frac_face = sd.tags["fracture_faces"]
        bc.is_neu[:, frac_face] = False
        bc.is_dir[:, frac_face] = True
        return bc

    def _initial_gap(self, sd):
        vals = getattr(self, "initial_gap", np.zeros(sd.num_cells))
        return vals

    def _dilation_angle(self, sd):
        vals = getattr(self, "dilation_angle", 0) * np.ones(sd.num_cells)
        return vals


if __name__ == "__main__":
    unittest.main()
