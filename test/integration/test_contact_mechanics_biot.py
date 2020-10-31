"""
Integration tests for the Biot modell, with and without contact mechanics.

We have the full Biot equations in the matrix, and mass conservation and contact
conditions in the fracture. For the contact mechanical part of this
test, please refer to test_contact_mechanics.
"""
import numpy as np
import unittest

import porepy as pp
import porepy.models.contact_mechanics_biot_model as model
import test.common.contact_mechanics_examples


class TestBiot(unittest.TestCase):
    def _solve(self, setup):
        pp.run_time_dependent_model(setup, {})

        gb = setup.gb

        g = gb.grids_of_dimension(setup.Nd)[0]
        d = gb.node_props(g)

        u = d[pp.STATE][setup.displacement_variable]
        p = d[pp.STATE][setup.scalar_variable]

        return u, p

    def test_pull_north_negative_scalar(self):

        setup = SetupContactMechanicsBiot(
            ux_south=0, uy_south=0, ux_north=0, uy_north=0.001
        )
        setup.with_fracture = False
        setup.mesh_args["mesh_size_bound"] = 0.5
        u, p = self._solve(setup)

        # By symmetry (reasonable to expect from this grid), the average x displacement should be zero
        self.assertTrue(np.abs(np.sum(u[0::2])) < 1e-8)
        # Check that the expansion yields a negative pressure
        self.assertTrue(np.all(p < -1e-8))


class TestContactMechanicsBiot(unittest.TestCase):
    def _solve(self, setup):
        pp.run_time_dependent_model(setup, {"convergence_tol": 1e-6})

        gb = setup.gb

        nd = setup.Nd

        g2 = gb.grids_of_dimension(nd)[0]
        g1 = gb.grids_of_dimension(nd - 1)[0]

        d_m = gb.edge_props((g1, g2))
        d_1 = gb.node_props(g1)

        mg = d_m["mortar_grid"]

        u_mortar = d_m[pp.STATE][setup.mortar_displacement_variable]
        contact_force = d_1[pp.STATE][setup.contact_traction_variable]
        fracture_pressure = d_1[pp.STATE][setup.scalar_variable]

        displacement_jump_global_coord = (
            mg.mortar_to_secondary_avg(nd=nd) * mg.sign_of_mortar_sides(nd=nd) * u_mortar
        )
        projection = d_m["tangential_normal_projection"]

        project_to_local = projection.project_tangential_normal(int(mg.num_cells / 2))
        u_mortar_local = project_to_local * displacement_jump_global_coord
        u_mortar_local_decomposed = u_mortar_local.reshape((nd, -1), order="F")

        contact_force = contact_force.reshape((nd, -1), order="F")

        return u_mortar_local_decomposed, contact_force, fracture_pressure

    def test_pull_north_positive_opening(self):

        setup = SetupContactMechanicsBiot(
            ux_south=0, uy_south=0, ux_north=0, uy_north=0.001
        )
        setup.mesh_args = [2, 2]
        setup.simplex = False
        # setup.subtract_fracture_pressure = False
        u_mortar, contact_force, fracture_pressure = self._solve(setup)
        # All components should be open in the normal direction
        self.assertTrue(np.all(u_mortar[1] > 0))

        # By symmetry (reasonable to expect from this grid), the jump in tangential
        # deformation should be zero.
        self.assertTrue(np.abs(np.sum(u_mortar[0])) < 1e-5)

        # The contact force in normal direction should be zero

        # NB: This assumes the contact force is expressed in local coordinates
        self.assertTrue(np.all(np.abs(contact_force) < 1e-7))

        # Check that the dilation of the fracture yields a negative fracture pressure
        self.assertTrue(np.all(fracture_pressure < -1e-7))

    def test_pull_south_positive_opening(self):

        setup = SetupContactMechanicsBiot(
            ux_south=0, uy_south=-0.001, ux_north=0, uy_north=0
        )

        u_mortar, contact_force, fracture_pressure = self._solve(setup)

        # All components should be open in the normal direction
        self.assertTrue(np.all(u_mortar[1] > 0))

        # By symmetry (reasonable to expect from this grid), the jump in tangential
        # deformation should be zero.
        self.assertTrue(np.abs(np.sum(u_mortar[0])) < 1e-5)

        # The contact force in normal direction should be zero

        # NB: This assumes the contact force is expressed in local coordinates
        self.assertTrue(np.all(np.abs(contact_force) < 1e-7))

        # Check that the dilation of the fracture yields a negative fracture pressure
        self.assertTrue(np.all(fracture_pressure < -1e-7))

    def test_push_north_zero_opening(self):

        setup = SetupContactMechanicsBiot(
            ux_south=0, uy_south=0, ux_north=0, uy_north=-0.001
        )

        u_mortar, contact_force, fracture_pressure = self._solve(setup)

        # All components should be closed in the normal direction
        self.assertTrue(np.abs(np.sum(u_mortar[1])) < 1e-5)

        # Contact force in normal direction should be negative
        self.assertTrue(np.all(contact_force[1] < 0))

        # Compression of the domain yields a (slightly) positive fracture pressure
        self.assertTrue(np.all(fracture_pressure > 1e-10))

    def test_push_south_zero_opening(self):

        setup = SetupContactMechanicsBiot(
            ux_south=0, uy_south=0.001, ux_north=0, uy_north=0
        )

        u_mortar, contact_force, fracture_pressure = self._solve(setup)

        # All components should be closed in the normal direction
        self.assertTrue(np.abs(np.sum(u_mortar[1])) < 1e-5)

        # Contact force in normal direction should be negative
        self.assertTrue(np.all(contact_force[1] < 0))

        # Compression of the domain yields a (slightly) positive fracture pressure
        self.assertTrue(np.all(fracture_pressure > 1e-10))

    def test_positive_fracture_pressure_positive_opening(self):

        setup = SetupContactMechanicsBiot(
            ux_south=0, uy_south=0, ux_north=0, uy_north=0, source_value=0.001
        )

        u_mortar, contact_force, fracture_pressure = self._solve(setup)

        # All components should be open in the normal direction
        self.assertTrue(np.all(u_mortar[1] > 0))

        # By symmetry (reasonable to expect from this grid), the jump in tangential
        # deformation should be zero.
        self.assertTrue(np.abs(np.sum(u_mortar[0])) < 1e-5)

        # The contact force in normal direction should be zero

        # NB: This assumes the contact force is expressed in local coordinates
        self.assertTrue(np.all(np.abs(contact_force) < 1e-7))

        # Fracture pressure is positive
        self.assertTrue(np.all(fracture_pressure > 1e-7))

    def test_time_dependent_pull_north_negative_scalar(self):
        """ To obtain the value used in the corresponding TM test, 
        test_pull_north_reduce_to_tm, uncomment the line
        setup.subtract_fracture_pressure = False
        """
        setup = SetupContactMechanicsBiot(
            ux_south=0, uy_south=0, ux_north=0, uy_north=0.001
        )
        setup.end_time *= 3
        setup.mesh_args = [2, 2]
        setup.simplex = False
        # setup.subtract_fracture_pressure = False
        u_mortar, contact_force, fracture_pressure = self._solve(setup)
        # All components should be open in the normal direction
        self.assertTrue(np.all(u_mortar[1] > 0))

        # By symmetry (reasonable to expect from this grid), the jump in tangential
        # deformation should be zero.
        self.assertTrue(np.abs(np.sum(u_mortar[0])) < 1e-5)

        # The contact force in normal direction should be zero

        # NB: This assumes the contact force is expressed in local coordinates
        self.assertTrue(np.all(np.abs(contact_force) < 1e-7))

        # Check that the dilation of the fracture yields a negative fracture pressure
        self.assertTrue(np.all(fracture_pressure < -1e-7))
        # If the update of the mechanical BC values for the previous time step used in
        # div u is missing, the effect is similar to if the pull on the north is
        # increased in each time step. This leads to a too small fracture pressure.
        self.assertTrue(np.all(np.isclose(fracture_pressure, -4.31072866e-06)))


class SetupContactMechanicsBiot(
    test.common.contact_mechanics_examples.ProblemDataTime, model.ContactMechanicsBiot
):
    def __init__(self, ux_south, uy_south, ux_north, uy_north, source_value=0):

        self.mesh_args = {
            "mesh_size_frac": 0.5,
            "mesh_size_min": 0.023,
            "mesh_size_bound": 0.5,
        }

        super().__init__()

        self.ux_south = ux_south
        self.uy_south = uy_south
        self.ux_north = ux_north
        self.uy_north = uy_north
        self.scalar_source_value = source_value


if __name__ == "__main__":
    unittest.main()
