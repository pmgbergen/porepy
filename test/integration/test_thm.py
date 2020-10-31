"""
Integration tests for the Biot modell, with and without contact mechanics.

We have the full Biot equations in the matrix, and mass conservation and contact
conditions in the non-intersecting fracture. For the contact mechanical part of this
test, please refer to test_contact_mechanics.
"""
import numpy as np
import unittest

import porepy as pp
import porepy.models.thm_model as model
from test.common.contact_mechanics_examples import ProblemDataTime


class TestTHM(unittest.TestCase):
    def _solve(self, setup):
        pp.run_time_dependent_model(setup, params={})

        gb = setup.gb

        g = gb.grids_of_dimension(setup.Nd)[0]
        d = gb.node_props(g)

        u = d[pp.STATE][setup.displacement_variable]
        p = d[pp.STATE][setup.scalar_variable]
        T = d[pp.STATE][setup.temperature_variable]

        return u, p, T

    def test_pull_north_negative_scalars(self):

        setup = SetupTHM(ux_south=0, uy_south=0, ux_north=0, uy_north=0.001)
        setup.with_fracture = False

        u, p, T = self._solve(setup)

        # By symmetry (reasonable to expect from this grid), the average x displacement should be zero
        self.assertTrue(np.abs(np.sum(u[0::2])) < 1e-8)
        # Check that the expansion yields a negative pressure
        self.assertTrue(np.all(p < -1e-8))
        # and a negative temperature
        self.assertTrue(np.all(T < -1e-8))

    def test_push_south_positive_scalars(self):

        setup = SetupTHM(ux_south=0, uy_south=0.001, ux_north=0, uy_north=0)
        setup.with_fracture = False

        u, p, T = self._solve(setup)

        # By symmetry (reasonable to expect from this grid), the average x displacement should be zero
        self.assertTrue(np.abs(np.sum(u[0::2])) < 1e-8)
        # Check that the expansion yields a positive pressure
        self.assertTrue(np.all(p > -1e-8))
        # and a positive temperature
        self.assertTrue(np.all(T > -1e-8))

    def test_pull_north_reduce_to_biot(self):
        """
        With vanishing TH and TM coupling terms, the p and u solutions should approach the
        pure HM case.
        The hardcoded values below are taken from the corresponding biot test, i.e.
        test_pull_north_negative_scalar of TestBiot in test_contact_mechanics_biot.
        They should be changed iff the values from that test change.
        """
        setup = SetupTHM(ux_south=0, uy_south=0, ux_north=0, uy_north=0.001)
        setup.with_fracture = False
        setup.beta, setup.gamma = 1e-10, 1e-10
        u, p, T = self._solve(setup)

        self.assertTrue(np.isclose(np.sum(p), -0.000560668482409941))
        self.assertTrue(np.isclose(np.sum(u), 0.0045))
        self.assertTrue(np.isclose(np.sum(T), 0.0))

    def test_pull_north_reduce_to_TM(self):
        """
        With vanishing TH and TM coupling terms, the p and u solutions should approach the
        pure HM case.
        The hardcoded values below are taken from the corresponding biot test, i.e.
        test_pull_north_negative_scalar of TestBiot in test_contact_mechanics_biot.
        They should be changed iff the values from that test change.
        """
        setup = SetupTHM(ux_south=0, uy_south=0, ux_north=0, uy_north=0.001)
        setup.with_fracture = False
        setup.T_0_Kelvin = 1
        setup.alpha, setup.gamma, setup.advection_weight = 1e-10, 1e-10, 1e-10
        u, p, T = self._solve(setup)

        self.assertTrue(np.isclose(np.sum(T), -0.000560668482409941))
        self.assertTrue(np.isclose(np.sum(u), 0.0045))
        self.assertTrue(np.isclose(np.sum(p), 0.0))


class TestContactMechanicsTHM(unittest.TestCase):
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
        fracture_temperature = d_1[pp.STATE][setup.temperature_variable]

        displacement_jump_global_coord = (
            mg.mortar_to_secondary_avg(nd=nd) * mg.sign_of_mortar_sides(nd=nd) * u_mortar
        )
        projection = d_m["tangential_normal_projection"]

        project_to_local = projection.project_tangential_normal(int(mg.num_cells / 2))
        u_mortar_local = project_to_local * displacement_jump_global_coord
        u_mortar_local_decomposed = u_mortar_local.reshape((nd, -1), order="F")

        contact_force = contact_force.reshape((nd, -1), order="F")

        return (
            u_mortar_local_decomposed,
            contact_force,
            fracture_pressure,
            fracture_temperature,
        )

    def test_pull_north_positive_opening(self):

        setup = SetupTHM(ux_south=0, uy_south=0, ux_north=0, uy_north=0.001)

        u_mortar, contact_force, fracture_pressure, fracture_temperature = self._solve(
            setup
        )

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
        # Check that the dilation of the fracture yields a negative fracture temperature
        self.assertTrue(np.all(fracture_temperature < -1e-7))

    def test_pull_south_positive_opening(self):

        setup = SetupTHM(ux_south=0, uy_south=-0.001, ux_north=0, uy_north=0)

        u_mortar, contact_force, fracture_pressure, fracture_temperature = self._solve(
            setup
        )

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

        setup = SetupTHM(ux_south=0, uy_south=0, ux_north=0, uy_north=-0.001)

        u_mortar, contact_force, fracture_pressure, fracture_temperature = self._solve(
            setup
        )

        # All components should be closed in the normal direction
        self.assertTrue(np.abs(np.sum(u_mortar[1])) < 1e-5)

        # Contact force in normal direction should be negative
        self.assertTrue(np.all(contact_force[1] < 0))

        # Compression of the domain yields a (slightly) positive fracture pressure
        self.assertTrue(np.all(fracture_pressure > 1e-10))
        # Compression of the domain yields a (slightly) positive fracture temperature
        self.assertTrue(np.all(fracture_temperature > 1e-10))

    def test_time_dependent_pull_north_positive_opening(self):

        setup = SetupTHM(ux_south=0, uy_south=0, ux_north=0, uy_north=0.001)
        setup.end_time *= 3
        setup.mesh_args = [2, 2]
        setup.simplex = False
        setup.gamma = -1e-2
        setup.T_0_Kelvin = 1
        u_mortar, contact_force, fracture_pressure, fracture_temperature = self._solve(
            setup
        )

        # All components should be open in the normal direction
        self.assertTrue(np.all(u_mortar[1] > 0))

        # By symmetry (reasonable to expect from this grid), the jump in tangential
        # deformation should be zero.
        self.assertTrue(np.abs(np.sum(u_mortar[0])) < 1e-5)

        # The contact force in normal direction should be zero

        # NB: This assumes the contact force is expressed in local coordinates
        self.assertTrue(np.all(np.abs(contact_force) < 1e-7))

        # If the update of the mechanical BC values for the previous time step used in
        # div u is missing, the effect is similar to if the pull on the north is
        # increased in each time step. This leads to a too small fracture pressure.
        # Note that the pressure is different to the corresponding HM test, as there
        # is a feedback from the temperature, which is also reduced.
        self.assertTrue(np.all(np.isclose(fracture_pressure, -3.67936546e-06)))
        self.assertTrue(np.all(np.isclose(fracture_temperature, -3.67936546e-06)))

    def test_pull_north_reduce_to_biot(self):
        """
        With vanishing TH and TM coupling terms, the p and u solutions should approach the
        pure HM case.
        The hardcoded values below are taken from the corresponding biot test, i.e.
        test_pull_north_negative_scalar of TestBiot in test_contact_mechanics_biot.
        They should be changed iff the values from that test change.
        """
        setup = SetupTHM(ux_south=0, uy_south=0, ux_north=0, uy_north=0.001)
        setup.beta, setup.gamma = 1e-12, 1e-12
        setup.end_time *= 3
        setup.mesh_args = [2, 2]
        setup.simplex = False
        #
        u_mortar, contact_force, fracture_pressure, fracture_temperature = self._solve(
            setup
        )
        self.assertTrue(np.all(np.isclose(fracture_pressure, -4.31072866e-06)))
        self.assertTrue(np.all(np.isclose(u_mortar[1], 9.99187742e-04)))
        self.assertTrue(np.all(np.isclose(fracture_temperature, 0.0)))

    def test_pull_north_reduce_to_TM(self):
        """
        With vanishing TH and TM coupling terms, the p and u solutions should approach the
        pure HM case.
        The hardcoded values below are taken from the corresponding biot test, i.e.
        test_pull_north_negative_scalar of TestBiot in test_contact_mechanics_biot.
        They should be changed iff the values from that test change. For compatability with
        TM, the HM variant should be run with subtract_fracture_pressure=False (this leads
        to slightly more opening).
        """
        setup = SetupTHM(ux_south=0, uy_south=0, ux_north=0, uy_north=0.001)
        setup.end_time *= 3
        setup.mesh_args = [2, 2]
        setup.simplex = False
        setup.T_0_Kelvin = 1
        setup.alpha, setup.gamma, setup.advection_weight = 1e-15, 1e-15, 1e-15
        setup.subtract_fracture_pressure = False
        u_mortar, contact_force, fracture_pressure, fracture_temperature = self._solve(
            setup
        )
        self.assertTrue(np.all(np.isclose(fracture_temperature, -2.05192594e-06)))
        self.assertTrue(np.all(np.isclose(u_mortar[1], 1.00061081e-03)))
        self.assertTrue(np.all(np.isclose(fracture_pressure, 0.0)))


class SetupTHM(ProblemDataTime, model.THM):
    def __init__(self, ux_south, uy_south, ux_north, uy_north):

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
        self.scalar_source_value = 0

    def biot_alpha(self, g):
        if hasattr(self, "alpha"):
            return self.alpha
        else:
            return super().biot_alpha(g)

    def biot_beta(self, g):
        if hasattr(self, "beta"):
            return self.beta
        else:
            return super().biot_beta(g)

    def scalar_temperature_coupling_coefficient(self, g):
        if hasattr(self, "gamma"):
            return self.gamma
        else:
            return super().scalar_temperature_coupling_coefficient(g)

    def set_temperature_parameters(self):
        super().set_temperature_parameters()
        if hasattr(self, "advection_weight"):
            w = self.advection_weight
            for g, d in self.gb:
                d[pp.PARAMETERS][self.temperature_parameter_key]["advection_weight"] = w


if __name__ == "__main__":
    unittest.main()
