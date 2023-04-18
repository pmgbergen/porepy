"""Tests for poromechanics.

The hardcoded fracture gap value of 0.042 ensures positive aperture for all simulation.
This is needed to avoid degenerate mass balance equation in fracture.

TODO: Clean up.
"""
from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import pytest

import porepy as pp


from .setup_utils import (
    BoundaryConditionsMassAndEnergyDirNorthSouth,
    Poromechanics,
    TimeDependentMechanicalBCsDirNorthSouth,
    compare_scaled_model_quantities,
    compare_scaled_primary_variables,
)


class NonzeroFractureGapPoromechanics:
    """Adjust bc values and initial condition."""

    domain_boundary_sides: Callable
    """Boundary sides of the domain. Normally defined in a mixin instance of
    :class:`~porepy.models.geometry.ModelGeometry`.

    """
    nd: int
    """Number of dimensions of the problem."""
    params: dict
    """Parameters for the model."""

    def bc_type_darcy(self, sd: pp.Grid) -> pp.BoundaryCondition:
        domain_sides = self.domain_boundary_sides(sd)
        return pp.BoundaryCondition(sd, domain_sides.north + domain_sides.south, "dir")

    def initial_condition(self):
        """Set initial condition.

        Set initial displacement compatible with fracture gap of 0.042 for matrix
        subdomain and matrix-fracture interface. Also initialize boundary conditions
        to 0.042 on top side.

        """
        super().initial_condition()
        # Initial pressure equals reference pressure (defaults to zero).
        self.equation_system.set_variable_values(
            self.fluid.pressure() * np.ones(self.mdg.num_subdomain_cells()),
            [self.pressure_variable],
            time_step_index=0,
            iterate_index=0,
        )
        sd, sd_data = self.mdg.subdomains(return_data=True)[0]
        # Initial displacement.
        if len(self.mdg.subdomains()) > 1:
            top_cells = sd.cell_centers[1] > 0.5
            vals = np.zeros((self.nd, sd.num_cells))
            vals[1, top_cells] = self.fluid.convert_units(0.042, "m")
            self.equation_system.set_variable_values(
                vals.ravel("F"),
                [self.displacement_variable],
                time_step_index=0,
                iterate_index=0,
            )
            # Find mortar cells on the top boundary
            intf = self.mdg.interfaces()[0]
            # Identify by normal vector in sd_primary
            faces_primary = intf.primary_to_mortar_int().tocsr().indices
            switcher = pp.grid_utils.switch_sign_if_inwards_normal(
                sd,
                self.nd,
                faces_primary,
            )

            normals = (switcher * sd.face_normals[: sd.dim].ravel("F")).reshape(
                sd.dim, -1, order="F"
            )
            intf_normals = normals[:, faces_primary]
            top_cells = intf_normals[1, :] < 0

            # Set mortar displacement to zero on bottom and 0.042 on top
            vals = np.zeros((self.nd, intf.num_cells))
            vals[1, top_cells] = self.fluid.convert_units(0.042, "m")
            self.equation_system.set_variable_values(
                vals.ravel("F"),
                [self.interface_displacement_variable],
                time_step_index=0,
                iterate_index=0,
            )

    def fracture_stress(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Fracture stress on interfaces.

        The "else" mimicks old subtract_p_frac=False behavior, which is
        useful for testing reference pressure.

        Parameters:
            interfaces: List of interfaces where the stress is defined.

        Returns:
            Poromechanical stress operator on matrix-fracture interfaces.

        """
        if not all([intf.dim == self.nd - 1 for intf in interfaces]):
            raise ValueError("Interfaces must be of dimension nd - 1.")

        if getattr(self, "subtract_p_frac", True):
            # Contact traction and fracture pressure.
            return super().fracture_stress(interfaces)
        else:
            # Only contact traction.
            return pp.constitutive_laws.LinearElasticMechanicalStress.fracture_stress(
                self, interfaces
            )

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fluid source term.

        Add a source term in fractures.

        Parameters:
            subdomains: List of subdomains where the source term is defined.

        Returns:
            Fluid source term operator.

        """
        internal_boundaries = super().fluid_source(subdomains)
        if "fracture_source_value" not in self.params:
            return internal_boundaries

        vals = []
        for sd in subdomains:
            if sd.dim == self.nd:
                vals.append(np.zeros(sd.num_cells))
            else:
                val = self.fluid.convert_units(
                    self.params["fracture_source_value"], "kg * s ^ -1"
                )
                vals.append(val * np.ones(sd.num_cells))
        fracture_source = pp.wrap_as_ad_array(
            np.hstack(vals), name="fracture_fluid_source"
        )
        return internal_boundaries + fracture_source


class TailoredPoromechanics(
    NonzeroFractureGapPoromechanics,
    TimeDependentMechanicalBCsDirNorthSouth,
    BoundaryConditionsMassAndEnergyDirNorthSouth,
    Poromechanics,
):
    pass


def create_fractured_setup(solid_vals, fluid_vals, uy_north):
    """Create a setup for a fractured domain.

    The domain is a unit square with two intersecting fractures.

    Parameters:
        solid_vals (dict): Parameters for the solid mechanics model.
        fluid_vals (dict): Parameters for the fluid mechanics model.
        uy_north (float): Displacement in y-direction on the north boundary.

    Returns:
        TailoredPoromechanics: A setup for a fractured domain.

    """
    # Instantiate constants and store in params.
    solid_vals["fracture_gap"] = 0.042
    solid_vals["residual_aperture"] = 1e-10
    solid_vals["biot_coefficient"] = 1.0
    fluid_vals["compressibility"] = 1
    solid = pp.SolidConstants(solid_vals)
    fluid = pp.FluidConstants(fluid_vals)

    params = {
        "suppress_export": True,  # Suppress output for tests
        "material_constants": {"solid": solid, "fluid": fluid},
        "uy_north": uy_north,
        "max_iterations": 20,
    }
    setup = TailoredPoromechanics(params)
    return setup


def get_variables(
    setup: TailoredPoromechanics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Utility function to extract variables from a setup.

    Parameters:
        setup (TailoredPoromechanics): A setup for a fractured domain.

    Returns:
        Tuple containing the following variables:
            np.ndarray: Displacement values.
            np.ndarray: Pressure values.
            np.ndarray: Fracture pressure values.
            np.ndarray: Displacement jump values.
            np.ndarray: Contact traction values.

    """
    sd = setup.mdg.subdomains(dim=setup.nd)[0]
    u_var = setup.equation_system.get_variables([setup.displacement_variable], [sd])
    u_vals = setup.equation_system.get_variable_values(
        variables=u_var, time_step_index=0
    ).reshape(setup.nd, -1, order="F")

    p_var = setup.equation_system.get_variables(
        [setup.pressure_variable], setup.mdg.subdomains()
    )
    p_vals = setup.equation_system.get_variable_values(
        variables=p_var, time_step_index=0
    )
    p_var = setup.equation_system.get_variables(
        [setup.pressure_variable], setup.mdg.subdomains(dim=setup.nd - 1)
    )
    p_frac = setup.equation_system.get_variable_values(
        variables=p_var, time_step_index=0
    )
    # Fracture
    sd_frac = setup.mdg.subdomains(dim=setup.nd - 1)
    jump = (
        setup.displacement_jump(sd_frac)
        .evaluate(setup.equation_system)
        .val.reshape(setup.nd, -1, order="F")
    )
    traction = (
        setup.contact_traction(sd_frac)
        .evaluate(setup.equation_system)
        .val.reshape(setup.nd, -1, order="F")
    )
    return u_vals, p_vals, p_frac, jump, traction


@pytest.mark.parametrize(
    "solid_vals,north_displacement",
    [
        ({}, 0.0),
        ({}, -0.1),
        ({"porosity": 0.5}, 0.1),
    ],
)
def test_2d_single_fracture(solid_vals, north_displacement):
    """Test that the solution is qualitatively sound.

    Parameters:
        solid_vals (dict): Dictionary with keys as those in :class:`pp.SolidConstants`
            and corresponding values.
        north_displacement (float): Value of displacement on the north boundary.
        expected_x_y (tuple): Expected values of the displacement in the x and y.
            directions. The values are used to infer sign of displacement solution.

    """

    setup = create_fractured_setup(solid_vals, {}, north_displacement)
    pp.run_time_dependent_model(setup, {})
    u_vals, p_vals, p_frac, jump, traction = get_variables(setup)

    # Create model and run simulation
    sd_nd = setup.mdg.subdomains(dim=setup.nd)[0]
    top = sd_nd.cell_centers[1] > 0.5
    bottom = sd_nd.cell_centers[1] < 0.5
    tol = 1e-10
    if np.isclose(north_displacement, 0.0):
        assert np.allclose(u_vals[:, bottom], 0)
        # Zero x and nonzero y displacement in top
        assert np.allclose(u_vals[0, top], 0)
        assert np.allclose(u_vals[1, top], 0.042)
        # Zero displacement relative to initial value implies zero pressure
        assert np.allclose(p_vals, 0)
    elif north_displacement < 0.0:
        # Boundary displacement is negative, so the y displacement should be
        # negative
        assert np.all(u_vals[1] < 0)

        # Check that x displacement is negative for left half and positive for right
        # half. Tolerance excludes cells at the centerline, where the displacement is
        # zero.
        left = sd_nd.cell_centers[0] < setup.domain.bounding_box["xmax"] / 2 - tol
        assert np.all(u_vals[0, left] < 0)
        right = sd_nd.cell_centers[0] > setup.domain.bounding_box["xmax"] / 2 + tol
        assert np.all(u_vals[0, right] > 0)
        # Compression implies pressure increase
        assert np.all(p_vals > 0 - tol)
    else:
        # Check that y displacement is positive in top half of domain
        assert np.all(u_vals[1, top] > 0)
        # Fracture cuts the domain in half, so the bottom half is only affected by
        # the pressure, which is negative, and thus the displacement should be
        # expansive, i.e. positive
        assert np.all(u_vals[1, bottom] > 0)
        # Expansion implies pressure reduction
        assert np.all(p_vals < 0 + tol)

    # Fracture
    if north_displacement > 0.0:
        # Normal component of displacement jump should be positive.
        assert np.all(jump[1] > -tol)
        # Traction should be zero
        assert np.allclose(traction, 0)
    else:
        # Displacement jump should be equal to initial displacement.
        assert np.allclose(jump[0], 0.0)
        assert np.allclose(jump[1], 0.042)
        # Normal traction should be non-positive. Zero if north_displacement equals
        # initial gap, negative otherwise.
        if north_displacement < 0.0:
            assert np.all(traction[setup.nd - 1 :: setup.nd] <= tol)
        else:
            assert np.allclose(traction, 0)


def test_poromechanics_model_no_modification():
    """Test that the poromechanics model with no modifications runs with no errors.

    Failure of this test would signify rather fundamental problems in the model.
    """
    mod = pp.poromechanics.Poromechanics({})
    pp.run_stationary_model(mod, {})


@pytest.mark.parametrize("biot_coefficient", [0, 0.5])
def test_without_fracture(biot_coefficient):
    fluid = pp.FluidConstants(constants={"compressibility": 0.5})
    solid = pp.SolidConstants(constants={"biot_coefficient": biot_coefficient})
    params = {
        "fracture_indices": [],
        "material_constants": {"fluid": fluid, "solid": solid},
        "uy_north": 0.001,
    }
    m = TailoredPoromechanics(params)
    pp.run_time_dependent_model(m, {})

    sd = m.mdg.subdomains(dim=m.nd)
    u = (
        m.displacement(sd)
        .evaluate(m.equation_system)
        .val.reshape((m.nd, -1), order="F")
    )
    p = m.pressure(sd).evaluate(m.equation_system).val

    # By symmetry (reasonable to expect from this grid), the average x displacement
    # should be zero
    tol = 1e-10
    assert np.abs(np.sum(u[0])) < tol
    # Check that y component lies between zero and applied boundary displacement
    assert np.all(u[1] > 0)
    assert np.all(u[1] < 0.001)
    # Check that the expansion yields a negative pressure
    if biot_coefficient == 0:
        assert np.allclose(p, 0)
    else:
        assert np.all(p < -tol)
        # Stronger test, could be relaxed.
        assert np.allclose(p, -4.16490713e-05)


def test_pull_north_positive_opening():
    """Check solution for a pull on the north side with one horizontal fracture."""
    setup = create_fractured_setup({}, {}, 0.001)
    pp.run_time_dependent_model(setup, {})
    _, _s, p_frac, jump, traction = get_variables(setup)

    # All components should be open in the normal direction
    assert np.all(jump[1] > 0)

    # By symmetry (reasonable to expect from this grid), the jump in tangential
    # deformation should be zero.
    assert np.abs(np.sum(jump[0])) < 1e-5

    # The contact force in normal direction should be zero

    # NB: This assumes the contact force is expressed in local coordinates
    assert np.all(np.abs(traction) < 1e-7)

    # Check that the dilation of the fracture yields a negative fracture pressure
    assert np.all(p_frac < -1e-7)


def test_pull_south_positive_opening():
    """Check solution for a pull on the south side with one horizontal fracture."""

    setup = create_fractured_setup({}, {}, 0)
    setup.params["uy_south"] = -0.001
    pp.run_time_dependent_model(setup, {})
    u_vals, p_vals, p_frac, jump, traction = get_variables(setup)

    # All components should be open in the normal direction
    assert np.all(jump[1] > 0)

    # By symmetry (reasonable to expect from this grid), the jump in tangential
    # deformation should be zero.
    assert np.abs(np.sum(jump[0])) < 1e-5

    # The contact force in normal direction should be zero

    # NB: This assumes the contact force is expressed in local coordinates
    assert np.all(np.abs(traction) < 1e-7)

    # Check that the dilation of the fracture yields a negative fracture pressure
    assert np.all(p_frac < -1e-7)


def test_push_north_zero_opening():
    setup = create_fractured_setup({}, {}, -0.001)
    pp.run_time_dependent_model(setup, {})
    u_vals, p_vals, p_frac, jump, traction = get_variables(setup)

    # All components should be closed in the normal direction
    assert np.allclose(jump[1], 0.042)

    # Contact force in normal direction should be negative
    assert np.all(traction[1] < 0)

    # Compression of the domain yields a (slightly) positive fracture pressure
    assert np.all(p_frac > 1e-10)


def test_positive_p_frac_positive_opening():
    setup = create_fractured_setup({}, {}, 0)
    setup.params["fracture_source_value"] = 0.001
    pp.run_time_dependent_model(setup, {})
    _, _, p_frac, jump, traction = get_variables(setup)

    # All components should be open in the normal direction
    assert np.all(jump[1] > 0.042)

    # By symmetry (reasonable to expect from this grid), the jump in tangential
    # deformation should be zero.
    assert np.abs(np.sum(jump[0])) < 1e-5

    # The contact force in normal direction should be zero

    # NB: This assumes the contact force is expressed in local coordinates
    assert np.all(np.abs(traction) < 1e-7)

    # Fracture pressure is positive
    assert np.all(p_frac > 4.7e-4)
    assert np.all(p_frac < 4.9e-4)


def test_pull_south_positive_reference_pressure():
    """Compare with and without nonzero reference (and initial) solution."""
    setup_ref = create_fractured_setup({}, {}, 0)
    setup_ref.subtract_p_frac = False
    setup_ref.params["uy_south"] = -0.001
    pp.run_time_dependent_model(setup_ref, {})
    u_vals_ref, p_vals_ref, p_frac_ref, jump_ref, traction_ref = get_variables(
        setup_ref
    )

    setup = create_fractured_setup({}, {"pressure": 1}, 0)
    setup.subtract_p_frac = False
    setup.params["uy_south"] = -0.001
    pp.run_time_dependent_model(setup, {})
    u_vals, p_vals, p_frac, jump, traction = get_variables(setup)

    assert np.allclose(jump, jump_ref)
    assert np.allclose(u_vals, u_vals_ref)
    assert np.allclose(traction, traction_ref)
    assert np.allclose(p_frac, p_frac_ref + 1)
    assert np.allclose(p_vals, p_vals_ref + 1)


@pytest.mark.parametrize(
    "units",
    [
        {"m": 0.2, "kg": 0.3, "K": 42},
    ],
)
def test_unit_conversion(units):
    """Test that solution is independent of units.

    Parameters:
        units (dict): Dictionary with keys as those in
            :class:`~pp.models.material_constants.MaterialConstants`.

    """

    params = {
        "suppress_export": True,  # Suppress output for tests
        "num_fracs": 1,
        "cartesian": True,
        "uy_north": 0.1,
    }
    reference_params = copy.deepcopy(params)
    reference_params["file_name"] = "unit_conversion_reference"

    # Create model and run simulation
    setup_0 = TailoredPoromechanics(reference_params)
    pp.run_time_dependent_model(setup_0, reference_params)

    params["units"] = pp.Units(**units)
    setup_1 = TailoredPoromechanics(params)

    pp.run_time_dependent_model(setup_1, params)
    variables = [
        setup_1.pressure_variable,
        setup_1.interface_darcy_flux_variable,
        setup_1.displacement_variable,
        setup_1.interface_displacement_variable,
    ]
    variable_units = ["Pa", "Pa * m^2 * s^-1", "m", "m"]
    compare_scaled_primary_variables(setup_0, setup_1, variables, variable_units)
    flux_names = ["darcy_flux", "fluid_flux", "stress", "fracture_stress"]
    flux_units = ["Pa * m^2 * s^-1", "kg * m^-1 * s^-1", "Pa * m", "Pa"]
    domain_dimensions = [None, None, 2, 1]
    compare_scaled_model_quantities(
        setup_0, setup_1, flux_names, flux_units, domain_dimensions
    )
