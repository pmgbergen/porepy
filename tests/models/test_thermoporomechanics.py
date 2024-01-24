"""Tests for thermoporomechanics.

The hardcoded fracture gap value of 0.042 ensures positive aperture for all simulation.
This is needed to avoid degenerate mass and energy balance equations in the fracture.

TODO: Clean up.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

import porepy as pp
from porepy.applications.md_grids import model_geometries
from porepy.applications.test_utils import well_models
from porepy.applications.test_utils.models import (
    Thermoporomechanics,
    compare_scaled_model_quantities,
    compare_scaled_primary_variables,
)

from .test_poromechanics import NonzeroFractureGapPoromechanics
from .test_poromechanics import get_variables as get_variables_poromechanics


class TailoredThermoporomechanics(
    NonzeroFractureGapPoromechanics,
    pp.model_boundary_conditions.TimeDependentMechanicalBCsDirNorthSouth,
    pp.model_boundary_conditions.BoundaryConditionsEnergyDirNorthSouth,
    pp.model_boundary_conditions.BoundaryConditionsMassDirNorthSouth,
    Thermoporomechanics,
):
    pass


def create_fractured_setup(
    solid_vals: dict, fluid_vals: dict, params: dict
) -> TailoredThermoporomechanics:
    """Create a setup for a 2d problem with a single fracture.

    Parameters:
        solid_vals: Dictionary with keys as those in :class:`pp.SolidConstants`
            and corresponding values.
        fluid_vals: Dictionary with keys as those in :class:`pp.FluidConstants`
            and corresponding values.
        params: Dictionary with keys as those in params of
            :class:`TailoredThermoporomechanics`.

    Returns:
        setup: Model object for the problem.

    """
    # Instantiate constants and store in params.
    solid_vals["fracture_gap"] = 0.042
    solid_vals["residual_aperture"] = 1e-10
    solid_vals["biot_coefficient"] = 1.0
    solid_vals["thermal_expansion"] = 1e-1
    fluid_vals["compressibility"] = 1
    fluid_vals["thermal_expansion"] = 1e-1
    solid = pp.SolidConstants(solid_vals)
    fluid = pp.FluidConstants(fluid_vals)

    default = {
        "suppress_export": True,  # Suppress output for tests
        "material_constants": {"solid": solid, "fluid": fluid},
        "max_iterations": 20,
    }
    default.update(params)
    setup = TailoredThermoporomechanics(default)
    return setup


def get_variables(setup):
    u_vals, p_vals, p_frac, jump, traction = get_variables_poromechanics(setup)
    t_var = setup.equation_system.get_variables(
        [setup.temperature_variable], setup.mdg.subdomains()
    )
    t_vals = setup.equation_system.get_variable_values(
        variables=t_var, time_step_index=0
    )
    t_var = setup.equation_system.get_variables(
        [setup.temperature_variable], setup.mdg.subdomains(dim=setup.nd - 1)
    )
    t_frac = setup.equation_system.get_variable_values(
        variables=t_var, time_step_index=0
    )
    return u_vals, p_vals, p_frac, jump, traction, t_vals, t_frac


@pytest.mark.parametrize(
    "solid_vals,uy_north",
    [
        ({}, 0.0),
        ({}, -0.1),
        ({"porosity": 0.5}, 0.1),
    ],
)
def test_2d_single_fracture(solid_vals: dict, uy_north: float):
    """Test that the solution is qualitatively sound.

    Parameters:
        solid_vals: Dictionary with keys as those in :class:`pp.SolidConstants`
            and corresponding values.
        uy_north: Value of displacement on the north boundary.

    """

    # Create model and run simulation
    setup = create_fractured_setup(solid_vals, {}, {"uy_north": uy_north})
    pp.run_time_dependent_model(setup, {})

    # Check that the pressure is linear
    sd = setup.mdg.subdomains(dim=setup.nd)[0]
    u_vals, p_vals, p_frac, jump, traction, t_vals, t_frac = get_variables(setup)

    top = sd.cell_centers[1] > 0.5
    bottom = sd.cell_centers[1] < 0.5
    tol = 1e-10
    if np.isclose(uy_north, 0.0):
        assert np.allclose(u_vals[:, bottom], 0)
        # Zero x and nonzero y displacement in top
        assert np.allclose(u_vals[0, top], 0)
        assert np.allclose(u_vals[1, top], 0.042)
        # Zero displacement relative to initial value implies zero pressure and
        # temperature
        assert np.allclose(p_vals, 0)
        assert np.allclose(t_vals, 0)
    elif uy_north < 0:
        # Boundary displacement is negative, so the y displacement should be
        # negative
        assert np.all(u_vals[1] < 0)

        # Check that x displacement is negative for left half and positive for right
        # half. Tolerance excludes cells at the centerline, where the displacement is
        # zero.
        left = sd.cell_centers[0] < setup.domain.bounding_box["xmax"] / 2 - tol
        assert np.all(u_vals[0, left] < 0)
        right = sd.cell_centers[0] > setup.domain.bounding_box["xmax"] / 2 + tol
        assert np.all(u_vals[0, right] > 0)
        # Compression implies pressure and temperature increase
        assert np.all(p_vals > 0 - tol)
        assert np.all(t_vals > 0 - tol)
    else:
        # Check that y displacement is positive in top half of domain
        assert np.all(u_vals[1, top] > 0)
        # Fracture cuts the domain in half, so the bottom half is only affected by
        # the pressure, which is negative, and thus the displacement should be
        # expansive, i.e. positive
        assert np.all(u_vals[1, bottom] > 0)
        # Expansion implies pressure reduction
        assert np.all(p_vals < 0 + tol)
        assert np.all(t_vals < 0 + tol)

    # Check that the displacement jump and traction are as expected.
    if uy_north > 0:
        # Normal component of displacement jump should be positive.
        assert np.all(jump[1] > -tol)
        # Traction should be zero
        assert np.allclose(traction, 0)
    else:
        # Displacement jump should be equal to initial displacement.
        assert np.allclose(jump[0], 0.0)
        assert np.allclose(jump[1], 0.042)
        # Normal traction should be non-positive. Zero if uy_north equals
        # initial gap, negative otherwise.
        if uy_north < 0:
            assert np.all(traction[setup.nd - 1 :: setup.nd] <= tol)
        else:
            assert np.allclose(traction, 0)


def test_thermoporomechanics_model_no_modification():
    """Test that the raw contact thermoporomechanics model with no modifications can be
    run with no error messages.

    Failure of this test would signify rather fundamental problems in the model.

    """
    mod = pp.thermoporomechanics.Thermoporomechanics({})
    pp.run_stationary_model(mod, {})


def test_pull_north_positive_opening():
    setup = create_fractured_setup({}, {}, {"uy_north": 0.001})
    pp.run_time_dependent_model(setup, {})
    u_vals, p_vals, p_frac, jump, traction, t_vals, t_frac = get_variables(setup)

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
    assert np.all(t_frac < -1e-7)


def test_pull_south_positive_opening():
    setup = create_fractured_setup({}, {}, {"uy_south": -0.001})
    pp.run_time_dependent_model(setup, {})
    u_vals, p_vals, p_frac, jump, traction, t_vals, t_frac = get_variables(setup)

    # All components should be open in the normal direction
    assert np.all(jump[1] > 0)

    # By symmetry (reasonable to expect from this grid), the jump in tangential
    # deformation should be zero.
    assert np.abs(np.sum(jump[0])) < 1e-5

    # The contact force in normal direction should be zero

    # NB: This assumes the contact force is expressed in local coordinates
    assert np.all(np.abs(traction) < 1e-7)

    # Check that the dilation yields a negative pressure and temperature
    assert np.all(p_vals < -1e-7)
    assert np.all(t_vals < -1e-7)


def test_push_north_zero_opening():
    setup = create_fractured_setup({}, {}, {"uy_north": -0.001})
    pp.run_time_dependent_model(setup, {})
    u_vals, p_vals, p_frac, jump, traction, t_vals, t_frac = get_variables(setup)

    # All components should be closed in the normal direction
    assert np.allclose(jump[1], 0.042)

    # Contact force in normal direction should be negative
    assert np.all(traction[1] < 0)

    # Compression of the domain yields a (slightly) positive fracture pressure
    assert np.all(p_frac > 1e-10)
    assert np.all(t_frac > 1e-10)


def test_positive_p_frac_positive_opening():
    setup = create_fractured_setup({}, {}, {"fracture_source_value": 0.001})
    pp.run_time_dependent_model(setup, {})
    _, _, p_frac, jump, traction, _, t_frac = get_variables(setup)

    # All components should be open in the normal direction.
    assert np.all(jump[1] > 0.042)

    # By symmetry (reasonable to expect from this grid), the jump in tangential
    # deformation should be zero.
    assert np.abs(np.sum(jump[0])) < 1e-5

    # The contact force in normal direction should be zero.

    # NB: This assumes the contact force is expressed in local coordinates.
    assert np.all(np.abs(traction) < 1e-7)

    # Fracture pressure and temperature are both positive.
    assert np.allclose(p_frac, 4.8e-4, atol=1e-5)
    assert np.allclose(t_frac, 8.3e-6, atol=1e-7)


@pytest.mark.parametrize(
    "units",
    [
        {"m": 0.29, "kg": 0.31, "K": 4.1},
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
        "fracture_indices": [0],
        "cartesian": True,
        "uy_north": -0.1,
    }
    reference_params = copy.deepcopy(params)

    # Create model and run simulation
    setup_0 = TailoredThermoporomechanics(reference_params)
    pp.run_time_dependent_model(setup_0, reference_params)

    params["units"] = pp.Units(**units)
    setup_1 = TailoredThermoporomechanics(params)

    pp.run_time_dependent_model(setup_1, params)
    variables = [
        setup_1.pressure_variable,
        setup_1.interface_darcy_flux_variable,
        setup_1.displacement_variable,
        setup_1.interface_displacement_variable,
        setup_1.temperature_variable,
        setup_1.interface_fourier_flux_variable,
        setup_1.interface_enthalpy_flux_variable,
    ]
    variable_units = [
        "Pa",
        "Pa * m^2 * s^-1",
        "m",
        "m",
        "K",
        "m^-1 * s^-1 * J",
        "m^-1 * s^-1 * J",
    ]
    compare_scaled_primary_variables(setup_0, setup_1, variables, variable_units)
    secondary_variables = ["darcy_flux", "fluid_flux", "stress", "porosity"]
    secondary_units = ["Pa * m^2 * s^-1", "kg * m^-1 * s^-1", "Pa * m", "-"]
    domain_dimensions = [None, None, 2, None]
    compare_scaled_model_quantities(
        setup_0, setup_1, secondary_variables, secondary_units, domain_dimensions
    )


class ThermoporomechanicsWell(
    well_models.OneVerticalWell,
    model_geometries.OrthogonalFractures3d,
    well_models.BoundaryConditionsWellSetup,
    pp.poromechanics.Poromechanics,
):
    def meshing_arguments(self) -> dict:
        # Length scale:
        ls = self.solid.convert_units(1, "m")
        h = 0.5 * ls
        mesh_sizes = {
            "cell_size": h,
        }
        return mesh_sizes


def test_thermoporomechanics_well():
    """Test that the thermoporomechanics model runs without errors."""
    # These parameters hopefully yield a relatively easy problem
    params = {
        "fracture_indices": [2],
        "well_flux": -1e-2,
    }
    setup = ThermoporomechanicsWell(params)
    pp.run_time_dependent_model(setup, {})
