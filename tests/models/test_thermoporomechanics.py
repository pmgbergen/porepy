"""Tests for thermoporomechanics.

The positive fracture gap value ensures positive aperture for all simulation. This is
needed to avoid degenerate mass and energy balance equations in the fracture.

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
        "times_to_export": [],  # Suppress output for tests
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
    setup = create_fractured_setup(solid_vals, {}, {"u_north": [0.0, uy_north]})
    pp.run_time_dependent_model(setup)

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
        assert np.allclose(u_vals[1, top], setup.solid.fracture_gap())
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
        assert np.allclose(jump[1], setup.solid.fracture_gap())
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

    setup = create_fractured_setup({}, {}, {"u_north": [0.0, 0.001]})
    pp.run_time_dependent_model(setup)
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

    setup = create_fractured_setup({}, {}, {"u_south": [0.0, -0.001]})
    pp.run_time_dependent_model(setup)
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

    setup = create_fractured_setup({}, {}, {"u_north": [0.0, -0.001]})
    pp.run_time_dependent_model(setup)
    u_vals, p_vals, p_frac, jump, traction, t_vals, t_frac = get_variables(setup)

    # All components should be closed in the normal direction
    assert np.allclose(jump[1], setup.solid.fracture_gap())

    # Contact force in normal direction should be negative
    assert np.all(traction[1] < 0)

    # Compression of the domain yields a (slightly) positive fracture pressure
    assert np.all(p_frac > 1e-10)
    assert np.all(t_frac > 1e-10)


def test_positive_p_frac_positive_opening():
    setup = create_fractured_setup({}, {}, {"fracture_source_value": 0.001})
    pp.run_time_dependent_model(setup)
    _, _, p_frac, jump, traction, _, t_frac = get_variables(setup)

    # All components should be open in the normal direction.
    assert np.all(jump[1] > setup.solid.fracture_gap())

    # By symmetry (reasonable to expect from this grid), the jump in tangential
    # deformation should be zero.
    assert np.abs(np.sum(jump[0])) < 1e-5

    # The contact force in normal direction should be zero.

    # NB: This assumes the contact force is expressed in local coordinates.
    assert np.all(np.abs(traction) < 1e-7)

    # Fracture pressure and temperature are both positive.
    assert np.allclose(p_frac, 4.8e-4, atol=1e-5)
    assert np.allclose(t_frac, 8.3e-6, atol=1e-7)


def test_robin_boundary_flux():
    """Tests model setup with Robin boundary conditions.

    Neumann and Robin boundary values are set by the same method
    (bc_values_"flux_name"). This test ensures that the correct values are assigned to
    the correct domain boundary sides (Robin values to Robin boundaries, Neumann values
    to Neumann boundaries). The test also covers checking if the Dirichlet values are
    assigned to the correct boundary sides. Thus, this test also checks that there is no
    overlap between the three boundary condition types.

    Model setup:
    * West boundary: Robin
    * East boundary: Neumann
    * North and south boundaries: Dirichlet

    Flux types that are tested: Mechanical stress, Fourier flux and Darcy flux.

    """

    class TailoredPoromechanicsRobin(
        pp.test_utils.models.RobinDirichletNeumannConditions,
        pp.models.thermoporomechanics.Thermoporomechanics,
    ):
        def set_domain(self) -> None:
            self._domain = pp.domains.unit_cube_domain(dimension=2)

    model_params = {
        "meshing_arguments": {"cell_size": 0.5},
        "grid_type": "cartesian",
        "pressure_north": 1e-3,
        "pressure_south": -1e-3,
        "darcy_flux_west": 1e-2,
        "darcy_flux_east": -1e-2,
        "fourier_flux_west": 2e-2,
        "fourier_flux_east": -2e-2,
        "mechanical_stress_west": 3e-2,
        "mechanical_stress_east": -3e-2,
    }

    model = TailoredPoromechanicsRobin(model_params)
    pp.run_time_dependent_model(model)

    subdomain = model.mdg.subdomains(dim=model.nd, return_data=True)[0][0]

    bc_operators = {
        "darcy_flux": model.combine_boundary_operators_darcy_flux,
        "fourier_flux": model.combine_boundary_operators_fourier_flux,
        "mechanical_stress": model.combine_boundary_operators_mechanical_stress,
    }

    # Create dictionary of evaluated boundary operators in bc_operators
    values = {
        key: operator([subdomain]).value(model.equation_system)
        for key, operator in bc_operators.items()
    }

    # Reshape mechanical stress values
    values["mechanical_stress"] = values["mechanical_stress"].reshape(
        (model.nd, subdomain.num_faces), order="F"
    )

    # Get boundary sides and assert boundary condition values
    bounds = model.domain_boundary_sides(subdomain)

    assert np.allclose(
        values["darcy_flux"][bounds.west], model_params["darcy_flux_west"]
    )
    assert np.allclose(
        values["darcy_flux"][bounds.east], model_params["darcy_flux_east"]
    )

    assert np.allclose(
        values["fourier_flux"][bounds.west], model_params["fourier_flux_west"]
    )
    assert np.allclose(
        values["fourier_flux"][bounds.east], model_params["fourier_flux_east"]
    )

    assert np.allclose(
        values["mechanical_stress"][0][bounds.west],
        model_params["mechanical_stress_west"],
    )
    assert np.allclose(
        values["mechanical_stress"][0][bounds.east],
        model_params["mechanical_stress_east"],
    )

    # Final check to see that the Dirichlet values are also assigned as expected. The
    # different bc types should not overlap. If all these tests pass, that suggest there
    # is no such overlap.

    # First constructing the pressure operator and evaluating it, then finding the
    # indices of the north and south boundaries in the pressure_values array, before
    # finally asserting that the values are correct.
    bg = model.mdg.subdomain_to_boundary_grid(subdomain)
    pressure_operator = model.pressure([bg])
    pressure_values = pressure_operator.value(model.equation_system)

    ind_north = np.nonzero(np.isin(bounds.all_bf, np.where(bounds.north)[0]))[0]
    ind_south = np.nonzero(np.isin(bounds.all_bf, np.where(bounds.south)[0]))[0]

    assert np.allclose(pressure_values[ind_north], model_params["pressure_north"])
    assert np.allclose(pressure_values[ind_south], model_params["pressure_south"])


@pytest.mark.parametrize(
    "units",
    [
        {"m": 0.29, "kg": 0.31, "K": 4.1},
    ],
)
def test_unit_conversion(units):
    """Test that solution is independent of units.

    Parameters:
        units: Dictionary with keys as those in
            :class:`~pp.models.material_constants.MaterialConstants`.

    """

    solid = pp.SolidConstants(pp.solid_values.extended_granite_values_for_testing)
    fluid = pp.FluidConstants(pp.fluid_values.extended_water_values_for_testing)

    model_params = {
        "times_to_export": [],  # Suppress output for tests
        "fracture_indices": [0],
        "cartesian": True,
        "u_north": [0.0, -1e-5],
        "material_constants": {"solid": solid, "fluid": fluid},
    }
    model_params_ref = copy.deepcopy(model_params)

    # Create model and run simulation
    setup_0 = TailoredThermoporomechanics(model_params_ref)
    pp.run_time_dependent_model(setup_0)

    model_params["units"] = pp.Units(**units)
    setup_1 = TailoredThermoporomechanics(model_params)

    pp.run_time_dependent_model(setup_1)
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
    model_params = {
        "fracture_indices": [2],
        "well_flux": -1e-2,
    }
    setup = ThermoporomechanicsWell(model_params)
    pp.run_time_dependent_model(setup)
