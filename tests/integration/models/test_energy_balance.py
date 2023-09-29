"""Tests for energy balance.

"""
from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import pytest

import porepy as pp

from .setup_utils import (
    MassAndEnergyBalance,
    compare_scaled_model_quantities,
    compare_scaled_primary_variables,
)
from .test_mass_balance import BoundaryConditionLinearPressure


class BoundaryConditionsEnergy(pp.energy_balance.BoundaryConditionsEnergyBalance):
    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        sides = self.domain_boundary_sides(boundary_grid)
        vals = np.zeros(boundary_grid.num_cells)
        vals[sides.west] = self.fluid.convert_units(1, "K")
        return vals

    def bc_type_fourier(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Dirichlet conditions on all external boundaries.

        Parameters:
            sd: Subdomain grid on which to define boundary conditions.

        Returns:
            Boundary condition object.

        """
        # Define boundary regions
        sides = self.domain_boundary_sides(sd)
        # Define Dirichlet conditions on the left and right boundaries
        return pp.BoundaryCondition(sd, sides.west + sides.east, "dir")

    def bc_type_enthalpy(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # Define boundary regions
        sides = self.domain_boundary_sides(sd)
        # Define Dirichlet conditions on the left and right boundaries
        return pp.BoundaryCondition(sd, sides.west, "dir")


class EnergyBalanceTailoredBCs(
    BoundaryConditionsEnergy,
    BoundaryConditionLinearPressure,
    MassAndEnergyBalance,
):
    pass


@pytest.mark.parametrize(
    "fluid_vals,solid_vals",
    [
        (
            {
                "thermal_conductivity": 1e-10,
                "specific_heat_capacity": 1e3,
            },
            {
                "thermal_conductivity": 1e-10,
                "permeability": 1e-2,
                "specific_heat_capacity": 1e3,
            },
        ),
        (
            {"thermal_conductivity": 1e5},
            {"thermal_conductivity": 1e4, "permeability": 1e-7},
        ),
    ],
)
def test_advection_or_diffusion_dominated(fluid_vals, solid_vals):
    """Test that the pressure solution is linear.

    With constant density (zero compressibility) and the specified boundary conditions,
    the pressure solution should be linear, i.e., ::math::
        p = 1 - x
        v_x = 1 * k / x_max

    We test one advection dominated and one diffusion dominated case. For the former,
    the influx is approx. w_in = v_x * specific_heat_capacity, yielding an added energy
    close to time * w_in. w_in also gives an upper bound for internal advective fluxes.
    For the latter, we check that a near linear temperature profile is obtained.

    Parameters:
        fluid_vals (dict): Dictionary with keys as those in :class:`pp.FluidConstants`
            and corresponding values.
        solid_vals (dict): Dictionary with keys as those in :class:`pp.SolidConstants`
            and corresponding values.

    """
    # Always use zero compressibility
    fluid_vals["compressibility"] = 0
    # Instantiate constants and store in params.
    fluid = pp.FluidConstants(fluid_vals)
    solid = pp.SolidConstants(solid_vals)
    params = {
        "suppress_export": True,  # Suppress output for tests
        "material_constants": {"fluid": fluid, "solid": solid},
    }

    # Create model and run simulation
    setup = EnergyBalanceTailoredBCs(params)
    pp.run_time_dependent_model(setup, params)

    if solid_vals["thermal_conductivity"] > 1:
        # Diffusion dominated case.
        # Check that the temperature is linear.
        for sd in setup.mdg.subdomains():
            var = setup.equation_system.get_variables(["temperature"], [sd])
            vals = setup.equation_system.get_variable_values(
                variables=var, time_step_index=0
            )
            assert np.allclose(
                vals,
                1 - sd.cell_centers[0] / setup.domain.bounding_box["xmax"],
                rtol=1e-4,
            )
    else:
        # Advection dominated case.
        # Check that the enthalpy flux over each face is bounded by the value
        # corresponding to a fully saturated domain.
        for sd in setup.mdg.subdomains():
            val = setup.enthalpy_flux([sd]).evaluate(setup.equation_system).val
            # Account for specific volume, default value of .01 in fractures.
            normals = np.abs(sd.face_normals[0]) * np.power(0.1, setup.nd - sd.dim)
            k = setup.solid.permeability() / setup.fluid.viscosity()
            grad = 1 / setup.domain.bounding_box["xmax"]
            enth = setup.fluid.specific_heat_capacity() * normals * grad * k
            assert np.all(np.abs(val) < np.abs(enth) + 1e-10)

        # Total advected matrix energy: (bc_val=1) * specific_heat * (time=1 s) * (total
        # influx =grad * dp * k=1/2*k)
        sds = setup.mdg.subdomains(dim=2)
        total_energy = (
            setup.volume_integral(
                setup.total_internal_energy(sds),
                sds,
                dim=1,
            )
            .evaluate(setup.equation_system)
            .val
        )
        expected = setup.fluid.specific_heat_capacity() * setup.solid.permeability() / 2
        assert np.allclose(np.sum(total_energy), expected, rtol=1e-3)


@pytest.mark.parametrize(
    "units",
    [
        {"m": 2, "kg": 3, "s": 1, "K": 7},
    ],
)
def test_unit_conversion(units):
    """Test that solution is independent of units.

    Parameters:
        units (dict): Dictionary with keys as those in
            :class:`~pp.models.material_constants.MaterialConstants`.

    """
    params = {"suppress_export": True, "fracture_indices": [0, 1], "cartesian": True}
    # Create model and run simulation
    reference_params = copy.deepcopy(params)
    reference_params["file_name"] = "unit_conversion_reference"
    reference_setup = EnergyBalanceTailoredBCs(reference_params)
    pp.run_time_dependent_model(reference_setup, reference_params)

    params["units"] = pp.Units(**units)
    setup = EnergyBalanceTailoredBCs(params)

    pp.run_time_dependent_model(setup, params)
    variable_names = [
        setup.temperature_variable,
        setup.pressure_variable,
        setup.interface_darcy_flux_variable,
        setup.interface_enthalpy_flux_variable,
    ]
    variable_units = ["K", "Pa", "m^2 * s^-1 * Pa", "m^-1 * s^-1 * J"]
    secondary_variables = ["darcy_flux", "enthalpy_flux", "fourier_flux"]
    secondary_units = ["m^2 * s^-1 * Pa", "m^-1 * s^-1 * J", "m^-1 * s^-1 * J"]
    # No domain restrictions.
    domain_dimensions = [None, None, None]
    compare_scaled_primary_variables(
        setup, reference_setup, variable_names, variable_units
    )
    compare_scaled_model_quantities(
        setup, reference_setup, secondary_variables, secondary_units, domain_dimensions
    )
