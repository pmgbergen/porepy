"""Tests for energy balance.

TODO: This test should be updated along the lines of test_single_phase_flow.py.

"""

from __future__ import annotations

import copy

import numpy as np
import pytest

import porepy as pp
from porepy.applications.test_utils import models, well_models


class BoundaryConditionLinearPressure(
    pp.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow
):
    """Overload the boundary condition to give a linear pressure profile.

    Homogeneous Neumann conditions on top and bottom, Dirichlet 1 and 0 on the
    left and right boundaries, respectively.

    """

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Dirichlet conditions on all external boundaries.

        Parameters:
            sd: Subdomain grid on which to define boundary conditions.

        Returns:
            Boundary condition object.

        """
        # Define boundary regions
        sides = self.domain_boundary_sides(sd)
        # Define Dirichlet conditions on the left and right boundaries
        return pp.BoundaryCondition(sd, sides.east + sides.west, "dir")

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Boundary values for the pressure.

        Parameters:
            boundary_grid: The boundary grid on which to define boundary conditions.

        Returns:
            Array of boundary values.

        """

        sides = self.domain_boundary_sides(boundary_grid)
        vals = np.zeros(boundary_grid.num_cells)
        vals[sides.west] = self.fluid.convert_units(1, "Pa")
        return vals


class BoundaryConditionsEnergy(pp.energy_balance.BoundaryConditionsEnergyBalance):
    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        sides = self.domain_boundary_sides(boundary_grid)
        vals = np.zeros(boundary_grid.num_cells)
        vals[sides.west] = self.fluid.convert_units(1, "K")
        return vals

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
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

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # Define boundary regions
        sides = self.domain_boundary_sides(sd)
        # Define Dirichlet conditions on the left and right boundaries
        return pp.BoundaryCondition(sd, sides.west, "dir")


class EnergyBalanceTailoredBCs(
    BoundaryConditionLinearPressure,
    BoundaryConditionsEnergy,
    models.MassAndEnergyBalance,
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
    model_params = {
        "times_to_export": [],  # Suppress output for tests
        "material_constants": {"fluid": fluid, "solid": solid},
    }

    # Create model and run simulation
    setup = EnergyBalanceTailoredBCs(model_params)
    pp.run_time_dependent_model(setup)

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
            val = setup.enthalpy_flux([sd]).value(setup.equation_system)
            # Account for specific volume, default value of .01 in fractures.
            normals = np.abs(sd.face_normals[0]) * np.power(0.1, setup.nd - sd.dim)
            k = setup.solid.permeability() / setup.fluid.viscosity()
            grad = 1 / setup.domain.bounding_box["xmax"]
            enth = setup.fluid.specific_heat_capacity() * normals * grad * k
            assert np.all(np.abs(val) < np.abs(enth) + 1e-10)

        # Total advected matrix energy: (bc_val=1) * specific_heat * (time=1 s) * (total
        # influx =grad * dp * k=1/2*k)
        sds = setup.mdg.subdomains(dim=2)
        total_energy = setup.volume_integral(
            setup.total_internal_energy(sds),
            sds,
            dim=1,
        ).value(setup.equation_system)
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

    class Model(EnergyBalanceTailoredBCs):
        def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
            """ """

            sides = self.domain_boundary_sides(boundary_grid)
            vals = np.zeros(boundary_grid.num_cells)
            vals[sides.west] = self.fluid.convert_units(10.0, "K")
            return vals

        def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
            """ """

            sides = self.domain_boundary_sides(boundary_grid)
            vals = np.zeros(boundary_grid.num_cells)
            vals[sides.west] = self.fluid.convert_units(1e4, "Pa")
            return vals

    solid_vals = pp.solid_values.extended_granite_values_for_testing
    fluid_vals = pp.fluid_values.extended_water_values_for_testing
    solid = pp.SolidConstants(solid_vals)
    fluid = pp.FluidConstants(fluid_vals)

    # Non-unitary time step needed for convergence
    dt = 1e5
    model_params = {
        "times_to_export": [],
        "fracture_indices": [0, 1],
        "cartesian": True,
        "material_constants": {"solid": solid, "fluid": fluid},
        "time_manager": pp.TimeManager(schedule=[0, dt], dt_init=dt, constant_dt=True),
    }

    # Create model and run simulation
    model_reference_params = copy.deepcopy(model_params)
    model_reference_params["file_name"] = "unit_conversion_reference"
    reference_setup = Model(model_reference_params)
    pp.run_time_dependent_model(reference_setup)

    model_params["units"] = pp.Units(**units)
    setup = Model(model_params)

    pp.run_time_dependent_model(setup)
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
    models.compare_scaled_primary_variables(
        setup, reference_setup, variable_names, variable_units
    )
    models.compare_scaled_model_quantities(
        setup, reference_setup, secondary_variables, secondary_units, domain_dimensions
    )


class MassAndEnergyWellModel(
    well_models.OneVerticalWell,
    models.OrthogonalFractures3d,
    well_models.BoundaryConditionsWellSetup,
    well_models.WellPermeability,
    pp.mass_and_energy_balance.MassAndEnergyBalance,
):
    pass


def test_energy_conservation():
    """One central vertical well, one horizontal fracture.

    The well is placed such that the pressure at the fracture is log distributed, up to
    the effect of the low-permeable matrix.

    """
    # We want low pressures, to ensure energy is not dominated by -p in fluid part.
    dt = 1e-4
    model_params = {
        # Set impermeable matrix
        "material_constants": {
            "solid": pp.SolidConstants(
                {
                    "specific_heat_capacity": 1e0,  # Ensure energy stays in domain.
                    "well_radius": 0.02,
                    "residual_aperture": 1.0,  # Ensure high permeability in fracture.
                    "thermal_conductivity": 1e-6,
                    "permeability": 1e4,  # Reduce pressure effect
                    "normal_permeability": 1e4,
                }
            ),
            "fluid": pp.FluidConstants(
                {
                    "specific_heat_capacity": 1e0,
                    "thermal_conductivity": 1e-6,
                    "normal_thermal_conductivity": 1e-6,
                }
            ),
        },
        # Use only the horizontal fracture of OrthogonalFractures3d
        "fracture_indices": [2],
        "time_manager": pp.TimeManager(schedule=[0, dt], dt_init=dt, constant_dt=True),
        "grid_type": "cartesian",
    }

    setup = MassAndEnergyWellModel(model_params)
    pp.run_time_dependent_model(setup)
    # Check that the total enthalpy equals the injected one.
    in_val = 1e7
    u_expected = in_val * dt

    sds = setup.mdg.subdomains()
    u = setup.volume_integral(setup.total_internal_energy(sds), sds, 1)
    h_f = setup.fluid_enthalpy(sds) * setup.porosity(sds)
    h_s = setup.solid_enthalpy(sds) * (pp.ad.Scalar(1) - setup.porosity(sds))
    h = setup.volume_integral(h_f + h_s, sds, 1)
    u_val = np.sum(u.value(setup.equation_system))
    h_val = np.sum(h.value(setup.equation_system))
    assert np.isclose(u_val, u_expected, rtol=1e-3)
    # u and h should be close with our setup
    assert np.isclose(u_val, h_val, rtol=1e-3)
    well_intf = setup.mdg.interfaces(codim=2)
    well_enthalpy_flux = setup.well_enthalpy_flux(well_intf).value(
        setup.equation_system
    )
    well_fluid_flux = setup.well_flux(well_intf).value(setup.equation_system)
    h_well = setup.fluid_enthalpy(setup.mdg.subdomains(dim=0)).value(
        setup.equation_system
    )
    # The well enthalpy flux should be equal to well enthalpy times well fluid flux
    assert np.isclose(well_enthalpy_flux, h_well * well_fluid_flux, rtol=1e-10)
    # Well fluid flux should equal the injected fluid. Minus for convention of interface
    # fluxes from higher to lower domain.
    assert np.isclose(well_fluid_flux, -1, rtol=1e-10)
