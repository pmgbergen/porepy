"""Tests for the mass balance model.

"""
from __future__ import annotations

import copy

import numpy as np
import pytest

import porepy as pp

from .setup_utils import (
    MassBalance,
    compare_scaled_model_quantities,
    compare_scaled_primary_variables,
)


class BoundaryConditionLinearPressure:
    """Overload the boundary condition to give a linear pressure profile.

    Homogeneous Neumann conditions on top and bottom, Dirichlet 1 and 0 on the
    left and right boundaries, respectively.

    """

    def bc_type_darcy(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Dirichlet conditions on all external boundaries.

        Parameters:
            sd: Subdomain grid on which to define boundary conditions.

        Returns:
            Boundary condition object.

        """
        # Define boundary regions
        all_bf, east, west, *_ = self.domain_boundary_sides(sd)
        # Define Dirichlet conditions on the left and right boundaries
        return pp.BoundaryCondition(sd, east + west, "dir")

    def bc_values_darcy(self, subdomains: list[pp.Grid]) -> pp.ad.Array:
        """Boundary values for the pressure.

        Parameters:
            subdomains: List of subdomains on which to define boundary conditions.

        Returns:
            Array of boundary values.

        """
        values = []
        for sd in subdomains:
            _, _, west, *_ = self.domain_boundary_sides(sd)
            val_loc = np.zeros(sd.num_faces)
            val_loc[west] = self.fluid.convert_units(1, "Pa")
            values.append(val_loc)
        if len(values) > 0:
            bc_values = np.hstack(values)
        else:
            bc_values = np.empty(0)
        return pp.wrap_as_ad_array(bc_values, name="bc_values_darcy")

    def bc_values_mobrho(self, subdomains: list[pp.Grid]) -> pp.ad.Array:
        """Boundary values for the mobility times density.

        Parameters:
            subdomains: List of subdomains on which to define boundary conditions.

        Returns:
            Array of boundary values.

        """
        # Define boundary regions
        values = []
        for sd in subdomains:
            all_bf, *_ = self.domain_boundary_sides(sd)
            val_loc = np.zeros(sd.num_faces)
            val_loc[all_bf] = self.fluid.density() / self.fluid.viscosity()
            values.append(val_loc)
        return pp.wrap_as_ad_array(np.hstack(values), name="bc_values_mobrho")


class LinearModel(BoundaryConditionLinearPressure, MassBalance):
    pass


@pytest.mark.parametrize(
    "fluid_vals,solid_vals",
    [
        ({}, {}),
        ({}, {"permeability": 10}),
        ({"viscosity": 0.1}, {"porosity": 0.5}),
    ],
)
def test_linear_pressure(fluid_vals, solid_vals):
    """Test that the pressure solution is linear.

    With constant density (zero compressibility) and the specified boundary conditions,
    the pressure solution should be linear, i.e.,
    ::math::
        p = 1 - x.

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
    setup = LinearModel(params)
    pp.run_time_dependent_model(setup, params)

    # Check that the pressure is linear
    for sd in setup.mdg.subdomains():
        var = setup.equation_system.get_variables(["pressure"], [sd])
        vals = setup.equation_system.get_variable_values(var)
        assert np.allclose(vals, 1 - sd.cell_centers[0] / setup.domain.bounding_box[
            "xmax"])

    # Check that the flux over each face is equal to the x component of the
    # normal vector
    for sd in setup.mdg.subdomains():
        val = setup.fluid_flux([sd]).evaluate(setup.equation_system).val
        # Account for specific volume, default value of .01 in fractures.
        normals = np.abs(sd.face_normals[0]) * np.power(0.1, setup.nd - sd.dim)
        k = setup.solid.permeability() / setup.fluid.viscosity()
        grad = 1 / setup.domain.bounding_box["xmax"]
        assert np.allclose(np.abs(val), normals * grad * k)


@pytest.mark.parametrize(
    "units",
    [
        {"m": 2, "kg": 3, "s": 1, "K": 1},
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
        "num_fracs": 2,
        "cartesian": True,
    }
    reference_params = copy.deepcopy(params)
    reference_params["file_name"] = "unit_conversion_reference"

    # Create model and run simulation
    setup_0 = LinearModel(reference_params)
    pp.run_time_dependent_model(setup_0, reference_params)

    params["units"] = pp.Units(**units)
    setup_1 = LinearModel(params)

    pp.run_time_dependent_model(setup_1, params)
    variables = [setup_1.pressure_variable, setup_1.interface_darcy_flux_variable]
    variable_units = ["Pa", "Pa * m^2 * s^-1"]
    compare_scaled_primary_variables(setup_0, setup_1, variables, variable_units)
    flux_names = ["darcy_flux", "fluid_flux"]
    flux_units = ["Pa * m^2 * s^-1", "kg * m^-1 * s^-1"]
    # No domain restrictions.
    domain_dimensions = [None, None]
    compare_scaled_model_quantities(
        setup_0, setup_1, flux_names, flux_units, domain_dimensions
    )
