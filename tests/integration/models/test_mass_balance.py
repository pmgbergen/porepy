"""Tests for model variables.

"""
import numpy as np
import pytest

import porepy as pp
from porepy.models.constitutive_laws import ad_wrapper

from .setup_utils import MassBalanceCombined


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

    def bc_values_darcy_flux(self, subdomains: list[pp.Grid]) -> pp.ad.Array:
        """
        Not sure where this one should reside.
        Note that we could remove the grid_operator BC and DirBC, probably also
        ParameterArray/Matrix (unless needed to get rid of pp.ad.Discretization. I don't see
        how it would be, though).
        Parameters:
            subdomains: List of subdomains on which to define boundary conditions.

        Returns:
            Array of boundary values.

        """
        # Define boundary regions
        values = []
        for sd in subdomains:
            _, _, west, *_ = self.domain_boundary_sides(sd)
            val_loc = np.zeros(sd.num_faces)
            val_loc[west] = 1
            values.append(val_loc)
        return ad_wrapper(np.hstack(values), True, name="bc_values_darcy")

    def bc_values_mobrho(self, subdomains: list[pp.Grid]) -> pp.ad.Array:
        """
        Not sure where this one should reside.
        Note that we could remove the grid_operator BC and DirBC, probably also
        ParameterArray/Matrix (unless needed to get rid of pp.ad.Discretization. I don't see
        how it would be, though).
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
        return ad_wrapper(np.hstack(values), True, name="bc_values_mobrho")


class LinearModel(BoundaryConditionLinearPressure, MassBalanceCombined):
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
        assert np.allclose(vals, 1 - sd.cell_centers[0] / setup.box["xmax"])

    # Check that the flux over each face is equal to the x component of the
    # normal vector
    for sd in setup.mdg.subdomains():
        val = setup.fluid_flux([sd]).evaluate(setup.equation_system).val
        # Account for specific volume, default value of .01 in fractures.
        normals = np.abs(sd.face_normals[0]) * np.power(0.1, setup.nd - sd.dim)
        k = setup.solid.permeability() / setup.fluid.viscosity()
        grad = 1 / setup.box["xmax"]
        assert np.allclose(np.abs(val), normals * grad * k)
