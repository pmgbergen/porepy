"""Tests for model variables.

"""
from __future__ import annotations

import copy

import numpy as np
import pytest

import porepy as pp
from porepy.applications.test_utils.models import (
    MomentumBalance,
    compare_scaled_model_quantities,
    compare_scaled_primary_variables,
)


class LinearModel(
    pp.model_boundary_conditions.BoundaryConditionsMechanicsDirNorthSouth,
    MomentumBalance,
):
    pass


@pytest.mark.parametrize(
    "solid_vals,north_displacement",
    [
        ({}, 0),
        ({}, -0.1),
        ({"porosity": 0.5}, 0.2),
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
    # Instantiate constants and store in params.
    solid = pp.SolidConstants(solid_vals)
    params = {
        "suppress_export": True,  # Suppress output for tests
        "material_constants": {"solid": solid},
        "uy_north": north_displacement,
    }

    # Create model and run simulation
    setup = LinearModel(params)
    pp.run_time_dependent_model(setup, params)

    # Check that the pressure is linear
    sd = setup.mdg.subdomains(dim=setup.nd)[0]
    var = setup.equation_system.get_variables([setup.displacement_variable], [sd])
    vals = setup.equation_system.get_variable_values(variables=var, time_step_index=0)
    if np.isclose(north_displacement, 0):
        assert np.allclose(vals, 0)
    else:
        if north_displacement < 0:
            # Boundary displacement is negative, so the y displacement should be
            # negative
            assert np.all(np.sign(vals[setup.nd - 1 :: setup.nd]) < 0)

            # Check that x displacement has the same sign as north_displacement for
            # x<0.5, and the opposite sign for x>0.5. To see why this makes sense, think
            # through what happens around the symmetry line of x=0.5 when pulling or
            # pushing the top (north) boundary.
            tol = 1e-10
            left = sd.cell_centers[0] < setup.domain.bounding_box["xmax"] / 2 - tol
            right = sd.cell_centers[0] > setup.domain.bounding_box["xmax"] / 2 + tol
            assert np.all(
                np.sign(vals[:: setup.nd][left]) == np.sign(north_displacement)
            )
            assert np.all(
                np.sign(vals[:: setup.nd][right]) == -np.sign(north_displacement)
            )
        else:
            # Check that y displacement is positive in top half of domain
            top = sd.cell_centers[1] > 0.5
            assert np.all(
                np.sign(vals[setup.nd - 1 :: setup.nd][top])
                == np.sign(north_displacement)
            )
            # Fracture cuts the domain in half, so the bottom half should be undispalced
            bottom = sd.cell_centers[1] < 0.5
            assert np.allclose(vals[setup.nd - 1 :: setup.nd][bottom], 0)
            # No displacement in x direction
            assert np.allclose(vals[:: setup.nd], 0)

    # Check that the displacement jump and traction are as expected
    sd_frac = setup.mdg.subdomains(dim=setup.nd - 1)
    jump = setup.displacement_jump(sd_frac).evaluate(setup.equation_system).val
    traction = setup.contact_traction(sd_frac).evaluate(setup.equation_system).val
    if north_displacement > 0:
        # Normal component of displacement jump should be positive
        assert np.all(jump[setup.nd - 1 :: setup.nd] > 0)
        # Traction should be zero
        assert np.allclose(traction, 0)
    else:
        # Displacement jump should be zero
        assert np.all(np.isclose(jump, 0))
        # Normal traction should be non-positive. Zero if north_displacement is zero.,
        if north_displacement < 0:
            assert np.all(traction[setup.nd - 1 :: setup.nd] <= 0)
        else:
            assert np.allclose(traction, 0)


@pytest.mark.parametrize("units", [{"m": 0.29, "kg": 0.31, "K": 4.1}])
@pytest.mark.parametrize("uy_north", [0.2, -0.1])
def test_unit_conversion(units, uy_north):
    """Test that solution is independent of units.

    Parameters:
        units (dict): Dictionary with keys as those in
            :class:`~pp.models.material_constants.MaterialConstants`.
        uy_north (float): Value of displacement on the north boundary.

    """

    params = {
        "suppress_export": True,  # Suppress output for tests
        "fracture_indices": [0, 1],
        "cartesian": True,
        "uy_north": uy_north,
        "max_iterations": 10,
    }
    reference_params = copy.deepcopy(params)

    # Create model and run simulation
    setup_0 = LinearModel(reference_params)
    pp.run_time_dependent_model(setup_0, reference_params)

    params["units"] = pp.Units(**units)
    setup_1 = LinearModel(params)

    pp.run_time_dependent_model(setup_1, params)
    variables = [
        setup_1.displacement_variable,
        setup_1.interface_displacement_variable,
        setup_1.contact_traction_variable,
    ]
    variable_units = ["m", "m", "Pa"]
    compare_scaled_primary_variables(setup_0, setup_1, variables, variable_units)
    secondary_variables = ["stress", "displacement_jump"]
    secondary_units = ["Pa * m", "m"]
    domain_dimensions = [2, 1]
    compare_scaled_model_quantities(
        setup_0, setup_1, secondary_variables, secondary_units, domain_dimensions
    )
