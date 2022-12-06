"""Tests for thermoporomechanics.

The hardcoded fracture gap value of 0.042 ensures positive aperture for all simulation.
This is needed to avoid degenerate mass and energy balance equations in the fracture.

TODO: Clean up.
"""
import numpy as np
import pytest

import porepy as pp

from .setup_utils import Thermoporomechanics
from .test_momentum_balance import BoundaryConditionsDirNorthSouth
from .test_poromechanics import NonzeroFractureGapPoromechanics


class TailoredThermoporomechanics(
    NonzeroFractureGapPoromechanics,
    BoundaryConditionsDirNorthSouth,
    Thermoporomechanics,
):
    ...


@pytest.mark.parametrize(
    "solid_vals,north_displacement",
    [
        ({}, 0.042),
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
    # Instantiate constants and store in params.
    solid_vals["fracture_gap"] = 0.042
    solid_vals["biot_coefficient"] = 0.8
    solid_vals["thermal_expansion"] = 1e-1
    fluid_vals = {"compressibility": 1, "thermal_expansion": 1e-1}
    solid = pp.SolidConstants(solid_vals)
    fluid = pp.FluidConstants(fluid_vals)

    params = {
        "suppress_export": False,  # Suppress output for tests
        "material_constants": {"solid": solid, "fluid": fluid},
        "north_displacement": north_displacement,
        "max_iterations": 50,
    }

    # Create model and run simulation
    setup = TailoredThermoporomechanics(params)
    pp.run_time_dependent_model(setup, params)

    # Check that the pressure is linear
    sd = setup.mdg.subdomains(dim=setup.nd)[0]
    u_var = setup.equation_system.get_variables([setup.displacement_variable], [sd])
    u_vals = setup.equation_system.get_variable_values(u_var).reshape(
        setup.nd, -1, order="F"
    )
    p_var = setup.equation_system.get_variables(
        [setup.pressure_variable], setup.mdg.subdomains()
    )
    p_vals = setup.equation_system.get_variable_values(p_var)
    t_var = setup.equation_system.get_variables(
        [setup.temperature_variable], setup.mdg.subdomains()
    )
    t_vals = setup.equation_system.get_variable_values(t_var)

    top = sd.cell_centers[1] > 0.5
    bottom = sd.cell_centers[1] < 0.5
    tol = 1e-10
    if np.isclose(north_displacement, 0.042):

        assert np.allclose(u_vals[:, bottom], 0)
        # Zero x and nonzero y displacement in top
        assert np.allclose(u_vals[0, top], 0)
        assert np.allclose(u_vals[1, top], 0.042)
        # Zero displacement relative to initial value implies zero pressure and
        # temperature
        assert np.allclose(p_vals, 0)
        assert np.allclose(t_vals, 0)
    elif north_displacement < 0.042:
        # Boundary displacement is negative, so the y displacement should be
        # negative
        assert np.all(u_vals[1] < 0)

        # Check that x displacement is negative for left half and positive for right
        # half. Tolerance excludes cells at the centerline, where the displacement is
        # zero.
        left = sd.cell_centers[0] < setup.box["xmax"] / 2 - tol
        assert np.all(u_vals[0, left] < 0)
        right = sd.cell_centers[0] > setup.box["xmax"] / 2 + tol
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

    # Check that the displacement jump and traction are as expected
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
    if north_displacement > 0.042:
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
        if north_displacement < 0.042:
            assert np.all(traction[setup.nd - 1 :: setup.nd] <= tol)
        else:
            assert np.allclose(traction, 0)
