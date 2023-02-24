from __future__ import annotations

import pytest

import porepy as pp

from .test_poromechanics import TailoredPoromechanics


def create_fractured_setup(solid_vals, fluid_vals, uy_north):
    """Create a setup for a fractured domain.

    This is an enhanced copy of .test_poromechanics.create_fractured_setup.
    It enables multiple time steps, and the export of the solution.

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
        "suppress_export": False,  # Suppress output for tests
        "material_constants": {"solid": solid, "fluid": fluid},
        "uy_north": uy_north,
        "max_iterations": 20,
        # TODO fix DataSavingMixin.write_pvd and use [0,1] with dt_init=0.5 here.
        "time_manager": pp.TimeManager(schedule=[0, 2], dt_init=1, constant_dt=True),
        "restart_options": {
            "restart": True,
            "file": "./restart_reference/previous_data.pvd",
        },
    }
    setup = TailoredPoromechanics(params)
    return setup


@pytest.mark.parametrize(
    "solid_vals,north_displacement",
    [
        ({"porosity": 0.5}, 0.1),
    ],
)
def test_restart_2d_single_fracture(solid_vals, north_displacement):
    """Restart version of .test_poromechanics.test_2d_single_fracture.

    Provided the exported data from a previous time step, restart the
    simulaton, continue running and compare the final state and exported
    vtu/pvd files with reference files.

    Parameters:
        solid_vals (dict): Dictionary with keys as those in :class:`pp.SolidConstants`
            and corresponding values.
        north_displacement (float): Value of displacement on the north boundary.
        expected_x_y (tuple): Expected values of the displacement in the x and y.
            directions. The values are used to infer sign of displacement solution.


    """
    setup = create_fractured_setup(solid_vals, {}, north_displacement)
    pp.run_time_dependent_model(setup, {})
    assert False
