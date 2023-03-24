"""Test for restart of a model.

Here, exemplarily for a mixed-dimensional poromechanics model with
time-varying boundary conditions.

"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

import porepy as pp

from ...unit.test_vtk import _compare_pvd_files, _compare_vtu_files
from .test_poromechanics import TailoredPoromechanics

# Store current directory
current_dir = os.path.dirname(os.path.realpath(__file__))


def create_fractured_setup(
    solid_vals: dict, fluid_vals: dict, uy_north: float, restart: bool
):
    """Create a setup for a fractured domain.

    This is an enhanced copy of .test_poromechanics.create_fractured_setup.
    It enables multiple time steps, and the export of the solution.

    Parameters:
        solid_vals: Parameters for the solid mechanics model.
        fluid_vals: Parameters for the fluid mechanics model.
        uy_north: Displacement in y-direction on the north boundary.
        restart: Flag controlling whether restart is used.

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
            "restart": restart,
            "reuse_dt": True,
            "file": current_dir + "/restart_reference/previous_data.pvd",
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

    This test also serves as minimal documentation of how to restart a model
    in a practical situation.

    Parameters:
        solid_vals (dict): Dictionary with keys as those in :class:`pp.SolidConstants`
            and corresponding values.
        north_displacement (float): Value of displacement on the north boundary.
        expected_x_y (tuple): Expected values of the displacement in the x and y.
            directions. The values are used to infer sign of displacement solution.

    """
    # Setup and run model for full time interval.
    setup = create_fractured_setup(solid_vals, {}, north_displacement, restart=False)
    pp.run_time_dependent_model(setup, {})

    # The run generates data for initial and the first two time steps.
    # In order to use the data as restart and reference data, move it
    # to a reference folder.
    pvd_files = list(Path(current_dir + "/visualization").glob("*.pvd"))
    vtu_files = list(Path(current_dir + "/visualization").glob("*.vtu"))
    for f in pvd_files + vtu_files:
        dst = Path(current_dir) / Path("restart_reference") / Path(f.stem + f.suffix)
        print(str(f), str(dst))
        shutil.move(str(f), str(dst))

    # Now use the reference data to restart the simulation.
    setup = create_fractured_setup(solid_vals, {}, north_displacement, restart=True)
    pp.run_time_dependent_model(setup, {})

    # To verify the restart capabilities, test whether:
    # - the states have been correctly initialized at restart time;
    # - the follow-up time step has been computed correctely;
    # - the overall solution stored in a global pvd file is compiled correctly.

    for ending in ["000001", "000002"]:
        for i in ["1", "2"]:
            assert _compare_vtu_files(
                current_dir + f"/visualization/data_{i}_{ending}.vtu",
                current_dir + f"/restart_reference/data_{i}_{ending}.vtu",
            )

            assert _compare_vtu_files(
                current_dir + f"/visualization/data_mortar_1_{ending}.vtu",
                current_dir + f"/restart_reference/data_mortar_1_{ending}.vtu",
            )

    for ending in ["_000002", ""]:
        assert _compare_pvd_files(
            current_dir + f"/visualization/data{ending}.pvd",
            current_dir + f"/restart_reference/data{ending}.pvd",
        )

    # Remove temporary visualization folder
    shutil.rmtree(current_dir + "/visualization")

    # Remove the reference data
    for f in pvd_files + vtu_files:
        src = Path(current_dir + "/restart_reference") / Path(f.stem + f.suffix)
        src.unlink()
