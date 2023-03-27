"""Test for restart of a model.

Here, exemplarily for a mixed-dimensional poromechanics model with time-varying boundary
conditions.

"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import porepy as pp

from ...unit.test_vtk import _compare_pvd_files, _compare_vtu_files
from .test_poromechanics import TailoredPoromechanics

# Store current directory, directory containing reference files, and temporary
# visualization folder.
current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
reference_dir = current_dir / Path("restart_reference")
visualization_dir = Path("visualization")


def create_fractured_setup(
    solid_vals: dict, fluid_vals: dict, uy_north: float, restart: bool
):
    """Create a setup for a fractured domain.

    This is an enhanced copy of .test_poromechanics.create_fractured_setup. It enables
    multiple time steps, and the export of the solution.

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
        "time_manager": pp.TimeManager(schedule=[0, 1], dt_init=0.5, constant_dt=True),
        "restart_options": {
            "restart": restart,
            "pvd_file": reference_dir / Path("previous_data.pvd"),
            "times_file": reference_dir / Path("previous_times.json"),
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

    Provided the exported data from a previous time step, restart the simulaton,
    continue running and compare the final state and exported vtu/pvd files with
    reference files.

    This test also serves as minimal documentation of how to restart a model in a
    practical situation.

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
    pvd_files = list(visualization_dir.glob("*.pvd"))
    vtu_files = list(visualization_dir.glob("*.vtu"))
    json_files = list(visualization_dir.glob("*.json"))
    for f in pvd_files + vtu_files + json_files:
        dst = reference_dir / Path(f.stem + f.suffix)
        shutil.move(f, dst)

    # Now use the reference data to restart the simulation.
    setup = create_fractured_setup(solid_vals, {}, north_displacement, restart=True)
    pp.run_time_dependent_model(setup, {})

    # To verify the restart capabilities, test whether:
    # - the states have been correctly initialized at restart time;
    # - the follow-up time step has been computed correctely;
    # - the overall solution stored in a global pvd file is compiled correctly;
    # - the logging of times and step sizes is correct.

    for ending in ["000001", "000002"]:
        for i in ["1", "2"]:
            assert _compare_vtu_files(
                visualization_dir / Path(f"data_{i}_{ending}.vtu"),
                reference_dir / Path(f"data_{i}_{ending}.vtu"),
            )

            assert _compare_vtu_files(
                visualization_dir / Path(f"data_mortar_1_{ending}.vtu"),
                reference_dir / Path(f"data_mortar_1_{ending}.vtu"),
            )

    for ending in ["_000002", ""]:
        assert _compare_pvd_files(
            visualization_dir / Path(f"data{ending}.pvd"),
            reference_dir / Path(f"data{ending}.pvd"),
        )

    restarted_times_json = open(visualization_dir / Path("times.json"))
    reference_times_json = open(reference_dir / Path("times.json"))
    restarted_times = json.load(restarted_times_json)
    reference_times = json.load(reference_times_json)
    for key in ["time", "dt"]:
        assert np.all(
            np.isclose(np.array(restarted_times[key]), np.array(reference_times[key]))
        )
    restarted_times_json.close()
    reference_times_json.close()

    # Remove temporary visualization folder
    shutil.rmtree(visualization_dir)

    # Clean up the reference data
    for f in pvd_files + vtu_files + json_files:
        src = reference_dir / Path(f.stem + f.suffix)
        src.unlink()
