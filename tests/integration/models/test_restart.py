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
from porepy.applications.test_utils.vtk import compare_pvd_files, compare_vtu_files

from .test_poromechanics import TailoredPoromechanics, create_fractured_setup

# Store current directory, directory containing reference files, and temporary
# visualization folder.
current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
reference_dir = current_dir / Path("restart_reference")
visualization_dir = Path("visualization")


def create_enhanced_fractured_setup(
    solid_vals: dict, fluid_vals: dict, uy_north: float, restart: bool
):
    # Create fractured setup
    fractured_setup = create_fractured_setup(solid_vals, fluid_vals, uy_north)

    # Fetch parameters for enhancing them
    params = fractured_setup.params

    # Enable exporting
    params["suppress_export"] = False

    # Add time stepping to the setup
    params["time_manager"] = pp.TimeManager(
        schedule=[0, 1], dt_init=0.5, constant_dt=True
    )

    # Add restart possibility
    params["restart_options"] = {
        "restart": restart,
        "pvd_file": reference_dir / Path("previous_data.pvd"),
        "times_file": reference_dir / Path("previous_times.json"),
    }

    # Redefine setup
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
    # Setup and run model for full time interval. With this generate reference files
    # for comparison with a restarted simulation. At the same time, this generates the
    # restart files.
    setup = create_enhanced_fractured_setup(
        solid_vals, {}, north_displacement, restart=False
    )
    pp.run_time_dependent_model(setup, {})

    # The run generates data for initial and the first two time steps. In order to use
    # the data as restart and reference data, move it to a reference folder.
    pvd_files = list(visualization_dir.glob("*.pvd"))
    vtu_files = list(visualization_dir.glob("*.vtu"))
    json_files = list(visualization_dir.glob("*.json"))
    for f in pvd_files + vtu_files + json_files:
        dst = reference_dir / Path(f.stem + f.suffix)
        shutil.move(f, dst)

    # Now use the reference data to restart the simulation. Note, the restart
    # capabilities of the models automatically use the last available time step for
    # restart, here the restart files contain information on the initial and first
    # time step. Thus, the simulation is restarted from the first time step.
    # Recompute the second time step which will serve as foundation for the comparison
    # to the above computed reference files.
    setup = create_enhanced_fractured_setup(
        solid_vals, {}, north_displacement, restart=True
    )
    pp.run_time_dependent_model(setup, {})

    # To verify the restart capabilities, perform five tests.

    # 1. Check whether the states have been correctly initialized at restart time.
    # Visit all dimensions and the mortar grids for this.
    for i in ["1", "2"]:
        assert compare_vtu_files(
            visualization_dir / Path(f"data_{i}_000001.vtu"),
            reference_dir / Path(f"data_{i}_000001.vtu"),
        )
    assert compare_vtu_files(
        visualization_dir / Path(f"data_mortar_1_000001.vtu"),
        reference_dir / Path(f"data_mortar_1_000001.vtu"),
    )

    # 2. Check whether the successive time step has been computed correctely.
    # Visit all dimensions and the mortar grids for this.
    for i in ["1", "2"]:
        assert compare_vtu_files(
            visualization_dir / Path(f"data_{i}_000002.vtu"),
            reference_dir / Path(f"data_{i}_000002.vtu"),
        )
    assert compare_vtu_files(
        visualization_dir / Path(f"data_mortar_1_000002.vtu"),
        reference_dir / Path(f"data_mortar_1_000002.vtu"),
    )

    # 3. Check whether the mdg pvd file is defined correctly.
    assert compare_pvd_files(
        visualization_dir / Path(f"data_000002.pvd"),
        reference_dir / Path(f"data_000002.pvd"),
    )

    # 4. Check whether the pvd file is compiled correctly, combining old and new data.
    assert compare_pvd_files(
        visualization_dir / Path(f"data.pvd"),
        reference_dir / Path(f"data.pvd"),
    )

    # 5. the logging of times and step sizes is correct.
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
