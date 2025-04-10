"""Tests of functionality found within data_saving_model_mixin.py.

The following is covered:
* Test that only the specified exported times are exported.

"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import porepy as pp
from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)
from porepy.models.momentum_balance import MomentumBalance


class DataSavingModelMixinModel(SquareDomainOrthogonalFractures, MomentumBalance):
    """Model for testing data saving."""

    def write_pvd_and_vtu(self) -> None:
        """Logger for the times that are exported.

        This method is called for every time step that is to be exported. It is now
        converted to a logger, meaning that every time step that is to be exported is
        logged in the model attribute exported_times.

        """
        self.exported_times.append(self.time_manager.time)


@pytest.mark.parametrize(
    "times_to_export", [None, [], [0.0, 0.5, 0.6], [0.0, 0.2, 0.5, 0.4, 1.0]]
)
def test_export_chosen_times(times_to_export):
    """Testing if only exported times are exported.

    We test exporting of:
    * All time steps
    * No time steps
    * A selection of time steps in ascending order
    * A selection of time steps in random order

    """
    time_steps = 10
    tf = 1.0
    dt = tf / time_steps

    time_manager = pp.TimeManager(
        schedule=[0.0, tf],
        dt_init=dt,
        constant_dt=True,
    )

    model_params = {
        "time_manager": time_manager,
        "times_to_export": times_to_export,
    }

    model = DataSavingModelMixinModel(model_params)
    model.exported_times = []
    pp.run_time_dependent_model(model)

    # The actual test of exported times based on the log stored in model.exported_times:
    if times_to_export is None:
        times_to_export = np.linspace(0.0, tf, time_steps + 1)
        assert np.allclose(model.exported_times, times_to_export)
    else:
        assert np.allclose(model.exported_times, np.sort(times_to_export))


@pytest.mark.parametrize(
    "times_to_export, expected_times",
    [
        (
            None,
            [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25],
        ),
        ([0.0, 0.1, 0.2], [0.0, 0.1, 0.2]),
    ],
)
def test_compare_exported_times_and_unique_timesteps(times_to_export, expected_times):
    """Test that exported times match the times in times.json and unique timesteps in
    PVD."""
    folder_name = "viz_test_data_saving"
    time_steps = 10
    tf = 0.25
    dt = tf / time_steps

    time_manager = pp.TimeManager(
        schedule=[0.0, tf],
        dt_init=dt,
        constant_dt=True,
    )

    params = {
        "time_manager": time_manager,
        "times_to_export": times_to_export,
        "folder_name": folder_name,
    }

    class ModelForTesting(pp.SinglePhaseFlow):
        """"""

    model = ModelForTesting(params)
    pp.run_time_dependent_model(model, params)

    # Read times.json to get time data.
    times_file = Path(folder_name) / "times.json"
    with open(times_file, "r") as f:
        times_data = json.load(f)

    # Compare exported times with the times in times.json.
    assert np.allclose(model.time_manager.exported_times, times_data["time"])

    # Compare exported times with the expected times.
    assert np.allclose(model.time_manager.exported_times, expected_times)

    # Check that the correct number of files are exported.
    # Parse the PVD file
    pvd_file = Path(folder_name) / "data.pvd"
    tree = ET.parse(pvd_file)
    root = tree.getroot()

    # Extract unique timesteps
    timesteps = set()
    for dataset in root.findall(".//DataSet"):
        timestep = dataset.get("timestep")
        timesteps.add(float(timestep))

    # Compare unique timesteps with times.json
    assert np.allclose(sorted(timesteps), times_data["time"])
