"""Tests of functionality found within data_saving_model_mixin.py.

The following is covered:
* Test that only the specified exported times are exported.

"""

import numpy as np

import porepy as pp
from porepy.models.momentum_balance import MomentumBalance
from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)

import pytest


class DataSavingModelMixinSetup(SquareDomainOrthogonalFractures, MomentumBalance):
    """Model setup for testing."""

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

    model = DataSavingModelMixinSetup(model_params)
    model.exported_times = []
    pp.run_time_dependent_model(model)

    # The actual test of exported times based on the log stored in model.exported_times:
    if times_to_export is None:
        times_to_export = np.linspace(0.0, tf, time_steps + 1)
        assert np.allclose(model.exported_times, times_to_export)
    else:
        assert np.allclose(model.exported_times, np.sort(times_to_export))
