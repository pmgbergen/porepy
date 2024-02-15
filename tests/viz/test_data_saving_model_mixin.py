"""Tests of functionality found within data_saving_model_mixin.py.

The following is covered:
* Test that only the specified exported times are exported.

"""

import porepy as pp
import numpy as np

from porepy.models.momentum_balance import MomentumBalance

import pytest


class DataSavingModelMixinSetup(MomentumBalance):
    """Model setup for testing."""

    def nd_rect_domain(self, x, y) -> pp.Domain:
        box: dict[str, pp.number] = {"xmin": 0, "xmax": x}

        box.update({"ymin": 0, "ymax": y})

        return pp.Domain(box)

    def set_domain(self) -> None:
        x = 1.0 / self.units.m
        y = 1.0 / self.units.m
        self._domain = self.nd_rect_domain(x, y)

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 0.1 / self.units.m}
        return mesh_args

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        bounds = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(
            sd,
            bounds.all_bf,
            "dir",
        )
        return bc

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        values = np.zeros((self.nd, bg.num_cells))
        bounds = self.domain_boundary_sides(bg)

        t = self.time_manager.time

        displacement_values = np.zeros((self.nd, bg.num_cells))

        # Time dependent sine Dirichlet condition
        values[1][bounds.west] += np.ones(
            len(displacement_values[0][bounds.west])
        ) * np.sin(t)

        return values.ravel("F")

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
        iter_max=10,
        print_info=True,
    )

    params = {
        "time_manager": time_manager,
        "times_to_export": times_to_export,
    }

    model = DataSavingModelMixinSetup(params)
    model.exported_times = []
    pp.run_time_dependent_model(model, params)

    # The actual test of exported times based on the log stored in model.exported_times:
    if times_to_export is None:
        scheduled_times = np.linspace(0.0, tf, time_steps + 1)
        assert len(model.exported_times) == len(scheduled_times)
    elif isinstance(times_to_export, list):
        assert np.all(np.isclose(model.exported_times, np.sort(times_to_export)))
