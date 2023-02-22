"""This mixin class is used to save data from a model to file.

It is combined with a Model class and provides methods for saving data to file.

We provide basic Exporter functionality, but the user is free to override
and extend this class to suit their needs. This could include, e.g., saving
data to a database, or to a file format other than vtu.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

import porepy as pp


class DataSavingMixin:
    """Class for saving data from a simulation model.

    Contract with other classes:
        The model should/may call save_data_time_step() at the end of each time step.
        The model should/may call finalize_save_data() at the end of the simulation.

    """

    equation_system: pp.EquationSystem
    """Equation system manager."""
    params: dict[str, Any]
    """Dictionary of parameters. May contain data saving parameters."""
    time_manager: pp.TimeManager
    """Time manager for the simulation."""
    mdg: pp.MixedDimensionalGrid
    """Mixed dimensional grid for the simulation."""

    def save_data_time_step(self) -> None:
        """Export the model state at a given time step."""
        if not self.suppress_export:
            self.exporter.write_vtu(
                self.data_to_export(),
                time_dependent=True,
            )

    def finalize_data_saving(self) -> None:
        """Export pvd file and finalize export."""
        if not self.suppress_export:
            self.exporter.write_pvd()

    def data_to_export(self):
        """Return data to be exported.

        Return type should comply with pp.exporter.DataInput.

        Returns:
            List containing all variable names.

        """
        var_names = [var.name for var in self.equation_system.variables]
        return var_names

    def initialize_data_saving(self) -> None:
        """Initialize data saving.

        This method is called by :meth:`prepare_simulation` to initialize the exporter,
        and any other data saving functionality (e.g., empty data containers to be
        appended in :meth:`save_data_time_step`).

        """
        self.exporter = pp.Exporter(
            self.mdg,
            self.params["file_name"],
            folder_name=self.params["folder_name"],
            export_constants_separately=self.params.get(
                "export_constants_separately", False
            ),
        )

    @property
    def suppress_export(self) -> bool:
        """Suppress export of data to file."""
        return self.params.get("suppress_export", False)


class VerificationDataSaving(DataSavingMixin):
    """Class to store relevant data for a generic verification setup."""

    _nonlinear_iteration: int
    """Number of non-linear iterations needed to solve the system. Used only as an
    indicator to avoid saving the initial conditions.

    """

    _is_time_dependent: Callable[[], bool]
    """Whether the problem is time-dependent."""

    results: list
    """List of objects containing the results of the verification."""

    def save_data_time_step(self) -> None:
        """Save data to the `results` list."""
        if not self._is_time_dependent():  # stationary problem
            if self._nonlinear_iteration > 0:  # avoid saving initial condition
                collected_data = self.collect_data()
                self.results.append(collected_data)
        else:  # time-dependent problem
            t = self.time_manager.time  # current time
            scheduled = self.time_manager.schedule[1:]  # scheduled times except t_init
            if any(np.isclose(t, scheduled)):
                collected_data = self.collect_data()
                self.results.append(collected_data)

    def collect_data(self):
        """Collect relevant data for the verification setup."""
        raise NotImplementedError()
