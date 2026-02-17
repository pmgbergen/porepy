"""Solver statistics object for non-linear solver loop."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Type

import numpy as np

from porepy.numerics.nonlinear.convergence_check import (
    ConvergenceInfoCollection,
    ConvergenceInfoHistory,
    ConvergenceStatusCollection,
    ConvergenceStatusHistory,
    SimulationStatus,
    _recursive_append,
)

logger = logging.getLogger(__name__)


# Auxiliary functions for appending dictionaries and exporting to json.
def _leafs_only(d: dict) -> dict:
    """Recursive function to extract only leafs of a dictionary."""
    if isinstance(d, dict):
        return {k: _leafs_only(v) for k, v in d.items()}
    elif isinstance(d, list):
        return d[-1]
    else:
        return d


class _NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy data types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class SolverStatistics:
    """Statistics object which keeps track of the convergence status of the solver.

    It is general enough for stationary and linear problems.

    """

    index: int = field(default=-1)
    """Current index of the statistics object - expected to be increased before use."""
    path: Optional[Path] = None
    """Path to save the statistics object to."""
    num_cells: dict[str, int] = field(default_factory=dict)
    """Number of cells in each dimension."""
    num_domains: dict[str, int] = field(default_factory=dict)
    """Number of domains in each dimension."""
    simulation_status: SimulationStatus = field(default=SimulationStatus.IN_PROGRESS)
    """Overall simulation status."""
    simulation_status_history: list[SimulationStatus] = field(default_factory=list)
    """Overall simulation status history."""
    custom_data: dict[str, Any] = field(default_factory=dict)
    """Custom data to be added to the statistics object."""

    def __replace__(self, **kwargs) -> "SolverStatistics":
        """Create a new instance with updated fields."""
        return type(self)(**{**self.__dict__, **kwargs})

    def increase_index(self) -> None:
        """Advance counting iterations, and reset the statistics object."""
        self.index += 1
        self.custom_data = dict[str, Any]()

    def log_mesh_information(self, subdomains: list, **kwargs) -> None:
        """Collect mesh information.

        Parameters:
            subdomains: List of subdomains in the model.
            **kwargs: Additional keyword arguments, for potential extension.

        """
        self.num_cells = dict[str, int]()
        self.num_domains = dict[str, int]()
        for sd in subdomains:
            dim_str = str(sd.dim)
            if dim_str not in self.num_cells:
                self.num_cells[dim_str] = 0
            self.num_cells[dim_str] += sd.num_cells
            if dim_str not in self.num_domains:
                self.num_domains[dim_str] = 0
            self.num_domains[dim_str] += 1

    def log_simulation_status(
        self, simulation_status: SimulationStatus, **kwargs
    ) -> None:
        """Log overall simulation status.

        Parameters:
            simulation_status: Overall simulation status.
            **kwargs: Additional keyword arguments, for potential extension.

        """
        if simulation_status is not None:
            self.simulation_status = simulation_status
            if len(self.simulation_status_history) <= self.index:
                self.simulation_status_history.append(self.simulation_status)
            else:
                self.simulation_status_history[-1] = self.simulation_status

    def log_custom_data(self, append: bool = False, **kwargs) -> None:
        """Log custom data to be added to the statistics object with custom keys.

        Has two modes:
        - If `append` is `False`, the custom data is added to the statistics object,
          potentially overwriting existing data with the same key.
        - If `append` is `True`, the custom data is appended to existing data with the
          same key, converting to a list if necessary.

        Parameters:
            append: Whether to append to existing data with the same key.
            **kwargs: Custom data to be added to the statistics object.

        """
        if append:
            # Append data to existing keys.
            self.custom_data = _recursive_append(self.custom_data, kwargs)
        else:
            # Overwrite existing data.
            self.custom_data.update(kwargs)

    def append_global_data(self, data: dict[str, dict]) -> dict[str, dict]:
        """Append the current statistics to the data dictionary.

        Parameters:
            data: Dictionary to append the statistics to.

        Returns:
            dict: Updated dictionary with global data.

        """

        str_simulation_status_history = [str(s) for s in self.simulation_status_history]
        final_str_simulation_status = (
            None
            if len(self.simulation_status_history) == 0
            else str_simulation_status_history[-1]
        )

        data["global"] = {
            "num_cells": self.num_cells,
            "num_domains": self.num_domains,
            "simulation_status_history": str_simulation_status_history,
            "final_simulation_status": final_str_simulation_status,
            "num_entries": self.index + 1,
        }

        return data

    def append_custom_data(self, data: dict[str, dict]) -> dict[str, dict]:
        """Update the statistics object with custom data.

        Parameters:
            data (dict): Dictionary to append the statistics to.

        Returns:
            dict: Updated dictionary with custom data.

        """
        data[str(self.index)].update(self.custom_data)
        return data

    def append_iterative_data(self, data: dict[str, dict]) -> dict[str, dict]:
        """Append the current statistics to the data dictionary at current index.

        Parameters:
            data: Dictionary to append the statistics to.

        Returns:
            dict: Updated dictionary with iterative data.

        """
        data[str(self.index)].update(
            {
                "simulation_status": str(self.simulation_status),
            }
        )
        return data

    def append_data(self, data: dict[str, dict]) -> dict[str, dict]:
        """Main function to append all data to the statistics dictionary.

        Parameters:
            data: Dictionary to append the statistics to.

        Returns:
            dict: Updated dictionary with all data.

        """
        # Disable appending in invalid mode.
        if self.index == -1:
            return data

        # Initialize index entry if non-existing.
        if str(self.index) not in data:
            data[str(self.index)] = {}

        # Append data.
        data = self.append_global_data(data)
        data = self.append_iterative_data(data)
        data = self.append_custom_data(data)

        return data

    def save(self) -> None:
        """Save the statistics object to a JSON file."""
        # Save to file.
        if self.path is not None:
            # Data management and data format safety.
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path = self.path.with_suffix(".json")

            # Load existing data if the file exists.
            if self.path.exists():
                with self.path.open("r") as file:
                    data = json.load(file)
                # Clean up obsolete information.
                available_indices = [int(k) for k in data.keys() if k.isdigit()]
                if len(available_indices) > 0:
                    max_index = max(int(k) for k in data.keys() if k.isdigit())
                    for k in range(self.index + 1, max_index + 1):
                        data.pop(str(k), None)
            else:
                data = {}

            # Update data
            data = self.append_data(data)

            # Save to file
            with self.path.open("w") as file:
                json.dump(data, file, indent=4, cls=_NumpyJSONEncoder)


@dataclass
class NonlinearSolverStatistics(SolverStatistics):
    """Statistics object for non-linear solver loop.

    This object keeps track of the number of non-linear iterations performed for the
    current time step, as well as increments and residuals for each iteration.

    Example:

        After storing solver statistics to file, we can load the file and plot the
        stored data, here for the first time step.

        >>> import matplotlib.pyplot as plt
        >>> import json
        >>> with open("solver_statistics.json", "r") as f:
        >>>     history = json.load(f)
        >>> time_step = str(1)
        >>> err = history[time_step]["residual_norms"]
        >>> plt.semilogy(err)
        >>> plt.xlabel("Iteration number")
        >>> plt.ylabel("Residual")
        >>> plt.title("Residual error")
        >>> plt.show()

    """

    num_iterations: int = field(default=0)
    """Number of (logged) non-linear iterations."""
    num_iterations_history: list[int] = field(default_factory=list)
    """History of number of iterations for entire run."""
    convergence_status: ConvergenceStatusHistory = field(
        default_factory=ConvergenceStatusHistory
    )
    """History of convergence status over nonlinear iterations."""
    convergence_info: ConvergenceInfoHistory = field(
        default_factory=ConvergenceInfoHistory
    )
    """History of convergence information over nonlinear iterations."""

    def __replace__(self, **kwargs) -> "NonlinearSolverStatistics":
        """Create a new instance with updated fields."""
        return type(self)(**{**self.__dict__, **kwargs})

    def increase_index(self) -> None:
        """Reset the statistics object."""
        super().increase_index()
        self.num_iterations = 0
        self.convergence_status.clear()
        self.convergence_info.clear()

    def log_convergence_status(
        self, convergence_status: ConvergenceStatusCollection, **kwargs
    ) -> None:
        """Log and collect the convergence status of the solver.

        Parameters:
            convergence_status: Convergence status of the solver.
            **kwargs: Additional keyword arguments, for potential extension.

        """
        # Append convergence status.
        self.convergence_status.append(convergence_status)

        # Update the number of iterations = length of the convergence status.
        num_convergence_status = [len(s) for s in self.convergence_status.values()]
        assert len(set(num_convergence_status)) == 1
        self.num_iterations = num_convergence_status[0]

        # Store global number of (attempted) iterations.
        if len(self.num_iterations_history) <= self.index:
            self.num_iterations_history.append(self.num_iterations)
        else:
            self.num_iterations_history[-1] = self.num_iterations

    def log_convergence_info(
        self, convergence_info: ConvergenceInfoCollection, **kwargs
    ) -> None:
        """Log info produced from convergence criteria.

        Parameters:
            convergence_info: Convergence information containing error norms.
            **kwargs: Additional keyword arguments, for potential extension.

        """
        self.convergence_info.append(convergence_info)

    def append_global_data(self, data: dict[str, dict]) -> dict[str, dict]:
        """Append the current statistics to the data dictionary.

        Stores the global number of iterations performed so far.

        Parameters:
            data: Dictionary to append the statistics to.

        Returns:
            dict: Updated dictionary with global data.

        """
        data = super().append_global_data(data)

        # Extract final convergence status.
        final_convergence_status = _leafs_only(self.convergence_status.to_str())

        # Determine number of waisted iterations.
        total_num_waisted_iterations = 0
        for simulation_status, num_iterations in zip(
            self.simulation_status_history, self.num_iterations_history
        ):
            if simulation_status != SimulationStatus.SUCCESSFUL:
                total_num_waisted_iterations += num_iterations

        # Update global data.
        data["global"].update(
            {
                "num_iterations_history": self.num_iterations_history,
                "total_num_iterations": sum(self.num_iterations_history),
                "total_num_waisted_iterations": total_num_waisted_iterations,
                "final_convergence_status": final_convergence_status,
            }
        )
        return data

    def append_iterative_data(self, data: dict[str, dict]) -> dict[str, dict]:
        """Append the current statistics to the data dictionary at current index."""

        data = super().append_iterative_data(data)
        data[str(self.index)].update(
            {
                "num_iterations": self.num_iterations,
                "simulation_status": str(
                    None
                    if len(self.simulation_status_history) == 0
                    else self.simulation_status_history[-1]
                ),
                "convergence_status": self.convergence_status.to_str().copy(),
                "convergence_info": self.convergence_info.copy(),
            }
        )

        return data


@dataclass
class TimeStatistics(SolverStatistics):
    """Mixin making SolverStatistics aware of time information for each iteration.

    Note: This class is intended to be used as a mixin with SolverStatistics.
    It assumes that the class it is mixed into has a `index` attribute.

    """

    time_index: int = field(default=0)
    """Current time step index."""
    time: float = field(default=0.0)
    """Current time."""
    dt: float = field(default=0.0)
    """Current time step size."""
    final_time_reached: bool = field(default=False)
    """Whether the final time has been reached."""

    def __replace__(self, **kwargs) -> "TimeStatistics":
        """Create a new instance with updated fields."""
        return type(self)(**{**self.__dict__, **kwargs})

    def log_time_information(
        self,
        time_index: int,
        time: float,
        dt: float,
        final_time_reached: bool,
        **kwargs,
    ) -> None:
        """Log time information.

        Parameters:
            time_index: Current time step index.
            time: Current time.
            dt: Current time step size.
            **kwargs: Additional keyword arguments, for potential extension.

        """
        self.time_index = time_index
        self.time = time
        self.dt = dt
        self.final_time_reached = final_time_reached

    def append_global_data(self, data: dict[str, dict]) -> dict[str, dict]:
        """Append the current statistics to the data dictionary.

        Parameters:
            data: Dictionary to append the statistics to.

        Returns:
            dict: Updated dictionary with global data.

        """
        data = super().append_global_data(data)

        # Determine number of total and failed time steps.
        # Simulation status identifies success of time step.
        total_num_time_steps = 0
        total_num_failed_time_steps = 0
        for simulation_status in self.simulation_status_history:
            total_num_time_steps += 1
            if simulation_status != SimulationStatus.SUCCESSFUL:
                total_num_failed_time_steps += 1

        # Update global data.
        data["global"].update(
            {
                "final_time_reached": int(self.final_time_reached),
                "total_num_time_steps": total_num_time_steps,
                "total_num_failed_time_steps": total_num_failed_time_steps,
            }
        )
        return data

    def append_iterative_data(self, data: dict[str, dict]) -> dict[str, dict]:
        """Append the current statistics to the data dictionary.

        Parameters:
            data: Dictionary to append the statistics to.

        Returns:
            dict: Updated dictionary with iterative data.
        """
        data = super().append_iterative_data(data)
        data[str(self.index)].update(
            {
                "final_time_reached": int(self.final_time_reached),
                "time_index": self.time_index,
                "time": self.time,
                "dt": self.dt,
            }
        )
        return data


class NonlinearSolverAndTimeStatistics(NonlinearSolverStatistics, TimeStatistics):
    """Combined statistics class for nonlinear solvers with time dependence.

    This class combines the statistics from both nonlinear solvers and
    time-dependent solvers.

    """

    def __replace__(self, **kwargs) -> "NonlinearSolverAndTimeStatistics":
        """Create a new instance with updated fields."""
        return type(self)(**{**self.__dict__, **kwargs})


# Collect all statistics types for type hinting.
StatisticsType = (
    Type[SolverStatistics]
    | Type[NonlinearSolverStatistics]
    | Type[TimeStatistics]
    | Type[NonlinearSolverAndTimeStatistics]
)


# Create a map from (nonlinear, time_dependent) to the appropriate statistics class.
_statistics_map: dict[tuple[bool, bool], StatisticsType] = {
    (True, True): NonlinearSolverAndTimeStatistics,
    (True, False): NonlinearSolverStatistics,
    (False, True): TimeStatistics,
    (False, False): SolverStatistics,
}


class SolverStatisticsFactory:
    """Factory class to create appropriate SolverStatistics subclasses."""

    @staticmethod
    def create_statistics_type(nonlinear: bool, time_dependent: bool) -> StatisticsType:
        return _statistics_map[(nonlinear, time_dependent)]
