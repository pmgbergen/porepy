"""Solver statistics object for non-linear solver loop."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Type

import numpy as np

from porepy.numerics.nonlinear.convergence_check import ConvergenceStatus
from copy import copy

logger = logging.getLogger(__name__)


def _recursive_append(d: dict, v: dict) -> dict:
    """Auxiliary function to recursively append dictionaries."""
    for key_v, value_v in v.items():
        if key_v not in d:
            d[key_v] = copy(value_v)
        else:
            if isinstance(d[key_v], dict):
                assert isinstance(value_v, dict)
                d[key_v] = _recursive_append(d[key_v], value_v)
            elif isinstance(d[key_v], list):
                assert isinstance(value_v, (float, int))
                d[key_v].append(value_v)
            else:
                assert isinstance(value_v, (float, int))
                d[key_v] = [d[key_v], value_v]

    return d


class NumpyJSONEncoder(json.JSONEncoder):
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

    counter: int = field(default=0)
    """Counter for the number of times the statistics object has been updated."""
    path: Optional[Path] = None
    """Path to save the statistics object to."""
    convergence_status: ConvergenceStatus = field(
        default=ConvergenceStatus.NOT_CONVERGED
    )
    """Convergence status of the solver."""
    num_cells: list[int] = field(default_factory=list)
    """Number of cells in each dimension."""
    custom_data: dict[str, Any] = field(default_factory=dict)
    """Custom data to be added to the statistics object."""

    def __replace__(self, **kwargs) -> "SolverStatistics":
        """Create a new instance with updated fields."""
        return type(self)(**{**self.__dict__, **kwargs})

    def log_mesh_information(self, subdomains: list, **kwargs) -> None:
        """Collect mesh information.

        Parameters:
            subdomains: List of subdomains in the model.
            **kwargs: Additional keyword arguments, for potential extension.

        """
        max_dim = -1
        for sd in subdomains:
            max_dim = max(max_dim, sd.dim)
        self.num_cells = [0] * (max_dim + 1)
        for sd in subdomains:
            self.num_cells[sd.dim] += sd.num_cells

    def log_convergence_status(
        self, convergence_status: ConvergenceStatus, **kwargs
    ) -> None:
        """Log the convergence status of the solver.

        Parameters:
            convergence_status: Convergence status of the solver.
            **kwargs: Additional keyword arguments, for potential extension.

        """
        self.convergence_status = convergence_status

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

    def reset(self) -> None:
        """Reset the statistics object, and restart counting iterations."""
        self.counter += 1
        self.convergence_status = ConvergenceStatus.NOT_CONVERGED
        self.custom_data = dict[str, Any]()

    def append_global_data(self, data: dict[str, dict]) -> dict[str, dict]:
        """Append the current statistics to the data dictionary.

        Parameters:
            data: Dictionary to append the statistics to.

        Returns:
            dict: Updated dictionary with global data.

        """

        # Store static geometry data.
        data["global"] = {
            "num_cells": self.num_cells,
            "convergence_status": str(self.convergence_status),
            "latest_counter": self.counter,
        }

        return data

    def append_custom_data(self, data: dict[str, dict]) -> dict[str, dict]:
        """Update the statistics object with custom data.

        Parameters:
            data (dict): Dictionary to append the statistics to.

        Returns:
            dict: Updated dictionary with custom data.

        """
        data[str(self.counter)].update(self.custom_data)
        return data

    def append_iterative_data(self, data: dict[str, dict]) -> dict[str, dict]:
        """Append the current statistics to the data dictionary at current counter.

        Parameters:
            data: Dictionary to append the statistics to.

        Returns:
            dict: Updated dictionary with iterative data.

        """
        return data

    def append_data(self, data: dict[str, dict]) -> dict[str, dict]:
        """Main function to append all data to the statistics dictionary.

        Parameters:
            data: Dictionary to append the statistics to.

        Returns:
            dict: Updated dictionary with all data.

        """
        data = self.append_global_data(data)
        data = self.append_iterative_data(data)
        data = self.append_custom_data(data)
        return data

    def save(self) -> None:
        """Save the statistics object to a JSON file."""
        # Save to file.
        if self.path is not None:
            # Load existing data if the file exists.
            if self.path.exists():
                with self.path.open("r") as file:
                    data = json.load(file)
                # Clean up obsolete information.
                max_counter = max(int(k) for k in data.keys() if k.isdigit())
                for k in range(self.counter + 1, max_counter + 1):
                    data.pop(str(k), None)
            else:
                data = {}

            # Update data
            data = self.append_data(data)

            # Save to file
            with self.path.open("w") as file:
                json.dump(data, file, indent=4, cls=NumpyJSONEncoder)


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

    num_iteration: int = field(default=0)
    """Number of non-linear iterations performed for current time step."""
    info: list[float] | dict[str, list[float]] = field(default_factory=list)
    """Convergence information containing error norms."""
    global_num_iteration: list[int] = field(default_factory=list)
    """List of number of iterations for entire run."""

    def __replace__(self, **kwargs) -> "NonlinearSolverStatistics":
        """Create a new instance with updated fields."""
        return type(self)(**{**self.__dict__, **kwargs})

    def advance_iteration(self) -> None:
        """Advance the iteration count by one."""
        self.num_iteration += 1

    def log_info(self, info: dict | float, **kwargs) -> None:
        """Log info produced from convergence criteria.

        Parameters:
            info: Convergence information containing error norms.
            **kwargs: Additional keyword arguments, for potential extension.

        """
        if isinstance(info, dict):
            if not isinstance(self.info, dict):
                assert len(self.info) == 0
                self.info = {}
            self.info = _recursive_append(self.info, info)
        elif isinstance(info, float):
            if not isinstance(self.info, list):
                assert len(self.info) == 0
                self.info = []
            self.info.append(info)
        else:
            raise TypeError(
                """Info must be either a float or a dictionary, """
                f"""got {type(info)}"""
            )

    def reset(self) -> None:
        """Reset the statistics object, and restart counting iterations."""
        super().reset()
        self.num_iteration = 0
        self.info.clear()

    def append_global_data(self, data: dict[str, dict]) -> dict[str, dict]:
        """Append the current statistics to the data dictionary.

        Stores the global number of iterations performed so far.

        Parameters:
            data: Dictionary to append the statistics to.

        Returns:
            dict: Updated dictionary with global data.

        """

        data = super().append_global_data(data)
        self.global_num_iteration.append(self.num_iteration)
        data["global"].update(
            {
                "num_iteration": self.global_num_iteration,
            }
        )
        return data

    def append_iterative_data(self, data: dict[str, dict]) -> dict[str, dict]:
        """Append the current statistics to the data dictionary at current counter."""

        data = super().append_iterative_data(data)
        data[str(self.counter)].update(
            {
                "num_iteration": self.num_iteration,
                "convergence_status": str(self.convergence_status),
                "info": self.info,
            }
        )

        return data


@dataclass
class TimeStatistics(SolverStatistics):
    """Mixin making SolverStatistics aware of time information for each iteration.

    Note: This class is intended to be used as a mixin with SolverStatistics.
    It assumes that the class it is mixed into has a `counter` attribute.

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
        data["global"].update(
            {
                "final_time_reached": int(self.final_time_reached),
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
        data[str(self.counter)] = {
            "final_time_reached": int(self.final_time_reached),
            "time_index": self.time_index,
            "time": self.time,
            "dt": self.dt,
        }
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
