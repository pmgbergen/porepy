"""Solver statistics object for non-linear solver loop."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SolverStatistics:
    """Statistics object for non-linear solver loop.

    This object keeps track of the number of non-linear iterations performed for the
    current time step, as well as increments and residuals for each iteration.

    Example:

        After storing solver statistics to file, we can load the file and plot
        the stored data, here for the first time step.

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

    num_iteration: int = 0
    """Number of non-linear iterations performed for current time step."""
    nonlinear_increment_norms: list[float] = field(default_factory=list)
    """List of increment magnitudes for each non-linear iteration."""
    residual_norms: list[float] = field(default_factory=list)
    """List of residual for each non-linear iteration."""
    path: Optional[Path] = None
    """Path to save the statistics object to."""

    def log_error(
        self, nonlinear_increment_norm: float, residual_norm: float, **kwargs
    ) -> None:
        """Log errors produced from convergence criteria.

        Parameters:
            nonlinear_increment_norm (float): Error in the increment.
            residual_norm (float): Error in the residual.
            **kwargs: Additional keyword arguments, for potential extension.

        """
        self.nonlinear_increment_norms.append(nonlinear_increment_norm)
        self.residual_norms.append(residual_norm)

    def reset(self) -> None:
        """Reset the statistics object."""
        self.num_iteration = 0
        self.nonlinear_increment_norms.clear()
        self.residual_norms.clear()

    def save(self) -> None:
        """Save the statistics object to a JSON file."""
        if self.path is not None:
            # Check if object exists and append to it
            if self.path.exists():
                with self.path.open("r") as file:
                    data = json.load(file)
            else:
                data = {}

            # Append data - assume the index corresponds to time step
            ind = len(data) + 1
            data[ind] = {
                "num_iteration": self.num_iteration,
                "nonlinear_increment_norms": self.nonlinear_increment_norms,
                "residual_norms": self.residual_norms,
            }

            # Save to file
            with self.path.open("w") as file:
                json.dump(data, file, indent=4)


@dataclass
class ExtendedSolverStatistics(SolverStatistics):
    """For tracking additional information."""

    time: float = 0.0
    """Time for which a non-linear loop is solved (previous time + current time step
    size)."""
    time_step_size: float = 0.0
    """Time step size with which the solver is applied."""

    residual_norms_per_equation: dict[str, list[float]] = field(default_factory=dict)
    """Dict of residual norms per equation, with the equation name as key."""

    increment_norms_per_variable: dict[str, list[float]] = field(default_factory=dict)
    """Dict of residual norms per nonlinear increment, with the variable name as key."""

    condition_numbers: dict[str, list[float]] = field(default_factory=dict)
    """Dict of condition numbers for the system or some subsystem, with some name as
    key."""

    def extended_log(
        self,
        time: float,
        time_step_size: float,
        res_norm_per_equation: dict[str, float],
        incr_norm_per_variable: dict[str, float],
        condition_numbers: dict[str, float],
    ) -> None:
        """Logs the additional fields."""
        self.time = time
        self.time_step_size = time_step_size

        for k, v in res_norm_per_equation.items():
            if k not in self.residual_norms_per_equation:
                self.residual_norms_per_equation[k] = []
            self.residual_norms_per_equation[k].append(v)

        for k, v in incr_norm_per_variable.items():
            if k not in self.increment_norms_per_variable:
                self.increment_norms_per_variable[k] = []
            self.increment_norms_per_variable[k].append(v)

        for k, v in condition_numbers.items():
            if k not in self.condition_numbers:
                self.condition_numbers[k] = []
            self.condition_numbers[k].append(v)

    def reset(self):
        """Clears the additional feels of this extentended dataclass."""
        super().reset()
        self.time = 0.0
        self.time_step_size = 0.0
        self.residual_norms_per_equation.clear()
        self.increment_norms_per_variable.clear()
        self.condition_numbers.clear()

    def save(self):
        """Save the new fields at the last index of the stored data."""
        super().save()
        if self.path is not None:
            # Check if object exists and append to it
            if self.path.exists():
                with self.path.open("r") as file:
                    data = json.load(file)
            else:
                data = {}

            # Append data to last entry
            ind = len(data)
            if str(ind) in data:
                ind = str(ind)

            new_data = {
                "time": self.time,
                "time_step_size": self.time_step_size,
            }
            new_data.update(self.residual_norms_per_equation)
            new_data.update(self.increment_norms_per_variable)
            new_data.update(self.condition_numbers)

            for k, v in new_data.items():
                data[ind][k] = v

            # Save to file
            with self.path.open("w") as file:
                json.dump(data, file, indent=4)
