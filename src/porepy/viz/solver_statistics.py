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
        analogous data, here for the first time step.

        >>> import matplotlib.pyplot as plt
        >>> import json
        >>> with open("solver_statistics.json", "r") as f:
        >>>     history = json.load(f)
        >>> ts = str(1)
        >>> err = history[ts]["residual_errors"]
        >>> plt.semilogy(err)
        >>> plt.xlabel("Iteration number")
        >>> plt.ylabel("Residual")
        >>> plt.title("Residual error")
        >>> plt.show()

    """

    num_iteration: int = 0
    """Number of non-linear iterations performed for current time step."""
    increment_errors: list[float] = field(default_factory=list)
    """List of increments for each non-linear iteration."""
    residual_errors: list[float] = field(default_factory=list)
    """List of residual for each non-linear iteration."""
    path: Optional[Path] = None
    """Path to save the statistics object to."""

    def log_error(self, increment_error: float, residual_error: float) -> None:
        """Log errors produced from convergence criteria.

        Parameters:
            increment_error (float): Error in the increment.
            residual_error (float): Error in the residual.

        """
        self.increment_errors.append(increment_error)
        self.residual_errors.append(residual_error)

    def reset(self) -> None:
        """Reset the statistics object."""
        self.num_iteration = 0
        self.increment_errors.clear()
        self.residual_errors.clear()

    def save(self) -> None:
        """Save the statistics object to a JSON file.

        Parameters:
            path (Path): Path to the file to save the statistics object to.

        """
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
                "increment_errors": self.increment_errors,
                "residual_errors": self.residual_errors,
            }

            # Save to file
            with self.path.open("w") as file:
                json.dump(data, file, indent=4)
