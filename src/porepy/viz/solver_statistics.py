"""Solver statistics object for non-linear solver loop."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

import porepy as pp

logger = logging.getLogger(__name__)


class SolverStatistics:
    """Statistics object for non-linear solver loop.

    This object keeps track of the number of non-linear iterations performed for the
    current time step, as well as increments and residuals for each iteration. The
    output is stored in a dictionary which is saved to a file in json format.

    We exemplify two use cases of the SolverStatistics object. The first example shows
    how to use the object inside a model, while the second demonstrates simple interaction
    as external post-processing step.

    Example:

        Within a model, we overwrite after_nonlinear_convergence() to plot the number of
        nonlinear iterations against the norm of the residual for the first time step.

        >>> def after_nonlinear_convergence(self, solution: np.ndarray) -> None:
        >>>     super().after_nonlinear_convergence(solution)
        >>>     import matplotlib.pyplot as plt
        >>>     ts = 0
        >>>     err = self.nonlinear_solver_statistics.history["residual_errors"][ts]
        >>>     plt.semilogy(err)
        >>>     plt.xlabel("Iteration number")
        >>>     plt.ylabel("Residual")
        >>>     plt.title("Residual error")
        >>>     plt.show()

    Example:

        After storing solver statistics to file, we can load the file and plot
        analogous data.

        >>> import matplotlib.pyplot as plt
        >>> import json
        >>> with open("solver_statistics.json", "r") as f:
        >>>     history = json.load(f)
        >>> ts = 0
        >>> err = history["residual_errors"][ts]
        >>> plt.semilogy(err)
        >>> plt.xlabel("Iteration number")
        >>> plt.ylabel("Residual")
        >>> plt.title("Residual error")
        >>> plt.show()

    """

    def __init__(self, model, path: Optional[Path]) -> None:

        self.nd = model.nd
        """Ambient dimension of the problem."""

        self._variable_data: dict[str, Sequence[dict[str, Any]]] = {}
        """Variable data for the model."""
        self._equations_data: dict[str, Sequence[dict[str, Any]]] = {}
        """Equation data for the model."""

        # Fetch variable and equation data for the model via the equation system.
        # The data is stored in dictionaries with keys "dense", "sparse", "subdomains_nd",
        # and "interfaces_nd", where nd is the ambient dimension of the problem.
        # "Dense" and "sparse" refer to the entire domain and the subdomains and interfaces,
        # respectively. "Subdomains_nd" and "interfaces_nd" refer to the subdomains and
        # interfaces in the model, respectively, grouped by dimension.

        self._variable_data["dense"] = pp.DiagnosticsMixin._variable_data(
            model,
            [[grid for grid in model.mdg.subdomains() + model.mdg.interfaces()]],
            add_grid_info=False,
        )
        self._variable_data["sparse"] = pp.DiagnosticsMixin._variable_data(
            model,
            [[grid] for grid in model.mdg.subdomains() + model.mdg.interfaces()],
            add_grid_info=True,
        )
        for nd in range(self.nd + 1):
            self._variable_data[f"subdomains_{nd}"] = (
                pp.DiagnosticsMixin._variable_data(
                    model,
                    [[grid for grid in model.mdg.subdomains(dim=nd)]],
                    add_grid_info=False,
                )
            )
        for nd in range(self.nd):
            self._variable_data[f"interfaces_{nd}"] = (
                pp.DiagnosticsMixin._variable_data(
                    model,
                    [[grid for grid in model.mdg.interfaces(dim=nd)]],
                    add_grid_info=False,
                )
            )

        self._equations_data["dense"] = pp.DiagnosticsMixin._equations_data(
            model,
            [[grid for grid in model.mdg.subdomains() + model.mdg.interfaces()]],
            add_grid_info=False,
        )
        self._equations_data["sparse"] = pp.DiagnosticsMixin._equations_data(
            model,
            [[grid] for grid in model.mdg.subdomains() + model.mdg.interfaces()],
            add_grid_info=True,
        )
        for nd in range(self.nd + 1):
            self._equations_data[f"subdomains_{nd}"] = (
                pp.DiagnosticsMixin._equations_data(
                    model,
                    [[grid for grid in model.mdg.subdomains(dim=nd)]],
                    add_grid_info=False,
                )
            )
        for nd in range(self.nd):
            self._equations_data[f"interfaces_{nd}"] = (
                pp.DiagnosticsMixin._equations_data(
                    model,
                    [[grid for grid in model.mdg.interfaces(dim=nd)]],
                    add_grid_info=False,
                )
            )

        self.init_complete: bool = not self._equations_data["dense"] == ()
        """Flag indicating if the initialization of the statistics object is complete."""

        self.path: Optional[Path] = path
        """Path to save the statistics object; if None, no saving is performed."""
        self.num_iteration: int = 0
        """Number of non-linear iterations performed for current time step."""
        self.increment_errors: list[float] = []
        """List of increments for each non-linear iteration."""
        self.residual_errors: list[float] = []
        """List of residual for each non-linear iteration."""
        self.sub_increment_errors: dict["str", list[float]] = {}
        """Dictionary of subincrements for each non-linear iteration."""
        self.sub_residual_errors: dict["str", list[float]] = {}
        """Dictionary of subresiduals for each non-linear iteration."""
        self.init_sub_residual_errors: dict["str", float] = {}
        """Dictionary of subinitial residuals for each time step."""
        self.history: dict = {
            "size": 0,
            "num_iteration": [],
            "increment_errors": [],
            "residual_errors": [],
            "sub_increment_errors": {},
            "sub_residual_errors": {},
            "init_sub_residual_errors": {},
        }
        """History of the statistics object over multiple nonlinear iterations."""

        # Initialize the subincrements dictionary and associated history
        for key in ["dense", "sparse"]:
            for var in self._variable_data[key]:
                self.sub_increment_errors[var["printed_name"]] = []
                self.history["sub_increment_errors"][var["printed_name"]] = []
        for nd in range(self.nd + 1):
            for var in self._variable_data[f"subdomains_{nd}"]:
                self.sub_increment_errors[var["printed_name"] + f" {nd}D"] = []
                self.history["sub_increment_errors"][
                    var["printed_name"] + f" {nd}D"
                ] = []
        for nd in range(self.nd):
            for var in self._variable_data[f"interfaces_{nd}"]:
                self.sub_increment_errors[var["printed_name"] + f" {nd}D, intf."] = []
                self.history["sub_increment_errors"][
                    var["printed_name"] + f" {nd}D, intf."
                ] = []

        # Initialize the subresiduals dictionary and associated history
        for key in ["dense", "sparse"]:
            for eq in self._equations_data[key]:
                self.sub_residual_errors[eq["printed_name"]] = []
                self.init_sub_residual_errors[eq["printed_name"]] = 0
                self.history["sub_residual_errors"][eq["printed_name"]] = []
                self.history["init_sub_residual_errors"][eq["printed_name"]] = []
        for nd in range(self.nd + 1):
            for eq in self._equations_data[f"subdomains_{nd}"]:
                self.sub_residual_errors[eq["printed_name"] + f" {nd}D"] = []
                self.init_sub_residual_errors[eq["printed_name"] + f" {nd}D"] = 0
                self.history["sub_residual_errors"][eq["printed_name"] + f" {nd}D"] = []
                self.history["init_sub_residual_errors"][
                    eq["printed_name"] + f" {nd}D"
                ] = []
        for nd in range(self.nd):
            for eq in self._equations_data[f"interfaces_{nd}"]:
                self.sub_residual_errors[eq["printed_name"] + f" {nd}D, intf."] = []
                self.init_sub_residual_errors[eq["printed_name"] + f" {nd}D, intf."] = 0
                self.history["sub_residual_errors"][
                    eq["printed_name"] + f" {nd}D, intf."
                ] = []
                self.history["init_sub_residual_errors"][
                    eq["printed_name"] + f" {nd}D, intf."
                ] = []

        self.set_info()

    def set_info(self) -> None:
        """Set the info attribute of the statistics object."""
        self.history["info"] = (
            f"""Statistics object for non-linear solver loop.
            The history attribute contains the following keys:
            - size: Counter for stacked metrics.
            - num_iteration: List of iteration counters.
            - increment_errors: List of evolutions of increments for each non-linear iteration.
            - residual_errors: List of evolutions of residuals for each non-linear iteration.
            - sub_increment_errors: Dictionary of subincrements for each non-linear iteration.
            - sub_residual_errors: Dictionary of subresiduals for each non-linear iteration.
            - init_sub_residual_errors: Dictionary of subinitial residuals for each time step.
            """
        )

    def log_error(self, increment_error: float, residual_error: float) -> None:
        """Log errors produced from convergence criteria."""
        self.increment_errors.append(increment_error)
        self.residual_errors.append(residual_error)

    def log_increment(self, increment: np.ndarray) -> None:
        """Log the increment for the current iteration."""
        for key in ["dense", "sparse"]:
            for var in self._variable_data[key]:
                self.sub_increment_errors[var["printed_name"]].append(
                    self._metric(increment[var["block_dofs"]])
                )
        for nd in range(self.nd + 1):
            for var in self._variable_data[f"subdomains_{nd}"]:
                self.sub_increment_errors[var["printed_name"] + f" {nd}D"].append(
                    self._metric(increment[var["block_dofs"]])
                )
        for nd in range(self.nd):
            for var in self._variable_data[f"interfaces_{nd}"]:
                self.sub_increment_errors[
                    var["printed_name"] + f" {nd}D, intf."
                ].append(self._metric(increment[var["block_dofs"]]))

    def log_residual(self, residual: np.ndarray, init_residual: np.ndarray) -> None:
        """Log the residual for the current iteration."""
        for key in ["dense", "sparse"]:
            for eq in self._equations_data[key]:
                self.sub_residual_errors[eq["printed_name"]].append(
                    self._metric(residual[eq["block_dofs"]])
                )
                self.init_sub_residual_errors[eq["printed_name"]] = self._metric(
                    init_residual[eq["block_dofs"]]
                )
        for nd in range(self.nd + 1):
            for eq in self._equations_data[f"subdomains_{nd}"]:
                self.sub_residual_errors[eq["printed_name"] + f" {nd}D"].append(
                    self._metric(residual[eq["block_dofs"]])
                )
                self.init_sub_residual_errors[eq["printed_name"] + f" {nd}D"] = (
                    self._metric(init_residual[eq["block_dofs"]])
                )
        for nd in range(self.nd):
            for eq in self._equations_data[f"interfaces_{nd}"]:
                self.sub_residual_errors[eq["printed_name"] + f" {nd}D, intf."].append(
                    self._metric(residual[eq["block_dofs"]])
                )
                self.init_sub_residual_errors[eq["printed_name"] + f" {nd}D, intf."] = (
                    self._metric(init_residual[eq["block_dofs"]])
                )

    def _metric(self, arr: np.ndarray) -> float:
        """Compute a metric for the array.

        NOTE: Use the same simple metric as used in the convergence check.

        """
        return np.linalg.norm(arr) / np.sqrt(arr.size)

    def log_timestep(self) -> None:
        """Stack the current values of the statistics object."""
        self.history["size"] += 1
        self.history["num_iteration"].append(self.num_iteration)
        self.history["increment_errors"].append(self.increment_errors)
        self.history["residual_errors"].append(self.residual_errors)
        for key in self.sub_increment_errors:
            self.history["sub_increment_errors"][key].append(
                self.sub_increment_errors[key]
            )
        for key in self.sub_residual_errors:
            self.history["sub_residual_errors"][key].append(
                self.sub_residual_errors[key]
            )
        for key in self.init_sub_residual_errors:
            self.history["init_sub_residual_errors"][key].append(
                self.init_sub_residual_errors[key]
            )
        self._save()

    def reset(self) -> None:
        """Reset the statistics object."""
        self.num_iteration = 0
        self.increment_errors.clear()
        self.residual_errors.clear()
        for key in self.sub_increment_errors:
            self.sub_increment_errors[key].clear()
        for key in self.sub_residual_errors:
            self.sub_residual_errors[key].clear()
        for key in self.init_sub_residual_errors:
            self.init_sub_residual_errors[key] = 0

    def _save(self) -> None:
        """Save the statistics object to a file in json format."""
        if self.path is not None:
            # Make sure the file has the correct suffix.
            if self.path.suffix != ".json":
                self.path = self.path.with_suffix(".json")
            # Make sure the folder exists.
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Dump the history to the file.
            with open(self.path, "w") as f:
                json.dump(self.history, f, indent=4)
