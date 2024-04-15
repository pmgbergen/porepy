"""Solver statistics object for non-linear solver loop."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

import porepy as pp

logger = logging.getLogger(__name__)


class SolverStatistics:
    """Statistics object for non-linear solver loop.

    This object keeps track of the number of non-linear iterations performed for the
    current time step, as well as increments and residuals for each iteration.

    An example usage is shown below. We have defined nonlinear_solver_statistics as an
    instance of SolverStatistics (see models.solution_strategy). We then define a
    mixin where the after_nonlinear_convergence() method is overwritten. We extract the
    number of nonlinear iterations and the norm of the residual for each iteration, and
    plot the results. Note that the index [0] refers to the first time step. More
    generally, index [i] would extract values from the i+1-th time step.

    def after_nonlinear_convergence(self, solution: np.ndarray) -> None:
        super().after_nonlinear_convergence(solution)
        itr = np.arange(0, self.nonlinear_solver_statistics.history["num_iteration"][0])
        err = self.nonlinear_solver_statistics.history["residual_errors"][0]
        plt.semilogy(itr, err)
        plt.xlabel("Iteration number")
        plt.ylabel("Residual")
        plt.title("Residual error")

    """

    def __init__(self, model, path: Optional[Path]) -> None:

        self.nd = model.nd
        """Ambient dimension of the problem."""

        self._variable_data: dict = {}
        """Variable data for the model."""
        self._equations_data: dict = {}
        """Equation data for the model."""

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
        self.increments: dict["str", list[float]] = {}
        """Dictionary of subincrements for each non-linear iteration."""
        self.residuals: dict["str", list[float]] = {}
        """Dictionary of subresiduals for each non-linear iteration."""
        self.init_residuals: dict["str", float] = {}
        """Dictionary of subinitial residuals for each time step."""
        self.history: dict = {
            "size": 0,
            "num_iteration": [],
            "increment_errors": [],
            "residual_errors": [],
            "increments": {},
            "residuals": {},
            "init_residuals": {},
        }
        """History of the statistics object over multiple nonlinear iterations."""

        # Initialize the subincrements dictionary and associated history
        for key in ["dense", "sparse"]:
            for var in self._variable_data[key]:
                self.increments[var["printed_name"]] = []
                self.history["increments"][var["printed_name"]] = []
        for nd in range(self.nd + 1):
            for var in self._variable_data[f"subdomains_{nd}"]:
                self.increments[var["printed_name"] + f" {nd}D"] = []
                self.history["increments"][var["printed_name"] + f" {nd}D"] = []
        for nd in range(self.nd):
            for var in self._variable_data[f"interfaces_{nd}"]:
                self.increments[var["printed_name"] + f" {nd}D, intf."]: list[
                    float
                ] = []
                self.history["increments"][var["printed_name"] + f" {nd}D, intf."] = []

        # Initialize the subresiduals dictionary and associated history
        for key in ["dense", "sparse"]:
            for eq in self._equations_data[key]:
                self.residuals[eq["printed_name"]]: list[float] = []
                self.init_residuals[eq["printed_name"]] = []
                self.history["residuals"][eq["printed_name"]] = []
                self.history["init_residuals"][eq["printed_name"]] = []
        for nd in range(self.nd + 1):
            for eq in self._equations_data[f"subdomains_{nd}"]:
                self.residuals[eq["printed_name"] + f" {nd}D"]: list[float] = []
                self.init_residuals[eq["printed_name"] + f" {nd}D"] = []
                self.history["residuals"][eq["printed_name"] + f" {nd}D"] = []
                self.history["init_residuals"][eq["printed_name"] + f" {nd}D"] = []
        for nd in range(self.nd):
            for eq in self._equations_data[f"interfaces_{nd}"]:
                self.residuals[eq["printed_name"] + f" {nd}D, intf."] = []
                self.init_residuals[eq["printed_name"] + f" {nd}D, intf."] = []
                self.history["residuals"][eq["printed_name"] + f" {nd}D, intf."] = []
                self.history["init_residuals"][
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
            - increments: Dictionary of subincrements for each non-linear iteration.
            - residuals: Dictionary of subresiduals for each non-linear iteration.
            - init_residuals: Dictionary of subinitial residuals for each time step.
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
                self.increments[var["printed_name"]].append(
                    self._metric(increment[var["block_dofs"]])
                )
        for nd in range(self.nd + 1):
            for var in self._variable_data[f"subdomains_{nd}"]:
                self.increments[var["printed_name"] + f" {nd}D"].append(
                    self._metric(increment[var["block_dofs"]])
                )
        for nd in range(self.nd):
            for var in self._variable_data[f"interfaces_{nd}"]:
                self.increments[var["printed_name"] + f" {nd}D, intf."].append(
                    self._metric(increment[var["block_dofs"]])
                )

    def log_residual(self, residual: np.ndarray, init_residual: np.ndarray) -> None:
        """Log the residual for the current iteration."""
        for key in ["dense", "sparse"]:
            for eq in self._equations_data[key]:
                self.residuals[eq["printed_name"]].append(
                    self._metric(residual[eq["block_dofs"]])
                )
                self.init_residuals[eq["printed_name"]] = self._metric(
                    init_residual[eq["block_dofs"]]
                )
        for nd in range(self.nd + 1):
            for eq in self._equations_data[f"subdomains_{nd}"]:
                self.residuals[eq["printed_name"] + f" {nd}D"].append(
                    self._metric(residual[eq["block_dofs"]])
                )
                self.init_residuals[eq["printed_name"] + f" {nd}D"] = self._metric(
                    init_residual[eq["block_dofs"]]
                )
        for nd in range(self.nd):
            for eq in self._equations_data[f"interfaces_{nd}"]:
                self.residuals[eq["printed_name"] + f" {nd}D, intf."].append(
                    self._metric(residual[eq["block_dofs"]])
                )
                self.init_residuals[eq["printed_name"] + f" {nd}D, intf."] = (
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
        for key in self.increments:
            self.history["increments"][key].append(self.increments[key])
        for key in self.residuals:
            self.history["residuals"][key].append(self.residuals[key])
        for key in self.init_residuals:
            self.history["init_residuals"][key].append(self.init_residuals[key])
        self._save()

    def reset(self) -> None:
        """Reset the statistics object."""
        self.num_iteration = 0
        self.increment_errors.clear()
        self.residual_errors.clear()
        for key in self.increments:
            self.increments[key].clear()
        for key in self.residuals:
            self.residuals[key].clear()
        for key in self.init_residuals:
            self.init_residuals[key] = 0

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
