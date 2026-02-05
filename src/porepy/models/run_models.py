"""This module contains functions to run stationary and time-dependent models."""

from __future__ import annotations

import logging
from typing import Optional, TypeVar, no_type_check

import numpy as np

import porepy as pp
from porepy.utils.ui_and_logging import DummyProgressBar, progressbar_class
from porepy.utils.ui_and_logging import (
    logging_redirect_tqdm_with_level as logging_redirect_tqdm,
)

# Module-wide logger
logger = logging.getLogger(__name__)

ModelType = TypeVar("ModelType", bound=pp.PorePyModel)
"""Type variable for objects inheriting from the PorePy model protocol."""


class ModelRunner:
    def __init__(self, model: ModelType, params: Optional[dict] = None) -> None:
        """_summary_



        Parameters:
            model: Model class containing all information on parameters, variables,
                discretization, geometry. Various methods such as those relating to
                solving the system, see the appropriate model for documentation.
            params: Parameters related to the solution procedure. Defaults to None.

        """
        # PvS: Is this clean code?
        self.params = params or {}
        self.model = model

        # Select a solver for the problem.
        self.solver = _choose_solver(self.model, self.params)


class StationaryModelRunner(ModelRunner):
    def run(self) -> None:
        """Run a stationary model."""
        self.model.prepare_simulation()
        self.solver.solve(self.model)
        self.model.after_simulation()


class TimeDependentModelRunner(ModelRunner):
    def __init__(self, model: ModelType, params: dict | None = None) -> None:
        super().__init__(model, params)
        self.initial_time_step: float = model.time_manager.dt
        self.init_time_progressbar()

        # To avoid checking the long name.
        self.dt_isconstant: bool = model.time_manager.is_constant

        # PvS: Shift to the base runner? Have this in a ``prepare_simulation`` method?
        if self.params.get("prepare_simulation", True):
            self.model.prepare_simulation()

    def init_time_progressbar(self) -> None:
        """

        Note:
            If the ``"progressbars"`` key in ``params`` is set to ``True`` (default is
            ``False``), the progress of nonlinear iterations will be shown on a
            progressbar. This requires the ``tqdm`` package to be installed. The package
            is not included in the dependencies, but can be installed with
            ```
            pip install tqdm
            ```



        """
        use_progress_bar = self.params.get("progressbars", False)
        if use_progress_bar and progressbar_class is DummyProgressBar:
            logger.warning(
                "Progress bars are requested, but `tqdm` is not installed. The time"
                " loop will run without progress bars."
            )
        # To display nested ``tqdm`` bars in the correct order, their positions have to
        # be specified. The orders are increasing, i.e., 0 is the lowest level, then 1,
        # etc.
        # When ``NewtonSolver`` is called inside ``run``, the
        # ``_nl_progress_bar_position`` parameter specifying the position of the
        # ``NewtonSolver`` progressbar is passed.
        self.params.update({"_nl_progress_bar_position": 1})

        # Check if the user wants a progress bar. Initialize an instance of the
        # progressbar_class, which is either :class:`~tqdm.trange` or
        # :class:`~DummyProgressbar` in case `tqdm` is not installed.
        if self.params.get("progressbars", False):
            # Create a time bar. The length is estimated as the time_steps predetermined
            # by the schedule and initial time step size.
            # NOTE: If, e.g., adaptive time stepping results in more time steps, the
            # time bar will increase with partial steps corresponding to the ratio of
            # the modified time step size to the initial time step size.
            expected_time_steps: int = int(
                np.round(
                    (
                        self.model.time_manager.schedule[-1]
                        - self.model.time_manager.schedule[0]
                    )
                    / self.initial_time_step
                )
            )
            self.time_progressbar = progressbar_class(
                range(expected_time_steps),
                desc="time loop",
                position=0,
                dynamic_ncols=True,
            )
        # Otherwise, use a dummy progress bar.
        else:
            self.time_progressbar = DummyProgressBar()

    def before_time_step(self) -> None:
        # Increase the simulation time.
        self.model.time_manager.increase_time()
        self.model.time_manager.increase_time_index()
        # Update the model's AD time step object.
        self.model.ad_time_step.set_value(self.model.time_manager.dt)

        # Logging and progressbar update.
        logger.info(
            f"\nTime step {self.model.time_manager.time_index} at time"
            + f" {self.model.time_manager.time:.1e}"
            + f" of {self.model.time_manager.time_final:.1e}"
            + f" with time step {self.model.time_manager.dt:.1e}"
        )
        self.time_progressbar.set_description_str(
            f"Time step {self.model.time_manager.time_index + 1}"  # Why plus 1? Be consistent!
        )

    # Define a function that does all the work during one time step, except
    # for everything ``tqdm`` related.
    def time_step(self) -> bool:
        """Does all the work during one time step.

        Returns:
            _description_

        """

        # Return convergence status s.t. the time loop can determine whether the time
        # step succeeded or failed.
        return self.solver.solve(self.model)

    def after_time_step(self) -> None:
        if self.ts_converged:
            # Update the time step magnitude if the dynamic scheme is used.
            if not self.dt_isconstant:
                self.model.time_manager.compute_time_step(
                    iterations=self.model.nonlinear_solver_statistics.num_iteration
                )

            # Update progressbar length.
            self.time_progressbar.update(
                n=self.model.time_manager.dt / self.initial_time_step
            )

    def run(self, model: ModelType, params: Optional[dict] = None) -> None:
        """Run a time dependent model.

        Note:
            If the ``"progressbars"`` key in ``params`` is set to ``True`` (default is
            ``False``), the progress of time steps and nonlinear iterations will be
            shown on a progressbar. This requires the ``tqdm`` package to be installed.
            The package is not included in the dependencies, but can be installed with
<<<<<<< HEAD
            using ``pip``.
=======
            ```
            pip install tqdm
            ```
>>>>>>> 07c2427a6ccd91c614c224c960ca71b9fd26e61f

        Parameters:
            model: Model class containing all information on parameters, variables,
                discretization, geometry. Various methods such as those relating to
                solving the system, see the appropriate solver for documentation.
            params: Parameters related to the solution procedure.

        """
        # Redirect the root logger, to avoid logger-progressbars interference.
        with logging_redirect_tqdm([logging.root]):
            # Time loop.
            while not self.model.time_manager.final_time_reached():
                self.before_time_step()
                self.ts_converged: bool = self.time_step()
                self.after_time_step()

        self.model.after_simulation()


def _choose_solver(model: ModelType, params: dict) -> pp.LinearSolver | pp.NewtonSolver:
    """Choose between linear and non-linear solver.

    Parameters:
        model: Model class containing all information on material parameters, variables,
            discretization and geometry. Various methods such as those relating to
            solving the system, see the appropriate solver for documentation.
        params: Parameters related to the solution procedure.

    """
    if "nonlinear_solver" in params:
        solver = params["nonlinear_solver"](params)
    elif model.is_nonlinear_problem():
        solver = pp.NewtonSolver(params)
    else:
        solver = pp.LinearSolver(params)
    return solver
