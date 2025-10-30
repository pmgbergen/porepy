"""
Module for the Linear Solver class, which is used to solve the linear
system when using the model classes for linear systems. Note that the
model object has its own system to assemble and solve the system; this
is just a wrapper around that, mostly for compliance with the nonlinear
case, see numerics.nonlinear.nonlinear_solvers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from porepy.models.solution_strategy import SolutionStrategy
from porepy.numerics.nonlinear.convergence_check import (
    ConvergenceInfo,
    ConvergenceStatus,
)
from porepy.viz.solver_statistics import TimeStatistics


class LinearSolver:
    """Wrapper around models that solves linear problems, and calls the methods in the
    model class before and after solving the problem.
    """

    def __init__(self, params: Optional[dict] = None) -> None:
        """Define linear solver.

        Parameters:
            params (dict): Parameters for the linear solver. Will be passed on to the
                model class. Thus the contect should be adapted to whatever needed for
                the problem at hand.

        """
        if params is None:
            params = {}
        self.params = params

    def solve(self, model: SolutionStrategy) -> ConvergenceStatus:
        """Solve a linear problem defined by the current state of the model.

        Parameters:
            model: Model to be solved.

        Returns:
            ConvergenceStatus: The status of the convergence.

        """
        # Prepare model for solving.
        model.before_nonlinear_loop()

        # For linear problems, the tolerance is irrelevant.
        # FIXME: This assumes a direct solver is applied, but it may also be that
        # parameters for linear solvers should be a property of the model, not the
        # solver. This needs clarification at some point.

        # Perform a single (Newton) iteration.
        model.assemble_linear_system()
        nonlinear_increment = model.solve_linear_system()

        # Monitor convergence.
        status, info = self.check_convergence(model, nonlinear_increment)

        # IMPLEMENTATION NOTE: The following is a bit awkward, and really shows
        # there is something wrong with how the linear and non-linear solvers
        # interact with the models (and it illustrates that the model convention for
        # the before_nonlinear_* and after_nonlinear_* methods is not ideal). Since
        # the model's after_nonlinear_convergence may expect that the converged
        # solution is already stored as an iterate (this may happen if a model is
        # implemented to be valid for both linear and non-linear problems, as is the
        # case for ContactMechanics and possibly others). Thus, we first call
        # after_nonlinear_iteration(), and then after_nonlinear_convergence()

        # Update model status.
        model.after_nonlinear_iteration(nonlinear_increment)

        # Update solver statistics.
        self.update_solver_statistics(model, status, info)

        # React to convergence status.
        if status.is_converged():
            model.after_nonlinear_convergence()
        elif status.is_failed():
            status = model.after_nonlinear_failure(status)
        else:
            raise ValueError(f"Unknown convergence status: {status}")
        return status

    def check_convergence(
        self, model: SolutionStrategy, nonlinear_increment
    ) -> tuple[ConvergenceStatus, ConvergenceInfo]:
        """Simple convergence check for linear problems checking NaN values."""
        residual = model.equation_system.assemble(evaluate_jacobian=False)
        if np.isnan(nonlinear_increment).any() or np.isnan(residual).any():
            return ConvergenceStatus.NAN, ConvergenceInfo(np.nan, np.nan)
        else:
            return ConvergenceStatus.CONVERGED, ConvergenceInfo(0.0, 0.0)

    def update_solver_statistics(
        self,
        model: SolutionStrategy,
        status: ConvergenceStatus,
        info: ConvergenceInfo,
    ) -> None:
        """Update the solver statistics in the model.

        Parameters:
            model: The model instance specifying the problem to be solved.
            status (ConvergenceStatus): Convergence status of the solver.
            info (dict): Dictionary containing norms and other information.

        """
        # Convergence-related information.
        model.nonlinear_solver_statistics.log_convergence_status(status)

        # Basic discretization-related information.
        model.nonlinear_solver_statistics.log_mesh_information(model.mdg.subdomains())
        if model._is_time_dependent():
            assert isinstance(model.nonlinear_solver_statistics, TimeStatistics)
            model.nonlinear_solver_statistics.log_time_information(
                model.time_manager.time_index,
                model.time_manager.time,
                model.time_manager.dt,
                model.time_manager.final_time_reached(),
            )
