"""
Module for the Linear Solver class, which is used to solve the linear
system when using the model classes for linear systems. Note that the
model object has its own system to assemble and solve the system; this
is just a wrapper around that, mostly for compliance with the nonlinear
case, see numerics.nonlinear.nonlinear_solvers.
"""

from __future__ import annotations

from warnings import warn

from porepy.models.solution_strategy import SolutionStrategy
from porepy.numerics.nonlinear.convergence_check import (
    ConvergenceCriteria,
    ConvergenceInfoCollection,
    ConvergenceStatusCollection,
    DivergenceCriteria,
    IncrementBasedNanCriterion,
    ResidualBasedNanCriterion,
    SimulationStatus,
)
from porepy.viz.solver_statistics import TimeStatistics


class LinearSolver:
    """Base solver class for PorePy models, assuming the model is linear and performing
    only a single linear solve.

    Parameters:
        params: ``default=None``

            Solver parameters. Defaults to empty dictionary.

    """

    def __init__(self, params: dict | None = None) -> None:
        self.params = params if isinstance(params, dict) else {}
        """Parameters passed during instantiation."""

        # Default parameters for convergence and divergence criteria
        if "nl_convergence_criteria" not in self.params:
            self.params["nl_convergence_criteria"] = {}
        self.convergence_criteria = ConvergenceCriteria(
            self.params.get("nl_convergence_criteria")  # type: ignore[arg-type]
        )
        """Convergence criterion used in the convergence check."""

        if "nl_divergence_criteria" not in self.params:
            self.params["nl_divergence_criteria"] = {
                "inc_nan": IncrementBasedNanCriterion(),
                "res_nan": ResidualBasedNanCriterion(),
            }
        self.divergence_criteria = DivergenceCriteria(
            self.params.get("nl_divergence_criteria")  # type: ignore[arg-type]
        )
        """Divergence criterion used in the convergence check."""

    def solve(self, model: SolutionStrategy) -> SimulationStatus:
        """Solve a linear problem defined by the current state of the model.

        The linear solver performs only one iteration and checks whether it converged.
        Based on that, the methods ``after_nonlinear_convergence`` or
        ``after_nonlinear_failure`` are called on the model.

        Parameters:
            model: Model to be solved.

        Returns:
            SimulationStatus: The status of the simulation.

        """
        # Make sure the model is nonlinear.
        if not model._is_nonlinear_problem():
            raise ValueError(
                "The linear solver is only applicable to linear problems. "
                "Please use a nonlinear solver for nonlinear problems."
            )

        # Prepare model for solving.
        model.before_nonlinear_loop()

        # For linear problems, the tolerance is irrelevant.
        # FIXME: This assumes a direct solver is applied, but it may also be that
        # parameters for linear solvers should be a property of the model, not the
        # solver. This needs clarification at some point.

        # Perform a single (Newton) iteration.
        model.before_nonlinear_iteration()
        model.assemble_linear_system()
        nonlinear_increment = model.solve_linear_system()

        # Update model status (iterate) before checking convergence, so that the
        # convergence check uses the updated state. Also, after_nonlinear_convergence
        # may expect the converged solution to already be stored as an iterate.
        model.after_nonlinear_iteration(nonlinear_increment)

        # Monitor convergence.
        status, _ = self.check_convergence(model, nonlinear_increment)

        # React to convergence status.
        if status.is_converged():
            solver_status = SimulationStatus.SUCCESSFUL
            model.after_nonlinear_convergence()
        elif status.is_diverged():
            solver_status = SimulationStatus.FAILED
            model.after_nonlinear_failure()
            warn("Failed to solve the (non)linear problem.", UserWarning)
        else:
            raise ValueError(f"Unknown convergence status: {status}")

        # Update (global) solver statistics.
        self.update_solver_statistics(model, solver_status)

        return solver_status

    def check_convergence(
        self, model: SolutionStrategy, nonlinear_increment
    ) -> tuple[ConvergenceStatusCollection, ConvergenceInfoCollection]:
        """Check convergence and divergence based on passed criteria.

        Parameters:
            model: The model instance specifying the problem to be solved.
            nonlinear_increment: The current nonlinear increment.

        Returns:
            tuple[ConvergenceStatusCollection, ConvergenceInfoCollection]: Status
                and info about convergence.

        """
        # Fetch the residual.
        residual = model.equation_system.assemble(evaluate_jacobian=False)

        # Check convergence status based on current iteration.
        convergence_status, convergence_info = self.convergence_criteria.check(
            increment=nonlinear_increment,
            residual=residual,
        )

        # Check divergence status based on current iteration.
        divergence_status = self.divergence_criteria.check(
            increment=nonlinear_increment,
            residual=residual,
        )

        # Combine convergence and divergence status.
        return (
            convergence_status.union(divergence_status),
            convergence_info,
        )

    def update_solver_statistics(
        self,
        model: SolutionStrategy,
        solver_status: SimulationStatus,
    ) -> None:
        """Update the solver statistics in the model.

        Parameters:
            model: The model instance specifying the problem to be solved.
            solver_status: Simulation status of the solver.
            info: Dictionary containing norms and other information.

        """
        # Basic discretization-related information and overall simulation status.
        model.nonlinear_solver_statistics.log_simulation_status(solver_status)
        model.nonlinear_solver_statistics.log_mesh_information(model.mdg.subdomains())
        if model._is_time_dependent():
            assert isinstance(model.nonlinear_solver_statistics, TimeStatistics)
            model.nonlinear_solver_statistics.log_time_information(
                model.time_manager.time_index,
                model.time_manager.time,
                model.time_manager.dt,
                model.time_manager.final_time_reached(),
            )
