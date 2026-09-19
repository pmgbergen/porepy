"""This module implements a sequential iterative nonlinear solver. See
sequential_nonlinear_solver_poromechanics.py for a usage example.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from numbers import Real
from typing import Optional
from warnings import warn

import numpy as np

import porepy as pp
from porepy.numerics.solvers.convergence_check import (
    ConvergenceCriteria,
    ConvergenceInfo,
    ConvergenceInfoCollection,
    ConvergenceStatus,
    ConvergenceStatusCollection,
    DivergenceCriteria,
    assemble_default_convergence_criteria,
    assemble_default_divergence_criteria,
    check_convergence,
)
from porepy.numerics.solvers.nonlinear_solvers import (
    NonlinearSolverBase,
    NonlinearSolverStatus,
    NonlinearSolverStatusConverged,
    NonlinearSolverStatusFailed,
)

__all__ = [
    "SequentialSolverConverged",
    "SequentialSolverFailed",
    "SequentialNonlinearSolver",
]

logger = logging.getLogger(__name__)


@dataclass
class SequentialSolverConverged(NonlinearSolverStatusConverged):
    subsolver_statuses: list[list[NonlinearSolverStatus]]
    """A list of lists of inner solver statuses. Outer list index denotes the solver
    iteration. Inner list index denotes the order of the inner solver within the
    iteration. E.g., if the sequential solver made 5 iterations, and we have two inner
    solvers, `subsolver_statuses[4][0]` accesses the 0-th inner solver data on the last
    outer iteration.

    """

    def number_of_iterations(self) -> int:
        return len(self.subsolver_statuses)


@dataclass
class SequentialSolverFailed(NonlinearSolverStatusFailed):
    subsolver_statuses: list[list[NonlinearSolverStatus]]
    """A list of lists of inner solver statuses. Outer list index denotes the solver
    iteration. Inner list index denotes the order of the inner solver within the
    iteration. E.g., if the sequential solver made 5 iterations, and we have two inner
    solvers, `subsolver_statuses[4][0]` accesses the 0-th inner solver data on the last
    outer iteration.

    """

    def number_of_iterations(self) -> int:
        return len(self.subsolver_statuses)


class SequentialNonlinearSolver(NonlinearSolverBase):
    """A sequential iterative nonlinear solver. Accepts `subsolvers`, each solves a
    restricted nonlinear problem (only some equations and variables). Iterates between
    them while the full problem converges.

    It is not generally guaranteed that the sequential iterative scheme is convergent.

    Parameters:
        subsolvers: List of subsolvers.
        max_iterations: Maximum number of iterations. A single iteration calls each
            subsolver once.
        convergence_criteria: For the full problem. If None (default), default criteria
            are initialized.
        divergence_criteria: For the full problem. If None (default), default criteria
            are initialized.

    """

    def __init__(
        self,
        subsolvers: list[NonlinearSolverBase],
        max_iterations: int = 10,
        convergence_criteria: Optional[ConvergenceCriteria] = None,
        divergence_criteria: Optional[DivergenceCriteria] = None,
    ) -> None:
        self.subsolvers = subsolvers
        """List of subsolvers."""
        self.max_iterations = max_iterations
        """Maximum number of iterations. A single iteration calls each subsolver once.
        
        """
        if convergence_criteria is None:
            convergence_criteria = assemble_default_convergence_criteria(
                is_nonlinear_problem=True,
                inc_atol=float("inf"),
                inc_rtol=1e-6,
                res_atol=float("inf"),
                res_rtol=1e-8,
                metric=pp.EuclideanMetric(),
            )
        if divergence_criteria is None:
            divergence_criteria = assemble_default_divergence_criteria(
                is_nonlinear_problem=True,
                max_iterations=max_iterations,
                inc_div_atol=1e10,
                res_div_atol=1e10,
                metric=pp.EuclideanMetric(),
            )
        self.convergence_criteria = convergence_criteria
        """For the full problem."""
        self.divergence_criteria = divergence_criteria
        """For the full problem."""

    def get_active_equations(
        self, model: pp.PorePyModel
    ) -> list[pp.ad.EquationOnDomain]:
        """Collects active equations of each subsolver.

        Overlapping equations are in principle permitted, but the user must know what
        they are doing. Since it is possible to do it by mistake, a warning is logged.

        Incomplete equations are also permited, e.g., when using nested sequential
        nonlinear solvers. But the user must know what they are doing, thus a warning is
        logged.

        Returns: Union of each subsolvers' active equations without duplicates.

        """
        all_equations = set(model.equation_system.equation_indexer.indices)
        return _get_active_operators(
            active_operators_per_subsolver=[
                subsolver.get_active_equations(model) for subsolver in self.subsolvers
            ],
            all_operators=all_equations,
        )

    def get_active_variables(self, model: pp.PorePyModel) -> list[pp.ad.Variable]:
        """Collects active variables of each subsolver.

        Overlapping variables are in principle permitted, but the user must know what
        they are doing. Since it is possible to do it by mistake, a warning is logged.

        Incomplete variables are also permited, e.g., when using nested sequential
        nonlinear solvers. But the user must know what they are doing, thus a warning is
        logged.

        Returns: Union of each subsolvers' active variables without duplicates.

        """
        return _get_active_operators(
            active_operators_per_subsolver=[
                subsolver.get_active_variables(model) for subsolver in self.subsolvers
            ],
            all_operators=set(model.equation_system.variables),
        )

    def solve(self, model: pp.PorePyModel) -> NonlinearSolverStatus:
        """Solve the nonlinear system with the sequential iterative scheme."""
        # Collection of results data.
        subsolver_statuses: list[list[NonlinearSolverStatus]] = []
        # Reset the criteria.
        self.convergence_criteria.reset()
        self.divergence_criteria.reset()

        for iteration_index in range(self.max_iterations):
            # Data storage for this iteration
            iteration_statuses: list[NonlinearSolverStatus] = []
            subsolver_statuses.append(iteration_statuses)

            # Save the solution array before this iteration for convergence criteria.
            old_iterate = model.equation_system.get_variable_values(
                variables=self.get_active_variables(model), iterate_index=0
            )

            # Apply subsolvers.
            for i, subsolver in enumerate(self.subsolvers):
                subsolver_status = subsolver.solve(model)
                iteration_statuses.append(subsolver_status)
                if subsolver_status.is_failed():
                    logger.warning(f"Subsolver {i} (counting from 0) failed.")
                    # Check if the inner failure led to nans and there is no point to
                    # continue.
                    full_residual = model.equation_system.assemble(
                        evaluate_jacobian=False,
                        equations=self.get_active_equations(model),
                    )
                    if np.all(np.isfinite(full_residual)):
                        # We can still recover.
                        continue

                    return SequentialSolverFailed(
                        convergence_statuses=pp.solvers.ConvergenceStatusCollection(),
                        divergence_statuses=pp.solvers.ConvergenceStatusCollection(),
                        subsolver_statuses=subsolver_statuses,
                    )

            # Fetch solution after this iteration for convergence criteria.
            new_iterate = model.equation_system.get_variable_values(
                variables=self.get_active_variables(model), iterate_index=0
            )
            # Assemble full system residual for convergence criteria.
            full_residual = model.equation_system.assemble(
                evaluate_jacobian=False, equations=self.get_active_equations(model)
            )
            convergence_status, divergence_status, convergence_info = check_convergence(
                convergence_criteria=self.convergence_criteria,
                divergence_criteria=self.divergence_criteria,
                nonlinear_increment=new_iterate - old_iterate,
                solution=new_iterate,
                residual=full_residual,
                iteration_index=iteration_index + 1,
            )
            # Iteration index is off by one, because MaxIterationsCriterion explicitly
            # starts count from 1, see its docstring.

            _log_convergence_info(
                inner_solver_statuses=iteration_statuses,
                iteration_index=iteration_index,
                max_iterations=self.max_iterations,
                convergence_info=convergence_info,
            )

            # Handle success or failure.
            if convergence_status.is_converged():
                return SequentialSolverConverged(
                    convergence_statuses=convergence_status,
                    divergence_statuses=divergence_status,
                    subsolver_statuses=subsolver_statuses,
                )
            elif divergence_status.is_failed():
                return SequentialSolverFailed(
                    convergence_statuses=convergence_status,
                    divergence_statuses=divergence_status,
                    subsolver_statuses=subsolver_statuses,
                )

        # This must be unreachable due to MaxIterationDivergenceCriterion. Keeping this
        # safeguard for consistency.
        return SequentialSolverFailed(
            convergence_statuses=ConvergenceStatusCollection(),
            divergence_statuses=ConvergenceStatusCollection(
                max_iter=ConvergenceStatus.FAILED,
            ),
            subsolver_statuses=subsolver_statuses,
        )


def _log_convergence_info(
    inner_solver_statuses: list[NonlinearSolverStatus],
    iteration_index: int,
    max_iterations: int,
    convergence_info: ConvergenceInfoCollection,
):
    """Format the status string and log it."""
    num_iterations = [x.number_of_iterations() for x in inner_solver_statuses]
    log_string = (
        f"Iter {iteration_index}/{max_iterations}. Inner #iters: {num_iterations}."
    )
    inc_abs = convergence_info.get("inc_abs", None)
    inc_rel = convergence_info.get("inc_rel", None)
    res_abs = convergence_info.get("res_abs", None)
    res_rel = convergence_info.get("res_rel", None)
    if isinstance(inc_abs, Real):
        log_string = f"{log_string} {inc_abs=:.1e}"
    if isinstance(inc_rel, Real):
        log_string = f"{log_string} {inc_rel=:.1e}"
    if isinstance(res_abs, Real):
        log_string = f"{log_string} {res_abs=:.1e}"
    if isinstance(res_rel, Real):
        log_string = f"{log_string} {res_rel=:.1e}"

    logger.info(log_string)


def _get_active_operators[T: (pp.ad.EquationOnDomain, pp.ad.Variable)](
    active_operators_per_subsolver: list[list[T]],
    all_operators: set[T],
) -> list[T]:
    """Collects active equations / variables of each subsolver and checks them for
    duplicates and overlapping.

    Overlapping equations / variables are in principle permitted, but the user must know
    what they are doing. Since it is possible to do it by mistake, a warning is logged.

    Incomplete equations / variables are also permited, e.g., when using nested
    sequential nonlinear solvers. But the user must know what they are doing, thus a
    warning is logged.

    Parameters:
        active_operators_per_subsolver: Active equations / variables for each subsolver.
        all_operators: All equations / variables known to the model.

    Returns:
        Union of subsolvers' active equations / variables without duplicates.

    """
    # Count with duplicates.
    num_active_operators = sum(len(x) for x in active_operators_per_subsolver)
    # Remove duplicates, preserve order.
    flat_active_operators = list(
        dict.fromkeys(y for x in active_operators_per_subsolver for y in x)
    )
    # Overlapping is permitted, but the used must know what they are doing.
    if len(flat_active_operators) != num_active_operators:
        warn(
            "Equations/variables in subsolvers are overlapping. Ensure it is intended.",
            stacklevel=3,
        )

    # One can use sequential solver as a part of a more complex nonlinear solver.
    # But again, only if they know what they are doing.
    if len(flat_active_operators) < len(all_operators):
        absent_equations = all_operators.difference(flat_active_operators)
        warn(
            "Sequential solver is solving an incomplete set of equations: "
            f"{absent_equations}. Ensure it is intended."
        )
    return flat_active_operators
