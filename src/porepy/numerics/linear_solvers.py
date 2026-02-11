"""
Module for the Linear Solver class, which is used to solve the linear
system when using the model classes for linear systems. Note that the
model object has its own system to assemble and solve the system; this
is just a wrapper around that, mostly for compliance with the nonlinear
case, see numerics.nonlinear.nonlinear_solvers.
"""

from __future__ import annotations

from porepy.models.model_runner import ModelInstance


class LinearSolver:
    """Base solver class for PorePy models, assuming the model is linear and performing
    only 1 linear solve.

    Parameters:
        params: ``default=None``

            Solver parameters. Defaults to empty dictionary.

    """

    def __init__(self, params: dict | None = None) -> None:
        self.params = params if isinstance(params, dict) else {}
        """Parameters passed during instantiation."""

    def solve(self, model: ModelInstance) -> bool:
        """Solve a linear problem defined by the current state of the model.

        The linear solver performs only one iteration and checks whether it converged.
        Based on that, the methods ``after_solver_convergence`` or
        ``after_solver_failure`` are called on the model.

        Parameters:
            model: Model to be solved.

        Returns:
            True if the linear solver converged, False otherwise.

        """
        # For linear problems, the tolerance is irrelevant.
        # FIXME: This assumes a direct solver is applied, but it may also be that
        # parameters for linear solvers should be a property of the model, not the
        # solver. This needs clarification at some point.
        model.before_solver_iteration()
        model.assemble_linear_system()
        residual = model.equation_system.assemble(evaluate_jacobian=False)
        nonlinear_increment = model.solve_linear_system()
        model.after_solver_iteration(nonlinear_increment)
        # NOTE: The linear solver performs only one iteration.
        # FIXME: Consider renaming the solver statistics to just "solver statistics".
        model.nonlinear_solver_statistics.num_iteration = 1

        is_converged, _ = model.check_convergence(
            nonlinear_increment, residual, residual.copy(), self.params
        )

        if is_converged:
            model.after_solver_convergence()
        else:
            model.after_solver_failure()
        return is_converged
