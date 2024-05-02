"""
Module for the Linear Solver class, which is used to solve the linear
system when using the model classes for linear systems. Note that the
setup object has its own system to assemble and solve the system; this
is just a wrapper around that, mostly for compliance with the nonlinear
case, see numerics.nonlinear.nonlinear_solvers.
"""

from __future__ import annotations

from typing import Optional

from porepy.models.solution_strategy import SolutionStrategy


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

    def solve(self, setup: SolutionStrategy) -> bool:
        """Solve a linear problem defined by the current state of the model.

        Parameters:
            setup (subclass of pp.SolutionStrategy): Model to be solved.

        Returns:
            boolean: True if the linear solver converged.

        """

        setup.before_nonlinear_loop()

        # For linear problems, the tolerance is irrelevant
        # FIXME: This assumes a direct solver is applied, but it may also be that parameters
        # for linear solvers should be a property of the model, not the solver. This
        # needs clarification at some point.

        setup.assemble_linear_system()
        residual = setup.equation_system.assemble(evaluate_jacobian=False)
        nonlinear_increment = setup.solve_linear_system()

        _, _, is_converged, _ = setup.check_convergence(
            nonlinear_increment, residual, residual.copy(), self.params
        )

        if is_converged:
            # IMPLEMENTATION NOTE: The following is a bit awkward, and really shows there is
            # something wrong with how the linear and non-linear solvers interact with the
            # models (and it illustrates that the model convention for the before_nonlinear_*
            # and after_nonlinear_* methods is not ideal).
            # Since the setup's after_nonlinear_convergence may expect that the converged
            # solution is already stored as an iterate (this may happen if a model is
            # implemented to be valid for both linear and non-linear problems, as is
            # the case for ContactMechanics and possibly others). Thus, we first call
            # after_nonlinear_iteration(), and then after_nonlinear_convergence()
            setup.after_nonlinear_iteration(nonlinear_increment)
            setup.after_nonlinear_convergence()
        else:
            setup.after_nonlinear_failure()
        return is_converged
