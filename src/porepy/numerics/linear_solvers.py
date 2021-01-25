"""
Module for the Linear Solver class, which is used to solve the linear
system when using the model classes for linear systems. Note that the
setup object has its own system to assemble and solve the system; this
is just a wrapper around that, mostly for compliance with the nonlinear
case, see numerics.nonlinear.nonlinear_solvers.
"""
from typing import Dict, Tuple

import porepy as pp
from porepy.models.abstract_model import AbstractModel

module_sections = ["numerics", "matrix"]


class LinearSolver:
    @pp.time_logger(sections=module_sections)
    def __init__(self, params: Dict = None) -> None:
        """Define linear solver.

        Parameters:
            params (dict): Parameters for the linear solver. Will be passed on to the
                model class. Thus the contect should be adapted to whatever needed for
                the problem at hand.

        """
        if params is None:
            params = {}
        # default_options.update(params)
        self.params = params  # default_options

    @pp.time_logger(sections=module_sections)
    def solve(self, setup: AbstractModel) -> Tuple[float, bool]:
        """Solve a linear problem defined by the current state of the model.

        Parameters:
            setup (subclass of pp.AbstractModel): Model to be solved.

        Returns:
            float: Norm of the error.
            boolean: True if the linear solver converged.

        """

        setup.before_newton_loop()
        prev_sol = setup.get_state_vector()
        # For linear problems, the tolerance is irrelevant
        sol = setup.assemble_and_solve_linear_system(tol=0)
        error_norm, is_converged, _ = setup.check_convergence(
            sol, prev_sol, prev_sol, self.params
        )

        if is_converged:
            setup.after_newton_convergence(sol, error_norm, iteration_counter=1)
        else:
            setup.after_newton_failure(sol, error_norm, iteration_counter=1)
        return error_norm, is_converged
