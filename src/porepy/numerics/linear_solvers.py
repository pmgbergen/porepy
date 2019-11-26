"""
Module for the Linear Solver class, which is used to solve the linear
system when using the model classes for linear systems. Note that the
setup object has its own system to assemble and solve the system; this
is just a wrapper around that, mostly for compliance with the nonlinear
case, see numerics.nonlinear.nonlinear_solvers.
"""


class LinearSolver:
    def __init__(self, params=None):
        if params is None:
            params = {}
        # default_options.update(params)
        self.params = params  # default_options

    def solve(self, setup):

        setup.before_newton_loop()
        prev_sol = setup.get_state_vector()
        # For linear problems, the tolerance is irrelevant
        sol = setup.assemble_and_solve_linear_system(tol=0)
        error_norm, is_converged, _ = setup.check_convergence(
            sol, prev_sol, self.params
        )

        if is_converged:
            setup.after_newton_convergence(sol)
        else:
            setup.after_newton_failure()
        return error_norm, is_converged
