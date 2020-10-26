"""


"""
import abc

import numpy as np


class AbstractModel(abc.ABC):
    """This is an abstract class that specifies methods that a model must implement to
    be compatible with the Newton and time stepping methods.

    """

    @abc.abstractmethod
    def get_state_vector(self):
        """Get a vector of the current state of the variables; with the same ordering
            as in the assembler.

        Returns:
            np.array: The current state of the system.

        """
        pass

    @abc.abstractmethod
    def prepare_simulation(self):
        """Method called prior to the start of time stepping, or prior to entering the
        non-linear solver for stationary problems.

        The intended use is to define parameters, geometry and grid, discretize linear
        and time-independent terms, and generally prepare for the simulation.

        """
        pass

    @abc.abstractmethod
    def before_newton_loop(self):
        """Method to be called before entering the non-linear solver, thus at the start
        of a new time step.

        Possible usage is to update time-dependent parameters, discertizations etc.

        """
        pass

    @abc.abstractmethod
    def before_newton_iteration(self):
        """Method to be called at the start of every non-linear iteration.

        Possible usage is to update non-linear parameters, discertizations etc.

        """
        pass

    @abc.abstractmethod
    def after_newton_iteration(self, solution_vector):
        """Method to be called after every non-linear iteration.

        Possible usage is to distribute information on the new trial state, visualize
        the current approximation etc.

        Parameters:
            np.array: The new solution state, as computed by the non-linear solver.

        """
        pass

    @abc.abstractmethod
    def after_newton_convergence(self, solution, errors, iteration_counter):
        """Method to be called after every non-linear iteration.

        Possible usage is to distribute information on the solution, visualization, etc.

        Parameters:
            np.array: The new solution state, as computed by the non-linear solver.

        """
        pass

    @abc.abstractmethod
    def after_newton_failure(self, solution, errors, iteration_counter):
        """Method called after a non-linear solver has failed.

        The failure can be due to divergence, or that the maximum number of iterations
        has been reached.

        The actual implementation depends on the model at hand.

        """
        pass

    @abc.abstractmethod
    def check_convergence(self, solution, prev_solution, init_solution, nl_params):
        """Implements a convergence check, to be called by a non-linear solver.

        Parameters:
            solution (np.array): Newly obtained solution vector
            prev_solution (np.array): Solution obtained in the previous non-linear
                iteration.
            init_solution (np.array): Solution obtained from the previous time-step.
            nl_params (dict): Dictionary of parameters used for the convergence check.
                Which items are required will depend on the converegence test to be
                implemented.

        Returns:
            double: Error, computed to the norm in question.
            boolean: True if the solution is converged according to the test
                implemented by this method.
            boolean: True if the solution is diverged according to the test
                implemented by this method.

        """
        pass

    @abc.abstractmethod
    def assemble_and_solve_linear_system(self, tol):
        """Assemble the linearized system, described by the current state of the model,
        solve and return the new solution vector.

        Parameters:
            tol (double): Target tolerance for the linear solver. May be used for
                inexact approaches.

        Returns:
            np.array: Solution vector.

        """
        pass

    def l2_norm_cell(self, g, u):
        """
        Compute the cell volume weighted norm of a vector-valued cellwise quantity for
        a given grid.

        Parameters:
            g (pp.Grid): Grid
            u (np.array): Vector-valued function.

        Returns:
            double: The computed L2-norm.

        """
        nc = g.num_cells
        sz = u.size
        if nc == sz:
            nd = 1
        elif nc * g.dim == sz:
            nd = g.dim
        else:
            raise ValueError("Have not conisdered this type of unknown vector")

        norm = np.sqrt(np.reshape(u ** 2, (nd, nc), order="F") * g.cell_volumes)

        return np.sum(norm)
