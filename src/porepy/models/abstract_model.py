""" Contains the mothed class for all models, that is, standardized setups for complex
problems.

"""
import abc
from typing import Any, Dict, Tuple

import numpy as np

import porepy as pp

module_sections = ["models", "numerics"]


class AbstractModel(abc.ABC):
    """This is an abstract class that specifies methods that a model must implement to
    be compatible with the Newton and time stepping methods.

    """

    @abc.abstractmethod
    @pp.time_logger(sections=module_sections)
    def get_state_vector(self) -> np.ndarray:
        """Get a vector of the current state of the variables; with the same ordering
            as in the assembler.

        Returns:
            np.array: The current state of the system.

        """
        pass

    @abc.abstractmethod
    @pp.time_logger(sections=module_sections)
    def prepare_simulation(self) -> None:
        """Method called prior to the start of time stepping, or prior to entering the
        non-linear solver for stationary problems.

        The intended use is to define parameters, geometry and grid, discretize linear
        and time-independent terms, and generally prepare for the simulation.

        """
        pass

    @abc.abstractmethod
    @pp.time_logger(sections=module_sections)
    def before_newton_loop(self) -> None:
        """Method to be called before entering the non-linear solver, thus at the start
        of a new time step.

        Possible usage is to update time-dependent parameters, discertizations etc.

        """
        pass

    @abc.abstractmethod
    @pp.time_logger(sections=module_sections)
    def before_newton_iteration(self) -> None:
        """Method to be called at the start of every non-linear iteration.

        Possible usage is to update non-linear parameters, discertizations etc.

        """
        pass

    @abc.abstractmethod
    @pp.time_logger(sections=module_sections)
    def after_newton_iteration(self, solution_vector: np.ndarray):
        """Method to be called after every non-linear iteration.

        Possible usage is to distribute information on the new trial state, visualize
        the current approximation etc.

        Parameters:
            np.array: The new solution state, as computed by the non-linear solver.

        """
        pass

    @abc.abstractmethod
    @pp.time_logger(sections=module_sections)
    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Method to be called after every non-linear iteration.

        Possible usage is to distribute information on the solution, visualization, etc.

        Parameters:
            np.array: The new solution state, as computed by the non-linear solver.

        """
        pass

    @abc.abstractmethod
    @pp.time_logger(sections=module_sections)
    def after_newton_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Method called after a non-linear solver has failed.

        The failure can be due to divergence, or that the maximum number of iterations
        has been reached.

        The actual implementation depends on the model at hand.

        """
        pass

    @abc.abstractmethod
    @pp.time_logger(sections=module_sections)
    def check_convergence(
        self,
        solution: np.ndarray,
        prev_solution: np.ndarray,
        init_solution: np.ndarray,
        nl_params: Dict[str, Any],
    ) -> Tuple[float, bool, bool]:
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
    @pp.time_logger(sections=module_sections)
    def assemble_and_solve_linear_system(self, tol: float) -> np.ndarray:
        """Assemble the linearized system, described by the current state of the model,
        solve and return the new solution vector.

        Parameters:
            tol (double): Target tolerance for the linear solver. May be used for
                inexact approaches.

        Returns:
            np.array: Solution vector.

        """
        pass

    @abc.abstractmethod
    @pp.time_logger(sections=module_sections)
    def after_simulation(self) -> None:
        """Run at the end of simulation. Can be used for cleaup etc."""

    @pp.time_logger(sections=module_sections)
    def _l2_norm_cell(self, g: pp.Grid, u: np.ndarray) -> float:
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
