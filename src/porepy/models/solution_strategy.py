"""Solution strategy classes.

This class is a modified version of relevant parts of AbstractModel.
In the future, it may be possible to merge the two classes. For now, we
keep them separate, to avoid breaking existing code (legacy models).
"""
from __future__ import annotations

import abc
import logging
import time
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sps

import porepy as pp

logger = logging.getLogger(__name__)


class SolutionStrategy(abc.ABC):
    """This is a class that specifies methods that a model must implement to
    be compatible with the linearization and time stepping methods.

    """

    def __init__(self, params: Optional[Dict] = None):
        """Initialize the solution strategy.

        Args:
            params (dict, optional): Parameters for the solution strategy. Defaults to
                None.

        """
        if params is None:
            params = {}
        default_params = {
            "folder_name": "visualization",
            "file_name": "data",
            "linear_solver": "pypardiso",
        }
        default_params.update(params)
        self.params = default_params
        """Dictionary of parameters."""

        # Set a convergence status. Not sure if a boolean is sufficient, or whether
        # we should have an enum here.
        self.convergence_status: bool = False
        """Whether the non-linear iteration has converged."""

        self._nonlinear_iteration: int = 0
        """Number of non-linear iterations performed for the current time step."""

        # Define attributes to be assigned later
        self.equation_system: pp.ad.EquationSystem
        """Equation system manager."""
        self.mdg: pp.MixedDimensionalGrid
        """Mixed-dimensional grid."""
        self.box: dict
        """Bounding box of the domain."""
        self.linear_system: tuple[sps.spmatrix, np.ndarray]
        """The linear system to be solved in each iteration of the non-linear solver.
        The tuple contains the sparse matrix and the right hand side residual vector."""
        self.exporter: pp.Exporter
        """Exporter for visualization."""

        self.units = params.get("units", pp.Units())
        """Units of the model."""

        self.time_manager = params.get(
            "time_manager",
            pp.TimeManager(schedule=[0, 1], dt_init=1, constant_dt=True),
        )
        """Time manager for the simulation."""

    def prepare_simulation(self) -> None:
        """Run at the start of simulation. Used for initialization etc."""
        # Set the geometry of the problem. This is a method that must be implemented
        # in a ModelGeometry class.
        self.set_geometry()
        # Exporter initialization must be done after grid creation.
        self.initialize_data_saving()

        # Set variables, constitutive relations, discretizations and equations.
        # Order of operations is important here.
        self.set_equation_system_manager()
        self.create_variables()
        self.initial_condition()
        self.set_materials()
        self.set_discretization_parameters()
        self.set_equations()

        # Export initial condition
        self.save_data_time_step()

        self.discretize()
        self._initialize_linear_solver()

    def set_equation_system_manager(self) -> None:
        """Create an equation_system manager on the mixed-dimensional grid."""
        self.equation_system = pp.EquationSystem(self.mdg)

    def initial_condition(self) -> None:
        """Set the initial condition for the problem."""
        vals = np.zeros(self.equation_system.num_dofs())
        self.equation_system.set_variable_values(vals, to_iterate=True, to_state=True)

    def before_newton_loop(self) -> None:
        """Wrap for legacy reasons. To be removed."""
        self.before_nonlinear_loop()  # or before_linearization?

    def before_nonlinear_loop(self) -> None:
        """Method to be called before entering the non-linear solver, thus at the start
        of a new time step.

        Possible usage is to update time-dependent parameters, discretizations etc.

        """
        self._nonlinear_iteration = 0

    def before_newton_iteration(self) -> None:
        self.before_nonlinear_iteration()

    def before_nonlinear_iteration(self) -> None:
        """Method to be called at the start of every non-linear iteration.

        Possible usage is to update non-linear parameters, discretizations etc.

        """
        pass

    def after_newton_iteration(self, solution_vector: np.ndarray):
        """Wrap for legacy reasons."""
        self.after_nonlinear_iteration(solution_vector)

    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        """Method to be called after every non-linear iteration.

        Possible usage is to distribute information on the new trial state, visualize
        the current approximation etc.

        Parameters:
            np.array: The new solution state, as computed by the non-linear solver.

        """
        self._nonlinear_iteration += 1
        self.equation_system.set_variable_values(
            values=solution_vector, additive=True, to_iterate=True
        )

    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        self.after_nonlinear_convergence(solution, errors, iteration_counter)

    def after_nonlinear_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Method to be called after every non-linear iteration.

        Possible usage is to distribute information on the solution, visualization, etc.

        Parameters:
            np.array: The new solution state, as computed by the non-linear solver.

        """
        solution = self.equation_system.get_variable_values(from_iterate=True)
        self.equation_system.set_variable_values(
            values=solution, to_state=True, additive=False
        )
        self.convergence_status = True
        self.save_data_time_step()

    def after_newton_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        self.after_nonlinear_failure(solution, errors, iteration_counter)

    def after_nonlinear_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        if self._is_nonlinear_problem():
            raise ValueError("Newton iterations did not converge")
        else:
            raise ValueError("Tried solving singular matrix for the linear problem.")

    def after_simulation(self) -> None:
        """Run at the end of simulation. Can be used for cleanup etc."""
        self.finalize_data_saving()

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
                Which items are required will depend on the convergence test to be
                implemented.

        Returns:
            float: Error, computed to the norm in question.
            boolean: True if the solution is converged according to the test
                implemented by this method.
            boolean: True if the solution is diverged according to the test
                implemented by this method.

        Raises: NotImplementedError if the problem is nonlinear and AD is not used.
            Convergence criteria are more involved in this case, so we do not risk
            providing a general method.

        """
        if not self._is_nonlinear_problem():
            # At least for the default direct solver, scipy.sparse.linalg.spsolve, no
            # error (but a warning) is raised for singular matrices, but a nan solution
            # is returned. We check for this.
            diverged = bool(np.any(np.isnan(solution)))
            converged: bool = not diverged
            error: float = np.nan if diverged else 0.0
            return error, converged, diverged
        else:
            # Simple but fairly robust convergence criterion.
            # More advanced options are e.g. considering errors for each variable
            # and/or each grid separately, possibly using _l2_norm_cell

            # We normalize by the size of the solution vector
            # Enforce float to make mypy happy
            error = float(np.linalg.norm(solution)) / np.sqrt(solution.size)
            logger.info(f"Normalized residual norm: {error:.2e}")
            converged = error < nl_params["nl_convergence_tol"]
            diverged = False
            return error, converged, diverged

    def _initialize_linear_solver(self) -> None:
        """Initialize linear solver.

        The default linear solver is Pardiso; this can be overridden by user choices.
        If Pardiso is not available, backup solvers will automatically be invoked in
        :meth:`solve_linear_system`.

        Raises:
            ValueError if the chosen solver is not among the three currently
            supported, see linear_solve.

            To use a custom solver in a model, override this method (and possibly
            :meth:`solve_linear_system`).

        """
        solver = self.params["linear_solver"]
        self.linear_solver = solver

        if solver not in ["scipy_sparse", "pypardiso", "umfpack"]:
            raise ValueError(f"Unknown linear solver {solver}")

    def assemble_linear_system(self) -> None:
        """Assemble the linearized system.

        The linear system is defined by the current state of the model.

        Attributes:
            linear_system is assigned.
        """
        t_0 = time.time()
        self.linear_system = self.equation_system.assemble()
        logger.debug(f"Assembled linear system in {t_0-time.time():.2e} seconds.")

    def solve_linear_system(self) -> np.ndarray:
        """Solve linear system.

        Default method is a direct solver. The linear solver is chosen in the
        initialize_linear_solver of this model. Implemented options are
            - scipy.sparse.spsolve with and without call to umfpack
            - pypardiso.spsolve

        Returns:
            np.ndarray: Solution vector.

        """
        A, b = self.linear_system
        t_0 = time.time()
        logger.debug(f"Max element in A {np.max(np.abs(A)):.2e}")
        logger.debug(
            f"""Max {np.max(np.sum(np.abs(A), axis=1)):.2e} and min
            {np.min(np.sum(np.abs(A), axis=1)):.2e} A sum."""
        )

        solver = self.linear_solver
        if solver == "pypardiso":
            # This is the default option which is invoked unless explicitly overridden by the
            # user. We need to check if the pypardiso package is available.
            try:
                from pypardiso import spsolve as sparse_solver  # type: ignore
            except ImportError:
                # Fall back on the standard scipy sparse solver.
                sparse_solver = sps.linalg.spsolve
                warnings.warn(
                    """PyPardiso could not be imported,
                    falling back on scipy.sparse.linalg.spsolve"""
                )
            x = sparse_solver(A, b)
        elif solver == "umfpack":
            # Following may be needed:
            # A.indices = A.indices.astype(np.int64)
            # A.indptr = A.indptr.astype(np.int64)
            x = sps.linalg.spsolve(A, b, use_umfpack=True)
        elif solver == "scipy_sparse":
            x = sps.linalg.spsolve(A, b)
        else:
            raise ValueError(
                f"AbstractModel does not know how to apply the linear solver {solver}"
            )
        logger.debug(f"Solved linear system in {t_0-time.time():.2e} seconds.")

        return np.atleast_1d(x)

    def _is_nonlinear_problem(self) -> bool:
        """Specifies whether the Model problem is nonlinear."""
        return True
