""" Contains the mother class for all models, that is, standardized setups for complex
problems.

Some methods may raise NotImplementedErrors instead of being abstract methods. This
is because they are only needed for some model types (typically nonlinear ones).

Note on running simulations: We provide run_time_dependent_model and run_stationary_model.
Both call the Model's prepare_simulation and after_simulation methods, as well as the
solve method of either a LinearSolver or NewtonSolver (the only non-linear solver available
for the time being). The Solvers call methods before_newton_loop, check_convergence,
after_newton_convergence/after_newton_divergence, (only NewtonSolver:) before_newton_iteration,
after_newton_iteration. The name "newton" is there for legacy reasons, a more fitting
and general name would be something like "equation_solve".
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


class AbstractModel:
    """This is a class that specifies methods that a model must implement to
    be compatible with the Newton and time stepping methods.

    Public attributes:
        params (Dict): Simulation specific parameters. Which items are admissible
            depends on the model in question.
        convergence_status (bool): Whether the non-linear iteration has converged.
        linear_solver (str): Specification of linear solver. Only known permissible
            value is 'direct'.
        dof_manager (pp.DofManager): Degree of freedom manager.
    """

    def __init__(self, params: Optional[Dict] = None):
        if params is None:
            self.params = {}
        else:
            self.params = params
        if params is None:
            params = {}
        default_params = {
            "folder_name": "visualization",
            "file_name": "data",
            "use_ad": True,
            # Set the default linear solver to Pardiso; this can be overridden by
            # user choices. If Pardiso is not available, backup solvers will automatically be
            # invoked.
            "linear_solver": "pypardiso",
        }

        default_params.update(params)
        self.params = default_params

        # Set a convergence status. Not sure if a boolean is sufficient, or whether
        # we should have an enum here.
        self.convergence_status: bool = False

        self._nonlinear_iteration: int = 0
        assert isinstance(self.params["use_ad"], bool)
        self._use_ad = self.params["use_ad"]

        # Define attributes to be assigned later
        self._eq_manager: pp.ad.EquationManager
        self.dof_manager: pp.DofManager
        self.mdg: pp.MixedDimensionalGrid
        self.box: dict
        self.linear_system: tuple[sps.spmatrix, np.ndarray]

    def create_grid(self) -> None:
        """Create the grid bucket.

        A unit square grid with no fractures is assigned by default.

        The method assigns the following attributes to self:
            mdg (pp.MixedDimensionalGrid): The produced grid bucket.
            box (dict): The bounding box of the domain, defined through minimum and
                maximum values in each dimension.
        """
        phys_dims = np.array([1, 1])
        n_cells = np.array([1, 1])
        self.box = pp.geometry.bounding_box.from_points(np.array([[0, 0], phys_dims]).T)
        g: pp.Grid = pp.CartGrid(n_cells, phys_dims)
        g.compute_geometry()
        self.mdg = pp.meshing.subdomains_to_mdg([[g]])
        # If fractures are present, it is advised to call
        # pp.contact_conditions.set_projections(self.mdg)

    def _initial_condition(self) -> None:
        """Set initial guess for the variables."""
        initial_values = np.zeros(self.equation_system.num_dofs())
        self.equation_system.set_variable_values(initial_values, to_state=True, to_iterate=True)

    def prepare_simulation(self) -> None:
        """Method called prior to the start of time stepping, or prior to entering the
        non-linear solver for stationary problems.

        The intended use is to define parameters, geometry and grid, discretize linear
        and time-independent terms, and generally prepare for the simulation.

        """
        raise NotImplementedError

    def before_newton_loop(self) -> None:
        """Method to be called before entering the non-linear solver, thus at the start
        of a new time step.

        Possible usage is to update time-dependent parameters, discretizations etc.

        """
        raise NotImplementedError

    def before_newton_iteration(self) -> None:
        """Method to be called at the start of every non-linear iteration.

        Possible usage is to update non-linear parameters, discretizations etc.

        """
        raise NotImplementedError

    def after_newton_iteration(self, solution_vector: np.ndarray):
        """Method to be called after every non-linear iteration.

        Possible usage is to distribute information on the new trial state, visualize
        the current approximation etc.

        Parameters:
            np.array: The new solution state, as computed by the non-linear solver.

        """
        raise NotImplementedError

    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Method to be called after every non-linear iteration.

        Possible usage is to distribute information on the solution, visualization, etc.

        Parameters:
            np.array: The new solution state, as computed by the non-linear solver.

        """
        raise NotImplementedError

    def after_newton_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        if self._is_nonlinear_problem():
            raise ValueError("Newton iterations did not converge")
        else:
            raise ValueError("Tried solving singular matrix for the linear problem.")

    def after_simulation(self) -> None:
        """Run at the end of simulation. Can be used for cleanup etc."""
        raise NotImplementedError

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
        elif self._use_ad:
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
        else:
            raise NotImplementedError

    def _initialize_linear_solver(self) -> None:
        """Initialize linear solver.

        Raises:
            ValueError if the chosen solver is not among the three currently
            supported, see linear_solve.

            To use a custom solver in a model, override this method (and
            linear_solve).

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
        if self._use_ad:
            A, b = self._eq_manager.assemble()
        else:
            A, b = self.assembler.assemble_matrix_rhs()  # type: ignore
        self.linear_system = (A, b)
        logger.debug(f"Assembled linear system in {t_0-time.time():.2e} seconds.")

    def solve_linear_system(self) -> np.ndarray:
        """Solve linear system.

        Default method is a direct solver. The linear solver is chosen in the
        initialize_linear_solver of this model. Implemented options are
            scipy.sparse.spsolve
                with and without call to umfpack
            pypardiso.spsolve

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

    @abc.abstractmethod
    def _is_nonlinear_problem(self) -> bool:
        """Specifies whether the Model problem is nonlinear."""
        pass

    ## Utility methods
    def _l2_norm_cell(self, g: pp.Grid, val: np.ndarray) -> float:
        """
        Compute the cell volume weighted norm of a vector-valued cell-wise quantity for
        a given grid.

        Parameters:
            g (pp.Grid): Grid
            val (np.array): Vector-valued function.

        Returns:
            double: The computed L2-norm.

        """
        nc = g.num_cells
        sz = val.size
        if nc == sz:
            nd = 1
        elif nc * g.dim == sz:
            nd = g.dim
        else:
            raise ValueError("Have not considered this type of unknown vector")

        norm = np.sqrt(np.reshape(val**2, (nd, nc), order="F") * g.cell_volumes)

        return np.sum(norm)

    def _nd_subdomain(self) -> pp.Grid:
        """Get the grid of the highest dimension. Assumes self.mdg is set."""
        return list(self.mdg.subdomains(dim=self.mdg.dim_max()))[0]

    def _export(self) -> None:
        """Method to export the solution to using an Exporter.

        Method is called after initialization and after solution convergence.
        """
        pass

    def _domain_boundary_sides(
        self, g: pp.Grid
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Obtain indices of the faces of a grid that lie on each side of the domain
        boundaries. It is assumed the domain is box shaped.
        """
        tol = 1e-10
        box = self.box
        east = g.face_centers[0] > box["xmax"] - tol
        west = g.face_centers[0] < box["xmin"] + tol
        if self.mdg.dim_max() == 1:
            north = np.zeros(g.num_faces, dtype=bool)
            south = north.copy()
        else:
            north = g.face_centers[1] > box["ymax"] - tol
            south = g.face_centers[1] < box["ymin"] + tol
        if self.mdg.dim_max() < 3:
            top = np.zeros(g.num_faces, dtype=bool)
            bottom = top.copy()
        else:
            top = g.face_centers[2] > box["zmax"] - tol
            bottom = g.face_centers[2] < box["zmin"] + tol
        all_bf = g.get_boundary_faces()
        return all_bf, east, west, north, south, top, bottom
