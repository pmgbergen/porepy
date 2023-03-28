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
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import scipy.sparse as sps

import porepy as pp

logger = logging.getLogger(__name__)


class SolutionStrategy(abc.ABC):
    """This is a class that specifies methods that a model must implement to
    be compatible with the linearization and time stepping methods.

    """

    nd: int
    """Ambient dimension of the problem. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """
    set_geometry: Callable[[], None]
    """Set the geometry of the model. Normally provided by a mixin instance of
    :class:`~porepy.models.geometry.ModelGeometry`.

    """
    initialize_data_saving: Callable[[], None]
    """Initialize data saving. Normally provided by a mixin instance of
    :class:`~porepy.viz.data_saving_model_mixin.DataSavingMixin`.

    """
    save_data_time_step: Callable[[], None]
    """Save data at a time step. Normally provided by a mixin instance of
    :class:`~porepy.viz.data_saving_model_mixin.DataSavingMixin`.

    """
    create_variables: Callable[[], None]
    """Create variables. Normally provided by a mixin instance of a Variable class
    relevant to the model.

    """
    set_equations: Callable[[], None]
    """Set the governing equations of the model. Normally provided by the solution
    strategy of a specific model (i.e. a subclass of this class).

    """
    load_data_from_vtu: Callable[[Path, int, Optional[Path]], None]
    """Load data from vtu to initialize the states, only applicable in restart mode.
    :class:`~porepy.viz.exporter.Exporter`.

    """
    load_data_from_pvd: Callable[[Path, bool, Optional[Path]], None]
    """Load data from pvd to initialize the states, only applicable in restart mode.
    :class:`~porepy.viz.exporter.Exporter`.

    """

    def __init__(self, params: Optional[dict] = None):
        """Initialize the solution strategy.

        Parameters:
            params: Parameters for the solution strategy. Defaults to
                None.

        """
        if params is None:
            params = {}

        # Set default parameters, these will be overwritten by any parameters passed.
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
        """Number of non-linear iterations performed for current time step."""

        # Define attributes to be assigned later
        self.equation_system: pp.ad.EquationSystem
        """Equation system manager. Will be set by :meth:`set_equation_system_manager`.

        """
        self.mdg: pp.MixedDimensionalGrid
        """Mixed-dimensional grid. Will normally be set by a mixin instance of
        :class:`~porepy.models.geometry.ModelGeometry`.

        """
        self.domain: pp.Domain
        """Box-shaped domain. Will normally be set by a mixin instance of
        :class:`~porepy.models.geometry.ModelGeometry`.

        """
        self.linear_system: tuple[sps.spmatrix, np.ndarray]
        """The linear system to be solved in each iteration of the non-linear solver.
        The tuple contains the sparse matrix and the right hand side residual vector.

        """
        self.exporter: pp.Exporter
        """Exporter for visualization."""

        self.units: pp.Units = params.get("units", pp.Units())
        """Units of the model. See also :meth:`set_units`."""

        self.fluid: pp.FluidConstants
        """Fluid constants. See also :meth:`set_materials`."""

        self.solid: pp.SolidConstants
        """Solid constants. See also :meth:`set_materials`."""

        self.time_manager = params.get(
            "time_manager",
            pp.TimeManager(schedule=[0, 1], dt_init=1, constant_dt=True),
        )
        """Time manager for the simulation."""

        self.restart_options: dict = params.get(
            "restart_options",
            {
                "restart": False,
                # Boolean flag controlling whether restart is active. Internally
                # assumed to be False.
                "pvd_file": None,
                # Path to pvd file collecting either multiple time steps (generated
                # through pp.Exporter.write_pvd()); or a pvd file associated to a single
                # time step (generated through pp.Exporter._export_mdg_pvd()).
                "is_mdg_pvd": False,
                # Boolean flag controlling whether prescribed pvd file is a mdg pvd
                # file, i.e., created through Exporter._export_mdg_pvd(). Otherwise,
                # it is assumed, the provided pvd file originates from
                # Exporter.write_pvd(). If not provided, assumed to be False.
                "vtu_files": None,
                # Path(s) to vtu file(s), (alternative to pvd files which are preferred
                # if present). Required if pvd_file is not specified.
                "times_file": None,
                # Path to json file containing evolution of exported time steps and
                # used time step size at that time. If 'None' a default value,
                # defined in :class:`~porepy.numerics.time_step_control.TimeManager
                # is internally used.
                "time_index": -1,
                # Index addressing history in times_file; only relevant if "vtu_files"
                # is not 'None' or "is_mdg_pvd" is 'True'. The index corresponds to
                # the single time step vtu/pvd files. If not provided, internally
                # assumed to address the last time step in times_file.
            },
        )
        """Restart options (template) for restart from pvd as expected restart routines
        within :class:`~porepy.viz.data_saving_model_mixin.DataSavingMixin`.
        """

    def prepare_simulation(self) -> None:
        """Run at the start of simulation. Used for initialization etc."""
        # Set the geometry of the problem. This is a method that must be implemented
        # in a ModelGeometry class.
        self.set_geometry()

        # Exporter initialization must be done after grid creation,
        # but prior to data initialization.
        self.initialize_data_saving()

        # Set variables, constitutive relations, discretizations and equations.
        # Order of operations is important here.
        self.set_equation_system_manager()
        self.set_materials()
        self.create_variables()
        self.initial_condition()
        self.set_equations()

        self.set_discretization_parameters()
        self.discretize()
        self._initialize_linear_solver()

        # Export initial condition
        self.save_data_time_step()

    def set_equation_system_manager(self) -> None:
        """Create an equation_system manager on the mixed-dimensional grid."""
        if not hasattr(self, "equation_system"):
            self.equation_system = pp.ad.EquationSystem(self.mdg)

    def set_discretization_parameters(self) -> None:
        """Set parameters for the discretization.

        This method is called before the discretization is performed. It is
        intended to be used to set parameters for the discretization, such as
        the permeability, the porosity, etc.

        """
        pass

    def initial_condition(self) -> None:
        """Set the initial condition for the problem."""
        vals = np.zeros(self.equation_system.num_dofs())
        self.equation_system.set_variable_values(vals, to_iterate=True, to_state=True)

        # Overwrite states from file if restart is enabled.
        if self.restart_options.get("restart", False):
            if "pvd_file" in self.restart_options:
                pvd_file = self.restart_options["pvd_file"]
                is_mdg_pvd = self.restart_options.get("is_mdg_pvd", False)
                times_file = self.restart_options.get("times_file", None)
                self.load_data_from_pvd(
                    pvd_file,
                    is_mdg_pvd,
                    times_file,
                )
            else:
                vtu_files = self.restart_options["vtu_files"]
                time_index = self.restart_options.get("time_index", -1)
                times_file = self.restart_options.get("times_file", None)
                self.load_data_from_vtu(
                    vtu_files,
                    time_index,
                    times_file,
                )
            vals = self.equation_system.get_variable_values()
            self.equation_system.set_variable_values(
                vals, to_iterate=True, to_state=True
            )

    def set_materials(self):
        """Set material parameters.

        In addition to adjusting the units (see ::method::set_units), materials are
        defined through ::class::pp.Material and the constants passed when initializing
        the materials. For most purposes, a user needs only pass the desired parameter
        values in params["material_constants"] when initializing a solution strategy,
        and should not need to modify the material classes. However, if a user wishes to
        modify to e.g. provide additional material parameters, this can be done by
        passing modified material classes in params["fluid"] and params["solid"].

        """
        # Default values
        constants = {}
        constants.update(self.params.get("material_constants", {}))
        # Use standard models for fluid and solid constants if not provided.
        if "fluid" not in constants:
            constants["fluid"] = pp.FluidConstants()
        if "solid" not in constants:
            constants["solid"] = pp.SolidConstants()

        # Loop over all constants objects (fluid, solid), and set units.
        for name, const in constants.items():
            # Check that the object is of the correct type
            assert isinstance(const, pp.models.material_constants.MaterialConstants)
            # Impose the units passed to initialization of the model.
            const.set_units(self.units)
            # This is where the constants (fluid, solid) are actually set as attributes
            setattr(self, name, const)

    def before_newton_loop(self) -> None:
        """Wrap for legacy reasons. Call :meth:`before_nonlinear_loop` instead."""
        # TODO: Remove and call before_nonlinear_loop directly.
        self.before_nonlinear_loop()

    def discretize(self) -> None:
        """Discretize all terms."""
        tic = time.time()
        # Do a discretization of the equations. More refined control of the
        # discretization process can be achieved by exploiting knowledge of the equation
        # system (e.g., which terms are linear and need not be discretized at every
        # iteration).
        self.set_discretization_parameters()
        self.equation_system.discretize()
        logger.info("Discretized in {} seconds".format(time.time() - tic))

    def before_nonlinear_loop(self) -> None:
        """Method to be called before entering the non-linear solver, thus at the start
        of a new time step.

        Possible usage is to update time-dependent parameters, discretizations etc.

        """
        self._nonlinear_iteration = 0

    def before_newton_iteration(self) -> None:
        """Wrap for legacy reasons. Call :meth:`before_nonlinear_iteration` instead."""
        self.before_nonlinear_iteration()

    def before_nonlinear_iteration(self) -> None:
        """Method to be called at the start of every non-linear iteration.

        Possible usage is to update non-linear parameters, discretizations etc.

        """
        pass

    def after_newton_iteration(self, solution_vector: np.ndarray):
        """Wrap for legacy reasons. Call :meth:`after_nonlinear_iteration` instead.

        Parameters:
            solution_vector: The new solution state, as computed by the non-linear
                solver.

        """
        # TODO: Remove and call after_nonlinear_iteration directly.
        self.after_nonlinear_iteration(solution_vector)

    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        """Method to be called after every non-linear iteration.

        Possible usage is to distribute information on the new trial state, visualize
        the current approximation etc.

        Parameters:
            solution_vector: The new solution state, as computed by the non-linear
                solver.

        """
        self._nonlinear_iteration += 1
        self.equation_system.set_variable_values(
            values=solution_vector, additive=True, to_iterate=True
        )

    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Wrap for legacy reasons. Call :meth:`after_nonlinear_convergence` instead.

        Parameters:
            solution: The new solution state, as computed by the non-linear solver.
            errors: The error in the solution, as computed by the non-linear solver.
            iteration_counter: The number of iterations performed by the non-linear
                solver.

        """
        self.after_nonlinear_convergence(solution, errors, iteration_counter)

    def after_nonlinear_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Method to be called after every non-linear iteration.

        Possible usage is to distribute information on the solution, visualization, etc.

        Parameters:
            solution: The new solution state, as computed by the non-linear solver.
            errors: The error in the solution, as computed by the non-linear solver.
            iteration_counter: The number of iterations performed by the non-linear
                solver.

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
        """Method to be called if the non-linear solver fails to converge.

        Parameters:
            solution: The new solution state, as computed by the non-linear solver.
            errors: The error in the solution, as computed by the non-linear solver.
            iteration_counter: The number of iterations performed by the non-linear
                solver.
        """
        self.after_nonlinear_failure(solution, errors, iteration_counter)

    def after_nonlinear_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Method to be called if the non-linear solver fails to converge.

        Parameters:
            solution: The new solution state, as computed by the non-linear solver.
            errors: The error in the solution, as computed by the non-linear solver.
            iteration_counter: The number of iterations performed by the non-linear
                solver.
        """
        if self._is_nonlinear_problem():
            raise ValueError("Newton iterations did not converge.")
        else:
            raise ValueError("Tried solving singular matrix for the linear problem.")

    def after_simulation(self) -> None:
        """Run at the end of simulation. Can be used for cleanup etc."""
        pass

    def check_convergence(
        self,
        solution: np.ndarray,
        prev_solution: np.ndarray,
        init_solution: np.ndarray,
        nl_params: dict[str, Any],
    ) -> tuple[float, bool, bool]:
        """Implements a convergence check, to be called by a non-linear solver.

        Parameters:
            solution: Newly obtained solution vector prev_solution: Solution obtained in
            the previous non-linear iteration. init_solution: Solution obtained from the
            previous time-step. nl_params: Dictionary of parameters used for the
            convergence check.
                Which items are required will depend on the convergence test to be
                implemented.

        Returns:
            The method returns the following tuple:

            float:
                Error, computed to the norm in question.
            boolean:
                True if the solution is converged according to the test implemented by
                this method.
            boolean:
                True if the solution is diverged according to the test implemented by
                this method.

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

        To use a custom solver in a model, override this method (and possibly
        :meth:`solve_linear_system`).

        Raises:
            ValueError if the chosen solver is not among the three currently supported,
            see linear_solve.

        """
        solver = self.params["linear_solver"]
        self.linear_solver = solver

        if solver not in ["scipy_sparse", "pypardiso", "umfpack"]:
            raise ValueError(f"Unknown linear solver {solver}")

    def assemble_linear_system(self) -> None:
        """Assemble the linearized system and store it in :attr:`linear_system`.

        The linear system is defined by the current state of the model.

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

        See also:
            :meth:`initialize_linear_solver`

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
            # This is the default option which is invoked unless explicitly overridden
            # by the user. We need to check if the pypardiso package is available.
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
        """Specifies whether the Model problem is nonlinear.

        Returns:
            bool: True if the problem is nonlinear, False otherwise.

        """
        return True

    def _is_time_dependent(self) -> bool:
        """Specifies whether the Model problem is time-dependent.

        Returns:
            bool: True if the problem is time-dependent, False otherwise.
        """
        return True

    def update_time_dependent_ad_arrays(self, initial: bool) -> None:
        """Update the time dependent arrays for the mechanics boundary conditions.

        Parameters:
            initial: If True, the array generating method is called for both state and
                iterate. If False, the array generating method is called only for the
                iterate, and the state is updated by copying the iterate.

        """
        pass
