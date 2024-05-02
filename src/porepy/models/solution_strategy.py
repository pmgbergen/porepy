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
    update_all_boundary_conditions: Callable[[], None]
    """Set the values of the boundary conditions for the new time step.
    Defined in :class:`~porepy.models.abstract_equations.BoundaryConditionsMixin`.

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

        # Define attributes to be assigned later
        self.equation_system: pp.ad.EquationSystem
        """Equation system manager. Will be set by :meth:`set_equation_system_manager`.

        """
        self.mdg: pp.MixedDimensionalGrid
        """Mixed-dimensional grid. Will normally be set by a mixin instance of
        :class:`~porepy.models.geometry.ModelGeometry`.

        """
        self._domain: pp.Domain
        """Box-shaped domain. Will normally be set by a mixin instance of
        :class:`~porepy.models.geometry.ModelGeometry`.

        """
        self.linear_system: tuple[sps.spmatrix, np.ndarray]
        """The linear system to be solved in each iteration of the non-linear solver.
        The tuple contains the sparse matrix and the right hand side residual vector.

        """
        self._nonlinear_discretizations: list[pp.ad._ad_utils.MergedOperator] = []
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
                # Path to pvd file (given as pathlib.Path) collecting either multiple
                # time steps (generated through pp.Exporter.write_pvd()); or a pvd file
                # associated to a single time step (generated through
                # pp.Exporter._export_mdg_pvd()).
                "is_mdg_pvd": False,
                # Boolean flag controlling whether prescribed pvd file is a mdg pvd
                # file, i.e., created through Exporter._export_mdg_pvd(). Otherwise,
                # it is assumed, the provided pvd file originates from
                # Exporter.write_pvd(). If not provided, assumed to be False.
                "vtu_files": None,
                # Path(s) to vtu file(s) (of type Path or list[Path]), (alternative
                # to 'pvd_file' which is preferred if available and not 'None').
                "times_file": None,
                # Path to json file (of type pathlib.Path) containing evolution of
                # exported time steps and used time step size at that time. If 'None'
                # a default value is used internally, as defined in
                # :class:`~porepy.numerics.time_step_control.TimeManager.
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
        self.ad_time_step = pp.ad.Scalar(self.time_manager.dt)
        """Time step as an automatic differentiation scalar.

        This is used to ensure that the time step is for all equations if the time step
        is adjusted during simulation. See :meth:`before_nonlinear_loop`.

        """

        self.nonlinear_solver_statistics = pp.SolverStatistics()
        """Statistics object for non-linear solver loop."""

    def prepare_simulation(self) -> None:
        """Run at the start of simulation. Used for initialization etc."""
        # Set the material and geometry of the problem. The geometry method must be
        # implemented in a ModelGeometry class.
        self.set_materials()
        self.set_geometry()

        # Exporter initialization must be done after grid creation,
        # but prior to data initialization.
        self.initialize_data_saving()

        # Set variables, constitutive relations, discretizations and equations.
        # Order of operations is important here.
        self.set_equation_system_manager()
        self.create_variables()
        self.initial_condition()
        self.reset_state_from_file()
        self.set_equations()

        self.set_discretization_parameters()
        self.discretize()
        self._initialize_linear_solver()
        self.set_nonlinear_discretizations()

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

    def initial_condition(self) -> None:
        """Set the initial condition for the problem.

        For each solution index stored in ``self.time_step_indices`` and
        ``self.iterate_indices`` a zero initial value will be assigned.

        """
        val = np.zeros(self.equation_system.num_dofs())
        for time_step_index in self.time_step_indices:
            self.equation_system.set_variable_values(
                val,
                time_step_index=time_step_index,
            )

        for iterate_index in self.iterate_indices:
            self.equation_system.set_variable_values(val, iterate_index=iterate_index)

        # Initialize time dependent ad arrays, including those for boundary values.
        self.update_time_dependent_ad_arrays()

    @property
    def domain(self) -> pp.Domain:
        """Domain of the problem."""
        return self._domain

    @property
    def time_step_indices(self) -> np.ndarray:
        """Indices for storing time step solutions.

        Returns:
            An array of the indices of which time step solutions will be stored.

        """
        return np.array([0])

    @property
    def iterate_indices(self) -> np.ndarray:
        """Indices for storing iterate solutions.

        Returns:
            An array of the indices of which iterate solutions will be stored.

        """
        return np.array([0])

    def reset_state_from_file(self) -> None:
        """Reset states but through a restart from file.

        Similar to :meth:`initial_condition`.

        """
        # Overwrite states from file if restart is enabled.
        if self.restart_options.get("restart", False):
            if self.restart_options.get("pvd_file", None) is not None:
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
            vals = self.equation_system.get_variable_values(time_step_index=0)
            self.equation_system.set_variable_values(
                vals, iterate_index=0, time_step_index=0
            )
            # Update the boundary conditions to both the time step and iterate solution.
            self.update_time_dependent_ad_arrays()

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

    def discretize(self) -> None:
        """Discretize all terms."""
        tic = time.time()
        # Do a discretization of the equations. More refined control of the
        # discretization process can be achieved by exploiting knowledge of the equation
        # system (e.g., which terms are linear and need not be discretized at every
        # iteration).
        self.equation_system.discretize()
        logger.info("Discretized in {} seconds".format(time.time() - tic))

    def rediscretize(self) -> None:
        """Discretize nonlinear terms."""
        tic = time.time()
        # Uniquify to save computational time, then discretize.
        unique_discr = pp.ad._ad_utils.uniquify_discretization_list(
            self.nonlinear_discretizations
        )
        pp.ad._ad_utils.discretize_from_list(unique_discr, self.mdg)
        logger.debug(
            "Re-discretized nonlinear terms in {} seconds".format(time.time() - tic)
        )

    @property
    def nonlinear_discretizations(self) -> list[pp.ad._ad_utils.MergedOperator]:
        """List of nonlinear discretizations in the equation system. The discretizations
        are recomputed  in :meth:`before_nonlinear_iteration`.

        """
        return self._nonlinear_discretizations

    def add_nonlinear_discretization(
        self, discretization: pp.ad._ad_utils.MergedOperator
    ) -> None:
        """Add an entry to the list of nonlinear discretizations.

        Parameters:
            discretization: The nonlinear discretization to be added.

        """
        # This guardrail is very weak. However, the discretization list is uniquified
        # before discretization, so it should not be a problem.
        if discretization not in self._nonlinear_discretizations:
            self._nonlinear_discretizations.append(discretization)

    def set_nonlinear_discretizations(self) -> None:
        """Set the list of nonlinear discretizations.

        This method is called before the discretization is performed. It is intended to
        be used to set the list of nonlinear discretizations.

        """

    def before_nonlinear_loop(self) -> None:
        """Method to be called before entering the non-linear solver, thus at the start
        of a new time step.

        Possible usage is to update time-dependent parameters, discretizations etc.

        """
        # Update time step size.
        self.ad_time_step.set_value(self.time_manager.dt)
        # Update the boundary conditions to both the time step and iterate solution.
        self.update_time_dependent_ad_arrays()
        # Empty the log in the statistics object.
        self.nonlinear_solver_statistics.reset()

    def before_nonlinear_iteration(self) -> None:
        """Method to be called at the start of every non-linear iteration.

        Possible usage is to update non-linear parameters, discretizations etc.

        """
        # Update parameters before re-discretizing.
        self.set_discretization_parameters()
        # Re-discretize nonlinear terms. If none have been added to
        # self.nonlinear_discretizations, this will be a no-op.
        self.rediscretize()

    def after_nonlinear_iteration(self, nonlinear_increment: np.ndarray) -> None:
        """Method to be called after every non-linear iteration.

        Possible usage is to distribute information on the new trial state, visualize
        the current approximation etc.

        Parameters:
            nonlinear_increment: The new solution, as computed by the non-linear solver.

        """
        self.equation_system.shift_iterate_values()
        self.equation_system.set_variable_values(
            values=nonlinear_increment, additive=True, iterate_index=0
        )
        self.nonlinear_solver_statistics.num_iteration += 1

    def after_nonlinear_convergence(self) -> None:
        """Method to be called after every non-linear iteration.

        Possible usage is to distribute information on the solution, visualization, etc.

        """
        solution = self.equation_system.get_variable_values(iterate_index=0)
        self.equation_system.shift_time_step_values()
        self.equation_system.set_variable_values(
            values=solution, time_step_index=0, additive=False
        )
        self.convergence_status = True
        self.save_data_time_step()

    def after_nonlinear_failure(self) -> None:
        """Method to be called if the non-linear solver fails to converge."""
        self.save_data_time_step()
        if self._is_nonlinear_problem():
            raise ValueError("Nonlinear iterations did not converge.")
        else:
            raise ValueError("Tried solving singular matrix for the linear problem.")

    def after_simulation(self) -> None:
        """Run at the end of simulation. Can be used for cleanup etc."""
        pass

    def check_convergence(
        self,
        nonlinear_increment: np.ndarray,
        residual: np.ndarray,
        reference_residual: np.ndarray,
        nl_params: dict[str, Any],
    ) -> tuple[float, float, bool, bool]:
        """Implements a convergence check, to be called by a non-linear solver.

        Parameters:
            nonlinear_increment: Newly obtained solution increment vector
            residual: Residual vector of non-linear system, evaluated at the newly
            obtained solution vector.
            reference_residual: Reference residual vector of non-linear system, evaluated
                for the initial guess at current time step.
            nl_params: Dictionary of parameters used for the convergence check.
                Which items are required will depend on the convergence test to be
                implemented.

        Returns:
            The method returns the following tuple:

            float:
                Residual norm, computed to the norm in question.
            float:
                Increment norm, computed to the norm in question.
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
            diverged = bool(np.any(np.isnan(nonlinear_increment)))
            converged: bool = not diverged
            residual_norm: float = np.nan if diverged else 0.0
            nonlinear_increment_norm: float = np.nan if diverged else 0.0
        else:
            # First a simple check for nan values.
            if np.any(np.isnan(nonlinear_increment)):
                # If the solution contains nan values, we have diverged.
                return np.nan, np.nan, False, True

            # nonlinear_increment based norm
            nonlinear_increment_norm = self.compute_nonlinear_increment_norm(
                nonlinear_increment
            )
            # Residual based norm
            residual_norm = self.compute_residual_norm(residual, reference_residual)
            logger.debug(
                f"Nonlinear increment norm: {nonlinear_increment_norm:.2e}, "
                f"Nonlinear residual norm: {residual_norm:.2e}"
            )
            # Check convergence requiring both the increment and residual to be small.
            converged_inc = nonlinear_increment_norm < nl_params["nl_convergence_tol"]
            converged_res = residual_norm < nl_params["nl_convergence_tol_res"]
            converged = converged_inc and converged_res
            diverged = False

        # Log the errors (here increments and residuals)
        self.nonlinear_solver_statistics.log_error(
            nonlinear_increment_norm, residual_norm
        )

        return residual_norm, nonlinear_increment_norm, converged, diverged

    def compute_residual_norm(
        self, residual: np.ndarray, reference_residual: np.ndarray
    ) -> float:
        """Compute the residual norm for a nonlinear iteration.

        Parameters:
            residual: Residual of current iteration.
            reference_residual: Reference residual value (initial residual expected),
                allowing for definiting relative criteria.

        Returns:
            float: Residual norm.

        """
        residual_norm = np.linalg.norm(residual) / np.sqrt(residual.size)
        return residual_norm

    def compute_nonlinear_increment_norm(
        self, nonlinear_increment: np.ndarray
    ) -> float:
        """Compute the norm based on the update increment for a nonlinear iteration.

        Parameters:
            nonlinear_increment: Solution to the linearization.

        Returns:
            float: Update increment norm.

        """
        # Simple but fairly robust convergence criterions. More advanced options are
        # e.g. considering norms for each variable and/or each grid separately,
        # possibly using _l2_norm_cell
        # We normalize by the size of the solution vector.
        nonlinear_increment_norm = np.linalg.norm(nonlinear_increment) / np.sqrt(
            nonlinear_increment.size
        )
        return nonlinear_increment_norm

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
        logger.debug(f"Assembled linear system in {time.time() - t_0:.2e} seconds.")

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
        logger.info(f"Solved linear system in {time.time() - t_0:.2e} seconds.")

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

    def update_time_dependent_ad_arrays(self) -> None:
        """Update the time dependent arrays before a new time step.

        The base implementation updates those for the boundary condition values.
        Override it to update other model-specific time dependent arrays.

        """
        self.update_all_boundary_conditions()
