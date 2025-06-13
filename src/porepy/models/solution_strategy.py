"""Solution strategy classes.

This class is a modified version of relevant parts of AbstractModel.
In the future, it may be possible to merge the two classes. For now, we
keep them separate, to avoid breaking existing code (legacy models).
"""

from __future__ import annotations

import logging
import time
import warnings
from functools import partial
from typing import Any, Callable, Literal, Optional, cast

import numpy as np
import scipy.sparse as sps

import porepy as pp

logger = logging.getLogger(__name__)


class SolutionStrategy(pp.PorePyModel):
    """This is a class that specifies methods that a model must implement to
    be compatible with the linearization and time stepping methods.

    """

    def __init__(self, params: Optional[dict] = None):
        """Initialize the solution strategy.

        Parameters:
            params: Parameters for the solution strategy. Defaults to None.

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
        self.convergence_status = False
        """Whether the non-linear iteration has converged."""
        self._nonlinear_discretizations: list[pp.ad._ad_utils.MergedOperator] = []
        """See :meth:`add_nonlinear_discretization`."""
        self._nonlinear_diffusive_flux_discretizations: list[
            pp.ad._ad_utils.MergedOperator
        ] = []
        """See :meth:`add_nonlinear_diffusive_flux_discretization`."""
        self.units = params.get("units", pp.Units())
        """Units of the model provided in ``params['units']``."""
        # get default or user-provided reference values
        reference_values: pp.ReferenceVariableValues = params.get(
            "reference_variable_values", pp.ReferenceVariableValues()
        )
        # Ensure the reference values are in the right units
        reference_values = reference_values.to_units(self.units)
        self.reference_variable_values = reference_values
        """The model reference values for variables, converted to simulation
        :attr:`units`.

        Reference values can be provided through ``params['reference_values']``.

        """

        self.time_manager = params.get(
            "time_manager",
            pp.TimeManager(schedule=[0, 1], dt_init=1, constant_dt=True),
        )
        """Time manager for the simulation."""
        self.restart_options = params.get(
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
        """Restart options. The template is provided in `SolutionStrategy.__init__`."""
        self.ad_time_step = pp.ad.Scalar(self.time_manager.dt)
        """Time step as an automatic differentiation scalar."""
        self.results: list[Any] = []
        """A list of results collected by the data saving mixin in
        :meth:`~porepy.viz.data_saving_model_mixin.DataSavingMixin.collect_data`."""

        self._operator_cache: dict[Any, pp.ad.Operator] = {}
        """Cache for storing the result of methods that return Ad operators. This is
        used to avoid re-construction of the same operator multiple times, but does not
        affect evaluation of the operator.

        An operator is added to the cache by adding the decorator @pp.ad.cache_operator
        to the method that returns the operator. It is considered good practice to use
        the cache sparingly, and only for operators that have been shown to be expensive
        to construct.
        """

        self._schur_complement_primary_variables: list[str] = []
        """See :meth:`schur_complement_primary_variables`."""

        self._schur_complement_primary_equations: list[str] = []
        """See :meth:`schur_complement_primary_equations`."""

        self.set_solver_statistics()

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
        # After fluid and variables are defined, we can define the secondary quantities
        # like fluid properties (which depend on variables). Creating fluid and
        # variables before defining secondary thermodynamic properties is critical in
        # the case where properties depend on some fractions. since the callables for
        # secondary variables are dynamically created during create_variables, as
        # opposed to e.g. pressure or temperature.
        self.assign_thermodynamic_properties_to_phases()
        self.initial_condition()
        self.initialize_previous_iterate_and_time_step_values()

        # Initialize time dependent ad arrays, including those for boundary values.
        self.update_time_dependent_ad_arrays()
        self.reset_state_from_file()
        self.set_equations()

        self.update_discretization_parameters()
        self.discretize()
        self._initialize_linear_solver()
        self.set_nonlinear_discretizations()

        # Export initial condition
        self.save_data_time_step()

    def initialize_previous_iterate_and_time_step_values(self) -> None:
        """Method to be called after initial values are set at ``iterate_index=0`` in
        the mixins for initial conditions.

        This methods copies respective values to all other iterate and time step indices
        to finalize the initialization procedure.

        """
        val = self.equation_system.get_variable_values(iterate_index=0)
        for iterate_index in self.iterate_indices:
            self.equation_system.set_variable_values(
                val,
                iterate_index=iterate_index,
            )
        for time_step_index in self.time_step_indices:
            self.equation_system.set_variable_values(
                val,
                time_step_index=time_step_index,
            )

    def set_equation_system_manager(self) -> None:
        """Create an equation_system manager on the mixed-dimensional grid."""
        if not hasattr(self, "equation_system"):
            self.equation_system = pp.ad.EquationSystem(self.mdg)

    def set_solver_statistics(self) -> None:
        """Set the solver statistics object.

        This method is called at initialization. It is intended to be used to set the
        solver statistics object(s). Currently, the solver statistics object is related
        to nonlinearity only. Statistics on other parts of the solution process, such as
        linear solvers, may be added in the future.

        Raises:
            ValueError: If the solver statistics object is not a subclass of
                pp.SolverStatistics.

        """
        # Retrieve the value with a default of pp.SolverStatistics.
        statistics = self.params.get("nonlinear_solver_statistics", pp.SolverStatistics)
        # Explicitly check if the retrieved value is a class and a subclass of
        # pp.SolverStatistics for type checking.
        if isinstance(statistics, type) and issubclass(statistics, pp.SolverStatistics):
            self.nonlinear_solver_statistics = statistics()

        else:
            raise ValueError(
                f"Expected a subclass of pp.SolverStatistics, got {statistics}."
            )

    @property
    def time_step_indices(self) -> np.ndarray:
        """Indices for storing time step solutions.

        Index 0 corresponds to the most recent time step with the know solution, 1 to
        the previous time step, etc.

        Returns:
            An array of the indices of which time step solutions will be stored,
            counting from 0. Defaults to storing the most recently computed solution
            only.

        """
        return np.array([0])

    @property
    def iterate_indices(self) -> np.ndarray:
        """Indices for storing iterate solutions.

        Returns:
            An array of the indices of which iterate solutions will be stored.

        """
        return np.array([0])

    @property
    def schur_complement_primary_equations(self) -> list[str]:
        """Names of the primary equations for the Schur complement reduction of the
        linear system.

        They define the row-block which does not contain the sub-matrix which is to be
        inverted for the Schur complement.

        See also:

            - :meth:`assemble_linear_system`
            - :meth:`~porepy.numerics.ad.equation_system.EquationSystem.
              assemble_schur_complement_system`

        Parameters:
            names: List of equation names to be set as primary equations.

        Raises:
            ValueError: If any name is not known to the model's equation system or the
                given names are not unique.

        Returns:
            The names of the equations (currently) defined as primary equations.

        """
        return self._schur_complement_primary_equations

    @schur_complement_primary_equations.setter
    def schur_complement_primary_equations(self, names: list[str]) -> None:
        known_equations = list(self.equation_system.equations.keys())
        for n in names:
            if n not in known_equations:
                raise ValueError(f"Equation {n} unknown to the equation system.")
        if len(set(names)) != len(names):
            raise ValueError("Primary equation names must be unique.")
        # Shallow copy for safety, and keep order of equation system.
        self._schur_complement_primary_equations = [
            n for n in known_equations if n in names
        ]

    @property
    def schur_complement_primary_variables(self) -> list[str]:
        """Names of the primary variables for the Schur complement reduction of the
        linear system.

        They define the column-block which does not contain the sub-matrix which is to
        be inverted for the Schur complement.

        See also:

            - :meth:`assemble_linear_system`
            - :meth:`~porepy.numerics.ad.equation_system.EquationSystem.
              assemble_schur_complement_system`

        Parameters:
            names: List of variable names to be set as primary variables.

        Raises:
            ValueError: If any name is not known to the model's equation system or the
                given names are not unique.

        Returns:
            The names of the variables (currently) defined as primary variables.

        """
        return self._schur_complement_primary_variables

    @schur_complement_primary_variables.setter
    def schur_complement_primary_variables(self, names: list[str]) -> None:
        known_variables = list(set(v.name for v in self.equation_system.variables))
        for n in names:
            if n not in known_variables:
                raise ValueError(f"Variable {n} unknown to the equation system.")
        if len(set(names)) != len(names):
            raise ValueError("Primary variables names must be unique.")
        # Shallow copy for safety
        self._schur_complement_primary_variables = [n for n in names]

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

        Searches for entries in ``params['material_constants']`` with keys ``'fluid'``
        and ``'solid'`` for respective material constant instances. If not found,
        default materials are instantiated.

        Provides the :attr:`solid` material constants as an attribute to the model, as
        well as the :attr:`fluid` object by calling :attr:`create_fluid`.

        By default, a 1-phase, 1-component fluid is created based on the fluid component
        provided in ``params['material_constants']``.

        """
        # User provided values, if any.
        constants = cast(
            dict[str, pp.Constants], self.params.get("material_constants", {})
        )
        # If the user provided material constants, assert they are in dictionary form
        assert isinstance(constants, dict), (
            "model.params['material_constants'] must be a dictionary."
        )

        # Use standard models for fluid, solid and numerical constants if not provided.
        # Otherwise get the given constants.
        solid = cast(
            pp.SolidConstants, constants.get("solid", pp.SolidConstants(name="solid"))
        )
        fluid = cast(
            pp.FluidComponent, constants.get("fluid", pp.FluidComponent(name="fluid"))
        )
        numerical = cast(
            pp.NumericalConstants,
            constants.get("numerical", pp.NumericalConstants(name="numerical")),
        )

        # Sanity check that users did not pass anything unexpected.
        assert isinstance(solid, pp.SolidConstants), (
            "model.params['material_constants']['fluid'] must be of type "
            + f"{pp.SolidConstants}"
        )
        assert isinstance(fluid, pp.FluidComponent), (
            "model.params['material_constants']['fluid'] must be of type "
            + f"{pp.FluidComponent}"
        )

        # Converting to units of simulation.
        fluid = fluid.to_units(self.units)
        solid = solid.to_units(self.units)
        numerical = numerical.to_units(self.units)

        # Set the solid for the model.
        # NOTE this will change with the generalization of the solid
        self.solid = solid

        # Set the numerical constants for the model.
        self.numerical = numerical

        # Store the fluid component to be accessible by the FluidMixin for creating the
        # default fluid object of the model
        if "material_constants" not in self.params:
            self.params["material_constants"] = {"fluid": fluid}
        else:
            # by logic, params['material_constants'] is ensured to be a dict
            self.params["material_constants"]["fluid"] = fluid
        self.create_fluid()

    def discretize(self) -> None:
        """Discretize all terms."""
        tic = time.time()
        # Do a discretization of the equations. More refined control of the
        # discretization process can be achieved by exploiting knowledge of the equation
        # system (e.g., which terms are linear and need not be discretized at every
        # iteration).
        self.equation_system.discretize()
        logger.info("Discretized in {} seconds".format(time.time() - tic))

    @property
    def nonlinear_discretizations(self) -> list[pp.ad._ad_utils.MergedOperator]:
        """List of nonlinear discretizations in the equation system.

        This list encompasses discretizations other than flux discretizations, such as
        Upwinding. It is crucial that fluxes are updated before Upwinding.

        See also:
            - :meth:`add_nonlinear_diffusive_flux_discretization`
            - :meth:`update_derived_quantities`

        Returns:
            A list of merged operators wrapping underlying discretizations.

        """
        return self._nonlinear_discretizations

    @property
    def nonlinear_diffusive_flux_discretizations(
        self,
    ) -> list[pp.ad._ad_utils.MergedOperator]:
        """List of nonlinear flux discretizations in the equation system.

        Not to be confused with other discretizations (:meth:`nonlinear_discretizations`
        ).

        Individual physics (flow, energy, mechanics) can add respective MPxA
        discretizations, if the second-order tensor is not constant.

        Fluxes are discretized before other discretizations, such as upwinding.

        See also:
            - :meth:`add_nonlinear_diffusive_flux_discretization`
            - :meth:`update_derived_quantities`

        Returns:
            A list of merged operators wrapping underlying discretizations.

        """
        return self._nonlinear_diffusive_flux_discretizations

    def add_nonlinear_discretization(
        self, discretization: pp.ad._ad_utils.MergedOperator
    ) -> None:
        """Add an entry to the list of :meth:`nonlinear_discretizations`.

        Parameters:
            discretization: The nonlinear discretization to be added.

        """
        discr_type = type(discretization._discr)
        admissible_types = [pp.Upwind, pp.UpwindCoupling]
        if discr_type not in admissible_types:
            raise TypeError(
                f"Expecting discretizations of type {admissible_types}."
                f" Got {discr_type}."
            )
        # This guardrail is very weak. However, the discretization list is uniquified
        # before discretization, so it should not be a problem.
        if discretization not in self._nonlinear_discretizations:
            self._nonlinear_discretizations.append(discretization)

    def add_nonlinear_diffusive_flux_discretization(
        self, discretization: pp.ad._ad_utils.MergedOperator
    ) -> None:
        """Add an entry to the list of :meth:`nonlinear_diffusive_flux_discretizations`.

        Parameters:
            discretization: The nonlinear flux discretization to be added.

        """
        discr_type = type(discretization._discr)
        admissible_types = [pp.Mpfa, pp.Tpfa]
        if discr_type not in admissible_types:
            raise TypeError(
                f"Expecting discretizations of type {admissible_types}."
                f" Got {discr_type}."
            )
        # This guardrail is very weak. However, the discretization list is uniquified
        # before discretization, so it should not be a problem.
        if discretization not in self._nonlinear_diffusive_flux_discretizations:
            self._nonlinear_diffusive_flux_discretizations.append(discretization)

    def set_nonlinear_discretizations(self) -> None:
        """Set the list of all nonlinear discretizations.

        This method is called before the discretization is performed. It is intended to
        be used to set the list of nonlinear discretizations.

        See also:
            - :meth:`add_nonlinear_discretization`
            - :meth:`add_nonlinear_diffusive_flux_discretization`

        """

    def before_nonlinear_loop(self) -> None:
        """Method to be called before entering the non-linear solver, thus at the start
        of a new time step.

        The base method does the following:

        1. Update the time step size in :attr:`ad_time_step`.
        2. Reset the nonlinear solver statistics :meth:`~porepy.viz.solver_statistics.
           SolverStatistics.reset`.
        3. Calls :meth:`update_time_dependent_ad_arrays`.
        4. Calls :meth:`update_derived_quantities`.

        """
        # Update time step size.
        self.ad_time_step.set_value(self.time_manager.dt)
        # Empty the log in the statistics object.
        self.nonlinear_solver_statistics.reset()
        # Update the boundary conditions to both the time step and iterate solution.
        self.update_time_dependent_ad_arrays()
        # Update other dependent quantities such as discretizations.
        self.update_derived_quantities()

    def before_nonlinear_iteration(self) -> None:
        """Method to be called at the start of every non-linear iteration.

        The base method only defines the method signature.

        """

    def after_nonlinear_iteration(self, nonlinear_increment: np.ndarray) -> None:
        """Method to be called after every non-linear iteration.

        The base method does the following:

        1. Shift the existing solutions backwards in the iterative sense.
        2. Store the ``nonlinear_increment`` in the current iterate additively.
        3. Calls :meth:`update_derived_quantities`.

        Parameters:
            nonlinear_increment: The new solution, as computed by the non-linear solver.

        """
        self.equation_system.shift_iterate_values(max_index=len(self.iterate_indices))
        self.equation_system.set_variable_values(
            values=nonlinear_increment, additive=True, iterate_index=0
        )
        self.update_derived_quantities()
        self.nonlinear_solver_statistics.num_iteration += 1

    def after_nonlinear_convergence(self) -> None:
        """Method to be called after the non-linear iterations converge.

        The base method does the following:

        1. Shift existing solutions backwards in time.
        2. Saves the current iterate values as the most recent time step values
           (see :meth:`update_solution`).
        3. Flags the model as converged (:attr:`convergence_status`).
        4. Calls :meth:`save_data_time_step`.

        Possible usage is to distribute information on the solution, visualization, etc.

        """
        solution = self.equation_system.get_variable_values(iterate_index=0)

        # Update the time step magnitude if the dynamic scheme is used.
        if not self.time_manager.is_constant:
            self.time_manager.compute_time_step(
                iterations=self.nonlinear_solver_statistics.num_iteration
            )
        self.update_solution(solution)

        self.convergence_status = True
        self.save_data_time_step()

    def update_solution(self, solution: np.ndarray) -> None:
        self.equation_system.shift_time_step_values(
            max_index=len(self.time_step_indices)
        )
        self.equation_system.set_variable_values(
            values=solution, time_step_index=0, additive=False
        )

    def after_nonlinear_failure(self) -> None:
        """Method to be called if the non-linear solver fails to converge."""
        self.save_data_time_step()
        if not self._is_nonlinear_problem():
            raise ValueError("Failed to solve linear system for the linear problem.")

        if self.time_manager.is_constant:
            # We cannot decrease the constant time step.
            raise ValueError("Nonlinear iterations did not converge.")
        else:
            # Update the time step magnitude if the dynamic scheme is used.
            # Note: It will also raise a ValueError if the minimal time step is reached.
            self.time_manager.compute_time_step(recompute_solution=True)

            # Reset the iterate values. This ensures that the initial guess for an
            # unknown time step equals the known time step.
            prev_solution = self.equation_system.get_variable_values(time_step_index=0)
            self.equation_system.set_variable_values(prev_solution, iterate_index=0)

    def after_simulation(self) -> None:
        """Run at the end of simulation. Can be used for cleanup etc."""
        pass

    def check_convergence(
        self,
        nonlinear_increment: np.ndarray,
        residual: Optional[np.ndarray],
        reference_residual: np.ndarray,
        nl_params: dict[str, Any],
    ) -> tuple[bool, bool]:
        """Implements a convergence check, to be called by a non-linear solver.

        Parameters:
            nonlinear_increment: Newly obtained solution increment vector
            residual: Residual vector of non-linear system, evaluated at the newly
                obtained solution vector. Potentially None, if not needed.
            reference_residual: Reference residual vector of non-linear system,
                evaluated for the initial guess at current time step.
            nl_params: Dictionary of parameters used for the convergence check.
                Which items are required will depend on the convergence test to be
                implemented.

        Returns:
            The method returns the following tuple:

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
                return False, True

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
            converged_inc = (
                nl_params["nl_convergence_tol"] is np.inf
                or nonlinear_increment_norm < nl_params["nl_convergence_tol"]
            )
            converged_res = (
                nl_params["nl_convergence_tol_res"] is np.inf
                or residual_norm < nl_params["nl_convergence_tol_res"]
            )
            converged = converged_inc and converged_res
            diverged = False

        # Log the errors (here increments and residuals)
        self.nonlinear_solver_statistics.log_error(
            nonlinear_increment_norm, residual_norm
        )

        return converged, diverged

    def compute_residual_norm(
        self, residual: Optional[np.ndarray], reference_residual: np.ndarray
    ) -> float:
        """Compute the residual norm for a nonlinear iteration.

        Parameters:
            residual: Residual of current iteration.
            reference_residual: Reference residual value (initial residual expected),
                allowing for defining relative criteria.

        Returns:
            float: Residual norm; np.nan if the residual is None.

        """
        if residual is None:
            return np.nan
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

        If ``params['apply_schur_complement_reduction']`` is True, the
        :meth:`schur_complement_primary_variables` and
        :meth:`schur_complement_primary_equations` are used to perform a Schur
        complement technique.

        To invert the secondary block, :meth:`~porepy.numerics.ad.equation_system.
        EquationSystem.default_schur_complement_inverter` is used by default.
        This inverter assumes the secondary equations to consist of non-overlapping
        blocks (local equations, block-diagonal matrix).
        The user can provide a custom inverter
        ``model.params['schur_complement_inverter']``, which is a callable taking a
        sparse matrix and returning the inverse (sparse) matrix.

        See Also:

            - :meth:`~porepy.numerics.ad.equation_system.EquationSystem.
              assemble_schur_complement_system`
            - :meth:`~porepy.numerics.ad.equation_system.EquationSystem.assemble`

        """
        t_0 = time.time()

        if self._apply_schur_complement_reduction():
            assert self.schur_complement_primary_variables, (
                "Primary column block for Schur technique not found."
            )
            assert self.schur_complement_primary_equations, (
                "Primary row block for Schur technique not defined."
            )
            self.linear_system = self.equation_system.assemble_schur_complement_system(
                self.schur_complement_primary_equations,
                self.schur_complement_primary_variables,
                inverter=cast(
                    Callable[[sps.spmatrix], sps.spmatrix],
                    self.params.get("schur_complement_inverter", None),
                ),
            )
        else:
            self.linear_system = self.equation_system.assemble()

        t_1 = time.time()
        logger.debug(f"Assembled linear system in {t_1 - t_0:.2e} seconds.")

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

        x = np.atleast_1d(x)
        if self._apply_schur_complement_reduction():
            x = self.equation_system.expand_schur_complement_solution(x)

        logger.info(f"Solved linear system in {time.time() - t_0:.2e} seconds.")
        return x

    def _apply_schur_complement_reduction(self) -> bool:
        """Returns the model parameter on whether the linear system should be reduced
        via Schur complement using the defined primary and secondary equations and
        variables.

        Can be set via ``model.params['apply_schur_complement_reduction'].
        Returns False by default.

        """
        return bool(self.params.get("apply_schur_complement_reduction", False))

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

    def _is_reference_phase_eliminated(self) -> bool:
        """Returns True if ``params['eliminate_reference_phase'] == True`.
        Defaults to True."""
        return bool(self.params.get("eliminate_reference_phase", True))

    def _is_reference_component_eliminated(self) -> bool:
        """Returns True if ``params['eliminate_reference_component'] == True`.
        Defaults to True."""
        return bool(self.params.get("eliminate_reference_component", True))

    def update_time_dependent_ad_arrays(self) -> None:
        """Update the time dependent arrays before a new time step.

        The base implementation updates those for the boundary condition values.
        Override it to update other model-specific time dependent arrays.

        """
        self.update_all_boundary_conditions()

    def update_derived_quantities(self) -> None:
        """Performs an update of derived and secondary quantities entering the
        equations.

        These updates include flux values and discretization matrices, or surrogate
        operators which wrap externalized computations. In principle, anything not part
        of the evaluation process in the AD framework, can be put here if it requires
        an update for the evaluation to lead to correct values.

        The base method performs the following updates:

        1. Update material properties (if necessary) based on the current state
           (see :meth:`update_material_properties`).
        2. Update discretization parameters, most crucially those entering the flux
           discretization (see :meth:`update_discretization_parameters`).
        3. Rediscretize the non-linear fluxes depending on above tensors
           (see :meth:`rediscretize_fluxes`).
        4. Evaluate and store fluxes for upstream discretizations
           (see :meth:`update_flux_values`).
        5. Rediscretize upstream (and possibly other) discretizations
           (see :meth:`rediscretize`).

        For a consistent evaluation of the system, this method is called in
        :meth:`after_nonlinear_iteration` (after the global state vector changes) and in
        :meth:`before_nonlinear_loop` (after the boundary conditions and other
        time-dependent quantities change).

        """
        self.update_material_properties()
        self.update_discretization_parameters()
        self.rediscretize_fluxes()
        self.update_flux_values()
        self.rediscretize()

    def update_material_properties(self) -> None:
        """Method for updating fluid and solid properties, which are not taken care of
        by the AD framework (external calculations and surrogate operators).

        The base method only defines the signature and individual physics model have to
        override this method. A super-call to trigger other physics' update is required.

        """

    def update_discretization_parameters(self) -> None:
        """Method for evaluating and storing discretization parameters required for
        discretizing fluxes and other discretizations.

        This primarily involves second order tensors such as permeability and thermal
        conductivity.

        The base method only defines the signature and individual physics model have to
        override this method. A super-call to trigger other physics' update is required.

        """

    def rediscretize_fluxes(self) -> None:
        """Discretize nonlinear fluxes."""
        tic = time.time()
        # Uniquify to save computational time, then discretize.
        unique_discr = pp.ad._ad_utils.uniquify_discretization_list(
            self.nonlinear_diffusive_flux_discretizations
        )
        pp.ad._ad_utils.discretize_from_list(unique_discr, self.mdg)
        logger.debug(f"Re-discretized nonlinear fluxes in {time.time() - tic} seconds.")

    def update_flux_values(self) -> None:
        """Method for updating and storing flux values, to be used for a subsequent
        discretization of upstream values.

        The base method only defines the signature and individual physics model have to
        override this method. A super-call to trigger other physics' update is required.

        """

    def rediscretize(self) -> None:
        """Discretize nonlinear terms."""
        tic = time.time()
        # Uniquify to save computational time, then discretize.
        unique_discr = pp.ad._ad_utils.uniquify_discretization_list(
            self.nonlinear_discretizations
        )
        pp.ad._ad_utils.discretize_from_list(unique_discr, self.mdg)
        logger.debug(f"Re-discretized nonlinear terms in {time.time() - tic} seconds.")

    def darcy_flux_storage_keywords(self) -> list[str]:
        """Return the keywords for which the Darcy flux values are stored.

        Returns:
            List of keywords for the Darcy flux values.

        """
        return []


class ContactIndicators(pp.PorePyModel):
    """Class for computing contact indicators used for tailored line search.

    This functionality is experimental and may be subject to change.

    The class is a mixin for the solution strategy classes for models with contact
    mechanics. The class provides methods for computing the opening and sliding
    indicators, which are used for tailored line search as defined in the class
    :class:`~porepy.numerics.nonlinear.line_search.ConstraintLineSearch`.

    By specifying the parameter `adaptive_indicator_scaling` in the model
    parameters, the indicators can be scaled adaptively by the characteristic
    fracture traction estimate based on the most recent iteration value.

    """

    contact_traction: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Contact traction operator."""

    contact_mechanics_numerical_constant: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Contact mechanics numerical constant."""

    fracture_gap: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Fracture gap operator."""

    friction_bound: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Friction bound operator."""

    def opening_indicator(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Function describing the state of the opening constraint.

        The function is a linear combination of the two arguments of the max function of
        the normal fracture deformation equation. Arbitrary sign convention: Negative
        for open fractures, positive for closed ones.

        The parameter `adaptive_indicator_scaling` scales the indicator by the contact
        traction estimate.

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            opening_indicator: Opening indicator operator.

        """
        nd_vec_to_normal = self.normal_component(subdomains)
        # The normal component of the contact traction and the displacement jump.
        t_n: pp.ad.Operator = nd_vec_to_normal @ self.contact_traction(subdomains)
        u_n: pp.ad.Operator = nd_vec_to_normal @ self.displacement_jump(subdomains)
        c_num = self.contact_mechanics_numerical_constant(subdomains)
        max_arg_1 = pp.ad.Scalar(-1.0) * t_n
        max_arg_2 = c_num * (u_n - self.fracture_gap(subdomains))
        ind = max_arg_1 - max_arg_2
        if self.params.get("adaptive_indicator_scaling", False):
            # Scale adaptively based on the contact traction estimate.
            # Base variable values from all fracture subdomains.
            all_subdomains = self.mdg.subdomains(dim=self.nd - 1)
            scale_op = self.contact_traction_estimate(all_subdomains)
            scale = self.compute_traction_norm(
                cast(np.ndarray, self.equation_system.evaluate(scale_op))
            )
            ind = ind / pp.ad.Scalar(scale)
        return ind

    def sliding_indicator(
        self,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """Function describing the state of the sliding constraint.

        The function is a linear combination of the two arguments of the max function of
        the tangential fracture deformation equation. Sign convention: Negative for
        sticking, positive for sliding:  ||T_t+c_t u_t||-b_p

        The parameter `adaptive_indicator_scaling` scales the indicator by the contact
        traction estimate.

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            sliding_indicator: Sliding indicator operator.

        """

        # Basis vector combinations
        num_cells = sum([sd.num_cells for sd in subdomains])
        # Mapping from a full vector to the tangential component
        nd_vec_to_tangential = self.tangential_component(subdomains)

        tangential_basis = self.basis(subdomains, dim=self.nd - 1)

        # Variables: The tangential component of the contact traction and the
        # displacement jump
        t_t: pp.ad.Operator = nd_vec_to_tangential @ self.contact_traction(subdomains)
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump(subdomains)
        # The time increment of the tangential displacement jump
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)
        zeros_frac = pp.ad.DenseArray(np.zeros(num_cells))

        f_max = pp.ad.Function(pp.ad.maximum, "max_function")
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")
        # Heaviside function. The 0 as the second argument to partial() implies
        # f_heaviside(0)=0, a choice that is not expected to affect the result in this
        # context.
        f_heaviside = pp.ad.Function(partial(pp.ad.heaviside, 0), "heaviside_function")

        c_num_as_scalar = self.contact_mechanics_numerical_constant(subdomains)

        basis_sum = pp.ad.sum_projection_list(tangential_basis)
        tangential_sum = t_t + (basis_sum @ c_num_as_scalar) * u_t_increment

        max_arg_1 = f_norm(tangential_sum)
        max_arg_1.set_name("norm_tangential")

        max_arg_2 = f_max(self.friction_bound(subdomains), zeros_frac)
        max_arg_2.set_name("b_p")

        h_oi = f_heaviside(self.opening_indicator(subdomains))
        ind = max_arg_1 - max_arg_2

        if self.params.get("adaptive_indicator_scaling", False):
            # Base on all fracture subdomains
            all_subdomains = self.mdg.subdomains(dim=self.nd - 1)
            scale_op = self.contact_traction_estimate(all_subdomains)
            scale = self.compute_traction_norm(
                cast(np.ndarray, self.equation_system.evaluate(scale_op))
            )
            ind = ind / pp.ad.Scalar(scale)
        return ind * h_oi

    def contact_traction_estimate(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Estimate the magnitude of contact traction.

        Parameters:
            subdomains: List of subdomains where the contact traction is defined.

        Returns:
            Characteristic fracture traction estimate.

        """
        t: pp.ad.Operator = self.contact_traction(subdomains)
        e_n = self.e_i(subdomains, dim=self.nd, i=self.nd - 1)

        u = self.displacement_jump(subdomains) - e_n @ self.fracture_gap(subdomains)
        c_num = self.contact_mechanics_numerical_constant(subdomains)
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd), "norm_function")
        return f_norm(t) + f_norm(c_num * u)

    def compute_traction_norm(self, val: np.ndarray) -> float:
        """Compute a norm of the traction estimate from the vector-valued traction.

        The scalar traction is computed as the p norm of the traction vector.

        Parameters:
            val: Vector-valued traction.

        Returns:
            Scalar traction.

        """
        val = val.clip(1e-8, 1e8)
        p = self.params.get("traction_estimate_p_mean", 5.0)
        p_mean = np.mean(val**p, axis=0) ** (1 / p)
        return float(p_mean)
