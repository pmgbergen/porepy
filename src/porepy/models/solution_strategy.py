"""Solution strategy classes.

This class is a modified version of relevant parts of AbstractModel.
In the future, it may be possible to merge the two classes. For now, we
keep them separate, to avoid breaking existing code (legacy models).
"""

from __future__ import annotations

import logging
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Never, Optional, cast
from warnings import warn

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics import solvers
from porepy.viz.solver_statistics import SolverStatisticsFactory

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

        # Print deprecation warning for an old parameter.
        if "linear_solver" in params:
            warn(
                "Linear solver was moved outside the PorePy model. If you previously "
                "passed 'linear_solver' backend string (e.g. 'pypardiso') in model "
                "params, replace it with "
                "pp.solvers.NewtonSolver(linear_solver=pp.LinearSolverDirect(backend="
                "'pypardiso')). The current passed value is ignored.",
                category=DeprecationWarning,
                stacklevel=2,
            )

        # Set default parameters, these will be overwritten by any parameters passed.
        default_params = {
            "folder_name": "visualization",
            "file_name": "data",
        }

        default_params.update(params)
        self.params = default_params
        """Dictionary of parameters."""

        """Whether the non-linear iteration has converged."""
        self._nonlinear_discretizations: list[pp.ad.MergedOperator] = []
        """See :meth:`add_nonlinear_discretization`."""
        self._nonlinear_diffusive_flux_discretizations: list[pp.ad.MergedOperator] = []
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
        self.time_data = pp.time_stepper.SimulationTimeData(
            time=0.0,
            dt=1.0,
            time_index_successful=0,
            schedule=np.array([0.0, 1.0]),
            constant_dt=True,
        )
        if "time_manager" in params:
            warn(message="", category=FutureWarning, stacklevel=2)
            time_manager = params["time_manager"]
            self.time_data = pp.time_stepper.SimulationTimeData(
                time=time_manager.schedule[0],
                dt=time_manager.dt_init,
                time_index_successful=0,
                schedule=time_manager.schedule,
                constant_dt=time_manager.is_constant,
            )

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
                # :class:`~porepy.time_stepper.time_step_control.TimeManager.
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

    @property
    def time_manager(self) -> pp.TimeManager:
        warn(message="", category=FutureWarning, stacklevel=2)
        time_manager = pp.TimeManager(
            schedule=self.time_data.schedule,
            dt_init=self.time_data.dt,
            time_index=self.time_data.time_index_successful,
            constant_dt=self.time_data.constant_dt,
        )
        time_manager._time = self.time_data.time
        return time_manager

    @time_manager.setter
    def time_manager(self, time_manager: pp.TimeManager) -> None:
        raise ValueError(
            "model.time_manager is deprecated. Please set ModelRunner(time_stepper="
            "TimeStepper(scheduler=assemble_default_time_scheduler(...)) instead."
        )

    def prepare_simulation(self) -> None:
        """Run at the start of simulation. Used for initialization etc."""
        # Set the material and geometry of the problem. The geometry method must be
        # implemented in a ModelGeometry class.
        self.set_materials()
        self.set_geometry()

        # Exporter initialization must be done after grid creation,
        # but prior to data initialization.
        self.set_nonlinear_solver_statistics()
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
        self.initialize_operator_reference_values_from_initial_state()
        self.initialize_previous_iterate_and_time_step_values()

        # Initialize time dependent ad arrays, including those for boundary values.
        self.update_time_dependent_ad_arrays()
        self.reset_state_from_file()
        self.set_equations()

        self.update_discretization_parameters()
        self.discretize()
        self.set_nonlinear_discretizations()

        # Export initial condition (only if time-dependent).
        if self._is_time_dependent():
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

    def initialize_operator_reference_values_from_initial_state(self) -> None:
        """Initialize AD operator reference values from iterate-0 state.

        This compatibility step aligns linearization references used by
        :meth:`porepy.numerics.ad.operators.Operator.perturbation_from_reference`
        with initialized primary-variable values for pressure and temperature.

        The behavior can be disabled by setting
        ``params['initialize_operator_reference_from_initial_values'] = False``.

        NOTE: This method is intended as a temporary bridge from PR #1696 until
        downstream PRs on initialization have been merged.

        """
        if not self.params.get(
            "initialize_operator_reference_from_initial_values", True
        ):
            return

        for quantity_name in ("pressure", "temperature"):
            variable_attr = f"{quantity_name}_variable"
            if not hasattr(self, variable_attr):
                continue

            reference_value = cast(
                pp.number, getattr(self.reference_variable_values, quantity_name, 0.0)
            )
            if np.isclose(reference_value, 0.0):
                continue

            variable_name = getattr(self, variable_attr)
            domains = cast(list[pp.GridLike], self.mdg.subdomains())
            variables = self.equation_system.get_variables([variable_name], domains)
            if len(variables) == 0:
                continue

            values = self.equation_system.get_variable_values(
                variables=variables, iterate_index=0
            )
            self.equation_system.set_variable_values(
                np.full_like(values, reference_value),
                variables=variables,
                reference=True,
            )

    def set_equation_system_manager(self) -> None:
        """Create an equation_system manager on the mixed-dimensional grid."""
        if not hasattr(self, "equation_system"):
            self.equation_system = pp.ad.EquationSystem(self.mdg)

    def set_nonlinear_solver_statistics(self) -> None:
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
        statistics = self.params.get(
            "nonlinear_solver_statistics",
            SolverStatisticsFactory.create_statistics_type(
                nonlinear=self._is_nonlinear_problem(),
                time_dependent=self._is_time_dependent(),
            ),
        )
        # Explicitly check if the retrieved value is a class and a subclass of
        # pp.SolverStatistics for type checking.
        if isinstance(statistics, type) and issubclass(statistics, pp.SolverStatistics):
            self.nonlinear_solver_statistics = statistics(
                path=cast(Path, self.params.get("solver_statistics_file_name"))
            )
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

    def before_time_step(self) -> None:
        """Called from the outside of the model at the start of each time step.

        The model must prepare its state for the target simulation time ``t``,
        available through :attr:`time_manager`. The nonlinear solver then attempts
        to advance the discretized problem to that time. If the solve fails, it may
        be retried with a different time-step size, so this method may be called
        multiple times with different target times.

        The base method does the following:

        1. Update :attr:`ad_time_step` with the current time step size.
        2. Call :meth:`update_time_dependent_ad_arrays` to update BC values and other
           time-dependent operators.
        3. Call :meth:`update_derived_quantities` to update them based on the
           time-dependent values.

        """
        self.ad_time_step.set_value(self.time_data.dt)
        self.initialize_nonlinear_solution()
        self.update_time_dependent_ad_arrays()
        self.update_derived_quantities()

    def before_nonlinear_loop(self) -> None:
        """Called before entering a nonlinear solver loop.

        With the default ``NewtonSolver``, each time step has a single nonlinear
        loop so this method is called once after ``before_time_step``. More advanced
        solvers may use multiple nonlinear loops per time step and call this method
        before each one.

        The base method does the following:

        1. Increases the statistics index.

        """
        self.nonlinear_solver_statistics.increase_index()

    def before_nonlinear_iteration(self) -> None:
        """Method to be called at the start of every non-linear iteration.

        The base method only defines the method signature.

        """

    def after_nonlinear_iteration(
        self,
        nonlinear_increment: np.ndarray,
        updated_variables: Optional[list[pp.ad.Variable]] = None,
    ) -> None:
        """Method to be called after every non-linear iteration.

        The base method does the following:

        1. Shift the existing solutions backwards in the iterative sense.
        2. Store the ``nonlinear_increment`` in the current iterate additively.
        3. Calls :meth:`update_derived_quantities`.

        Parameters:
            nonlinear_increment: The new solution, as computed by the non-linear solver.
            updated_variables: Variables to update with `nonlinear_increment`. If
                `None`, all variables are updated.

        """
        self.equation_system.shift_iterate_values(max_index=len(self.iterate_indices))
        self.equation_system.set_variable_values(
            values=nonlinear_increment,
            variables=updated_variables,
            additive=True,
            iterate_index=0,
        )
        self.update_derived_quantities()

    def after_nonlinear_convergence(self) -> None:
        """Called after a nonlinear solver loop converges.

        With the default ``NewtonSolver``, each time step has a single nonlinear
        loop, so this method is called exactly once, before
        ``after_time_step_convergence``. More advanced solvers may use multiple
        nonlinear loops per time step and call this method after each converged
        loop.

        Use this method for loop-specific post-processing.

        """

    def after_nonlinear_failure(self) -> None:
        """Called after a nonlinear solver loop fails to converge.

        With the default ``NewtonSolver``, each time step has a single nonlinear
        loop, so this method is called exactly once, before
        ``after_time_step_failure``. More advanced solvers may use multiple
        nonlinear loops per time step and call this method after each failed loop.

        """

    def after_time_step_convergence(self) -> None:
        """Called after a new time step solution has been achieved.

        The base method does the following:

        1. Call :meth:`update_time_step_solution`.
        2. Call :meth:`save_data_time_step`.
        3. Call :meth:`save_statistics`.

        """
        self.update_time_step_solution()
        self.save_data_time_step()
        self.save_statistics()

    def after_time_step_failure(self) -> None:
        """Called after a time step has failed to converge.

        The base method reverts the trial time step being executed.
        It also calls :meth:`save_statistics`.

        """
        self.revert_trial_time_step_solution()
        self.save_statistics()

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

    def set_materials(self) -> None:
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
            self.params["material_constants"] = {"fluid": fluid}  # type:ignore[assignment]
        else:
            # by logic, params['material_constants'] is ensured to be a dict
            self.params["material_constants"]["fluid"] = fluid  # type:ignore[index]
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
    def nonlinear_discretizations(self) -> list[pp.ad.MergedOperator]:
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
    ) -> list[pp.ad.MergedOperator]:
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
        self, discretization: pp.ad.MergedOperator
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

    def remove_nonlinear_discretization(
        self, discretization: pp.ad.MergedOperator
    ) -> bool:
        """Remove an entry from the list of :meth:`nonlinear_discretizations`.

        Parameters:
            discretization: The nonlinear discretization to be removed.

        Returns:
            True if the discretization was found and removed, False otherwise.
        """
        if discretization in self._nonlinear_discretizations:
            self._nonlinear_discretizations.remove(discretization)
            return True
        else:
            return False

    def add_nonlinear_diffusive_flux_discretization(
        self, discretization: pp.ad.MergedOperator
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

    def remove_nonlinear_diffusive_flux_discretization(
        self, discretization: pp.ad.MergedOperator
    ) -> bool:
        """Remove an entry from the list of
        :meth:`nonlinear_diffusive_flux_discretizations`.

        Parameters:
            discretization: The nonlinear flux discretization to be removed.

        Returns:
            True if the discretization was found and removed, False otherwise.
        """
        if discretization in self._nonlinear_diffusive_flux_discretizations:
            self._nonlinear_diffusive_flux_discretizations.remove(discretization)
            return True
        else:
            return False

    def set_nonlinear_discretizations(self) -> None:
        """Set the list of all nonlinear discretizations.

        This method is called before the discretization is performed. It is intended to
        be used to set the list of nonlinear discretizations.

        See also:
            - :meth:`add_nonlinear_discretization`
            - :meth:`add_nonlinear_diffusive_flux_discretization`

        """

    def initialize_nonlinear_solution(self) -> None:
        """Set the previous time step solution as the initial guess for the nonlinear
        solver.

        """
        prev_solution = self.equation_system.get_variable_values(time_step_index=0)
        self.equation_system.set_variable_values(prev_solution, iterate_index=0)

    def update_time_step_solution(self) -> None:
        """Shifts the solution per time step index and sets the provided solution
        as the recent time step solution.

        Parameters:
            solution: Global, accepted solution vector.

        """
        self.equation_system.shift_time_step_values(
            max_index=len(self.time_step_indices)
        )
        solution = self.equation_system.get_variable_values(iterate_index=0)
        self.equation_system.set_variable_values(
            values=solution, time_step_index=0, additive=False
        )

    def revert_trial_time_step_solution(self) -> None:
        """Revert the solution to the previous time step solution.

        This method is intended to be used in the case of a failed time step, where the
        trial solution should be reverted to the last known good solution. All iterate
        indices are updated to the previous time step solution, which is stored at time
        step index 0. I.e., we assume time step solution has *not* been shifted yet.

        """
        prev_solution = self.equation_system.get_variable_values(time_step_index=0)
        for ind in self.iterate_indices:
            self.equation_system.set_variable_values(prev_solution, iterate_index=ind)

    def after_simulation(self) -> None:
        """Run at the end of simulation. Can be used for cleanup etc."""
        self.save_statistics()

    def assemble_linear_system(self) -> solvers.LinearSystem:
        """Assemble the linearized system.

        The linear system is defined by the current state of the model.

        See Also:
            - :meth:`~porepy.numerics.ad.equation_system.EquationSystem.assemble`

        Returns:
            The assembled matrix and right-hand side vector.

        """
        warn(
            "The method model.assemble_linear_system is deprecated and will be removed."
            " Use model.equation_system.assemble instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        t_0 = time.time()

        linear_system = self.equation_system.assemble()

        t_1 = time.time()
        logger.debug(f"Assembled linear system in {t_1 - t_0:.2e} seconds.")
        return linear_system

    def solve_linear_system(self) -> Never:
        raise AttributeError(
            "Linear solver was moved outside the PorePy model. If you override this "
            "function, provide a custom linear solver to the nonlinear solver, e.g.: "
            "pp.NewtonSolver(linear_solver=CustomLinearSolver())"
        )

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
        3. Rediscretize the nonlinear fluxes depending on above tensors
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
        unique_discr = pp.ad.uniquify_discretization_list(
            self.nonlinear_diffusive_flux_discretizations
        )
        pp.ad.discretize_from_list(unique_discr, self.mdg)
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
        unique_discr = pp.ad.uniquify_discretization_list(
            self.nonlinear_discretizations
        )
        pp.ad.discretize_from_list(unique_discr, self.mdg)
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
