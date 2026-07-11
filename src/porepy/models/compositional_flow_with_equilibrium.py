"""Extensions of model mixins for compositional flow (CF) to acount for local
equilibrium (LE) equations and and the usage of a flash instance.

The most important difference is, that in this fully, thermodynamically consistent
formulation of the compositional flow problem, there are no dangling variables.
No separate constitutive modelling is required because thermodynamic properties and
secondary variables are fully determined by the result of the local equilibrium problem.

Also, equilibrium calculations (in the unified setting) introduce extended fractions.
Partial fractions become dependent operators (normalization of extended fractions).

Provides a fully formulated CF model with local equilibrium equations formulated as
a p-h flash, where phase properties are represented using surrogate operators and their
properties are obtained by the flash.

"""

from __future__ import annotations

import logging
from functools import cached_property, partial
from typing import TYPE_CHECKING, Callable, Optional, Sequence, cast

import numpy as np

import porepy as pp
import porepy.compositional as pc

from . import compositional_flow as cf
from .persistent_variable_equilibrium import PersistentEquilibriumEquations

# NOTE: Avoid actual import and triggering of compilation. We only need this for type
# checking
if TYPE_CHECKING:
    from numpy.typing import NDArray

    from porepy.compositional.flash.abstract_flash import (
        AbstractFlash,
        FlashResults,
        StateSpecDict,
    )


logger = logging.getLogger(__name__)


class BoundaryConditionsEquilibrium(cf.BoundaryConditionsPhaseProperties):
    """BC mixin for CF models with equilibrium and flash instance.

    This class uses the flash instance to provide BC values for secondary variables
    and thermodynamic properties of phases, using BC values for pressure, temperature
    and overall fractions of components.

    If the BC are not constant, the user needs to flag this in the model parameters and
    this class will perform the boundary flash in every time step to update respective
    values.

    Note:
        As of now, the flash is only performed on the matrix boundary.

        Also, BC values for thermodynamic state functions are not stored!
        That includes pressure, temperature, energies and volume. Only fractional
        unknowns on the boundary are implemented using methods with name
        ``bc_values_*``. You must override ``bc_values_enthalpy`` for example and return
        the values stored in :meth:`boundary_equilibrium_results` if you want to utilize
        the results of the boundary equilibrium.

    Supports the following model parameters:

    - ``'has_time_dependent_boundary_values'``: Defaults to False.
      A bool indicating whether Dirichlet BC for pressure, temperature or
      feed fractions are time-dependent.

      If True, the boundary equilibrium will be re-computed at the beginning of every
      time step.

    """

    flash: AbstractFlash

    bc_values_pressure: Callable[[pp.BoundaryGrid], np.ndarray]
    bc_values_temperature: Callable[[pp.BoundaryGrid], np.ndarray]
    bc_values_overall_fraction: Callable[[pp.Component, pp.BoundaryGrid], np.ndarray]

    # Provided by CompositionalVariablesMixin
    has_independent_fraction: Callable[[pp.Component], bool]
    has_independent_saturation: Callable[[pp.Phase], bool]
    has_independent_partial_fraction: Callable[[pp.Component, pp.Phase], bool]
    has_independent_extended_fraction: Callable[[pp.Component, pp.Phase], bool]
    _saturation_variable: Callable[[pp.Phase], str]
    _partial_fraction_variable: Callable[[pp.Component, pp.Phase], str]

    @property
    def _boundary_equilibrium_required(self) -> bool:
        """Internally used flag triggering the boundary flash during prepare simulation
        and in the course of simulations, if BC values are time-dependent."""

        start_of_simulation: bool = (
            self.time_manager.time_init == self.time_manager.time
        )

        # NOTE This will stop working if for some reason equations are suddenly set
        # before the state is set from the files.
        simulation_restarted: bool = self.restart_options.get(
            "restart", False
        ) and not bool(self.equation_system.equations)

        if (
            start_of_simulation
            or simulation_restarted
            or self.params.get("has_time_dependent_boundary_values", False)
        ):
            return True
        else:
            return False

    @cached_property
    def boundary_equilibrium_results(
        self,
    ) -> dict[pp.BoundaryGrid, pc.FluidProperties]:
        """The results of the boundary flash are stored here (per boundary grid) for
        further processing."""
        return {}

    def update_boundary_values_phase_properties(self) -> None:
        """Instead of performing the update using underlying EoS, a flash is performed
        to compute the updates for phase properties, as well as for (extended) partial
        fractions and saturations.

        Calls :meth:`boundary_equilibrium` for all boundary grids, using p-T
        equilibrium conditions.

        """

        for sd in self.mdg.subdomains():
            bg = self.mdg.subdomain_to_boundary_grid(sd)
            if bg is not None:
                p = self.bc_values_pressure(bg)
                T = self.bc_values_temperature(bg)
                self.boundary_equilibrium(sd, {"p": p, "T": T})

    def boundary_equilibrium(
        self, sd: pp.Grid, equilibrium_specs: StateSpecDict
    ) -> None:
        """This method performs the p-T flash on the boundary of the given grid.

        The results are stored in :meth:`boundary_equilibrium_results`.

        The method can be called any time once the model is initialized, especially for
        non-constant BC.

        Note:
            :meth:`bc_type_equilibrium` is used as a mask for cells, where the flash is
            not to be applied. For boundary faces not tagged as ``'dir'``, trivial
            values are stored.

        Parameters:
            sd: A grid on whose boundary the flash should be performed.
            equilibrium_specs: Definition of the equilibrium condition in terms of
                state functions and their values.

                See also :meth:`~porepy.compositional.flash.abstract_flash.
                AbstractFlash.flash`.

        Raises:
            ValueError: If the flash did not succeed everywhere.

        """

        # DO nothing if not at beginning of simulation or BC not time dependent.
        if not self._boundary_equilibrium_required:
            return

        bg = self.mdg.subdomain_to_boundary_grid(sd)
        assert bg is not None, "Boundary grid of given subdomain not found."

        # Boundary faces flagged as dir are used for bc flash.
        flash_idx = self._boundary_equilibrium_cells(bg)

        # Define by default trivial values so that the system can be evaluated.
        # On cells not flagged for flash, add some eps to avoid division by zero.
        bg_state = self._default_boundary_state(bg, ~flash_idx)

        # Perform flash on tagged faces and prolong solution to whole boundary.
        if np.any(flash_idx):
            # The bc_values method is only called for independent components.
            feed = [
                self.bc_values_overall_fraction(comp, bg)[flash_idx]
                for comp in self.fluid.components
                if self.has_independent_fraction(comp)
            ]
            z_r = 1.0 - pc.safe_sum(feed)
            feed = (
                feed[: self.fluid.reference_component_index]
                + [z_r]
                + feed[self.fluid.reference_component_index :]
            )
            for k, v in equilibrium_specs.items():
                equilibrium_specs[k] = v[flash_idx]  # type:ignore

            # Performing flash, asserting everything is successful.
            logger.info(
                f"Equilibration on boundary {bg.id} at t={self.time_manager.time:.3e}."
            )
            state = self.flash.flash(
                equilibrium_specs,
                feed,
                params=self.params.get("flash_params", None),
            )

            if not np.all(state.converged):
                raise ValueError(f"Boundary flash not successful on boundary {bg.id}")

            # Prolong solution.
            bg_state.p[flash_idx] = state.p
            bg_state.T[flash_idx] = state.T
            bg_state.h[flash_idx] = state.h
            bg_state.u[flash_idx] = state.u
            bg_state.rho[flash_idx] = state.rho
            bg_state.y[:, flash_idx] = state.y
            bg_state.sat[:, flash_idx] = state.sat

            for j in range(self.fluid.num_phases):
                bg_state.phases[j].h[flash_idx] = state.phases[j].h
                bg_state.phases[j].u[flash_idx] = state.phases[j].u
                bg_state.phases[j].rho[flash_idx] = state.phases[j].rho
                bg_state.phases[j].mu[flash_idx] = state.phases[j].mu
                bg_state.phases[j].kappa[flash_idx] = state.phases[j].kappa
                bg_state.phases[j].phis[:, flash_idx] = state.phases[j].phis
                bg_state.phases[j].x[:, flash_idx] = state.phases[j].x

        self.boundary_equilibrium_results[bg] = bg_state

    def bc_type_equilibrium(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Method for defining boundary faces on which to perform the flash for boundary
        conditions.

        Faces on the boundary tagged as ``'dir'`` are used to evaluate the target state
        and perform the flash.

        Note:
            The user must ensure that propper pressure, temperature and overall fraction
            values are defined on respectie faces.

        Parameters:
            sd: A subdomain in the md-grid.

        Returns:
            A boundary conditions object. By default fall faces are tagged as ``'dir'``
            and the flash is performed everywhere on the boundary.

        """
        # Define boundary faces.
        boundary_faces = self.domain_boundary_sides(sd).all_bf
        # Define boundary condition on all boundary faces.
        return pp.BoundaryCondition(sd, boundary_faces, "dir")

    def _boundary_equilibrium_cells(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Returns a boolean array indicating which cells of the boundary grid are
        flagged for performing equilibrium calculations.

        The flags are primary used to not perform the calculations, where not required
        and to avoid division by zero errors, where not required.

        Parameters:
            bg: A boundary grid.

        Returns:
            A boolean array with ``shape=(bg.num_cells,)``.

        """
        return self.bc_type_equilibrium(bg.parent).is_dir[
            self.domain_boundary_sides(bg.parent).all_bf
        ]

    def _default_boundary_state(
        self, bg: pp.BoundaryGrid, cell_idx: np.ndarray, eps: float = 1e-10
    ) -> pc.FluidProperties:
        """Returns a fluid property instance with trivial values for a given boundary.

        Adds a small ``eps`` to viscosity and partial fractions to avoid a division by
        zero when evaluating mobility terms, and the propagation of nan into the system.

        These ``eps`` are cancled out by zero density values.

        Parameters:
            bg: A Boundary grid.
            cell_idx: Boolean array which cell values should be augmented with ``eps``.
            eps: ``default=1e-10``

                Close-to-zero value for viscosity and partial fractions.

        Returns:
            An almost trivial fluid property structure.

        """
        ncomp = len(self.fluid.components)
        nphase = len(self.fluid.phases)
        phase_states = [phase.state for phase in self.fluid.phases]
        n = bg.num_cells

        bg_state = pc.initialize_fluid_properties(
            n,
            ncomp,
            nphase,
            phase_states,
            with_derivatives=False,  # No diffs on bg.
        )

        for j in range(nphase):
            bg_state.phases[j].mu[cell_idx] = eps
            bg_state.phases[j].x[:, cell_idx] = eps

        return bg_state

    def update_all_boundary_conditions(self):
        """Updates BC values of phase properties (surrogate operators) and secondary
        variables appearing in the non-linear weights on the boundary.

        The update is performed using the results of the BC flash.

        """
        super().update_all_boundary_conditions()

        if cf.is_fractional_flow(self):
            return

        for phase in self.fluid.phases:
            self._update_phase_properties_on_boundaries(phase)

            # Updating values of saturations of independent phases.
            if self.has_independent_saturation(phase):
                bc_values_saturation = cast(
                    Callable[[pp.BoundaryGrid], np.ndarray],
                    partial(self.bc_values_saturation, phase),
                )
                self.update_boundary_condition(
                    self._saturation_variable(phase),
                    bc_values_saturation,
                )

            for component in phase:
                if self.has_independent_extended_fraction(
                    component, phase
                ) or self.has_independent_partial_fraction(component, phase):
                    bc_values_partial_fraction = cast(
                        Callable[[pp.BoundaryGrid], np.ndarray],
                        partial(self.bc_values_partial_fraction, component, phase),
                    )
                    self.update_boundary_condition(
                        self._partial_fraction_variable(component, phase),
                        bc_values_partial_fraction,
                    )

    def _update_phase_properties_on_boundaries(self, phase: pp.Phase) -> None:
        """Method updating the phase properties of a phase on all boundary grids for
        which results of the boundary flash are stored in
        :meth:`boundary_equilibrium_results`."""

        def update(
            prop: pp.ExtendedDomainFunctionType, val: np.ndarray, bg: pp.BoundaryGrid
        ) -> None:
            if isinstance(prop, pp.ad.SurrogateFactory):
                prop.update_boundary_values(val, bg, depth=self.time_step_indices.size)

        for bg, fluid_props in self.boundary_equilibrium_results.items():
            j = self.fluid.phases.index(phase)
            phase_props = fluid_props.phases[j]
            to_update = [
                (phase.density, phase_props.rho),
                (phase.specific_volume, phase_props.v),
                (phase.specific_enthalpy, phase_props.h),
                (phase.specific_internal_energy, phase_props.u),
                (phase.viscosity, phase_props.mu),
                (phase.thermal_conductivity, phase_props.kappa),
            ]
            for prop, val in to_update:
                update(prop, val, bg)

    def bc_values_saturation(self, phase: pp.Phase, bg: pp.BoundaryGrid) -> np.ndarray:
        """Boundary condition for saturation values of a ``phase``.

        This method is only called for independent phases.

        Parameters:
            phase: A phase in fluid.
            bg: A boundary grid.

        Returns:
            If results are stored for the passed boundary grid in
            :meth:`boundary_equilibrium_results`, they are returned. Otherwise a zero
            array is returned.

        """
        if bg in self.boundary_equilibrium_results:
            saturations = self.boundary_equilibrium_results[bg].sat
            j = self.fluid.phases.index(phase)
            return saturations[j]
        else:
            return np.zeros(bg.num_cells)

    def bc_values_partial_fraction(
        self, component: pp.Component, phase: pp.Phase, bg: pp.BoundaryGrid
    ) -> np.ndarray:
        """Boundary condition for the (extended) partial fraction of ``component`` in
        ``phase``.

        This method is called for every (independent) component in every phase.

        Parameters:
            component: A component in the phase.
            phase: A phase in fluid.
            bg: A boundary grid.

        Returns:
            If results are stored for the passed boundary grid in
            :meth:`boundary_equilibrium_results`, they are returned. Otherwise a zero
            array is returned.

        """
        if bg in self.boundary_equilibrium_results:
            j = self.fluid.phases.index(phase)
            i = phase.components.index(component)
            return self.boundary_equilibrium_results[bg].phases[j].x[i]
        else:
            return np.zeros(bg.num_cells)


class BoundaryConditionsCFLE(
    # NOTE The order here is critical, since primary variables must be updated first in
    # order for the BC flash to work.
    BoundaryConditionsEquilibrium,
    cf.BoundaryConditionsMulticomponent,
    pp.mass_and_energy_balance.BoundaryConditionsFluidMassAndEnergy,
):
    """Collection of boundary condition mixins for compositional flow with local
    equilibrium."""


class BoundaryConditionsCFFLE(
    # NOTE The order here is critical for the functionality. Primary variables must be
    # set first, followed by the BC flash execution. As a last step, the values of
    # fractional flow weights can be assembled.
    cf.BoundaryConditionsFractionalFlow,
    BoundaryConditionsEquilibrium,
    cf.BoundaryConditionsMulticomponent,
    pp.mass_and_energy_balance.BoundaryConditionsFluidMassAndEnergy,
):
    """BC mixin for CFLE models in the fractional flow formulation.

    The results of the boundary flash are used to provide values of the fractional flow
    weights on the boundary.

    """

    # TODO this needs a better solution, depending on how relative_permeability is
    # finally implemented.
    def _bc_value_phase_mobility(
        self, phase_index: int, fluid_properties: pc.FluidProperties
    ) -> np.ndarray:
        return (
            fluid_properties.sat[phase_index] / fluid_properties.phases[phase_index].mu
        )

    def _bc_value_component_mass_mobility(
        self, component: pp.FluidComponent, bg: pp.BoundaryGrid
    ) -> np.ndarray:
        """Helper method to evaluate the component mass mobility of a ``component`` on a
        boundary grid.

        Parameters:
            component: A component in the fluid.
            bg: A boundary grid.

        Returns:
            The value of the component mass mobility based on the results of the
            boundary flash.

        """
        fluid_props = self.boundary_equilibrium_results[bg]
        vals = np.zeros(bg.num_cells)

        for j, phase_props in enumerate(zip(fluid_props.phases, self.fluid.phases)):
            props, phase = phase_props
            if component in phase:
                x_ij = cast(
                    np.ndarray, props.x_normalized[phase.components.index(component)]
                )
                vals += x_ij * props.rho * self._bc_value_phase_mobility(j, fluid_props)

        return vals

    def _bc_value_total_mass_mobility(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Helper method to evaluate the total mass mobility on a boundary grid.

        Parameters:
            bg: A boundary grid.

        Returns:
            The value of the total mass mobility based on the results of the boundary
            flash.

        """
        fluid_props = self.boundary_equilibrium_results[bg]
        vals = np.zeros(bg.num_cells)

        for j, phase_props in enumerate(fluid_props.phases):
            vals += phase_props.rho * self._bc_value_phase_mobility(j, fluid_props)

        return vals

    def _bc_value_advected_enthalpy(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Helper method to evaluate the advected enthalpy on a boundary grid.

        Parameters:
            bg: A boundary grid.

        Returns:
            The value of the advected enthalpy based on the results of the boundary
            flash.

        """
        fluid_props = self.boundary_equilibrium_results[bg]
        vals = np.zeros(bg.num_cells)

        for j, phase_props in enumerate(fluid_props.phases):
            vals += (
                phase_props.h
                * phase_props.rho
                * self._bc_value_phase_mobility(j, fluid_props)
            )

        return vals

    def bc_values_fractional_flow_component(self, component, bg):
        """Computes the values based on the result from the boundary flash, if the flash
        was performed for the boundary grid ``bg``, and inserts it in the cells flagged
        for the boundary equilibrium."""

        vals = super().bc_values_fractional_flow_energy(bg)

        if bg in self.boundary_equilibrium_results:
            idx = self._boundary_equilibrium_cells(bg)
            component_mass_mobility = self._bc_value_component_mass_mobility(
                component, bg
            )[idx]
            total_mass_mobility = self._bc_value_total_mass_mobility(bg)[idx]
            vals[idx] = component_mass_mobility / total_mass_mobility

        return vals

    def bc_values_fractional_flow_energy(self, bg):
        """Computes the values based on the result from the boundary flash, if the flash
        was performed for the boundary grid ``bg``, and inserts it in the cells flagged
        for the boundary equilibrium."""

        vals = super().bc_values_fractional_flow_energy(bg)

        if bg in self.boundary_equilibrium_results:
            idx = self._boundary_equilibrium_cells(bg)
            advected_enthalpy = self._bc_value_advected_enthalpy(bg)[idx]
            total_mass_mobility = self._bc_value_total_mass_mobility(bg)[idx]
            vals[idx] = advected_enthalpy / total_mass_mobility

        return vals


class InitialConditionsEquilibrium(cf.InitialConditionsPhaseProperties):
    """Modified initialization procedure for compositional flow problem with
    equilibrium conditions and a flash instance.

    This class uses the flash to perform the 'initial flash' to calculate values
    for secondary variables and secondary operators representing the thermodynamic
    properties of phases.

    It performs a p-T flash i.e., enthalpy (though primary) is also initialized using
    the flash results.

    """

    flash: AbstractFlash
    """See :class:`SolutionStrategyCFLE`."""

    # Provided by CompositionalVariablesMixin
    has_independent_saturation: Callable[[pp.Phase], bool]
    has_independent_fraction: Callable[[pp.Phase | pp.Component], bool]
    has_independent_partial_fraction: Callable[[pp.Component, pp.Phase], bool]
    has_independent_extended_fraction: Callable[[pp.Component, pp.Phase], bool]

    # Provided by initial condition mixins
    ic_values_pressure: Callable[[pp.Grid], np.ndarray]
    ic_values_temperature: Callable[[pp.Grid], np.ndarray]
    ic_values_overall_fraction: Callable[[pp.Component, pp.Grid], np.ndarray]

    def set_initial_values_phase_properties(self) -> None:
        """Instead of computing the initial values using the underlying EoS, it performs
        the initial flash.

        The default implementation performs a p-T flash on every subdomain by calling
        :meth:`initial_equilibrium`.

        It performes a p-T flash, hence initial conditions for enthalpy are not
        required, but computed by this class.

        Values for phase properties, as well as secondary fractions and enthalpy are
        then initialized using the results, for all iterate and time step indices.

        Derivative values for properties are stored at the current iterate.

        """

        for sd in self.mdg.subdomains():
            # pressure, temperature and overall fractions
            p = self.ic_values_pressure(sd)
            T = self.ic_values_temperature(sd)
            self.initial_equilibrium(sd, {"p": p, "T": T})

    def initial_equilibrium(
        self, sd: pp.Grid, equilibrium_specs: StateSpecDict
    ) -> None:
        """Performs the flash on a given grid with given equilibrium conditions.

        Additionally to performing the flash, the results are used to provide
        initial values for saturations, phase fractions and partial fractions, as well
        as phase properties and their derivatives.

        The results are stored at iterate index 0.

        For properties appearing in accumulation terms (density and enthalpy), the
        time step indices are additionally initialized.

        State functions not defined by ``equilibrium_specs`` will also be provided
        with an initial value. E.g., if this method is called with ``'p','T'`` in
        ``equilibrium_specs``, the enthalpy value of from the flash result will be used
        to initialize the (fluid) enthalpy variable.

        Note:
            The initial feed fractions will be parsed directly in this method, since
            they are required in any case.

        Parameters:
            sd: A subdomain.
            equilibrium_specs: Definition of the equilibrium condition in terms of
                state functions and their values.

                See also :meth:`~porepy.compositional.flash.abstract_flash.
                AbstractFlash.flash`.

        """

        # IC values for potentially dependent component are never called directly.
        feed = [
            self.ic_values_overall_fraction(comp, sd)
            for comp in self.fluid.components
            if self.has_independent_fraction(comp)
        ]
        z_r = 1.0 - pc.safe_sum(feed)
        feed = (
            feed[: self.fluid.reference_component_index]
            + [z_r]
            + feed[self.fluid.reference_component_index :]
        )

        # Computing initial equilibrium.
        logger.info(f"Initial equilibration on grid {sd.id}.")
        results = self.flash.flash(
            equilibrium_specs,
            feed,
            params=self.params.get("flash_params", None),
        )

        self.postprocess_initial_equilibrium(sd, results)

        # NOTE Multiple ingores for mypy because the return type of several
        # callables is a general operator, while by logic it is indeed a variable.

        # Initializing values for unknown state functions.
        if isinstance(self, pp.fluid_mass_balance.VariablesSinglePhaseFlow):
            if results.specification >= pc.FlashSpec.vT:
                self.equation_system.set_variable_values(
                    results.p,
                    [self.pressure([sd])],  # type: ignore[arg-type]
                    iterate_index=0,
                )
            elif self.has_fluid_volume_variable:
                self.equation_system.set_variable_values(
                    results.v,
                    [self.specific_fluid_volume([sd])],  # type: ignore[arg-type]
                    iterate_index=0,
                )

        if isinstance(self, pp.energy_balance.VariablesEnergyBalance):
            if results.specification not in [pc.FlashSpec.pT, pc.FlashSpec.vT]:
                self.equation_system.set_variable_values(
                    results.T,
                    [self.temperature([sd])],  # type: ignore[arg-type]
                    iterate_index=0,
                )

            if (
                results.specification not in [pc.FlashSpec.ph, pc.FlashSpec.vh]
                and self.has_fluid_enthalpy_variable
            ):
                self.equation_system.set_variable_values(
                    results.h,
                    [self.specific_fluid_enthalpy([sd])],  # type: ignore[arg-type]
                    iterate_index=0,
                )

            if (
                results.specification != pc.FlashSpec.vu
                and self.has_fluid_internal_energy_variable
            ):
                self.equation_system.set_variable_values(
                    results.u,
                    [self.specific_fluid_internal_energy([sd])],  # type: ignore[arg-type]
                    iterate_index=0,
                )

        # Setting initial values for all fractional variables and phase properties.
        is_persistent = pc.is_persistent_variable_form(self)
        for j, phase in enumerate(self.fluid.phases):
            if self.has_independent_fraction(phase):
                self.equation_system.set_variable_values(
                    results.y[j],
                    [phase.fraction([sd])],  # type: ignore[arg-type]
                    iterate_index=0,
                )
            if self.has_independent_saturation(phase):
                self.equation_system.set_variable_values(
                    results.sat[j],
                    [phase.saturation([sd])],  # type: ignore[arg-type]
                    iterate_index=0,
                )

            # fractions of component in phase
            for k, comp in enumerate(phase.components):
                # Extended or partial, one of them is independent
                if self.has_independent_extended_fraction(comp, phase):
                    self.equation_system.set_variable_values(
                        results.phases[j].x[k],
                        [phase.extended_fraction_of[comp]([sd])],  # type: ignore[arg-type]
                        iterate_index=0,
                    )
                elif self.has_independent_partial_fraction(comp, phase):
                    self.equation_system.set_variable_values(
                        results.phases[j].x_normalized[k],
                        [phase.partial_fraction_of[comp]([sd])],  # type: ignore[arg-type]
                        iterate_index=0,
                    )

            # Update values and derivatives for current iterate.
            # Extend derivatives from partial to extended fractions, in the case of
            # unified equilibrium formulations.
            cf.update_phase_properties(
                sd,
                phase,
                results.phases[j],
                0,
                use_extended_derivatives=is_persistent,
                update_fugacities=True,
            )

    def postprocess_initial_equilibrium(
        self, sd: pp.Grid, results: FlashResults
    ) -> None:
        """Postprocessing of initial equilibrium calculations.

        The base method asserts that all calculations converged.

        Parameters:
            sd: Grid on which the equilibrium was calculated.
            results: Flash results for given grid.

        """
        if not np.all(results.converged):
            raise ValueError(f"Initial flash not successful on grid {sd.id}")


class InitialConditionsCFLE(
    InitialConditionsEquilibrium,
    cf.InitialConditionsCF,
):
    """Collection of initial condition mixins for compositional flow with local
    equilibrium."""


class SolutionStrategyEquilibrium(cf.SolutionStrategyPhaseProperties):
    """A solution strategy for compositional flow with local equilibrium conditions in
    the form of algebraic equations.

    Performs nonlinear preconditioning in the form of local equilibrium calculations.
    I.e., it solves the respective subsystem and updates secondary quantities using the
    results from the flash.

    Important:
        Compositional flow models with local equilibrium equations assume that the
        model is closed in the sense that secondary variables are completely determined
        by the local equilibrium equations.

        Hence no secondary variable (as defined by the base variable mixin for CF) is
        eliminated by some constitutive expression.

    Supports the following model parameters:

    - ``'equilibrium_specification'``: Defaults to None. See
      :func:`~porepy.compositional.compositional_mixins.get_equilibrium_specification`.
    - ``'flash_params'``: Defaults to None. Parameter dictionary used for flash
      initialization and calling the flash method.

    """

    flash: AbstractFlash
    """The flash class set by this solution strategy."""

    # Provided by respective variable mixins.
    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]

    has_independent_saturation: Callable[[pp.Phase], bool]
    has_independent_fraction: Callable[[pp.Phase | pp.Component], bool]
    has_independent_partial_fraction: Callable[[pp.Component, pp.Phase], bool]
    has_independent_extended_fraction: Callable[[pp.Component, pp.Phase], bool]

    def set_materials(self):
        """Asserts that local equilibrium conditions are specified before setting the
        flash."""
        super().set_materials()

        assert pc.has_equilibrium_specified(self), (
            "Local equilibrium condition not defined in model parameters."
        )

        self.set_flash()

    def set_flash(self) -> None:
        """Sub-routine of :meth:`set_materials` to set the flash class for equilibrium
        calculations, after the fluid is defined.

        The base method uses the :class:`~porepy.compositional.flash.
        persistent_variable_flash.CompiledPersistentVariableFlash` and the
        :class:`~porepy.compositional.flash.flash_initializer.
        HeuristicVLInitializer`.

        """

        # Import here for runtime reasons of global import (compilation).
        import porepy.compositional.flash as pf

        # Setting default flash params.
        if "flash_params" not in self.params:
            self.params["flash_params"] = {
                "compile": True,
                "compile_args": (),
            }

        assert isinstance(self.params["flash_params"], dict), (
            "params['flash_params'] expected to be dictionary."
        )

        self.flash = pf.CompiledPersistentVariableFlash(
            self.fluid, params=self.params["flash_params"]
        )

        if self.params["flash_params"]["compile"]:
            self.flash.compile(*self.params["flash_params"]["compile_args"])

    def update_derived_quantities(self):
        """Normalizes fractional variables in the case of violation of the bound
        [0,1], before calling the base method."""
        self.make_fractions_feasible()
        self.make_statefunctions_feasible()
        super().update_derived_quantities()

    def make_fractions_feasible(self) -> None:
        """Sub-routine to bind fractions to interval [0,1] and normalize where
        applicable.

        This method overwrites the state of the system, i.e. values stored for DOFs.

        """
        subdomains = self.mdg.subdomains()

        # NOTE: eps=0 to not modify feed fractions.
        state = self.current_fluid_state(subdomains, eps=0.0)

        for z_i, comp in zip(state.z, self.fluid.components):
            if self.has_independent_fraction(comp):
                self.equation_system.set_variable_values(
                    z_i,
                    [comp.fraction(subdomains)],  # type:ignore[arg-type]
                    iterate_index=0,
                )
        for j, data in enumerate(zip(state.sat, state.y, self.fluid.phases)):
            s_j, y_j, phase = data
            if self.has_independent_saturation(phase):
                self.equation_system.set_variable_values(
                    s_j,
                    [phase.saturation(subdomains)],  # type:ignore[arg-type]
                    iterate_index=0,
                )
            if self.has_independent_fraction(phase):
                self.equation_system.set_variable_values(
                    y_j,
                    [phase.fraction(subdomains)],  # type:ignore[arg-type]
                    iterate_index=0,
                )

            for i, comp in enumerate(self.fluid.components):
                if self.has_independent_extended_fraction(comp, phase):
                    self.equation_system.set_variable_values(
                        state.phases[j].x[i],
                        [phase.extended_fraction_of[comp](subdomains)],  # type:ignore[arg-type]
                        iterate_index=0,
                    )
                elif self.has_independent_partial_fraction(comp, phase):
                    self.equation_system.set_variable_values(
                        state.phases[j].x[i],
                        [phase.partial_fraction_of[comp](subdomains)],  # type:ignore[arg-type]
                        iterate_index=0,
                    )

    def make_statefunctions_feasible(self, min_val: float = 1e-7) -> None:
        """Makes state functions, which must be positive, strictly positive.

        This includes pressure, temperature and specific fluid volume, if they are
        variables.

        Parameters:
            min_val: Minimal value for state functions.

        """
        assert min_val > 0.0, "Minimal value expected to be strictly positive."

        if isinstance(self, pp.fluid_mass_balance.VariablesSinglePhaseFlow):
            p = self.equation_system.get_variable_values(
                [self.pressure_variable], iterate_index=0
            )
            p = np.maximum(p, min_val)
            self.equation_system.set_variable_values(
                p, [self.pressure_variable], iterate_index=0
            )
            if self.has_fluid_volume_variable:
                v = self.equation_system.get_variable_values(
                    [self.specific_fluid_volume_variable], iterate_index=0
                )
                v = np.maximum(v, min_val)
                self.equation_system.set_variable_values(
                    v, [self.specific_fluid_volume_variable], iterate_index=0
                )

        if isinstance(self, pp.energy_balance.VariablesEnergyBalance):
            T = self.equation_system.get_variable_values(
                [self.temperature_variable], iterate_index=0
            )
            T = np.maximum(T, min_val)
            self.equation_system.set_variable_values(
                T, [self.temperature_variable], iterate_index=0
            )

    def do_flash_preconditioning(self) -> bool:
        """Checks whether the flash should be done for the given iteration as specified
        in ``params["flash_params"]["global_iteration_stride"]``.

        The parameter can be set to some boolean value to activate or deactivate the
        preconditioning using the flash.
        It can also be set to an integer, specifying that it should be done every
        n-th iteration.

        Returns:
            A boolean indicating if the flash is requested.

        """
        stride = self.params.get("flash_params", {}).get("global_iteration_stride", 1)  # type:ignore
        do_flash = False

        # If non-integer or non-positive value, return boolean equivalent.
        if not isinstance(stride, int):
            do_flash = bool(stride)
        # If positive integer value, do calculations.
        else:
            # If problem is nonlinear, do it every n-th iteration.
            if (
                isinstance(
                    self.nonlinear_solver_statistics, pp.NonlinearSolverStatistics
                )
                and stride > 0
            ):
                n = self.nonlinear_solver_statistics.num_iterations
                # NOTE The iteration counter is increased after the iteration.
                do_flash = (n + 1) % stride == 0
            # For possibly linear problems, just return boolean value again.
            else:
                do_flash = bool(stride)

        return do_flash

    def before_nonlinear_iteration(self) -> None:
        """Calls :meth:`nonlinear_flash_preconditioning` after the super-call."""
        super().before_nonlinear_iteration()  # type:ignore[safe-super]
        self.nonlinear_flash_preconditioning()

    def nonlinear_flash_preconditioning(self):
        """Uses the defined flash instance to solve the local equilibrium subproblem.

        If requested, loops over subdomains and equilibrates the fluid using the
        default equilibrium specifications and the current state as initial guess.

        See also:
            :meth:`do_flash_preconditioning`, :meth:`local_equilibrium`

        """

        if self.do_flash_preconditioning():
            for grid in self.mdg.subdomains():
                self.local_equilibrium(grid)

    def current_fluid_state(
        self,
        subdomains: Sequence[pp.Grid] | pp.Grid,
        state: Optional[np.ndarray] = None,
        feasible_fractions: bool = True,
        eps: float = 1e-7,
    ) -> pc.FluidProperties:
        """Method to assemble the state of the fluid at the current iterate.

        The returned fluid state contains only quantities considered unknowns (fractions
        and equilibrium state functions), and not fluid properties. It also returns only
        the intensive state functions pressure and temperature. Energies and volume are
        not considered as possible variables in the thermodynamic sense.

        Intended use for the returned fluid property instance is as the initial guess
        for the flash performed in :meth:`local_equilibrium`.

        This method provides room to pre-process data before the flash is called.

        Parameters:
            subdomains: One or multiple subdomains in the md-grid.
            state: A global state vector for evaluating the state variables.
            feasible_fractions: If true, fractional quantities are ensured to be bound
                in the interval [0,1] and fulfill the unity constraint.
            eps: Used to bind overall fractions away from zero and detect absent
                phases.

        Returns:
            The base method returns a fluid state containing the current iterate values
            for all fractional variables, as well as pressureand temperature.

        """

        if isinstance(subdomains, pp.Grid):
            subdomains = [subdomains]

        is_persistent = pc.is_persistent_variable_form(self)
        nc = sum([sd.num_cells for sd in subdomains])

        # EPS for fractions of absent phases in persistent form.
        # Used to detect absence of phase (y) and to bind extended fractions away
        # from zero.
        eps_persistent = 1e-8
        # EPS for (extended) partial fractions to avoid numerical issues.
        # Must never be zero.
        eps_machine = 1e-14

        # NOTE: In the case of 1 component, z is implemented as the scalar 1.
        if self.fluid.num_components == 1:
            z = np.ones((1, nc))
        else:
            z = np.array(
                [
                    self.equation_system.evaluate(
                        component.fraction(subdomains), state=state
                    )
                    for component in self.fluid.components
                ]
            )

        y = np.array(
            [
                self.equation_system.evaluate(phase.fraction(subdomains), state=state)
                for phase in self.fluid.phases
            ]
        )

        sat = np.array(
            [
                self.equation_system.evaluate(phase.saturation(subdomains), state=state)
                for phase in self.fluid.phases
            ]
        )

        x = [
            np.array(
                [
                    (
                        self.equation_system.evaluate(
                            phase.extended_fraction_of[component](subdomains),
                            state=state,
                        )
                        if is_persistent
                        else self.equation_system.evaluate(
                            phase.partial_fraction_of[component](subdomains),
                            state=state,
                        )
                    )
                    for component in phase
                ]
            )
            for phase in self.fluid.phases
        ]

        if feasible_fractions:
            z[z < 0] = eps
            z[z > 1] = 1.0 - eps
            z = pc.normalize_rows(z.T).T

            sat[sat < 0] = 0.0
            sat[sat > 1] = 1.0
            sat = pc.normalize_rows(sat.T).T

            y[y < 0] = 0.0
            y[y > 1] = 1.0
            y = pc.normalize_rows(y.T).T

            for y_, x_ in zip(y, x):
                x_[x_ < 0] = eps_machine
                x_[x_ > 1] = 1.0
                # NOTE: In persistent-variable form, phase can be absent despite
                # numerically small fraction. Extended partial fractions do not fulfill
                # unity constraint there. We avoid normalization to not mess with this
                # sensitivity.
                idx = (
                    y_ > max(eps, eps_persistent)
                    if is_persistent
                    else np.ones_like(y_, dtype=bool)
                )
                if np.any(idx):
                    x_[:, idx] = pc.normalize_rows(x_[:, idx].T).T

                # If extended fractions (phase vanished) violate unity, we can safely
                # normalize them as they have no physical meaning. The flash should
                # resolve the correct values.
                idx = (x_.sum(axis=0) > 1) & (y_ <= max(eps, eps_persistent))
                if np.any(idx):
                    x_[:, idx] = pc.normalize_rows(x_[:, idx].T).T

        # NOTE
        if hasattr(self, "pressure"):
            p = cast(
                np.ndarray,
                self.equation_system.evaluate(self.pressure(subdomains), state=state),
            )
        else:
            p = np.zeros(nc)

        if hasattr(self, "temperature"):
            T = cast(
                np.ndarray,
                self.equation_system.evaluate(
                    self.temperature(subdomains), state=state
                ),
            )
        else:
            T = np.zeros(nc)

        return pc.FluidProperties(
            z=z,
            y=y,
            sat=sat,
            p=p,
            T=T,
            phases=[
                pc.PhaseProperties(state=phase.state, x=x_)
                for x_, phase in zip(x, self.fluid.phases)
            ],
        )

    def local_equilibrium(
        self,
        grid: pp.Grid,
        /,
        *,
        specification: Optional[StateSpecDict] = None,
        initial_guess_from_current_state: bool = True,
        update_secondary_quantities: bool = True,
        state: Optional[np.ndarray] = None,
        cell_mask: Optional[np.ndarray] = None,
    ) -> tuple[FlashResults, NDArray[np.bool_]]:
        """Performs flash calculations on the given grid and updates the fluid
        properties at the current iterate.

        Performs a full flash (with initial guess), where the flash based on the global
        iterate state did not succeed.

        Calls :meth:`postprocess_equilibrium` and applies the update where indicated.

        See also:
            :meth:`~porepy.compositional.flash.abstract_flash.AbstractFlash.flash`,
            :meth:`update_secondary_quantities_from_flash_result`,
            :meth:`current_fluid_state`,

        Parameters:
            grid: A subdomain in the md-grid.
            specification: ``default=None``

                Definition of the equilibrium condition in terms of state functions and
                their values.

                If None, the equilibrium condition is parsed from the model paramters.
            initial_guess_from_current_state: ``default=True``

                If True, the initial fluid state for the flash is evaluated from the
                current solution values at iterate 0.

            update_secondary_quantities: ``default=True``

                If True, the flash results are used to update the values of variables
                of the equilibrium problem at iterate 0, additionally to the fluid
                properties.

                Besides updates of various fractions, this includes also an update
                of pressure or temperature for example, if they are not defined in
                ``specification``.
            state: ``default=None``

                Global state vector to evaluate the equilibrium state functions.
            cell_mask: ``default=None``

                If a boolean array is given, the flash is only performed on the
                masked cells. If None, all cells are equilibrated.

        Returns:
            A 2-tuple containing the flash results and a boolean array indicating where
            the results were used to update the state of the system and values of fluid
            properties. The latter depends on ``cell_mask`` as well as the success of
            the flash and the results of :meth:`postprocess_equilibrium`.

        """

        if cell_mask is None:
            cell_mask = np.ones(grid.num_cells, dtype=np.bool_)

        if specification is None:
            model_specs = pc.get_equilibrium_specifications(self)
            flash_spec = [s for s in model_specs if isinstance(s, pc.FlashSpec)][0]

            spec = {}

            if flash_spec < pc.FlashSpec.vT:
                spec["p"] = self.equation_system.evaluate(
                    self.pressure([grid]), state=state
                )

                if flash_spec == pc.FlashSpec.pT:
                    spec["T"] = self.equation_system.evaluate(
                        self.temperature([grid]), state=state
                    )
                elif flash_spec == pc.FlashSpec.ph:
                    h_func = self.fluid.specific_enthalpy
                    if isinstance(self, pp.energy_balance.VariablesEnergyBalance):
                        if self.has_fluid_enthalpy_variable:
                            h_func = self.specific_fluid_enthalpy  # type: ignore[assignment]
                    spec["h"] = self.equation_system.evaluate(
                        h_func([grid]), state=state
                    )
                else:
                    raise NotImplementedError(
                        f"Isobaric specification {flash_spec} not implemented."
                    )
            else:
                v_func = self.fluid.specific_volume

                if isinstance(self, pp.fluid_mass_balance.VariablesSinglePhaseFlow):
                    if self.has_fluid_volume_variable:
                        v_func = self.specific_fluid_volume  # type: ignore[assignment]

                spec["v"] = self.equation_system.evaluate(v_func([grid]), state=state)

                if flash_spec == pc.FlashSpec.vT:
                    spec["T"] = self.equation_system.evaluate(
                        self.temperature([grid]), state=state
                    )
                elif flash_spec == pc.FlashSpec.vh:
                    h_func = self.fluid.specific_enthalpy
                    if isinstance(self, pp.energy_balance.VariablesEnergyBalance):
                        if self.has_fluid_enthalpy_variable:
                            h_func = self.specific_fluid_enthalpy  # type: ignore[assignment]
                    spec["h"] = self.equation_system.evaluate(
                        h_func([grid]), state=state
                    )
                elif flash_spec == pc.FlashSpec.vu:
                    u_func = self.fluid.specific_internal_energy
                    if isinstance(self, pp.energy_balance.VariablesEnergyBalance):
                        if self.has_fluid_internal_energy_variable:
                            u_func = self.specific_fluid_internal_energy
                    spec["u"] = self.equation_system.evaluate(
                        u_func([grid]), state=state
                    )
                else:
                    raise NotImplementedError(
                        f"Isochoric specification {flash_spec} not implemented."
                    )

            specification = spec  # type:ignore

        initial_state: pc.FluidProperties | None
        feed: np.ndarray

        if initial_guess_from_current_state:
            initial_state = self.current_fluid_state(grid, state=state)
            feed = initial_state.z
        else:
            initial_state = None
            feed = np.array(
                [
                    self.equation_system.evaluate(comp.fraction([grid]), state=state)
                    for comp in self.fluid.components
                ]
            )

        assert specification is not None, "Failed to specify flash."
        results = self.flash.flash(
            specification,
            [z for z in feed],
            initial_state=initial_state,
            params=cast(dict | None, self.params.get("flash_params", None)),
        )

        # Perform the full flash where the initial guess from the current state caused
        # failures.
        not_converged = ~results.converged
        if np.any(not_converged) and initial_guess_from_current_state:
            logger.info(
                f"Flash from iterate state failed in {not_converged.sum()} cells on"
                + f" grid {grid.id}. Performing full flash."
            )
            self._full_equilibrium(results, specification)

        success_mask = self.postprocess_equilibrium(grid, results, state=state)
        update_mask = success_mask & cell_mask
        results.postprocess_fractions()
        results.evaluate_saturations()
        results.evaluate_extensive_state()

        if update_secondary_quantities:
            self.update_secondary_quantities_from_flash_result(
                [grid], results, update_mask
            )

        logger.info(
            f"Fluid {results.specification.name}-equilibrated on grid {grid.id}: "
            + f"t= {self.time_manager.time:.3e} "
            + f"i= {self.nonlinear_solver_statistics.num_iterations}"  # type: ignore[attr-defined]
        )
        self.nonlinear_solver_statistics.log_custom_data(
            append=True,
            flash_clocktime=results.clocktime_solve + results.clocktime_init,
        )
        self.nonlinear_solver_statistics.log_custom_data(
            append=True,
            flash_iterations=int(results.num_iter.sum()),
        )

        return results, update_mask

    def _full_equilibrium(
        self,
        results: FlashResults,
        flash_spec: StateSpecDict,
    ) -> None:
        """A method to perform the full equilibrium calculations, including an initial
        guess, where the given flash results indicate failure.

        This is meant as a robust fall-back strategy in case equilibration based on
        current fluid state (from global solver) fails.

        Parameters:
            results: First flash results where some failure occurred.
            flash_spec: The equilibrium specification used in the flash.

        """
        failure = ~results.converged
        spec: StateSpecDict = dict([(k, v[failure]) for k, v in flash_spec.items()])  # type:ignore

        sub_results = self.flash.flash(
            spec,
            cast(Sequence[np.ndarray], results.z[:, failure]),
            params=cast(dict | None, self.params.get("flash_params", None)),
        )

        results.clocktime_solve += sub_results.clocktime_solve
        results.clocktime_init += sub_results.clocktime_init
        # We count the full flash iterations in addition to the previous ones.
        results.num_iter[failure] += sub_results.num_iter
        # NOTE: setitem overwrites exitcodes only.
        results[failure] = sub_results

    def postprocess_equilibrium(
        self,
        grid: pp.Grid,
        results: FlashResults,
        /,
        *,
        state: Optional[np.ndarray] = None,
    ) -> NDArray[np.bool_]:
        """A method called by :meth:`local_equilibrium` to post-process flash results
        and indicate which results to use to update values in the global system.

        The base method returns True where the flash converged.

        Parameters:
            grid: The subdomain on which the flash was performed.
            results: The resulting fluid properties and success flags from
                the call to :meth:`local_equilibrium`.
            state: A global state vector from which the state variables were evaluated
                for the flash.

        Returns:
            A boolean array indicating where the flash results should replace the
            current iterate values of phase properties and, in the thermodynamic sense,
            dependent variables.

        """
        not_converged = ~results.converged
        n = int(not_converged.sum())
        if n > 0:
            logger.warning(
                f"{results.specification.name}-flash failed in {n} cells on grid"
                f" {grid.id}."
            )
        return results.converged

    def update_secondary_quantities_from_flash_result(
        self,
        subdomains: list[pp.Grid],
        results: FlashResults,
        cell_mask: NDArray[np.bool_] | None = None,
    ) -> None:
        """Updates values for secondary quantities stored in subdomain data
        dictionaries.

        Secondary quantities include all phase properties and their derivatives, as well
        as fluid state functions which are mathematical variables in the PorePy model
        but not fixed by the flash specification stored in ``results``.

        Parameters:
            subdomains: A sequence of domains for which data is contained in
                ``results``.
            results: Flash results data structure containing cell-wise data for all
                subdomains.
            cell_mask: ``default=None``

                If given, the update is only performed on respective cells.

        """
        nc = int(sum([grid.num_cells for grid in subdomains]))
        if cell_mask is None:
            cell_mask = np.ones(nc, dtype=np.bool_)

        # Updating fluid properties.
        n = 0
        is_persistent = pc.is_persistent_variable_form(self)
        for grid in subdomains:
            n1p = n + grid.num_cells
            idx = np.zeros(nc, dtype=np.bool_)
            idx[n:n1p] = True
            for phase, phase_state in zip(self.fluid.phases, results.phases):
                cf.update_phase_properties(
                    grid,
                    phase,
                    phase_state[idx],
                    0,
                    use_extended_derivatives=is_persistent,
                    update_fugacities=True,
                    mask=cell_mask[idx],
                )
            n = n1p

        def update(var: pp.ad.Operator, vals: np.ndarray) -> None:
            assert isinstance(var, (pp.ad.MixedDimensionalVariable, pp.ad.Variable)), (
                f"Operator {var.name} not independent variable."
            )
            current_vals = self.equation_system.get_variable_values(
                [var], iterate_index=0
            )
            current_vals[cell_mask] = vals[cell_mask]
            self.equation_system.set_variable_values(
                current_vals, [var], iterate_index=0
            )

        # Updating variables which are always unknowns in the equilibrium problem.
        for j, phase in enumerate(self.fluid.phases):
            if self.has_independent_fraction(phase):
                update(phase.fraction(subdomains), results.y[j])

            if self.has_independent_saturation(phase):
                update(phase.saturation(subdomains), results.sat[j])

            for i, comp in enumerate(phase.components):
                if self.has_independent_extended_fraction(comp, phase):
                    var = phase.extended_fraction_of[comp](subdomains)
                elif self.has_independent_partial_fraction(comp, phase):
                    var = phase.partial_fraction_of[comp](subdomains)
                else:
                    continue

                update(var, results.phases[j].x[i])

        # Updating state variables. If isochoric, update pressure. If isobaric, update
        # fluid volume.
        if isinstance(self, pp.fluid_mass_balance.VariablesSinglePhaseFlow):
            if results.specification >= pc.FlashSpec.vT:
                update(self.pressure(subdomains), results.p)
            elif self.has_fluid_volume_variable:
                update(self.specific_fluid_volume(subdomains), results.v)

        # Update energy-related variables if applicable.
        if isinstance(self, pp.energy_balance.VariablesEnergyBalance):
            if results.specification not in [pc.FlashSpec.pT, pc.FlashSpec.vT]:
                update(self.temperature(subdomains), results.T)

            if (
                results.specification
                not in [
                    pc.FlashSpec.ph,
                    pc.FlashSpec.vh,
                ]
                and self.has_fluid_enthalpy_variable
            ):
                update(self.specific_fluid_enthalpy(subdomains), results.h)

            if (
                results.specification != pc.FlashSpec.vu
                and self.has_fluid_internal_energy_variable
            ):
                update(self.specific_fluid_internal_energy(subdomains), results.u)


class SolutionStrategyCFLE(
    SolutionStrategyEquilibrium,
    cf.SolutionStrategyExtendedFluidMassAndEnergy,
):
    """Collection of solution strategies for compositional flow with local
    equilibrium."""


class CFLEModelTemplate(  # type: ignore[misc]
    cf.ConstitutiveLawsCF,
    cf.VolumeBalanceFormulation,
    PersistentEquilibriumEquations,
    cf.PrimaryEquationsCF,
    cf.VariablesCF,
    BoundaryConditionsCFLE,
    InitialConditionsCFLE,
    SolutionStrategyCFLE,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """Base class for compositional flow with local equilibrium equations."""


class CFFLEModelTemplate(  # type: ignore[misc]
    cf.ConstitutiveLawsCF,
    cf.VolumeBalanceFormulation,
    PersistentEquilibriumEquations,
    cf.PrimaryEquationsCFF,
    cf.VariablesCF,
    BoundaryConditionsCFFLE,
    InitialConditionsCFLE,
    SolutionStrategyCFLE,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """Base class for compositional fractional flow with local equilibrium equations."""


class IsothermalCFLEModelTemplate(  # type: ignore[misc]
    cf.ConstitutiveLawsCF,
    cf.VolumeBalanceFormulation,
    PersistentEquilibriumEquations,
    cf.ComponentMassBalanceEquations,
    pp.fluid_mass_balance.FluidMassBalanceEquations,
    pc.CompositionalVariables,
    pp.fluid_mass_balance.VariablesSinglePhaseFlow,
    BoundaryConditionsEquilibrium,
    cf.BoundaryConditionsMulticomponent,
    pp.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow,
    InitialConditionsEquilibrium,
    cf.InitialConditionsFractions,
    pp.fluid_mass_balance.InitialConditionsSinglePhaseFlow,
    SolutionStrategyEquilibrium,
    pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """Isothermal model template for case 2.

    Temperature is constant, but for compatibility with the framework, every
    temperature-related function is implemented, returning the constant value
    :attr:`_T_IN` in some form.

    """

    _T_IN: float

    def temperature(self, subdomains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        nc = sum([sd.num_cells for sd in subdomains])
        return pp.wrap_as_dense_ad_array(self._T_IN, nc, "temperature")

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        return np.ones(sd.num_cells) * self._T_IN

    def bc_values_temperature(self, bg: pp.BoundaryGrid) -> np.ndarray:
        return np.ones(bg.num_cells) * self._T_IN

    def postprocess_equilibrium(
        self,
        grid: pp.Grid,
        results: FlashResults,
        /,
        *,
        state: Optional[np.ndarray] = None,
    ) -> NDArray[np.bool_]:
        """Removes temperature-dependency from derivatives of phase properties."""
        row_idx = np.array(
            [True, False] + [True] * self.fluid.num_components, dtype=np.bool_
        )

        for phase in results.phases:
            phase.dh = phase.dh[row_idx, :]
            phase.drho = phase.drho[row_idx, :]
            phase.du = phase.du[row_idx, :]
            phase.dmu = phase.dmu[row_idx, :]
            phase.dkappa = phase.dkappa[row_idx, :]
            phase.dphis = np.array([dphis[row_idx, :] for dphis in phase.dphis])

        return super().postprocess_equilibrium(grid, results, state=state)

    def postprocess_initial_equilibrium(
        self, sd: pp.Grid, results: FlashResults
    ) -> None:
        """Removes temperature-dependency from derivatives of phase properties."""
        row_idx = np.array(
            [True, False] + [True] * self.fluid.num_components, dtype=np.bool_
        )

        for phase in results.phases:
            phase.dh = phase.dh[row_idx, :]
            phase.drho = phase.drho[row_idx, :]
            phase.du = phase.du[row_idx, :]
            phase.dmu = phase.dmu[row_idx, :]
            phase.dkappa = phase.dkappa[row_idx, :]
            phase.dphis = np.array([dphis[row_idx, :] for dphis in phase.dphis])

        super().postprocess_initial_equilibrium(sd, results)

    def update_thermodynamic_properties_of_phases_on_grid(
        self, grid: pp.Grid, state: Optional[np.ndarray] = None
    ) -> None:
        """Handling of constant temperature case for EoS computations."""

        row_idx = np.array(
            [True, False] + [True] * self.fluid.num_components, dtype=np.bool_
        )

        equilibrium_defined = pc.has_equilibrium_specified(self)
        is_persistent = pc.is_persistent_variable_form(self)

        for phase in self.fluid.phases:
            dep_vals = [
                self.equation_system.evaluate(d([grid]), state=state)
                for d in self.dependencies_of_phase_properties(phase)
            ]
            dep_vals = (
                dep_vals[:1] + [np.ones(grid.num_cells) * self._T_IN] + dep_vals[1:]
            )
            phase_state = phase.compute_properties(
                *cast(list[np.ndarray], dep_vals),
                params=self.params.get("phase_property_params", None),  # type:ignore[arg-type]
            )

            phase_state.dh = phase_state.dh[row_idx, :]
            phase_state.drho = phase_state.drho[row_idx, :]
            phase_state.du = phase_state.du[row_idx, :]
            phase_state.dmu = phase_state.dmu[row_idx, :]
            phase_state.dkappa = phase_state.dkappa[row_idx, :]
            phase_state.dphis = np.array(
                [dphis[row_idx, :] for dphis in phase_state.dphis]
            )

            cf.update_phase_properties(
                grid,
                phase,
                phase_state,
                0,
                use_extended_derivatives=is_persistent,
                update_fugacities=equilibrium_defined,
            )
