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
import time
from functools import cached_property, partial
from typing import (
    TYPE_CHECKING,
    Callable,
    Optional,
    Sequence,
    cast,
)

import numpy as np

import porepy as pp
import porepy.compositional as pc

from . import compositional_flow as cf
from .persistent_variable_equilibrium import PH_PVEEquations

# NOTE: Avoid actual import and triggering of compilation. We only need this for type
# checking
if TYPE_CHECKING:
    from porepy.compositional.flash.abstract_flash import (
        AbstractFlash,
        FlashResults,
        StateSpecDict,
    )


logger = logging.getLogger(__name__)


class LocalIsenthalpicEquilibriumEquations(PH_PVEEquations):
    """Equations for closing compositional flow models with isobaric-isenthalpic
    equilibrium conditions.

    Due to saturations and molar fractions being independent variables, the model is
    closed with local phase mass conservation equations.

    Note:
        Using an independent fluid enthalpy variable, these model equations are suitable
        for both, isenthalpic and isothermal flash procedures.

    """

    has_independent_saturation: Callable[[pp.Phase], bool]
    """See :class:`~porepy.compositional.compositional_mixins._MixtureDOFHandler`."""

    def set_equations(self):
        """Assembles primary balance equations, local equilibrium equations and
        local phase mass conservation, in that order.

        The phase fraction variable usually appears in equilibrium formulations.
        Since saturations are variables as well, the system must be closed by relating
        those two phase-related quantities to each other.

        """
        super().set_equations()

        subdomains = self.mdg.subdomains()
        for phase in self.fluid.phases:
            if self.has_independent_saturation(phase):
                equ = self.mass_constraint_for_phase(phase, subdomains)
                self.equation_system.set_equation(equ, subdomains, {"cells": 1})


class EnthalpyBasedEquationsCFLE(
    LocalIsenthalpicEquilibriumEquations,
    cf.PrimaryEquationsCF,
):
    """CFLE model equations with a p-h equilibrium.

    Notably, this model uses the fluid mass balance (non-fractional flow) and the
    unified p-h equilibrium, with a local closure for saturations in the form of mass
    constraints per independent phase.

    """


class EnthalpyBasedEquationsCFFLE(
    LocalIsenthalpicEquilibriumEquations,
    cf.PrimaryEquationsCFF,
):
    """CFFLE model equations with a p-h equilibrium.

    Contrary to :class:`EnthalpyBasedEquationsCFLE`, this collection of equations which
    uses the pressure equation in the fractional-flow formulation, and relies hence on
    re-discretization of fluxes.

    """


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

    Supports the following model parameters:

    - ``'has_time_dependent_boundary_values'``: Defaults to False.
      A bool indicating whether Dirichlet BC for pressure, temperature or
      feed fractions are time-dependent.

      If True, the boundary equilibrium will be re-computed at the beginning of every
      time step.

    """

    flash: AbstractFlash
    """See :class:`SolutionStrategyCFLE`."""

    bc_values_pressure: Callable[[pp.BoundaryGrid], np.ndarray]
    """See :class:`~porepy.models.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow`.
    """
    bc_values_temperature: Callable[[pp.BoundaryGrid], np.ndarray]
    """See :class:`~porepy.models.energy_balance.BoundaryConditionsEnergy`."""
    bc_values_overall_fraction: Callable[[pp.Component, pp.BoundaryGrid], np.ndarray]
    """See :class:`~porepy.models.compositional_flow.BoundaryConditionsMulticomponent`.
    """

    has_independent_fraction: Callable[[pp.Component], bool]
    """Provided by mixin for compositional variables."""

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


class BoundaryConditionsCFLE(
    # NOTE The order here is critical, since primary variables must be updated first in
    # order for the BC flash to work.
    BoundaryConditionsEquilibrium,
    cf.BoundaryConditionsMulticomponent,
    pp.mass_and_energy_balance.BoundaryConditionsFluidMassAndEnergy,
):
    """BC mixin for CFLE models in the standard formulation (not fractional flow).

    The results of the boundary flash are used to update values of phase properties and
    secondary variables such as partial fractions, which are relevant on the boundary.

    Note:
        This mixin is built on the same assumption as
        :class:`BoundaryConditionsEquilibrium`, in terms of which variables are required
        on the boundary for the flash. Hence no BC values for enthalpy.

    """

    # Provided by CompositionalVariablesMixin
    has_independent_saturation: Callable[[pp.Phase], bool]
    has_independent_partial_fraction: Callable[[pp.Component, pp.Phase], bool]
    has_independent_extended_fraction: Callable[[pp.Component, pp.Phase], bool]
    _saturation_variable: Callable[[pp.Phase], str]
    _partial_fraction_variable: Callable[[pp.Component, pp.Phase], str]

    def update_all_boundary_conditions(self):
        """Updates BC values of phase properties (surrogate operators) and secondary
        variables appearing in the non-linear weights on the boundary.

        The update is performed using the results of the BC flash.

        """
        super().update_all_boundary_conditions()

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

        nt = self.time_step_indices.size

        for bg, fluid_props in self.boundary_equilibrium_results.items():
            j = self.fluid.phases.index(phase)
            phase_props = fluid_props.phases[j]
            if isinstance(phase.density, pp.ad.SurrogateFactory):
                phase.density.update_boundary_values(phase_props.rho, bg, depth=nt)
            if isinstance(phase.specific_enthalpy, pp.ad.SurrogateFactory):
                phase.specific_enthalpy.update_boundary_values(
                    phase_props.h, bg, depth=nt
                )
            if isinstance(phase.viscosity, pp.ad.SurrogateFactory):
                phase.viscosity.update_boundary_values(phase_props.mu, bg, depth=nt)
            if isinstance(phase.thermal_conductivity, pp.ad.SurrogateFactory):
                phase.thermal_conductivity.update_boundary_values(
                    phase_props.kappa, bg, depth=nt
                )

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
            )
            total_mass_mobility = self._bc_value_total_mass_mobility(bg)
            vals[idx] = (component_mass_mobility / total_mass_mobility)[idx]

        return vals

    def bc_values_fractional_flow_energy(self, bg):
        """Computes the values based on the result from the boundary flash, if the flash
        was performed for the boundary grid ``bg``, and inserts it in the cells flagged
        for the boundary equilibrium."""

        vals = super().bc_values_fractional_flow_energy(bg)

        if bg in self.boundary_equilibrium_results:
            idx = self._boundary_equilibrium_cells(bg)
            advected_enthalpy = self._bc_value_advected_enthalpy(bg)
            total_mass_mobility = self._bc_value_total_mass_mobility(bg)
            vals[idx] = (advected_enthalpy / total_mass_mobility)[idx]

        return vals


class InitialConditionsEquilibrium(cf.InitialConditionsCF):
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

        if not np.all(results.converged):
            raise ValueError(f"Initial flash not successful on grid {sd.id}")

        # NOTE Multiple ingores for mypy because the return type of several
        # callables is a general operator, while by logic it is indeed a variable.

        # Initializing values for unknown state functions.
        if results.specification >= pc.FlashSpec.vT:
            self.equation_system.set_variable_values(
                results.p,
                [self.pressure([sd])],  # type: ignore[arg-type]
                iterate_index=0,
            )
        if results.specification not in [pc.FlashSpec.pT, pc.FlashSpec.vT]:
            self.equation_system.set_variable_values(
                results.T,
                [self.temperature([sd])],  # type: ignore[arg-type]
                iterate_index=0,
            )
        if results.specification not in [
            pc.FlashSpec.ph,
            pc.FlashSpec.vh,
        ] and isinstance(self, pp.energy_balance.EnthalpyVariable):
            self.equation_system.set_variable_values(
                results.h,
                [self.enthalpy([sd])],  # type: ignore[arg-type]
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


# mypy: disable-error-code="union-attr"
# NOTE When using the flash it is clear that the properties are surrogate
# factories.
class SolutionStrategyCFLE(cf.SolutionStrategyCF):
    """A solution strategy for compositional flow with local equilibrium conditions in
    the form of algebraic equations.

    Updates of secondary variables and expressions (thermodynamic properties) are
    performed using the provided flash instance.

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

    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`."""
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.energy_balance.VariablesEnergyBalance`."""
    enthalpy: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.energy_balance.EnthalpyVariable`."""

    has_independent_saturation: Callable[[pp.Phase], bool]
    """See :class:`~porepy.compositional.compositional_mixins._MixtureDOFHandler`."""
    has_independent_fraction: Callable[[pp.Phase | pp.Component], bool]
    """See :class:`~porepy.compositional.compositional_mixins._MixtureDOFHandler`."""
    has_independent_partial_fraction: Callable[[pp.Component, pp.Phase], bool]
    """See :class:`~porepy.compositional.compositional_mixins._MixtureDOFHandler`."""
    has_independent_extended_fraction: Callable[[pp.Component, pp.Phase], bool]
    """See :class:`~porepy.compositional.compositional_mixins._MixtureDOFHandler`."""

    def set_materials(self):
        """Sets the flash class after defining the fluid via super-call.

        By default, the unified flash is used, assuming all phases use the same
        (compiled) equation of state.

        If ``params['compile']`` is True (default), both flash and EoS are compiled.

        """
        super().set_materials()

        assert pc.has_equilibrium_specified(self), (
            "Local equilibrium condition not defined in model parameters."
        )

        self.set_flash()

    def set_flash(self) -> None:
        """Sub-routine of :meth:`set_materials` to set the flash class for equilibrium
        calculations, after the fluid is defined."""

        # Import here for runtime reasons of global import (compilation).
        import porepy.compositional.compiled_eos as ceos
        import porepy.compositional.flash as pf

        self.flash = pf.CompiledPersistentVariableFlash(
            self.fluid,
            self.params.get("flash_params", None),  # type:ignore[arg-type]
        )

        if self.params.get("compile", True):
            assert isinstance(self.fluid.reference_phase.eos, ceos.CompiledEoS), (
                "EoS of phases must be instance of CompiledEoS."
            )
            self.flash.compile(*self.params.get("flash_compiler_args", tuple()))

    def update_derived_quantities(self):
        """Normalizes fractional variables in the case of violation of the bound
        [0,1], before calling the base method."""
        self.make_fractions_feasible()
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

    def update_thermodynamic_properties_of_phases(
        self, state: Optional[np.ndarray] = None
    ) -> None:
        """The solution strategy for CF with LE uses this step of the
        algorithm to compute the flash and update the values of thermodynamic
        properties of phases, as well as secondary variables based on the
        flash results.

        This splits the solution strategy into two parts, by resolving the instantaneous
        equilibrium time scale and giving secondary quantities and variables an
        intermediate update by solving the local equilibrium problem with fixed primary
        variables.

        This method loops over all subdomains and and calls :meth:`local_equilibrium`,
        with default arguments.

        Note:
            The update performed here is not an update in the iterative sense.
            It is an update to the values of the current iterate.

        Parameters:
            state: Global state vector to evaluate the flash input from.
                Passed to :meth:`local_equilibrium`.

        """
        stride = self.params.get("flash_params", {}).get("global_iteration_stride", 1)  # type:ignore
        do_flash = False
        if isinstance(stride, int):
            # NOTE Iteration counter is increased after iteration, and 0 modulo anything
            # is zero.
            assert stride > 0, "Global iteration stride must be positive."
            n = self.nonlinear_solver_statistics.num_iteration
            do_flash = (n + 1) % stride == 0 or n == 0

        for sd in self.mdg.subdomains():
            if do_flash:
                self.local_equilibrium(sd, state=state)  # type:ignore
            else:
                self.update_thermodynamic_properties_of_phases_on_grid(sd, state=state)

    def current_fluid_state(
        self,
        subdomains: Sequence[pp.Grid] | pp.Grid,
        state: Optional[np.ndarray] = None,
        feasible_fractions: bool = True,
        eps: float = 1e-7,
    ) -> pc.FluidProperties:
        """Method to assemble the state of the fluid at the current iterate.

        The returned fluid state contains only quantities considered unknowns (fractions
        and equilibrium state functions), and not fluid properties.

        Intended use for the returned fluid property instance is as the initial guess
        for the flash performed in :meth:`local_equilibrium`.

        This method provides room to pre-process data before the flash is called

        Parameters:
            subdomains: One or multiple subdomains in the md-grid.
            state: A global state vector for evaluating the state variables.
            feasible_fractions: If true, fractional quantities are ensured to be bound
                in the interval [0,1] and fulfill the unity constraint.
            eps: Used to bind overall fractions away from zero and detect absent
                phases.

        Returns:
            The base method returns a fluid state containing the current iterate values
            for all fractional variables, as well as pressure, temperature and enthalpy
            (if defined).

        """

        if isinstance(subdomains, pp.Grid):
            subdomains = [subdomains]

        is_persistent = pc.is_persistent_variable_form(self)

        # EPS for fractions of absent phases in persistent form.
        # Used to detect absence of phase (y) and to bind extended fractions away
        # from zero.
        eps_persistent = 1e-5

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
                x_[x_ < 0] = 0.0
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

        p = cast(
            np.ndarray,
            self.equation_system.evaluate(self.pressure(subdomains), state=state),
        )
        T = cast(
            np.ndarray,
            self.equation_system.evaluate(self.temperature(subdomains), state=state),
        )

        if isinstance(self, pp.energy_balance.EnthalpyVariable):
            h = cast(
                np.ndarray,
                self.equation_system.evaluate(self.enthalpy(subdomains), state=state),
            )
        else:
            h = cast(
                np.ndarray,
                self.equation_system.evaluate(
                    self.fluid.specific_enthalpy(subdomains), state=state
                ),
            )

        return pc.FluidProperties(
            z=z,
            y=y,
            sat=sat,
            p=p,
            T=T,
            h=h,
            phases=[
                pc.PhaseProperties(state=phase.state, x=x_)
                for x_, phase in zip(x, self.fluid.phases)
            ],
        )

    def local_equilibrium(
        self,
        sd: pp.Grid,
        state: Optional[np.ndarray] = None,
        specification: Optional[StateSpecDict] = None,
        initial_guess_from_current_state: bool = True,
        update_secondary_variables: bool = True,
    ) -> FlashResults:
        """Performs flash calculations on the given grid and updates the fluid
        properties at the current iterate.

        Performs a full flash (with initial guess), where the flash based on the global
        iterate state did not succeed.

        Calls :meth:`postprocess_equilibrium` at the end.

        See also:
            :meth:`~porepy.compositional.flash.abstract_flash.AbstractFlash.flash`

            :meth:`current_fluid_state`

        Parameters:
            sd: A subdomain in the md-grid.
            state: ``default=None``

                Global state vector to evaluate the equilibrium state functions.
            specification: ``default=None``

                Definition of the equilibrium condition in terms of state functions and
                their values.

                If None, the equilibrium condition is parsed from the model paramters.
            initial_guess_from_current_state: ``default=True``

                If True, the initial fluid state for the flash is evaluated from the
                current solution values at iterate 0.

            update_secondary_variables: ``default=True``

                If True, the flash results are used to update the values of variables
                of the equilibrium problem at iterate 0, additionally to the fluid
                properties.

                Besides updates of various fractions, this includes also an update
                of pressure or temperature for example, if they are not defined in
                ``specification``.

        """

        logger.info(
            f"Equilibration on grid {sd.id} at t={self.time_manager.time:.3e},"
            + f" iter={self.nonlinear_solver_statistics.num_iteration}."
        )
        start = time.time()

        if specification is None:
            model_specs = pc.get_equilibrium_specifications(self)
            flash_spec = [s for s in model_specs if isinstance(s, pc.FlashSpec)][0]

            spec = {}

            if flash_spec < pc.FlashSpec.vT:
                spec["p"] = self.equation_system.evaluate(
                    self.pressure([sd]), state=state
                )
                if flash_spec == pc.FlashSpec.pT:
                    spec["T"] = self.equation_system.evaluate(
                        self.temperature([sd]), state=state
                    )
                elif flash_spec == pc.FlashSpec.ph:
                    if isinstance(self, pp.energy_balance.EnthalpyVariable):
                        _h = self.equation_system.evaluate(
                            self.enthalpy([sd]), state=state
                        )
                    else:
                        _h = self.equation_system.evaluate(
                            self.fluid.specific_enthalpy([sd]), state=state
                        )
                    spec["h"] = _h
                else:
                    raise NotImplementedError(
                        f"Isobaric specification {flash_spec} not yet implemented."
                    )
            else:
                raise NotImplementedError(
                    "Isochoric specifications not yet implemented."
                )

            specification = spec  # type:ignore

        initial_state: pc.FluidProperties | None
        feed: np.ndarray

        if initial_guess_from_current_state:
            initial_state = self.current_fluid_state(sd, state=state)
            feed = initial_state.z
        else:
            initial_state = None
            feed = np.array(
                [
                    self.equation_system.evaluate(comp.fraction([sd]), state=state)
                    for comp in self.fluid.components
                ]
            )

        assert specification is not None, "Failed to specify flash."
        results = self.flash.flash(
            specification,
            [z for z in feed],
            initial_state=initial_state,
            params=self.params.get("flash_params", None),  # type:ignore[arg-type]
        )

        # Perform the full flash where the initial guess from the current state caused
        # failures.
        not_converged = ~results.converged
        if np.any(not_converged) and initial_guess_from_current_state:
            logger.info(
                f"Flash from iterate state failed in {not_converged.sum()} cells on"
                + f" grid {sd.id}. Performing full flash."
            )
            self._full_equilibrium(results, specification)

        self.postprocess_equilibrium(results, sd, state=state)
        results.postprocess_fractions()
        results.evaluate_saturations()
        results.evaluate_extensive_state()

        # Updating fluid properties.
        is_persistent = pc.is_persistent_variable_form(self)
        for phase, phase_state in zip(self.fluid.phases, results.phases):
            # extend derivatives from partial to extended fractions.
            # NOTE The flash returns properties with derivatives w.r.t
            # partial/physical fractions by default. Must be extended since
            # here extended fractions are used (chain rule for normalization)
            # NOTE also, that the progress_* methods with depth 0, don't shift
            # the iterate values, but overwrite only the current one at iterate
            # index 0
            cf.update_phase_properties(
                sd,
                phase,
                phase_state,
                0,
                use_extended_derivatives=is_persistent,
                update_fugacities=True,
            )

        # Updating variables which are also unknowns in the equilibrium problem.
        if update_secondary_variables:
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

                for i, comp in enumerate(phase.components):
                    if self.has_independent_extended_fraction(comp, phase):
                        self.equation_system.set_variable_values(
                            results.phases[j].x[i],
                            [phase.extended_fraction_of[comp]([sd])],  # type: ignore[arg-type]
                            iterate_index=0,
                        )
                    elif self.has_independent_partial_fraction(comp, phase):
                        self.equation_system.set_variable_values(
                            results.phases[j].x[i],
                            [phase.partial_fraction_of[comp]([sd])],  # type: ignore[arg-type]
                            iterate_index=0,
                        )

            # Updating other state functions, if they are were not used to compute the
            # equilibrium.
            if results.specification >= pc.FlashSpec.vT:
                self.equation_system.set_variable_values(
                    results.p,
                    [self.pressure([sd])],  # type: ignore[arg-type]
                    iterate_index=0,
                )
            if results.specification not in [pc.FlashSpec.pT, pc.FlashSpec.vT]:
                self.equation_system.set_variable_values(
                    results.T,
                    [self.temperature([sd])],  # type: ignore[arg-type]
                    iterate_index=0,
                )
            if results.specification not in [
                pc.FlashSpec.ph,
                pc.FlashSpec.vh,
            ] and isinstance(self, pp.energy_balance.EnthalpyVariable):
                self.equation_system.set_variable_values(
                    results.h,
                    [self.enthalpy([sd])],  # type: ignore[arg-type]
                    iterate_index=0,
                )

        logger.debug(
            f"Fluid equilibrated on grid {sd.id}"
            + " (elapsed time: %.5f (s))." % (time.time() - start)
        )

        return results

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
            params=self.params.get("flash_params", None),  # type:ignore[arg-type]
        )

        # Update parent state with sub state values.
        results.T[failure] = sub_results.T
        results.h[failure] = sub_results.h
        results.rho[failure] = sub_results.rho
        results.u[failure] = sub_results.u

        # We count the full flash iterations in addition to the previous ones.
        results.num_iter[failure] += sub_results.num_iter

        # We treat max iter reached as success, and hope for the best globally.
        # sub_results.exitcode[sub_results.exitcode == 1] = 0
        results.exitcode[failure] = sub_results.exitcode

        # Update phase properties.
        for j in range(len(results.phases)):
            results.sat[j][failure] = sub_results.sat[j]
            results.y[j][failure] = sub_results.y[j]

            results.phases[j].x[:, failure] = sub_results.phases[j].x

            results.phases[j].rho[failure] = sub_results.phases[j].rho
            results.phases[j].h[failure] = sub_results.phases[j].h
            results.phases[j].u[failure] = sub_results.phases[j].u
            results.phases[j].mu[failure] = sub_results.phases[j].mu
            results.phases[j].kappa[failure] = sub_results.phases[j].kappa

            results.phases[j].drho[:, failure] = sub_results.phases[j].drho
            results.phases[j].dh[:, failure] = sub_results.phases[j].dh
            results.phases[j].du[:, failure] = sub_results.phases[j].du
            results.phases[j].dmu[:, failure] = sub_results.phases[j].dmu
            results.phases[j].dkappa[:, failure] = sub_results.phases[j].dkappa

            results.phases[j].phis[:, failure] = sub_results.phases[j].phis
            results.phases[j].dphis[:, :, failure] = sub_results.phases[j].dphis

    def postprocess_equilibrium(
        self,
        results: FlashResults,
        sd: pp.Grid,
        state: Optional[np.ndarray] = None,
    ) -> None:
        """A method called by :meth:`local_equilibrium` to post-process failures if
        any.

        The base method asserts that the success flags returned by the flash are zero
        everywhere.

        It allows also to define a flag ``'fall_back_to_iterate'`` in the model
        paramters ```flash_params``` for the flash to fall back to the current iterate
        values where the flash failed.

        Parameters:
            results: The resulting fluid properties and success flags from
                the call to :meth:`local_equilibrium`.
            sd: The grid on which the flash was performed.
            state: A global state vector from which the state variables were evaluated
                for the flash.

        Raises:
            ValueError: If any success flag in ``results`` is not zero.

        """

        if not np.all(results.converged):
            num_fail = (~results.converged).sum()

            if self.params.get("flash_params", {}).get("fallback_to_iterate", False):  # type:ignore
                logger.info(
                    f"Flash failed in {num_fail} cells on grid"
                    + f" {sd.id}. Falling back to previous iterate values."
                )
                self._fall_back_to_current_values(results, sd)
            else:
                raise ValueError(
                    f"Flash strategy failed in {num_fail} / {results.size} cases."
                )

    def _fall_back_to_current_values(
        self,
        results: FlashResults,
        sd: pp.Grid,
    ) -> None:
        """A method to fall back to the current values of the fluid state stored in
        the grid data dictionary.

        Can be used as a last resort if the flash proves to be unsuccessful.
        The global solver can often take care of the convergence issues.

        Paramters:
            results: The flash results where some failures occurred.
            sd: The grid on which the flash was performed. Used to fetch stored iterate
                data and populate the failed entries in ``results``.

        """
        idx = ~results.converged
        subdomains = [sd]
        data = self.mdg.subdomain_data(sd)

        if results.specification not in [pc.FlashSpec.pT, pc.FlashSpec.vT]:
            results.T[idx] = pp.get_solution_values(
                self.temperature_variable, data, iterate_index=0
            )[idx]
        if results.specification >= pc.FlashSpec.vT:
            results.p[idx] = pp.get_solution_values(
                self.pressure_variable, data, iterate_index=0
            )[idx]

        for j, phase in enumerate(self.fluid.phases):
            if phase != self.fluid.reference_phase:
                results.sat[j][idx] = pp.get_solution_values(
                    phase.saturation(subdomains).name, data, iterate_index=0
                )[idx]
                results.y[j][idx] = pp.get_solution_values(
                    phase.fraction(subdomains).name, data, iterate_index=0
                )[idx]

            results.phases[j].rho[idx] = pp.get_solution_values(
                phase.density.name, data, iterate_index=0
            )[idx]
            results.phases[j].h[idx] = pp.get_solution_values(
                phase.specific_enthalpy.name, data, iterate_index=0
            )[idx]
            results.phases[j].mu[idx] = pp.get_solution_values(
                phase.viscosity.name, data, iterate_index=0
            )[idx]
            results.phases[j].kappa[idx] = pp.get_solution_values(
                phase.thermal_conductivity.name, data, iterate_index=0
            )[idx]

            results.phases[j].drho[:, idx] = pp.get_solution_values(
                phase.density._name_derivatives, data, iterate_index=0
            )[:, idx]
            results.phases[j].dh[:, idx] = pp.get_solution_values(
                phase.specific_enthalpy._name_derivatives, data, iterate_index=0
            )[:, idx]
            results.phases[j].dmu[:, idx] = pp.get_solution_values(
                phase.viscosity._name_derivatives, data, iterate_index=0
            )[:, idx]
            results.phases[j].dkappa[:, idx] = pp.get_solution_values(
                phase.thermal_conductivity._name_derivatives, data, iterate_index=0
            )[:, idx]

            for i, comp in enumerate(phase):
                results.phases[j].x[i, idx] = pp.get_solution_values(
                    phase.extended_fraction_of[comp](subdomains).name,
                    data,
                    iterate_index=0,
                )[idx]

                results.phases[j].phis[i, idx] = pp.get_solution_values(
                    phase.fugacity_coefficient_of[comp].name, data, iterate_index=0
                )[idx]
                # NOTE numpy does some weird transpositions when dealing with 3D arrays
                dphi = results.phases[j].dphis[i]
                dphi[:, idx] = pp.get_solution_values(
                    phase.fugacity_coefficient_of[comp]._name_derivatives,
                    data,
                    iterate_index=0,
                )[:, idx]
                results.phases[j].dphis[i, :, :] = dphi

        # reference phase fractions and saturations must be computed, since not stored.
        results.y[self.fluid.reference_phase_index, :] = 1 - np.sum(
            results.y[1:, :], axis=0
        )
        results.sat[self.fluid.reference_phase_index, :] = 1 - np.sum(
            results.sat[1:, :], axis=0
        )


class EnthalpyBasedCFLETemplate(  # type: ignore[misc]
    EnthalpyBasedEquationsCFLE,
    cf.VariablesCF,
    cf.ConstitutiveLawsCF,
    InitialConditionsEquilibrium,
    BoundaryConditionsCFLE,
    SolutionStrategyCFLE,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """Base class for compositional flow with local equilibrium problem in terms of
    pressure and enthalpy."""


class EnthalpyBasedCFFLETemplate(  # type: ignore[misc]
    EnthalpyBasedEquationsCFFLE,
    cf.VariablesCF,
    cf.ConstitutiveLawsCF,
    InitialConditionsEquilibrium,
    BoundaryConditionsCFFLE,
    SolutionStrategyCFLE,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """Base class for compositional flow with local equilibrium problem in terms of
    pressure and enthalpy."""
