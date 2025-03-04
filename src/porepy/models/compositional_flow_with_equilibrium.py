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
from typing import Any, Callable, Optional, Sequence, cast

import numpy as np

import porepy as pp

from . import compositional_flow as cf
from .unified_local_equilibrium import Unified_ph_Equilibrium

logger = logging.getLogger(__name__)


class LocalIsenthalpicEquilibriumEquations(Unified_ph_Equilibrium):
    """Equations for closing compositional flow models with isobaric-isenthalpic equilibrium
    conditions.

    Due to saturations and molar fractions being independent variables, the model is
    closed with local phase mass conservation equations.

    Note:
        Using an independent fluid enthalpy variable, these model equations are suitable
        for both, isenthalpic and isothermal flash procedures.

    """

    has_independent_fraction: Callable[[pp.Phase | pp.Component], bool]
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
            if self.has_independent_fraction(phase):
                equ = self.mass_constraint_for_phase(phase, subdomains)
                self.equation_system.set_equation(equ, subdomains, {"cells": 1})


class EnthalpyBasedEquationsCFLE(
    LocalIsenthalpicEquilibriumEquations,
    cf.PrimaryEquationsCF,
):
    """CFLE model equations with a p-h equilibrium.

    Notably, this model uses the fluid mass balance (non-fractional flow) and the unified p-h
    equilibrium, with a local closure for saturations in the form of mass constraints per
    independent phase.

    """


class EnthalpyBasedEquationsCFFLE(
    LocalIsenthalpicEquilibriumEquations,
    cf.PrimaryEquationsCFF,
):
    """CFFLE model equations with a p-h equilibrium.

    Contrary to :class:`EnthalpyBasedEquationsCFLE`, this collection of equations which uses
    the pressure equation in the fractional-flow formulation, and relies hence on
    re-discretization of fluxes.

    """


class BoundaryConditionsFlash(cf.BoundaryConditionsPhaseProperties):
    """BC mixin for CF models with equilibrium and flash instance.

    This class uses the flash instance to provide BC values for secondary variables
    and thermodynamic properties of phases, using BC values for pressure, temperature and
    overall fractions of components.

    If the BC are not constant, the user needs to flag this in the model parameters and
    this class will perform the boundary flash in every time step to update respective
    values.

    Note:
        As of now, the flash is only performed on the matrix boundary (grid dimension = ambient
        dimension).

    Supports the following model parameters:

    - ``'has_time_dependent_boundary_values'``: Defaults to False.
      A bool indicating whether Dirichlet BC for pressure, temperature or
      feed fractions are time-dependent.

      If True, the boundary equilibrium will be re-computed at the beginning of every
      time step.

    """

    flash: pp.compositional.flash.Flash
    """See :class:`SolutionStrategyFlash`."""

    bc_values_pressure: Callable[[pp.BoundaryGrid], np.ndarray]
    """See :class:`~porepy.models.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow`."""
    bc_values_temperature: Callable[[pp.BoundaryGrid], np.ndarray]
    """See :class:`~porepy.models.energy_balance.BoundaryConditionsEnergy`."""
    bc_values_overall_fraction: Callable[[pp.Component, pp.BoundaryGrid], np.ndarray]
    """See :class:`~porepy.models.compositional_flow.BoundaryConditionsMulticomponent`."""

    has_independent_fraction: Callable[[pp.Component], bool]
    """Provided by mixin for compositional variables."""

    @property
    def _boundary_flash_required(self) -> bool:
        """Internally used flag triggering the boundary flash during prepare simulation and
        in the course of simulations, if BC values are time-dependent."""

        start_of_simulation: bool = (
            self.time_manager.time_init == self.time_manager.time
        )

        if start_of_simulation or self.params.get(
            "has_time_dependent_boundary_values", False
        ):
            return True
        else:
            return False

    @cached_property
    def boundary_flash_results(
        self,
    ) -> dict[pp.BoundaryGrid, pp.compositional.FluidProperties]:
        """The results of the boundary flash are stored here (per boundary grid) for further
        processing."""
        return {}

    def update_boundary_values_phase_properties(self) -> None:
        """Instead of performing the update using underlying EoS, a flash is performed
        to compute the updates for phase properties, as well as for (extended) partial
        fractions and saturations.

        Calls :meth:`boundary_flash` at the beginning of the simulation, and in the course of
        it if BC values are time-dependent.

        """
        if self._boundary_flash_required:
            self.boundary_flash()

    def boundary_flash(self) -> None:
        """This method performs the p-T flash on the boundary of the matrix.

        The results are stored in :meth:`boundary_flash_results`.

        The method can be called any time once the model is initialized, especially for
        non-constant BC.

        Important:
            The flash is performed on the whole boundary. It is up to the user to provide
            values for pressure, temperature and overall fractions even on faces, on which they
            are not used, or to implement a flash class handling zero-values (default values).

            This might change in the future after some work on the BC framework.

        Raises:
            ValueError: If the flash did not succeed everywhere.

        """
        # Matrix = (only) grid with ambient dimension
        sd = self.mdg.subdomains(dim=self.nd)[0]
        bg = self.mdg.subdomain_to_boundary_grid(sd)
        assert bg is not None, "Boundary grid of matrix not found."
        assert bg.num_cells > 0, "Matrix boundary grid has no cells."

        p = self.bc_values_pressure(bg)
        T = self.bc_values_temperature(bg)
        # This is required to uphold the promise of the BC mixin for multi-component models:
        # The bc_values method is only called for independent components
        feed = [
            self.bc_values_overall_fraction(comp, bg)
            for comp in self.fluid.components
            if self.has_independent_fraction(comp)
        ]
        z_r = 1.0 - pp.compositional.safe_sum(feed)
        feed = (
            feed[: self.fluid.reference_component_index]
            + [z_r]
            + feed[self.fluid.reference_component_index :]
        )

        # Performing flash, asserting everything is successful, and storing results.
        logger.debug(f"Computing equilibrium on boundary {bg.id}")
        boundary_state, success, _ = self.flash.flash(
            z=feed,
            p=p,
            T=T,
            params=self.params.get("flash_params", None),
        )

        if not np.all(success == 0):
            raise ValueError("Boundary flash did not succeed.")

        self.boundary_flash_results[bg] = boundary_state


class BoundaryConditionsCFLE(
    # NOTE The order here is critical, since primary variables must be updated first in order
    # for the BC flash to work.
    BoundaryConditionsFlash,
    cf.BoundaryConditionsMulticomponent,
    pp.mass_and_energy_balance.BoundaryConditionsFluidMassAndEnergy,
):
    """BC mixin for CFLE models in the standard formulation (not fractional flow).

    The results of the boundary flash are used to update values of phase properties and
    secondary variables such as partial fractions, which are relevant on the boundary.

    Note:
        This mixin is built on the same assumption as :class:`BoundaryConditionsFlash`, in
        terms of which variables are required on the boundary for the flash. Hence no BC values
        for enthalpy.

    """

    # Provided by CompositionalVariablesMixin
    has_independent_saturation: Callable[[pp.Phase], bool]
    has_independent_partial_fraction: Callable[[pp.Component, pp.Phase], bool]
    has_independent_extended_fraction: Callable[[pp.Component, pp.Phase], bool]
    _saturation_variable: Callable[[pp.Phase], str]
    _partial_fraction_variable: Callable[[pp.Component, pp.Phase], str]

    def update_all_boundary_conditions(self):
        """Updates BC values of phase properties (surrogate operators) and secondary variables
        appearing in the non-linear weights on the boundary.

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
                        self._partial_fraction_variable(phase),
                        bc_values_partial_fraction,
                    )

    def _update_phase_properties_on_boundaries(self, phase: pp.Phase) -> None:
        """Method updating the phase properties of a phase on all boundary grids for
        which results of the boundary flash are stored in :meth:`boundary_flash_results`."""

        nt = self.time_step_indices.size

        assert isinstance(phase.density, pp.ad.SurrogateFactory)
        assert isinstance(phase.specific_enthalpy, pp.ad.SurrogateFactory)
        assert isinstance(phase.viscosity, pp.ad.SurrogateFactory)
        assert isinstance(phase.thermal_conductivity, pp.ad.SurrogateFactory)

        for bg, fluid_props in self.boundary_flash_results.items():
            j = self.fluid.phases.index(phase)
            phase_props = fluid_props.phases[j]
            phase.density.update_boundary_values(phase_props.rho, bg, depth=nt)
            phase.specific_enthalpy.update_boundary_values(phase_props.h, bg, depth=nt)
            phase.viscosity.update_boundary_values(phase_props.mu, bg, depth=nt)
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
            :meth:`boundary_flash_results`, they are returned. Otherwise a zero array is
            returned.

        """
        if bg in self.boundary_flash_results:
            saturations = self.boundary_flash_results[bg].sat
            j = self.fluid.phases.index(phase)
            return saturations[j]
        else:
            return np.zeros(bg.num_cells)

    def bc_values_partial_fraction(
        self, component: pp.Component, phase: pp.Phase, bg: pp.BoundaryGrid
    ) -> np.ndarray:
        """Boundary condition for the (extended) partial fraction of ``component`` in ``phase``.

        This method is called for every (independent) component in every phase.

        Parameters:
            component: A component in the phase.
            phase: A phase in fluid.
            bg: A boundary grid.

        Returns:
            If results are stored for the passed boundary grid in
            :meth:`boundary_flash_results`, they are returned. Otherwise a zero array is
            returned.

        """
        if bg in self.boundary_flash_results:
            j = self.fluid.phases.index(phase)
            i = phase.components.index(component)
            return self.boundary_flash_results[bg].phases[j].x[i]
        else:
            return np.zeros(bg.num_cells)


class BoundaryConditionsCFFLE(
    # NOTE The order here is critical for the functionality. Primary variables must be set
    # first, followed by the BC flash execution. As a last step, the values of fractional flow
    # weights can be assembled.
    cf.BoundaryConditionsFractionalFlow,
    BoundaryConditionsFlash,
    cf.BoundaryConditionsMulticomponent,
    pp.mass_and_energy_balance.BoundaryConditionsFluidMassAndEnergy,
):
    """BC mixin for CFLE models in the fractional flow formulation.

    The results of the boundary flash are used to provide values of the fractional flow weights
    on the boundary.

    """

    # TODO this needs a better solution, depending on how relative_permeability is finally
    # implemented.
    relative_permeability: Callable[..., np.ndarray]

    def _bc_value_component_mass_mobility(
        self, component: pp.FluidComponent, bg: pp.BoundaryGrid
    ) -> np.ndarray:
        """Helper method to evaluate the component mass mobility of a ``component`` on a
        boundary grid.

        Parameters:
            component: A component in the fluid.
            bg: A boundary grid.

        Returns:
            The value of the component mass mobility based on the results of the boundary
            flash.

        """
        fluid_props = self.boundary_flash_results[bg]
        vals = np.zeros(bg.num_cells)

        for j, phase_props in enumerate(zip(fluid_props.phases, self.fluid.phases)):
            props, phase = phase_props
            if component in phase:
                x_ij = cast(
                    np.ndarray, props.x_normalized[phase.components.index(component)]
                )
                vals += (
                    x_ij
                    * props.rho
                    * self.relative_permeability(fluid_props.sat[j])
                    / props.mu
                )

        return vals

    def _bc_value_total_mass_mobility(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Helper method to evaluate the total mass mobility on a boundary grid.

        Parameters:
            bg: A boundary grid.

        Returns:
            The value of the total mass mobility based on the results of the boundary flash.

        """
        fluid_props = self.boundary_flash_results[bg]
        vals = np.zeros(bg.num_cells)

        for j, phase_props in enumerate(fluid_props.phases):
            vals += (
                phase_props.rho
                * self.relative_permeability(fluid_props.sat[j])
                / phase_props.mu
            )

        return vals

    def _bc_value_advected_enthalpy(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Helper method to evaluate the advected enthalpy on a boundary grid.

        Parameters:
            bg: A boundary grid.

        Returns:
            The value of the advected enthalpy based on the results of the boundary flash.

        """
        fluid_props = self.boundary_flash_results[bg]
        vals = np.zeros(bg.num_cells)

        for j, phase_props in enumerate(fluid_props.phases):
            vals += (
                phase_props.h
                * phase_props.rho
                * self.relative_permeability(fluid_props.sat[j])
                / phase_props.mu
            )

        return vals

    def bc_values_fractional_flow_component(self, component, bg):
        """Computes the values based on the result from the boundary flash, if the flash
        was performed for the boundary grid ``bg``.

        Performs a super-call otherwise.

        """

        if bg in self.boundary_flash_results:
            component_mass_mobility = self._bc_value_component_mass_mobility(
                component, bg
            )
            total_mass_mobility = self._bc_value_total_mass_mobility(bg)
            return component_mass_mobility / total_mass_mobility
        else:
            return super().bc_values_fractional_flow_component(component, bg)

    def bc_values_fractional_flow_energy(self, bg):
        """Computes the values based on the result from the boundary flash, if the flash
        was performed for the boundary grid ``bg``.

        Performs a super-call otherwise.

        """

        if bg in self.boundary_flash_results:
            advected_enthalpy = self._bc_value_advected_enthalpy(bg)
            total_mass_mobility = self._bc_value_total_mass_mobility(bg)
            return advected_enthalpy / total_mass_mobility
        else:
            return super().bc_values_fractional_flow_energy(bg)


class InitialConditionsCFLE(cf.InitialConditionsCF):
    """Modified initialization procedure for compositional flow problem with
    equilibrium conditions and a flash instance.

    This class uses the flash to perform the 'initial flash' to calculate values
    for secondary variables and secondary operators representing the thermodynamic
    properties of phases.

    It performs a p-T flash i.e., enthalpy (though primary) is also initialized using the
    flash results.

    """

    flash: pp.compositional.flash.Flash
    """See :class:`SolutionStrategyFlash`."""

    # Provided by CompositionalVariablesMixin
    has_independent_saturation: Callable[[pp.Phase], bool]
    has_independent_fraction: Callable[[pp.Phase], bool]
    has_independent_partial_fraction: Callable[[pp.Component, pp.Phase], bool]
    has_independent_extended_fraction: Callable[[pp.Component, pp.Phase], bool]

    def set_intial_values_phase_properties(self) -> None:
        """Instead of computing the initial values using the underlying EoS, it performs
        the initial flash.

        It performes a p-T flash, hence initial conditions for enthalpy are not
        required, but computed by this class.

        Values for phase properties, as well as secondary fractions and enthalpy are
        then initialized using the results, for all iterate and time step indices.

        Derivative values for properties are stored at the current iterate.

        """
        subdomains = self.mdg.subdomains()

        nt = self.time_step_indices.size
        ni = self.iterate_indices.size
        has_unified_equilibrium = pp.compositional.has_unified_equilibrium(self)

        for grid in subdomains:
            logger.debug(f"Computing initial equilibrium on grid {grid.id}")
            # pressure, temperature and overall fractions
            p = self.ic_values_pressure(grid)
            T = self.ic_values_temperature(grid)
            # IC values for potentially dependent component are never to be called directly
            feed = [
                self.ic_values_overall_fraction(comp, grid)
                for comp in self.fluid.components
            ]
            z_r = 1.0 - pp.compositional.safe_sum(feed)
            feed = (
                feed[: self.fluid.reference_component_index]
                + [z_r]
                + feed[self.fluid.reference_component_index :]
            )

            # computing initial equilibrium
            state, success, _ = self.flash.flash(
                feed, p=p, T=T, params=self.params.get("flash_params", None)
            )

            if not np.all(success == 0):
                raise ValueError(f"Initial equilibriam not successful on grid {grid}")

            # NOTE Multiple ingores for mypy because the return type of several callables is a
            # general Ad operator, while by logic it is indeed a variable.
            # setting initial values for enthalpy
            # NOTE that in the initialization, h is dependent compared to p, T, z
            self.equation_system.set_variable_values(
                state.h,
                [self.enthalpy([grid])],  # type: ignore[arg-type]
                iterate_index=0,
            )

            # setting initial values for all fractional variables and phase properties
            for j, phase in enumerate(self.fluid.phases):
                if self.has_independent_fraction(phase):
                    self.equation_system.set_variable_values(
                        state.y[j],
                        [phase.fraction([grid])],  # type: ignore[arg-type]
                        iterate_index=0,
                    )
                if self.has_independent_saturation(phase):
                    self.equation_system.set_variable_values(
                        state.sat[j],
                        [phase.saturation([grid])],  # type: ignore[arg-type]
                        iterate_index=0,
                    )

                # fractions of component in phase
                for k, comp in enumerate(phase.components):
                    # Extended or partial, one of them is independent
                    if self.has_independent_extended_fraction(comp, phase):
                        self.equation_system.set_variable_values(
                            state.phases[j].x[k],
                            [phase.extended_fraction_of[comp]([grid])],  # type: ignore[arg-type]
                            iterate_index=0,
                        )
                    elif self.has_independent_partial_fraction(comp, phase):
                        self.equation_system.set_variable_values(
                            state.phases[j].x_normalized[k],
                            [phase.partial_fraction_of[comp]([grid])],  # type: ignore[arg-type]
                            iterate_index=0,
                        )

                # Update values and derivatives for current iterate
                # Extend derivatives from partial to extended fractions, in the case of
                # unified equilibrium formulations.
                cf.update_phase_properties(
                    grid,
                    phase,
                    state.phases[j],
                    ni,
                    update_derivatives=True,
                    use_extended_derivatives=has_unified_equilibrium,
                )

                # Appeasing mypy
                assert isinstance(phase.density, pp.ad.SurrogateFactory)
                assert isinstance(phase.specific_enthalpy, pp.ad.SurrogateFactory)
                assert isinstance(phase.viscosity, pp.ad.SurrogateFactory)
                assert isinstance(phase.thermal_conductivity, pp.ad.SurrogateFactory)
                # progress iterates values to all indices
                for _ in self.iterate_indices:
                    phase.density.progress_iterate_values_on_grid(
                        state.phases[j].rho, grid, depth=ni
                    )
                    phase.specific_enthalpy.progress_iterate_values_on_grid(
                        state.phases[j].h, grid, depth=ni
                    )
                    phase.viscosity.progress_iterate_values_on_grid(
                        state.phases[j].mu, grid, depth=ni
                    )
                    phase.thermal_conductivity.progress_iterate_values_on_grid(
                        state.phases[j].kappa, grid, depth=ni
                    )

                # fugacities are not covered by update_phase_properties
                dphis = (
                    state.phases[j].dphis_ext
                    if has_unified_equilibrium
                    else state.phases[j].dphis
                )
                for k, comp in enumerate(phase.components):
                    phi = phase.fugacity_coefficient_of[comp]
                    assert isinstance(phi, pp.ad.SurrogateFactory)
                    for _ in self.iterate_indices:
                        phi.progress_iterate_values_on_grid(
                            state.phases[j].phis[k], grid, depth=ni
                        )
                    phi.set_derivatives_on_grid(dphis[k], grid)

                # progress property values in time on subdomain
                for _ in self.time_step_indices:
                    phase.density.progress_values_in_time([grid], depth=nt)
                    phase.specific_enthalpy.progress_values_in_time([grid], depth=nt)


class SolutionStrategyFlash(pp.PorePyModel):
    """A solution strategy for compositional flow with local equilibrium conditions in
    the unified setting.

    Updates of secondary variables and expressions (thermodynamic properties) are
    performed using the provided flash instance.

    Note:
        CFLE models have extended fractions (as per unified flash assumptions).
        Partial fractions are dependent operators.

    Important:
        Compositional flow models with local equilibrium equations assume that the
        model is closed in the sense that secondary variables are completely determined
        by the local equilibrium equations.

        Hence no secondary variable (as defined by the base variable mixin for CF) is
        eliminated by some constitutive expression.

    Supports the following model parameters:

    - ``'equilibrium_type'``: Defaults to None. If the model contains an equilibrium
      part, it should be a string indicating the fixed state of the local phase
      equilibrium problem e.g., ``'p-T'``,``'p-h'``. The string can also contain other
      qualifiers providing information about the equilibrium model, for example
      ``'unified-p-h'``.
    - ``'flash_params'``: Defaults to None. Parameter dictionary used for flash initialization
      and calling the flash method.

    """

    flash: pp.compositional.flash.Flash
    """The flash class set by this solution strategy."""

    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`."""
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.energy_balance.VariablesEnergyBalance`."""
    enthalpy: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.energy_balance.EnthalpyVariable`."""

    has_independent_saturation: Callable[[pp.Phase], bool]
    """See :class:`~porepy.compositional.compositional_mixins._MixtureDOFHandler`."""
    has_independent_fraction: Callable[[pp.Phase], bool]
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

        assert isinstance(self, pp.SolutionStrategy), (
            "This is a mixin. Require SolutionStrategy as base."
        )
        super().set_materials()  # type:ignore[safe-super]

        assert pp.compositional.get_equilibrium_type(self) is not None, (
            "Equilibrium type not defined in model parameters."
        )

        self.flash = pp.compositional.flash.CompiledUnifiedFlash(
            self.fluid, self.params.get("flash_params", None)
        )

        if self.params.get("compile", True):
            assert isinstance(
                self.fluid.reference_phase.eos, pp.compositional.EoSCompiler
            ), "EoS of phases must be instance of EoSCompiler."
            self.flash.compile()

    def update_thermodynamic_properties_of_phases(self) -> None:
        """The solution strategy for CF with LE uses this step of the
        algorithm to compute the flash and update the values of thermodynamic
        properties of phases, as well as secondary variables based on the
        flash results.

        This splits the solution strategy into two parts, by resolving the instantaneous
        equilibrium time scale and giving secondary quantities and variables an
        intermediate update by solving the local equilibrium problem with fixed primary
        variables.

        Note:
            The update performed here is not an update in the iterative sense.
            It is an update to the values of the current iterate.

        """

        equilibrium_type = str(pp.compositional.get_equilibrium_type(self))
        has_unified_equilibrium = pp.compositional.has_unified_equilibrium(self)

        logger.info(
            f"Updating thermodynamic state of fluid with {equilibrium_type} flash."
        )

        for sd in self.mdg.subdomains():
            logger.debug(f"Flashing on grid {sd.id}")
            start = time.time()
            fluid = self.postprocess_flash(
                sd,
                *self.equilibrate_fluid([sd], None, self.get_fluid_state([sd], None)),
            )
            logger.info(
                f"Fluid equilibriated on grid {sd.id}"
                + " (elapsed time: %.5f (s))." % (time.time() - start)
            )

            ### Updating variables which are unknown to the specific equilibrium type
            for j, phase in enumerate(self.fluid.phases):
                if self.has_independent_fraction(phase):
                    self.equation_system.set_variable_values(
                        fluid.y[j],
                        [phase.fraction([sd])],  # type: ignore[arg-type]
                        iterate_index=0,
                    )
                if self.has_independent_saturation(phase):
                    self.equation_system.set_variable_values(
                        fluid.sat[j],
                        [phase.saturation([sd])],  # type: ignore[arg-type]
                        iterate_index=0,
                    )

                for i, comp in enumerate(phase.components):
                    if self.has_independent_extended_fraction(comp, phase):
                        self.equation_system.set_variable_values(
                            fluid.phases[j].x[i],
                            [phase.extended_fraction_of[comp]([sd])],  # type: ignore[arg-type]
                            iterate_index=0,
                        )
                    elif self.has_independent_partial_fraction(comp, phase):
                        self.equation_system.set_variable_values(
                            fluid.phases[j].x[i],
                            [phase.partial_fraction_of[comp]([sd])],  # type: ignore[arg-type]
                            iterate_index=0,
                        )

            # setting state function values, depending on equilibrium definition
            if "T" not in equilibrium_type:
                self.equation_system.set_variable_values(
                    fluid.T,
                    [self.temperature([sd])],  # type: ignore[arg-type]
                    iterate_index=0,
                )
            if "h" not in equilibrium_type:
                self.equation_system.set_variable_values(
                    fluid.h,
                    [self.enthalpy([sd])],  # type: ignore[arg-type]
                    iterate_index=0,
                )
            if "p" not in equilibrium_type:
                self.equation_system.set_variable_values(
                    fluid.p,
                    [self.pressure([sd])],  # type: ignore[arg-type]
                    iterate_index=0,
                )

            ### update dependen quantities/ secondary expressions
            for phase, state in zip(self.fluid.phases, fluid.phases):
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
                    state,
                    0,
                    update_derivatives=True,
                    use_extended_derivatives=has_unified_equilibrium,
                )

                dphis = state.dphis_ext if has_unified_equilibrium else state.dphis

                for k, comp in enumerate(phase.components):
                    phi = phase.fugacity_coefficient_of[comp]
                    assert isinstance(phi, pp.ad.SurrogateFactory)
                    phi.progress_iterate_values_on_grid(state.phis[k], sd)
                    phi.set_derivatives_on_grid(dphis[k], sd)

    def get_fluid_state(
        self, subdomains: Sequence[pp.Grid], state: Optional[np.ndarray] = None
    ) -> pp.compositional.FluidProperties:
        """Method to assemble a fluid state in the iterative procedure, which
        should be passed to :meth:`equilibrate_fluid`.

        This method provides room to pre-process data before the flash is called with
        the returned fluid state as the initial guess.

        Parameters:
            subdomains: Subdomains for which the state functions should be evaluated
            state: ``default=None``

                Global state vector to be passed to the Ad framework when evaluating the
                current state (fractions, pressure, temperature, enthalpy,..)

        Returns:
            The base method returns a fluid state containing the current iterate value
            of the unknowns of respective flash subproblem (p-T, p-h,...).

        """

        # Extracting the current, iterative state to use as initial guess for the flash
        fluid_state = self._fractional_state_from_vector(subdomains, state)
        equilibrium_type = str(pp.compositional.get_equilibrium_type(self))

        # Evaluate temperature as initial guess, if not fixed in equilibrium type
        if "T" not in equilibrium_type:
            # initial guess for T from iterate
            fluid_state.T = cast(
                np.ndarray,
                self.equation_system.evaluate(
                    self.temperature(subdomains), state=state
                ),
            )
        # evaluate pressure, if volume is fixed. NOTE saturations are also fractions
        # and already included
        if "p" not in equilibrium_type:
            fluid_state.p = cast(
                np.ndarray,
                self.equation_system.evaluate(self.pressure(subdomains), state=state),
            )

        return fluid_state

    def equilibrate_fluid(
        self,
        subdomains: Sequence[pp.Grid],
        state: Optional[np.ndarray] = None,
        initial_fluid_state: Optional[pp.compositional.FluidProperties] = None,
    ) -> tuple[pp.compositional.FluidProperties, np.ndarray]:
        """Convenience method perform the flash based on model specifications.

        This method is called in :meth:`update_thermodynamic_properties_of_phases` to
        use the flash for computing fluid properties and as a predictor for secondary
        variables during nonlinear iterations.

        Parameters:
            subdomains: Subdomains on which to evaluate the target state functions.
            state: ``default=None``

                Global state vector to be passed to the Ad framework when evaluating the
                state functions.
            initial_fluid_state: ``default=None``

                Initial guess passed to :meth:`~porepy.compositional.flash.Flash.flash`.
                Note that if None, the flash computes the initial guess itself.

        Returns:
            The equilibriated state of the fluid and an indicator where the flash was
            successful (or not).

            For more information on the `success`-indicators, see respective flash
            object.

        """

        if initial_fluid_state is None:
            z = np.array(
                [
                    self.equation_system.evaluate(
                        comp.fraction(subdomains), state=state
                    )
                    for comp in self.fluid.components
                ]
            )
        else:
            z = initial_fluid_state.z

        flash_kwargs: dict[str, Any] = {
            "z": z,
            "initial_state": initial_fluid_state,
            "params": self.params.get("flash_params", None),
        }

        equilibrium_type = str(pp.compositional.get_equilibrium_type(self))

        if "p-T" in equilibrium_type:
            flash_kwargs.update(
                {
                    "p": self.equation_system.evaluate(
                        self.pressure(subdomains), state=state
                    ),
                    "T": self.equation_system.evaluate(
                        self.temperature(subdomains), state=state
                    ),
                }
            )
        elif "p-h" in equilibrium_type:
            flash_kwargs.update(
                {
                    "p": self.equation_system.evaluate(
                        self.pressure(subdomains), state=state
                    ),
                    "h": self.equation_system.evaluate(
                        self.enthalpy(subdomains), state=state
                    ),
                }
            )
        # TODO enable once volume is available in code.
        # elif "v-h" in equilibrium_type:
        #     flash_kwargs.update(
        #         {
        #           "v": self.equation_system.evaluate(self.volume(subdomains), state=state),
        #           "h": self.equation_system.evaluate(self.enthalpy(subdomains), state=state),
        #         }
        #     )
        else:
            raise NotImplementedError(
                "Attempting to equilibriate fluid with uncovered equilibrium type"
                + f" {equilibrium_type}."
            )

        result_state, succes, _ = self.flash.flash(**flash_kwargs)

        return result_state, succes

    def postprocess_flash(
        self,
        subdomain: pp.Grid,
        fluid_state: pp.compositional.FluidProperties,
        success: np.ndarray,
    ) -> pp.compositional.FluidProperties:
        """A method called after :meth:`equilibrate_fluid` to post-process failures if
        any.

        The base method asserts that ``success`` is zero everywhere.

        Parameters:
            subdomain: A grid for which ``fluid_state`` contains the values.
            fluid_state: Fluid state returned from :meth:`equilibrate_fluid`.
            success: Success flags returned along the fluid state.

        Returns:
            A final fluid state, with treatment of values where the flash did not
            succeed.

        """
        # nothing to do if everything successful
        if np.all(success == 0):
            return fluid_state
        else:
            raise ValueError(
                "Flash strategy did not succeed in"
                + f" {(success > 0).sum()} / {len(success)} cases."
            )

    def _fractional_state_from_vector(
        self,
        subdomains: Sequence[pp.Grid],
        state: Optional[np.ndarray] = None,
    ) -> pp.compositional.FluidProperties:
        """Uses the AD framework to create a fluid state from currently stored values of
        fractions.

        Convenience function to get the values for fractions in iterative procedures.

        Evaluates:

        1. Overall fractions per component
        2. Fractions per phase
        3. Volumetric fractions per phase (saturations)
        4. Fractions per phase per component
           (extended if equilibrium defined, else partial)

        Parameters:
            state: ``default=None``

                See :meth:`~porepy.numerics.ad.operators.Operator.value`.

        Returns:
            A partially filled fluid state data structure containing the above
            fractional values.

        """

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
                        if pp.compositional.has_unified_equilibrium(self)
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

        return pp.compositional.FluidProperties(
            z=z,
            y=y,
            sat=sat,
            phases=[pp.compositional.PhaseProperties(x=x_) for x_ in x],
        )


class SolutionStrategyCFLE(
    SolutionStrategyFlash,
    cf.SolutionStrategyCF,
):
    """CFLE solution strategy which uses the flash to solve the local equilibrium to
    update phase properties and secondary variables."""


class SolutionStrategyCFFLE(
    SolutionStrategyFlash,
    cf.SolutionStrategyCFF,
):
    """Analogous to :class:`SolutionstrategyCFLE`, but for fractional flow formulations."""


class EnthalpyBasedCFLETemplate(  # type: ignore[misc]
    EnthalpyBasedEquationsCFLE,
    cf.VariablesCF,
    cf.ConstitutiveLawsCF,
    InitialConditionsCFLE,
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
    InitialConditionsCFLE,
    BoundaryConditionsCFFLE,
    SolutionStrategyCFFLE,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """Base class for compositional flow with local equilibrium problem in terms of
    pressure and enthalpy."""
