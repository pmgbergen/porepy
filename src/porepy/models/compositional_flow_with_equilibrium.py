"""Extensions of model mixins for compositional flow (CF) to acount for local
equilibrium (LE) equations and and the usage of a flash instance.

The most important difference is, that in this fully, thermodynamically consistent
formulation of the compositional flow problem, there are no dangling variables.
No separate constitutive modelling is required because thermodynamic properties and
secondary variables are fully determined by the result of the local equilibrium problem.

Also, equilibrium calculations (in the unified setting) introduce extended fractions.
Partial fractions become dependent operators (normalization of extended fractions).

Provides a fully formulated CF model with local equilibrium equations formulated as
a p-h flash.

"""

from __future__ import annotations

import logging
import time
from typing import Callable, cast

import numpy as np

import porepy as pp
import porepy.compositional as ppc

from . import compositional_flow as cf

logger = logging.getLogger(__name__)


class EquationsCFLE_ph(
    cf.PrimaryEquationsCF,
    ppc.Unified_ph_Equilibrium,
):
    """Model equations for compositional flow with isobaric-isenthalpic equilibrium
    conditions.

    Due to saturations and molar fractions being independent variables, the model is
    closed with local phase mass conservation equations.

    Note:
        Using an independent fluid enthalpy variable, these model equations are suitable
        for both, isenthalpic and isothermal flash procedures.

    """

    has_independent_fraction: Callable[[ppc.Phase], bool]
    """See :class:`~porepy.compositional.compositional_mixins._MixtureDOFHandler`."""

    def set_equations(self):
        """Assembles primary balance equations, local equilibrium equations and
        local phase mass conservation, in that order."""
        cf.PrimaryEquationsCF.set_equations(self)
        ppc.Unified_ph_Equilibrium.set_equations(self)
        self.set_mass_conservations_for_phases()

    def set_mass_conservations_for_phases(self) -> None:
        """Method setting the local mass conservation equation for each phase which has
        an independent fraction variable.

        The phase fraction variable usually appears in equilibrium formulations.
        Since saturations are variables as well, the system must be closed by relating
        those two phase-related quantities to each other.

        """
        subdomains = self.mdg.subdomains()
        for phase in self.fluid_mixture.phases:
            if self.has_independent_fraction(phase):
                equ = self.mass_constraint_for_phase(phase, subdomains)
                self.equation_system.set_equation(equ, subdomains, {"cells": 1})


class BoundaryConditionsCFLE(cf.BoundaryConditionsCF):
    """BC mixin for CF models with equilibrium and flash instance.

    This class uses the flash instance to provide BC values for secondary variables
    and thermodynamic properties of phases.

    The 'boundary flash' needs to be performed once by the
    :class:`SolutionStrategyCFLE` during initialization.

    If the BC are not constant, the user needs to flag this in the model parameters and
    this class will perform the boundary flash in every time step to update respective
    values.

    """

    flash: ppc.Flash
    """See :class:`~porepy.compositional.flash.FlashMixin`."""
    flash_params: dict
    """See :class:`~porepy.compositional.flash.FlashMixin`."""

    has_independent_saturation: Callable[[ppc.Phase], bool]
    has_independent_partial_fraction: Callable[[ppc.Component, ppc.Phase], bool]
    has_independent_extended_fraction: Callable[[ppc.Component, ppc.Phase], bool]

    def update_boundary_values_phase_properties(self) -> None:
        """Instead of performing the update using underlying EoS, a flash is performed
        to compute the updates for phase properties, as well as for (extended) partial
        fractions and saturations.

        Calls :meth:`boundary_flash` if the model parameters contains
        ``params['has_time_dependent_boundary_equilibrium'] == True``.

        """
        if self.params.get("has_time_dependent_boundary_equilibrium", False):
            self.boundary_flash()

    def boundary_flash(self) -> None:
        """This method performs the p-T flash on the Dirichlet boundary, where pressure
        and temperature are positive.

        The results are stored in the secondary expressions representing thermodynmaic
        properties of phases.

        Results for secondary variables (saturations, relative fractions), are also
        passed to :meth:`update_boundary_condition` in form of lambda functions.

        The method can be called any time once the model is initialized, especially for
        non-constant BC.

        Important:
            The flash is performed on boundaries where the non-linear advective terms
            are required. As of now, this is indicated by ``is_dir`` from
            :meth:`bc_type_advective_flux`.

            The user must provide values for p, T, and z on those boundaries!

        Raises:
            ValueError: If the flash did not succeed everywhere.

        """

        # structure for storing values of fractional variables on the boundaries
        # used to update time-dependent dense arrays
        fracs_on_bgs: dict[pp.BoundaryGrid, dict[str, np.ndarray]] = dict()
        nt = self.time_step_indices.size

        # First loop to compute values and to set them for thermodynamic properties
        for bg in self.mdg.boundaries():
            logger.debug(f"Computing equilibrium on boundary {bg.id}")

            # populate fractional values with default value of zero
            phase_states: list[ppc.PhaseProperties] = list()
            fracs_on_bgs[bg] = dict()

            # NOTE IMPORTANT: Indicator for boundary cells, where is_dir indicates
            # where values are required for upwinding.
            dir_bc = self.bc_type_advective_flux(bg.parent).is_dir[
                self.domain_boundary_sides(bg.parent).all_bf
            ]

            # set zero values if not required anywhere (completeness)
            if bg.num_cells == 0:
                for phase in self.fluid_mixture.phases:
                    # default values of properties are zero-sized arrays.
                    # So this will not raise an error since bg.num_cells == 0
                    state_j = ppc.PhaseProperties()
                    # NOTE ones to avoid division by zero. Cancelled out anyways.
                    # NOTE I also don't know why numpy is performing the operation on
                    # empty arrays
                    state_j.mu = np.ones(0)
                    phase_states.append(state_j)

                    fracs_on_bgs[bg][self._saturation_variable(phase)] = np.zeros(
                        bg.num_cells
                    )
                    for comp in phase:
                        # NOTE trace amounts to avoid division by zero errors when
                        # evaluationg partial fractions by normalization
                        fracs_on_bgs[bg][
                            self._partial_fraction_variable(comp, phase)
                        ] = (np.ones(bg.num_cells) * 1e-16)
            else:
                assert np.all(dir_bc), "Missing logic in BC conditions for flash"
                p = self.bc_values_pressure(bg)
                T = self.bc_values_temperature(bg)
                feed = [
                    self.bc_values_overall_fraction(comp, bg)
                    for comp in self.fluid_mixture.components
                ]

                boundary_state, success, _ = self.flash.flash(
                    z=[z for z in feed],
                    p=p,
                    T=T,
                    parameters=self.flash_params,
                )

                if not np.all(success == 0):
                    raise ValueError("Boundary flash did not succeed.")

                # storing fractional values on boundaries temporarily, and progressing
                # secondary expressions in time, for which boundary values are required.
                for j, phase in enumerate(self.fluid_mixture.phases):

                    # Update for saturation values
                    fracs_on_bgs[bg][self._saturation_variable(phase)] = (
                        boundary_state.sat[j]
                    )
                    state_j = boundary_state.phases[j]

                    # Update for relative fractions
                    for k, comp in enumerate(phase.components):
                        fracs_on_bgs[bg][
                            self._partial_fraction_variable(comp, phase)
                        ] = state_j.x[k]

                    phase_states.append(state_j)

            # After the states are computes, update the boundary values of phase
            # properties, where boundary values are required
            for phase, state in zip(self.fluid_mixture.phases, phase_states):
                # Update BC values of phase properties in time on boundaries
                phase.density.update_boundary_values(state.rho, bg, depth=nt)
                phase.specific_enthalpy.update_boundary_values(state.h, bg, depth=nt)
                phase.viscosity.update_boundary_values(state.mu, bg, depth=nt)
        # Second loop to call the base method for updating time-dependent dense arrays
        # on boundaries. Used to update values of fractional unknowns, which appear
        # in the advective fluxes.
        # NOTE this loop is done additionally to the first loop, because
        # update_boundary_condition itself loops over all boundaries
        for phase in self.fluid_mixture.phases:

            # BC values for saturations are required in mobility terms.
            if self.has_independent_saturation(phase):
                var_name = self._saturation_variable(phase)
                s_j_bc = lambda bg: fracs_on_bgs[bg][var_name]
                s_j_bc = cast(Callable[[pp.BoundaryGrid], np.ndarray], s_j_bc)
                self.update_boundary_condition(var_name, s_j_bc)

            # BC values for fractions in phase (extended or partial, one of them is
            # independent), are also required for the mobility terms on the BC
            for k, comp in enumerate(phase.components):
                if self.has_independent_extended_fraction(
                    comp, phase
                ) or self.has_independent_partial_fraction(comp, phase):
                    var_name = self._partial_fraction_variable(comp, phase)
                    x_ij_bc = lambda bg: fracs_on_bgs[bg][var_name]
                    x_ij_bc = cast(Callable[[pp.BoundaryGrid], np.ndarray], x_ij_bc)
                    self.update_boundary_condition(var_name, x_ij_bc)


class InitialConditionsCFLE(cf.InitialConditionsCF):
    """Modified initialization procedure for compositional flow problem with
    equilibrium conditions and a flash instance.

    This class uses the flash to perform the 'initial flash' to calculate values
    for secondary variables and secondary expressions representing the thermodynamic
    properties of phases.

    Note:
        Like in the corresponding solution strategy, this class assumes that the model
        is closed in the sense that seconcary variables are not eliminated with some
        constitutive expressions.

    """

    flash: ppc.Flash
    """See :class:`~porepy.compositional.flash.FlashMixin`."""
    flash_params: dict
    """See :class:`~porepy.compositional.flash.FlashMixin`."""

    _has_unified_equilibrium: bool
    has_independent_saturation: Callable[[ppc.Phase], bool]
    has_independent_fraction: Callable[[ppc.Phase], bool]
    has_independent_partial_fraction: Callable[[ppc.Component, ppc.Phase], bool]
    has_independent_extended_fraction: Callable[[ppc.Component, ppc.Phase], bool]

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

        for grid in subdomains:
            logger.debug(f"Computing initial equilibrium on grid {grid.id}")
            # pressure, temperature and overall fractions
            p = self.initial_pressure(grid)
            T = self.initial_temperature(grid)
            z = [
                self.initial_overall_fraction(comp, grid)
                for comp in self.fluid_mixture.components
            ]

            # computing initial equilibrium
            state, success, _ = self.flash.flash(
                z, p=p, T=T, parameters=self.flash_params
            )

            if not np.all(success == 0):
                raise ValueError(f"Initial equilibriam not successful on grid {grid}")

            # setting initial values for enthalpy
            # NOTE that in the initialization, h is dependent compared to p, T, z
            self.equation_system.set_variable_values(
                state.h, [self.enthalpy([grid])], iterate_index=0
            )

            # setting initial values for all fractional variables and phase properties
            for j, phase in enumerate(self.fluid_mixture.phases):
                if self.has_independent_fraction(phase):
                    self.equation_system.set_variable_values(
                        state.y[j], [phase.fraction([grid])], iterate_index=0
                    )
                if self.has_independent_saturation(phase):
                    self.equation_system.set_variable_values(
                        state.sat[j], [phase.saturation([grid])], iterate_index=0
                    )

                # fractions of component in phase
                for k, comp in enumerate(phase.components):
                    # Extended or partial, one of them is independent
                    if self.has_independent_extended_fraction(comp, phase):
                        self.equation_system.set_variable_values(
                            state.phases[j].x[k],
                            [phase.extended_fraction_of[comp]([grid])],
                            iterate_index=0,
                        )
                    elif self.has_independent_partial_fraction(comp, phase):
                        self.equation_system.set_variable_values(
                            state.phases[j].x[k],
                            [phase.partial_fraction_of[comp]([grid])],
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
                    use_extended_derivatives=self._has_unified_equilibrium,
                )

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
                    phase.conductivity.progress_iterate_values_on_grid(
                        state.phases[j].kappa, grid, depth=ni
                    )

                # fugacities are not covered by update_phase_properties
                dphis = (
                    state.phases[j].dphis_ext
                    if self._has_unified_equilibrium
                    else state.phases[j].dphis
                )
                for k, comp in enumerate(phase.components):
                    for _ in self.iterate_indices:
                        phase.fugacity_coefficient_of[
                            comp
                        ].progress_iterate_values_on_grid(
                            state.phases[j].phis[k], grid, depth=ni
                        )
                    phase.fugacity_coefficient_of[comp].set_derivatives_on_grid(
                        dphis[k], grid
                    )

                # progress property values in time on subdomain
                for _ in self.time_step_indices:
                    phase.density.progress_values_in_time([grid], depth=nt)
                    phase.specific_enthalpy.progress_values_in_time([grid], depth=nt)


class SolutionStrategyCFLE(cf.SolutionStrategyCF, ppc.FlashMixin):
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

    New model parameters introduced for instantiation:

    - ``'has_time_dependent_boundary_equilibrium'``: Defaults to False.
      A bool indicating whether Dirichlet BC for pressure, temperature or
      feed fractions are time-dependent.

      If True, the boundary equilibrium will be re-computed at the beginning of every
      time step. This is required to provide e.g., values of the advective weights on
      the boundary for upwinding.

      Cannot be True if :attr:`SolutionStrategyCFLE.equilibrium_type` is set to None
      (and hence no flash method was introduced).

    """

    boundary_flash: Callable[[], None]
    """See :class:`BoundaryConditionsCF`"""
    initial_flash: Callable[[], None]
    """See :class:`InitialConditionsCF`."""

    has_independent_saturation: Callable[[ppc.Phase], bool]
    """See :class:`~porepy.compositional.compositional_mixins._MixtureDOFHandler`."""
    has_independent_fraction: Callable[[ppc.Phase], bool]
    """See :class:`~porepy.compositional.compositional_mixins._MixtureDOFHandler`."""
    has_independent_partial_fraction: Callable[[ppc.Component, ppc.Phase], bool]
    """See :class:`~porepy.compositional.compositional_mixins._MixtureDOFHandler`."""
    has_independent_extended_fraction: Callable[[ppc.Component, ppc.Phase], bool]
    """See :class:`~porepy.compositional.compositional_mixins._MixtureDOFHandler`."""

    _has_unified_equilibrium: bool
    """See :class:`~porepy.compositional.compositional_mixins._MixtureDOFHandler`."""

    def __init__(self, params: dict | None = None) -> None:

        super().__init__(params)

        # Input validation for set-up
        if self.equilibrium_type is None:
            raise ppc.CompositionalModellingError(
                "Using the solution-strategy for CFLE without providing a specification"
                + " of the local equilibrium type."
            )

    def initial_condition(self) -> None:
        """The initialization of the solutionstrategy for compositional flow with
        local equilibrium equations involves setting up the flash instance,
        calling the parent method to set initial conditions,
        and finally computing the initial boundary equilibrium to set values for
        boundary conditions.

        Note:
            For models without time-depenendent BC, the first BC values for various
            secondary expressions need to be set explicitly with the boundary flash.
            This is for performance reasons not done in every time step by the BC mixin.

        """
        self.set_up_flasher()
        super().initial_condition()
        self.boundary_flash()

    def update_thermodynamic_properties_of_phases(self) -> None:
        """The solution strategy for CF with LE uses this step of the
        algorithm to compute the flash and update the values of thermodynamic
        properites of phases, as well as secondary variables based on the
        flash results.

        This splits the solution strategy into two parts, by resolving the instantaneous
        equilibrium time scale and giving secondary quantities and variables an
        intermediate update by solving the local equilibrium problem with fixed primary
        variables.

        Note:
            The update performed here is not an update in the iterative sense.
            It is an update to the values of the current iterate.

        """

        logger.info(
            f"Updating thermodynamic state of fluid with {self.equilibrium_type} flash."
        )

        for sd in self.mdg.subdomains():
            logger.debug(f"Flashing on grid {sd.id}")
            start = time.time()
            fluid = self.postprocess_flash(
                sd,
                *self.equilibriate_fluid([sd], None, self.get_fluid_state([sd], None)),
            )
            logger.info(
                f"Fluid equilibriated on grid {sd.id}"
                + " (elapsed time: %.5f (s))." % (time.time() - start)
            )

            ### Updating variables which are unknown to the specific equilibrium type
            for j, phase in enumerate(self.fluid_mixture.phases):
                if self.has_independent_fraction(phase):
                    self.equation_system.set_variable_values(
                        fluid.y[j], [phase.fraction([sd])], iterate_index=0
                    )
                if self.has_independent_saturation(phase):
                    self.equation_system.set_variable_values(
                        fluid.sat[j], [phase.saturation([sd])], iterate_index=0
                    )

                for i, comp in enumerate(phase.components):
                    if self.has_independent_extended_fraction(comp, phase):
                        self.equation_system.set_variable_values(
                            fluid.phases[j].x[i],
                            [phase.extended_fraction_of[comp]([sd])],
                            iterate_index=0,
                        )
                    elif self.has_independent_partial_fraction(comp, phase):
                        self.equation_system.set_variable_values(
                            fluid.phases[j].x[i],
                            [phase.partial_fraction_of[comp]([sd])],
                            iterate_index=0,
                        )

            # setting state function values, depending on equilibrium definition
            if "T" not in self.equilibrium_type:
                self.equation_system.set_variable_values(
                    fluid.T, [self.temperature([sd])], iterate_index=0
                )
            if "h" not in self.equilibrium_type:
                self.equation_system.set_variable_values(
                    fluid.h, [self.enthalpy([sd])], iterate_index=0
                )
            if "p" not in self.equilibrium_type:
                self.equation_system.set_variable_values(
                    fluid.p, [self.pressure([sd])], iterate_index=0
                )

            ### update dependen quantities/ secondary expressions
            for phase, state in zip(self.fluid_mixture.phases, fluid.phases):
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
                    use_extended_derivatives=self._has_unified_equilibrium,
                )

                # The update of fugacity coefficients is done only in the equilibrium
                # model, hence it is not part of the routine `update_phase_properties`
                dphis = (
                    state.dphis_ext if self._has_unified_equilibrium else state.dphis
                )

                for k, comp in enumerate(phase.components):
                    phase.fugacity_coefficient_of[comp].progress_iterate_values_on_grid(
                        state.phis[k], sd
                    )
                    phase.fugacity_coefficient_of[comp].set_derivatives_on_grid(
                        dphis[k], sd
                    )


class CFLEModelMixin_ph(
    EquationsCFLE_ph,
    InitialConditionsCFLE,
    BoundaryConditionsCFLE,
    SolutionStrategyCFLE,
    cf.CFModelMixin,
):
    """Base class for compositional flow with local equilibrium problem in terms of
    pressure and enthalpy."""
