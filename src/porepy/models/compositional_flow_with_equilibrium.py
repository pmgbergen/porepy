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
    closed with local density relations.

    """

    def set_equations(self):
        """Assembles primary balance equations, local equilibrium equations and
        local phase density relations, in that order."""
        cf.PrimaryEquationsCF.set_equations(self)
        ppc.Unified_ph_Equilibrium.set_equations(self)
        self.set_density_relations_for_phases()

    def set_density_relations_for_phases(self) -> None:
        """Method setting the density relation for each independent phase."""
        rphase = self.fluid_mixture.reference_phase
        subdomains = self.mdg.subdomains()
        if self.fluid_mixture.num_phases > 1:
            for phase in self.fluid_mixture.phases:
                if phase == rphase and self.eliminate_reference_phase:
                    continue
                equ = self.density_relation_for_phase(phase, subdomains)
                self.equation_system.set_equation(equ, subdomains, {"cells": 1})


class BoundaryConditionsCFLE(cf.BoundaryConditionsCF):
    """BC mixin for CF models with equilibrium and flash instance.

    This class uses the flash instance to provide BC values for secondary variables
    and thermodynamic properties of phases.

    The 'boundary flash' needs to be performed once by the
    :class:`SolutionStrategyCFLE` during initialization.

    If the BC are not constant, the user needs to flag this and this class will
    perform the boundary flash in every time step to update respective values.

    """

    has_time_dependent_boundary_equilibrium: bool = False
    """A bool indicating whether Dirichlet BC for pressure, temperature or
    feed fractions are time-dependent.

    If True, the boundary equilibrium will be re-computed at the beginning of every
    time step. This is required to provide e.g., values of the advective weights on
    the boundary for upwinding.

    Cannot be True if :attr:`SolutionStrategyCFLE.equilibrium_type` is set to None
    (and hence no flash method was introduced).

    Defaults to False.

    """

    flash: ppc.Flash
    """Provided by :class:`~porepy.compositional.equilibrium_mixins.FlashMixin`."""
    flash_params: dict
    """Provided by :class:`~porepy.compositional.equilibrium_mixins.FlashMixin`."""

    def update_boundary_values_phase_properties(self) -> None:
        """Instead of performing the update using underlying EoS, a flash is performed
        to compute the updates for phase properties, as well as for (extended) partial
        fractions and saturations.

        Calls :meth:`boundary_flash` if :attr:`has_time_dependent_boundary_equilibrium`
        is True.

        """
        if self.has_time_dependent_boundary_equilibrium:
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
            phase_states: list[ppc.PhaseState] = list()
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
                    state_j = ppc.PhaseState()
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
                            self._relative_fraction_variable(comp, phase)
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
                            self._relative_fraction_variable(comp, phase)
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

            if (
                self.eliminate_reference_phase
                and phase == self.fluid_mixture.reference_phase
            ):
                pass
            else:
                var_name = self._saturation_variable(phase)
                s_j_bc = lambda bg: fracs_on_bgs[bg][var_name]
                s_j_bc = cast(Callable[[pp.BoundaryGrid], np.ndarray], s_j_bc)
                self.update_boundary_condition(var_name, s_j_bc)

            for k, comp in enumerate(phase.components):
                var_name = self._relative_fraction_variable(comp, phase)
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
    """Provided by :class:`~porepy.compositional.equilibrium_mixins.FlashMixin`."""
    flash_params: dict
    """Provided by :class:`~porepy.compositional.equilibrium_mixins.FlashMixin`."""

    _relative_fraction_variable: Callable[[ppc.Component, ppc.Phase], str]
    """Provided by :class:`~porepy.compositional.compositional_mixins.CompositeVariables`."""

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

        for sd in subdomains:
            logger.debug(f"Computing initial equilibrium on grid {sd.id}")
            # pressure, temperature and overall fractions
            p = self.initial_pressure(sd)
            T = self.initial_temperature(sd)
            z = [
                self.initial_overall_fraction(comp, sd)
                for comp in self.fluid_mixture.components
            ]

            # computing initial equilibrium
            state, success, _ = self.flash.flash(
                z, p=p, T=T, parameters=self.flash_params
            )

            if not np.all(success == 0):
                raise ValueError(f"Initial equilibriam not successful on grid {sd}")

            # setting initial values for enthalpy
            # NOTE that in the initialization, h is dependent compared to p, T, z
            self.equation_system.set_variable_values(
                state.h, [self.enthalpy([sd])], iterate_index=0
            )

            # setting initial values for all fractional variables and phase properties
            for j, phase in enumerate(self.fluid_mixture.phases):
                # phase fractions and saturations
                if (
                    phase == self.fluid_mixture.reference_phase
                    and self.eliminate_reference_phase
                ):
                    pass  # y and s of ref phase are dependent operators
                else:
                    self.equation_system.set_variable_values(
                        state.y[j], [phase.fraction([sd])], iterate_index=0
                    )
                    self.equation_system.set_variable_values(
                        state.sat[j], [phase.saturation([sd])], iterate_index=0
                    )
                # extended fractions
                for k, comp in enumerate(phase.components):
                    self.equation_system.set_variable_values(
                        state.phases[j].x[k],
                        [phase.extended_fraction_of[comp]([sd])],
                        iterate_index=0,
                    )

                # progress iterates values to all indices
                for _ in self.iterate_indices:
                    phase.density.progress_iterate_values_on_grid(
                        state.phases[j].rho, sd, depth=ni
                    )
                    phase.specific_enthalpy.progress_iterate_values_on_grid(
                        state.phases[j].h, sd, depth=ni
                    )
                    phase.viscosity.progress_iterate_values_on_grid(
                        state.phases[j].mu, sd, depth=ni
                    )
                    phase.conductivity.progress_iterate_values_on_grid(
                        state.phases[j].kappa, sd, depth=ni
                    )
                # progress derivative values only once (current iterate)
                # extend derivatives from partial to extended fractions.
                # NOTE This revers the hack performed by the composite mixins when creating
                # secondary expressions which depend on extended fractions (independent)
                # quantities, but should actually depend on partial fractions (dependent).
                phase.density.progress_iterate_derivatives_on_grid(
                    state.phases[j].drho_ext, sd
                )
                phase.specific_enthalpy.progress_iterate_derivatives_on_grid(
                    state.phases[j].dh_ext, sd
                )
                phase.viscosity.progress_iterate_derivatives_on_grid(
                    state.phases[j].dmu_ext, sd
                )
                phase.conductivity.progress_iterate_derivatives_on_grid(
                    state.phases[j].dkappa_ext, sd
                )

                # fugacities
                dphis_ext = state.phases[j].dphis_ext
                for k, comp in enumerate(phase.components):
                    for _ in self.iterate_indices:
                        phase.fugacity_coefficient_of[
                            comp
                        ].progress_iterate_values_on_grid(
                            state.phases[j].phis[k], sd, depth=ni
                        )
                    phase.fugacity_coefficient_of[
                        comp
                    ].progress_iterate_derivatives_on_grid(dphis_ext[k], sd)

        # progress property values in time on subdomain
        for phase in self.fluid_mixture.phases:
            for _ in self.time_step_indices:
                phase.density.progress_values_in_time(subdomains, depth=nt)
                phase.specific_enthalpy.progress_values_in_time(subdomains, depth=nt)


class SolutionStrategyCFLE(cf.SolutionStrategyCF, ppc.FlashMixin):
    """A solution strategy for compositional flow with local equilibrium conditions.

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

    """

    boundary_flash: Callable[[], None]
    """Provided by :class:`BoundaryConditionsCF`"""
    initial_flash: Callable[[], None]
    """Provided by :class:`InitialConditionsCF`."""

    _phase_fraction_variable: Callable[[ppc.Phase], str]
    """Provided by :class:`~porepy.compositional.compositional_mixins.CompositeVariables`."""
    _saturation_variable: Callable[[ppc.Phase], str]
    """Provided by :class:`~porepy.compositional.compositional_mixins.CompositeVariables`."""
    _relative_fraction_variable: Callable[[ppc.Component, ppc.Phase], str]
    """Provided by :class:`~porepy.compositional.compositional_mixins.CompositeVariables`."""

    has_time_dependent_boundary_equilibrium: bool
    """Provided by :class:`BoundaryConditionsCF`"""

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)

        # Input validation for set-up
        if (
            self.equilibrium_type is None
            and self.has_time_dependent_boundary_equilibrium
        ):
            raise ppc.CompositionalModellingError(
                f"Conflicting model set-up: Time-dependent boundary flash calculations"
                + f" requested but no equilibrium type defined."
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

        Hence, this turns the strategy into a non-linear Schur elimination, where
        secondary quantities and variables recieve an intermediate update by
        solving the local equilibrium problem with fixed primary variables.

        """
        self._log_res("before flash")
        logger.info(
            f"Updating thermodynamic state of fluid with {self.equilibrium_type} flash."
        )
        res = self.equation_system.assemble(evaluate_jacobian=False)
        self._stats[-1][-1]["residual_before_flash"] = np.linalg.norm(res) / np.sqrt(
            res.size
        )
        t_0 = time.time()
        for sd in self.mdg.subdomains():
            logger.debug(f"Flashing on grid {sd.id}")
            start = time.time()
            fluid = self.postprocess_failures(
                sd,
                *self.equilibriate_fluid([sd], None, self.get_fluid_state([sd], None)),
            )
            logger.info(
                f"Fluid equilibriated on grid {sd.id}"
                + " (elapsed time: %.5f (s))." % (time.time() - start)
            )

            ### Updating variables which are unknown to the specific equilibrium type
            for j, phase in enumerate(self.fluid_mixture.phases):
                if (
                    phase == self.fluid_mixture.reference_phase
                    and self.eliminate_reference_phase
                ):
                    pass
                else:
                    self.equation_system.set_variable_values(
                        fluid.sat[j],
                        [phase.saturation([sd])],
                        iterate_index=0,
                    )
                    self.equation_system.set_variable_values(
                        fluid.y[j], [phase.fraction([sd])], iterate_index=0
                    )

                for i, comp in enumerate(phase.components):
                    self.equation_system.set_variable_values(
                        fluid.phases[j].x[i],
                        [phase.extended_fraction_of[comp]([sd])],
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
            # TODO THis update is not an iterative progress, but replaces the values
            # of the current iterate

            ### update dependen quantities/ secondary expressions
            for phase, state in zip(self.fluid_mixture.phases, fluid.phases):
                # extend derivatives from partial to extended fractions.
                # NOTE The flash returns properties with derivatives w.r.t
                # partial/physical fractions by default. Must be extended since
                # here extended fractions are used (chain rule for normalization)
                # NOTE also, that the progress_* methods with depth 0, don't shift
                # the iterate values, but overwrite only the current one at iterate
                # index 0
                phase.density.progress_iterate_values_on_grid(state.rho, sd)
                phase.specific_enthalpy.progress_iterate_values_on_grid(state.h, sd)
                phase.viscosity.progress_iterate_values_on_grid(state.mu, sd)
                phase.conductivity.progress_iterate_values_on_grid(state.kappa, sd)

                phase.density.progress_iterate_derivatives_on_grid(state.drho_ext, sd)
                phase.specific_enthalpy.progress_iterate_derivatives_on_grid(
                    state.dh_ext, sd
                )
                phase.viscosity.progress_iterate_derivatives_on_grid(state.dmu_ext, sd)
                phase.conductivity.progress_iterate_derivatives_on_grid(
                    state.dkappa_ext, sd
                )

                dphis_ext = state.dphis_ext

                for k, comp in enumerate(phase.components):
                    phase.fugacity_coefficient_of[comp].progress_iterate_values_on_grid(
                        state.phis[k], sd
                    )
                    phase.fugacity_coefficient_of[
                        comp
                    ].progress_iterate_derivatives_on_grid(dphis_ext[k], sd)

        t_1 = time.time()
        self._stats[-1][-1]["time_flash"] = t_1 - t_0
        self._log_res("after flash")
        res = self.equation_system.assemble(evaluate_jacobian=False)
        self._stats[-1][-1]["residual_after_flash"] = np.linalg.norm(res) / np.sqrt(
            res.size
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
