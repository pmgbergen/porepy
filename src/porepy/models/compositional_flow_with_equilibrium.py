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
from typing import Callable

import numpy as np

import porepy as pp
import porepy.composite as ppc
from porepy.composite.utils_c import extended_compositional_derivatives_v as _extend

from . import compositional_flow as cf
from . import mass_and_energy_balance as mass_energy

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
    :class:`SolutionStrategyCFLE`.

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
    """Provided by :class:`~porepy.composite.equilibrium_mixins.FlashMixin`."""
    flash_params: dict
    """Provided by :class:`~porepy.composite.equilibrium_mixins.FlashMixin`."""

    _boundary_fluid_state: dict[pp.BoundaryGrid, dict[str, np.ndarray]]
    """Provided by :class:`SolutionStrategyCFLE`."""

    def update_all_boundary_conditions(self) -> None:
        """Boundary conditions for problems with equilibrium calculations can be
        done using the provided flash class (see :meth:`boundary_flash`)

        If :attr:`has_time_dependent_boundary_equilibrium` is True,
        this method performs flash calculations on the boundary using the provided
        pressure, temperature and overall fractions.
        It then updates the BC values for all secondary, fractional variables and
        phase properties.

        """
        # cover the base update of parent class and primary CF variables
        mass_energy.BoundaryConditionsFluidMassAndEnergy.update_all_boundary_conditions(
            self
        )
        self.update_essential_boundary_values()

        if self.has_time_dependent_boundary_equilibrium:
            self.boundary_flash()

        # flash results for fractions are stored in the boundary fluid states.
        # Use regular way of updating to acces them.
        self.update_boundary_values_secondary_fractions()

    def boundary_flash(self) -> None:
        """This method performs the p-T flash on the Dirichlet boundary, where pressure
        and temperature are positive.

        The results are stored in the secondary expressions representing thermodynmaic
        properties of phases.

        Results for secondary variables (saturations, relative fractions), are also
        passed to :meth:`update_boundary_condition` in form of lambda functions

        The method can be called any time once the model is initialized, especially for
        non-constant BC.

        Important:
            If p or T are non-positive, the respective secondary expressions are stored
            as zero. Might have some implications for the simulation in weird cases.

        Raises:
            ValueError: If the flash did not succeed everywhere.

        """

        for bg in self.mdg.boundaries():
            vec = np.zeros(bg.num_cells)

            # grids without cells (boundaries of lines) need no fraction values
            # the methods return by default zero arrays of size bg.num_cells
            if bg.num_cells == 0:
                for phase in self.fluid_mixture.phases:
                    phase.density.update_boundary_value(vec.cop(), bg)
                    phase.volume.update_boundary_value(vec.copy(), bg)
                    phase.enthalpy.update_boundary_value(vec.copy(), bg)
                    phase.viscosity.update_boundary_value(vec.copy(), bg)
            # if at least 1 cell, perform flash, update properties and store fraction
            # values
            else:
                sd = bg.parent
                # indexation on boundary grid
                # equilibrium is computable where pressure is given and positive
                dbc = self.bc_type_darcy_flux(sd).is_dir
                # reduce vector with all faces to vector with boundary faces
                bf = self.domain_boundary_sides(sd).all_bf
                dbc = dbc[bf]
                p = self.bc_values_pressure(bg)
                dir_idx = dbc & (p > 0.0)

                # set zero values if not required anywhere (completeness)
                if not np.any(dir_idx):
                    for phase in self.fluid_mixture.phases:
                        phase.density.update_boundary_value(vec.copy(), bg)
                        phase.volume.update_boundary_value(vec.copy(), bg)
                        phase.enthalpy.update_boundary_value(vec.copy(), bg)
                        phase.viscosity.update_boundary_value(vec.copy(), bg)
                else:
                    # BC consistency checks ensure that z, T are non-trivial where p is
                    # non-trivial
                    T = self.bc_values_temperature(bg)
                    feed = [
                        self.bc_values_overall_fraction(comp, bg)
                        for comp in self.fluid_mixture.components
                    ]

                    boundary_state, success, _ = self.flash.flash(
                        z=[z[dir_idx] for z in feed],
                        p=p[dir_idx],
                        T=T[dir_idx],
                        parameters=self.flash_params,
                    )

                    if not np.all(success == 0):
                        raise ValueError("Boundary flash did not succeed.")

                    for j, phase in enumerate(self.fluid_mixture.phases):
                        state_j = boundary_state.phases[j]

                        # store update for saturation values
                        sat_j = vec.copy()
                        sat_j[dir_idx] = boundary_state.sat[j]
                        self._boundary_fluid_state[bg].update(
                            {self._saturation_variable(phase): sat_j}
                        )

                        # store update for relative fractions
                        for k, comp in enumerate(phase.components):
                            x_kj = vec.copy()
                            x_kj[dir_idx] = state_j.x[k]
                            self._boundary_fluid_state[bg].update(
                                {self._relative_fraction_variable(comp, phase): x_kj}
                            )

                        # progress boundary values of phase properties in time
                        val = vec.copy()
                        val[dir_idx] = state_j.rho
                        phase.density.update_boundary_value(val, bg)
                        val = vec.copy()
                        val[dir_idx] = state_j.v
                        phase.volume.update_boundary_value(val, bg)
                        val = vec.copy()
                        val[dir_idx] = state_j.h
                        phase.enthalpy.update_boundary_value(val, bg)
                        val = vec.copy()
                        val[dir_idx] = state_j.mu
                        phase.viscosity.update_boundary_value(val, bg)

    def bc_values_saturation(
        self, phase: ppc.Phase, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        """Method returns the data stored in the boundary fluid states
        (boundary flash results)"""
        if boundary_grid not in self._boundary_fluid_state:
            vals = np.zeros(boundary_grid.num_cells)
        else:
            vals = self._boundary_fluid_state[boundary_grid][
                self._saturation_variable(phase)
            ]
            assert vals.shape == (boundary_grid.num_cells,), (
                f"Mismatch in required phase enthalpy values for phase {phase.name}"
                + f" on boundary {boundary_grid}."
            )
        return vals

    def bc_values_relative_fraction(
        self, component: ppc.Component, phase: ppc.Phase, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        """Method returns the data stored in the boundary fluid states
        (boundary flash results)"""
        if boundary_grid not in self._boundary_fluid_state:
            vals = np.zeros(boundary_grid.num_cells)
        else:
            vals = self._boundary_fluid_state[boundary_grid][
                self._relative_fraction_variable(component, phase)
            ]
            assert vals.shape == (boundary_grid.num_cells,), (
                f"Mismatch in required phase enthalpy values for phase {phase.name}"
                + f" on boundary {boundary_grid}."
            )
        return vals


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
    """Provided by :class:`~porepy.composite.equilibrium_mixins.FlashMixin`."""
    flash_params: dict
    """Provided by :class:`~porepy.composite.equilibrium_mixins.FlashMixin`."""

    def set_initial_values(self) -> None:
        """Changes the order of steps, when compared to the parent method, because the
        secondary variables are computed in the initial flash.

        Shifting of time and iterate values must occur at the end.

        """
        self.set_initial_values_primary_variables()
        self.set_initial_values_secondary_variables()
        self.set_intial_values_phase_properties()
        # updating variable values from current time step, to all previous and iterate
        val = self.equation_system.get_variable_values(iterate_index=0)
        self.equation_system.shift_iterate_values()
        for time_step_index in self.time_step_indices:
            self.equation_system.set_variable_values(
                val,
                time_step_index=time_step_index,
            )

    def set_intial_values_phase_properties(self) -> None:
        """This method hacks into the parent strategy to compute the thermodynamic
        properties, as well as values for secondary variables, based on the initial
        values for primary variables.

        It performes a p-T flash, hence initial conditions for enthalpy are not
        required, but computed by this class.

        Values of properties and secondary variables are stored for all time and
        iterate indices.
        Derivative values for properties are stored at the current iterate.

        """
        subdomains = self.mdg.subdomains()

        for sd in subdomains:
            # pressure, temperature and overall fractions
            p = self.intial_pressure(sd)
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
                        [phase.fraction_of[comp]([sd])],
                        iterate_index=0,
                    )

                # pprogress iterates values to all indices
                for _ in range(self.iterate_indices):
                    phase.density.progress_iterate_values_on_grid(
                        state.phases[j].rho, sd
                    )
                    phase.volume.progress_iterate_values_on_grid(state.phases[j].v, sd)
                    phase.enthalpy.progress_iterate_values_on_grid(
                        state.phases[j].h, sd
                    )
                    phase.viscosity.progress_iterate_values_on_grid(
                        state.phases[j].mu, sd
                    )
                    phase.conductivity.progress_iterate_values_on_grid(
                        state.phases[j].kappa, sd
                    )
                # progress derivative values only once (current iterate)
                phase.density.progress_iterate_derivatives_on_grid(
                    state.phases[j].drho, sd
                )
                phase.volume.progress_iterate_derivatives_on_grid(
                    state.phases[j].dv, sd
                )
                phase.enthalpy.progress_iterate_derivatives_on_grid(
                    state.phases[j].dh, sd
                )
                phase.viscosity.progress_iterate_derivatives_on_grid(
                    state.phases[j].dmu, sd
                )
                phase.conductivity.progress_iterate_derivatives_on_grid(
                    state.phases[j].dkappa, sd
                )

                # fugacities
                for k, comp in enumerate(phase.components):
                    for _ in self.iterate_indices:
                        phase.fugacity_of[comp].progress_iterate_values_on_grid(
                            state.phases[j].phis[k], sd
                        )
                    phase.fugacity_of[comp].progress_iterate_derivatives_on_grid(
                        state.phases[j].dphis[k], sd
                    )

        # progress property values in time on subdomain
        for phase in self.fluid_mixture.phases:
            for _ in self.time_step_indices:
                phase.density.progress_values_in_time(subdomains)
                phase.volume.progress_values_in_time(subdomains)
                phase.enthalpy.progress_values_in_time(subdomains)
                phase.viscosity.progress_values_in_time(subdomains)
                phase.conductivity.progress_values_in_time(subdomains)

                for comp in phase:
                    phase.fugacity_of[comp].progress_values_in_time(subdomains)


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
        eliminated by some constitutive expression, *because the local equilibrium
        problem is the constitutive law*.

    """

    boundary_flash: Callable[[], None]
    """Provided by :class:`BoundaryConditionsCF`"""
    initial_flash: Callable[[], None]
    """Provided by :class:`InitialConditionsCF`."""

    _phase_fraction_variable: Callable[[ppc.Phase], str]
    """Provided by :class:`~porepy.composite.composite_mixins.CompositeVariables`."""
    _saturation_variable: Callable[[ppc.Phase], str]
    """Provided by :class:`~porepy.composite.composite_mixins.CompositeVariables`."""
    _relative_fraction_variable: Callable[[ppc.Component, ppc.Phase], str]
    """Provided by :class:`~porepy.composite.composite_mixins.CompositeVariables`."""

    has_time_dependent_boundary_equilibrium: bool
    """Provided by :class:`BoundaryConditionsCF`"""

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)

        self._boundary_fluid_state: dict[pp.BoundaryGrid, dict[str, np.ndarray]]
        """Data structure to store results from the boundary flash, required for
        advective weights on the boundary."""

        # Input validation for set-up
        if (
            self.equilibrium_type is None
            and self.has_time_dependent_boundary_equilibrium
        ):
            raise ppc.CompositeModellingError(
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

    def initialize_timestep_and_iterate_indices(self) -> None:
        """Propagates the thermodynamic properties of phases, assuming the initial
        flash was performed before.

        In a model with equilibrium conditions and flash calculations, phase
        properties don't need to be registered as secondary expressions with the
        mixin for constitutive laws, since the equilibrium equations *are* the
        constitutive law.

        Note:
            This method progresses values for phase properties
            **on all subdomains**, including

            - phase densities
            - phase volumes
            - phase enthalpies
            - phase viscosities

            Viscosities, conductivities and fugacities are not progressed in time,
            since they are not expected in the accumulation term.

            It also progresses boundary values of phase properties on
            **all boundary grids**, for which boundary values are required.
            That **excludes** the fugacity coefficients.

        """

        subdomains = self.mdg.subdomains()

        # copying the current value of secondary expressions to all indices
        # NOTE Only values, not derivatives
        for phase in self.fluid_mixture.phases:
            # phase properties and their derivatives on each subdomain
            rho_j = phase.density.subdomain_values
            v_j = phase.volume.subdomain_values
            h_j = phase.enthalpy.subdomain_values
            mu_j = phase.viscosity.subdomain_values
            kappa_j = phase.conductivity.subdomain_values

            # all properties have iterate values, use framework from sec. expressions
            # to push back values
            for _ in self.iterate_indices:
                phase.density.subdomain_values = rho_j
                phase.volume.subdomain_values = v_j
                phase.enthalpy.subdomain_values = h_j
                phase.viscosity.subdomain_values = mu_j
                phase.conductivity.subdomain_values = kappa_j

            # all properties have time step values, progress sec. exp. in time
            for _ in self.time_step_indices:
                phase.density.progress_values_in_time(subdomains)
                phase.volume.progress_values_in_time(subdomains)
                phase.enthalpy.progress_values_in_time(subdomains)
            # NOTE viscosity and conductivity are not progressed in time

            # fugacity coeffs
            # NOTE their values are not progressed in time.
            for comp in phase:
                phi = phase.fugacity_of[comp].subdomain_values
                d_phi = phase.fugacity_of[comp].subdomain_derivatives

                for _ in self.iterate_indices:
                    phase.fugacity_of[comp].subdomain_values = phi
                    phase.fugacity_of[comp].subdomain_derivatives = d_phi

            # properties have also (time-dependent) values on boundaries
            # NOTE the different usage of progressing in time on boundaries
            # NOTE fugacities have no boundary values
            bc_rho_j = phase.density.boundary_values
            bc_v_j = phase.volume.boundary_values
            bc_h_j = phase.enthalpy.boundary_values
            bc_mu_j = phase.viscosity.boundary_values
            bc_kappa_j = phase.conductivity.boundary_values
            for _ in self.time_step_indices:
                phase.density.boundary_values = bc_rho_j
                phase.volume.boundary_values = bc_v_j
                phase.enthalpy.boundary_values = bc_h_j
                phase.viscosity.boundary_values = bc_mu_j
                phase.conductivity.boundary_values = bc_kappa_j

    def update_thermodynamic_properties_of_phases(self) -> None:
        """The solution strategy for CF with LE hacks into this step of the
        algorithm to compute the flash and update the values of thermodynamic
        properites of phases, as well as secondary variables based on the
        flash results.

        Hence, this turns the strategy into a non-linear Schur elimination, where
        secondary quantities and variables recieve an intermediate update by
        solving the local equilibrium problem with fixed primary variables.

        """
        fluid = self.postprocess_failures(*self.equilibriate_fluid(None))

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
                    [self._phase_fraction_variable(phase)],
                    iterate_index=0,
                )
                self.equation_system.set_variable_values(
                    fluid.y[j], [self._saturation_variable(phase)], iterate_index=0
                )

            for i, comp in enumerate(phase.components):
                self.equation_system.set_variable_values(
                    fluid.phases[j].x[i],
                    [self._relative_fraction_variable(comp, phase)],
                    iterate_index=0,
                )

        # setting Temperature and pressure values, depending on equilibrium definition
        if "T" not in self.equilibrium_type:
            self.equation_system.set_variable_values(
                fluid.T, [self.temperature_variable], iterate_index=0
            )
        if "p" not in self.equilibrium_type:
            self.equation_system.set_variable_values(
                fluid.p, [self.pressure_variable], iterate_index=0
            )
        # TODO resulting enthalpy can change due to numerics, update as well?

        ### update dependen quantities/ secondary expressions
        for phase, state in zip(self.fluid_mixture.phases, fluid.phases):
            phase.density.subdomain_values = state.rho
            phase.volume.subdomain_values = state.v
            phase.enthalpy.subdomain_values = state.h
            phase.viscosity.subdomain_values = state.mu
            phase.conductivity.subdomain_values = state.kappa

            # extend derivatives from partial to extended fractions.
            # NOTE This revers the hack performed by the composite mixins when creating
            # secondary expressions which depend on extended fractions (independent)
            # quantities, but should actually depend on partial fractions (dependent).
            x = np.array(
                [
                    self.equation_system.get_variable_values(
                        [self._relative_fraction_variable(comp, phase)]
                    )
                    for comp in phase
                ]
            )

            for k, comp in enumerate(phase.components):
                phase.fugacity_of[comp].subdomain_values = state.phis[k]
                phase.fugacity_of[comp].subdomain_derivatives = _extend(
                    state.dphis[k], x
                )

            phase.density.subdomain_derivatives = _extend(state.drho, x)
            phase.volume.subdomain_derivatives = _extend(state.dv, x)
            phase.enthalpy.subdomain_derivatives = _extend(state.dh, x)
            phase.viscosity.subdomain_derivatives = _extend(state.dmu, x)
            phase.conductivity.subdomain_derivatives = _extend(state.dkappa, x)


class CFLEModelMixin_ph(
    EquationsCFLE_ph,
    InitialConditionsCFLE,
    SolutionStrategyCFLE,
    cf.CFModelMixin,
):
    """Base class for compositional flow with local equilibrium problem in terms of
    pressure and enthalpy."""
