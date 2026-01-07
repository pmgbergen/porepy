"""Model mixins for isothermal compositional flow.

This module provides building blocks for setting up isothermal, multiphase,
multi-component flow and transport models in PorePy. It omits the energy
balance equation and related thermal variables/properties, assuming a constant
temperature.

The model is based on:

- A total mass balance equation (pressure equation).
- Component mass balance equations for each independent component.

The primary variables are assumed to be a single pressure and overall fraction variables.
They are collected in :class:`VariablesICF`.

Phase properties (density, viscosity) are handled using SurrogateFactory,
whose values are computed based on pressure and component fractions.

The template :class:`IsothermalCompositionalFlowTemplate` serves as a starting point
for users to define their specific fluid properties and local closure equations.
"""

from __future__ import annotations

import logging
# from functools import partial
from typing import Optional, cast

import numpy as np

import porepy as pp
import porepy.compositional as compositional
from porepy.models.compositional_flow import (
    ComponentMassBalanceEquations,
    BoundaryConditionsMulticomponent,
    BoundaryConditionsPhaseProperties,
    InitialConditionsFractions,
    SolutionStrategySchurComplement,
)


logger = logging.getLogger(__name__)


def update_phase_properties_isothermal(
    sd: pp.Grid,
    phase: pp.Phase,
    props: compositional.PhaseProperties,
    depth: int,
    update_derivatives: bool = True,
) -> None:
    """Helper method to update the phase properties (density, viscosity)
    and its derivatives.

    This is an isothermal adaptation of update_phase_properties from
    compositional_flow.py, excluding enthalpy and thermal conductivity.
    """
    if isinstance(phase.density, pp.ad.SurrogateFactory):
        phase.density.progress_iterate_values_on_grid(props.rho, sd, depth=depth)
        if update_derivatives:
            phase.density.set_derivatives_on_grid(props.drho, sd)
    if isinstance(phase.viscosity, pp.ad.SurrogateFactory):
        phase.viscosity.progress_iterate_values_on_grid(props.mu, sd, depth=depth)
        if update_derivatives:
            phase.viscosity.set_derivatives_on_grid(props.dmu, sd)


# region general PDEs.

# Re-using ComponentMassBalanceEquations from compositional_flow.py
# This class is suitable as it handles component mass balance without explicit
# thermal dependencies in its core logic for advective fluxes.


# endregion
# region Intermediate mixins collecting variables, equations and constitutive laws.


class PrimaryEquationsICF(
    ComponentMassBalanceEquations,
    pp.fluid_mass_balance.FluidMassBalanceEquations,
):
    """A collection of primary equations for isothermal compositional flow.

    They are PDEs consisting of:

    - 1 fluid mass balance equation,
    - mass balance equations per component,

    in this order (reverse order to the base classes).
    """


class VariablesICF(
    compositional.CompositionalVariables,
    pp.fluid_mass_balance.VariablesSinglePhaseFlow,  # Provides pressure
):
    """Bundles standard variables for isothermal flow (pressure) with
    fractional variables."""


class ConstitutiveLawsSolidSkeletonICF(
    pp.constitutive_laws.MassWeightedPermeability,
    pp.constitutive_laws.ConstantPorosity,
    pp.constitutive_laws.ConstantSolidDensity,
):
    """Collection of constitutive laws for the solid skeleton in the
    isothermal compositional flow framework.
    """

    def relative_permeability(
        self, phase: pp.Phase, domains: pp.SubdomainsOrBoundaries
    ) -> pp.ad.Operator:
        """Constitutive law implementing the relative permeability.

        Parameters:
            phase: A phase in the fluid.
            domains: A list of subdomains or boundaries.

        Returns:
            The base class method implements the linear law.
        """
        return phase.saturation(domains)


class ConstitutiveLawsICF(
    compositional.FluidMixin,
    ConstitutiveLawsSolidSkeletonICF,
    pp.constitutive_laws.FluidMobility,
    pp.constitutive_laws.ZeroGravityForce,
    pp.constitutive_laws.SecondOrderTensorUtils,
    pp.constitutive_laws.DimensionReduction,  # Added for mixed-dimensional handling
    pp.constitutive_laws.AdvectiveFlux,
    pp.constitutive_laws.DarcysLaw,
    pp.constitutive_laws.PeacemanWellFlux,
):
    """Constitutive laws for isothermal compositional flow.

    It puts the FluidMixin on top to overwrite the base class treatment of
    thermodynamic phase properties with general surrogate factories provided
    by the fluid mixin (density and viscosity, but not thermal properties).
    """


# endregion
# region Boundary condition mixins.

# Re-using BoundaryConditionsMulticomponent from compositional_flow.py
# This class is suitable for handling component boundary conditions.


class BoundaryConditionsICF(
    # Put on top for proper MRO and handling of multi-component BCs
    BoundaryConditionsPhaseProperties,
    pp.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow,
    BoundaryConditionsMulticomponent,
):
    """Collection of BC values update routines required for Isothermal CF.
    Handles pressure and component-related boundary conditions.
    """


# endregion
# region Initial condition mixins.

# Re-using InitialConditionsFractions from compositional_flow.py
# This class is suitable for handling component fraction initial conditions.


class InitialConditionsPhasePropertiesICF(pp.InitialConditionMixin):
    """Extension of the initial condition mixing to provide a method which
    initializes values and derivative values for phase properties
    (density and viscosity).

    This class assumes that phase properties are given as surrogate factories,
    which can get values assigned after initial values for their dependencies
    are set.

    Allows the user to define ``model.params['phase_property_params']`` which
    are passed as ``params`` to :meth:`~porepy.compositional.base.Phase.compute_properties`.
    """

    def initial_condition(self) -> None:
        """Calls :meth:`set_initial_values_phase_properties` after the
        super-call."""
        super().initial_condition()
        self.set_initial_values_phase_properties()

    def set_initial_values_phase_properties(self) -> None:
        """Method to set the initial values and derivative values of phase
        properties (density and viscosity), which are surrogate factories with
        some dependencies.

        This method also fills all time and iterate indices with the initial values.
        Derivative values are only stored for the current iterate.
        """

        # Set the initial values on individual grids for the iterate indices.
        for sd in self.mdg.subdomains():
            for phase in self.fluid.phases:
                dep_vals = [
                    self.equation_system.evaluate(d([sd]))
                    for d in self.dependencies_of_phase_properties(phase)
                ]

                phase_props = phase.compute_properties(
                    *cast(list[np.ndarray], dep_vals),
                    params=self.params.get("phase_property_params", None),
                )

                # Set current iterate indices of values and derivatives.
                # NOTE: Setting depth to zero does not shift the properties in the
                # iterative sense, but updates only the current iterate.
                update_phase_properties_isothermal(
                    sd, phase, phase_props, 0, update_derivatives=True
                )

    def initialize_previous_iterate_and_time_step_values(self) -> None:
        """Attaches to the iterate and time step initialization and copies the
        values of phase properties found at iterate index 0 to all other
        iterate and time step indices.

        This is done for all phases on all subdomains.

        While iterate indices are copied for all properties, time step indices
        are copied only for density, as it is expected in accumulation terms in
        balance equations.
        """
        super().initialize_previous_iterate_and_time_step_values()  # type:ignore
        ni = self.iterate_indices.size
        nt = self.time_step_indices.size

        for sd in self.mdg.subdomains():
            for phase in self.fluid.phases:
                # Progress iterate values to all iterate indices.
                for _ in self.iterate_indices:
                    if isinstance(phase.density, pp.ad.SurrogateFactory):
                        vals = phase.density.get_values_on_grid(sd, iterate_index=0)
                        phase.density.progress_iterate_values_on_grid(
                            vals, sd, depth=ni
                        )
                    if isinstance(phase.viscosity, pp.ad.SurrogateFactory):
                        vals = phase.viscosity.get_values_on_grid(sd, iterate_index=0)
                        phase.viscosity.progress_iterate_values_on_grid(
                            vals, sd, depth=ni
                        )

                # Copy values to all time step indices.
                # Only density values is copied because it is involved in 
                # time-dependent accumulation terms.
                for _ in self.time_step_indices:
                    if isinstance(phase.density, pp.ad.SurrogateFactory):
                        phase.density.progress_values_in_time([sd], depth=nt)


class InitialConditionsICF(
    # Put this on top because it overrides initial_condition.
    InitialConditionsPhasePropertiesICF,
    pp.fluid_mass_balance.InitialConditionsSinglePhaseFlow,  # For pressure ICs
    InitialConditionsFractions,
):
    """Collection of initialization procedures for the general Isothermal CF model."""


# endregion
# region Solution strategies.

class SolutionStrategyPhasePropertiesICF(pp.PorePyModel):
    """A mixin solution strategy for Isothermal CF models which use surrogate
    operators for phase properties (density and viscosity).

    In this case, the phase properties must be evaluated and respective values
    and derivative values stored. The EoS of each phase is used to perform
    respective evaluation.

    This is a proper mixin providing only overloads of some methods. It is to
    be used in a model on top of a fully functional solution strategy.
    """

    def update_thermodynamic_properties_of_phases(
        self, state: Optional[np.ndarray] = None
    ) -> None:
        """This method uses for each phase the underlying EoS to calculate
        new values and derivative values of phase properties (density, viscosity)
        and to update them in the iterative sense, on all subdomains.

        It is called in :meth:`before_nonlinear_iteration`.
        """

        subdomains = self.mdg.subdomains()

        for grid in subdomains:
            for phase in self.fluid.phases:
                # Compute the values of variables/state functions on which the phase
                # properties depend.
                dep_vals = [
                    self.equation_system.evaluate(d([grid]), state=state)
                    for d in self.dependencies_of_phase_properties(phase)
                ]
                # Compute phase properties using the phase EoS.
                phase_state = phase.compute_properties(
                    *cast(list[np.ndarray], dep_vals),
                    params=self.params.get("phase_property_params", None),
                )

                # Set current iterate indices of values and derivatives.
                # NOTE: Setting depth to zero does not shift the properties in the
                # iterative sense, but updates only the current iterate.
                update_phase_properties_isothermal(
                    grid,
                    phase,
                    phase_state,
                    0,
                    update_derivatives=True,
                )

    def before_nonlinear_iteration(self) -> None:
        """Overwrites parent methods to perform an update of phase properties
        before performing a super-call.

        Fluid properties (surrogate operators) and their values must be updated
        before any re-discretization due to discretizations depending on these
        values. They appear in the non-linear part of various fluxes.
        """
        self.update_thermodynamic_properties_of_phases()
        assert isinstance(self, pp.SolutionStrategy), (
            "This is a mixin. Require SolutionStrategy as base."
        )
        super().before_nonlinear_iteration()  # type:ignore[safe-super]

    def after_nonlinear_convergence(self) -> None:
        """Progresses phase properties (density) in time, if they are surrogate
        factories.

        The progression is performed after the super-call.
        """
        assert isinstance(self, pp.SolutionStrategy), (
            "This is a mixin. Require SolutionStrategy as base."
        )
        super().after_nonlinear_convergence()  # type:ignore[safe-super]

        subdomains = self.mdg.subdomains()
        nt = self.time_step_indices.size
        for phase in self.fluid.phases:
            if isinstance(phase.density, pp.ad.SurrogateFactory):
                phase.density.progress_values_in_time(subdomains, depth=nt)


# Re-using SolutionStrategySchurComplement from compositional_flow.py
# This class is suitable for Schur complement reduction if desired.


class SolutionStrategyICF(
    # NOTE: The MRO order here is critical for the execution of update routines before
    # the linear system is solved.
    SolutionStrategyPhasePropertiesICF,
    SolutionStrategySchurComplement,
    pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow,  # Base solution strategy for mass balance
):
    """Solution strategy for general isothermal compositional flow.

    The generality refers to the fluid phase properties (density and viscosity)
    being surrogate operators. I.e, they are given by some underlying EoS and
    their values must be computed and stored explicitly at several steps in the
    algorithm.

    It uses a mixed-in solution strategy for phase property updates and is based
    on the fully functional solution strategy for fluid mass balance equations.

    Supports the following model parameters:

    - ``'eliminate_reference_phase'``: Defaults to True. If True, the molar
      fraction and saturation of the reference phase are eliminated by unity,
      reducing the size of the system. If False, more work is required by the
      modeller.
    - ``'eliminate_reference_component'``: Defaults to True. If True, the
      overall fraction of the reference component is eliminated by unity,
      reducing the number of unknowns. Also, the mass balance equation for the
      reference component is removed as an equation. If False, the modeller must
      close the system.
    """


# endregion


class IsothermalCompositionalFlowTemplate(  # type: ignore[misc]
    ConstitutiveLawsICF,
    PrimaryEquationsICF,
    VariablesICF,
    BoundaryConditionsICF,
    InitialConditionsICF,
    SolutionStrategyICF,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    """General class for setting up an isothermal multi-phase multi-component flow model,
    with thermodynamic properties of phases being represented as surrogate factories.

    The model can be used as a starting point to add various thermodynamic models and
    correlations (constitutive modelling).

    The primary, transportable variables are:

    - pressure
    - overall fractions per independent component
    - tracer fractions for pure transport without equilibrium (if any)

    The secondary, local variables are:

    - saturations per independent phase
    - phase fractions per independent phase (if any, related to equilibrium formulation)
    - fractions of components in phases (extended or partial)

    The primary block of equations consists of:

    - pressure equation / transport of total mass
    - transport equations for each independent component

    The secondary block of equations must be provided using constitutive relations or an
    equilibrium model for the fluid.

    Important:
        This model is not runable. It is a skeleton for isothermal compositional
        flow. To close it, constitutive modelling is required.

    Note:
        The model inherits the md-treatment of Darcy flux and advective component fluxes.
        Some interface variables and interface equations are introduced there.
        They are treated as secondary equations and variables.
    """
