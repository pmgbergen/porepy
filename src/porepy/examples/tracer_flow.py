"""Module containing a simple tracer flow setup, modelled as a single phase, 2-component
flow."""

from __future__ import annotations

from typing import Sequence, TypedDict, cast

import porepy as pp
from porepy.applications.material_values.fluid_values import water
from porepy.compositional.compositional_mixins import CompositionalVariables
from porepy.models.compositional_flow import (
    BoundaryConditionsFractions,
    ComponentMassBalanceEquations,
    InitialConditionsFractions,
    TotalMassBalanceEquation,
)
from porepy.models.fluid_mass_balance import (
    BoundaryConditionsSinglePhaseFlow,
    ConstitutiveLawsSinglePhaseFlow,
    InitialConditionsSinglePhaseFlow,
    SinglePhaseFlow,
    VariablesSinglePhaseFlow,
)
from porepy.models.fluid_property_library import MobilityCF


class WaterDict(TypedDict):
    """For mypy when using **water in combination with FluidComponent."""

    compressibility: float
    density: float
    specific_heat_capacity: float
    thermal_conductivity: float
    thermal_expansion: float
    viscosity: float


class TracerFlowDomain(pp.ModelGeometry):
    """A simple rectangle for left to right flow with a diagonal, non-permeable
    fracture embedded."""


class TracerFluid(pp.FluidMixin):
    """Setting up a 2-component fluid."""

    def get_components(self) -> Sequence[pp.FluidComponent]:
        """Mixed in method defining water as the reference component and a simple
        tracer as the second component."""

        component_1 = pp.FluidComponent(name="water", **cast(WaterDict, water))
        component_2 = pp.FluidComponent(name="tracer")
        return [component_1, component_2]


class TracerFlowConstitutiveLaws(
    MobilityCF,
    ConstitutiveLawsSinglePhaseFlow,
):
    """Atop the constitutive laws for single-phase flow, we add the constitutive laws
    required for the multi-component case with fractional mobilities.

    Note:
        The constitutive laws for the single phase are all based on the reference
        component, which is water in this case.
        The tracer is just a transported quantity in this simple setup without any
        effect on the physics, besides being represented by a overall fraction and
        including a transport equation.

    """


class TracerFlowVariables(
    VariablesSinglePhaseFlow,
    CompositionalVariables,
):
    """Combines the pressure variable from single-phase flow with the fractional
    variables for a multi-component mixture."""


class TracerFlowEquations(TotalMassBalanceEquation, ComponentMassBalanceEquations):
    """The isothermal tracer flow setup requires only the total mass balance equation
    and the transport equation for the independent component."""


class TracerBC(
    BoundaryConditionsFractions,
    BoundaryConditionsSinglePhaseFlow,
):
    """Combination of boundary condition routines for fractions and pressure."""


class TracerIC(
    InitialConditionsFractions,
    InitialConditionsSinglePhaseFlow,
):
    """Combination of initial condition mixins for fractions and pressure."""


class TracerFlowSetup(  # type: ignore[misc]
    TracerFlowDomain,
    TracerFlowVariables,
    TracerFlowEquations,
    TracerBC,
    TracerIC,
    TracerFlowConstitutiveLaws,
    SinglePhaseFlow,
):
    """Complete set-up for tracer flow modelled as a single phase, 2-component flow
    problem."""
