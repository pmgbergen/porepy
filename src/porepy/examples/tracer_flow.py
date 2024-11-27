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
)
from porepy.models.fluid_mass_balance import SinglePhaseFlow


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


class TracerFluid:
    """Setting up a 2-component fluid."""

    def get_components(self) -> Sequence[pp.FluidComponent]:
        """Mixed in method defining water as the reference component and a simple
        tracer as the second component."""

        component_1 = pp.FluidComponent(name="water", **cast(WaterDict, water))
        component_2 = pp.FluidComponent(name="tracer")
        return [component_1, component_2]


class TracerBC(BoundaryConditionsFractions):
    """Inherits treatmens of BC for fractions and mixes in custom BC for pressure."""


class TracerIC(InitialConditionsFractions):
    """Inherits treatmens of IC for fractions and mixes in IC for pressure."""


class TracerFlowSetup(  # type: ignore[misc]
    TracerFlowDomain,
    TracerFluid,
    CompositionalVariables,
    ComponentMassBalanceEquations,
    TracerBC,
    TracerIC,
    SinglePhaseFlow,
):
    """Complete set-up for tracer flow modelled as a single phase, 2-component flow
    problem."""
