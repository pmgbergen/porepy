"""This module contains simplified representations of phases and some components,
as well as a simple composition class to use them.

The phases are modelled using different equations of state,
i.e. this simple representation of a mixture is **not** thermodynamically consistent.

It serves solely for demonstrating what is necessary to set up a composition using the
unified approach.

"""
from __future__ import annotations

from typing import Optional

import numpy as np

import porepy as pp

from ._composite_utils import CP_REF, H_REF, P_REF, R_IDEAL, T_REF, V_REF
from .component import Component
from .composition import Composition
from .peng_robinson import CO2_ps, H2O_ps
from .phase import Phase

__all__ = ["SimpleWater", "SimpleCO2", "SimpleComposition"]


class IncompressibleFluid(Phase):
    """Ideal, incompressible fluid with constant density of 1e6 mol water per ``V_REF``.

    The EoS is

        ``rho = 1e6 / V_REF``,
        ``V = V_REF``.

    """

    rho0: float = 1e6 / V_REF

    def density(self, p, T):
        return pp.ad.Array(
            np.ones(self.ad_system.mdg.num_subdomain_cells()) * self.rho0
        )

    def specific_enthalpy(self, p, T):
        return H_REF + CP_REF * (T - T_REF) + V_REF * (p - P_REF)

    def dynamic_viscosity(self, p, T):
        return pp.ad.Scalar(1.0)

    def thermal_conductivity(self, p, T):
        return pp.ad.Scalar(1.0)

    def fugacity_of(self, p, T, component) -> pp.ad.Operator:
        return pp.ad.Scalar(component.k_value)


class IdealGas(Phase):
    """Ideal gas phase with EoS

    ``rho = n / V  = p / (R * T)``.

    """

    def density(self, p, T):
        # pressure needs to be scaled from MPa to Pa -> *1e6
        # R_IDEAL needs to be rescaled from kJ to J -> * 1e-3
        return p / (T * R_IDEAL) * 1e3

    def specific_enthalpy(self, p, T):
        # enthalpy at reference state is
        # h = u + p / rho(p,T)
        # which due to the ideal gas law simplifies to
        # h = u + R * T
        return H_REF + CP_REF * (T - T_REF)

    def dynamic_viscosity(self, p, T):
        return pp.ad.Scalar(0.1)

    def thermal_conductivity(self, p, T):
        return pp.ad.Scalar(0.1)

    def fugacity_of(self, p, T, component) -> pp.ad.Operator:
        return pp.ad.Scalar(1)


class SimpleWater(Component, H2O_ps):
    """Simple representation of water."""

    k_value: float = 1.2
    """A constant k-value for liquid-gas-equilibrium calculations."""


class SimpleCO2(Component, CO2_ps):
    """Simple representation of Sodium Chloride."""

    k_value: float = 2.0
    """A constant k-value for liquid-gas-equilibrium calculations."""


class SimpleComposition(Composition):
    """A simple composition where

    - the liquid phase is represented by an incompressible fluid
    - the gaseous phase is represented by the ideal gas law
    - constant k-values must be set for each component

    Constant k-values must be set per component using :data:`k_values`.

    The liquid phase ``L`` is set as reference phase, with the other phase being the gas
    phase ``G``.

    Provides additionally an initial guess strategy based on feed fractions using the
    keyword ``initial_guess='feed'`` for :meth:`flash`.

    """

    def __init__(self, ad_system: Optional[pp.ad.EquationSystem] = None) -> None:
        super().__init__(ad_system)

        self._phases: list[Phase] = [
            IncompressibleFluid(self.ad_system, name="L"),
            IdealGas(self.ad_system, name="G"),
        ]
