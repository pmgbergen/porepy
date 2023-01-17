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
            np.ones(self.ad_system.dof_manager.mdg.num_subdomain_cells()) * self.rho0
        )

    def specific_enthalpy(self, p, T):
        return H_REF + CP_REF * (T - T_REF) + V_REF * (p - P_REF)

    def dynamic_viscosity(self, p, T):
        return pp.ad.Scalar(1.0)

    def thermal_conductivity(self, p, T):
        return pp.ad.Scalar(1.0)


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


class SimpleWater(Component, H2O_ps):
    """Simple representation of water."""

    pass


class SimpleCO2(Component, CO2_ps):
    """Simple representation of Sodium Chloride."""

    pass


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

    def __init__(self, ad_system: Optional[pp.ad.ADSystem] = None) -> None:
        super().__init__(ad_system)

        self._phases: list[Phase] = [
            IncompressibleFluid(self.ad_system, name="L"),
            IdealGas(self.ad_system, name="G"),
        ]

        ### PUBLIC
        self.k_values: dict[Component, float] = dict()
        """A dictionary containing constant k-values per component.

        The k-values are to be formulated w.r.t the reference phase, i.e.

            ``x_cG - k_c * x_cL = 0``.

        """

    def set_fugacities(self) -> None:
        """Sets constant fugacities for this simple model, where the fugacity in the
        gas phase is set to 1 and the fugacity in the liquid phase is equal the
        set :data:`k_values`."""
        L = self._phases[0]
        G = self._phases[1]

        one = pp.ad.Scalar(1.0)

        for comp in self.components:
            k_val = pp.ad.Scalar(self.k_values[comp])

            self.fugacity_coeffs[comp] = {L: k_val, G: one}

    def print_state(self, from_iterate: bool = False) -> None:
        """Helper method to print the state of the composition to the console."""
        L = self._phases[0]
        G = self._phases[1]
        if from_iterate:
            print("THERMODYNAMIC ITERATE state:")
        else:
            print("THERMODYNAMIC STATE:")
        print("--- thermodynamic INPUT:")
        for C in self.components:
            print(
                C.fraction_name,
                self.ad_system.get_var_values(C.fraction_name, from_iterate),
            )
        print(self.p_name, self.ad_system.get_var_values(self.p_name, from_iterate))
        print(self.T_name, self.ad_system.get_var_values(self.T_name, from_iterate))
        print(self.h_name, self.ad_system.get_var_values(self.h_name, from_iterate))
        print("--- thermodynamic OUTPUT:")
        print(
            L.fraction_name,
            self.ad_system.get_var_values(L.fraction_name, from_iterate),
        )
        print(
            G.fraction_name,
            self.ad_system.get_var_values(G.fraction_name, from_iterate),
        )
        print("---")
        print(
            L.saturation_name,
            self.ad_system.get_var_values(L.saturation_name, from_iterate),
        )
        print(
            G.saturation_name,
            self.ad_system.get_var_values(G.saturation_name, from_iterate),
        )
        print("---")
        for C in self.components:
            name = L.fraction_of_component_name(C)
            print(name, self.ad_system.get_var_values(name, from_iterate))
        for C in self.components:
            name = G.fraction_of_component_name(C)
            print(name, self.ad_system.get_var_values(name, from_iterate))
        print("---")
