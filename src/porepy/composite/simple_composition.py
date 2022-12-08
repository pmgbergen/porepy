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
    - the initial guess strategy ``feed`` based on feed fractions is implemented

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

        # name of equilibrium equation
        self._equilibrium: str = "flash_k-value"

    def initialize(self) -> None:
        """Sets the equilibrium equations using constant k-values,
        after the super-call to the parent-method."""

        super().initialize()

        ### equilibrium equations
        equations = dict()
        for component in self.components:
            name = f"{self._equilibrium}_{component.name}"
            equ = self.get_equilibrium_equation(component)
            equations.update({name: equ})

        # append equation names to both subsystems
        for name in equations.keys():
            self.pT_subsystem["equations"].append(name)
            self.ph_subsystem["equations"].append(name)

        # adding equations to AD system
        image_info = dict()
        for sd in self.ad_system.dof_manager.mdg.subdomains():
            image_info.update({sd: {"cells": 1}})
        for name, equ in equations.items():
            self.ad_system.set_equation(name, equ, num_equ_per_dof=image_info)

    def get_equilibrium_equation(self, component: Component) -> pp.ad.Operator:
        """Constant k-value equations for a given component.

            ``xi_cG - k_c * xi_cL = 0``

        Parameters:
            component: A component in this composition.

        Returns:
            AD operator representing the left-hand side of the equation (rhs=0).

        """
        equation = self._phases[1].fraction_of_component(component) - self.k_values[
            component
        ] * self._phases[0].fraction_of_component(component)
        return equation

    def _set_initial_guess(self, initial_guess: str) -> None:
        """Initial guess strategy based on feed is introduced here."""

        if initial_guess == "feed":
            # use feed fractions as basis for all initial guesses
            feed: dict[Component, np.ndarray] = dict()
            # setting the values for liquid and gas phase composition
            liquid = self._phases[0]
            gas = self._phases[1]
            for component in self.components:
                k_val = self.k_values[component]
                z_c = self.ad_system.get_var_values(component.fraction_name, True)
                feed.update({component: z_c})
                # this initial guess fullfils the k-value equation for component c
                xi_c_L = z_c
                xi_c_V = k_val * xi_c_L

                self.ad_system.set_var_values(
                    liquid.fraction_of_component_name(component),
                    xi_c_L,
                )
                self.ad_system.set_var_values(
                    gas.fraction_of_component_name(component),
                    xi_c_V,
                )
            # for an initial guess for gas fraction we take the feed of the reference component
            # if its only one component, we use 0.5
            if self.num_components == 1:
                y_V = feed[self.reference_component] * 0.5
            else:
                y_V = feed[self.reference_component]
            y_L = 1 - y_V
            self.ad_system.set_var_values(
                liquid.fraction_name,
                y_L,
            )
            self.ad_system.set_var_values(
                gas.fraction_name,
                y_V,
            )
        else:
            super()._set_initial_guess(initial_guess)

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
        for C in self.components:
            name = L.normalized_fraction_of_component_name(C)
            print(name, self.ad_system.get_var_values(name, from_iterate))
        for C in self.components:
            name = G.normalized_fraction_of_component_name(C)
            print(name, self.ad_system.get_var_values(name, from_iterate))
        print("---")
