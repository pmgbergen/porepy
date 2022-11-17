"""Composition class for the Peng-Robinson equation of state."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

import porepy as pp

from ._composite_utils import R_IDEAL
from .composition import Composition
from .component import Component
from .phase import Phase

__all__ = ["PengRobinsonComposition"]


class PR_Phase(Phase):
    """Representation of a phase using the Peng-Robinson EoS.
    
    Thermodynamic properties are represented by references to callables, which are set by the
    Peng-Robinson composition class.

    This class is not intended to be used or instantiated except by the respective composition
    class.

    """

    def __init__(self, name: str, ad_system: pp.ad.ADSystem) -> None:
        super().__init__(name, ad_system)

        self._h: Callable
        self._rho: Callable
        self._mu: Callable
        self._kappa: Callable
    
    def density(self, p, T):
        X = (self.ext_fraction_of_component(component) for component in self)
        return self._rho(p, T, *X)

    def specific_enthalpy(self, p, T):
        X = (self.ext_fraction_of_component(component) for component in self)
        return self._h(p, T, *X)

    def dynamic_viscosity(self, p, T):
        X = (self.ext_fraction_of_component(component) for component in self)
        return self._mu(p, T, *X)

    def thermal_conductivity(self, p, T):
        X = (self.ext_fraction_of_component(component) for component in self)
        return self._kappa(p, T, *X)


class PengRobinsonComposition(Composition):
    """A composition modelled using the Peng-Robinson equation of state

        ``p = R * T / (v - b) - a / (b**2 + 2 * v * b - b**2)``,

    with the characteristic polynomial

        ``Z**3 + (B - 1) * Z**2 + (A - 2 * B - 3 * B**2) * Z + (B**3 + B**2 - A * B) = 0``,

    where

        ``Z = p * v / (R * T)``,
        ``A = p * a / (R * T)**2``,
        ``B = P * b / (R * T)``.

    Mixing rules according to [1] are applied to ``a`` and ``b``.

    Note:
        - This class currently supports only a liquid and a gaseous phase.
        - The various properties of this class depend on the thermodynamic state. They are
          continuously re-computed and must therefore be accessed by reference.

    References:
        [1]: `Peng, Robinson (1976) <https://doi.org/10.1021/i160057a011>`_

    """

    def __init__(self, ad_system: Optional[pp.ad.ADSystem] = None) -> None:
        super().__init__(ad_system)

        # setting of currently supported phases
        self._phases: list[Phase] = [
            PR_Phase("L", self.ad_system),
            PR_Phase("G", self.ad_system),
        ]

        ### PRIVATE
        # (extended) liquid and gas root of the characteristic polynomial
        self._Z_L: np.ndarray
        self._Z_G: np.ndarray

        # attraction and co-volume, assembled during initialization
        # (based on mixing-rule and present components)
        self._a: pp.ad.Operator
        self._b: pp.ad.Operator

        # name of equilibrium equation
        self._fugacity_equation: str = "flash_fugacity"
        # dictionary containing fugacity functions per component per phase
        self._fugacities: dict[Component, dict[Phase, Callable]] = dict()
    
    def initialize(self) -> None:
        """Before initializing the p-h and p-T subsystems, this method additionally assigns
        callables for thermodynamic properties of phases, according to the equation of state
        and present components."""
        ## TODO
        ## defining the attraction value
        ## defining the co-volume
        ## setting callables representing the phase densities
        self._assign_phase_densities()
        ## setting callables representing the specific phase enthalpies
        ## setting callables defining the dynamic viscosity and heat conductivity of phases
        self._assign_phase_viscosities()
        self._assign_phase_conductivities()
        ### settings callables for fugacities
        self._assign_fugacities()

        # super call to initialize p-h and p-T subsystems
        super().initialize()

        ### equilibrium equations
        equations = dict()
        for component in self.components:
            name = f"{self._fugacity_equation}_{component.name}"
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

    ### EoS parameters ------------------------------------------------------------------------

    @property
    def attraction(self) -> pp.ad.Operator:
        """An operator representing ``a`` in the Peng-Robinson EoS."""
        return self._a

    @property
    def covolume(self) -> pp.ad.Operator:
        """An operator representing ``b`` in the Peng-Robinson EoS."""
        return self._b

    @property
    def A(self) -> pp.ad.Operator:
        """An operator representing ``A`` in the characteristic polynomial of the EoS."""
        return (self.attraction * self.p) / (R_IDEAL**2 * self.T * self.T)

    @property
    def B(self) -> pp.ad.Operator:
        """An operator representing ``B`` in the characteristic polynomial of the EoS."""
        return (self.covolume * self.p) / (R_IDEAL * self.T)

    ### EoS root finding methods --------------------------------------------------------------

    @property
    def c2(self) -> pp.ad.Operator:
        """An operator representing the coefficient of the monomial ``Z**2`` in the
        characteristic polynomial."""
        return self.B - 1

    @property
    def c1(self) -> pp.ad.Operator:
        """An operator representing the coefficient of the monomial ``Z`` in the
        characteristic polynomial."""
        return self.A - 3 * self.B * self.B - 2 * self.B

    @property
    def c0(self) -> pp.ad.Operator:
        """An operator representing the coefficient of the monomial ``Z**0`` in the
        characteristic polynomial."""
        # TODO as soon as power overload is available for operators, this needs to change
        # at multiple points in this class
        return self.B * self.B * self.B + self.B * self.B - self.A * self.B

    @property
    def discriminant(self) -> np.ndarray:
        """An operator representing the discriminant of the characteristic polynomial, based
        on the current thermodynamic state.

        The sign of the discriminant can be used to distinguish between 1 and 2-phase regions:

        - ``< 0``: single-phase region
        - ``> 0``: 2-phase region

        Warning:
            The case where the discriminant is zero (or close) is not covered. This corresponds
            to the case where there are 3 real roots and two are equal.
            That can happen mathematically, but the thermodynamic literature is scars on that
            topic.

        """
        return (
            (
                self.c2 * self.c2 * self.c1 * self.c1
                - 4 * self.c1 * self.c1 * self.c1
                - 4 * self.c2 * self.c2 * self.c2 * self.c0
                - 27 * self.c0 * self.c0
                + 18 * self.c2 * self.c1 * self.c0
            )
            .evaluate(self.ad_system.dof_manager)
            .val
        )

    @property
    def Z_L(self) -> np.ndarray:
        """An array representing cell-wise the extended root of the characteristic polynomial
        associated with the liquid phase.

        The values depend on the thermodynamic state and are recomputed internally.

        """
        return self._Z_L

    @property
    def Z_G(self) -> np.ndarray:
        """An array representing cell-wise the extended root of the characteristic polynomial
        associated with the gaseous phase.

        The values depend on the thermodynamic state and are recomputed internally.

        """
        return self._Z_G

    @property
    def _Q(self) -> pp.ad.Operator:
        """Intermediate coefficient for the cubic formula."""
        return (3 * self.c1 - self.c2 * self.c2) / 9

    @property
    def _R(self) -> pp.ad.Operator:
        """Intermediate coefficient for the cubic formula."""
        return (9 * self.c2 * self.c1 - 27 * self.c0 - 2 * self.c2 * self.c2 * self.c2) / 54

    @property
    def _D(self) -> pp.ad.Operator:
        """Modified version of the discriminant for the cubic formula
        (including sign-convention)."""
        return self._Q * self._Q * self._Q + self._R * self._R

    def _compute_roots(self) -> None:
        """(Re-) compute the extended roots based on the current thermodynamic state using
        the cubic formula.

        References:
            [1]: `Cubic formula <https://mathworld.wolfram.com/CubicFormula.html>`_
            [2]: `Ben Gharbia et al. (2021) <https://doi.org/10.1051/m2an/2021075>`_

        """
        R = self._R.evaluate(self.ad_system.dof_manager).val
        D = self._D.evaluate(self.ad_system.dof_manager).val
        c2 = self.c2.evaluate(self.ad_system.dof_manager).val
        S = np.power(R + np.power(D, .5, dtype=complex), 1 / 3, dtype=complex)
        T = np.power(R - np.power(D, .5, dtype=complex), 1 / 3, dtype=complex)

        # based on the sign of the discriminant, we distinguish the regions
        # note that this discriminant uses a different sign convention than the standard one.
        three_root_region = D < 0
        one_root_region = D > 0

        # regular (unextended) roots of the polynomial
        Z1 = -c2 / 3 + S + T
        Z2 = -c2 / 3 - (S + T) / 2 + 1j * np.sqrt(3) / 2 * (S - T)
        Z3 = -c2 / 3 - (S + T) / 2 - 1j * np.sqrt(3) / 2 * (S - T)

        nc = self.ad_system.dof_manager.mdg.num_subdomain_cells()
        # (extended) roots of the liquid and gas phase (per cell)
        Z_L = np.zeros(nc)
        Z_G = np.zeros(nc)

        # this case is not covered by the thermodynamic literature (to my knowing)
        # if it happens, we have to alert the user
        if np.any(np.isclose(D, 0.0)):
            raise RuntimeError(
                "Encountered real double-root in characteristic polynomial."
            )

        # in the three-root-region, the largest root (Z1) corresponds to the gaseous phase
        # the smallest root (Z2) corresponds to the liquid phase
        Z_L[three_root_region] = np.real(Z2[three_root_region])
        Z_G[three_root_region] = np.real(Z1[three_root_region])

        ## Domain extension outside of three-root-region (Ben Gharbia et al.) TODO

        # storing the extended roots for further computations
        self._Z_L = Z_L
        self._Z_G = Z_G

    ### Model equations -----------------------------------------------------------------------

    def get_equilibrium_equation(self, component: Component) -> pp.ad.Operator:
        """The equilibrium equation for the Peng-Robinson EoS is defined using fugacities
        
            ``f_cG(p,T,X) * xi_cG - f_cL(p,T,X) * xi_cR = 0``,
        
        where ``f_cG, f_cR`` are fugacities for component ``c`` in gaseous and liquid phase
        respectively.

        """
        pass

    def _assign_fugacities(self) -> None:
        """Creates and stores callables representing fugacities ``f_ce(p,T,X)`` per component
        per phase."""
        pass

    def _assign_phase_densities(self) -> None:
        """Constructs callable objects representing phase densities and assigns them to the
        ``PR_Phase``-classes."""

        # constructing callable objects with references to the roots
        # densities in cubic EoS are given as inverse of specific molar volume v_e
        # rho_e = 1/ v_e
        # Z_e = p * v_e / (R * T)
        # as of now, the composition X does not influence the density
        def _rho_L(p, T, *X):
            return p / (R_IDEAL * T * self.Z_L)

        def _rho_G(p, T, *X):
            return p / (R_IDEAL * T * self.Z_G)

        # assigning the callable to respective thermodynamic property of the PR_Phase
        self._phases[0]._rho = _rho_L
        self._phases[1]._rho = _rho_G
    
    def _assign_phase_viscosities(self) -> None:
        """Constructs callable objects representing phase dynamic viscosities and assigns them
        to the ``PR_Phase``-classes.
        
        WIP: Assigns currently only unity TODO
        """

        def _mu_L(p, T, *X):
            return pp.ad.Scalar(1.)

        def _mu_G(p, T, *X):
            return pp.ad.Scalar(1.)

        # assigning the callable to respective thermodynamic property of the PR_Phase
        self._phases[0]._mu = _mu_L
        self._phases[1]._mu = _mu_G

    def _assign_phase_conductivities(self) -> None:
        """Constructs callable objects representing phase conductivities and assigns them
        to the ``PR_Phase``-classes.
        
        WIP: Assigns currently only unity TODO
        """

        def _kappa_L(p, T, *X):
            return pp.ad.Scalar(1.)

        def _kappa_G(p, T, *X):
            return pp.ad.Scalar(1.)

        # assigning the callable to respective thermodynamic property of the PR_Phase
        self._phases[0]._kappa = _kappa_L
        self._phases[1]._kappa = _kappa_G