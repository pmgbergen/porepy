"""Composition class for the Peng-Robinson equation of state."""
from __future__ import annotations

from typing import Optional

import numpy as np

import porepy as pp

from .._composite_utils import R_IDEAL
from ..composition import Composition
from .pr_bip import get_PR_BIP
from .pr_component import PR_Component
from .pr_phase import PR_Phase

__all__ = ["PR_Composition"]


class PR_Composition(Composition):
    """A composition modelled using the Peng-Robinson equation of state

        ``p = R * T / (v - b) - a / (b**2 + 2 * v * b - b**2)``,

    with the characteristic polynomial

        ``Z**3 + (B - 1) * Z**2 + (A - 2 * B - 3 * B**2) * Z + (B**3 + B**2 - A * B) = 0``,

    where

        ``Z = p * v / (R * T)``,
        ``A = p * a / (R * T)**2``,
        ``B = P * b / (R * T)``.

    Mixing rules according to Peng and Robinson are applied to ``a`` and ``b``.

    Note:
        - This class currently supports only a liquid and a gaseous phase.
        - The various properties of this class depend on the thermodynamic state. They are
          continuously re-computed and must therefore be accessed by reference.

    """

    def __init__(self, ad_system: Optional[pp.ad.ADSystem] = None) -> None:
        super().__init__(ad_system)

        # setting of currently supported phases
        self._phases: list[PR_Phase] = [
            PR_Phase(self.ad_system, name="L"),
            PR_Phase(self.ad_system, name="G"),
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
        self._fugacities: dict[PR_Component, dict[PR_Phase, pp.ad.Operator]] = dict()

    def initialize(self) -> None:
        """Before initializing the p-h and p-T subsystems, this method additionally assigns
        callables for thermodynamic properties of phases, according to the equation of state
        and present components, and constructs the attraction and co-volume factors in the EoS.

        After that, it performs a super-call to :meth:`~Composition.initialize` and as a third
        and final step it assigns equilibrium equations in the form of equality of fugacities.

        Raises:
            AssertionError: If the mixture is empty (no components).

        """
        # assert non-empty mixture
        assert self.num_components >= 1

        ## defining the attraction value
        self._assign_attraction()
        ## defining the co-volume
        self._assign_covolume()
        ## setting callables representing the phase densities
        self._assign_phase_densities()
        ## setting callables representing the specific phase enthalpies
        self._assign_phase_enthalpies()
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

    @property  # TODO Z_L and Z_G to Ad_array? Derivatives might be needed for flash
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
        return (
            9 * self.c2 * self.c1 - 27 * self.c0 - 2 * self.c2 * self.c2 * self.c2
        ) / 54

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
        S = np.power(R + np.power(D, 0.5, dtype=complex), 1 / 3, dtype=complex)
        T = np.power(R - np.power(D, 0.5, dtype=complex), 1 / 3, dtype=complex)

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

    def get_equilibrium_equation(self, component: PR_Component) -> pp.ad.Operator:
        """The equilibrium equation for the Peng-Robinson EoS is defined using fugacities

            ``f_cG(p,T,X) * xi_cG - f_cL(p,T,X) * xi_cR = 0``,

        where ``f_cG, f_cR`` are fugacities for component ``c`` in gaseous and liquid phase
        respectively.

        """
        L = self._phases[0]
        G = self._phases[1]

        equation = self._fugacities[component][G] * G.fraction_of_component(
            component
        ) - self._fugacities[component][L] * L.fraction_of_component(component)

        return equation

    def _assign_attraction(self) -> None:
        """Creates the attraction parameter for a mixture according to PR."""
        components: list[PR_Component] = [c for c in self.components]  # type: ignore

        # First we sum over the diagonal elements of the mixture matrix
        a = (
            components[0].fraction
            * components[0].fraction
            * components[0].attraction(self.T)
        )

        if len(components) > 1:
            # sqrt AD function
            sqrt = pp.ad.Function(pp.ad.sqrt, "sqrt")
            # add remaining diagonal elements
            for comp in components[1:]:
                a += comp.fraction * comp.fraction * comp.attraction(self.T)

            # adding off-diagonal elements, including BIPs
            for comp_i in components:
                for comp_j in components:
                    if comp_i != comp_j:
                        a_ij = (
                            comp_i.fraction
                            * comp_j.fraction
                            * sqrt(
                                comp_i.attraction(self.T) * comp_j.attraction(self.T)
                            )
                        )

                        bip, order = get_PR_BIP(comp_i.name, comp_j.name)
                        # assert there is a bip, to appease mypy
                        # this check is performed in add_component though
                        assert bip is not None
                        # call to BIP and multiply with a_ij
                        if order:
                            a_ij *= 1 - bip(comp_i, comp_j)
                        else:
                            a_ij *= 1 - bip(comp_j, comp_i)
                        # add off-diagonal element
                        a += a_ij
            # store attraction parameter
            self._a = a

    def _assign_covolume(self) -> None:
        """Creates the co-volume of the mixture according to VdW mixing rule."""
        components: list[PR_Component] = [c for c in self.components]  # type: ignore

        b = components[0].fraction * components[0].covolume

        if len(components) > 1:
            for comp in components[1:]:
                b += comp.fraction * comp.covolume

        self._b = b

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

    def _assign_phase_enthalpies(self) -> None:  # TODO
        """Constructs callable objects representing phase enthalpies and assings them to the
        ``PR_Phase``-classe."""
        pass

    def _assign_phase_viscosities(self) -> None:  # TODO
        """Constructs callable objects representing phase dynamic viscosities and assigns them
        to the ``PR_Phase``-classes.
        """

        def _mu_L(p, T, *X):
            return pp.ad.Scalar(1.0)

        def _mu_G(p, T, *X):
            return pp.ad.Scalar(1.0)

        # assigning the callable to respective thermodynamic property of the PR_Phase
        self._phases[0]._mu = _mu_L
        self._phases[1]._mu = _mu_G

    def _assign_phase_conductivities(self) -> None:  # TODO
        """Constructs callable objects representing phase conductivities and assigns them
        to the ``PR_Phase``-classes.
        """

        def _kappa_L(p, T, *X):
            return pp.ad.Scalar(1.0)

        def _kappa_G(p, T, *X):
            return pp.ad.Scalar(1.0)

        # assigning the callable to respective thermodynamic property of the PR_Phase
        self._phases[0]._kappa = _kappa_L
        self._phases[1]._kappa = _kappa_G

    def _assign_fugacities(self) -> None:  # TODO
        """Creates and stores operators representing fugacities ``f_ce(p,T,X)`` per component
        per phase."""
        pass
