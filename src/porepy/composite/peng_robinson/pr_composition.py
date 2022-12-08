"""Composition class for the Peng-Robinson equation of state."""
from __future__ import annotations

from typing import Generator, Optional

import numpy as np

import porepy as pp

from .._composite_utils import H_REF, R_IDEAL
from ..composition import Composition
from .pr_bip import get_PR_BIP
from .pr_component import PR_Component
from .pr_phase import PR_Phase
from .pr_roots import PR_Roots
from .pr_utils import _log, _power, _sqrt

__all__ = ["PR_Composition"]

_div_sqrt = pp.ad.Scalar(-1 / 2)


class PR_Composition(Composition):
    """A composition modelled using the Peng-Robinson equation of state.

        ``p = R * T / (v - b) - a / (b**2 + 2 * v * b - b**2)``.

    Mixing rules according to Peng and Robinson are applied to ``a`` and ``b``.

    Note:
        - This class currently supports only a liquid and a gaseous phase.
        - The various properties of this class depend on the thermodynamic state. They are
          continuously re-computed and must therefore be accessed by reference.

    """

    def __init__(self, ad_system: Optional[pp.ad.ADSystem] = None) -> None:
        super().__init__(ad_system)

        self.roots: PR_Roots
        """The roots of the cubic equation of state.

        This object is instantiated during initialization.

        """

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
        self._dT_a: pp.ad.Operator
        self._b: pp.ad.Operator

        # name of equilibrium equation
        self._fugacity_equation: str = "flash_fugacity_PR"
        # dictionary containing fugacity functions per component per phase
        self._fugacities: dict[PR_Component, dict[PR_Phase, pp.ad.Operator]] = dict()

    def add_component(self, component: PR_Component | list[PR_Component]) -> None:
        """This child class method checks additionally if BIPs are defined for components to be
        added and components already added.

        For the Peng-Robinson EoS to work as intended, BIPs must be available for any
        combination of two present components and present in
        :data:`~porepy.composite.peng_robinson.pr_bip.PR_BIP_MAP`.

        Parameters:
            component: one or multiple model (PR-) components for this EoS.

        Raises:
            NotImplementedError: If a BIP is not available for any combination of modelled
                components.

        """
        if isinstance(component, PR_Component):
            component = [component]  # type: ignore

        missing_bips: list[tuple[str, str]] = list()

        # check for missing bips between new components and present components
        for comp_new in component:
            for comp_present in self.components:
                # there is no bip between a component and itself
                if comp_new != comp_present:
                    bip, *_ = get_PR_BIP(comp_new.name, comp_present.name)
                    # if bip is not available, add pair to missing bips
                    if bip is None:
                        missing_bips.append((comp_new.name, comp_present.name))

        # check for missing bips between new components
        for comp_1 in component:
            for comp_2 in self.components:
                # no bip between a component and itself
                if comp_2 != comp_1:
                    bip, *_ = get_PR_BIP(comp_1.name, comp_2.name)
                    if bip is None:
                        missing_bips.append((comp_1.name, comp_2.name))

        # if missing bips detected, raise error
        if missing_bips:
            raise NotImplementedError(
                f"BIPs not available for the following component-pairs:\n\t{missing_bips}"
            )
        # if no missing bips, we proceed adding the components using the parent method.
        super().add_component(component)

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
        ## settings callables for fugacities
        self._assign_fugacities()

        ## create roots object
        self.roots = PR_Roots(self.ad_system, self.A, self.B)

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

    @property
    def components(self) -> Generator[PR_Component, None, None]:
        """Function overload to specify special return type for PR compositions."""
        for C in self._components:
            yield C

    ### EoS parameters ------------------------------------------------------------------------

    @property
    def attraction(self) -> pp.ad.Operator:
        """An operator representing ``a`` in the Peng-Robinson EoS."""
        return self._a

    @property
    def dT_attraction(self) -> pp.ad.Operator:
        """An operator representing the derivative of :meth:`attraction` w.r.t. temperature."""
        return self._dT_a

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
        """Creates the attraction parameter for a mixture according to PR, as well as its
        derivative w.r.t. temperature."""
        components: list[PR_Component] = [c for c in self.components]  # type: ignore

        # First we sum over the diagonal elements of the mixture matrix,
        # starting with the first component
        comp_0 = components[0]
        a = self._vdW_a_ij(comp_0, comp_0)
        dT_a = self._vdW_dT_a_ij(comp_0, comp_0)

        if len(components) > 1:
            # add remaining diagonal elements
            for comp in components[1:]:
                a += self._vdW_a_ij(comp, comp)
                dT_a += self._vdW_dT_a_ij(comp, comp)

            # adding off-diagonal elements, including BIPs
            for comp_i in components:
                for comp_j in components:
                    if comp_i != comp_j:
                        # computing the attraction between components i and j
                        a += self._vdW_a_ij(comp_i, comp_j)
                        # computing the derivative w.r.t temperature of attraction terms
                        dT_a += self._vdW_dT_a_ij(comp_i, comp_j)

        # store attraction parameters
        self._a = a
        self._dT_a = dT_a

    def _vdW_a_ij(
        self,
        comp_i: PR_Component,
        comp_j: PR_Component,
        multiply_fractions: bool = True,
    ) -> pp.ad.Operator:
        """Get ``z_i * z_j * a_ij`` from the van der Waals - mixing rule

            ``a = sum_i sum_j z_i * z_j * sqrt(a_i * a_j) * (1 - delta_ij)``,

        where ``delta_ij`` is the BIP for ``i!=j``, and 0 for ``i==j``.

        """
        # for different components, the expression is more complex
        if comp_i != comp_j:
            # first part without BIP
            a_ij = _sqrt(comp_i.attraction(self.T) * comp_j.attraction(self.T))
            # get bip
            bip, _, order = get_PR_BIP(comp_i.name, comp_j.name)
            # assert there is a bip, to appease mypy
            # this check is performed in add_component anyways
            assert bip is not None

            # call to BIP and multiply with a_ij
            if order:
                a_ij *= 1 - bip(self.T, comp_i, comp_j)
            else:
                a_ij *= 1 - bip(self.T, comp_j, comp_i)
        # for same components, the expression can be simplified
        else:
            a_ij = comp_i.attraction(self.T)

        # multiply with fractions and return
        if multiply_fractions:
            return a_ij * comp_i.fraction * comp_j.fraction
        else:
            return a_ij

    def _vdW_dT_a_ij(
        self,
        comp_i: PR_Component,
        comp_j: PR_Component,
        multiply_fractions: bool = True,
    ) -> pp.ad.Operator:
        """Get the derivative w.r.t. temperature of ``z_i * z_j * a_ij``
        (see :meth:`_get_a_ij`)."""
        # the expression for two different components
        if comp_i != comp_j:
            # the derivative of a_ij
            dT_a_ij = (
                _power(comp_i.attraction(self.T), _div_sqrt)
                * comp_i.dT_attraction(self.T)
                * _sqrt(comp_j.attraction(self.T))
                + _sqrt(comp_i.attraction(self.T))
                * _power(comp_j.attraction(self.T), _div_sqrt)
                * comp_j.dT_attraction(self.T)
            ) / 2

            bip, dT_bip, order = get_PR_BIP(comp_i.name, comp_j.name)
            # assert there is a bip, to appease mypy
            # this check is performed in add_component anyways
            assert bip is not None

            # multiplying with BIP
            if order:
                dT_a_ij *= 1 - bip(self.T, comp_i, comp_j)

                # if the derivative of the BIP is not trivial, add respective part
                if dT_bip:
                    dT_a_ij -= _sqrt(
                        comp_i.attraction(self.T) * comp_j.attraction(self.T)
                    ) * dT_bip(self.T, comp_i, comp_j)
            else:
                dT_a_ij *= 1 - bip(self.T, comp_j, comp_i)

                # if the derivative of the BIP is not trivial, add respective part
                if dT_bip:
                    dT_a_ij -= _sqrt(
                        comp_i.attraction(self.T) * comp_j.attraction(self.T)
                    ) * dT_bip(self.T, comp_j, comp_i)
        # if the components are the same, the expression simplifies
        else:
            dT_a_ij = comp_i.dT_attraction(self.T)

        # multiply with fractions and return
        if multiply_fractions:
            return dT_a_ij * comp_i.fraction * comp_j.fraction
        else:
            return dT_a_ij

    def _assign_covolume(self) -> None:
        """Creates the co-volume of the mixture according to van der Waals- mixing rule."""
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
            return p / (R_IDEAL * T * self.roots.liquid_root)

        def _rho_G(p, T, *X):
            return p / (R_IDEAL * T * self.roots.gas_root)

        # assigning the callable to respective thermodynamic property of the PR_Phase
        self._phases[0]._rho = _rho_L
        self._phases[1]._rho = _rho_G

    def _assign_phase_enthalpies(self) -> None:
        """Constructs callable objects representing phase enthalpies and assigns them to the
        ``PR_Phase``-classes."""

        coeff = (
            _power(2 * self.covolume, pp.ad.Scalar(-1 / 2))
            / 2
            * (self.T * self.dT_attraction - self.attraction)
        )

        def _h_L(p, T, *X):
            ln_l = _log(
                (self.roots.liquid_root + (1 - np.sqrt(2)) * self.B)
                / (self.roots.liquid_root + (1 + np.sqrt(2)) * self.B)
            )
            return (
                coeff * ln_l + R_IDEAL * self.T * (self.roots.liquid_root - 1) + H_REF
            )  # TODO check relation H_REF and H_0 in standard formula

        def _h_G(p, T, *X):
            ln_l = _log(
                (self.roots.gas_root + (1 - np.sqrt(2)) * self.B)
                / (self.roots.gas_root + (1 + np.sqrt(2)) * self.B)
            )
            return coeff * ln_l + R_IDEAL * self.T * (self.roots.gas_root - 1) + H_REF

        # assigning the callable to respective thermodynamic property of the PR_Phase
        self._phases[0]._h = _h_L
        self._phases[1]._h = _h_G

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
        per phase.

        References:
            [1]: `Zhu et al. (2014), equ. A-4 <https://doi.org/10.1016/j.fluid.2014.07.003>`_
            [2]: `ePaper <https://www.yumpu.com/en/document/view/36008448/
                 1-derivation-of-the-fugacity-coefficient-for-the-peng-robinson-fet>`_

        """

        for component in self.components:
            # shorten namespace
            b_c = component.covolume
            Z_L = self.roots.liquid_root
            Z_G = self.roots.gas_root

            a_c = [
                other_c.fraction * self._vdW_a_ij(component, other_c, False)
                for other_c in self.components
            ]
            a_c = sum(a_c)

            # fugacity coefficient for component c in liquid phase
            # TODO the last term containing a_c / a is unclear,
            # sources say different thing... might be sum_i z_i * a_ci / a
            phi_c_L = (
                b_c / self.covolume * (Z_L - 1)
                - _log(Z_L - self.B)
                - self.attraction
                / (self.covolume * R_IDEAL * self.T * np.sqrt(8))
                * _log(
                    (Z_L + (1 + np.sqrt(2)) * self.B)
                    / (Z_L + (1 - np.sqrt(2)) * self.B)
                )
                * (2 * a_c / self.attraction - b_c / self.covolume)
            )
            # fugacity coefficient for component c in gas phase
            phi_c_G = (
                b_c / self.covolume * (Z_G - 1)
                - _log(Z_G - self.B)
                - self.attraction
                / (self.covolume * R_IDEAL * self.T * np.sqrt(8))
                * _log(
                    (Z_G + (1 + np.sqrt(2)) * self.B)
                    / (Z_G + (1 - np.sqrt(2)) * self.B)
                )
                * (2 * a_c / self.attraction - b_c / self.covolume)
            )

            self._fugacities[component] = {
                self._phases[0]: phi_c_L,
                self._phases[1]: phi_c_G,
            }
