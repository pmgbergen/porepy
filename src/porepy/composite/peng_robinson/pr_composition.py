"""This module contains a composition class for the Peng-Robinson equation of state.

As of now, it supports a liquid and a gas phase, and several modelled components.

The formulation is thermodynamically consistent. The PR composition creates and assigns
thermodynamic properties of phases, based on the roots of the cubic polynomial and
added components.

For the equilibrium equations, formulae for fugacity values for each component in each
phase are implemented.

This framework is highly non-linear and active research code.

"""
from __future__ import annotations

from typing import Callable, Generator, Literal, Optional

import numpy as np
import scipy.sparse as sps

import porepy as pp

from .._composite_utils import H_REF, R_IDEAL
from ..composition import Composition
from .pr_bip import get_PR_BIP
from .pr_component import PR_Component
from .pr_phase import PR_Phase
from .pr_roots import PR_Roots
from .pr_utils import _exp, _log, _power, _sqrt

__all__ = ["PR_Composition"]

_div_sqrt = pp.ad.Scalar(-1 / 2)


class PR_Composition(Composition):
    """A composition modelled using the Peng-Robinson equation of state.

        ``p = R * T / (v - b) - a / (b**2 + 2 * v * b - b**2)``.

    Van der Waals mixing rules are applied to ``a`` and ``b``.

    Note:
        - This class currently supports only a liquid and a gaseous phase.
        - The various properties of this class depend on the thermodynamic state.
          They are continuously re-computed and must therefore be accessed by reference.

    """

    def __init__(self, ad_system: Optional[pp.ad.EquationSystem] = None) -> None:
        super().__init__(ad_system)

        self.roots: PR_Roots
        """The roots of the cubic equation of state.

        This object is instantiated during initialization.

        """

        self.log_fugacity_coeffs: dict[
            PR_Component, dict[PR_Phase, pp.ad.Operator]
        ] = dict()
        """A nested dictionary containing an operator representing the logarithm of the
        fugacity coefficient for a given ``[component][phase]``.

        This dictionary must be created in :meth:`set_fugacities`, which in return
        is called during :meth:`initialize`.

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

        # cohesion and covolume, assembled during initialization
        # (based on mixing-rule and present components)
        self._a: pp.ad.Operator
        self._dT_a: pp.ad.Operator
        self._b: pp.ad.Operator

    def add_component(self, component: PR_Component | list[PR_Component]) -> None:
        """This child class method checks additionally if BIPs are defined for
        components to be added and components already added.

        For the Peng-Robinson EoS to work as intended, BIPs must be available for any
        combination of two present components and present in
        :data:`~porepy.composite.peng_robinson.pr_bip.PR_BIP_MAP`.

        Parameters:
            component: One or multiple model (PR-) components for this EoS.

        Raises:
            NotImplementedError: If a BIP is not available for any combination of
                modelled components.

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
            for comp_2 in component:
                # no bip between a component and itself
                if comp_2 != comp_1:
                    bip, *_ = get_PR_BIP(comp_1.name, comp_2.name)
                    if bip is None:
                        missing_bips.append((comp_1.name, comp_2.name))

        # if missing bips detected, raise error
        if missing_bips:
            raise NotImplementedError(
                f"BIPs not available for following component-pairs:\n\t{missing_bips}"
            )
        # if no missing bips, we proceed adding the components using the parent method.
        super().add_component(component)

    def initialize(self) -> None:
        """Creates and assigns thermodynamic properties, as well as equilibrium
        equations using PR.

        Before initializing the p-h and p-T subsystems,
        this method additionally assigns callables for thd. properties of phases,
        according to the equation of state and present components,
        and constructs the cohesion and covolume factors in the EoS.

        After that, it performs a super-call to :meth:`~Composition.initialize`.

        As a third and final step it assigns equilibrium equations in the form of
        equality of fugacities.

        Raises:
            AssertionError: If the mixture is empty (no components).

        """
        # assert non-empty mixture
        assert self.num_components >= 1

        ## defining the cohesion value
        self._assign_cohesion()
        ## defining the covolume
        self._assign_covolume()

        ## create roots object
        self.roots = PR_Roots(self.ad_system, self.A, self.B)

        ## setting callables representing the phase densities
        self._assign_phase_densities()
        ## setting callables representing the specific phase enthalpies
        self._assign_phase_enthalpies()
        ## setting callables defining the viscosity and conductivity of phases
        self._assign_phase_viscosities()
        self._assign_phase_conductivities()

        # super call to initialize p-h and p-T subsystems
        super().initialize()

    @property
    def components(self) -> Generator[PR_Component, None, None]:
        """Function overload to specify special return type for PR compositions."""
        for C in self._components:
            yield C

    ### EoS parameters -----------------------------------------------------------------

    @property
    def cohesion(self) -> pp.ad.Operator:
        """An operator representing ``a`` in the Peng-Robinson EoS."""
        return self._a

    @property
    def dT_cohesion(self) -> pp.ad.Operator:
        """An operator representing the derivative of :meth:`cohesion` w.r.t.
        temperature."""
        return self._dT_a

    @property
    def covolume(self) -> pp.ad.Operator:
        """An operator representing ``b`` in the Peng-Robinson EoS."""
        return self._b

    @property
    def A(self) -> pp.ad.Operator:
        """An operator representing ``A`` in the characteristic polynomial of the EoS."""
        return (self.cohesion * self.p) / (R_IDEAL**2 * self.T * self.T)

    @property
    def B(self) -> pp.ad.Operator:
        """An operator representing ``B`` in the characteristic polynomial of the EoS."""
        return (self.covolume * self.p) / (R_IDEAL * self.T)

    ### Subsystem assembly method ------------------------------------------------------

    def linearize_subsystem(
        self,
        flash_type: Literal["isenthalpic", "isothermal"],
        other_vars: Optional[list[str]] = None,
        other_eqns: Optional[list[str]] = None,
        state: Optional[np.ndarray] = None,
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Before the system is linearized by a super call to the parent method,
        the PR mixture computes the EoS Roots for (iterative) updates."""
        self.roots.compute_roots(state=state)
        return super().linearize_subsystem(flash_type, other_vars, other_eqns, state)

    ### Model equations ----------------------------------------------------------------

    def _assign_cohesion(self) -> None:
        """Creates the cohesion parameter for a mixture according to PR,
        as well as its derivative w.r.t. temperature."""
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
                        # computing the cohesion between components i and j
                        a += self._vdW_a_ij(comp_i, comp_j)
                        # computing the derivative w.r.t temperature of cohesion terms
                        dT_a += self._vdW_dT_a_ij(comp_i, comp_j)

        # store cohesion parameters
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
            a_ij = _sqrt(comp_i.cohesion(self.T) * comp_j.cohesion(self.T))
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
            a_ij = comp_i.cohesion(self.T)

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
        (see :meth:`_vdW_a_ij`)."""
        # the expression for two different components
        if comp_i != comp_j:
            # the derivative of a_ij
            dT_a_ij = (
                _power(comp_i.cohesion(self.T), _div_sqrt)
                * comp_i.dT_cohesion(self.T)
                * _sqrt(comp_j.cohesion(self.T))
                + _sqrt(comp_i.cohesion(self.T))
                * _power(comp_j.cohesion(self.T), _div_sqrt)
                * comp_j.dT_cohesion(self.T)
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
                        comp_i.cohesion(self.T) * comp_j.cohesion(self.T)
                    ) * dT_bip(self.T, comp_i, comp_j)
            else:
                dT_a_ij *= 1 - bip(self.T, comp_j, comp_i)

                # if the derivative of the BIP is not trivial, add respective part
                if dT_bip:
                    dT_a_ij -= _sqrt(
                        comp_i.cohesion(self.T) * comp_j.cohesion(self.T)
                    ) * dT_bip(self.T, comp_j, comp_i)
        # if the components are the same, the expression simplifies
        else:
            dT_a_ij = comp_i.dT_cohesion(self.T)

        # multiply with fractions and return
        if multiply_fractions:
            return dT_a_ij * comp_i.fraction * comp_j.fraction
        else:
            return dT_a_ij

    def _assign_covolume(self) -> None:
        """Creates the covolume of the mixture according to van der Waals- mixing rule."""
        components: list[PR_Component] = [c for c in self.components]  # type: ignore

        b = components[0].fraction * components[0].covolume

        if len(components) > 1:
            for comp in components[1:]:
                b += comp.fraction * comp.covolume

        self._b = b

    def _assign_phase_densities(self) -> None:
        """Constructs callable objects representing phase densities and assigns them to
        the ``PR_Phase``-classes."""
        # assigning the callable to respective thermodynamic property of the PR_Phase
        self._phases[0]._rho = self._rho_Z(self.roots.liquid_root)
        self._phases[1]._rho = self._rho_Z(self.roots.gas_root)

    def _rho_Z(self, Z: pp.ad.Operator) -> Callable:
        """Returns a callable representing the density in the PR EoS for a
        specific root.

        Parameters:
            Z: reference to an AD array representing an extended root of the cubic
                polynomial.

        """
        # densities in cubic EoS are given as inverse of specific molar volume v_e
        # rho = 1/ v
        # Z = p * v / (R * T)
        # as of now, the composition X does not influence the density
        def rho_z(p, T, *X):
            return p / (R_IDEAL * T * Z)

        return rho_z

    def _assign_phase_enthalpies(self) -> None:
        """Constructs callable objects representing phase enthalpies and assigns them to
        the ``PR_Phase``-classes."""
        # assigning the callable to respective thermodynamic property of the PR_Phase
        self._phases[0]._h = self._h_Z(self.roots.liquid_root)
        self._phases[1]._h = self._h_Z(self.roots.gas_root)

    def _h_Z(self, Z: pp.ad.Operator) -> Callable:
        """Returns a callable representing the enthalpy in the PR EoS for a
        specific root.

        References:
            [1]: `Connolly et al. (2021), eq. B-15 <https://doi.org/10.1016/
                 j.ces.2020.116150>`_

        Parameters:
            Z: reference to an AD array representing an extended root of the cubic
                polynomial.

        """
        coeff = (
            _power(2 * self.covolume, pp.ad.Scalar(-1 / 2))
            / 2
            * (self.T * self.dT_cohesion - self.cohesion)
        )

        def h_Z(p, T, *X):
            ln_Z = _log(
                (Z + (1 - np.sqrt(2)) * self.B) / (Z + (1 + np.sqrt(2)) * self.B)
            )
            return (
                coeff * ln_Z + R_IDEAL * self.T * (Z - 1) + H_REF
            )  # TODO check relation H_REF and H_0 in standard formula

        return h_Z

    def _assign_phase_viscosities(self) -> None:
        """Constructs callable objects representing phase dynamic viscosities and
        assigns them to the ``PR_Phase``-classes."""
        # assigning the callable to respective thermodynamic property of the PR_Phase
        self._phases[0]._mu = self._mu_Z(self.roots.liquid_root)
        self._phases[1]._mu = self._mu_Z(self.roots.gas_root)

    def _mu_Z(self, Z: pp.ad.Operator) -> Callable:  # TODO
        """Returns a callable representing the viscosity in the PR EoS for a
        specific root.

        Parameters:
            Z: reference to an AD array representing an extended root of the cubic
                polynomial.

        """

        def mu_Z(p, T, *X):
            return pp.ad.Scalar(1.0)

        return mu_Z

    def _assign_phase_conductivities(self) -> None:
        """Constructs callable objects representing phase conductivities and assigns
        them to the ``PR_Phase``-classes."""
        # assigning the callable to respective thermodynamic property of the PR_Phase
        self._phases[0]._kappa = self._kappa_Z(self.roots.liquid_root)
        self._phases[1]._kappa = self._kappa_Z(self.roots.gas_root)

    def _kappa_Z(self, Z: pp.ad.Operator) -> Callable:  # TODO
        """Returns a callable representing the conductivity in the PR EoS for a
        specific root.

        Parameters:
            Z: reference to an AD array representing an extended root of the cubic
                polynomial.

        """

        def kappa_Z(p, T, *X):
            return pp.ad.Scalar(1.0)

        return kappa_Z

    def set_fugacities(self) -> None:
        """Creates logarithmic and normal fugacity coefficients for each component using
        the liquid and gas root."""

        for component in self.components:
            self.log_fugacity_coeffs[component] = {
                self._phases[0]: self._log_phi_c_e(component, self.roots.liquid_root),
                self._phases[1]: self._log_phi_c_e(component, self.roots.gas_root),
            }

            self.fugacity_coeffs[component] = {
                self._phases[0]: self._phi_c_e(component, self.roots.liquid_root),
                self._phases[1]: self._phi_c_e(component, self.roots.gas_root),
            }

    def _phi_c_e(
        self,
        component: PR_Component,
        Z: pp.ad.Operator,
        phase: Optional[PR_Phase] = None,
    ) -> pp.ad.Operator:
        """Returns an AD operator representing the fugacity coefficients of
        ``component``.

        The root ``Z`` must be provided.

        Optionally, a phase can be provided to use contextual mole fractions,
        i.e. phase compositions instead of overall mixture composition.

        References:
            [1]: `Zhu et al. (2014), equ. A-4
                 <https://doi.org/10.1016/j.fluid.2014.07.003>`_
            [2]: `ePaper <https://www.yumpu.com/en/document/view/36008448/
                 1-derivation-of-the-fugacity-coefficient-for-the-peng-robinson-fet>`_
            [3]: `therm <https://thermo.readthedocs.io/
                 thermo.eos_mix.html#thermo.eos_mix.PRMIX.fugacity_coefficients>`_

        Parameters:
            component: A PR component in this composition.
            Z: reference to an AD array representing an extended root of the cubic
                polynomial.
            phase: ``default=None``

                An optional phase to return contextual fugacity coefficients.
                If given, the phase composition will be used in the mixing rule for
                ``A`` and ``B``.
                If None, the component overall fractions are used.

        """
        log_phi_c_e = self._log_phi_c_e(component, Z, phase)
        phi_c_e = _exp(log_phi_c_e)

        return phi_c_e

    def _log_phi_c_e(
        self,
        component: PR_Component,
        Z: pp.ad.Operator,
        phase: Optional[PR_Phase] = None,
    ) -> pp.ad.Operator:
        """Returns an AD operator representing the logarithmic fugacity coefficients of
        ``component``.

        For more information see :meth:`_phi_c_e`.

        """
        # if phase is given, use phase compositions for mixing rule
        if phase:
            # index c for component
            # index m for mixture
            b_c = component.covolume
            b_m = sum(
                [
                    phase.normalized_fraction_of_component(comp) * comp.covolume
                    for comp in self.components
                ]
            )
            B_m = (b_m * self.p) / (R_IDEAL * self.T)

            a_c = sum(
                [
                    phase.normalized_fraction_of_component(other_c)
                    * self._vdW_a_ij(component, other_c, False)
                    for other_c in self.components
                ]
            )
            a_m = sum(
                [
                    phase.normalized_fraction_of_component(comp_i)
                    * phase.normalized_fraction_of_component(comp_j)
                    * self._vdW_a_ij(comp_i, comp_j, False)
                    for comp_i in self.components
                    for comp_j in self.components
                ]
            )
        # if not given, use overall component fractions
        else:
            # index c for component
            # index m for mixture
            b_c = component.covolume
            b_m = self.covolume
            B_m = self.B

            a_c = sum(
                [
                    other_c.fraction * self._vdW_a_ij(component, other_c, False)
                    for other_c in self.components
                ]
            )
            a_m = self.cohesion

        log_phi_c_e = (
            b_c / b_m * (Z - 1)
            - _log(Z - B_m)
            + _log((Z + (1 + np.sqrt(2)) * B_m) / (Z + (1 - np.sqrt(2)) * B_m))
            * a_m
            / (b_m * R_IDEAL * self.T * np.sqrt(8))
            * (b_c / b_m - 2 * a_c / a_m)
        )

        return log_phi_c_e
