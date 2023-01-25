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

from typing import Generator, Literal, Optional

import numpy as np
import scipy.sparse as sps

import porepy as pp

from .._composite_utils import R_IDEAL
from ..composition import Composition
from .pr_bip import get_PR_BIP
from .pr_component import PR_Component
from .pr_mixing import VdW_a_ij, dT_VdW_a_ij
from .pr_phase import PR_Phase
from .pr_roots import PR_Roots

__all__ = ["PR_Composition"]


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

        ### PRIVATE

        # setting of currently supported phases
        self._phases: list[PR_Phase] = [
            PR_Phase(self.ad_system, name="L"),
            PR_Phase(self.ad_system, name="G"),
        ]

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
        this method additionally constructs the cohesion and covolume factors in the EoS.

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

        # super call to initialize p-h and p-T subsystems
        super().initialize()

    ### EoS parameters -----------------------------------------------------------------

    @property
    def cohesion(self) -> pp.ad.Operator:
        """An operator representing ``a`` in the Peng-Robinson EoS using the component
        feed fraction and Van der Waals mixing rule."""
        return self._a

    @property
    def dT_cohesion(self) -> pp.ad.Operator:
        """An operator representing the derivative of :meth:`cohesion` w.r.t.
        temperature."""
        return self._dT_a

    @property
    def covolume(self) -> pp.ad.Operator:
        """An operator representing ``b`` in the Peng-Robinson EoS using the component
        feed fraction and Van der Waals mixing rule."""
        return self._b

    @property
    def A(self) -> pp.ad.Operator:
        """An operator representing ``A`` in the characteristic polynomial of the EoS.

        It is based on :meth:`cohesion` and its representation."""
        return (self.cohesion * self.p) / (R_IDEAL**2 * self.T * self.T)

    @property
    def B(self) -> pp.ad.Operator:
        """An operator representing ``B`` in the characteristic polynomial of the EoS.

        It is based on :meth:`covolume` and its representation."""
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
        a = VdW_a_ij(self.T, comp_0, comp_0) * comp_0.fraction * comp_0.fraction
        dT_a = dT_VdW_a_ij(self.T, comp_0, comp_0) * comp_0.fraction * comp_0.fraction

        if len(components) > 1:
            # add remaining diagonal elements
            for comp in components[1:]:
                a += VdW_a_ij(self.T, comp, comp) * comp.fraction * comp.fraction
                dT_a += dT_VdW_a_ij(self.T, comp, comp) * comp.fraction * comp.fraction

            # adding off-diagonal elements, including BIPs
            for comp_i in components:
                for comp_j in components:
                    if comp_i != comp_j:
                        # computing the cohesion between components i and j
                        a += (
                            VdW_a_ij(self.T, comp_i, comp_j)
                            * comp_i.fraction
                            * comp_j.fraction
                        )
                        # computing the derivative w.r.t temperature of cohesion terms
                        dT_a += (
                            dT_VdW_a_ij(self.T, comp_i, comp_j)
                            * comp_i.fraction
                            * comp_j.fraction
                        )

        # store cohesion parameters
        self._a = a
        self._dT_a = dT_a

    def _assign_covolume(self) -> None:
        """Creates the covolume of the mixture according to van der Waals- mixing rule."""
        components: list[PR_Component] = [c for c in self.components]  # type: ignore

        b = components[0].fraction * components[0].covolume

        if len(components) > 1:
            for comp in components[1:]:
                b += comp.fraction * comp.covolume

        self._b = b
