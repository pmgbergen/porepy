"""This module contains a composition class for the Peng-Robinson equation of state.

As of now, it supports a liquid and a gas phase, and several modelled components.

The formulation is thermodynamically consistent. The PR composition creates and assigns
thermodynamic properties of phases, based on the roots of the cubic polynomial and
added components.

For the equilibrium equations, formulae for fugacity values for each component in each
phase are implemented.

This framework is highly non-linear and active research code.

References:
    [1]: `Ben Gharbia et al. (2021) <https://doi.org/10.1051/m2an/2021075>`_
    [2]: `Vu et al. (2021) <https://doi.org/10.1016/j.matcom.2021.07.015>`_

"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import scipy.sparse as sps

from ..composition import Composition
from .pr_bip import get_PR_BIP
from .pr_component import PR_Component

__all__ = ["PR_Composition"]


class PR_Composition(Composition):
    """A composition modelled using the Peng-Robinson equation of state.

    This composition class is in principle applicable to any cubic EoS,
    where the roots of the cubic polynomial must be computed before the flash system
    is linearized.

    """

    def add_components(self, components: list[PR_Component]) -> None:
        """This child class method checks additionally if BIPs are defined for
        components to be added and components already added.

        For the Peng-Robinson EoS to work as intended, BIPs must be available for any
        combination of two present components and present in
        :data:`~porepy.composite.peng_robinson.pr_bip.PR_BIP_MAP`.

        Parameters:
            components: Peng-Robinson component(s) for this composition.

        Raises:
            NotImplementedError: If a BIP is not available for any combination of
                modelled components.

        """

        missing_bips: list[tuple[str, str]] = list()

        # check for missing bips between new components and present components
        for comp_new in components:
            for comp_present in self.components:
                # there is no bip between a component and itself
                if comp_new != comp_present:
                    bip, *_ = get_PR_BIP(comp_new.name, comp_present.name)
                    # if bip is not available, add pair to missing bips
                    if bip is None:
                        missing_bips.append((comp_new.name, comp_present.name))

        # check for missing bips between new components
        for comp_1 in components:
            for comp_2 in components:
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
        super().add_components(components)

    def linearize_subsystem(
        self,
        flash_type: Literal["isenthalpic", "isothermal"],
        other_vars: Optional[list[str]] = None,
        other_eqns: Optional[list[str]] = None,
        state: Optional[np.ndarray] = None,
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Before the system is linearized by a super call to the parent method,
        the PR mixture computes the EoS Roots in each phase for (iterative) updates,
        including the smoothing procedure by default.

        Note:
            This is for performance reasons, since the roots can be evaluated only
            once and used for all thermodynamic properties.
            If they are computed during each call to every property, this becomes
            very inefficient.

        """

        # compute roots and thermodynamic properties ONCE, since they depend on
        # fractions, pressure and temperature.
        self.compute_roots(state=state, apply_smoother=True)
        return super().linearize_subsystem(flash_type, other_vars, other_eqns, state)

    ### root computation ---------------------------------------------------------------

    def compute_roots(
        self,
        state: Optional[np.ndarray] = None,
        apply_smoother: bool = False,
    ) -> None:
        """Invokes the computation of the compressibility factor for each present phase
        (see :meth:`~porepy.composite.peng_robinson.pr_eos.PR_EoS.compute`).

        If ``state`` is not given, values for ``p``, ``T`` and ``X`` are assembled
        and passed to the method.

        Otherwise respective values are extracted from ``state`` and used.

        Parameters:
            state: ``default=None``

                An optional (global) state vector for the AD system, containing the
                thermodynamic state of the system.
            apply_smoother: ``default=False``

                If True, a smoothing procedure is applied in the three-root-region,
                where the intermediate root approaches one of the other roots.

                This is to be used **within** iterative procedures for numerical
                reasons. Once convergence is reached, the true roots should be computed
                without smoothing.

        """
        # evaluate variables in AD form to get the derivatives
        pressure = self.p.evaluate(self.ad_system, state=state)
        temperature = self.T.evaluate(self.ad_system, state=state)

        for phase in self.phases:
            X = [
                phase.normalized_fraction_of_component(comp).evaluate(
                    self.ad_system, state=state
                )
                for comp in self.components
            ]
            phase.eos.compute(pressure, temperature, *X, apply_smoother=apply_smoother)
