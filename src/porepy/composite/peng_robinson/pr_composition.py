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

import porepy as pp

from ..composition import Composition
from .pr_bip import get_PR_BIP
from .pr_component import PR_Component
from .pr_phase import PR_Phase

__all__ = ["PR_Composition"]


class PR_Composition(Composition):
    """A composition modelled using the Peng-Robinson equation of state.

    This composition class is in principle applicable to any cubic EoS,
    where the roots of the cubic polynomial must be computed before the flash system
    is linearized.

    Note:
        - This class currently supports only a liquid and a gaseous phase.
        - The various properties of this class depend on the thermodynamic state.
          They are defined during initialization since they depend on all components.

    """

    def __init__(self, ad_system: Optional[pp.ad.EquationSystem] = None) -> None:
        super().__init__(ad_system)

        ### PRIVATE

        # setting of currently supported phases
        self._phases: list[PR_Phase] = [
            PR_Phase(self.ad_system, name="L"),
            PR_Phase(self.ad_system, name="G"),
        ]

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

    ### Subsystem assembly method ------------------------------------------------------

    def linearize_subsystem(
        self,
        flash_type: Literal["isenthalpic", "isothermal"],
        other_vars: Optional[list[str]] = None,
        other_eqns: Optional[list[str]] = None,
        state: Optional[np.ndarray] = None,
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Before the system is linearized by a super call to the parent method,
        the PR mixture computes the EoS Roots for (iterative) updates.

        Warning:
            If the isothermal system is assembled, the roots of the EoS must be
            computed beforehand using :meth:`compute_roots`.

            In the isenthalpic system they are computed here due to the
            temperature-dependency of the roots.

            This is for performance reasons, since the roots must be evaluated only
            once in the isothermal flash.

        """

        # compute roots and thermodynamic properties ONCE, since they depend on
        # fractions, pressure and temperature.
        self.compute_roots(state=state)
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
        pass
