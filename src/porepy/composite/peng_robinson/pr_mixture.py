"""This module contains a mixture class for the Peng-Robinson equation of state.

It must be used in combination with respective component and phase classes.

This framework is highly non-linear and active research code.

References:
    [1]: `Ben Gharbia et al. (2021) <https://doi.org/10.1051/m2an/2021075>`_
    [2]: `Vu et al. (2021) <https://doi.org/10.1016/j.matcom.2021.07.015>`_

"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..mixture import Mixture
from .pr_bip import get_PR_BIP
from .pr_component import PR_Component
from .pr_phase import PR_Phase

__all__ = ["PengRobinsonMixture"]


class PengRobinsonMixture(Mixture):
    """A mixture modelled using the original Peng-Robinson equation of state.

    This mixture class is in principle applicable to any cubic EoS,
    where the roots of the cubic polynomial must be computed before the flash system
    is linearized.

    """

    def add(self, components: list[PR_Component], phases: list[PR_Phase]) -> None:
        """This child class method checks additionally if BIPs are defined for
        components to be added and components already added.

        For the Peng-Robinson EoS to work as intended, BIPs must be available for any
        combination of two present components and present in
        :data:`~porepy.composite.peng_robinson.pr_bip.PR_BIP_MAP`.

        Parameters:
            components: Peng-Robinson component(s) for this mixture.

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
        super().add(components, phases)

    ### root computation ---------------------------------------------------------------

    def precompute(
        self,
        *args,
        state: Optional[np.ndarray] = None,
        apply_smoother: bool = True,
        **kwargs,
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
            apply_smoother: ``default=True``

                If True, a smoothing procedure is applied in the three-root-region,
                where the intermediate root approaches one of the other roots.

                This is to be used **within** iterative procedures for numerical
                reasons. Once convergence is reached, the true roots should be computed
                without smoothing.

        """
        # evaluate variables in AD form to get the derivatives
        ads = self.AD.system
        pressure = self.AD.p.evaluate(ads, state=state)
        temperature = self.AD.T.evaluate(ads, state=state)

        for phase in self.phases:
            X = [
                phase.normalized_fraction_of_component(comp).evaluate(ads, state=state)
                for comp in self.components
            ]
            phase.eos.compute(pressure, temperature, *X, apply_smoother=apply_smoother)
