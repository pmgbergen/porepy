"""This module contains an extension of the abstract phase class for the
Peng-Robinson EoS."""
from __future__ import annotations

import porepy as pp

from ..phase import Phase
from .pr_component import PR_Component
from .pr_eos import PR_EoS


class PR_Phase(Phase):
    """Representation of a phase using the Peng-Robinson EoS and the Van der Waals
    mixing rule (see :class:`~porepy.composite.peng_robinson.pr_eos.PR_EoS`).

    For further information on thermodynamic properties and how they are computed,
    see respective EoS class.

    """

    def __init__(self, ad_system: pp.ad.EquationSystem, name: str = "") -> None:
        super().__init__(ad_system, name=name)

        self.eos: PR_EoS

    def fugacity_of(
        self,
        p: pp.ad.MixedDimensionalVariable,
        T: pp.ad.MixedDimensionalVariable,
        component: PR_Component,
    ) -> pp.ad.Operator:
        """See :class:`~porepy.composite.peng_robinson.pr_eos.PR_EoS`.

        Since for efficiency reasons roots are ought to be computed once before
        linearization,
        this method wraps :data:`~porepy.composite.peng_robinson.pr_eos.PR_EoS.phi`
        into an AD operator function.

        """
        if component in self:
            pass  # TODO
        else:
            return pp.ad.Scalar(0.0)

    def density(self, p, T):
        """See :class:`~porepy.composite.peng_robinson.pr_eos.PR_EoS`.

        Since for efficiency reasons roots are ought to be computed once before
        linearization,
        this method wraps :data:`~porepy.composite.peng_robinson.pr_eos.PR_EoS.rho`
        into an AD operator function.

        """
        pass

    def specific_enthalpy(self, p, T):
        """Returns the sum of fraction weighted ideal component enthalpies (normalized)
        added to the departer enthalpy of the EoS.

        The departer enthalpy is wrapped into an AD operator function using
        :data:`~porepy.composite.peng_robinson.pr_eos.PR_EoS.h_dep`.
        """

        # TODO
        h_departure = 0

        h_ideal = sum(
            [self.fraction_of_component(comp) * comp.h_ideal(p, T) for comp in self]
        ) / sum([self.fraction_of_component(comp) for comp in self])

        return h_ideal + h_departure

    def dynamic_viscosity(self, p, T):  # TODO
        return pp.ad.Scalar(1.0)

    def thermal_conductivity(self, p, T):  # TODO
        return pp.ad.Scalar(1.0)
