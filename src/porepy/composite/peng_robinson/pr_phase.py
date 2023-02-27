"""This module contains an extension of the abstract phase class for the
Peng-Robinson EoS."""
from __future__ import annotations

import porepy as pp

from ..phase import Phase
from .pr_component import PR_Component
from .pr_eos import PR_EoS

__all__ = ["PR_Phase"]


class PR_Phase(Phase):
    """Representation of a phase using the Peng-Robinson EoS and the Van der Waals
    mixing rule (see :class:`~porepy.composite.peng_robinson.pr_eos.PR_EoS`).

    For further information on thermodynamic properties and how they are computed,
    see respective EoS class.

    """

    def __init__(
        self, ad_system: pp.ad.EquationSystem, gas_like: bool, name: str = ""
    ) -> None:
        super().__init__(ad_system, name=name)

        self.eos: PR_EoS = PR_EoS(gas_like)
        """The equation of state providing property computations for the Peng-Robinson
        framework.

        For the correct values, the user must call
        :meth:`~porepy.composite.peng_robinson.pr_eos.PR_EoS.compute`
        before evaluating any phase property.

        """

    @property
    def components(self) -> list[PR_Component]:
        """Additional to the functionalities of the parent property, the setter
        of the child property also sets the components in
        :data:`eos`.

        """
        return super().components

    @components.setter
    def components(self, components: list[PR_Component]) -> None:
        Phase.components.fset(self, components)
        self.eos.components = [comp for comp in components]

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
        into an AD operator function for each ``component``.

        """
        if component in self:

            # wrapping the fugacity coefficient in a component specific
            # AD operator function
            @pp.ad.admethod
            def phi_i(p_, T_):
                return self.eos.phi[component]

            return phi_i(p, T)
        else:
            return pp.ad.Scalar(0.0)

    @pp.ad.admethod
    def density(self, p, T):
        """See :class:`~porepy.composite.peng_robinson.pr_eos.PR_EoS`.

        Since for efficiency reasons roots are ought to be computed once before
        linearization,
        this method wraps :data:`~porepy.composite.peng_robinson.pr_eos.PR_EoS.rho`
        into an AD operator function.

        """
        return self.eos.rho

    @pp.ad.admethod
    def specific_enthalpy(self, p, T, *X):
        """Returns the sum of fraction weighted ideal component enthalpies (normalized)
        added to the departer enthalpy of the EoS.

        The departer enthalpy is wrapped into an AD operator function using
        :data:`~porepy.composite.peng_robinson.pr_eos.PR_EoS.h_dep`.

        Parameters:
            *X: Normalized component fractions in this phase. This is used for the
                ideal part, to be able to evaluate it w.r.t. specific values.

        """

        h_ideal = sum([x * comp.h_ideal(p, T) for x, comp in zip(X, self)]) / sum(X)

        return h_ideal + self.eos.h_dep

    def dynamic_viscosity(self, p, T):  # TODO
        return pp.ad.Scalar(1.0)

    def thermal_conductivity(self, p, T):  # TODO
        return pp.ad.Scalar(10.0)
