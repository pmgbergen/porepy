"""This module contains extended, abstract base classes representing components and
their properties and parameters used in the Peng-Robinson EoS.

"""
from __future__ import annotations

import abc

import porepy as pp
from porepy.numerics.ad.operator_functions import NumericType

from ..composite_utils import R_IDEAL
from ..component import Component, Compound
from .pr_utils import A_CRIT, B_CRIT


class PR_Component(Component):
    """Intermediate abstraction layer for (fluid) components in a Peng-Robinson mixture.

    Serves for the abstraction of cohesion and covolume in the Peng-Robinson EoS,
    associated with this component.

    """

    @property
    def critical_cohesion(self) -> float:
        """The critical cohesion parameter

            ``a_critical = A_CRIT * (R_IDEAL**2 * T_critical**2) / p_critical``

        is part of the cohesion ``a``

            ``a = a_critical * alpha(T_reduced, omega)``.

        """
        return (
            A_CRIT
            * (R_IDEAL**2 * self.critical_temperature() ** 2)
            / self.critical_pressure()
        )

    @property
    def cohesion_correction_weight(self) -> float:
        """Weight ``kappa`` of the linearized alpha-correction of the
        cohesion parameter:

            ``a = a_critical * alpha``,
            ``alpha = 1 + kappa * (1 - sqrt(T_reduced))``.

        References:
            `Zhu et al. (2014), Appendix A
            <https://doi.org/10.1016/j.fluid.2014.07.003>`_

        """
        if self.acentric_factor < 0.49:
            return (
                0.37464
                + 1.54226 * self.acentric_factor
                - 0.26992 * self.acentric_factor**2
            )
        else:
            return (
                0.379642
                + 1.48503 * self.acentric_factor
                - 0.164423 * self.acentric_factor**2
                + 0.016666 * self.acentric_factor**3
            )

    def cohesion_correction(self, T: NumericType) -> NumericType:
        """Returns the root of the linearized alpha-correction for the
        cohesion parameter."""
        return 1 + self.cohesion_correction_weight * (
            1 - pp.ad.sqrt(T / self.critical_temperature())
        )

    def cohesion(self, T: NumericType) -> NumericType:
        """Returns an expression for ``a`` in the EoS for this component."""
        return self.critical_cohesion * pp.ad.power(self.cohesion_correction(T), 2)

    def dT_cohesion(self, T: NumericType) -> NumericType:
        """Returns an expression for the derivative of ``a`` with respect to
        temperature."""

        # external derivative of alpha term
        dt_a = 2 * self.critical_cohesion * self.cohesion_correction(T)
        # internal derivative of alpha term
        dt_a *= (
            -self.cohesion_correction_weight / (2 * self.critical_temperature())
        ) * pp.ad.power(T / self.critical_temperature(), -1 / 2)

        return dt_a

    @property
    def covolume(self) -> NumericType:
        """The constant covolume ``b`` in the Peng-Robinson EoS

            ``b = B_CRIT * (R_IDEAL * T_critical) / p_critical

        wrapped in an AD operator.

        """
        return (
            B_CRIT * (R_IDEAL * self.critical_temperature()) / self.critical_pressure()
        )


class PR_Compound(PR_Component, Compound):
    """Intermediate abstraction layer for (fluid) compounds in a Peng-Robinson mixture.

    Serves for the abstraction of cohesion and covolume in the Peng-Robinson EoS,
    associated with this component.

    Compared to the PR-component, the cohesion correction ``alpha`` remains abstract,
    since it depends on the present solutes.

    """

    @abc.abstractmethod
    def cohesion_correction(self, T: NumericType) -> NumericType:
        """Abstraction of the corrective term in ``a``.

        To be implemented in child classes using heuristic laws depending on present
        solutes.

        """
        pass
