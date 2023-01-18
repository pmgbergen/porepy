"""This module contains extended, abstract base classes representing components and
their properties and parameters used in the Peng-Robinson EoS.

"""
from __future__ import annotations

import abc

import porepy as pp

from .._composite_utils import R_IDEAL
from ..component import Component, Compound
from .pr_utils import A_CRIT, B_CRIT, _power, _sqrt


class PR_Component(Component):
    """Intermediate abstraction layer for (fluid) components in a Peng-Robinson mixture.

    Serves for the abstraction of cohesion and covolume in the Peng-Robinson EoS,
    associated with this component.

    """

    @property
    @abc.abstractmethod
    def acentric_factor(self) -> float:
        """This is a constant value, hence to be a static function.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [-]

        Returns:
            Acentric factor.

        """
        pass

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

    def cohesion_correction(self, T: pp.ad.MergedVariable) -> pp.ad.Operator:
        """Returns the linearized alpha-correction for the cohesion parameter"""

        alpha_root = 1 + self.cohesion_correction_weight * (
            1 - _sqrt(T / self.critical_temperature())
        )

        return alpha_root * alpha_root

    def cohesion(self, T: pp.ad.MergedVariable) -> pp.ad.Operator:
        """Returns an expression for ``a`` in the EoS for this component."""
        return self.critical_cohesion * self.cohesion_correction(T)

    def dT_cohesion(self, T: pp.ad.MergedVariable) -> pp.ad.Operator:
        """Returns an expression for the derivative of ``a`` with respect to
        temperature."""
        T_r = T / self.critical_temperature()

        # external derivative of cohesion correction squared
        dt_a = (
            self.critical_cohesion
            * 2
            * (1 + self.cohesion_correction_weight * (1 - _sqrt(T_r)))
        )
        # internal derivative of cohesion correction
        dt_a *= (
            self.cohesion_correction_weight
            / (-2)
            / self.critical_temperature()
            * _power(T_r, pp.ad.Scalar(-1 / 2))
        )

        return dt_a

    @property
    def covolume(self) -> pp.ad.Operator:
        """The constant covolume ``b`` in the Peng-Robinson EoS

            ``b = B_CRIT * (R_IDEAL * T_critical) / p_critical

        wrapped in an AD operator.

        """
        return pp.ad.Scalar(
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
    def cohesion_correction(self, T: pp.ad.MergedVariable) -> pp.ad.Operator:
        """Abstraction of the corrective term in ``a``.

        To be implemented in child classes using heuristic laws depending on present
        solutes.

        """
        pass
