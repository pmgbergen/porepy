"""Contains extended, abstract classes representing components and their properties and
parameters used in the Peng-Robinson EoS.

"""
from __future__ import annotations

import abc

import porepy as pp

from .._composite_utils import R_IDEAL
from ..component import Component, Compound
from ..phase import VarLike


class PR_Component(Component):
    """Intermediate abstraction layer for (fluid) components in a Peng-Robinson mixture.

    Serves for the abstraction of attraction and co-volume in the Peng-Robinson EoS,
    associated with this component.

    """

    @property
    @abc.abstractmethod
    def acentric_factor(self) -> float:
        """This is a constant value, hence to be a static function.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [-]

        Returns: acentric factor.

        """
        pass

    @property
    def attraction_critical(self) -> float:
        """Returns the critical attraction parameter
        ``a = a_critical * alpha(T_reduced, omega)``
        in the Peng-Robinson EoS,
        without the scaling by acentric factor and reduced temperature:

            ``a_critical = 0.457235529 * (R_IDEAL**2 * T_critical**2) / p_critical``

        """
        return (
            0.457235529
            * (R_IDEAL**2 * self.critical_temperature() ** 2)
            / self.critical_pressure()
        )

    @property
    def attraction_correction_weight(self) -> float:
        """Weight ``kappa`` of the linearized alpha-correction of the attraction parameter in
        the Peng-Robinson EoS:

            ``a = a_cricital * alpha``,
            ``alpha = 1 + kappa * (1 - sqrt(T_reduced))``.

        """
        return (
            0.37464
            + 1.54226 * self.acentric_factor
            - 0.26992 * self.acentric_factor**2
        )

    def attraction_correction(self, T: VarLike) -> VarLike:
        """Returns the linearized alpha-correction for the attraction parameter"""
        sqrt = pp.ad.Function(pp.ad.sqrt, "sqrt")

        alpha_root = 1 + self.attraction_correction_weight * (
            1 - sqrt(T / self.critical_temperature())
        )

        return alpha_root * alpha_root

    @property
    def covolume(self) -> float:
        """Returns the constant co-volume ``b`` in the Peng-Robinson EoS:

        ``b = 0.077796072 * (R_IDEAL * T_critical) / p_critical

        """
        return (
            0.077796072
            * (R_IDEAL * self.critical_temperature())
            / self.critical_pressure()
        )

    def attraction(self, T: VarLike) -> VarLike:
        """Returns an expression for ``a`` in the EoS for this component."""
        return self.attraction_critical * self.attraction_correction(T)


class PR_Compound(PR_Component, Compound):
    """Intermediate abstraction layer for (fluid) compounds in a Peng-Robinson mixture.

    Serves for the abstraction of attraction and co-volume in the Peng-Robinson EoS,
    associated with this component.

    Compared to the PR-component, the attraction correction ``alpha`` remains abstract,
    since it depends on the present solutes.

    """

    @abc.abstractmethod
    def attraction_correction(self, T: VarLike) -> VarLike:
        """Abstraction of the corrective term in ``a``.

        To be implemented in child classes using heuristic laws depending on present solutes.

        """
        pass
