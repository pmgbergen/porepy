"""Contains extended, abstract classes representing components and their properties and
parameters used in the Peng-Robinson EoS.

"""
from __future__ import annotations

import abc

from .._composite_utils import R_IDEAL
from ..component import Component


class PR_Component(Component):
    """Intermediate abstraction layer for components in a Peng-Robinson mixture."""

    @staticmethod
    @abc.abstractmethod
    def critical_pressure() -> float:
        """This is a constant value, hence to be a static function.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [kPa]

        Returns: critical pressure for this component (critical point in p-T diagram).

        """
        pass

    @staticmethod
    @abc.abstractmethod
    def critical_temperature() -> float:
        """This is a constant value, hence to be a static function.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [K]

        Returns: critical temperature for this component (critical point in p-T diagram).

        """
        pass


class PR_FluidComponent(PR_Component):
    """Intermediate abstraction layer for components which are only expected in a fluid phase
    of a Peng-Robinson mixture.

    Serves for the abstraction of properties which are usually only associated with this type
    of component.

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
    def covolume(self) -> float:
        """Returns the constant co-volume ``b`` in the Peng-Robinson EoS:

        ``b = 0.077796072 * (R_IDEAL * T_critical) / p_critical

        """
        return (
            0.077796072
            * (R_IDEAL * self.critical_temperature())
            / self.critical_pressure()
        )

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


class PR_SoluteComponent(PR_Component):
    """Intermediate abstraction layer for components which are only expected as solutes in a
    fluid phase of the Peng-Robinson mixture.

    Serves for the abstraction of properties which are usually only associated with this type
    of component.

    """

    pass
