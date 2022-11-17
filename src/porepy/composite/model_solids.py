""" Contains concrete implementations of components which appear as solutes in fluids,
but also possibly in the porous medium.
"""

from .component import SolidComponent

__all__ = ["NaCl"]


class NaCl(SolidComponent):
    """Solute component representing sodium chloride.

    The physical properties and parameters are found in respective references.

    """

    @staticmethod
    def molar_mass():
        """`Source <https://en.wikipedia.org/wiki/Sodium_chloride>`_."""
        return 0.058443

    @staticmethod
    def critical_pressure():
        """`Source <https://doi.org/10.1016/j.gca.2006.01.033>`_."""
        return 18200

    @staticmethod
    def critical_temperature():
        """`Source <https://doi.org/10.1016/j.gca.2006.01.033>`_."""
        return 3841.15

    @staticmethod
    def triple_point_pressure():
        """`Source <https://en.wikipedia.org/wiki/Sodium_chloride_(data_page)>`_."""
        return 0.03

    @staticmethod
    def triple_point_temperature():
        """`Source <https://en.wikipedia.org/wiki/Sodium_chloride_(data_page)>`_."""
        return 1074
