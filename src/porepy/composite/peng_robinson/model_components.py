"""Contains concrete implementations of modelled components for the Peng-Robinson EoS."""
from __future__ import annotations

from .pr_component import PR_FluidComponent, PR_SoluteComponent

__all__ = ["H2O", "NaCl"]


class H2O(PR_FluidComponent):
    """Fluid component representing water as a fluid for the Peng-Robinson EoS.

    The physical properties and parameters are found in respective references.

    """

    @staticmethod
    def molar_mass():
        """`Source <https://en.wikipedia.org/wiki/Water_vapor>`_."""
        return 0.0180152833

    @staticmethod
    def critical_pressure():
        """`Source <https://doi.org/10.1016/j.gca.2006.01.033>`_."""
        return 22.060

    @staticmethod
    def critical_temperature():
        """`Source <https://doi.org/10.1016/j.gca.2006.01.033>`_."""
        return 647.096

    @staticmethod
    def triple_point_pressure():
        """`Source <https://en.wikipedia.org/wiki/Water_(data_page)>`_."""
        return 0.00061173

    @staticmethod
    def triple_point_temperature():
        """`Source <https://en.wikipedia.org/wiki/Water_(data_page)>`_."""
        return 273.1600

    @property
    def acentric_factor(self) -> float:
        """`Source
        <https://ebookcentral.proquest.com/lib/bergen-ebooks/detail.action?docID=317176>`_."""
        return 0.3449


class NaCl(PR_SoluteComponent):
    """Solute component representing sodium chloride for the Peng-Robinson EoS with adaptions
    for solutes.

    The physical properties and parameters are found in respective references.

    """

    @staticmethod
    def molar_mass():
        """`Source <https://en.wikipedia.org/wiki/Sodium_chloride>`_."""
        return 0.058443

    @staticmethod
    def critical_pressure():
        """`Source <https://doi.org/10.1016/j.gca.2006.01.033>`_."""
        return 18.200

    @staticmethod
    def critical_temperature():
        """`Source <https://doi.org/10.1016/j.gca.2006.01.033>`_."""
        return 3841.15

    @staticmethod
    def triple_point_pressure():
        """`Source <https://en.wikipedia.org/wiki/Sodium_chloride_(data_page)>`_."""
        return 0.00003

    @staticmethod
    def triple_point_temperature():
        """`Source <https://en.wikipedia.org/wiki/Sodium_chloride_(data_page)>`_."""
        return 1074
