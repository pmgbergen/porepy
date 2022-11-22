"""Contains concrete implementations of modelled components for the Peng-Robinson EoS."""
from __future__ import annotations

from ..component import PseudoComponent
from .pr_component import PR_Component, PR_Compound

__all__ = [
    "BINARY_INTERACTION_PARAMETERS",
    "H2O_ps",
    "NaCl_ps",
    "CO2_ps",
    "H2S_ps",
    "H2O",
    "CO2",
    "H2S",
]


BINARY_INTERACTION_PARAMETERS: dict[tuple[str, str], float] = {("H2O", "NaCl"): 1.0}
"""Contains for a pair of component names (key) the respective binary interaction parameter."""


class H2O_ps(PseudoComponent):
    """Pseud-representation of water, including triple point data."""

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


class NaCl_ps(PseudoComponent):
    """Pseudo-representation of Sodium Chloride, including triple point data."""

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


class CO2_ps(PseudoComponent):
    """Pseud-representation of carbon dioxide."""

    @staticmethod
    def molar_mass():
        """`Source <https://en.wikipedia.org/wiki/Carbon_dioxide>`_."""
        return 0.044009

    @staticmethod
    def critical_pressure():
        """`Source <https://doi.org/10.1016/0378-3812(92)85105-H>`_."""
        return 7.38

    @staticmethod
    def critical_temperature():
        """`Source <https://doi.org/10.1016/0378-3812(92)85105-H>`_."""
        return 304.2


class H2S_ps(PseudoComponent):
    """Pseud-representation of hydrogen sulfide."""

    @staticmethod
    def molar_mass():
        """`Source <https://en.wikipedia.org/wiki/Hydrogen_sulfide>`_."""
        return 0.03408

    @staticmethod
    def critical_pressure():
        """`Source <https://doi.org/10.1016/0378-3812(92)85105-H>`_."""
        return 8.94

    @staticmethod
    def critical_temperature():
        """`Source <https://doi.org/10.1016/0378-3812(92)85105-H>`_."""
        return 373.2


class H2O(PR_Component, H2O_ps):
    """Component representing water as a fluid for the Peng-Robinson EoS.

    Constant physical properties are inherited from respective pseudo-component.

    """

    @property
    def acentric_factor(self) -> float:
        """`Source <https://doi.org/10.1016/0378-3812(92)85105-H>`_."""
        return 0.3434


class CO2(PR_Component, CO2_ps):
    """Component representing CO2 as a fluid for the Peng-Robinson EoS.

    Constant physical properties are inherited from respective pseudo-component.

    """

    @property
    def acentric_factor(self) -> float:
        """`Source <https://doi.org/10.1016/0378-3812(92)85105-H>`_."""
        return 0.2273


class H2S(PR_Component, H2S_ps):
    """Component representing H2S as a fluid for the Peng-Robinson EoS.

    Constant physical properties are inherited from respective pseudo-component.

    """

    @property
    def acentric_factor(self) -> float:
        """`Source <https://doi.org/10.1016/0378-3812(92)85105-H>`_."""
        return 0.1081
