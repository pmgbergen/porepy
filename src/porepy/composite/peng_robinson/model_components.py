"""Contains concrete implementations of modelled components for the Peng-Robinson EoS."""
from __future__ import annotations

import porepy as pp

from ..component import PseudoComponent
from ..phase import VarLike
from .pr_component import PR_Component, PR_Compound

__all__ = [
    "H2O_ps",
    "NaCl_ps",
    "CO2_ps",
    "H2S_ps",
    "N2_ps",
    "H2O",
    "CO2",
    "H2S",
    "N2",
    "NaClBrine",
]

# region Pseudo-components --------------------------------------------------------------------


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


class N2_ps(PseudoComponent):
    """Pseud-representation of nitrogen."""

    @staticmethod
    def molar_mass():
        """`Source <https://doi.org/10.1016/0378-3812(92)85105-H>`_."""
        return 0.028

    @staticmethod
    def critical_pressure():
        """`Source <https://en.wikipedia.org/wiki/Nitrogen>`_."""
        return 3.39

    @staticmethod
    def critical_temperature():
        """`Source <https://en.wikipedia.org/wiki/Nitrogen>`_."""
        return 126.21


# endregion
# region Components ---------------------------------------------------------------------------


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


class N2(PR_Component, N2_ps):
    """Component representing N2 as a fluid for the Peng-Robinson EoS.

    Constant physical properties are inherited from respective pseudo-component.

    """

    @property
    def acentric_factor(self) -> float:
        """`Source <https://doi.org/10.1016/0378-3812(92)85105-H>`_."""
        return 0.0403


# endregion
# region Compounds ----------------------------------------------------------------------------


class NaClBrine(PR_Compound, H2O):
    """A compound representing water - sodium chloride brine, where water is the solvent
    and NaCl the solute.

    This class instantiates :class:`H2O_ps` and :class:`NaCl_ps` internally and assigns
    them as solvent and solute.

    Adaptions to attraction correction are made according to given
    references.

    The acentric factor is given by inheritance from :class:`H2O`, as well as the critical
    temperature.

    """

    def __init__(self, ad_system: pp.ad.ADSystem) -> None:

        # instantiate H2O_ps as the solvent
        solvent = H2O_ps(ad_system=ad_system)

        # super call to constructor with above solvent
        super().__init__(ad_system, solvent)

        # instantiate NaCl_ps as a solute
        solute = NaCl_ps(ad_system=ad_system)
        # add solute to self
        self.add_solute(solute)
        # store NaCl for quick access
        self.NaCl: PseudoComponent = solute
        """Quick reference to the pseudo-component representing NaCl."""

    def attraction_correction(self, T: VarLike) -> VarLike:
        """The attraction correction for NaCl-brine based on molal salinity can be found in
        `Soereide (1992), equation 9 <https://doi.org/10.1016/0378-3812(92)85105-H>`_

        """
        # molal salinity
        T_r = T / self.critical_temperature()
        cw = self.molality_of(self.NaCl)
        power = pp.ad.Function(pp.ad.power, "power")
        exponent_1 = pp.ad.Scalar(1.1)
        exponent_2 = pp.ad.Scalar(-3)

        alpha_root = (
            1
            + 0.453 * (1 - T_r * (1 - 0.0103 * power(cw, exponent_1)))
            + 0.0034 * (power(T_r, exponent_2) - 1)
        )

        return alpha_root * alpha_root


# endregion
