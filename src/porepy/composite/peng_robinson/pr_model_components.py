"""This module contains concrete implementations of components for the
Peng-Robinson EoS."""
from __future__ import annotations

import porepy as pp
from porepy.numerics.ad.operator_functions import NumericType

from .._core import R_IDEAL, T_REF
from ..component import PseudoComponent
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

# region Pseudo-components -------------------------------------------------------------


class H2O_ps(PseudoComponent):
    """Pseud-representation of water, including triple point data."""

    @staticmethod
    def molar_mass():
        """`Source <https://en.wikipedia.org/wiki/Water_vapor>`_."""
        return 0.0180152833

    @staticmethod
    def critical_pressure():
        """`Source <https://doi.org/10.1016/j.gca.2006.01.033>`_."""
        # return 22.06
        return 22.04832

    @staticmethod
    def critical_temperature():
        """`Source <https://doi.org/10.1016/j.gca.2006.01.033>`_."""
        # return 647.096
        return 647.14

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
        # return 7.38
        return 7.376460

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
# region Components --------------------------------------------------------------------


class H2O(PR_Component, H2O_ps):
    """Component representing water as a fluid for the Peng-Robinson EoS.

    Constant physical properties are inherited from respective pseudo-component.

    """

    cp1: float = 0.0322
    """``ci`` are heat capacity coefficients at constant pressure
    (see :meth:`h_ideal`) given in [kJ / mol K^i]."""
    cp2: float = 1.904e-6
    cp3: float = 1.055e-8
    cp4: float = -3.596e-12

    @property
    def acentric_factor(self) -> float:
        """`Source <https://doi.org/10.1016/0378-3812(92)85105-H>`_."""
        # return 0.3434
        return 0.344

    def h_ideal(self, p: NumericType, T: NumericType) -> NumericType:
        """The specific molar enthalpy of is constructed using
        heat capacity coefficients from
        `Zhu, Okuno (2015) <https://onepetro.org/spersc/proceedings/15RSS/
        1-15RSS/D011S001R002/183434>`_ .

        """
        return (
            self.cp1 * (T - T_REF)
            + self.cp2 / 2 * (pp.ad.power(T, 2) - T_REF**2)
            + self.cp3 / 3 * (pp.ad.power(T, 3) - T_REF**3)
            + self.cp4 / 4 * (pp.ad.power(T, 4) - T_REF**4)
        )


class CO2(PR_Component, CO2_ps):
    """Component representing CO2 as a fluid for the Peng-Robinson EoS.

    Constant physical properties are inherited from respective pseudo-component.

    """

    cp1: float = 0.019795
    """``ci`` are heat capacity coefficients at constant pressure
    (see :meth:`h_ideal`) given in [kJ / mol K^i]."""
    cp2: float = 7.343e-5
    cp3: float = -5.602e-8
    cp4: float = 1.715e-11

    @property
    def acentric_factor(self) -> float:
        """`Source <https://doi.org/10.1016/0378-3812(92)85105-H>`_."""
        # return 0.2273
        return 0.2252

    def h_ideal(self, p: NumericType, T: NumericType) -> NumericType:
        """The specific molar enthalpy of is constructed using
        heat capacity coefficients from
        `Zhu, Okuno (2014) <http://dx.doi.org/10.1016/j.fluid.2014.07.003>`_ .

        """
        return (
            self.cp1 * (T - T_REF)
            + self.cp2 / 2 * (pp.ad.power(T, 2) - T_REF**2)
            + self.cp3 / 3 * (pp.ad.power(T, 3) - T_REF**3)
            + self.cp4 / 4 * (pp.ad.power(T, 4) - T_REF**4)
        )


class H2S(PR_Component, H2S_ps):
    """Component representing H2S as a fluid for the Peng-Robinson EoS.

    Constant physical properties are inherited from respective pseudo-component.

    """

    cp1: float = 3.931
    """``ci`` are heat capacity coefficients at constant pressure
    (see :meth:`h_ideal`) given in [kJ / mol K^i]."""
    cp2: float = 1.49 - 3
    cp3: float = -0.232e5

    @property
    def acentric_factor(self) -> float:
        """`Source <https://doi.org/10.1016/0378-3812(92)85105-H>`_."""
        return 0.1081

    def h_ideal(self, p: NumericType, T: NumericType) -> NumericType:
        """The specific molar enthalpy of is constructed using
        heat capacity coefficients from
        `de Nevers (2012), table A.9 <https://onlinelibrary.wiley.com/doi/
        book/10.1002/9781118135341>`_ .

        """
        return R_IDEAL * (
            self.cp1 * (T - T_REF)
            + self.cp2 / 2 * (pp.ad.power(T, 2) - T_REF**2)
            - self.cp3 * (pp.ad.power(T, -1) - T_REF ** (-1))
        )


class N2(PR_Component, N2_ps):
    """Component representing N2 as a fluid for the Peng-Robinson EoS.

    Constant physical properties are inherited from respective pseudo-component.

    """

    cp1: float = 3.280
    """``ci`` are heat capacity coefficients at constant pressure
    (see :meth:`h_ideal`) given in [kJ / mol K^i]."""
    cp2: float = 0.593e-3
    cp3: float = 0.04e5

    @property
    def acentric_factor(self) -> float:
        """`Source <https://doi.org/10.1016/0378-3812(92)85105-H>`_."""
        return 0.0403

    def h_ideal(self, p: NumericType, T: NumericType) -> NumericType:
        """The specific molar enthalpy of is constructed using
        heat capacity coefficients from
        `de Nevers (2012), table A.9 <https://onlinelibrary.wiley.com/doi/
        book/10.1002/9781118135341>`_ .

        """
        return R_IDEAL * (
            self.cp1 * (T - T_REF)
            + self.cp2 / 2 * (pp.ad.power(T, 2) - T_REF**2)
            - self.cp3 * (pp.ad.power(T, -1) - T_REF ** (-1))
        )


# endregion
# region Compounds ---------------------------------------------------------------------


class NaClBrine(PR_Compound, H2O):
    """A compound representing water - sodium chloride brine, where water is the solvent
    and NaCl the solute.

    This class instantiates :class:`H2O_ps` and :class:`NaCl_ps` internally and assigns
    them as solvent and solute.

    Adaptions to attraction correction are made according to given references.

    The acentric factor is given by inheritance from :class:`H2O`,
    as well as the critical temperature.

    """

    def __init__(self, ad_system: pp.ad.EquationSystem) -> None:

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
        """Reference to the pseudo-component representing NaCl."""

    def cohesion_correction(self, T: NumericType) -> NumericType:
        """The attraction correction for NaCl-brine based on molal salinity can be found
        in
        `Soereide (1992), equation 9 <https://doi.org/10.1016/0378-3812(92)85105-H>`_

        """
        # molal salinity
        T_r = T / self.critical_temperature()
        cw = self.molality_of(self.NaCl)

        alpha = (
            1
            + 0.453 * (1 - T_r * (1 - 0.0103 * pp.ad.power(cw, 1.1)))
            + 0.0034 * (pp.ad.power(T_r, -3) - 1)
        )

        return alpha


# endregion
