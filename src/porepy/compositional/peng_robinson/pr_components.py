"""This module contains components for the Peng-Robinson EoS.

It contains an intermediate abstraction of a component through the class
:class:`Component_PR`, as well as various concrete implementations.

The intermediate abstraction :class:`Component_PR` contains optional instance
attributes, which are utilized by the EoS class
:class:`~porepy.compositional.peng_robinson.eos.PengRobinsonEoS`.
They are optional in the sense that they are declared, but not implemented.
Custom implementation must be done in the constructor for child classes.

"""

from __future__ import annotations

from typing import Callable, TypeVar, Union

import numpy as np

import porepy as pp

from .._core import R_IDEAL_MOL, T_REF
from ..base import Component, Compound
from ..chem_species import load_species

__all__ = [
    "ComponentPR",
    "H2O",
    "CO2",
    "H2S",
    "N2",
    "NaClBrine",
]


NumericType = Union[pp.number, np.ndarray, pp.ad.AdArray]

# NOTE Consider adding typing of objects which support arithmetic overloads
# see https://stackoverflow.com/questions/76821158/
# specify-that-a-typevar-supports-the-operator-among-its-values
_Any = TypeVar("_Any")


def h_ideal_H2O(T: _Any) -> _Any:
    """Specific, ideal enthalpy of water based on below reference.

    Can be called with any object supporting overloads of ``+,-,*, **``.

    References:
        `Zhu, Okuno (2015) <https://onepetro.org/spersc/proceedings/15RSS/
        1-15RSS/D011S001R002/183434>`_

    """
    # Units cp_i in [J / mol K^i]
    cp1: float = 32.2
    cp2: float = 1.907e-3
    cp3: float = 1.055e-5
    cp4: float = -3.596e-9
    return (
        cp1 * (T - T_REF)
        + cp2 / 2 * (T**2 - T_REF**2)
        + cp3 / 3 * (T**3 - T_REF**3)
        + cp4 / 4 * (T**4 - T_REF**4)
    )


def h_ideal_CO2(T: _Any) -> _Any:
    """Specific, ideal enthalpy of CO2 based on below reference.

    Can be called with any object supporting overloads of ``+,-,*, **``.

    References:
        `Zhu, Okuno (2014) <http://dx.doi.org/10.1016/j.fluid.2014.07.003>`_

    """
    # Units cp_i in [J / mol K^i]
    cp1: float = 19.795
    cp2: float = 7.343e-2
    cp3: float = -5.602e-5
    cp4: float = 1.715e-8

    return (
        cp1 * (T - T_REF)
        + cp2 / 2 * (T**2 - T_REF**2)
        + cp3 / 3 * (T**3 - T_REF**3)
        + cp4 / 4 * (T**4 - T_REF**4)
    )


def h_ideal_H2S(T: _Any) -> _Any:
    """Specific, ideal enthalpy of CO2 based on below reference.

    Can be called with any object supporting overloads of ``+,-,*, **``.

    References:
        `de Nevers (2012), table A.9 <https://onlinelibrary.wiley.com/doi/
        book/10.1002/9781118135341>`_ .

    """
    # Units cp_i in [J / mol K^i]
    cp1: float = 3.931
    cp2: float = 1.49e-3
    cp3: float = -0.232e5
    return R_IDEAL_MOL * (
        cp1 * (T - T_REF)
        + cp2 / 2 * (T**2 - T_REF**2)
        - cp3 * (T ** (-1) - T_REF ** (-1))
    )


def h_ideal_N2(T: _Any) -> _Any:
    """Specific, ideal enthalpy of N2 based on below reference.

    Can be called with any object supporting overloads of ``+,-,*, **``.

    References:
        `de Nevers (2012), table A.9 <https://onlinelibrary.wiley.com/doi/
        book/10.1002/9781118135341>`_ .

    """
    # Units cp_i in [J / mol K^i]
    cp1: float = 3.280
    cp2: float = 0.593e-3
    cp3: float = 0.04e5
    return R_IDEAL_MOL * (
        cp1 * (T - T_REF)
        + cp2 / 2 * (T**2 - T_REF**2)
        - cp3 * (T ** (-1) - T_REF ** (-1))
    )


class ComponentPR(Component):
    """Intermediate abstraction layer for (fluid) components in a Peng-Robinson mixture.

    Serves for the abstraction of some EoS-specific quantities
    (see attribute declarations).

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.alpha: Callable[[NumericType], NumericType]
        """Abstraction of the cohesion correction term.

        The cohesion ``a`` is given by

        .. math::

            a = a_{crit} \\alpha(T)~,

        consisting of its critical value and a corrective term ``alpha``.

        Per standard Peng-Robinson EoS, ``alpha`` is a linearized expression dependent
        on the acentric factor and the temperature. This is computed by the equation of
        state :class:`~porepy.compositional.peng_robinson.pr_eos.PengRobinsonEoS`.

        If a more specific formula for ``alpha`` is required by a special model,
        this can be implemented by defining a callable ``alpha`` as an attribute of this
        class during construction.

        The EoS will check if ``alpha`` is defined and try to call by passing the
        temperature in some numeric format.

        If ``alpha`` is not an attribute of the Peng-Robinson-Component, the EoS
        will use the default linearization.

        """

        self.bip_map: dict[
            str, Callable[[NumericType], tuple[NumericType, NumericType]]
        ]
        """Abstraction of boundary interaction parameters between this component
        and others.

        The BIP enters into the mixture cohesion formula

        .. math::

            a_{ij} = \\sqrt{a_i a_j} (1 - k_{ij})~,

        where the cohesion term between components ``i != j`` is a non-trivial number
        ``k_ij``, which in general depends on the temperature.

        The other component is defined by its CAS registry number, which is a key for
        this dict.

        The value is, in general, a temperature dependent BIP and its
        temperature-derivative.

        These two must be implemented by a Callable, which takes a numeric
        temperature-value and return the numeric BIP value and its
        temperature-derivative.

        Per standard Peng-Robinson EoS, the BIP is a constant, real number.

        If ``bip_map`` is not defined for a concrete Peng-Robinson component, the
        EOS-class :class:`~porepy.compositional.peng_robinson.pr_eos.PengRobinsonEoS`
        uses :func:`~porepy.compositional.peng_robinson.pr_bip.load_bip` to obtain a
        constant value.

        If ``bip_map`` is defined in a concrete implementation, it uses the
        callables given here to calculate BIPs by passing the temperature
        during a call to
        :meth:`~porepy.compositional.peng_robinson.pr_eos.PengRobinsonEoS.compute`.

        """


class H2O(ComponentPR):
    """Component representing water as a fluid for the Peng-Robinson EoS."""

    cp1: float = 32.2
    """``ci`` are heat capacity coefficients at constant pressure
    (see :meth:`h_ideal`) given in ``[J / mol K^i]``."""
    cp2: float = 1.907e-3
    cp3: float = 1.055e-5
    cp4: float = -3.596e-9

    def h_ideal(self, p: NumericType, T: NumericType) -> NumericType:
        """The specific molar enthalpy of is constructed using
        heat capacity coefficients from
        `Zhu, Okuno (2015) <https://onepetro.org/spersc/proceedings/15RSS/
        1-15RSS/D011S001R002/183434>`_ .

        """
        return (
            self.cp1 * (T - T_REF)
            + self.cp2 / 2 * (T**2 - T_REF**2)
            + self.cp3 / 3 * (T**3 - T_REF**3)
            + self.cp4 / 4 * (T**4 - T_REF**4)
        )


class CO2(ComponentPR):
    """Component representing carbon dioxide as a fluid for the Peng-Robinson EoS."""

    cp1: float = 19.795
    """``ci`` are heat capacity coefficients at constant pressure
    (see :meth:`h_ideal`) given in ``[J / mol K^i]``."""
    cp2: float = 7.343e-2
    cp3: float = -5.602e-5
    cp4: float = 1.715e-8

    def h_ideal(self, p: NumericType, T: NumericType) -> NumericType:
        """The specific molar enthalpy of is constructed using
        heat capacity coefficients from
        `Zhu, Okuno (2014) <http://dx.doi.org/10.1016/j.fluid.2014.07.003>`_ .

        """
        return (
            self.cp1 * (T - T_REF)
            + self.cp2 / 2 * (T**2 - T_REF**2)
            + self.cp3 / 3 * (T**3 - T_REF**3)
            + self.cp4 / 4 * (T**4 - T_REF**4)
        )


class H2S(ComponentPR):
    """Component representing hydrogen sulfide as a fluid for the Peng-Robinson EoS."""

    cp1: float = 3.931
    """``ci`` are heat capacity coefficients at constant pressure
    (see :meth:`h_ideal`) given in ``[J / mol K^i]``."""
    cp2: float = 1.49e-3
    cp3: float = -0.232e5

    def h_ideal(self, p: NumericType, T: NumericType) -> NumericType:
        """The specific molar enthalpy of is constructed using
        heat capacity coefficients from
        `de Nevers (2012), table A.9 <https://onlinelibrary.wiley.com/doi/
        book/10.1002/9781118135341>`_ .

        """
        return R_IDEAL_MOL * (
            self.cp1 * (T - T_REF)
            + self.cp2 / 2 * (T**2 - T_REF**2)
            - self.cp3 * (T ** (-1) - T_REF ** (-1))
        )


class N2(ComponentPR):
    """Component representing nitrogen as a fluid for the Peng-Robinson EoS.

    There is a questionable implementations for BIP between water and nitrogen.
    Reference can be found `here <https://www.mdpi.com/1996-1073/14/17/5239>`_,
    but it was designed for another EoS (SRK), with a value 0.385438.

    """

    cp1: float = 3.280
    """``ci`` are heat capacity coefficients at constant pressure
    (see :meth:`h_ideal`) given in ``[J / mol K^i]``."""
    cp2: float = 0.593e-3
    cp3: float = 0.04e5

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        cas_water = load_species(["H2O"])[0].CASr_number

        def bip_water(T: NumericType) -> tuple[NumericType, NumericType]:
            return 0.385438, 0

        self.bip_map = {cas_water: bip_water}

    def h_ideal(self, p: NumericType, T: NumericType) -> NumericType:
        """The specific molar enthalpy of is constructed using
        heat capacity coefficients from
        `de Nevers (2012), table A.9 <https://onlinelibrary.wiley.com/doi/
        book/10.1002/9781118135341>`_ .

        """
        return R_IDEAL_MOL * (
            self.cp1 * (T - T_REF)
            + self.cp2 / 2 * (T**2 - T_REF**2)
            - self.cp3 * (T ** (-1) - T_REF ** (-1))
        )


class NaClBrine(Compound, H2O):
    """A compound representing water - sodium chloride brine, where water is the solvent
    and NaCl a solute.

    A special model for the cohesion correction :attr:`alpha` and BIPs are implemented,
    see reference.

    References:
        [1] `Soereide (1992) <https://doi.org/10.1016/0378-3812(92)85105-H>`_

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # instantiate NaCl_ps as a solute
        solute = load_species(["NaCl"], species_type="basic")
        # add solute to self
        self.pseudo_components = solute

        # store NaCl for quick access
        self.NaCl = solute
        """Reference to the pseudo-component representing NaCl."""

        def alpha(T: NumericType) -> NumericType:
            # molal salinity
            T_r = T / self.T_crit
            b = self.molalities[1]

            alpha_ = (
                1 + 0.453 * (1 - T_r * (1 - 0.0103 * b**1.1)) + 0.0034 * (T_r**-3 - 1)
            )

            return alpha_

        self.alpha = alpha

        co2, h2s, n2 = load_species(["CO2", "H2S", "N2"])

        def bip_co2(T: NumericType) -> tuple[NumericType, NumericType]:
            T_r = T / co2.T_crit
            b = self.molalities[1]

            return (
                T_r * 0.23580 * (1 + 0.17837 * b**0.979)
                - 21.2566 * pp.ad.exp(-6.7222 * T_r - b)
                - 0.31092 * (1 + 0.15587 * b**0.7505)
            ), (
                21.2566 * pp.ad.exp(-6.7222 * T_r - b) * (6.7222 / co2.T_crit)
                + 0.23580 * (1 + 0.17837 * b**0.979) / co2.T_crit
            )

        def bip_h2s(T: NumericType) -> tuple[NumericType, NumericType]:
            T_r = T / h2s.T_crit
            return (-0.20441 + 0.23426 * T_r, 0.23426 / h2s.T_crit)

        def bip_n2(T: NumericType) -> tuple[NumericType, NumericType]:
            T_r = T / n2.T_crit
            b = self.molalities[1]

            return (
                T_r * 0.44338 * (1 + 0.08126 * b**0.75)
                - 1.70235 * (1 + 0.25587 * b**0.75)
            ), (0.44338 * (1 + 0.08126 * b**0.75) / n2.T_crit)

        self.bip_map = {
            co2.CASr_number: bip_co2,
            h2s.CASr_number: bip_h2s,
            n2.CASr_number: bip_n2,
        }
