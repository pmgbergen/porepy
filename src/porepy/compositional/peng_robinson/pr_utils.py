"""This module contains functionality to set up mixtures using the (Peng-Robinson) EoS.

it contains models for some ideal thermodynamic funcionts required for consistent
computations of mixture properties, as well as utility funtions to get e.g., the
binary interaction parameters (BIP) for a mixture of components.

Note:
    Obtaining BIPs requires usually third-party software which is not necessarily
    included in PorePy's requirements (e.g. :mod:`thermo`).

"""

from __future__ import annotations

import warnings
from typing import Callable, TypeVar, Union

import numpy as np

import porepy as pp

from .._core import R_IDEAL_MOL, T_REF
from ..base import Component, Compound
from ..chem_species import load_species

__all__ = [
    "thd_function_type",
    "h_ideal_H2O",
    "h_ideal_CO2",
    "h_ideal_H2S",
    "h_ideal_N2",
    "get_bip_matrix",
    "NaClBrine",
]


NumericType = Union[pp.number, np.ndarray, pp.ad.AdArray]

# NOTE Consider adding typing of objects which support arithmetic overloads
# see https://stackoverflow.com/questions/76821158/
# specify-that-a-typevar-supports-the-operator-among-its-values
_Any = TypeVar("_Any")
"""Type Variable representing a type which supports basic arithmetic operations
``+,-,*, **``, which are used to type the abstract callables represnting thermodynamic
functions in this module."""


thd_function_type = Callable[[_Any], _Any]
"""Type alias for a 1-D, scalar function, taking any type supporting basic arithmetic
operations and returning an instance of that type."""


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


def get_bip_matrix(components: list[Component], package: str = "thermo") -> np.ndarray:
    """Loads the Peng-Robinson binary interaction parameters from a
    third-party database.

    Parameters:
        components: A list of components with valid CASs registry numbers.
        package: ``default='thermo'``

            Third-party package containing databases from which the BIP are loaded.
            Currently supported packages include:

            - ``'thermo'``

    Raises:
        NotImplementedError: If an unsupported package is passed as argument.

    Returns:
        A symmtric 2D array ``bip_matrix`` containing BIP values.

        The row/column order for BIPs corresponds to the order of ``components``.
        I.e., the BIP between ``components[i]`` and ``components[j]`` is given by
        ``bip_matrix[i, j]``.

        Note that ``bip_matrix[i, i]`` is always zero. Zeros in the upper and lower
        triangle of the matrix are most likely a result of missing data in the used
        package. A warnings is issued if that is the case.

    """
    nc = len(components)
    bip_mat = np.zeros((nc, nc))

    # type-hinting how a package-specific BIP fetching function should look like
    # to obtain the BIP for two components identified with their CASs registry number
    # in string format
    fetcher: Callable[[str, str], float]

    if package == "thermo":
        from thermo.interaction_parameters import IPDB

        def fetcher(cas_1: str, cas_2: str) -> float:
            bip = IPDB.get_ip_automatic(CASs=[cas_1, cas_2], ip_type="PR kij", ip="kij")
            return float(bip)

    else:
        raise NotImplementedError(f"Unsupported package `{package}`.")

    for i in range(nc):
        comp_i = components[i]
        for j in range(i + 1, nc):
            comp_j = components[j]

            bip_ij = fetcher(comp_i.CASr_number, comp_j.CASr_number)

            if bip_ij == 0.0:
                warnings.warn(
                    f"Fetched BIP ({package}) for components ({comp_i.name}, "
                    + f"{comp_j.name}) is zero. Most likely due to missing data in"
                    + " third-party package."
                )

            bip_mat[i, j] = bip_ij

    return bip_mat + bip_mat.T


class NaClBrine(Compound):
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
