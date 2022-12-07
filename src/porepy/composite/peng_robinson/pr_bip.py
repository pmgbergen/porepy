"""This modules contains Binary Interaction Parameters for components modelled for the
Peng-Robinson EoS.

BIPs are intended for genuine components (and compounds), but not for pseudo-components.
The effect of pseudo-components is integrated in respective interaction law involving
compounds.

References for these largely heuristic laws and values can be found in respective
implementations.

BIPs are implemented as callable objects. This module provides a map ``BIP_MAP`` which maps
between two components and their respective BIP.

The BIP between a component/compound and itself is assumed to be 0, and hence not given here.

It is recommended to use the function :meth:`get_PR_BIP` to get a reference to the respective
callable. This function also provides information on how to use a BIP in terms of argument
order (component 1 and component 2).

Examples:
    >>> import porepy as pp
    >>> composition = pp.composite.PR_Composition()
    >>> H2O = pp.composite.H2O(composition.ad_system)
    >>> CO2 = pp.composite.CO2(composition.ad_system)
    >>> bip, order = pp.composite.get_PR_BIP(H2O.name, CO2.name)
    >>> bip_co2_h2o = bip(composition.T, H2O, CO2) if order else bip(composition.T, CO2, H2O)

    Note that the last line will raise an error if there is no BIP implemented for `H2O` and
    `CO2` (`bip` is None in that case).

"""
from __future__ import annotations

from typing import Callable, Literal

import porepy as pp

from ..phase import VarLike
from .pr_model_components import *
from .pr_utils import _power, _exp

__all__ = [
    "PR_BIP_MAP",
    "get_PR_BIP",
]


def bip_H2O_CO2(T: VarLike, h2o: H2O, co2: CO2) -> pp.ad.Operator:
    """(Constant) BIP for water and carbon dioxide.

    The value is taken from
    `thermo <https://thermo.readthedocs.io/thermo.interaction_parameters.html>`_,
    ``IPDB.get_ip_automatic`` with CAS numbers:

    - H2O: 7732-18-5
    - CO2: 124-38-9

    Returns:
        An AD Operator representing the constant scalar BIP.

    """
    return pp.ad.Scalar(0.0952)


def bip_H2O_H2S(T: VarLike, h20: H2O, h2s: H2S) -> pp.ad.Operator:
    """(Constant) BIP for water and hydrogen sulfide.

    The value is taken from
    `thermo <https://thermo.readthedocs.io/thermo.interaction_parameters.html>`_,
    ``IPDB.get_ip_automatic`` with CAS numbers:

    - H2O: 7732-18-5
    - H2S: 7783-06-4

    Returns:
        An AD Operator representing the constant scalar BIP.

    """
    return pp.ad.Scalar(0.0394)


def bip_H2O_N2(T: VarLike, h2o: H2O, n2: N2) -> pp.ad.Operator:
    """(Constant) BIP for water and nitrogen.

    The law is taken from
    `Haghighi et al. (2009), equation 11 <https://doi.org/10.1016/j.fluid.2008.10.006>`_.

    Warning:
        The validity of this law is highly questionable, since it was evaluated for the
        Cubic+ EoS.

        An alternative would be `this one <https://www.mdpi.com/1996-1073/14/17/5239>`_,
        but it was also designed for another EoS (SRK), with a value 0.385438.

    Returns:
        An AD Operator representing the constant scalar BIP.

    """
    return 0.9909 - 379.9691 / T


def dT_bip_H2O_N2(T: VarLike, h2o: H2O, n2: N2) -> pp.ad.Operator:
    """Analytical derivative of :func:`bip_H2O_N2` w.r.t. temperature ``T``."""
    return 379.9691 / (T * T)


def bip_CO2_H2S(T: VarLike, co2: CO2, h2s: H2S) -> pp.ad.Operator:
    """(Constant) BIP for carbon dioxide and hydrogen sulfide.

    The value is taken from
    `thermo <https://thermo.readthedocs.io/thermo.interaction_parameters.html>`_,
    ``IPDB.get_ip_automatic`` with CAS numbers:

    - CO2: 124-38-9
    - H2S: 7783-06-4

    Returns:
        An AD Operator representing the constant scalar BIP.

    """
    return pp.ad.Scalar(0.0967)


def bip_CO2_N2(T: VarLike, co2: CO2, n2: N2) -> pp.ad.Operator:
    """(Constant) BIP for carbon dioxide and nitrogen.

    The value is taken from
    `thermo <https://thermo.readthedocs.io/thermo.interaction_parameters.html>`_,
    ``IPDB.get_ip_automatic`` with CAS numbers:

    - CO2: 124-38-9
    - N2: 7727-37-9

    Returns:
        An AD Operator representing the constant scalar BIP.

    """
    return pp.ad.Scalar(-0.0122)


def bip_N2_H2S(T: VarLike, n2: N2, h2s: H2S) -> pp.ad.Operator:
    """(Constant) BIP for nitrogen and hydrogen sulfide.

    The value is taken from
    `thermo <https://thermo.readthedocs.io/thermo.interaction_parameters.html>`_,
    ``IPDB.get_ip_automatic`` with CAS numbers:

    - H2S: 7783-06-4
    - N2: 7727-37-9

    Returns:
        An AD Operator representing the constant scalar BIP.

    """
    return pp.ad.Scalar(0.1652)


def bip_NaClBrine_H2S(T: VarLike, naclbrine: NaClBrine, h2s: H2S) -> pp.ad.Operator:
    """Temperature- and salinity-dependent BIP for NaCl-brine and hydrogen sulfide.

    The law is taken from
    `Soereide (1992), equation 15 <https://doi.org/10.1016/0378-3812(92)85105-H>`_.

    Returns:
        An AD Operator representing the BIP.

    """
    T_r = T / h2s.critical_temperature()
    return -0.20441 + 0.23426 * T_r


def dT_bip_NaClBrine_H2S(T: VarLike, naclbrine: NaClBrine, h2s: H2S) -> pp.ad.Operator:
    """Analytical derivative of :func:`bip_NaClBrine_H2S` w.r.t. temperature ``T``."""
    return 0.23426 / h2s.critical_temperature()


def bip_NaClBrine_CO2(T: VarLike, naclbrine: NaClBrine, co2: CO2) -> pp.ad.Operator:
    """Temperature- and salinity-dependent BIP for NaCl-brine and carbon dioxide.

    The law is taken from
    `Soereide (1992), equation 14 <https://doi.org/10.1016/0378-3812(92)85105-H>`_.

    Returns:
        An AD Operator representing the BIP.

    """
    T_r = T / co2.critical_temperature()
    molality = naclbrine.molality_of(naclbrine.NaCl)
    exponent_1 = pp.ad.Scalar(0.7505)
    exponent_2 = pp.ad.Scalar(0.979)

    return (
        -0.31092 * (1 + 0.15587 * _power(molality, exponent_1))
        + 0.23580 * (1 + 0.17837 * _power(molality, exponent_2)) * T_r
        - 21.2566 * _exp(-6.7222 * T_r - molality)
    )


def dT_bip_NaClBrine_CO2(T: VarLike, naclbrine: NaClBrine, co2: CO2) -> pp.ad.Operator:
    """Analytical derivative of :func:`bip_NaClBrine_CO2` w.r.t. temperature ``T``."""
    T_r = T / co2.critical_temperature()
    molality = naclbrine.molality_of(naclbrine.NaCl)
    exponent_2 = pp.ad.Scalar(0.979)

    return 0.23580 * (
        1 + 0.17837 * _power(molality, exponent_2)
    ) / co2.critical_temperature() + 21.2566 * _exp(-6.7222 * T_r - molality) * (
        6.7222 / co2.critical_temperature()
    )


def bip_NaClBrine_N2(T: VarLike, naclbrine: NaClBrine, n2: N2) -> pp.ad.Operator:
    """Temperature- and salinity-dependent BIP for NaCl-brine and nitrogen.

    The law is taken from
    `Soereide (1992), equation 13 <https://doi.org/10.1016/0378-3812(92)85105-H>`_.

    Returns:
        An AD Operator representing the BIP.

    """
    T_r = T / n2.critical_temperature()
    molality = naclbrine.molality_of(naclbrine.NaCl)
    exponent = pp.ad.Scalar(0.75)

    return (
        -1.70235 * (1 + 0.25587 * _power(molality, exponent))
        + 0.44338 * (1 + 0.08126 * _power(molality, exponent)) * T_r
    )


def dT_bip_NaClBrine_N2(T: VarLike, naclbrine: NaClBrine, n2: N2) -> pp.ad.Operator:
    """Analytical derivative of :func:`bip_NaClBrine_N2` w.r.t. temperature ``T``."""
    molality = naclbrine.molality_of(naclbrine.NaCl)
    exponent = pp.ad.Scalar(0.75)

    return (
        0.44338 * (1 + 0.08126 * _power(molality, exponent)) / n2.critical_temperature()
    )


PR_BIP_MAP: dict[tuple[str, str], Callable] = {
    ("H2O", "CO2"): (bip_H2O_CO2, 0),
    ("H2O", "H2S"): (bip_H2O_H2S, 0),
    ("H2O", "N2"): (bip_H2O_N2, dT_bip_H2O_N2),
    ("CO2", "H2S"): (bip_CO2_H2S, 0),
    ("CO2", "N2"): (bip_CO2_N2, 0),
    ("N2", "H2S"): (bip_N2_H2S, 0),
    ("NaClBrine", "H2S"): (bip_NaClBrine_H2S, dT_bip_NaClBrine_H2S),
    ("NaClBrine", "CO2"): (bip_NaClBrine_CO2, dT_bip_NaClBrine_CO2),
    ("NaClBrine", "N2"): (bip_NaClBrine_N2, dT_bip_NaClBrine_N2),
}
"""Contains for a pair of component/compound names (key) the respective
binary interaction parameter for the Peng-Robinson EoS and their derivative w.r.t. temperature,
in form of a tuple of callables, or a callable and 0 if the derivative is trivial
(constant BIP).

This map serves the Peng-Robinson composition to assemble the attraction parameter of the
mixture and its intended use is only there.

"""


def get_PR_BIP(
    component1: str, component2: str
) -> tuple[Callable | None, Callable | Literal[0], bool]:
    """Returns the callables representing a BIP and its derivative for two given component
    names, in the Peng-Robinson EoS.

    This function is a wrapper for accessing :data:`BIP_MAP`, which is not sensitive
    the order in the 2-tuple containing component names.

    Parameters:
        component1: name of the first component
        component2: name of the second component

    Returns:
        The returned 3-tuple contains:

        1. A callable implemented which represents the BIP for given components.
           It is ``None``, if the BIP is not available.
        2. A second callable, or 0, for the derivative of the BIP w.r.t. temperature.
           A zero indicates that the derivative is trivial.
        3. A bool indicating whether the order of input arguments
           fits the order for the BIP arguments. It is ``False``, if the BIP argument order is
           ``component2, component1``. If no BIP is found, the bool has no meaning.

    """
    # try input order
    BIP, dT_BIP = PR_BIP_MAP.get((component1, component2), None)
    order = True

    # try reverse order
    if BIP is None:
        BIP, dT_BIP = PR_BIP_MAP.get((component2, component1), None)
        order = False

    # return what is found, possibly None
    return BIP, dT_BIP, order
