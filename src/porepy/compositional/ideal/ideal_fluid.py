"""Module containing functionality representing ideal properties of a fluid, as well
as examples for H2O, CO2, H2S and N2.

Heat capacity coefficients for the provided examples is taken from below reference.

References:
    [1] `de Nevers (2012)
    <https://onlinelibrary.wiley.com/doi/book/10.1002/9781118135341>`_ .

"""

from __future__ import annotations

import logging
from typing import Callable, Literal, Optional, TypeAlias, Union

import numba as nb
import numpy as np

from .._global_thermodynamic_reference_state import R_U
from .._numba_interface import NUMBA_FAST_MATH, njit

__all__ = [
    "IdealProperty_T",
    "IdealProperty_pT",
    "GradIdealProperty_pT",
    "IdealProperty",
    "ideal_rho",
    "grad_ideal_rho",
    "ideal_v",
    "grad_ideal_v",
    "IdealFluid",
]


logger = logging.getLogger(__name__)


IdealProperty_T: TypeAlias = Callable[[float], float]
"""Typing of ideal property functions which depends only on 1 intensive state functions
(temperature).

Used for enthalpy and internal energy.

"""


IdealProperty_pT: TypeAlias = Callable[[float, float], float]
"""Type of ideal property function which depends on 2 intensive state functions,
pressure and temperature.

Used for Gibbs energy and entropy.

"""

GradIdealProperty_pT: TypeAlias = Callable[[float, float], np.ndarray]
"""Type of derivative of ideal property function which depends on 2 intensive state
functions, pressure and temperature.

Used to type derivatives of a :data:`IdealProperty_pT`.

"""

IdealProperty: TypeAlias = Union[
    IdealProperty_T, IdealProperty_pT, GradIdealProperty_pT
]
"""Union alias for all ideal property signatures."""


_IDP_T_COMPILER = njit(nb.f8(nb.f8))
"""Compiler for ideal properties depending only on Temperature."""


_IDP_pT_COMPILER = njit(nb.f8(nb.f8, nb.f8))
"""Compiler for ideal properties depending on both pressure and temperature."""


@njit(nb.f8(nb.f8, nb.f8), fastmath=NUMBA_FAST_MATH, cache=True)
def ideal_rho(p: float, T: float) -> float:
    """Ideal gas density.

    Parameters:
        p: Pressure.
        T: Temperature.

    Returns:
        :math:`\\frac{p}{R T}`

    """
    return p / (R_U * T)


@njit(nb.f8[:](nb.f8, nb.f8), fastmath=NUMBA_FAST_MATH, cache=True)
def grad_ideal_rho(p: float, T: float) -> np.ndarray:
    """Gradient of :func:`rho` with respect to pressure and temperature.

    Parameters:
        p: Pressure.
        T: Temperature.

    Returns:
        A 1D array of shape ``(2,)`` containing the derivatives.

    """
    RT = R_U * T
    return np.array((1.0 / RT, -p / (RT * T)))


@njit(nb.f8(nb.f8, nb.f8), fastmath=NUMBA_FAST_MATH, cache=True)
def ideal_v(p: float, T: float) -> float:
    """Specific volume of an ideal gas.

    Parameters:
        p: Pressure.
        T: Temperature.

    Returns:
        :math:`\\frac{R T}{p}`

    """
    return R_U * T / p


@njit(nb.f8[:](nb.f8, nb.f8), fastmath=NUMBA_FAST_MATH, cache=True)
def grad_ideal_v(p: float, T: float) -> np.ndarray:
    """Gradient of :func:`ideal_v` with respect to pressure and temperature.

    Parameters:
        p: Pressure.
        T: Temperature.

    Returns:
        A 1D array of shape ``(2,)`` containing the derivatives.

    """
    return np.array((-R_U * T / (p * p), R_U / p))


class IdealFluid:
    def __init__(
        self,
        name: str,
        /,
        h: Optional[IdealProperty_T] = None,
        dhdT: Optional[IdealProperty_T] = None,
        u: Optional[IdealProperty_T] = None,
        dudT: Optional[IdealProperty_T] = None,
    ):
        self.name: str = str(name)
        """Given name for the ideal fluid. Used only for user support at runtime."""

        if h is None and u is None:
            raise ValueError("Require at least one: ideal enthalpy or internal energy.")
        if h is not None and dhdT is None:
            raise ValueError("Require dhdT if h is given.")
        if u is not None and dudT is None:
            raise ValueError("Require dudT if u is given.")

        self.funcs_raw: dict[
            Literal["u", "h", "dh", "du"], IdealProperty_T | IdealProperty_pT | None
        ] = {"h": h, "u": u, "dh": dhdT, "du": dudT}
        """Contains the functions passed at instantiation."""

        self.funcs: dict[
            Literal["u", "h", "dh", "du", "rho", "drho", "v", "dv"], IdealProperty
        ] = {}
        """Contains compiled versions of :attr:`funcs_raw` and the functions for
        ideal density and specific volume."""

        self.is_compiled: bool = False
        """Flag indicating if already compiled."""

    def compile(self) -> None:
        """Compiles the raw functions passed at instantiation and stores them in
        :attr:`funcs`."""

        if self.is_compiled:
            return

        h_c: IdealProperty_T
        u_c: IdealProperty_T
        dh_c: IdealProperty_T
        du_c: IdealProperty_T

        if self.funcs_raw["u"] is not None and self.funcs_raw["du"] is not None:
            logger.info("Compiling ideal u and h(u)..")

            u_c = _IDP_T_COMPILER(self.funcs_raw["u"])
            du_c = _IDP_T_COMPILER(self.funcs_raw["du"])

            @_IDP_T_COMPILER
            def h_c(T: float) -> float:
                return u_c(T) + R_U * T

            @_IDP_T_COMPILER
            def dh_c(T: float) -> float:
                return du_c(T) + R_U

        elif self.funcs_raw["h"] is not None and self.funcs_raw["dh"] is not None:
            logger.info("Compiling ideal h and u(h)..")

            h_c = _IDP_T_COMPILER(self.funcs_raw["h"])
            dh_c = _IDP_T_COMPILER(self.funcs_raw["dh"])

            @_IDP_T_COMPILER
            def u_c(T: float) -> float:
                return h_c(T) - R_U * T

            @_IDP_T_COMPILER
            def du_c(T: float) -> float:
                return dh_c(T) - R_U

        else:
            raise RuntimeError("Lost references to raw functions u/h.")

        self.funcs = {
            "u": u_c,
            "du": du_c,
            "h": h_c,
            "dh": dh_c,
            "rho": ideal_rho,
            "drho": grad_ideal_rho,
            "v": ideal_v,
            "dv": grad_ideal_v,
        }
        self.is_compiled = True
