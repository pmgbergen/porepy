"""Module containing functionality representing ideal properties of a fluid, as well
as examples for H2O, CO2, H2S and N2.

Heat capacity coefficients for the provided examples is taken from below reference.

References:
    [1] `de Nevers (2012)
    <https://onlinelibrary.wiley.com/doi/book/10.1002/9781118135341>`_ .

"""

from __future__ import annotations

import logging
from typing import Callable, Literal, Optional, TypeAlias

import numba as nb
import numpy as np

from porepy.compositional._core import H_REF, NUMBA_FAST_MATH, R_U_MOL, T_REF, njit

__all__ = [
    "IdealProperty_T",
    "IdealProperty_pT",
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


IDP_T_COMPILER = njit(nb.f8(nb.f8))
"""Compiler for ideal properties depending only on Temperature."""


IDP_pT_COMPILER = njit(nb.f8(nb.f8, nb.f8))
"""Compiler for ideal properties depending on both pressure and temperature."""


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
            Literal["u", "h", "dhdT", "dudT"], IdealProperty_T | None
        ] = {"h": h, "u": u, "dhdT": dhdT, "dudT": dudT}
        """Contains the functions passed at instantiation."""

        self.funcs: dict[Literal["u", "h", "dhdT", "dudT"], IdealProperty_T] = {}
        """Contains compiled versions of :attr:`funcs_raw`"""

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

        if self.funcs_raw["u"] is not None and self.funcs_raw["dudT"] is not None:
            logger.info("Compiling ideal u and h(u)..")

            u_c = IDP_T_COMPILER(self.funcs_raw["u"])
            du_c = IDP_T_COMPILER(self.funcs_raw["dudT"])

            @IDP_T_COMPILER
            def h_c(T: float) -> float:
                return u_c(T) + R_U_MOL * T

            @IDP_T_COMPILER
            def dh_c(T: float) -> float:
                return du_c(T) + R_U_MOL

        elif self.funcs_raw["h"] is not None and self.funcs_raw["dhdT"] is not None:
            logger.info("Compiling ideal h and u(h)..")

            h_c = IDP_T_COMPILER(self.funcs_raw["h"])
            dh_c = IDP_T_COMPILER(self.funcs_raw["dhdT"])

            @IDP_T_COMPILER
            def u_c(T: float) -> float:
                return h_c(T) - R_U_MOL * T

            @IDP_T_COMPILER
            def du_c(T: float) -> float:
                return dh_c(T) - R_U_MOL

        else:
            raise RuntimeError("Lost references to raw functions u/h.")

        self.funcs = {
            "u": u_c,
            "dudT": du_c,
            "h": h_c,
            "dhdT": dh_c,
        }
        self.is_compiled = True
