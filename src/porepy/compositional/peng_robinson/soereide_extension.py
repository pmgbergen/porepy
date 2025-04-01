"""Soereide extension of the Peng-Robinson EoS for H2O, CO2, H2S, N2 mixtures containing
NaCl and forming a brine.

One of the components passed to :class:`PengRobinsonSoereideCompiler` must be named
``'brine'`` and it will be treated like water.
The other supported components are optional.

Note:
    The modifications herein for water with salt hold for temperature ranges 15-325 deg
    Celsius, and a salt molality up to 8.

"""

from __future__ import annotations

from typing import Callable, Literal, Optional, Sequence

import numpy as np
import sympy as sp

import porepy as pp

from . import eos, eos_symbolic
from .utils import thd_function_type

__all__ = [
    "PengRobinsonSoereideSymbolic",
    "PengRobinsonSoereideCompiler",
]


class PengRobinsonSoereideSymbolic(eos_symbolic.PengRobinsonSymbolic):
    """Extension of the symbolic PR EoS to account for salinity in the binary
    interaction parameters and cohesion.

    Supports only fluid components defined by :attr:`SUPPORTED_FLUID_COMPONENTS`.

    Must have at least the component ``'brine'``, which represents water with NaCl.

    """

    SUPPORTED_FLUID_COMPONENTS: tuple[Literal["brine", "CO2", "H2S", "N2"], ...] = (
        "brine",
        "CO2",
        "H2S",
        "N2",
    )
    """Names of fluid components supported by this EoS extension.

    The component named ``'brine'`` is treated as ``'H2O'``.

    """

    c_s: sp.Symbol = sp.Symbol("c_NaCl")
    """Symbolic representation of the salt molality in the mixture."""

    def __init__(
        self,
        components: Sequence[pp.FluidComponent],
        ideal_enthalpies: Sequence[thd_function_type],
    ) -> None:
        # Check of assumptions.
        brine: Optional[pp.FluidComponent] = None
        for comp in components:
            if comp.name not in self.SUPPORTED_FLUID_COMPONENTS:
                raise ValueError(
                    f"Unsupported component {comp.name}. Is 'brine' defined?"
                )
            if comp.name == "brine":
                brine = comp

        if brine is None:
            raise ValueError("Soereide extension requires fluid component 'brine'.")

        # Create special BIPs depending on salinity and temperature.
        nc = len(components)
        bip_matrix = np.zeros((nc, nc))
        for i in range(nc):
            ci = components[i]
            for j in range(i + 1, nc):
                cj = components[j]
                if ci.name == "brine":
                    bip = self._bip_map[cj.name](cj.critical_temperature)
                elif cj.name == "brine":
                    bip = self._bip_map[ci.name](ci.critical_temperature)
                else:
                    continue
                bip_matrix[i][j] = bip

        # BIP matrix is symmetric.
        bip_matrix = bip_matrix + bip_matrix.T

        super().__init__(components, ideal_enthalpies, bip_matrix)

        T_r = self.T_s / brine.critical_temperature
        self._alpha: sp.Expr = (
            1
            + 0.453 * (1 - T_r * (1 - 0.0103 * self.c_s**1.1))
            + 0.0034 * (T_r**-3 - 1)
        )
        """Cohesion correction for brine component depending on salinity."""

        self._brine_index = list(components).index(brine)
        """Index of brine component."""

    @property
    def _bip_map(self) -> dict[str, Callable[[float], sp.Expr]]:
        """Utility property returning the callable BIP functions for supported fluid
        components."""
        return {
            "CO2": self.bip_co2,
            "H2S": self.bip_h2s,
            "N2": self.bip_n2,
        }

    def bip_co2(self, T_crit: float) -> sp.Expr:
        """Returns the temperature and salinity-dependent BIP between CO2 and brine,
        given the critical temperature of CO2."""
        T_r = self.T_s / T_crit
        return (
            T_r * 0.23580 * (1 + 0.17837 * self.c_s**0.979)
            - 21.2566 * sp.exp(-6.7222 * T_r - self.c_s)
            - 0.31092 * (1 + 0.15587 * self.c_s**0.7505)
        )

    def bip_h2s(self, T_crit: float) -> sp.Expr:
        """Returns the temperature and salinity-dependent BIP between H2S and brine,
        given the critical temperature of H2S."""
        return -0.20441 + 0.23426 * self.T_s / T_crit

    def bip_n2(self, T_crit: float) -> sp.Expr:
        """Returns the temperature and salinity-dependent BIP between N2 and brine,
        given the critical temperature of N2."""
        return self.T_s / T_crit * 0.44338 * (
            1 + 0.08126 * self.c_s**0.75
        ) - 1.70235 * (1 + 0.25587 * self.c_s**0.75)

    @property
    def alphas(self) -> list[sp.Expr]:
        """Overloads the parent method to insert the salinity-dependent :math:`\\alpha`
        for the brine component."""
        # alphas: list[sp.Expr] = eos_symbolic.PengRobinsonSymbolic.alphas.fget(self)
        alphas = super().alphas
        alphas[self._brine_index] = self._alpha
        return alphas

    @property
    def A_func(self) -> Callable[[float, float, np.ndarray], float]:
        """The cohesion of the extension depends also on molal salinity, which is
        appended as the last argument after pressure, temperature and component
        fractions."""
        arg = (self.p_s, self.T_s, self.x_s, self.c_s)
        return sp.lambdify(arg, self.A)

    @property
    def grad_pTx_A_func(self) -> Callable[[float, float, np.ndarray], list[float]]:
        """Lambdified expression :meth:`grad_pTx_A` returning a list of floats of length
        ``2 + num_comp``, representing the derivatives w.r.t. pressure, temperature and
        component fractions.

        Like :meth:`A`, molal salinity is added as an argument, but the respective
        derivative is not!

        """
        arg = (self.p_s, self.T_s, self.x_s, self.c_s)
        return sp.lambdify(arg, self.grad_pTx_A)


class PengRobinsonSoereideCompiler(eos.PengRobinsonCompiler):
    """Extension of the compiled PR EoS which expects the salinity as a parameter
    for the preargument functions.

    Does not take the ``bip_matrix`` argument, since BIPs are customized in this
    extension.

    """

    def __init__(
        self,
        components: Sequence[pp.FluidComponent],
        ideal_enthalpies: Sequence[thd_function_type],
        params: Optional[dict[str, float]] = None,
    ) -> None:
        # Dummy BIPs for super call.
        nc = len(components)
        super().__init__(components, ideal_enthalpies, np.zeros((nc, nc)), params)

        self.symbolic: PengRobinsonSoereideSymbolic = PengRobinsonSoereideSymbolic(
            components, ideal_enthalpies
        )
