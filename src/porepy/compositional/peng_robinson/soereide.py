"""Soereide extension of the Peng-Robinson EoS for H2O, CO2, H2S, N2 mixtures containing
NaCl and forming a brine.

One of the components passed to :class:`PengRobinsonSoereideCompiler` must be named
``'brine'`` and it will be treated like water (see :class:`NaClBrine`).
The other supported components are optional.

References:
    [1] Ingolf Søreide, Curtis H. Whitson,
        Peng-Robinson predictions for hydrocarbons, CO2, N2, and H2 S with pure water
        and NaCI brine,
        Fluid Phase Equilibria,
        Volume 77,
        1992,
        https://doi.org/10.1016/0378-3812(92)85105-H

Note:
    The modifications herein for water with salt hold for temperature ranges 15-325 deg
    Celsius, and a salt molality up to 8.

"""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Literal, Optional, Sequence

import numba as nb
import numpy as np
import sympy as sp

import porepy as pp

from .._core import NUMBA_FAST_MATH
from ..base import Compound
from ..materials import FluidComponent
from . import eos
from .compressibility_factor import (
    get_compressibility_factor,
    get_compressibility_factor_derivatives,
)
from .utils import thd_function_type

__all__ = [
    "NaClBrine",
    "SymbolicPengRobinsonSoereide",
    "CompiledPengRobinsonSoereide",
]


class NaClBrine(FluidComponent, Compound):
    """(Fluid-) compound consiting of water and NaCl as an active tracer.

    Uses :func:`~porepy.compositional.load_fluid_constants` to obtain parameters for the
    2 chemical species.
    """

    def __init__(self):
        h2o, nacl = pp.compositional.load_fluid_constants(["H2O", "NaCl"], "chemicals")
        h2o_vals = asdict(h2o)
        h2o_vals["name"] = "brine"
        del h2o_vals["constants_in_SI"]
        del h2o_vals["_initialized"]
        super().__init__(**h2o_vals)

        self.active_tracers = [nacl]


class SymbolicPengRobinsonSoereide(eos.SymbolicPengRobinson):
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


class CompiledPengRobinsonSoereide(eos.CompiledPengRobinson):
    """Extension of the compiled PR EoS which expects the salinity as a parameter
    for the preargument functions.

    Does not take the ``bip_matrix`` argument, since BIPs are customized in this
    extension.

    The parameter array for the pre-argument function can have up to 4 entries
    (see also :attr:`params`):

    1. ``'salinity'``: Molality of NaCl in the brine.
    2. ``'smoothing_multiphase'`` : Portion of 2-phase region used for smoothing roots
       near phase borders
    3. ``'eps'``: Numerical tolerance to determine zero (root case computation).

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

        self.symbolic: SymbolicPengRobinsonSoereide = SymbolicPengRobinsonSoereide(
            components, ideal_enthalpies
        )

        # If salinity is not provided, set default value to zero.
        if params is not None:
            s = params.get("salinity", 0.0)
        else:
            s = 0.0

        self.params["salinity"] = float(s)
        """The EoS has an additional parameter ``'salinity'`` which represents the
        molality of NaCl in the brine. Defaults to zero."""

    def _get_cohesion(self) -> eos.ScalarFunction:
        """Cohesion takes molal salinity as last argument."""
        return nb.njit(nb.f8(nb.f8, nb.f8, nb.f8[:], nb.f8))(self.symbolic.A_func)

    def _get_cohesion_derivatives(self) -> eos.VectorFunction:
        """Cohesion derivatives take molal salinity as last argument."""
        df = nb.njit(self.symbolic.grad_pTx_A_func, fastmath=NUMBA_FAST_MATH)

        @nb.njit(nb.f8[:](nb.f8, nb.f8, nb.f8[:], nb.f8), fastmath=NUMBA_FAST_MATH)
        def inner(p_, T_, X_, c_):
            return np.array(df(p_, T_, X_, c_), dtype=np.float64)

        return inner

    def get_prearg_for_values(self) -> eos.VectorFunction:
        """Modified pre-argument for values expecting molal salinity as the first
        element in the parameters array argument."""
        A_c = self._cfuncs["A"]
        B_c = self._cfuncs["B"]

        eps = self.params["eps"]
        s_m = self.params["smoothing_multiphase"]
        sal = self.params["salinity"]

        @nb.njit(nb.f8[:](nb.i1, nb.f8, nb.f8, nb.f8[:], nb.f8[:]))
        def prearg_val_c(
            phasetype: int, p: float, T: float, xn: np.ndarray, params: np.ndarray
        ) -> np.ndarray:
            prearg = np.empty((3,), dtype=np.float64)

            s_m_ = s_m
            eps_ = eps
            sal_ = sal
            if params.size >= 1:
                sal_ = params[0]
            if params.size >= 2:
                s_m_ = params[1]
            if params.size >= 3:
                eps_ = params[2]

            if phasetype == 1:
                gaslike = True
            elif phasetype == 0:
                gaslike = False
            else:
                raise NotImplementedError(f"Unsupported phase type: {phasetype}")

            A = A_c(p, T, xn, sal_)
            B = B_c(p, T, xn)

            prearg[0] = A_c(p, T, xn)
            prearg[1] = B_c(p, T, xn)
            prearg[2] = get_compressibility_factor(A, B, gaslike, eps_, s_m_)

            return prearg

        return prearg_val_c

    def get_prearg_for_derivatives(self) -> eos.VectorFunction:
        """Modified pre-argument for derivatives expecting molal salinity as the first
        element in the parameters array argument."""
        A_c = self._cfuncs["A"]
        B_c = self._cfuncs["B"]
        dA_c = self._cfuncs["dA"]
        dB_c = self._cfuncs["dB"]
        # number of derivatives for A, B, Z (p, T, and per component fraction)
        d = 2 + self._nc

        eps = self.params["eps"]
        s_m = self.params["smoothing_multiphase"]
        sal = self.params["salinity"]

        @nb.njit(nb.f8[:](nb.i1, nb.f8, nb.f8, nb.f8[:], nb.f8[:]))
        def prearg_jac_c(
            phasetype: int, p: float, T: float, xn: np.ndarray, params: np.ndarray
        ) -> np.ndarray:
            # the pre-arg for the jacobian contains the derivatives of A, B, Z
            # w.r.t. p, T, and fractions.
            prearg = np.empty((3 * d,), dtype=np.float64)

            s_m_ = s_m
            eps_ = eps
            sal_ = sal
            if params.size >= 1:
                sal_ = params[0]
            if params.size >= 2:
                s_m_ = params[1]
            if params.size >= 4:
                eps_ = params[2]

            if phasetype == 1:
                gaslike = True
            elif phasetype == 0:
                gaslike = False
            else:
                raise NotImplementedError(f"Unsupported phase type: {phasetype}")

            A = A_c(p, T, xn, sal_)
            B = B_c(p, T, xn)

            dA = dA_c(p, T, xn)
            dB = dB_c(p, T, xn)
            dZ_ = get_compressibility_factor_derivatives(A, B, gaslike, eps_, s_m_)
            dZ = dZ_[0] * dA + dZ_[1] * dB

            prearg[0:d] = dA
            prearg[d : 2 * d] = dB
            prearg[2 * d : 3 * d] = dZ

            return prearg

        return prearg_jac_c
