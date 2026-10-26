"""Constants of the manufactured-solution convergence test, in SI units."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import sympy as sp


@dataclass(frozen=True)
class HaliteMMSConstants:
    """Immutable container of the constants of the manufactured-solution convergence test.

    Groups: phase and rock properties, solubility law (``chi0``, ``beta_T``,
    ``beta_p``), reference state, geometry, matrix feedback (``phi0``, ``K0``),
    interface law (``K_k``), aperture law (``a0``, ``a_min``, clogging exponent
    ``theta``) and manufactured amplitudes (``xi``, ``P``, ``H1``, ``Z1``, ``S0``, ``S1``).
    """

    rho_liq: float = 1.0
    rho_hal: float = 2.160
    mu_liq: float = 1.0
    cp: float = 1.0
    kappa_liq: float = 8.0e-3
    kappa_hal: float = 1.6e-2
    kappa_r: float = 2.0
    rho_r: float = 2.7
    cp_r: float = 1.0
    chi0: float = 0.2
    beta_T: float = 5.0e-2
    beta_p: float = 5.0e-2
    T0: float = 0.0
    p0: float = 0.0
    h0: float = 1.0
    z0: float = 0.4
    Lx: float = 1.0
    Ly: float = 1.0
    x_frac: float = 0.5
    y_tip_lo: float = 0.25
    y_tip_hi: float = 0.75
    phi0: float = 0.3
    K0: float = 1.0e-2
    K_k: float = 5.0e-5
    ell_target: float = 0.072
    a0: float = 1.02e-3 
    a_res: float = 1.0e-3
    a_min: float = 1.0e-4
    theta: float = 0.2
    xi: float = 1.5
    P: float = 1.0
    H1: float = 1.0
    Z1: float = 0.5
    S0: float = 0.1
    S1: float = 0.2

    @property
    def chi_star_ref(self) -> float:
        """Saturated liquid salt fraction at the reference state ``(p0, h0)``."""
        return self.chi0 + self.beta_T / self.cp * self.h0

    @property
    def two_phase_margin_ref(self) -> float:
        """Two-phase margin ``z0 - chi_star(p0, h0)``; positive inside the liquid + halite region."""
        return self.z0 - self.chi_star_ref

    @property
    def aperture_bounds(self) -> tuple[float, float]:
        """Smallest and largest aperture for halite saturations in ``[S0, S0 + S1]``."""
        a_lo = self.a0 * (1.0 - (self.S0 + self.S1)) ** self.theta
        a_hi = self.a0 * (1.0 - self.S0) ** self.theta
        return a_lo, a_hi

    def _trace_halite_saturation(self) -> float:
        """Halite saturation at the matrix trace of the interface, a constant.

        Returns:
            The trace halite saturation.

        """
        margin = self.two_phase_margin_ref

        return self.rho_liq * margin / (
            self.rho_hal * (1.0 - self.z0) + self.rho_liq * margin
        )

    @property
    def coupling_length_pressure(self) -> float:
        """Pressure coupling length ``a_l K_h / (2 K_k)`` at ``t = 0``."""
        s_trace = self._trace_halite_saturation()
        K_h = self.K0 * (1.0 - s_trace) ** 2
        return self.aperture_bounds[1] * K_h / (2.0 * self.K_k)

    @property
    def coupling_length_enthalpy(self) -> float:
        """Enthalpy coupling length ``a_l D_h / (2 D_l)`` at ``t = 0``."""
        s_trace = self._trace_halite_saturation()
        phi_h = self.phi0 * (1.0 - s_trace)
        kappa_f_h = (1.0 - s_trace) * self.kappa_liq + s_trace * self.kappa_hal
        D_h = phi_h * kappa_f_h + (1.0 - phi_h) * self.kappa_r
        # Fracture porosity is unity, so the effective conductivity there is the
        # fluid conductivity alone -- which is also used as the
        # normal conductivity on the interface.
        D_l = (1.0 - self.S0) * self.kappa_liq + self.S0 * self.kappa_hal
        return self.aperture_bounds[1] * D_h / (2.0 * D_l)

    def as_float_dict(self) -> dict[str, float]:
        """Stored constants as a name-keyed float mapping.

        Returns:
            Constant name to value.

        """
        return asdict(self)

    def as_sympy_dict(self, symbols: dict[str, sp.Symbol]) -> dict[sp.Symbol, float]:
        """Symbol-keyed substitution mapping for ``sympy.lambdify``.

        Parameters:
            symbols: Constant name to its SymPy symbol.

        Returns:
            Symbol to value.

        Raises:
            KeyError: If ``symbols`` lacks a constant name.

        """
        return {symbols[name]: value for name, value in self.as_float_dict().items()}


MMS_HALITE = HaliteMMSConstants()
"""HaliteMMSConstants: Canonical constants instance."""

_NAMES = list(MMS_HALITE.as_float_dict().keys())

SYMBOLS: dict[str, sp.Symbol] = {name: sp.symbols(name, real=True) for name in _NAMES}
"""dict[str, sympy.Symbol]: One real SymPy symbol per constant name."""

DEFAULT_PARAMS: dict[sp.Symbol, float] = MMS_HALITE.as_sympy_dict(SYMBOLS)
"""dict[sympy.Symbol, float]: Symbol-to-value substitution map."""