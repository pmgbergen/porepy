"""Module containing various data structures to store thermodynamic state values for
fluid mixtures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, cast

import numpy as np

from .composite_utils import safe_sum
from .utils_c import compute_saturations, extend_fractional_derivatives

__all__ = [
    "ExtensiveState",
    "IntensiveState",
    "PhaseState",
    "FluidState",
    "initialize_fluid_state",
]


@dataclass
class IntensiveState:
    """Dataclass for storing intensive thermodynamic properties of a fluid mixture.

    Storage is intended to be in array format for usage in flow and transport, where
    the vectorized format reflects the degrees of freedom on a grid.

    Intensive properties include:

    - pressure [Pa]
    - temperature [K]
    - overall molar fraction per fluid component [-]

    """

    p: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """Pressure values. As of now, an equal pressure across all phases is assumed at
    equilibrium."""

    T: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """Temperature values. As of now, an equal temperatue across all phases is assumed
    at equilibrium."""

    z: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    """Overall molar fractions (feed fractions) per component, stored in a 2D array
    with row-wise values per component.
    The first fraction is always the feed fraction of the reference component.."""


@dataclass
class ExtensiveState:
    """Dataclass for storing extensive thermodynamic properties
    (analogous to :class:`IntensiveState`).

    As of now, the extensive state does not encompass all of the physical quantities,
    but those which are relevant for the flow and transport model in PorePy.

    These include:

    - specific molar enthalpy [J / mol]
    - specific molar density [mol / m^3]
    - specific molar volume [m^3 / mol]

    """

    h: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """Specific molar enthalpy."""

    rho: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """Density."""

    @property
    def v(self) -> np.ndarray:
        """Specific molar volume as the reciprocal of :attr:`rho`."""
        v = np.zeros_like(self.rho)
        # special treatment for zero values to avoid division-by zero errors
        idx = self.rho > 0.0
        v[idx] = 1.0 / self.rho[idx]
        return v


@dataclass
class PhaseState(ExtensiveState):
    """An extended state description for physical phases, including derivatives
    with respect to pressure temperature and fractions of component in this phase.

    The derivatives are always in that order:

    1. pressure,
    2. temperature,
    3. ``dx`` for each component

    The state of a phase is additionally characterized by an integer representing the
    phase type, and the values of fractions per component.

    """

    phasetype: int = 0
    """Type of the phase. Defaults to 0 (liquid)."""

    x: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    """Molar fractions for each component in a phase, stored row-wise per component
    in a 2D array.

    Fractions of components in a phase are relative to the moles in a phase.

    The first one is assumed to belong to the reference component.

    """

    phis: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    """Fugacity coefficients per component in this phase, stored row-wise per component
    in a 2D array."""

    dh: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    """Derivatives of the specific molar enthalpy with respect to pressure, temperature
    and each ``x`` in :attr:`x`.

    The derivatives are stored row-wise in a 2D array.
    The length of ``dh`` is ``2 + len(x)``.

    """

    drho: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    """Derivatives of the density with respect to pressure, temperature
    and each ``x`` in :attr:`x`.

    The derivatives are stored row-wise in a 2D array.
    The length of ``drho`` is ``2 + len(x)``.

    """

    dphis: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    """Derivatives of fugacity coefficients w.r.t. pressure, temperature and each
    ``x`` in :attr:`x`.

    Derivatives are stored in a 3D array, where the first axis is associated with
    components in this phase and the second axis with the derivatives.
    The third axis is for the values.

    """

    mu: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """Dynamic molar viscosity."""

    dmu: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    """Derivatives of the dynamic molar viscosity w.r.t. pressure, temperature and
    each ``x`` in :attr:`x`.

    The derivatives are stored row-wise in a 2D array.
    The length of ``dmu`` is ``2 + len(x)``.

    """

    kappa: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """Thermal conductivity."""

    dkappa: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    """Derivatives of the thermal conductivity w.r.t. pressure, temperature and
    each ``x`` in :attr:`x`.

    The derivatives are stored row-wise in a 2D array.
    The length of ``dkapp`` is ``2 + len(x)``.

    """

    @property
    def dv(self) -> np.ndarray:
        """Derivatives of the specific molar volume, expressed as the reciprocal of
        density.

        The chainrule is applied to compute ``dv`` from ``d(1/rho)``.

        """
        # Treatment to avoid division by zero errors
        idx = self.rho > 0.0
        outer = np.zeros_like(self.v)
        outer[idx] = -1 / self.rho[idx] ** 2
        return np.array([outer * d for d in self.drho])

    @property
    def x_normalized(self) -> np.ndarray:
        """Normalized values of fractions found in :attr:`x`."""
        x_sum = safe_sum(self.x)
        return np.array([x / x_sum for x in self.x])

    @property
    def drho_ext(self) -> np.ndarray:
        """Returning the derivatives of :attr:`rho` with respect to pressure,
        temperature and the extended partial fractions."""
        return self._extend(self.drho)

    @property
    def dv_ext(self) -> np.ndarray:
        """Returning the derivatives of :meth:`v` with respect to pressure,
        temperature and the extended partial fractions."""
        return self._extend(self.dv)

    @property
    def dh_ext(self) -> np.ndarray:
        """Returning the derivatives of :attr:`h` with respect to pressure,
        temperature and the extended partial fractions."""
        return self._extend(self.dh)

    @property
    def dmu_ext(self) -> np.ndarray:
        """Returning the derivatives of :attr:`mu` with respect to pressure,
        temperature and the extended partial fractions."""
        return self._extend(self.dmu)

    @property
    def dkappa_ext(self) -> np.ndarray:
        """Returning the derivatives of :attr:`kappa` with respect to pressure,
        temperature and the extended partial fractions."""
        return self._extend(self.dkappa)

    @property
    def dphis_ext(self) -> np.ndarray:
        """Returning the derivatives of :attr:`phis` with respect to pressure,
        temperature and the extended partial fractions."""
        return np.array([self._extend(dphi) for dphi in self.dphis])

    def _extend(self, df_dx: np.ndarray) -> np.ndarray:
        """Helper method to extend the fractional derivatives.

        Note:
            The extended derivatives are used in the unified CFLE setting.
            But it seems that the model converges even when using the un-extended
            derivatives (Jacobian is not exact).
            Full implications unclear as of now.

        """
        return extend_fractional_derivatives(df_dx, self.x)
        # return df_dx


@dataclass
class FluidState(IntensiveState, ExtensiveState):
    """Nested dataclass characterizing the thermodynamic state of a
    multiphase multicomponent fluid.

    This is a collection of intensive and extensive states of the fluid,
    as well as a collection of :class:`PhaseState` isntances characterizing individual
    phases.

    Important:
        The first phase is always assumed to be the reference phase.
        I.e., its fractional values are usually dependent by unity of fractions.

    The complete fluid state includes additionally:

    - volumetric phase fractions (saturations) [-]
    - molar phase fractions [-]

    Contrary to :class:`PhaseState`, this dataclass does not support derivatives of
    extensive properties on a mixture-level.
    Since the derivatives w.r.t. to molar or volumetric phase fractions are trivially
    the respective property of the phase, this can be done easily by the user without
    having the same values stored at two different places.

    """

    y: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    """Molar phase fractions for each phase in :attr:`phases`."""

    sat: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    """Saturation for each phase in :attr:`phases`."""

    phases: Sequence[PhaseState] = field(default_factory=lambda: list())
    """A collection of phase state descriptions for phases anticipated in the fluid
    mixture."""

    def evaluate_saturations(self, eps: float = 1e-10) -> None:
        """Calculates the values for volumetric phase fractions from stored
        :attr:`y` and density values, and stores them in :attr:`sat`.

        The calculation is performed using the relation
        :math:`y_j \\rho  = s_j \\rho_j`.

        Parameters:
            eps: ``default=1e-10``

                Tolerance to detect saturated phases.

        """

        if not isinstance(self.y, np.ndarray):
            y = np.array(self.y)
        else:
            y = self.y

        rho = np.array([phase.rho for phase in self.phases])
        assert y.shape == rho.shape, "Mismatch in values for fractions and densities."
        self.sat = compute_saturations(y, rho, eps)

    def evaluate_extensive_state(self) -> None:
        """Evaluates the mixture properties based on the currently stored phase
        properties, molar phase fractions and volumetric phase fractions.

        Stores them in the respective attribute found in :class:`ExtensiveState`.

        """

        self.h = safe_sum([y * state.h for y, state in zip(self.y, self.phases)])
        rho = cast(
            np.ndarray,
            safe_sum([s * state.rho for s, state in zip(self.sat, self.phases)]),
        )
        self.rho = rho


def initialize_fluid_state(
    n: int,
    ncomp: int | np.ndarray,
    nphase: int,
    phase_types: Optional[np.ndarray] = None,
    with_derivatives: bool = False,
) -> FluidState:
    """Creates a fluid state with filled with zero values of defined size.

    Parameters:
        n: Number of values per thermodynamic quantity.
        ncomp: Number of components. Either as a number or an array with numbers per
            phase.
        nphase: Number of phases
        phase_types: ``default=None``

            Phase types (integers) per phase. If None, all phases are assigned type 0.
        with_derivatives: ``default=False``.

            If True, the derivatives are also initialized with zero values, otherwise
            they are left empty.

    """
    state = FluidState()
    state.p = np.zeros(n)
    state.T = np.zeros(n)
    state.h = np.zeros(n)
    state.rho = np.zeros(n)
    state.y = np.zeros((nphase, n))
    state.sat = np.zeros((nphase, n))

    if phase_types is None:
        phase_types = np.zeros(nphase, dtype=int)
    if isinstance(ncomp, int):
        ncomp = np.ones(nphase, dtype=int) * ncomp
    else:
        assert ncomp.shape == (nphase,), "Need number of components for every phase."

    # to cover all components, independent of their modelling in phases
    state.z = np.zeros((ncomp.max(), n))

    state.phases = list()
    for j in range(nphase):
        phase_state = PhaseState(
            h=np.zeros(n),
            rho=np.zeros(n),
            phasetype=phase_types[j],
            x=np.zeros((ncomp[j], n)),
            phis=np.zeros((ncomp[j], n)),
            mu=np.zeros(n),
            kappa=np.zeros(n),
        )

        if with_derivatives:
            phase_state.dh = np.zeros((2 + ncomp[j], n))
            phase_state.drho = np.zeros((2 + ncomp[j], n))
            phase_state.dphis = np.zeros((ncomp[j], 2 + ncomp[j], n))
            phase_state.dmu = np.zeros((2 + ncomp[j], n))
            phase_state.dkappa = np.zeros((2 + ncomp[j], n))

        state.phases.append(phase_state)
