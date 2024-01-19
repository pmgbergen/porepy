"""Module containing various data structures to store thermodynamic state values for
fluid mixtures."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .composite_utils import safe_sum
from .utils_c import compute_saturations, compute_saturations_v

__all__ = [
    "ExtensiveState",
    "IntensiveState",
    "PhaseState",
    "FluidState",
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

    p: np.ndarray = np.zeros(1)
    """Pressure values. As of now, an equal pressure across all phases is assumed at
    equilibrium. Default is zero."""

    T: np.ndarray = np.zeros(1)
    """Temperature values. As of now, an equal temperatue across all phases is assumed
    at equilibrium. Default is zero."""

    z: Sequence[np.ndarray] = field(default_factory=lambda: [np.zeros(1)])
    """Overall molar fractions (feed fractions) per component.
    The first fraction is always the feed fraction of the reference component.
    Default is a sequence of zeros."""


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

    h: np.ndarray = np.zeros(1)
    """Specific molar enthalpy. Default is zero."""

    v: np.ndarray = np.ones(1)
    """Specific molar volume. Default is one."""

    @property
    def rho(self) -> np.ndarray:
        """Specific molar density as the reciprocal of :attr:`v`."""
        return 1 / self.v


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

    x: Sequence[np.ndarray] = field(default_factory=lambda: [np.zeros(1)])
    """Extended molar fractions for each component.

    Fractions of components in a phase are relative to the moles in a phase.

    The first one is assumed to belong to the reference component.

    Default is a list of zeros.

    """

    phis: Sequence[np.ndarray] = field(default_factory=lambda: [np.zeros(1)])
    """Fugacity coefficients per component in this phase. Default is a list of zeros."""

    dh: Sequence[np.ndarray] = field(default_factory=lambda: [np.zeros(1)])
    """Derivatives of the specific molar enthalpy with respect to pressure, temperature
    and each ``x`` in :attr:`x`.

    The length of ``dh`` is ``2 + len(x)``.

    Default is a list of zeros.

    """

    dv: Sequence[np.ndarray] = field(default_factory=lambda: [np.zeros(1)])
    """Derivatives of the specific molar volume with respect to pressure, temperature
    and each ``x`` in :attr:`x`.

    The length of ``dv`` is ``2 + len(x)``.

    Default is a list of zeros.

    """

    dphis: Sequence[Sequence[np.ndarray]] = field(
        default_factory=lambda: [np.zeros((1, 1))]
    )
    """Derivatives of fugacity coefficients per component in this phase.

    The elements themselves are sequences, containing derivatives with respect to
    pressure, temperature and fractions.

    Default is a list of zero 2D-arrays."""

    @property
    def drho(self) -> Sequence[np.ndarray]:
        """Derivatives of the specific molar volume, expressed as the reciprocal
        of volume.

        The chainrule is applied to compute ``drho`` from ``d(1/v)``.

        """
        outer = -1 / self.v**2
        return [outer * dv for dv in self.dv]

    @property
    def xn(self) -> Sequence[np.ndarray]:
        """Normalized values of fractions found in :attr:`x`."""
        x_sum = safe_sum(self.x)
        return [x / x_sum for x in self.x]


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

    y: Sequence[np.ndarray] = field(default_factory=lambda: [np.zeros(1)])
    """Molar phase fractions for each phase in :attr:`phases`."""

    sat: Sequence[np.ndarray] = field(default_factory=lambda: [np.zeros(1)])
    """Saturation for each phase in :attr:`phases`."""

    phases: Sequence[PhaseState] = field(default_factory=lambda: [PhaseState()])
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

        if len(y.shape) == 1:
            self.sat = compute_saturations(y, rho, eps).reshape((y.shape[0], 1))
        else:
            self.sat = compute_saturations_v(y, rho, eps)

    def evaluate_extensive_state(self) -> None:
        """Evaluates the mixture properties based on the currently stored phase
        properties, molar phase fractions and volumetric phase fractions.

        Stores them in the respective attribute found in :class:`ExtensiveState`.

        """

        self.h = safe_sum([y * state.h for y, state in zip(self.y, self.phases)])
        self.v = 1 / safe_sum(
            [s * state.rho for s, state in zip(self.sat, self.phases)]
        )
