"""Module containing various data structures to store thermodynamic state values for
fluid mixtures.

Note:
    State data structures are required as a contract for which thermodynamic properties
    are required to describe a fluid in PorePy's flow & transport problems.

    The compositional framework does not rely purely on :mod:`porepy.numerics.ad`, but
    has multiple interfaces to obtaining values.

    1. Flash calculations: Must be fast, efficient and parallelized, hence no AD but
       generic interfaces and data structures.
    2. OBL: Operator-based linearizations introduces interpolated data into the
       framework. To safely and generically broadcast the data into the Jacobian and
       residual, these data structures are used.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, cast

import numpy as np

from ._core import PhysicalState
from .utils import (
    chainrule_fractional_derivatives,
    compute_saturations,
    normalize_rows,
    safe_sum,
)

__all__ = [
    "ExtensiveProperties",
    "IntensiveProperties",
    "PhaseProperties",
    "FluidProperties",
    "initialize_fluid_properties",
]


@dataclass
class IntensiveProperties:
    """Dataclass for storing intensive thermodynamic properties of a fluid mixture.

    Storage is intended to be in array format for usage in flow and transport, where
    the vectorized format reflects the degrees of freedom on a grid.

    """

    p: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """Pressure values. As of now, an equal pressure across all phases is assumed at
    equilibrium."""

    T: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """Temperature values. As of now, an equal temperatue across all phases is assumed
    at equilibrium."""

    z: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    """Overall molar fractions (feed fractions) per component, stored in a 2D array
    with row-wise values per component."""


@dataclass
class ExtensiveProperties:
    """Dataclass for storing extensive thermodynamic properties
    (analogous to :class:`IntensiveProperties`).

    As of now, the extensive state does not encompass all of the physical quantities,
    but those which are relevant for the flow & transport model in PorePy.

    """

    h: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """Specific enthalpy."""

    rho: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """Density."""

    @property
    def v(self) -> np.ndarray:
        """Specific volume as the reciprocal of :attr:`rho`.

        Returns zeros, where :attr:`rho` is zero.

        """
        v = np.zeros_like(self.rho)
        # special treatment for zero values to avoid division-by zero errors
        idx = self.rho > 0.0
        v[idx] = 1.0 / self.rho[idx]
        return v


@dataclass
class PhaseProperties(ExtensiveProperties):
    """An extended state description for physical phases, including derivatives
    of extensive properties and properties which are not state functions such
    as viscosity and thermal conductivity.

    Derivatives are denoted with a ``d*`` and the derivative values are stored
    along the first axis of a 2D array (i.e. each row represents the derivative with
    respect to 1 dependency)

    The state of a phase is additionally characterized by an integer representing the
    phase type, and the values of fractions per component.

    """

    state: PhysicalState = PhysicalState.liquid
    """Physical state of the phase. Defaults to liquid-like."""

    x: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    """Fractions for each component in a phase, stored row-wise per component
    in a 2D array.

    Fractions of components in a phase are relative to the fraction of a phase.

    """

    phis: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    """Fugacity coefficients per component in this phase, stored row-wise per component
    in a 2D array."""

    dh: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    """Derivatives of the specific enthalpy."""

    drho: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    """Derivatives of the density."""

    dphis: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    """Derivatives of fugacity coefficients.

    This is a 3D array!

    The first axis is associated with components in this phase and the second axis with
    the derivatives. The third axis is for the values.

    """

    mu: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """Dynamic viscosity."""

    dmu: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    """Derivatives of the dynamic viscosity."""

    kappa: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """Thermal conductivity."""

    dkappa: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    """Derivatives of the thermal conductivity."""

    @property
    def dv(self) -> np.ndarray:
        """Derivatives of the specific volume, expressed as the reciprocal of
        density.

        The chainrule is applied to compute ``dv`` from ``d(1/rho)``.

        The derivatives are set to zero, where :attr:`rho` is zero.

        """
        # Treatment to avoid division by zero errors
        idx = self.rho > 0.0
        outer = np.zeros_like(self.v)
        outer[idx] = -1 / self.rho[idx] ** 2
        return np.array([outer * d for d in self.drho])

    @property
    def x_normalized(self) -> np.ndarray:
        """Normalized values of fractions found in :attr:`x`.
        The normalization is performed along the 2nd axis."""
        return normalize_rows(self.x.T).T

    @property
    def drho_ext(self) -> np.ndarray:
        """Assuming the derivatives in :attr:`drho` are with respect to (physical)
        partial fractions (last :attr:`x` ``.shape[0]`` rows), this property returns the
        derivatives w.r.t. extended fractions in the unified setting.

        The extended fractions are assumed to be stored in :attr:`x`,
        whereas the partial fractions are given by :attr:`x_normalized`.

        For more information, see
        :func:`~porepy.compositional.utils.chainrule_fractional_derivatives`.

        """
        return self._for_extended_fractions(self.drho)

    @property
    def dv_ext(self) -> np.ndarray:
        """See :meth:`drho_ext` for more information."""
        return self._for_extended_fractions(self.dv)

    @property
    def dh_ext(self) -> np.ndarray:
        """See :meth:`drho_ext` for more information."""
        return self._for_extended_fractions(self.dh)

    @property
    def dmu_ext(self) -> np.ndarray:
        """See :meth:`drho_ext` for more information."""
        return self._for_extended_fractions(self.dmu)

    @property
    def dkappa_ext(self) -> np.ndarray:
        """See :meth:`drho_ext` for more information."""
        return self._for_extended_fractions(self.dkappa)

    @property
    def dphis_ext(self) -> np.ndarray:
        """See :meth:`drho_ext` for more information."""
        return np.array([self._for_extended_fractions(dphi) for dphi in self.dphis])

    def _for_extended_fractions(self, df_dx: np.ndarray) -> np.ndarray:
        """Helper method to apply the chainrule to fractional derivatives, switching
        from partial fraction to extended fractions in the unified formulation.

        See Also:
            :func:`~porepy.compositional.utils.chainrule_fractional_derivatives`

        """
        # NOTE development & debug:
        # A switch to Quasi-Newton can be handled here by omitting the chainrule.
        # Increased robustness (challenging EoS extension) and the Jacobian containing
        # the physical derivatives are the consequences
        return chainrule_fractional_derivatives(df_dx, self.x)


@dataclass
class FluidProperties(IntensiveProperties, ExtensiveProperties):
    """Nested dataclass characterizing the thermodynamic state of a
    multiphase multicomponent fluid.

    This is a collection of intensive and extensive states of the fluid,
    as well as a collection of :class:`PhaseProperties` isntances characterizing individual
    phases.

    Note:
        The first phase is always assumed to be the reference phase
        (see :class:`~porepy.compositional.base.FluidMixture`).
        I.e., its fractional values are usually dependent by unity of fractions.

    Contrary to :class:`PhaseProperties`, this dataclass does not support derivatives of
    extensive properties on a mixture-level.
    Since the derivatives w.r.t. to phase fractions or saturations are trivially
    the respective property of the phase, this can be done easily by the user without
    having the same values stored at two different places.

    """

    y: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    """Phase fractions for each phase in :attr:`phases`, stored row-wise."""

    sat: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    """Saturation for each phase in :attr:`phases`, stored row-wise."""

    phases: Sequence[PhaseProperties] = field(default_factory=lambda: list())
    """A collection of phase state descriptions per phase in the fluid mixture."""

    def evaluate_saturations(self, eps: float = 1e-10) -> None:
        """Calculates the values for volumetric phase fractions from stored
        :attr:`y` and density values, and stores them in :attr:`sat`.

        See Also:
            :func:`~porepy.compositional.utils.compute_saturations`

        Parameters:
            eps: ``default=1e-10``

                Tolerance to detect saturated phases.

        """

        if not isinstance(self.y, np.ndarray):
            y = np.array(self.y)
        else:
            y = self.y

        rho = np.array([phase.rho for phase in self.phases])
        self.sat = compute_saturations(y, rho, eps)

    def evaluate_extensive_state(self) -> None:
        """Evaluates the extensive mixture properties based on the currently stored
        phase properties, phase fractions and saturations.

        1. :attr:`h` of the mixture (weights :attr:`y`)
        2. :attr:`rho` of the mixture (weights :attr:`sat`)

        Each one is computed by summing corresponding phase properties weighed with
        respective weights.

        """

        self.h = safe_sum([y * state.h for y, state in zip(self.y, self.phases)])
        rho = cast(
            np.ndarray,
            safe_sum([s * state.rho for s, state in zip(self.sat, self.phases)]),
        )
        self.rho = rho


def initialize_fluid_properties(
    n: int,
    ncomp: int | np.ndarray,
    nphase: int,
    phase_states: Optional[Sequence[PhysicalState]] = None,
    with_derivatives: bool = False,
) -> FluidProperties:
    """Creates a fluid property structure with filled with zero values of defined size.

    Parameters:
        n: Number of values per thermodynamic quantity.
        ncomp: Number of components. Either as a number or an array with numbers per
            phase.
        nphase: Number of phases
        phase_states: ``default=None``

            Physical states per phase. If None, all phases are assigned a liquid state.
        with_derivatives: ``default=False``.

            If True, the derivatives are also initialized with zero values, otherwise
            they are left empty.

    """
    state = FluidProperties()
    state.p = np.zeros(n)
    state.T = np.zeros(n)
    state.h = np.zeros(n)
    state.rho = np.zeros(n)
    state.y = np.zeros((nphase, n))
    state.sat = np.zeros((nphase, n))

    if phase_states is None:
        phase_states = [PhysicalState.liquid] * nphase
    if isinstance(ncomp, int):
        ncomp = np.ones(nphase, dtype=int) * ncomp
    else:
        assert ncomp.shape == (nphase,), "Need number of components for every phase."

    # to cover all components, independent of their modelling in phases
    state.z = np.zeros((ncomp.max(), n))

    state.phases = []
    for j in range(nphase):
        phase_state = PhaseProperties(
            h=np.zeros(n),
            rho=np.zeros(n),
            state=phase_states[j],
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

    return state
