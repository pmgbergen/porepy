"""Module containing various data structures to store thermodynamic state values for
fluid mixtures."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pypardiso
import scipy.sparse as sps

from .composite_utils import safe_sum

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

        num_phases = len(self.y)
        densities = [state.rho for state in self.phases]
        assert num_phases == len(
            densities
        ), "Need equal amount of phases and molar phase fractionsfractions"

        num_vals = len(self.y[0])

        s: list[np.ndarray] = [np.zeros(num_vals) for _ in range(num_phases)]

        if num_phases == 1:
            s[0] = np.ones(num_vals)
        else:
            if num_phases == 2:
                # 2-phase saturation evaluation can be done analytically
                rho_1, rho_2 = densities
                y_1, y_2 = self.y

                s_1 = np.zeros_like(y_1)
                s_2 = np.zeros_like(y_2)

                phase_1_saturated = (y_1 > 1 - eps) & (y_1 < 1 + eps)
                phase_2_saturated = (y_2 > 1 - eps) & (y_2 < 1 + eps)
                idx = ~phase_2_saturated
                y2_idx = y_2[idx]
                rho1_idx = rho_1[idx]
                rho2_idx = rho_2[idx]
                s_1[idx] = 1.0 / (1.0 + y2_idx / (1.0 - y2_idx) * rho1_idx / rho2_idx)
                s_1[phase_1_saturated] = 1.0
                s_1[phase_2_saturated] = 0.0

                idx = ~phase_1_saturated
                y1_idx = y_1[idx]
                rho1_idx = rho_1[idx]
                rho2_idx = rho_2[idx]
                s_2[idx] = 1.0 / (1.0 + y1_idx / (1.0 - y1_idx) * rho2_idx / rho1_idx)
                s_2[phase_1_saturated] = 0.0
                s_2[phase_2_saturated] = 1.0

                s[0] = s_1
                s[1] = s_2
            else:
                # More than 2 phases requires the inversion of the matrix given by
                # phase fraction relations
                mats = list()

                # list of indicators per phase, where the phase is fully saturated
                saturated = list()
                # where one phase is saturated, the other vanish
                vanished = [np.zeros(num_vals, dtype=bool) for _ in range(num_phases)]

                for j2 in range(num_phases):
                    # get the DOFS where one phase is fully saturated
                    # TODO check sensitivity of this
                    saturated_j = (self.y[j2] > 1 - eps) & (self.y[j2] < 1 + eps)
                    saturated.append(saturated_j)
                    # store information that other phases vanish at these DOFs
                    for j2 in range(num_phases):
                        if j2 != j2:
                            # where phase j is saturated, phase j2 vanishes
                            # logical OR acts cumulatively
                            vanished[j2] = vanished[j2] | saturated_j

                # stacked indicator which DOFs
                saturated = np.hstack(saturated)
                # staacked indicator which DOFs vanish
                vanished = np.hstack(vanished)
                # all other DOFs are in multiphase regions
                multiphase = ~(saturated | vanished)

                # construct the matrix for saturation flash
                # first loop, per block row (equation per phase)
                for j in range(num_phases):
                    mats_row = list()
                    # second loop, per block column (block per phase per equation)
                    for j2 in range(num_phases):
                        # diagonal values are zero
                        # This matrix is just a placeholder
                        if j == j2:
                            mats.append(sps.diags([np.zeros(num_vals)]))
                        # diagonals of blocks which are not on the main diagonal,
                        # are non-zero
                        else:
                            d = 1 - self.y[j]
                            # to avoid a division by zero error, we set it to one
                            # this is arbitrary, but respective matrix entries are
                            # sliced out since they correspond to cells where one phase
                            # is saturated,
                            # i.e. the respective saturation is 1., the other 0.
                            d[d == 0.0] = 1.0
                            diag = 1.0 + densities[j2] / densities[j] * self.y[j] / d
                            mats_row.append(sps.diags([diag]))

                    # row-block per phase fraction relation
                    mats.append(sps.hstack(mats_row, format="csr"))

                # Stack matrices per equation on each other
                # This matrix corresponds to the vector of stacked saturations per phase
                mat = sps.vstack(mats, format="csr")
                # TODO Matrix has large band width

                # projection matrix to DOFs in multiphase region
                # start with identity in CSR format
                projection: sps.spmatrix = sps.diags([np.ones(len(multiphase))]).tocsr()
                # slice image of canonical projection out of identity
                projection = projection[multiphase]
                projection_transposed = projection.transpose()

                # get sliced system
                rhs = projection * np.ones(num_vals * num_phases)
                mat = projection * mat * projection_transposed

                s = pypardiso.spsolve(mat, rhs)

                # prolongate the values from the multiphase region to global DOFs
                saturations = projection_transposed * s
                # set values where phases are saturated or have vanished
                saturations[saturated] = 1.0
                saturations[vanished] = 0.0

                # distribute results to the saturation variables
                for j in range(num_phases):
                    s[j] = saturations[j * num_vals : (j + 1) * num_vals]

        self.sat = s

    def evaluate_extensive_state(self) -> None:
        """Evaluates the mixture properties based on the currently stored phase
        properties, molar phase fractions and volumetric phase fractions.

        Stores them in the respective attribute found in :class:`ExtensiveState`.

        """

        self.h = safe_sum([y * state.h for y, state in zip(self.y, self.phases)])
        self.v = 1 / safe_sum(
            [s * state.rho for s, state in zip(self.sat, self.phases)]
        )
