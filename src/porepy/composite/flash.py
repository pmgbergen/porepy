"""Module containing an abstraction layer for the flash procedure."""

from __future__ import annotations

import abc
import logging
from typing import Any, Literal, Optional, Sequence

import numpy as np

import porepy as pp
import porepy.composite as ppc

from .composite_utils import safe_sum
from .states import FluidState

__all__ = ["Flash"]

logger = logging.getLogger(__name__)


class Flash(abc.ABC):
    """Abstract base class for flash algorithms defining the interface of flash objects.

    The definition of the interface is done mainly as an orientation for compatibility
    with the remainder of PorePy's framework (especially the compositional flow.)


    """

    def __init__(self, mixture: ppc.Mixture) -> None:
        super().__init__()

        ncomp = mixture.num_components
        nphase = mixture.num_phases

        self.npnc: tuple[int, int] = (nphase, ncomp)
        """Number of phases and components present in mixture."""

        self.nc_per_phase: tuple[int, ...] = tuple(
            [phase.num_components for phase in mixture.phases]
        )
        """Number of components modelled in each phase."""

        ref_idx: int = 0
        for comp in mixture.components:
            if comp == mixture.reference_component:
                break
            else:
                ref_idx += 1

        self._ref_component_idx: int = ref_idx
        """Index of reference component. Relevant for when eliminating the mass balance
        of the reference component, or constructing a generic flash argument."""

        ref_idx: int = 0
        for phase in mixture.phases:
            if phase == mixture.reference_phase:
                break
            else:
                ref_idx += 1

        self._ref_phase_idx: int = ref_idx
        """Index of reference phase. Relevant for when eliminating its fraction and
        saturation as an unknown."""

        self.tolerance: float = 1e-8
        """Convergence criterion for the flash algorithm. Defaults to ``1e-8``."""

        self.max_iter: int = 150
        """Maximal number of iterations for the flash algorithms. Defaults to 150."""

        self.last_flash_stats: dict[str, Any] = dict()
        """Dictionary to store some information about the last flash procedure called.

        Can be filled at the end of :meth:`flash` and is printed by
        :meth:`log_last_flash`.

        Examples:

            - ``'type'``: String. Type of the flash (p-T, p-h,...)
            - ``'init_time'``: Float. Real time taken to compute initial guess in
              seconds.
            - ``'minim_time'``: Float. Real time taken to solve the minimization problem
              in seconds.
            - ``'num_flash'``: Int. Number of flash problems solved
              (if vectorized input)
            - ``'num_max_iter'``: Int. Number of flash procedures which reached the
              prescribed number of iterations.
            - ``'num_failure'``: Int. Number of failed flash procedures
              (failure in evaluation of residual or Jacobian).
            - ``'num_diverged'``: Int. Number of flash procedures which diverged.

        """

    def log_last_flash(self):
        """Prints statistics found in :attr:`last_flash_stats` in the console."""
        msg = "Last flash overview:\n"
        for k, v in self.last_flash_stats.items():
            msg += f"---\t{k}: {v}\n"
        msg += "\n"
        print(msg)

    def parse_flash_input(
        self,
        z: Sequence[np.ndarray | pp.number],
        p: Optional[np.ndarray | pp.number] = None,
        T: Optional[np.ndarray | pp.number] = None,
        h: Optional[np.ndarray | pp.number] = None,
        v: Optional[np.ndarray | pp.number] = None,
        initial_state: Optional[FluidState] = None,
    ) -> tuple[
        FluidState,
        Literal["p-T", "p-h", "v-T", "v-h"],
        int,
        int,
    ]:
        """Helper method to parse the input and construct a provisorical fluid state
        with uniform input (numpy arrays of same size).

        The parameters are described in :meth:`flash`.

        Determines also the equilibrium definition, the size of the (local) flash
        problem and the number of values for vectorized input.

        Hint:
            The returned fluid state structure can be used by :meth:`flash` to fill it
            up with results and return it to the caller of the flash.

        Raises:
            AssertionError: If an insufficient amount of any fraction set has been
                passed.
            ValueError: If any feed fraction violates the strict bound ``(0,1)``.
            AssertionError: If the sum of any fraction set violates the unity
                cosntraint.
            TypeError: If the parser fails to broadcast the state input into a uniform
                format (numpy arrays of equal length).

        Returns:
            A tuple containing

            1. The fluid state consisting of feed fractions and equilibrium state.
               If ``initial_state`` is not none, it includes the values.
            2. A string denoting the equilibrium type.
            3. A number denoting the size of the locall equilibrium problem (dofs)
            4. A number denoting the size of the input after uniformization.

        """

        nphase, ncomp = self.npnc

        assert len(z) == ncomp, f"Expecting {ncomp} feed fractions, {len(z)} provided."

        for i, z_ in enumerate(z):
            if np.any(z_ <= 0) or np.any(z_ >= 1):
                raise ValueError(
                    f"Violation of strict bound (0, 1) for feed fraction {i + 1}."
                )

        z_sum = safe_sum(z)
        assert np.all(z_sum == 1.0), "Feed fractions violate unity."

        # Declaring output
        fluid_state: FluidState
        flash_type: Literal["p-T", "p-h", "v-T", "v-h"]
        f_dim: int  # Dimension of flash system (unknowns & equations including NPIPM)
        NF: int  # number of vectorized target states

        if p is not None and T is not None and (h is None and v is None):
            flash_type = "p-T"
            f_dim = nphase - 1 + nphase * ncomp
            state_1 = p
            state_2 = T
        elif p is not None and h is not None and (T is None and v is None):
            flash_type = "p-h"
            f_dim = nphase - 1 + nphase * ncomp + 1
            state_1 = p
            state_2 = h
        elif v is not None and T is not None and (p is None and h is None):
            flash_type = "v-T"
            f_dim = 2 * (nphase - 1) + nphase * ncomp + 1
        elif v is not None and h is not None and (T is None and p is None):
            flash_type = "v-h"
            f_dim = 2 * (nphase - 1) + nphase * ncomp + 2
            state_1 = v
            state_2 = h
        else:
            raise NotImplementedError(
                f"Unsupported flash with state definitions {p, T, h, v}"
            )

        # broadcasting vectorized input
        t = z_sum + state_1 + state_2
        if isinstance(t, np.ndarray):
            NF = t.shape[0]
        elif isinstance(t, pp.number):
            NF = 1
        else:
            raise TypeError(
                f"Could not unify types of input arguments: "
                + f"z_sum={type(z_sum)} "
                + f"state_1={type(state_1)} "
                + f"state_2={type(state_2)} "
            )

        if isinstance(initial_state, FluidState):
            n = len(initial_state.y)
            assert n == nphase, f"Expecting {nphase} phase fractions, {n} provided."
            assert np.allclose(
                safe_sum(initial_state.y), 1.0
            ), f"Initial phase fractions violate strong unity constraint."

            for j in range(nphase):
                assert np.all(
                    safe_sum(initial_state.phases[j].x) <= 1.0 + 1e-7
                ), f"Component fractions in phase {j} violate weak unity constraint."
                n = len(initial_state.phases[j].x)
                assert n == self.nc_per_phase[j], (
                    f"Expexting {self.nc_per_phase[j]} fractions of components in phase"
                    + f" {j}, {n} provided."
                )

            if "v" in flash_type:
                n = len(initial_state.sat)
                assert (
                    n == nphase
                ), f"Expecting {nphase} phase saturations, {n} provided."
                assert np.allclose(
                    safe_sum(initial_state.sat), 1.0
                ), f"Initial phase saturations violate strong unity constraint."
            fluid_state = initial_state
        else:
            fluid_state = FluidState()

        # Uniformization of state values
        try:
            s_1 = np.zeros(NF)
            s_2 = np.zeros(NF)
            s_1[:] = state_1
            s_2[:] = state_2

            Z = list()
            for z_ in z:
                _z = np.zeros(NF)
                _z[:] = z_
                Z.append(_z)
            fluid_state.z = np.array(Z)
        except ValueError as err:
            if "broadcast" in str(err):
                raise TypeError(
                    f"Failed to uniformize vectorized input for:\n"
                    + f"state 1: len = {len(state_1)}"
                    + f"state 2: len = {len(state_2)}"
                    + safe_sum(
                        [f"feed {i}: len = {len(z_)}\n" for i, z_ in enumerate(z)]
                    )
                )

        if flash_type == "p-T":
            fluid_state.p = s_1
            fluid_state.T = s_2
        elif flash_type == "p-h":
            fluid_state.p = s_1
            fluid_state.h = s_2
        elif flash_type == "v-T":
            fluid_state.v = s_1
            fluid_state.T = s_2
        elif flash_type == "v-h":
            fluid_state.v = s_1
            fluid_state.h = s_2
        else:
            # alert developers if sth missing, error should be catched above
            assert False, "Missing parsing of fluid input state"

        # uniformization of initial values if provided
        if isinstance(initial_state, FluidState):
            try:
                # molar fractions
                Y = list()
                for j in range(nphase):
                    y = np.zeros(NF)
                    y[:] = fluid_state.y[j]
                    Y.append(y)
                    # fractions of components in phase
                    X = list()
                    for i in range(self.nc_per_phase[j]):
                        x = np.zeros(NF)
                        x[:] = fluid_state.phases[j].x[i]
                        X.append(x)
                    fluid_state.phases[j].x = np.array(X)
                fluid_state.y = np.array(Y)

                if "v" in flash_type:
                    S = list()
                    for j in range(nphase):
                        s = np.zeros(NF)
                        s[:] = fluid_state.sat[j]
                        S.append(s)
                    fluid_state.sat = np.array(S)
                    p = np.zeros(NF)
                    p[:] = fluid_state.p
                    fluid_state.p = p
                if "T" not in flash_type:
                    T = np.zeros(NF)
                    T[:] = fluid_state.T
                    fluid_state.T = T
            except ValueError as err:
                if "broadcast" in str(err):
                    xl = [[len(x) for x in phase.x] for phase in fluid_state.phases]
                    raise TypeError(
                        f"Failed to uniformize input state for:\n"
                        + f"y: lengths ({[len(y for y in fluid_state.y)]})\n"
                        + f"s: lengths ({[len(s for s in fluid_state.sat)]})\n"
                        + f"s: lengths {xl}"
                    )

        return fluid_state, flash_type, f_dim, NF

    @abc.abstractmethod
    def flash(
        self,
        z: Sequence[np.ndarray],
        p: Optional[np.ndarray] = None,
        T: Optional[np.ndarray] = None,
        h: Optional[np.ndarray] = None,
        v: Optional[np.ndarray] = None,
        initial_state: Optional[FluidState] = None,
        parameters: dict = dict(),
    ) -> tuple[FluidState, np.ndarray, np.ndarray]:
        """Abstract method for performing a flash procedure.

        Exactly 2 thermodynamic state must be defined in terms of ``p, T, h`` or ``v``
        for an equilibrium problem to be well-defined.

        One state must relate to pressure or volume.
        The other to temperature or energy.

        Parameters:
            z: ``len=num_components``

                A squence of feed fractions per component.
            p: Pressure at equilibrium.
            T: Temperature at equilibrium.
            h: Specific enthalpy of the mixture at equilibrium,
            v: Specific volume of the mixture at equilibrium,
            initial_state: ``default=None``

                If not given, an initial guess must be computed by the flash class.

                If given, it must have at least values for phase fractions and
                compositions.

                It must have additionally values for temperature, for
                a state definition where temperature is not known at equilibrium.

                It must have additionally values for pressure and saturations, for
                state definitions where pressure is not known at equilibrium.
            params: ``default={}``

                Optional dictionary containing anything else required for custom flash
                classes.

        Returns:
            A 3-tuple containing the results, success flags and number of iterations.
            The results are stored in a fluid state structure.

            Important:
                If the equilibrium state is not defined in terms of pressure or
                temperature, the resulting volume or enthalpy values of the fluid might
                differ slightly from the input values, due to precision and convergence
                criterion.
                Extensive properties are always returned in terms of the computed
                pressure or temperature.

        """
        ...
