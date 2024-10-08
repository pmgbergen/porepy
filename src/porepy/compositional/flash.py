"""Module containing an abstraction layer for the flash procedure."""

from __future__ import annotations

import abc
import logging
from typing import Callable, Literal, Optional, Sequence

import numpy as np

import porepy as pp
import porepy.compositional as ppc

from .base import FluidMixture
from .states import FluidProperties
from .utils import CompositionalModellingError, safe_sum

__all__ = ["Flash", "FlashMixin"]

logger = logging.getLogger(__name__)


class Flash(abc.ABC):
    """Abstract base class for flash algorithms defining the interface of flash objects.

    The definition of the interface is done mainly as an orientation for compatibility
    with the remainder of PorePy's framework (especially the compositional flow).

    """

    def __init__(self, mixture: ppc.FluidMixture) -> None:
        super().__init__()

        ncomp = mixture.num_components
        nphase = mixture.num_phases

        self.npnc: tuple[int, int] = (nphase, ncomp)
        """Number of phases and components present in mixture."""

        self.nc_per_phase: tuple[int, ...] = tuple(
            [phase.num_components for phase in mixture.phases]
        )
        """Number of components modelled in each phase."""

        self.tolerance: float = 1e-8
        """Convergence criterion for the flash algorithm. Defaults to ``1e-8``."""

        self.max_iter: int = 150
        """Maximal number of iterations for the flash algorithms. Defaults to 150."""

    def parse_flash_input(
        self,
        z: Sequence[np.ndarray | pp.number],
        p: Optional[np.ndarray | pp.number] = None,
        T: Optional[np.ndarray | pp.number] = None,
        h: Optional[np.ndarray | pp.number] = None,
        v: Optional[np.ndarray | pp.number] = None,
        initial_state: Optional[FluidProperties] = None,
    ) -> tuple[
        FluidProperties,
        Literal["p-T", "p-h", "v-T", "v-h"],
        int,
        int,
    ]:
        """Helper method to parse the input and construct a provisorical fluid state
        with uniform input (numpy arrays of same size).

        The parameters are described in :meth:`flash`.

        Determins also the equilibrium definition, the size of the (local) flash
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
            3. A number denoting the size of the locall equilibrium problem (dofs).
            4. A number denoting the size of the input after uniformization.

        """

        nphase, ncomp = self.npnc

        assert len(z) == ncomp, f"Expecting {ncomp} feed fractions, {len(z)} provided."

        for i, z_ in enumerate(z):
            if np.any(z_ <= 0) or np.any(z_ >= 1):
                raise ValueError(
                    f"Violation of bound [0, 1] for feed fraction {i + 1}."
                )

        z_sum = safe_sum(z)
        assert np.all(z_sum == 1.0), "Feed fractions violate unity."

        # Declaring output
        fluid_state: FluidProperties
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

        if isinstance(initial_state, FluidProperties):
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
            fluid_state = FluidProperties()

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
        # NOTE the state cannot set v, as it is always the reciprocal of rho
        # set rho as the reciprocal of target v
        elif flash_type == "v-T":
            fluid_state.rho = 1.0 / s_1
            fluid_state.T = s_2
        elif flash_type == "v-h":
            fluid_state.rho = 1.0 / s_1
            fluid_state.h = s_2
        else:
            # alert developers if something is missing, error should be catched above
            assert False, "Missing parsing of fluid input state"

        # uniformization of initial values if provided
        if isinstance(initial_state, FluidProperties):
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
        initial_state: Optional[FluidProperties] = None,
        parameters: dict = dict(),
    ) -> tuple[FluidProperties, np.ndarray, np.ndarray]:
        """Abstract method for performing a flash procedure.

        Exactly 2 thermodynamic states must be defined in terms of ``p, T, h`` or ``v``
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
            parameters: ``default={}``

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


class FlashMixin:
    """Mixin class to introduce the unified flash procedure into the solution strategy.

    Main ideas of the FlashMixin:

    1. Instantiation of Flash object and make it available for other mixins.
       :meth:`set_up_flasher`.
    2. Convenience methods to equilibriate the fluid.
    3. Abstraction to enable customization.

    """

    flash: Flash
    """A flasher object able to compute the fluid phase equilibrium for a mixture
    defined in the mixture mixin.

    This object should be created here during :meth:`set_up_flasher`.

    """

    flash_params: dict = dict()
    """The dictionary to be passed to a flash algorithm, whenever it is called."""

    mdg: pp.MixedDimensionalGrid
    """See :class:`~porepy.models.geometry.ModelGeometry`."""
    equation_system: pp.ad.EquationSystem
    """See :class:`~porepy.models.solution_strategy.SolutionStrategy`."""
    fluid_mixture: FluidMixture
    """See :class:`FluidMixtureMixin`."""

    fractional_state_from_vector: Callable[
        [Sequence[pp.Grid], Optional[np.ndarray]], FluidProperties
    ]
    """See :class:`~porepy.compositional.compositional_mixins.CompositionalVariables`.
    """

    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.
    """
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.energy_balance.VariablesEnergyBalance`."""
    enthalpy: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """See :class:`~porepy.models.compositional_flow.VariablesCF`."""
    volume: Callable[[list[pp.Grid]], pp.ad.Operator]
    """See :class:`~porepy.models.compositional_flow.SolidSkeletonCF`."""

    equilibrium_type: Optional[str]
    """See :class:`~porepy.models.compositional_flow.SolutionStrategyCF`"""

    def set_up_flasher(self) -> None:
        """Method to introduce the flash class, if an equilibrium is defined.

        This method is called by the solution strategy after the model is set up.

        """
        raise CompositionalModellingError(
            "Call to mixin method. No flash object defined."
        )

    def get_fluid_state(
        self, subdomains: Sequence[pp.Grid], state: Optional[np.ndarray] = None
    ) -> FluidProperties:
        """Method to assemble a fluid state in the iterative procedure, which
        should be passed to :meth:`equilibriate_fluid`.

        This method provides room to pre-process data before the flash is called with
        the returned fluid state as the initial guess.

        Parameters:
            subdomains: Subdomains for which the state functions should be evaluated
            state: ``default=None``

                Global state vector to be passed to the Ad framework when evaluating the
                current state (fractions, pressure, temperature, enthalpy,..)

        Returns:
            The base method returns a fluid state containing the current iterate value
            of the unknowns of respective flash subproblem (p-T, p-h,...).

        """

        # Extracting the current, iterative state to use as initial guess for the flash
        fluid_state = self.fractional_state_from_vector(subdomains, state)

        # Evaluate temperature as initial guess, if not fixed in equilibrium type
        if "T" not in self.equilibrium_type:
            # initial guess for T from iterate
            fluid_state.T = self.temperature(subdomains).value(
                self.equation_system, state
            )
        # evaluate pressure, if volume is fixed. NOTE saturations are also fractions
        # and already included
        if "v" in self.equilibrium_type:
            fluid_state.p = self.pressure(subdomains).value(self.equation_system, state)

        return fluid_state

    def equilibriate_fluid(
        self,
        subdomains: Sequence[pp.Grid],
        state: Optional[np.ndarray] = None,
        initial_fluid_state: Optional[FluidProperties] = None,
    ) -> tuple[FluidProperties, np.ndarray]:
        """Convenience method perform the flash based on model specifications.

        This method is called in
        :meth:`~porepy.models.compositional_flow.SolutionStrategyCF.
        before_nonlinear_iteration` to use the flash as a predictor during nonlinear
        iterations.

        Parameters:
            subdomains: Subdomains on which to evaluate the target state functions.
            state: ``default=None``

                Global state vector to be passed to the Ad framework when evaluating the
                state functions.
            initial_fluid_state: ``default=None``

                Initial guess passed to :meth:`~porepy.compositional.flash.Flash.flash`.
                Note that if None, the flash computes the initial guess itself.

        Returns:
            The equilibriated state of the fluid and an indicator where the flash was
            successful (or not).

            For more information on the `success`-indicators, see respective flash
            object.

        """

        if initial_fluid_state is None:
            z = np.array(
                [
                    comp.fraction(subdomains).value(self.equation_system)
                    for comp in self.fluid_mixture.components
                ]
            )
        else:
            z = initial_fluid_state.z

        flash_kwargs = {
            "z": z,
            "initial_state": initial_fluid_state,
            "parameters": self.flash_params,
        }

        if "p-T" in self.equilibrium_type:
            flash_kwargs.update(
                {
                    "p": self.pressure(subdomains).value(self.equation_system, state),
                    "T": self.temperature(subdomains).value(
                        self.equation_system, state
                    ),
                }
            )
        elif "p-h" in self.equilibrium_type:
            flash_kwargs.update(
                {
                    "p": self.pressure(subdomains).value(self.equation_system, state),
                    "h": self.enthalpy(subdomains).value(self.equation_system, state),
                }
            )
        elif "v-h" in self.equilibrium_type:
            flash_kwargs.update(
                {
                    "v": self.volume(subdomains).value(self.equation_system, state),
                    "h": self.enthalpy(subdomains).value(self.equation_system, state),
                }
            )
        else:
            raise CompositionalModellingError(
                "Attempting to equilibriate fluid with uncovered equilibrium type"
                + f" {self.equilibrium_type}."
            )

        result_state, succes, _ = self.flash.flash(**flash_kwargs)

        return result_state, succes

    def postprocess_flash(
        self, subdomain: pp.Grid, fluid_state: FluidProperties, success: np.ndarray
    ) -> FluidProperties:
        """A method called after :meth:`equilibriate_fluid` to post-process failures if
        any.

        The base method asserts that ``success`` is zero everywhere.

        Parameters:
            subdomain: A grid for which ``fluid_state`` contains the values.
            fluid_state: Fluid state returned from :meth:`equilibriate_fluid`.
            success: Success flags returned along the fluid state.

        Returns:
            A final fluid state, with treatment of values where the flash did not
            succeed.

        """
        # nothing to do if everything successful
        if np.all(success == 0):
            return fluid_state
        else:
            raise ValueError(
                "Flash strategy did not succeed in"
                + f" {(success > 0).sum()} / {len(success)} cases."
            )
