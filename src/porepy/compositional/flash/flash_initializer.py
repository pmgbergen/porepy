"""Module containing functionality to provide initial guesses for the equilibrium
problem."""

from __future__ import annotations

import abc
import logging
import time
from functools import partial
from typing import Callable, Optional, cast

import numba as nb
import numpy as np

import porepy as pp

from .._global_thermodynamic_reference_state import R_U
from .._global_thermodynamic_reference_state import T as T_REF
from .._numba_interface import (
    NUMBA_CACHE,
    NUMBA_FAST_MATH,
    NUMBA_PARALLEL,
    cfunc,
    get_empty_numba_dict,
    njit,
    typeof,
)
from ..compiled_eos import CompiledEoS
from ..utils import (
    FlashSpec,
    FlashSpec_NUMBA_TYPE,
    PhysicalState,
    _compute_saturations,
    compute_saturations,
    normalize_rows,
)
from .flash_equations import (
    assemble_generic_arg,
    assemble_vectorized_generic_arg,
    first_order_constraint_jac,
    first_order_constraint_res,
    parse_generic_arg,
    parse_vectorized_generic_arg,
    phase_mass_constraints_jac,
    phase_mass_constraints_res,
)
from .solvers._core import SOLVER_PARAMETERS_TYPE

__all__ = [
    "FlashInitializer",
    "UniformFlashInitializer",
    "HeuristicVLInitializer",
    "K_Wilson",
    "dT_K_Wilson",
    "cubic_mix",
    "critical_pressure_guess",
    "get_dew_point_T",
    "get_bubble_point_T",
]

logger = logging.getLogger(__name__)

_COMPILER = njit
"""Decorator for compiling functions in this module.

Uses :func:`~porepy.compositional._numba_interface.njit`

"""


# region Helper methods


@cfunc(nb.f8[:, :](nb.f8, nb.f8, nb.f8[:, :], nb.f8[:]), cache=True)
def get_K_values_template_func(
    p: float, T: float, x: np.ndarray, params: np.ndarray
) -> np.ndarray:
    """Template c-function for K-value computations.

    Parameters:
        p: Pressure.
        T: Temperature.
        x: 2D array containing row-wise extended partial fractions per phase.
        params: 1D array containing the parameters stored in the generic argument.

    Returns:
        K-values w.r.t. to the reference phase (first row in ``x``).

    """
    return x * p * T


@cfunc(nb.f8[:](nb.f8[:], SOLVER_PARAMETERS_TYPE), cache=True)
def update_state_template_func(
    X_gen: np.ndarray, params: dict[str, float]
) -> np.ndarray:
    """Template c-functions for methods which update state functions such as pressure
    or temperature.

    Parameters:
        X_gen: Generic flash argument.
        params: Initialization parameters.

    Returns:
        Updated ``X_gen``.

    """
    return X_gen * params["0"]


@_COMPILER(nb.f8[:](nb.f8, nb.f8, nb.f8[:], nb.f8[:], nb.f8[:]), cache=True)
def K_Wilson(
    p: float, T: float, p_cs: np.ndarray, T_cs: np.ndarray, omegas: np.ndarray
) -> np.ndarray:
    """Wilson correlation for K-values (ratio of liquid to gas fugacity coefficient).

    Parameters:
        p: Pressure [Pa].
        T: Temperature [K].
        p_cs: Critical pressures per component.
        T_cs: Critical temperatures per component.
        omegas: Acentric factors per component.

    Returns:
        An array of size ``T_cs``/``p_cs`` containing K-values per component (ratio
        of fugacity in liquid and fugacity in vapor).

    """
    return np.exp(5.37 * (1 + omegas) * (1 - T_cs / T)) * p_cs / p + 1e-10


@_COMPILER(nb.f8[:](nb.f8, nb.f8, nb.f8[:], nb.f8[:], nb.f8[:]), cache=True)
def dT_K_Wilson(
    p: float, T: float, p_cs: np.ndarray, T_cs: np.ndarray, omegas: np.ndarray
) -> np.ndarray:
    """Temperature-derivative of Wilson correlation for K-values.

    Parameters:
        p: Pressure [Pa].
        T: Temperature [K].
        p_cs: Critical pressures per component.
        T_cs: Critical temperatures per component.
        omegas: Acentric factors per component.

    Returns:
        An array of size ``T_cs``/``p_cs`` containing the ``dK/dT``.

    """
    c = 5.37 * (1 + omegas)
    return np.exp(c * (1 - T_cs / T)) * c * T_cs * p_cs / (p * T**2)


@_COMPILER(nb.f8(nb.f8[:], nb.f8[:]), fastmath=NUMBA_FAST_MATH, cache=True)
def cubic_mix(x: np.ndarray, phis: np.ndarray) -> float:
    """Advanced mixing rule of Lorentz-Berthelot-type, used for volume-like quantities.

    Parameters:
        x: Fractions.
        phis: Quantity to be mixed.

    Returns:
        Approximation of the quantity for the mixture corresponding to the fractions.

    """
    n = x.size
    phi_mix = 0.0
    cphis = np.cbrt(phis)
    for i in range(n):
        phi_mix += phis[i] * x[i] ** 2
        phi_mix += x[i] * np.sum(x[i + 1 :] * (cphis[i] + cphis[i + 1 :]) ** 3) / 4.0
    return phi_mix


@_COMPILER(
    nb.f8(nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:]),
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def critical_pressure_guess(
    x: np.ndarray, p_cs: np.ndarray, T_cs: np.ndarray, v_cs: np.ndarray
) -> float:
    """Guess for critical pressure of a mixture using heuristics.

    See also:

        - Kay's rule, Prausnitz-Gunn rule, Lorentz-Berthelot-type mixing.
        - :func:`cubic_mix`
        - `Saha, Carrol 1997: The isoenergetic-isochoric flash
          <https://doi.org/10.1016/S0378-3812(97)00151-9>`_

    Parameters:
        x: Fractions.
        p_pcs: Critical pressure values.
        T_pcs: Critical temperature values.
        v_pcs: Critical specific volume values.

    Returns:
        A guess for the pseudo-critical pressure.

    """
    x_max = x.max()
    # Kay's rule if 1 component clearly dominates.
    if x_max >= 0.9:
        p_pc = np.dot(x, p_cs)
    # Prausnitz-Gunn rule if 1 component is almost dominant.
    # Other components influence pseudo-critical value more.
    elif 0.5 < x_max < 0.9:
        T_pc = np.dot(x, T_cs)
        p_pc = T_pc / np.dot(x, T_cs / p_cs)
    # Modified Prausnitz-Gunn rule if no clear dominance.
    # Include information on critical specific volume.
    else:
        T_pc = np.dot(x, T_cs)
        v_pc = cubic_mix(x, v_cs)
        v_pc_lin = np.dot(x, v_cs)
        # Pseudo-critical compressibility factor.
        Z_pc = np.dot(x, p_cs * v_cs / T_cs) / R_U
        p_pc_cub = Z_pc * R_U * T_pc / v_pc
        p_pc_lin = np.dot(x, p_cs)
        # For high-variability mixtures, average with Kay's rule.
        v_var = np.abs((v_cs - v_pc_lin) / v_pc_lin).max()
        # NOTE: The weighing towards Kay's rule should be influenced by strong
        # variability. For polar fluids like water mixed with CO2/H2S for example, it
        # might be more beneficial to lean more towards Kay.
        if v_var > 0.1:
            # 0.7 is upper bound, at most that much Kay's rule.
            # 0.2 is lower bound, at least that much Kay's rule.
            # 1.2 is a slope for variability, more variability, more Kay's rule.
            w = min(0.7, 0.2 + 1.2 * v_var)
            p_pc = w * p_pc_lin + (1 - w) * p_pc_cub
        # Else use the cubic rule.
        else:
            p_pc = p_pc_cub

    return p_pc


@_COMPILER(
    nb.f8(nb.f8, nb.f8, nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:]),
    cache=NUMBA_CACHE,
)
def get_dew_point_T(
    T0: float,
    p: float,
    z: np.ndarray,
    p_cs: np.ndarray,
    T_cs: np.ndarray,
    omegas: np.ndarray,
) -> float:
    r"""Approximates the dew point temperature using Rachford-rice equations and
    Wilson-correlations for K-values.

    Applies Newton to obtain ``T`` at fixed ``p``:

    ..math::

        \sum_i \frac{z_i}{K_i(p, T)} = 1

    Parameters:
        T0: Initial guess for ``T``.
        p: Pressure value [Pa].
        z: Feed fractions per component.
        p_cs: Critical pressures per component [Pa].
        T_cs: Critical temperatures per component [Pa].
        omegas: acentric factors per component [Pa].

    Returns:
        A temperature solving above equation approximately.

    """
    Ti = T0
    T_r = T_cs.max()
    for _ in range(15):  # Simple newton loop.
        K_i = K_Wilson(p, Ti, p_cs, T_cs, omegas)
        r_i = np.sum(z / K_i) - 1.0  # residual
        if np.abs(r_i) < 1e-7:
            break

        dKdT_i = dT_K_Wilson(p, Ti, p_cs, T_cs, omegas)
        J_i = np.sum(-z / K_i**2 * dKdT_i) * T_r  # Jacobian
        dT = -r_i / J_i
        dT = np.sign(dT) * min(np.abs(dT), 0.1)
        Ti += dT * T_r

    return Ti


@_COMPILER(
    nb.f8(nb.f8, nb.f8, nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:]),
    cache=NUMBA_CACHE,
)
def get_bubble_point_T(
    T0: float,
    p: float,
    z: np.ndarray,
    p_cs: np.ndarray,
    T_cs: np.ndarray,
    omegas: np.ndarray,
) -> float:
    r"""Approximates the bubble point temperature using Rachford-rice equations and
    Wilson-correlations for K-values.

    Applies Newton to obtain ``T`` at fixed ``p``:

    ..math::

        \sum_i z_i K_i(p, T) = 1

    Parameters:
        T0: Initial guess for ``T``.
        p: Pressure value [Pa].
        z: Feed fractions per component.
        p_cs: Critical pressures per component [Pa].
        T_cs: Critical temperatures per component [Pa].
        omegas: acentric factors per component [Pa].

    Returns:
        A temperature solving above equation approximately.

    """
    Ti = T0
    T_r = T_cs.max()
    for _ in range(15):  # Similar to dew point.
        K_i = K_Wilson(p, Ti, p_cs, T_cs, omegas)
        r_i = np.sum(z * K_i) - 1.0
        if np.abs(r_i) < 1e-7:
            break

        dKdT_i = dT_K_Wilson(p, Ti, p_cs, T_cs, omegas)
        J_i = np.sum(z * dKdT_i) * T_r
        dT = -r_i / J_i
        dT = np.sign(dT) * min(np.abs(dT), 0.1)
        Ti += dT * T_r

    return Ti


# endregion
# region Rachford-Rice equations


@_COMPILER(nb.f8[:](nb.f8[:], nb.f8[:, :]), fastmath=NUMBA_FAST_MATH, cache=True)
def _rr_poles(y: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Parameters:
        y: ``shape=(num_phases,)``

            Phase fractions, assuming the first one belongs to the reference phase.
        K: Matrix of K-values per independent phase (row) per component (column)

    Returns:
        A vector of length ``num_components`` containing the denominators in the
        RR-equation related to K-values per component.
        Each demoninator is given by :math:`1 + \\sum_{j\\neq r} y_j (K_{ji} - 1)`.

    """
    # tensordot is the fastes option for non-contigous arrays,
    # but currently unsupported by numba TODO
    # return 1 + np.tensordot(K.T - 1, y[1:], axes=1)
    return 1 + (K.T - 1) @ y[1:]  # K-values given for each independent phase


@_COMPILER(nb.f8(nb.f8[:], nb.f8[:]), fastmath=NUMBA_FAST_MATH, cache=True)
def _rr_binary_vle_inversion(z: np.ndarray, K: np.ndarray) -> float:
    """Inverts the Rachford-Rice equation for the binary 2-phase case.

    Parameters:
        z: ``shape=(num_components,)``

            Vector of feed fractions.
        K: ``shape=(num_components,)``

            Matrix of K-values per component between vapor and liquid phase.

    Returns:
        The corresponding value of the vapor fraction.

    """
    ncomp = z.shape[0]
    n = np.sum((1 - K) * z)
    d = np.empty(ncomp)
    for i in range(ncomp):
        d[i] = (K[i] - 1) * np.sum(np.delete(K, i) - 1) * z[i]

    return n / np.sum(d)


@_COMPILER(nb.f8(nb.f8[:], nb.f8[:], nb.f8[:, :]), cache=NUMBA_CACHE)
def _rr_potential(z: np.ndarray, y: np.ndarray, K: np.ndarray) -> float:
    r"""Calculates the potential according to [1] for the j-th Rachford-Rice equation.

    With :math:`n_c` components, :math:`n_p` phases and :math:`R` the reference phase,
    the potential is given by

    .. math::

        F = \sum_i -z_i ln(1 - \sum_{j\neq R}(1 - K_{ij})y_j)

    References:
        [1] `Okuno and Sepehrnoori (2010) <https://doi.org/10.2118/117752-PA>`_

    Parameters:
        z: ``shape=(num_components,)``

            Vector of feed fractions.
        y: ``shape=(num_phases,)``

            Vector of phase fractions.
        K: ``shape=(num_phases - 1, num_components)``

            Matrix of K-values per independent phase (row) per component (column).

    Returns:
        The value of the potential based on above formula.

    """
    return np.sum(-z * np.log(np.abs(_rr_poles(y, K))))


@_COMPILER(
    nb.f8[:](
        typeof(get_K_values_template_func),
        nb.f8[:],
        SOLVER_PARAMETERS_TYPE,
        FlashSpec_NUMBA_TYPE,
        nb.bool,
    ),
    cache=NUMBA_CACHE,
)
def fractions_from_rr(
    get_K_values: Callable[[float, float, np.ndarray, np.ndarray], np.ndarray],
    X_gen: np.ndarray,
    params: dict[str, float],
    spec: FlashSpec,
    use_wilson: bool,
) -> np.ndarray:
    """Guessing fractions for a single flash configuration.

    Supports currently only 2-phase, 2-component mixtures.

    Parameters:
        get_K_values: See :func:`get_K_values_template_func`.
        X_gen: Generic flash argument.
        params: Parameter dictionary.
            Require ``'num_phases','num_components'`` and ``'N1'``, which is the number
            of loops to perform here. Require also critical pressures, temperatures and
            acentric factors for each component.
        flash_type: A string denoting the flash type to parse ``X_gen``.
            See :func:`~porepy.compositional.flash.flash_equations.parse_generic_arg`
        use_wilson: Flag to use the Wilson correlation for the first K-value guess.

    Returns:
        A generic flash argument with updated fractions.

    """
    # Parsing parameters and generic arg.
    nphase = int(params["num_phases"])
    ncomp = int(params["num_components"])
    N1 = int(params["N1"])
    s, x, y, z, p, T, s1, s2, x_p = parse_generic_arg(X_gen, ncomp, nphase, spec)

    omegas = np.empty(ncomp)
    T_cs = np.empty(ncomp)
    p_cs = np.empty(ncomp)
    for i in range(ncomp):
        T_cs[i] = params[f"_T_crit_{i}"]
        p_cs[i] = params[f"_p_crit_{i}"]
        omegas[i] = params[f"_omega_{i}"]

    # Pseudo-critical quantities.
    T_pc = np.sum(z * T_cs)
    p_pc = np.sum(z * p_cs)

    if use_wilson:
        K = np.empty((nphase - 1, ncomp), dtype=np.float64)
        for j in range(nphase - 1):
            K[j, :] = K_Wilson(p, T, p_cs, T_cs, omegas)
    else:
        K = get_K_values(p, T, x, x_p)

    # NOTE If only 1 component, we do not iterate using the Rachford Rice equations.
    # We check the K value. If they are between 0 and 1, it's liquid, if they are above
    # 1, its vapor. If it is around 1, the vapor fraction is not determinable without
    # energy, hence we set it to 0.5 and return.
    if ncomp == 1:
        N1 = 1

    # Starting iterations using Rachford Rice.
    for n in range(N1):
        # Solving RR for phase fractions.
        if nphase == 2:
            # Only one independent phase assumed.
            K_ = K[0]
            if ncomp == 1:
                if K_[0] < 1.0 - 1e-4:
                    y_ = 0.0
                elif K_[0] > 1.0 + 1e-4:
                    y_ = 1.0
                else:
                    y_ = 0.5
            elif ncomp == 2:
                y_ = _rr_binary_vle_inversion(z, K_)
            else:  # TODO  efficient BRENT method (scipy.optimize.brentq)
                raise NotImplementedError("Multicomponent RR solution not implemented.")

            negative = y_ < 0.0
            exceeds = y_ > 1.0
            invalid = exceeds | negative

            # Correction of invalid gas phase values.
            if invalid:
                # Assuming gas saturated for correction using RR potential.
                y_test = np.array([0.0, 1.0], dtype=np.float64)
                rr_pot = _rr_potential(z, y_test, K)
                # Checking if y is feasible
                # For more information see Equation 10 in
                # `Okuno et al. (2010) <https://doi.org/10.2118/117752-PA>`_
                t_i = _rr_poles(y_test, K)
                cond_1 = t_i - z >= 0.0
                # Tests holds for arbitrary number of phases
                # reflected by implementation, despite nph == 2
                cond_2 = K * z - t_i <= 0.0
                gas_feasible = np.all(cond_1) & np.all(cond_2)

                if rr_pot > 0.0:
                    y_ = 0.0
                elif (rr_pot < 0.0) & gas_feasible:
                    y_ = 1.0

                # Clearly liquid.
                if (T < T_pc) & (p > p_pc):
                    y_ = 0.0
                # Clearly gas.
                elif (T > T_pc) & (p < p_pc):
                    y_ = 1.0

                # Correction based on negative flash
                # value of y_ must be between innermost poles
                K_min = np.min(K_)
                K_max = np.max(K_)
                y_1 = 1 / (1 - K_max)
                y_2 = 1 / (1 - K_min)
                if y_1 <= y_2:
                    y_feasible = y_1 < y_ < y_2
                else:
                    y_feasible = y_2 < y_ < y_1

                if y_feasible & negative:
                    y_ = 0.0
                elif y_feasible & exceeds:
                    y_ = 1.0

                # If all K-values are smaller than 1 and gas fraction is negative,
                # the liquid phase is clearly saturated.
                # Vice versa, if fraction above 1 and K-.
                if negative & np.all(K_ < 1.0):
                    y_ = 0.0
                elif exceeds & np.all(K_ > 1.0):
                    y_ = 1.0

            y[1] = y_
            y[0] = 1.0 - y_
        else:
            raise NotImplementedError(
                "Fractions guess for more than 2 phases not implemented."
            )

        # resolve compositions
        t = _rr_poles(y, K)
        x[0] = z / t  # fraction in reference phase
        x[1:] = K * x[0]  # fraction in independent phases

        # update K-values if another iteration comes
        if n < N1 - 1:
            K = get_K_values(p, T, x, x_p)

    return assemble_generic_arg(s, x, y, z, p, T, s1, s2, x_p, spec)


@_COMPILER(
    nb.f8[:, :](
        typeof(get_K_values_template_func),
        nb.f8[:, :],
        SOLVER_PARAMETERS_TYPE,
    ),
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_CACHE,
)
def rachford_rice_initializer(
    get_K_values: Callable[[float, float, np.ndarray, np.ndarray], np.ndarray],
    X_gen: np.ndarray,
    params: dict[str, float],
) -> np.ndarray:
    """pT initializer as a parallelized loop over all rows in the vectorized generic
    flash argument.

    Uses the Rachford-Rice equations to compute some guess for phase fractions and
    extended partial fractions.

    Parameters:
        get_K_values: See :func:`get_K_values_template_func`.
        X_gen: Vectorized generic flash argument.
        params: Initialization parameters. See :func:`fractions_from_rr` for a list of
            required parameters.

    Returns:
        ``X_gen`` with initialized fraction values.

    """
    for f in nb.prange(X_gen.shape[0]):
        X_gen[f] = fractions_from_rr(get_K_values, X_gen[f], params, FlashSpec.pT, True)
    return X_gen


# endregion


@_COMPILER(
    nb.f8[:, :](
        typeof(get_K_values_template_func),
        typeof(update_state_template_func),
        FlashSpec_NUMBA_TYPE,
        nb.f8[:, :],
        SOLVER_PARAMETERS_TYPE,
    ),
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_CACHE,
)
def nested_initializer(
    get_K_values: Callable[[float, float, np.ndarray, np.ndarray], np.ndarray],
    update_state_func: Callable[[np.ndarray, dict[str, float]], np.ndarray],
    spec: FlashSpec,
    X_gen: np.ndarray,
    params: dict[str, float],
) -> np.ndarray:
    """Nested initializer for alternating updates of fractions and other state functions
    like pressure and temperature.

    Parameters:
        get_K_values: See :func:`fractions_from_rr`.
        update_state_func: A callable taking a generic flash argument and updating
            some state values. Will be called first, then ``fractions_from_rr``.
        flash_type: A string denoting the flash type. Required for parsing the
            generic flash argument.
        X_gen: Vectorized generic flash argument, such that the initialization is
            performed row-wise.
        params: Initialization parameters. Required is ``'N3'``, denoting the number
            of alternations between state and fraction update.

    Returns:
        The updated/initialized ``X_gen``.

    """
    N3 = int(params["N3"])
    for f in nb.prange(X_gen.shape[0]):
        xf = X_gen[f]
        for _ in range(N3):
            xf = update_state_func(xf, params)
            xf = fractions_from_rr(get_K_values, xf, params, spec, False)
        X_gen[f] = xf
    return X_gen


class FlashInitializer(abc.ABC):
    """Abstract flash initializer defining the API.

    This is a container for initialization routines per flash specification.
    It can be compiled, if required for the numba framework.

    Supports only non-trivial fluid mixtures, i.e. with at least two phases and one
    component.

    Parameters:
        fluid: A fluid mixture.
        params: ``default=None``

            Initialization parameters.

    """

    def __init__(
        self,
        fluid: pp.Fluid[pp.FluidComponent, pp.Phase[pp.FluidComponent]],
        params: Optional[dict[str, float]] = None,
    ) -> None:
        super().__init__()

        ncomp = fluid.num_components
        nphase = fluid.num_phases

        assert nphase >= 2, "Require at least two phases."
        assert ncomp >= 1, "Require at least one component."

        self._n_PC: tuple[int, int] = (nphase, ncomp)
        """Tuple containing the number of phases and components in the fluid."""

        self.params: dict[str, float | int | bool] = (
            params if isinstance(params, dict) else {}
        )
        """Parameters for initialization routines passed at instantiation.
        
        Defaults to empty dict.
    
        """

    @abc.abstractmethod
    def __getitem__(self, key: FlashSpec) -> Callable[[np.ndarray], np.ndarray]:
        """Abstract getter defining the interface to access initialization routines per
        flash specification.

        An initialization routine takes a generic flash argument and returns a populated
        one containing the initial guess. It may take vectorized input.

        Parameters:
            key: A supported flash specification.

        Returns:
            A callable taking a (vectorized) generic flash argument and returning
            a populated argument with initial values.

        """

    def compile(self, *args: FlashSpec) -> None:
        """Compilation interface for initialization routines per flash specification.

        The base method is empty, not abstract.

        Parameters:
            *args: Specify subset of flash types which should be compiled to safe time.

        """


class UniformFlashInitializer(FlashInitializer):
    """Simple class providing initial values for the flash using a uniform distribution
    of mass for phases and components in phases.

    The default initialization uses a uniform distribution for all fractions.

    Two types of bias can be specified:

    ``feed_bias``:

    The uniform distribution is averaged with the feed fraction for each
    component. Favorable when a phase is present at equilibrium.

    ``liquid_bias`` for liquid phases only:

    If multiple liquid phases, each phase is assumed to be
    dominated by 1 component, i.e., its partial fraction is set to be larger than
    uniform. The value for how large can be set with ``params['liquid_bias'] = 0.9``.
    The rest is split uniformly accross the remaining components in that phase.
    If the ``feed_bias`` is also active, it is applied. Note, however, if the
    feed fraction is such that the average would be pulled below uniform value,
    the feed bias is skipped for this liquid phase, as it runs into conflict with the
    liquid bias. The liquid bias is favorable if the liquid phase is present at
    equilibrium. Makes only sense if the value is greater than ``1/num_phases``.

    Pseudo-critical values for pressure or temperature are computed as an
    initial guess, if the specification requires it. Saturations are set to be equal to
    phase fractions.

    Parameters:
        fluid: A fluid containing at least 2 phases and 1 component.
        params: ``default=None``

            Initial parametrization, defaulting to no bias.

    """

    def __init__(
        self,
        fluid: pp.Fluid[pp.FluidComponent, pp.Phase[pp.FluidComponent]],
        params: Optional[dict[str, float]] = None,
    ) -> None:
        super().__init__(fluid, params)
        assert self._n_PC[0] <= self._n_PC[1] + 1, (
            "Not expecting more phases than components + gas phase."
        )

        self._pcrits: np.ndarray = np.array(
            [comp.critical_pressure for comp in fluid.components]
        )
        """Critical pressures per component."""
        self._Tcrits: np.ndarray = np.array(
            [comp.critical_temperature for comp in fluid.components]
        )
        """A list containing critical temperatures per component."""
        self._vcrits: np.ndarray = np.array(
            [comp.critical_specific_volume for comp in fluid.components]
        )
        """A list containing critical volumes per component."""
        self._omegas: np.ndarray = np.array(
            [comp.acentric_factor for comp in fluid.components]
        )
        """A list containing acentric factors per component."""
        self._phasestates: tuple[PhysicalState, ...] = tuple(
            [phase.state for phase in fluid.phases]
        )
        """A sequence containing the physical phase state per phase."""
        self._gas_phase_index: Optional[int] = fluid.gas_phase_index
        """The index of the gas phase. None if gas not existent."""

        self._initializer: Callable[[FlashSpec, dict, np.ndarray], np.ndarray]
        """The actual initialization routine created by this class during compilation.

        Takes the flash specification, the parameter dictionary and an initial generic
        flash argument, and returns the initial guess.

        """

        # Provide new default parameters, if not already present.
        default_params: dict[str, float | int | bool] = {
            "liquid_bias": 0.9,
            "feed_bias": False,
        }
        default_params.update(self.params)
        self.params = default_params

        self.nb_params: dict[str, float]
        """Numba-compiled version of :attr:`params`.
        
        Must be created during the first call to the getter of this class using
        :meth:`compile_nb_params`

        This supports only floats as values as per convention in the flash solver
        package.

        """

    def __getitem__(self, key: FlashSpec) -> Callable[[np.ndarray], np.ndarray]:
        """Shortcut for accessing flash initial guess methods for flash types denoted by
        ``key``.

        The key is meaningless at this point, since the same routine is used for all
        flashes.

        Raises:
            KeyError: If initializer is not compiled.

        """

        if not hasattr(self, "_initializer"):
            raise KeyError("Uniform flash initializer not compiled.")

        if not hasattr(self, "nb_params"):
            self.compile_nb_params()

        def initializer(x: np.ndarray) -> np.ndarray:
            """Wrapper for initialization routine, updating parameters and feeding
            them to the initialization method."""
            params = self.nb_params
            for k, v in self.params.items():
                params[str(k)] = float(v)
            return self._initializer(key, params, x)

        return initializer

    def compile_nb_params(self) -> None:
        """Creates :attr:`nb_params` during the first call to the getter."""
        assert not hasattr(self, "nb_params"), (
            "Numba-parameter dictionary already compiled."
        )

        d: dict[str, float] = get_empty_numba_dict()
        self.nb_params = cast(dict[str, float], d)
        self.nb_params["num_phases"] = float(self._n_PC[0])
        self.nb_params["num_components"] = float(self._n_PC[1])
        self.nb_params["gas_phase_index"] = float(
            -1 if self._gas_phase_index is None else self._gas_phase_index
        )
        self.nb_params["liquid_bias"] = float(self.params["liquid_bias"])
        self.nb_params["feed_bias"] = float(self.params["feed_bias"])

        # Adding also some component parameters which are required
        for i in range(self._n_PC[1]):
            self.nb_params[f"_T_crit_{i}"] = float(self._Tcrits[i])
            self.nb_params[f"_p_crit_{i}"] = float(self._pcrits[i])
            self.nb_params[f"_v_crit_{i}"] = float(self._vcrits[i])
            self.nb_params[f"_omega_{i}"] = float(self._omegas[i])

    def compile(self, *args: FlashSpec) -> None:
        """Triggers the compilation of initialization routine.

        Parameters:
            *args: Specify subset of flash types which should be compiled to safe time.
                The uniform initializer supports all flash types. This signature is
                left for inheritance reasons.

        """

        logger.info(f"Compiling uniform flash initialization ..")
        start = time.time()

        @_COMPILER(nb.f8[:](FlashSpec_NUMBA_TYPE, SOLVER_PARAMETERS_TYPE, nb.f8[:]))
        def initializer(
            spec: FlashSpec, params: dict[str, float], X_gen: np.ndarray
        ) -> np.ndarray:
            # Parsing parameters.
            nphase = int(params["num_phases"])
            ncomp = int(params["num_components"])
            gas_phase_idx = int(params["gas_phase_index"])
            liquid_bias = params["liquid_bias"]
            feed_bias = params["feed_bias"]

            # Critical values per component.
            T_crits = np.empty(ncomp)
            p_crits = np.empty(ncomp)
            v_crits = np.empty(ncomp)
            for i in range(ncomp):
                T_crits[i] = params[f"_T_crit_{i}"]
                p_crits[i] = params[f"_p_crit_{i}"]
                v_crits[i] = params[f"_v_crit_{i}"]

            approx_T = spec not in (FlashSpec.pT, FlashSpec.vT)
            approx_p = spec >= FlashSpec.vT

            _, _, _, z, p, T, s1, s2, x_p = parse_generic_arg(
                X_gen, ncomp, nphase, spec
            )
            # Critical value approximations for pressure and temperature.
            if approx_T:
                T = np.dot(z, T_crits)

            if approx_p:
                p = critical_pressure_guess(z, p_crits, T_crits, v_crits)

            # Phase fractions and saturations are always uniformly guessed.
            y = np.ones(nphase) / nphase
            s = y.copy()

            # Uniform distribution as starting point for partial fractions.
            x = np.ones((nphase, ncomp)) / ncomp

            # Applying bias.
            if max(feed_bias, liquid_bias) > 0:
                # Cap liquid bias at 1.0 for safety.
                liquid_bias = float(min(1.0, liquid_bias))
                # Rest of mass in case of liquid biase is distributed uniformly accross
                # other components.
                lr = (1 - liquid_bias) / max(ncomp - 1, 1)
                # Component index for keeping track of liquid bias in multiphase case.
                k = 0

                # Apply liquid-bias only if more than 1 liquid phase and parameter is
                # not zero. If no gas, gas_phase_idx is -1.
                apply_liq_bias = (liquid_bias > 0) and (
                    (nphase - (1 if gas_phase_idx >= 0 else 0)) > 1
                )

                for j in range(nphase):
                    # Averaging weight according to number of bias applied.
                    w = 1

                    # Feed bias is applicable to any phase.
                    if feed_bias > 0:
                        xfb = z
                        w += 1
                    else:
                        xfb = np.zeros(ncomp)

                    # Apply liquid bias only to liquid phases.
                    # If no gas, index is -1 and never equal to j.
                    if apply_liq_bias and j != gas_phase_idx:
                        xlb = np.ones(ncomp) * lr
                        xlb[k] = liquid_bias
                        w += 1
                        # Cancel the feed bias, if the resulting value for the partial
                        # fraction of the dominant component would be smaller than
                        # uniform.
                        if (
                            feed_bias > 0
                            and (x[j, k] + liquid_bias + z[k]) / 3 <= x[j, k]
                        ):
                            xfb = np.zeros(ncomp)
                            w -= 1
                        k += 1
                    else:
                        xlb = np.zeros(ncomp)

                    x[j] = (x[j] + xfb + xlb) / w

                # Unity should be achieved by above maths, but we play safe.
                x = normalize_rows(x)

            return assemble_generic_arg(s, x, y, z, p, T, s1, s2, x_p, spec)

        @_COMPILER(
            nb.f8[:, :](FlashSpec_NUMBA_TYPE, SOLVER_PARAMETERS_TYPE, nb.f8[:, :]),
            parallel=NUMBA_PARALLEL,
        )
        def init_par(
            spec: FlashSpec, params: dict[str, float], X_gen: np.ndarray
        ) -> np.ndarray:
            for i in nb.prange(X_gen.shape[0]):
                X_gen[i] = initializer(spec, params, X_gen[i].copy())

            return X_gen

        self._initializer = init_par
        logger.info(
            "Flash initialization routine compiled"
            + " (elapsed time: %.4f (s))." % (time.time() - start)
        )


class HeuristicVLInitializer(UniformFlashInitializer):
    """Initializer using heuristics and Rachford-Rice equations to provide initial
    values for fractions, pressure and temperature, depending on the flash type.

    Important:
        If pressure and temperature should be guessed, they must be passed as zeros
        in the generic flash argument.

    Parameters:
        fluid: The fluid for which the flash is compiled. Supports currently only
            2-phase, gas-liquid mixtures.
        params: Initialization parameters (see :attr:`params`).

    """

    SUPPORTED_SPECIFICATIONS: tuple[FlashSpec, ...] = (
        FlashSpec.pT,
        FlashSpec.ph,
        FlashSpec.vh,
        FlashSpec.vu,
    )
    """Supported flash types. Used for checking flash input."""

    def __init__(
        self,
        fluid: pp.Fluid[pp.FluidComponent, pp.Phase[pp.FluidComponent]],
        params: Optional[dict[str, float]] = None,
    ) -> None:
        super().__init__(fluid, params)

        assert self._n_PC[0] == 2, "Supports only 2-phase mixtures."

        eos = fluid.reference_phase.eos
        assert isinstance(eos, CompiledEoS), "Suppors only mixtures with compiled EoS."
        self._eos: CompiledEoS = eos
        """Compiled EoS of the reference phase, assuming all phases have the same EoS.
        """

        self._initializers: dict[
            FlashSpec,
            Callable[[np.ndarray, dict[str, float]], np.ndarray],
        ] = {}
        """Storage of initialization routines per flash.
        
        Initialization routines take a generic argument and a parameter dictionary as
        arguments, and return the updated generic argument.

        The generic argument can be vectorized (flash per row).

        """

        self._get_K_values: Callable[[float, float, np.ndarray, np.ndarray], np.ndarray]
        """Helper function computing K-values. Created during compilation."""

        # Provide new default parameters, if not already present.
        default_params: dict[str, float] = {
            "N1": 3.0,
            "N2": 1.0,
            "N3": 1.0,
            "atol": 1e-4,
        }
        default_params.update(self.params)
        self.params = default_params

    def __getitem__(self, key: FlashSpec) -> Callable[[np.ndarray], np.ndarray]:
        """Accesses the right initialization routine for the requested flash.

        Raises:
            KeyError: If the requested flash initialization is not compiled.

        """
        _ = self._initializers[key]
        if key not in self._initializers:
            raise KeyError(f"{key.name} flash initialization not compiled.")

        if not hasattr(self, "nb_params"):
            self.compile_nb_params()

        def initializer(x: np.ndarray) -> np.ndarray:
            """Wrapper for initialization routine, updating parameters and feeding
            them to the initialization method."""
            params = self.nb_params
            for k, v in self.params.items():
                params[str(k)] = float(v)
            return self._initializers[key](x, params)

        return initializer

    def compile(self, *args: FlashSpec) -> None:
        """Triggers the compilation of initialization routines.

        Parameters:
            *args: Specify subset of flash types which should be compiled to safe time.
                Due to some internal structures, the pT initializer is always compiled.

        Raises:
            ValueError: If unsupported flash specification is passed as an argument.

        """

        # If not specified, compile all.
        if not args:
            args = self.SUPPORTED_SPECIFICATIONS

        if not self._eos.is_compiled:
            self._eos.compile()

        for a in args:
            if a not in self.SUPPORTED_SPECIFICATIONS:
                raise ValueError(f"Unsupported flash specification {a.name}")

        # Setting outer scope variables to avoid referencing self in JIT functions.
        nphase, ncomp = self._n_PC
        phasestates = self._phasestates

        prearg_val_c = self._eos.funcs["prearg_val"]
        prearg_jac_c = self._eos.funcs["prearg_jac"]
        phi_c = self._eos.funcs["phis"]
        h_c = self._eos.funcs["h"]
        dh_c = self._eos.funcs["dh"]
        u_c = self._eos.funcs["u"]
        du_c = self._eos.funcs["du"]
        rho_c = self._eos.funcs["rho"]
        drho_c = self._eos.funcs["drho"]

        logger.info(f"Compiling {[a.name for a in args]} flash initializations ..")
        start = time.time()

        if not hasattr(self, "_get_K_values"):
            logger.debug("Compiling K-value computation ..")

            @_COMPILER(nb.f8[:, :](nb.f8, nb.f8, nb.f8[:, :], nb.f8[:]))
            def get_K_values(
                p: float, T: float, x: np.ndarray, xp: np.ndarray
            ) -> np.ndarray:
                """See :func:`get_K_values_template_func`."""
                # To avoid overflow in exp differences.
                cap = np.log(np.finfo(np.float64).max) - 10.0
                nphase, ncomp = x.shape
                K = np.empty((nphase - 1, ncomp), dtype=np.float64)
                xn = normalize_rows(x)
                pre_0 = prearg_val_c(phasestates[0], p, T, xn[0], xp)
                phi_0 = phi_c(pre_0, p, T, xn[0])
                # NOTE phis are given as ln phis, but K-values are ratios of phis
                for j in range(1, nphase):
                    pre_j = prearg_val_c(phasestates[j], p, T, xn[j], xp)
                    phi_j = phi_c(pre_j, p, T, xn[j])
                    # Binding K-values away from zero.
                    # K[j - 1, :] = phi_0 / phi_j + 1e-10
                    K[j - 1, :] = np.exp(np.minimum(phi_0 - phi_j, cap)) + 1e-10
                return K

            self._get_K_values = get_K_values

        get_K_values = self._get_K_values

        if FlashSpec.pT in args and FlashSpec.pT not in self._initializers:
            logger.debug("Compiling pT-initialization ..")

            self._initializers[FlashSpec.pT] = partial(
                rachford_rice_initializer, get_K_values
            )

        if FlashSpec.ph in args and FlashSpec.ph not in self._initializers:
            logger.debug("Compiling ph-initialization ..")

            @_COMPILER(
                nb.f8[:, :](
                    nb.f8[:, :],
                    SOLVER_PARAMETERS_TYPE,
                ),
                parallel=NUMBA_PARALLEL,
            )
            def ph_init(
                X_gen: np.ndarray,
                params: dict[str, float],
            ) -> np.ndarray:
                nphase = int(params["num_phases"])
                ncomp = int(params["num_components"])
                N2 = int(params["N2"])
                tol = params["atol"]

                T_cs = np.empty(ncomp)
                p_cs = np.empty(ncomp)
                v_cs = np.empty(ncomp)
                omegas = np.empty(ncomp)
                for i in range(ncomp):
                    T_cs[i] = params[f"_T_crit_{i}"]
                    p_cs[i] = params[f"_p_crit_{i}"]
                    v_cs[i] = params[f"_v_crit_{i}"]
                    omegas[i] = params[f"_omega_{i}"]

                for k in nb.prange(X_gen.shape[0]):
                    Xk = X_gen[k]

                    s, x, y, z, p, T, s1, s2, x_p = parse_generic_arg(
                        Xk, ncomp, nphase, FlashSpec.ph
                    )
                    # NOTE local copy for simplicity of compilation.
                    p_cs_ = p_cs.copy()
                    T_cs_ = T_cs.copy()
                    v_cs_ = v_cs.copy()
                    omegas_ = omegas.copy()

                    # Compute pseudo-critical estimate of enthalpy.
                    T_pc = (z * T_cs_).sum()
                    p_pc = critical_pressure_guess(z, p_cs_, T_cs_, v_cs_)

                    pre_g_pc = prearg_val_c(PhysicalState.gas, p_pc, T_pc, z, x_p)
                    pre_l_pc = prearg_val_c(PhysicalState.liquid, p_pc, T_pc, z, x_p)
                    h_g_pc = h_c(pre_g_pc, p_pc, T_pc, z)
                    h_l_pc = h_c(pre_l_pc, p_pc, T_pc, z)
                    # Pseudo-critical estimates are not exact, hence these two can
                    # differ slightly. Take average and obtain pseudo-critical h.
                    h_pc = (h_g_pc + h_l_pc) * 0.5

                    itr_gas = False
                    itr_liq = False

                    # We now refine the T guess by dividing the ph plane.
                    # Above the pseudo-critical pressure, we iterate over the enthalpy
                    # constraint. If we are left of the h_pc, use h_liq, otherwise use
                    # h_gas.
                    if p >= p_pc:
                        T = T_pc  # Start with pseudo-critical T.
                        if s2 < h_pc:
                            itr_liq = True
                        else:
                            itr_gas = True
                    # Below p_pc, approximate bubble and dew-point temperature, and
                    # compute enthalpies at points. If we are left of h_bub, iterate
                    # using h_liq, if we are right of h_dew iterate using h_liq.
                    # If we are in between, interpolate T and not refine anymore.
                    else:
                        # NOTE Starting from pseudo-critical alone is often unstable.
                        T0 = (T_REF + T_pc) * 0.5
                        T_dew = get_dew_point_T(T0, p, z, p_cs_, T_cs_, omegas_)
                        T_bub = get_bubble_point_T(T_dew, p, z, p_cs_, T_cs_, omegas_)

                        # Compute enthalpies at points.
                        pre_g_dew = prearg_val_c(PhysicalState.gas, p, T_dew, z, x_p)
                        h_dew = h_c(pre_g_dew, p, T_dew, z)
                        pre_l_bub = prearg_val_c(PhysicalState.liquid, p, T_bub, z, x_p)
                        h_bub = h_c(pre_l_bub, p, T_bub, z)

                        if s2 > h_dew:  # Clearly gas-like.
                            T = T_dew
                            itr_gas = True
                        elif s2 < h_bub:  # Clearly liquid-like.
                            T = T_bub
                            itr_liq = True
                        else:  # If not clear, interpolate between bubble and dew point.
                            w = np.abs(s2 - h_bub) / np.abs(h_dew - h_bub)
                            T = (1.0 - w) * T_bub + w * T_dew

                    if itr_gas or itr_liq:
                        if itr_gas:
                            ps = PhysicalState.gas
                        else:
                            ps = PhysicalState.liquid
                        T_r = T_cs_.max()

                        # Simple Newton on energy constraint.
                        # Single-phase is always well-behaved.
                        for i in range(N2):
                            pre_v_k = prearg_val_c(ps, p, T, z, x_p)
                            h_i = h_c(pre_v_k, p, T, z)

                            r_i = h_i / s2 - 1.0  # residual energy constraint
                            if np.abs(r_i) < tol:
                                break

                            pre_j_k = prearg_jac_c(pre_v_k, p, T, z, x_p)
                            dhdT_i = dh_c(pre_v_k, pre_j_k, p, T, z)[1]

                            J_i = dhdT_i / s2 * T_r  # Scaled derivative.
                            dT = -r_i / J_i  # Newton step.
                            if dT in (np.nan, np.inf, -np.inf):  # Fail -> steep. desc.
                                dT = -r_i
                            dT = np.sign(dT) * min(np.abs(dT), 0.1) * T_r  # Chop.
                            T += dT

                    Xk = assemble_generic_arg(
                        s, x, y, z, p, T, s1, s2, x_p, FlashSpec.ph
                    )
                    X_gen[k] = fractions_from_rr(
                        get_K_values, Xk, params, FlashSpec.ph, True
                    )
                return X_gen

            self._initializers[FlashSpec.ph] = ph_init

        if FlashSpec.vh in args and FlashSpec.vh not in self._initializers:
            logger.debug("Compiling vh-initialization ..")

            @_COMPILER(nb.f8[:](nb.f8[:], SOLVER_PARAMETERS_TYPE))
            def update_pT_guess(
                X_gen: np.ndarray, params: dict[str, float]
            ) -> np.ndarray:
                """Helper function to update pT guess for vh flash by solving
                respective equations using Newton and some corrections."""

                # Parsing parameters
                N2 = int(params["N2"])
                tol = params["atol"]
                gas_phase_idx = int(params["gas_phase_index"])

                # Local system size.
                M = 2 + nphase - 1

                res = np.zeros(M)
                jac = np.zeros((M, M))

                # s1 and s2 are target volume and enthalpy respectively
                s, x, y, z, p, T, s1, s2, x_p = parse_generic_arg(
                    X_gen, ncomp, nphase, FlashSpec.vh
                )

                # Assume no guess, fetch later if otherwise.
                y_g = 0.0

                # If no p or T value are provided at all, create initial guess using
                # pseudo-critical values.
                if p == 0.0 or T == 0.0:
                    T_cs = np.empty(ncomp)
                    v_cs = np.empty(ncomp)
                    p_cs = np.empty(ncomp)
                    for i in range(ncomp):
                        T_cs[i] = params[f"_T_crit_{i}"]
                        v_cs[i] = params[f"_v_crit_{i}"]
                        p_cs[i] = params[f"_p_crit_{i}"]
                    # pseudo_critical T_guess
                    T = np.dot(z, T_cs)

                    # pseudo-critical pressure guess
                    v_pc = cubic_mix(z, v_cs)
                    p = critical_pressure_guess(z, p_cs, T_cs, v_cs)
                    # Pseudo-critical compressibility factor.
                    Z_pc = np.dot(z, p_cs * v_cs / T_cs) / R_U

                    # Refining pressure and temperature guess based on ratio of
                    # pseudo-critical volume and given volume.
                    R = v_pc / s1
                    if R > 1:  # liquid-like
                        p *= 0.2 / Z_pc
                        T = T / np.sqrt(R)
                    else:  # gas-like
                        p *= 0.7 / Z_pc

                    # Make first fraction guess based on pseudo-critical values.
                    xf = assemble_generic_arg(
                        s, x, y, z, p, T, s1, s2, x_p, FlashSpec.vh
                    )
                    xf = fractions_from_rr(get_K_values, xf, params, FlashSpec.vh, True)
                    s, x, y, z, p, T, s1, s2, x_p = parse_generic_arg(
                        xf, ncomp, nphase, FlashSpec.vh
                    )

                    # Correct pressure if no gas phase
                    if gas_phase_idx >= 0:
                        y_g = y[gas_phase_idx]

                    if y_g < 1e-3:
                        p *= 0.7
                        # T *= 1.1
                        # Refine fraction guess based on corrected pressure.
                        xf = assemble_generic_arg(
                            s, x, y, z, p, T, s1, s2, x_p, FlashSpec.vh
                        )
                        xf = fractions_from_rr(
                            get_K_values, xf, params, FlashSpec.vh, False
                        )
                        s, x, y, z, p, T, s1, s2, x_p = parse_generic_arg(
                            xf, ncomp, nphase, FlashSpec.vh
                        )

                xn = normalize_rows(x)
                if gas_phase_idx >= 0:
                    y_g = y[gas_phase_idx]

                rhos = np.empty(nphase)
                hs = np.empty(nphase)
                dhs = np.empty((nphase, 2 + ncomp))
                drhos = np.empty((nphase, 2 + ncomp))

                for _ in range(N2):
                    # Assembling volume and enthalpy constraints with derivatives for
                    # s-pT.

                    for j in range(nphase):
                        pre_val_j = prearg_val_c(phasestates[j], p, T, xn[j], x_p)
                        pre_jac_j = prearg_jac_c(pre_val_j, p, T, xn[j], x_p)
                        rhos[j] = rho_c(pre_val_j, p, T, xn[j])
                        drhos[j] = drho_c(pre_val_j, pre_jac_j, p, T, xn[j])
                        hs[j] = h_c(pre_val_j, p, T, xn[j])
                        dhs[j] = dh_c(pre_val_j, pre_jac_j, p, T, xn[j])

                    # Saturations are only used locally, hence we refer to sat, not s
                    # which is in the generic arg.
                    sat = _compute_saturations(y, rhos, 1e-10)
                    v_mix = 1.0 / (sat * rhos).sum()
                    h_mix = (y * hs).sum()

                    res[0] = first_order_constraint_res(s2, y, hs)[0] / s2  # / T**2
                    res[1] = first_order_constraint_res(1.0, sat, s1 * rhos)[0]
                    res[2:] = phase_mass_constraints_res(sat, y, rhos)

                    jac[0] = first_order_constraint_jac(y, hs, dhs, True)[0, :M]
                    # jac[0, 1] -= 2 / T * res[0]
                    jac[0] /= s2
                    jac[1] = (
                        first_order_constraint_jac(sat, rhos, drhos, False)[0, :M] * s1
                    )
                    jac[2:] = phase_mass_constraints_jac(sat, y, rhos, drhos)[:, :M]

                    if np.linalg.norm(res) <= tol:
                        break
                    else:
                        dspT = np.linalg.solve(jac, -res)

                        # update corrections
                        dp = dspT[-2]
                        dT = dspT[-1]
                        if np.abs(dT) > T:
                            dT = 0.1 * T * np.sign(dT)
                        if np.abs(dp) > p:
                            dp = 0.2 * p * np.sign(dp)

                        fp = 1 - np.abs(dp) / p
                        fT = 1 - np.abs(dT) / T

                        # give preferance to pressure update if gas present and volume
                        # too large
                        if y_g > 1e-3 and v_mix > s1:
                            # volume contraction only by positive p update, not neg. T
                            if dT < 0.0:
                                dT = 0.0
                            # unfeasible update (should compress if v_mix bigger than v)
                            if dp < 0.0:
                                dp = 0.0

                        T_ = T + fT * dT
                        p_ = p + fp * dp

                        # correction for gas-like mixture and volume too large,
                        # increase p significantly
                        if y_g >= 1.0 and v_mix > s1:
                            p_ *= 2 - fp
                        # correction for liquid-like mixtures, h is very sensitive to p
                        # because h = u + pv, v small (liquid)
                        # then cancel the update
                        if y_g < 1e-1 and h_mix < s2:  # and p_ > p:
                            p_ *= 1.1

                        p = p_
                        T = T_

                return assemble_generic_arg(s, x, y, z, p, T, s1, s2, x_p, FlashSpec.vh)

            def vh_init(X_gen: np.ndarray, params: dict[str, float]):
                X_gen = nested_initializer(
                    get_K_values, update_pT_guess, FlashSpec.vh, X_gen, params
                )
                # Performing final saturation update, after guessing fractions and p,T
                s, x, y, z, p, T, s1, s2, x_p = parse_vectorized_generic_arg(
                    X_gen, ncomp, nphase, FlashSpec.vh
                )
                rhos = np.empty(y.shape)
                for j in range(nphase):
                    x_j = x[j, :, :]
                    # NOTE accessing the gufuncs directly require the fractions
                    # column-wise.
                    xn = normalize_rows(x_j.T)
                    pre = self._eos.gufuncs["prearg_val"](
                        phasestates[j], p, T, xn, x_p.T
                    )
                    rhos[j] = self._eos.gufuncs["rho"](pre, p, T, xn)
                s = compute_saturations(y, rhos)

                return assemble_vectorized_generic_arg(
                    s, x, y, z, p, T, s1, s2, x_p, FlashSpec.vh
                )

            self._initializers[FlashSpec.vh] = vh_init

        if FlashSpec.vu in args and FlashSpec.vu not in self._initializers:
            logger.debug("Compiling vu-initialization ..")

            @_COMPILER(nb.f8[:](nb.f8[:], SOLVER_PARAMETERS_TYPE))
            def update_pT_guess_saha(
                X_gen: np.ndarray, params: dict[str, float]
            ) -> np.ndarray:
                """Helper function to update pT guess for vh flash by solving
                respective equations using Newton and some corrections."""

                # Parsing parameters
                gas_phase_idx = int(params["gas_phase_index"])

                # s1 and s2 are target volume and enthalpy respectively
                s, x, y, z, p, T, s1, s2, x_p = parse_generic_arg(
                    X_gen, ncomp, nphase, FlashSpec.vu
                )

                # Assume no gas, fetch later if otherwise.
                y_g = 0.0

                # If no p or T value are provided at all, create initial guess using
                # pseudo-critical values
                if p == 0.0 or T == 0.0:
                    T_cs = np.empty(ncomp)
                    v_cs = np.empty(ncomp)
                    p_cs = np.empty(ncomp)
                    for i in range(ncomp):
                        T_cs[i] = params[f"_T_crit_{i}"]
                        v_cs[i] = params[f"_v_crit_{i}"]
                        p_cs[i] = params[f"_p_crit_{i}"]
                    # pseudo_critical T_guess
                    T = np.dot(z, T_cs)

                    # pseudo-critical pressure guess
                    v_pc = cubic_mix(z, v_cs)
                    p = critical_pressure_guess(z, p_cs, T_cs, v_cs)
                    # Pseudo-critical compressibility factor.
                    Z_pc = np.dot(z, p_cs * v_cs / T_cs) / R_U

                    # Refining pressure and temperature guess based on ratio of
                    # pseudo-critical volume and given volume.
                    R = v_pc / s1
                    if R > 1:  # liquid-like
                        p *= 0.2 / Z_pc
                        T = T / np.sqrt(R)
                    else:  # gas-like
                        p *= 0.7 / Z_pc

                    # Make first fraction guess based on pseudo-critical values.
                    xf = assemble_generic_arg(
                        s, x, y, z, p, T, s1, s2, x_p, FlashSpec.vu
                    )
                    xf = fractions_from_rr(get_K_values, xf, params, FlashSpec.vu, True)
                    s, x, y, z, p, T, s1, s2, x_p = parse_generic_arg(
                        xf, ncomp, nphase, FlashSpec.vu
                    )

                    # Correct pressure if no gas phase
                    if gas_phase_idx >= 0:
                        y_g = y[gas_phase_idx]

                    if y_g < 1e-3:
                        p *= 0.7
                        T *= 1.1
                        # Refine fraction guess.
                        xf = assemble_generic_arg(
                            s, x, y, z, p, T, s1, s2, x_p, FlashSpec.vu
                        )
                        xf = fractions_from_rr(
                            get_K_values, xf, params, FlashSpec.vu, False
                        )
                        s, x, y, z, p, T, s1, s2, x_p = parse_generic_arg(
                            xf, ncomp, nphase, FlashSpec.vu
                        )

                xn = normalize_rows(x)
                if gas_phase_idx >= 0:
                    y_g = y[gas_phase_idx]

                us = np.empty(nphase)
                dus = np.empty((nphase, 2 + ncomp))
                rhos = np.empty(nphase)
                drhos = np.empty((nphase, 2 + ncomp))

                for j in range(nphase):
                    pre_val_j = prearg_val_c(phasestates[j], p, T, xn[j], x_p)
                    pre_jac_j = prearg_jac_c(pre_val_j, p, T, xn[j], x_p)
                    us[j] = u_c(pre_val_j, p, T, xn[j])
                    dus[j] = du_c(pre_val_j, pre_jac_j, p, T, xn[j])
                    rhos[j] = rho_c(pre_val_j, p, T, xn[j])
                    drhos[j] = drho_c(pre_val_j, pre_jac_j, p, T, xn[j])

                s = _compute_saturations(y, rhos, 1e-10)
                rho_mix = np.dot(s, rhos)
                outer = -1.0 / rho_mix**2
                v_new = 1.0 / rho_mix
                dv_new_dp = outer * np.dot(s, drhos[:, 0])
                dv_new_dT = outer * np.dot(s, drhos[:, 1])

                u_new = np.dot(y, us) - p * v_new
                du_new_dT = np.dot(y, dus[:, 1]) - p * dv_new_dT

                dT = (s2 - u_new) / du_new_dT
                fc = 1 - np.abs(dT) / T

                if y_g > 1e-3:
                    if y_g > 1.0 - 1e-8:
                        fc = 1.0
                    p += fc * dT * dv_new_dT / np.abs(dv_new_dp)
                else:
                    p *= 2.0 - fc

                T += fc * dT

                return assemble_generic_arg(s, x, y, z, p, T, s1, s2, x_p, FlashSpec.vu)

            def vu_init(X_gen: np.ndarray, params: dict[str, float]):
                X_gen = nested_initializer(
                    get_K_values, update_pT_guess_saha, FlashSpec.vu, X_gen, params
                )
                # Performing final saturation update, after guessing fractions and p,T
                s, x, y, z, p, T, s1, s2, x_p = parse_vectorized_generic_arg(
                    X_gen, ncomp, nphase, FlashSpec.vu
                )
                rhos = np.empty(y.shape)
                for j in range(nphase):
                    x_j = x[j, :, :]
                    # NOTE accessing the gufuncs directly require the fractions
                    # column-wise.
                    xn = normalize_rows(x_j.T)
                    pre = self._eos.gufuncs["prearg_val"](
                        phasestates[j], p, T, xn, x_p.T
                    )
                    rhos[j] = self._eos.gufuncs["rho"](pre, p, T, xn)
                s = compute_saturations(y, rhos)

                return assemble_vectorized_generic_arg(
                    s, x, y, z, p, T, s1, s2, x_p, FlashSpec.vu
                )

            self._initializers[FlashSpec.vu] = vu_init

        logger.info(
            "Flash initialization routines compiled"
            + " (elapsed time: %.4f (s))." % (time.time() - start)
        )
