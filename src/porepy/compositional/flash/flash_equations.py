"""Module containing compiled assembly of equations which are part of the equilibrium
problem, as well as the parsing and assembly of the generic argument for all flash
configurations.

The structure of the generic argument is as follows:

*(params, overall fractions, target state 1, target state 2, pressure, temperature,
saturations, phase fractions, partial fractions in phase 1, ... , partial fractions in
phase n)*

This is the most general layout, reflecting especially the order of values in the
array.

This layout is adapted to individual flash specifications to avoid redundant entries.

The generic argument formulation enables us to formulate any flash system as a function
``F(X_gen)``, and hence solve it with mathematical means.

Most importantly, the order of the elements in the generic argument reflects the order
of derivatives (columns) in the Jacobian ``DF(X_gen)``.

Various ``*_jac``-functions returning partial Jacobians of the unified flash system
always return the full Jacobian w.r.t. to **all** possible dependencies:

*(pressure, temperature, saturations, phase fractions, partial fractions)*

Individual flash systems must assemble the partial Jacobians they need and slice them.

The rows in every flash system are expected to be of a particular order:

1. Local mass constraints (``num_components - 1``)
2. Local energy and/or volume constraints (1+)
3. Isofugacity equations (``num_components * (num_phases - 1)``)
4. Complementary conditions (``num_phases``).

Follow this pattern for maximum compatibility when assembling flash systems.

"""

from __future__ import annotations

from typing import Optional, Sequence

import numba as nb
import numpy as np

from .._numba_interface import NUMBA_CACHE, NUMBA_FAST_MATH, NUMBA_PARALLEL, njit
from ..utils import FlashSpec, FlashSpec_NUMBA_TYPE
from .abstract_flash import FlashResults

__all__ = [
    "generic_arg_from_flash_results",
    "dim_gen_arg",
    "parse_xy",
    "parse_generic_arg",
    "assemble_generic_arg",
    "parse_vectorized_generic_arg",
    "assemble_vectorized_generic_arg",
    "mass_constraint_res",
    "mass_constraint_jac",
    "complementary_conditions_res",
    "complementary_conditions_jac",
    "isofugacity_constraints_res",
    "isofugacity_constraints_jac",
    "first_order_constraint_res",
    "first_order_constraint_jac",
]


_COMPILER = njit
"""Decorator for compiling functions in this module."""


def generic_arg_from_flash_results(
    results: FlashResults,
    ncomp: int,
    nphase: int,
    state_is_initialized: bool = False,
    params: Optional[Sequence[np.ndarray | float] | float | np.ndarray] = None,
) -> np.ndarray:
    """Assembles a generic argument from a given flash results data structure.

    ``results`` must at least contained the fields indicated by its flash specification.

    See also:
        :func:`assemble_generic_arg`, :func:`assemble_vectorized_generic_arg`

    Parameters:
        results: Flash result data structure with valid flash specification and
            equilibrium state values.
        ncomp: Number of components.
        nphase: Number of phases.
        state_is_initialized: If True, values for partial fractions and other degrees of
            freedom according to the flash specification are extracted from ``results``
            as well. Otherwise they are instantiated with zero.
        params: Parameter array.

    Returns:
        The generic argument as returned by :func:`assemble_vectorized_generic_arg`.

    """
    spec = results.specification
    N = results.size

    if params is None:
        params = np.zeros((0, N))
    elif isinstance(params, float):
        params = np.ones((1, N)) * params
    elif isinstance(params, np.ndarray):
        if params.ndim == 1:
            params = params.reshape((1, params.size))
        else:
            assert params.ndim == 2, "Parameter array must be 1D or 2D."
            assert params.shape[1] == N, (
                "Parameter array must have shape (n_params, N) with N = number of "
                "flash problems."
            )
    else:  # Assume sequence
        Np = len(params)
        param_array = np.empty((Np, N))
        for i, p in enumerate(params):
            param_array[i, :] = p
        params = param_array.astype(np.float64)

    assert isinstance(params, np.ndarray), 'Failure to convert "params" to array.'
    # Target states depending on flash type.
    z = results.z
    assert z.shape == (ncomp, N), "Overall compositions of unexpected shape."
    state1: np.ndarray = getattr(results, results.specification.name[0])
    state2: np.ndarray = getattr(results, results.specification.name[1])

    if (spec >= FlashSpec.vT and state_is_initialized) or spec < FlashSpec.vT:
        p = results.p
    else:
        p = np.zeros(N)
    if (spec not in [FlashSpec.pT, FlashSpec.vT] and state_is_initialized) or spec in [
        FlashSpec.pT,
        FlashSpec.vT,
    ]:
        T = results.T
    else:
        T = np.zeros(N)

    if state_is_initialized:
        y = results.y
        x = np.array([phase.x for phase in results.phases])
        assert x.shape == (nphase, ncomp, N), "Partial fractions of unexpected shape."
    else:
        y = np.zeros((nphase, N))
        x = np.zeros((nphase, ncomp, N))

    return assemble_vectorized_generic_arg(x, y, z, p, T, state1, state2, params, spec)


@_COMPILER(
    nb.int_(nb.int_, nb.int_, FlashSpec_NUMBA_TYPE),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def dim_gen_arg(ncomp: int, nphase: int, spec: FlashSpec) -> int:
    """Returns the base dimension (no parameters) of the generic flash argument for
    a specified flash type.

    Parameters:
        ncomp: Number of components.
        nphase: Number of phases.
        spec: The Flash specification in terms of target state functions.

    Returns:
        ``n`` where the generic argument for a single flash problem has shape ``(n,)``,
        assuming no parameters are stored within.

    """
    # Number of independent phases.
    n_P1m = nphase - 1

    # Base dimension is for all the same.
    # NOTE: Pressure and temperature could be the same as the target state for many
    # of the state definitions, but this simplifies
    d = (
        n_P1m  # Number of independent phase fractions.
        + nphase * ncomp  # Number of partial fractions.
        + ncomp
        - 1  # Independent overall compositions.
        + 2  # Pressure and temperature.
    )
    if spec == FlashSpec.none:
        raise ValueError("Dimension not determinable if flash not specified.")

    # If it is isobaric and T is not among the target states, we need 1 more target
    # state value related to energy
    if FlashSpec.pT < spec < FlashSpec.vT:
        d += 1
    # If isochoric specifications, volume values are part of the generic argument.
    if spec >= FlashSpec.vT:
        d += 1
    # If isochoric and not isothermal, we need an additional energy-related state
    # variable.
    if spec > FlashSpec.vT:
        d += 1

    return d


@_COMPILER(
    nb.types.Tuple((nb.f8[:, :], nb.f8[:]))(nb.f8[:], nb.int_, nb.int_),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def parse_xy(
    X_gen: np.ndarray, ncomp: int, nphase: int
) -> tuple[np.ndarray, np.ndarray]:
    """Helper function to extract phase compositions and fractions from generic
    argument.

    Parameters:
        Xgen: Generic argument, shape at least ``(nphase * ncomp + nphase,)``.
        ncomp: Number of components.
        nphase: Number of phases.

    Returns:
        Tuple containing:

        - Phase compositions, shape ``(nphase, ncomp)``.
        - Phase fractions, shape ``(nphase,)``.

    """
    n_PC = nphase * ncomp
    x = X_gen[-n_PC:].copy().reshape((nphase, ncomp))
    # Phase fractions
    y = np.zeros(nphase)
    y[1:] = X_gen[-(n_PC + nphase - 1) : -n_PC]
    y[0] = 1.0 - y.sum()

    return x, y


@_COMPILER(
    nb.types.Tuple(
        (
            nb.f8[:, :],
            nb.f8[:],
            nb.f8[:],
            nb.f8,
            nb.f8,
            nb.f8,
            nb.f8,
            nb.f8[:],
        )
    )(nb.f8[:], nb.int_, nb.int_, FlashSpec_NUMBA_TYPE),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def parse_generic_arg(
    X_gen: np.ndarray, ncomp: int, nphase: int, spec: FlashSpec
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    np.ndarray,
]:
    """Parses the generic flash argument and returns the unknowns and parameters of
    the flash problem.

    Parameters:
        X_gen: Generic flash argument (1D array).
        ncomp: Number of components.
        nphase: Number of phases.
        spec: The Flash specification in terms of target state functions.

    Returns:
        A tuple containing

        1. 2D array of (extended) partial fractions, row-wise per phase,
        2. 1D array of phase fractions,
        3. 1D array of overall fractions,
        4. pressure value,
        5. temperature value,
        6. first target state according to ``spec``,
        7. second target state according to ``spec``,
        8. other parameters stored in the generic argument.

        The first and second target state can coincide with the pressure or temperature
        value, if pressure or temperature are defined as target state values in
        ``spec``.

        All fractions contain values for reference phase and component. They are
        always stored as the first value.

        If no parameters are stored, the parameter array is a zero array.

    """
    if spec == FlashSpec.none:
        raise ValueError("No safe parsing possible if flash not specified.")

    y = np.zeros(nphase)
    z = np.zeros(ncomp)
    x = np.zeros((nphase, ncomp))

    # The last nphase * ncomp values are the extended partial fractions
    i = nphase * ncomp  # Keeping track of accessed indices (from back to front).
    # NOTE Numba requires a contiguous array for reshape, which is created with copy.
    x = X_gen[-i:].copy().reshape((nphase, ncomp))

    # Phase fractions
    y[1:] = X_gen[-(i + nphase - 1) : -i]
    y[0] = 1.0 - y.sum()
    i += nphase - 1

    # pressure and temperature are always the last (seen from back) unknowns.
    p, T = X_gen[-(i + 2) : -i]
    i += 2

    # Now come the state definitions, where the indexing is flash-type-specific.
    # Isobaric
    if spec < FlashSpec.vT:
        state1 = p
        # Non-isothermal, additional state value expected.
        if FlashSpec.pT < spec:
            state2 = X_gen[-(i + 1)]
            i += 1
        # Isothermal, no additional values
        else:
            state2 = T
    # Isochoric, volume is an additional state value in the generic argument
    else:
        state1 = X_gen[-(i + 1)]
        i += 1
        # Non-isothermal, additional state value expected
        if spec > FlashSpec.vT:
            state2 = X_gen[-(i + 1)]
            i += 1
        # Isothermal, temperature is already contained.
        else:
            state2 = T

    # The final standard elements of the generic argument are the independent overall
    # compositions.
    z[1:] = X_gen[-(i + ncomp - 1) : -i]
    z[0] = 1.0 - z.sum()
    i += ncomp - 1

    # Other parameters, if any.
    params = X_gen[:-i]

    # Sanity check, avoid accessing random memory in compiled version.
    assert X_gen.shape[0] == i + params.shape[0], (
        f"Parsing generic argument failed with specification {spec}."
    )

    return x, y, z, p, T, state1, state2, params


@_COMPILER(
    nb.f8[:](
        nb.f8[:, :],
        nb.f8[:],
        nb.f8[:],
        nb.f8,
        nb.f8,
        nb.f8,
        nb.f8,
        nb.f8[:],
        FlashSpec_NUMBA_TYPE,
    ),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def assemble_generic_arg(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    p: float,
    T: float,
    state1: float,
    state2: float,
    params: np.ndarray,
    spec: FlashSpec,
) -> np.ndarray:
    """Inverse operation of :func:`parse_generic_arg`.

    Note:
        Though not practical, every potential part of the generic argument has to be
        passed for every flash type. This is due to the static signature of this
        numba function, otherwise the compilation and usage will become less efficient
        and clear.

        It also makes this (and the parser) usable for any flash type.

    Parameters:
        x: ``shape=(num_phases, num_components)``

            (Extended) partial fractions per phase.
        y: ``shape=(num_phases,)``

            Phase fractions.
        z: ``shape=(num_components,)``

            Overall component fractions.
        p: Pressure value.
        T: Temperature value.
        state1: First target state (f.e. pressure in ph flash).
        state2: Second target state (f.e. enthalpy in ph flash).
        params: ``shape=(n,)``

            Vector of other parameters.
        spec: The Flash specification in terms of target state functions.

    Returns:
        The generic argument corresponding to the ``spec``.

    """
    nphase, ncomp = x.shape

    ## Allocating parts of the generic argument.
    # Fractions which are always unknowns.
    X_gen_yx = np.zeros(nphase - 1 + nphase * ncomp)
    X_gen_yx[: nphase - 1] = y[1:]
    X_gen_yx[nphase - 1 :] = x.copy().reshape((nphase * ncomp,))

    # Keeping track of the size.
    i = nphase - 1 + nphase * ncomp

    # Pressure and temperature values.
    X_gen_pT = np.array([p, T])
    i += 2

    # Other state definitions.
    # Isobaric, non-isothermal.
    if FlashSpec.pT < spec < FlashSpec.vT:
        X_gen_state = np.ones(1) * state2
        i += 1
    # Isochoric.
    elif FlashSpec.vT <= spec:
        # Non-isothermal.
        if FlashSpec.vT < spec:
            X_gen_state = np.array([state2, state1])
            i += 2
        # Isothermal.
        else:
            X_gen_state = np.array([state1])
            i += 1
    # Isobaric, isothermal: No additional state value required
    else:
        X_gen_state = np.zeros((0,))

    # Finally, overall fractions.
    if ncomp > 1:
        X_gen_z = z[1:]
    else:
        X_gen_z = np.zeros((0,))
    i += ncomp - 1

    # Create generic argument.
    X_gen = np.hstack((params, X_gen_z, X_gen_state, X_gen_pT, X_gen_yx))

    # Sanity check.
    assert X_gen.shape[0] == i + params.shape[0]

    return X_gen


@_COMPILER(
    nb.types.Tuple(
        (
            nb.f8[:, :, :],
            nb.f8[:, :],
            nb.f8[:, :],
            nb.f8[:],
            nb.f8[:],
            nb.f8[:],
            nb.f8[:],
            nb.f8[:, :],
        )
    )(nb.f8[:, :], nb.int_, nb.int_, FlashSpec_NUMBA_TYPE),
    fastmath=NUMBA_FAST_MATH,
    parallel=NUMBA_PARALLEL,
    cache=NUMBA_CACHE,
)
def parse_vectorized_generic_arg(
    X_gen: np.ndarray, ncomp: int, nphase: int, spec: FlashSpec
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Parallelized version of :func:`parse_generic_arg` for vectorized input.

    Parsing is performed over the rows of ``X_gen``.

    """
    n, m = X_gen.shape

    x = np.empty((nphase, ncomp, n), dtype=np.float64)
    y = np.empty((nphase, n), dtype=np.float64)
    z = np.empty((ncomp, n), dtype=np.float64)
    p = np.empty((n,), dtype=np.float64)
    T = np.empty((n,), dtype=np.float64)
    state1 = np.empty((n,), dtype=np.float64)
    state2 = np.empty((n,), dtype=np.float64)

    # Fetching number of paramters stored.
    dim_params = m - dim_gen_arg(ncomp, nphase, spec)
    params = np.empty((dim_params, n), dtype=np.float64)

    for i in nb.prange(n):
        x_i, y_i, z_i, p_i, T_i, state1_i, state2_i, x_p_i = parse_generic_arg(
            X_gen[i], ncomp, nphase, spec
        )

        x[:, :, i] = x_i
        y[:, i] = y_i
        z[:, i] = z_i
        p[i] = p_i
        T[i] = T_i
        state1[i] = state1_i
        state2[i] = state2_i
        params[:, i] = x_p_i

    return x, y, z, p, T, state1, state2, params


@_COMPILER(
    nb.f8[:, :](
        nb.f8[:, :, :],
        nb.f8[:, :],
        nb.f8[:, :],
        nb.f8[:],
        nb.f8[:],
        nb.f8[:],
        nb.f8[:],
        nb.f8[:, :],
        FlashSpec_NUMBA_TYPE,
    ),
    parallel=NUMBA_PARALLEL,
    fastmath=NUMBA_FAST_MATH,
    cache=NUMBA_CACHE,
)
def assemble_vectorized_generic_arg(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    p: np.ndarray,
    T: np.ndarray,
    state1: np.ndarray,
    state2: np.ndarray,
    params: np.ndarray,
    spec: FlashSpec,
) -> np.ndarray:
    """Parallelized version of :func:`assemble_generic_arg` for vectorized input.

    Assembly is performed such that 1 row in the return value represents one generic
    flash argument.

    """
    ncomp = z.shape[0]
    nphase = y.shape[0]
    n = p.shape[0]
    n_param = params.shape[0]

    d = dim_gen_arg(ncomp, nphase, spec)
    X_gen = np.empty((n, d + n_param), dtype=np.float64)

    for i in nb.prange(n):
        X_gen[i] = assemble_generic_arg(
            x[:, :, i],
            y[:, i],
            z[:, i],
            p[i],
            T[i],
            state1[i],
            state2[i],
            params[:, i],
            spec,
        )
    return X_gen


@_COMPILER(
    nb.f8[:](nb.f8[:, :], nb.f8[:], nb.f8[:]),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def mass_constraint_res(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    r"""Assembles the residual of the mass conservation equations.

    For each component ``i``, except reference component, it holds

    ... math::

        \sum_j y_j x_{ij}  - z_i = 0.

    Number of phases and components is determined from the chape of ``x``.

    Note:
        In the 1-component case, the mass conservation equation can be obtained by
        summing the complementarity conditions and applying unity of fractions.

        In the multicomponent case, the mass conservation of the reference component can
        also be obtained by summation of complementarity conditions and other mass
        conservation equations.

        It is hence in all cases an redundant equation for the reference component, and
        the respective result is implemented as an empty array.

    Parameters:
        x: ``shape=(num_phases, num_components)``

            (Extended) partial fractions per phase.
        y: ``shape=(num_phases,)``

            Phase fractions.
        z: ``shape=(num_components,)``

            Overall fractions per component.

    Returns:
        An array with ``shape=(num_components - 1,)`` containing the residual of the
        mass conservation equation (left-hand side of above equation) for each
        component, except the first one (in ``z``).

    """
    if z.size > 1:
        return (np.dot(y, x) - z)[1:]
    else:
        return np.empty(0)


@_COMPILER(
    nb.f8[:, :](nb.f8[:, :], nb.f8[:]),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def mass_constraint_jac(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Returns the Jacobian of the residual described in
    :func:`mass_constraint_res`.

    Derivatives are computed w.r.t. independent phase fractions and extended partial
    fractions, where the order of derivatives for the latter is phase-major.
    The first phase fraction is assumed dependent.

    Parameters:
        x: ``shape=(num_phases, num_components)``

            (Extended) partial fractions per phase.
        y: ``shape=(num_phases,)``

            Phase fractions.

    Returns:
        The Jacobian of shape ``(num_components - 1, 2 + 2 * (num_phases - 1) +
        num_phases * num_components)``.

    """
    n_P, n_C = x.shape
    n_P1p = n_P + 1  # Independent phases and p, T.

    # Must fill with zeros, since slightly sparse and below fill-up does not cover
    # elements which are zero.
    jac = np.zeros((n_C - 1, n_P1p + n_P * n_C), dtype=np.float64)

    for i in range(n_C - 1):
        # (1 - sum_j y_j) x_ir + y_j x_ij is there, per phase.
        # Hence d mass_i / d y_j = x_ij - x_ir
        # i + 1 to skip ref component.
        jac[i, 2:n_P1p] = x[1:, i + 1] - x[0, i + 1]

        # d.r.t. w.r.t x_ij is always y_j for all j per mass conv.
        jac[i, 2 + n_P + i :: n_C] = y  # nphase -1 + i + 1 to skip ref component

    return jac


@_COMPILER(
    nb.f8[:](nb.f8[:, :], nb.f8[:]),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def complementary_conditions_res(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""Assembles the residual of the complementary conditions.

    For each phase ``j`` it holds

    ... math::

        y_j \cdot \left(1 - \sum_i x_{ij}\right) = 0.

    Number of phases and components is determined from the chape of ``x``.

    Parameters:
        x: ``shape=(num_phases, num_components)``

            (Extended) partial fractions per phase.
        y: ``shape=(num_phases,)``

            Phase fractions.

    Returns:
        An array with ``shape=(num_phases,)`` containing the residual of the
        complementary condition per phase.

    """
    return y * (1 - np.sum(x, axis=1))


@_COMPILER(
    nb.f8[:, :](nb.f8[:, :], nb.f8[:]),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def complementary_conditions_jac(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Returns the Jacobian of the residual described in
    :func:`complementary_conditions_res`.

    Derivatives are computed w.r.t. independent phase fractions and extended partial
    fractions, where the order of derivatives for the latter is phase-major.
    The first phase fraction is assumed dependent.

    Parameters:
        x: ``shape=(num_phases, num_components)``

            (Extended) partial fractions per phase.
        y: ``shape=(num_phases,)``

            Phase fractions.

    Returns:
        The Jacobian of shape ``(num_phases, 2 + 2 * (num_phases - 1) + num_phases *
        num_components)``.

    """
    n_P, n_C = x.shape
    n_P1p = n_P + 1  # Independent phase fractions and p, T.

    jac = np.zeros((n_P, n_P1p + n_P * n_C), dtype=np.float64)

    unities = 1 - np.sum(x, axis=1)

    # first complementary condition is w.r.t. to reference phase
    # (1 - sum_j y_j) * (1 - sum_i x_i0)
    jac[0, 2:n_P1p] = -unities[0]
    jac[0, n_P1p : n_P1p + n_C] = -y[0]
    for j in range(1, n_P):
        # for the other phases, it's slightly easier since y_j * (1 - sum_i x_ij)
        jac[j, 1 + j] = unities[j]  # 2 + j - 1 to skip reference phase.
        jac[j, n_P1p + j * n_C : n_P1p + (j + 1) * n_C] = -y[j]

    return jac


@_COMPILER(
    nb.f8[:](nb.f8[:, :], nb.f8[:, :]),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def isofugacity_constraints_res(x: np.ndarray, lnphis: np.ndarray) -> np.ndarray:
    r"""Assembles the residual of the isofugacity constraints in log space.

    For each independent phase ``j``, and each component ``i`` it holds

    ... math::

        \log{x_{ij}} + \varphi_{ij} - \log{x_{i0}} - \varphi_{i0} = 0.

    Number of phases and components is determined from the chape of ``x``.
    The reference phase is assumed to be the first one.

    Parameters:
        x: ``shape=(num_phases, num_components)``

            (Extended) partial fractions per phase.
        lnphis: ``shape=(num_phases, num_components)``

            Logarithm of fugacity coefficients per phase and component.

    Returns:
        An array with ``shape=((num_phases - 1) * num_components,)`` containing the
        residual of the isofugacity constraints per independent phase and for each
        component.

    """
    n_P, n_C = x.shape
    res = np.zeros(n_C * (n_P - 1), dtype=np.float64)
    eps = 1e-14

    for j in range(1, n_P):
        res[(j - 1) * n_C : j * n_C] = (
            np.log(np.maximum(x[j], eps))
            + lnphis[j]
            - np.log(np.maximum(x[0], eps))
            - lnphis[0]
        )

    return res


@_COMPILER(
    nb.f8[:, :](nb.f8[:, :], nb.f8[:, :, :]),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def isofugacity_constraints_jac(x: np.ndarray, dlnphis: np.ndarray) -> np.ndarray:
    """Returns the Jacobian of the residual described in
    :func:`isofugacity_constraints_res`.

    Parameters:
        x: ``shape=(num_phases, num_components)``

            (Extended) partial fractions per phase.
        phis: ``shape=(num_phases, num_components)``

            Fugacity coefficients per phase and component.
        dlnphis: ``shape=(num_phases, num_components, 2 + num_diffs)``

            Derivatives of log-fugacity coefficients per phase and component.

    Returns:
        The Jacobian of shape ``((num_phases -1) * num_components, 2 + num_phases - 1 +
        num_phases * num_components)``.

    """
    n_P, n_C = x.shape
    n_P1m = n_P - 1  # Independent phases.
    # Allocating for pTx derivatives
    jac = np.zeros((n_C * n_P1m, 2 + n_C * n_P))
    eps = 1e-14

    # Creating block of derivatives of expression x_{i0} + phi_{i0}
    # product rule: x * dphi
    # block_0 = (dphis[0, :, :].T * x[0]).T
    block_0 = dlnphis[0, :, :]
    # + phi * dx  (minding the first two columns which contain the dp dT)
    # block_0[:, 2:] += np.diag(phis[0])
    block_0[:, 2:] += np.diag(1.0 / np.maximum(x[0], eps))

    # Loop over row blocks associated with constraints between an independent phase
    # and the reference phase, for all components.
    for j in range(1, n_P):
        # Creating block of derivatives of expression x_{ij} phi_{ij}
        # block_j = (dphis[j, :, :].T * x[j]).T
        block_j = dlnphis[j, :, :]
        # block_j[:, 2:] += np.diag(phis[j])
        block_j[:, 2:] += np.diag(1.0 / np.maximum(x[j], eps))

        # p, T derivatives
        idx = (j - 1) * n_C  # start of row block
        jac[idx : idx + n_C, :2] = block_j[:, :2] - block_0[:, :2]

        # Derivatives w.r.t. partial fractions.
        # d(x_ij * phi_ij - x_ir * phi_ir)
        # Hence every row-block associated with an independent phase contains -block_0
        jac[idx : idx + n_C, 2 : 2 + n_C] = -block_0[:, 2:]
        # Derivatives w.r.t. fractions in independent phase j
        jac[idx : idx + n_C, 2 + j * n_C : 2 + (j + 1) * n_C] = block_j[:, 2:]

    # Adding trivial columns for derivatives w.r.t. phase fractions and saturations.
    return np.hstack((jac[:, :2], np.zeros((n_C * n_P1m, n_P1m)), jac[:, 2:]))


@_COMPILER(nb.f8[:](nb.f8, nb.f8[:], nb.f8[:]), fastmath=NUMBA_FAST_MATH, cache=True)
def first_order_constraint_res(
    phi_target: float, w: np.ndarray, phis: np.ndarray
) -> np.ndarray:
    r"""Assembles the constraint of a first-order thermodynamic function

    .. math::

        \sum_j w_j \phi_j - \hat{\phi} = 0,

    where :math:`\phi_j` is some phase-related quantity, and :math:`\hat{\phi}` the
    target value of respective quantity for the fluid.

    Used to assemble the enthalpy constraint for example.

    Parameters:
        phi_target: Target value of the constraint function.
        w: ``shape=(num_phases,)``

            Phase fractions/saturations.
        phis: ``shape=(num_phases,)``

            Phase-related partial value of constrained function.

    Returns:
        The value of the left-hand-side of above equation, wrapped in an array with
        shape ``(1,)``. THe wrapping is performed for convenience since we expect this
        residual to be stacked with other equations.

    """
    return np.ones(1, dtype=np.float64) * ((w * phis).sum() - phi_target)


@_COMPILER(
    nb.f8[:, :](nb.f8[:], nb.f8[:], nb.f8[:, :]),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def first_order_constraint_jac(
    w: np.ndarray, phis: np.ndarray, dphis: np.ndarray
) -> np.ndarray:
    """Assembles the Jacobian of the first order constraint given by
    :func:`first_order_constraint_res`.

    Parameters:
        w: ``shape=(num_phases,)``

            Weights.
        phis: ``shape=(num_phases,)``

            Phase-related partial value of constrained function.
        dphis: ``shape=(num_phases, 2 + num_diffs)``

            Derivatives of phase-related partial value of constrained function.
            The derivatives must be such that the first two columns contain the
            derivatives w.r.t. to the same variables (like pressure and temperature).
            The remaining columns can be phase-related variables (like partial
            fractions), but they must be equal in number per phase (``num_diffs``).

    Returns:
        The Jacobian of shape ``(1, 2 + num_phases - 1 + num_phases * num_components)``.

    """
    n_P = w.shape[0]
    # Number of derivatives excluding p and T derivatives is equal to number of partial
    # fractions (components).
    n_C = dphis.shape[1] - 2
    n_P1p = n_P + 1  # Independent phases and p, T.

    # Allocate correct number of derivatives.
    jac = np.zeros(n_P1p + n_P * n_C, dtype=np.float64)

    # Derivatives w.r.t. p and T
    jac[:2] = (dphis[:, :2].T * w).T.sum(axis=0)
    # Derivatives w.r.t weights. Keep in mind that w_0 = 1 - w_1 - w_2 ...
    jac[2:n_P1p] = phis[1:] - phis[0]
    # Derivatives w.r.t. partial fractions per phase.
    # NOTE ().T will introduce somewhere a Fortran-order which numba cannot process.
    # This workaround using transpose().copy() will create a contiguous C-order array.
    # See https://github.com/numba/numba/issues/5433
    jac[n_P1p:] = np.transpose(dphis[:, 2:].T * w).copy().reshape((n_P * n_C,))

    # Reshaping because this is expected to be a row in a larger Jacobian.
    return jac.reshape((1, jac.shape[0]))
