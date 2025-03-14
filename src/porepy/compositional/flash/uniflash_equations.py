"""Module containing compiled assembly of equations for the unified flash, as well as
the parsing and assembly of the generic argument for all flash configurations.

The structure of the generic argument is as follows:

*(params, overall fractions, target state 1, target state 2, pressure, temperature,
saturations, phase fractions, partial fractions in phase 1, ... , partial fractions in
phase n)*

This is the most general layout, reflecting especially the order of values in the
array.

Individual flash types cherry pick the elements they need for their respective,
generic argument.

Examples:
    The generic argument for the p-T flash is of form

    *(params, overall fractions, pressure, temperature, phase fractions, partial
    fractions)*.

    The generic argument for the p-h flash is of form

    *(params, overall fractions, enthalpy, pressure, temperature, phase fractions,
    partial fractions)*

    The generic argument for the v-h (and analogously u-h) flash is as above,
    with target state 1 and two being target volume and target enthalpy respectively.

The generic argument formulation enables us to formulate any flash system as a function
``F(X_gen)``, and hence solve it with mathematical means.

Most importantly, the order of the elements in the generic argument reflects the order
of derivatives (columns) in the Jacobian ``DF(X_gen)``.

Various ``*_jac``-functions returning partial Jacobians of the unified flash system
always return the full Jacobian w.r.t. to **all** possible dependencies:

*(pressure, temperature, saturations, phase fractions, partial fractions)*

Individual flash systems must assemble the partial Jacobians they need, and slice.

"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import numba as nb
import numpy as np

import porepy as pp

from .._core import NUMBA_FAST_MATH, NUMBA_PARALLEL


def generic_arg_from_fluid_state(
    flash_type: str,
    num_components: int,
    num_phases: int,
    num_values: int,
    target_state: pp.compositional.FluidProperties,
    state_is_initialized: bool = False,
    params: Optional[Sequence[np.ndarray]] = None,
) -> np.ndarray:
    # Unknowns in all flash types are independent phase fractions and partial fractions.
    f_dim: int = num_phases - 1 + num_phases * num_components

    # Target states depending on flash type.
    state_1: np.ndarray
    state_2: np.ndarray

    if flash_type == "p-T":
        state_1 = target_state.p
        state_2 = target_state.T
    elif flash_type == "p-h":
        f_dim += 1
        # Note reverse order because p has special treatment
        state_1 = target_state.h
        state_2 = target_state.p
    elif flash_type == "v-h":
        f_dim += 2 + num_phases - 1
        state_1 = target_state.v
        state_2 = target_state.h

    # Second dimension for vectorized input.
    X_gen = np.zeros((num_components + 1 + f_dim, num_values))

    # Filling the independent overall fractions into gen arg
    for i in range(num_components - 1):
        X_gen[i] = target_state.z[i + 1]
    # Filling of target state
    X_gen[num_components - 1] = state_1
    X_gen[num_components] = state_2

    # If initial fluid state is given, insert values in gen arg.
    if state_is_initialized:
        # Index of first fractional variable.
        idx_f = -(num_phases * num_components + num_phases - 1)
        # Phase fractions and partial fractions.
        for j in range(num_phases):
            # Skip dependent phase fraction.
            if j > 0:
                X_gen[idx_f + j] = target_state.y[j]
            for i in range(num_components):
                X_gen[-((num_phases - j) * num_components) + i] = target_state.phases[
                    j
                ].x[i]

        # If isochoric specifications, saturations are variables.
        if "v" in flash_type:
            # Index of first fractional variable changes.
            idx_f -= num_phases - 1
            for j in range(1, num_phases):
                X_gen[idx_f + j - 1] = target_state.sat[j]

        # For any flash type, p and T are always stored right before fractional values.
        X_gen[idx_f - 1] = target_state.T
        X_gen[idx_f - 2] = target_state.p

    # If parameters are given, store them as first elements.
    if params is not None:
        X_params = np.zeros((len(params), num_values))
        for i, p in enumerate(params):
            X_params[i] = p
        X_gen = np.vstack((X_params, X_gen))

    # Transpose for vectorization of input over rows.
    return X_gen.T


@nb.njit(
    nb.i4(nb.types.UniTuple((nb.i4, 2)), nb.types.unicode_type),
    fastmath=True,
    cache=True,
)
def dim_gen_arg(npnc: tuple[int, int], flash_type: str) -> int:
    """Returns the base dimension (no parameters) of the generic flash argument for
    a specified flash type.

    Parameters:
        npnc: Tuple containing the number of phases and components.
        flash_type: A string denoting the flash type.

    Returns:
        ``n`` where the generic argument for a single flash problem has shape ``(n,)``,
        assuming no parameters are stored within.

    """
    nphase, ncomp = npnc
    nip = nphase - 1

    # Base dimension involes phase fractions and partial fractions, as well as two
    # target state values and ncomp - 1 independent overall fractions
    d = nip + nphase * ncomp + 2 + ncomp - 1
    # T is a variable, if not specified
    if "t" not in flash_type.lower():
        d += 1
    # p is a variable if not specified.
    if "p" not in flash_type.lower():
        d += 1
    # saturations are variables in isochoric specifications.
    if "v" in flash_type.lower():
        d += nip

    return d


@nb.njit(
    nb.types.Tuple(
        (
            nb.f8[:, :],
            nb.f8[:, :, :],
            nb.f8[:, :],
            nb.f8[:, :],
            nb.f8[:],
            nb.f8[:],
            nb.f8[:],
            nb.f8[:],
            nb.f8[:, :],
        )
    )(nb.f8[:], nb.types.UniTuple((nb.i4, 2)), nb.types.unicode_type),
    fastmath=True,
    parallel=NUMBA_PARALLEL,
    cache=True,
)
def parse_vectorized_generic_arg(
    X_gen: np.ndarray, npnc: tuple[int, int], flash_type: str
) -> tuple[
    np.ndarray,
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
    nphase, ncomp = npnc
    n = X_gen.shape[0]

    s = np.empty((nphase, n), dtype=np.float64)
    x = np.empty((nphase, ncomp, n), dtype=np.float64)
    y = np.empty((nphase, n), dtype=np.float64)
    z = np.empty((ncomp, n), dtype=np.float64)
    p = np.empty(n, dtype=np.float64)
    T = np.empty(n, dtype=np.float64)
    state_1 = np.empty(n, dtype=np.float64)
    state_2 = np.empty(n, dtype=np.float64)

    # Fetching number of paramters stored.
    dim_params = X_gen.shape[1] - dim_gen_arg(npnc, flash_type)
    params = np.empty((dim_params, n), dtype=np.float64)

    for i in nb.prange(n):
        s_i, x_i, y_i, z_i, p_i, T_i, s1_i, s2_i, x_p_i = parse_generic_arg(
            X_gen[i], npnc, flash_type
        )

        s[:, i] = s_i
        x[:, :, i] = x_i
        y[:, i] = y_i
        z[:, i] = z_i
        p[i] = p_i
        T[i] = T_i
        state_1[i] = s1_i
        state_2[i] = s2_i
        params[:, i] = x_p_i

    return s, x, y, z, p, T, state_1, state_2, params


@nb.njit(
    nb.types.Tuple(
        (
            nb.f8[:],
            nb.f8[:, :],
            nb.f8[:],
            nb.f8[:],
            nb.f8,
            nb.f8,
            nb.f8,
            nb.f8,
            nb.f8[:],
        )
    )(nb.f8[:], nb.types.UniTuple((nb.i4, 2)), nb.types.unicode_type),
    fastmath=True,
    cache=True,
)
def parse_generic_arg(
    X_gen: np.ndarray, npnc: tuple[int, int], flash_type: str
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    np.ndarray,
]:
    """ "Parses the generic flash argument and returns the unknowns and parameters of
    the flash problem.

    Parameters:
        X_gen: Generic flash argument (1D array).
        npnc: A 2-tuple containing the number of phases and components.
        flash_type: A string denoting the flash type/ target state (e.g. ``'p-T'``)

    Returns:
        A tuple containing

        1. 1D array of saturations
        2. 2D array of (extended) partial fractions, row-wise per phase.
        3. 1D array of phase fractions.
        4. 1D array of overall fractions
        5. pressure value
        6. temperature value
        7. first target state
        8. second target state
        9. other parameters stored in the generic argument.

        The first and second target state can coincide with the pressure or temperature
        value, if pressure or temperature are defined as target state values in
        ``flash_type``.

        All fractions contain values for reference phase and component. They are
        always stored as the first value.

        Saturations are returned as zero, if the flash is not isochoric.

        If no parameters are stored, the parameter array returned last is of shape (0,).

    """
    nphase, ncomp = npnc

    s = np.zeros(nphase)
    y = np.zeros(nphase)
    z = np.zeros(ncomp)

    # The last nphase * ncomp values are the extended partial fractions
    i = nphase * ncomp  # Keeping track of accessed indices (from back to front).
    x = X_gen[-i:].reshape((nphase, ncomp))

    # Phase fractions
    y[1:] = X_gen[-(i + nphase - 1) : i]
    y[0] = 1.0 - y.sum()
    i += nphase - 1

    # If pressure is an unknown, saturations are as well (isochoric flash).
    if "p" not in flash_type.lower():
        s[1:] = X_gen[-(i + nphase - 1) : i]
        s[0] = 1.0 - s.sum()
        i += nphase - 1

    # pressure and temperature are always the last (seen from back) unknowns.
    p, T = X_gen[-(i + 2) : i]
    i += 2

    # Now come the state definitions, where the indexing is flash-type-specific.
    if "p-t" in flash_type.lower():
        state_1 = p
        state_2 = T
    elif "p-h" in flash_type.lower():
        state_1 = p
        state_2 = X_gen[-(i + 1)]
        i += 1
    elif "v-h" in flash_type.lower():
        state_1, state_2 = X_gen[-(i + 2) : i]
        i += 2
    else:
        raise NotImplementedError(f"Unknown flash type {flash_type}")

    # The final standard elements of the generic argument are the independent overall
    # fractions.
    z[1:] = X_gen[-(i + ncomp - 1) : i]
    z[0] = 1.0 - z.sum()

    # Other parameters, if any.
    params = X_gen[:-i]

    # Sanity check.
    assert X_gen.shape[0] == i + params.shape[0]

    return s, x, y, z, p, T, state_1, state_2, params


@nb.njit(
    nb.f8[:, :](
        nb.f8[:, :],
        nb.f8[:, :, :],
        nb.f8[:, :],
        nb.f8[:, :],
        nb.f8[:],
        nb.f8[:],
        nb.f8[:],
        nb.f8[:],
        nb.f8[:, :],
        nb.types.unicode_type,
    ),
    parallel=NUMBA_PARALLEL,
    fastmath=True,
    cache=True,
)
def assemble_vectorized_generic_arg(
    s: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    p: np.ndarray,
    T: np.ndarray,
    state_1: np.ndarray,
    state_2: np.ndarray,
    params: np.ndarray,
    flash_type: str,
) -> np.ndarray:
    """Parallelized version of :func:`assemble_generic_arg` for vectorized input.

    Assembly is performed such that 1 row in the return value represents one generic
    flash argument.

    """
    ncomp = z.shape[0]
    nphase = y.shape[0]
    n = p.shape[0]
    n_param = params.shape[0]

    d = dim_gen_arg((nphase, ncomp), flash_type)
    X_gen = np.empty((n, d + n_param), dtype=np.float64)

    for i in nb.prange(n):
        X_gen[i] = assemble_generic_arg(
            s[:, i],
            x[:, :, i],
            y[:, i],
            z[:, i],
            p[i],
            T[i],
            state_1[i],
            state_2[i],
            params[:, i],
            flash_type,
        )
    return X_gen


@nb.njit(
    nb.f8[:](
        nb.f8[:],
        nb.f8[:, :],
        nb.f8[:],
        nb.f8[:],
        nb.f8,
        nb.f8,
        nb.f8,
        nb.f8,
        nb.f8[:],
        nb.types.unicode_type,
    ),
    fastmath=True,
    cache=True,
)
def assemble_generic_arg(
    s: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    p: float,
    T: float,
    state_1: float,
    state_2: float,
    params: np.ndarray,
    flash_type: str,
) -> np.ndarray:
    """Inverse operation of :func:`parse_generic_arg`.

    Note:
        Though not practical, every potential part of the generic argument has to be
        passed for every flash type. This is due to the static signature of this
        numba function, otherwise the compilation and usage will become less efficient
        and clear.

        It also makes this (and the parser) usable for any flash type.

    Parameters:
        s: ``shape=(num_phases,)``

            Phase saturations.
        x: ``shape=(num_phases, num_components)``

            (Extended) partial fractions per phase.
        y: ``shape=(num_phases,)``

            Phase fractions.
        z: ``shape=(num_components,)``

            Overall component fractions.
        p: Pressure value.
        T: Temperature value.
        state_1: First target state (f.e. pressure in p-h flash).
        state_2: Second target state (f.e. enthalpy in p-h flash).
        params: ``shape=(n,)``

            Vector of other parameters.
        flash_type: String denoting flash type.

    Returns:
        The generic argument corresponding to the ``flash_type``.

    """
    nphase, ncomp = x.shape

    ## Allocating parts of the generic argument.
    # Fractions which are always unknowns.
    X_gen_yx = np.zeros(nphase - 1 + nphase * ncomp)
    X_gen_yx[: nphase - 1] = y[1:]
    X_gen_yx[nphase - 1 :] = x.reshape((nphase * ncomp,))

    # Keeping track of the size.
    i = nphase - 1 + nphase * ncomp

    # If pressure is an unknown, saturations are as well
    if "p" in flash_type.lower():
        X_gen_s = s[1:]
        i += nphase - 1
    else:
        X_gen_s = np.zeros((0,))

    # Non-fractional values enter the generic argument depending on the flash type.
    if "p-t" in flash_type.lower():
        X_gen_state = np.zeros(2)
        X_gen_state[0] = p
        X_gen_state[1] = T
        i += 2
    elif "p-h" in flash_type.lower():
        X_gen_state = np.zeros(3)
        X_gen_state[0] = state_2
        X_gen_state[1] = p
        X_gen_state[2] = T
        i += 3
    elif "v-h" in flash_type.lower():
        X_gen_state = np.zeros(4)
        X_gen_state[0] = state_1
        X_gen_state[1] = state_2
        X_gen_state[2] = p
        X_gen_state[3] = T
        i += 4
    else:
        raise NotImplementedError(f"Unknown flash type {flash_type}.")

    # Finally, overall fractions.
    X_gen_z = z[1:]
    i += ncomp - 1

    # Create generic argument.
    X_gen = np.hstack((params, X_gen_z, X_gen_state, X_gen_s, X_gen_yx))

    # Sanity check.
    assert X_gen.shape[0] == i + params.shape[0]

    return X_gen


@nb.njit(
    nb.f8[:](nb.f8[:, :], nb.f8[:], nb.f8[:]),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def mass_conservation_res(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    r"""Assembles the residual of the mass conservation equations.

    For each component ``i``, except reference component, it holds

    ... math::

        \sum_j y_j x_{ij}  - z_i = 0.

    Number of phases and components is determined from the chape of ``x``.

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
    # tensordot is the fastest option for non-contiguous arrays,
    # but currently unsupported by numba TODO
    # return (z - np.tensordot(y, x, axes=1))[1:]
    return (np.dot(y, x) - z)[1:]


@nb.njit(
    nb.f8[:, :](nb.f8[:, :], nb.f8[:]),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def mass_conservation_jac(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Returns the Jacobian of the residual described in
    :func:`mass_conservation_res`.

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
    nphase, ncomp = x.shape
    nip = nphase - 1  # number of independent phases

    # Must fill with zeros, since slightly sparse and below fill-up does not cover
    # elements which are zero.
    jac = np.zeros((ncomp - 1, nip + nphase * ncomp), dtype=np.float64)

    for i in range(ncomp - 1):
        # (1 - sum_j y_j) x_ir + y_j x_ij is there, per phase
        # hence d mass_i / d y_j = x_ij - x_ir
        jac[i, :nip] = x[1:, i + 1] - x[0, i + 1]  # i + 1 to skip ref component

        # d.r.t. w.r.t x_ij is always y_j for all j per mass conv.
        jac[i, 2 + nip + i :: nphase] = y  # nphase -1 + i + 1 to skip ref component

    # Adding trivial derivatives w.r.t. p, T and saturations
    return np.hstack((np.zeros((ncomp - 1, 2 + nip), dtype=np.float64), jac))


@nb.njit(
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


@nb.njit(
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
    nphase, ncomp = x.shape
    nip = nphase - 1  # number of independent phases
    # must fill with zeros, since matrix sparsely populated.
    jac = np.zeros((nphase, nip + nphase * ncomp), dtype=np.float64)

    unities = 1 - np.sum(x, axis=1)

    # first complementary condition is w.r.t. to reference phase
    # (1 - sum_j y_j) * (1 - sum_i x_i0)
    jac[0, :nip] = (-1) * unities[0]
    jac[0, nip : nip + ncomp] = y[0] * (-1)
    for j in range(1, nphase):
        # for the other phases, it's slightly easier since y_j * (1 - sum_i x_ij)
        jac[j, j - 1] = unities[j]
        jac[j, nip + j * ncomp : nip + (j + 1) * ncomp] = y[j] * (-1)

    return np.hstack((np.zeros((nphase, 2 + nip), dtype=np.float64), jac))


@nb.njit(
    nb.f8[:](nb.f8[:, :], nb.f8[:, :]),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def isofugacity_constraints_res(x: np.ndarray, phis: np.ndarray) -> np.ndarray:
    r"""Assembles the residual of the isofugacity constraints.

    For each independent phase ``j``, and each component ``i`` it holds

    ... math::

        x_{ij}\varphi_{ij} - x_{i0}\varphi_{i0} = 0.

    Number of phases and components is determined from the chape of ``x``.
    The reference phase is assumed to be the first one.

    Parameters:
        x: ``shape=(num_phases, num_components)``

            (Extended) partial fractions per phase.
        phis: ``shape=(num_phases, num_components)``

            Fugacities per phase and component.

    Returns:
        An array with ``shape=(num_phases - 1,)`` containing the residual of the
        isofugacity constraints per independent phase.

    """
    nphase, ncomp = x.shape
    res = np.zeros(ncomp * (nphase - 1), dtype=np.float64)

    for j in range(1, nphase):
        res[(j - 1) * ncomp : j * ncomp] = x[j] * phis[j] - x[0] * phis[0]
        # isofug[(j - 1) * ncomp : j * ncomp] = x[j] * phis[j] / phis[0] - x[0]

    return res


@nb.njit(
    nb.f8[:, :](nb.f8[:, :], nb.f8[:, :], nb.f8[:, :, :]),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def isofugacity_constraints_jac(
    x: np.ndarray, phis: np.ndarray, dphis: np.ndarray
) -> np.ndarray:
    """"""
    nphase, ncomp = x.shape
    nip = nphase - 1  # number of independent phases
    # Allocating for pTx derivatives
    jac = np.zeros((ncomp * nip, 2 + ncomp * nphase))

    # Creating block of derivatives of expression x_{i0} phi_{i0}
    # product rule: x * dphi
    block_0 = (dphis[0, :, :].T * x[0]).T
    # + phi * dx  (minding the first two columns which contain the dp dT)
    block_0[:, 2:] += np.diag(phis[0])

    # Loop over row blocks associated with constraints between an independent phase
    # and the reference phase, for all components.
    for j in range(1, nphase):
        # Creating block of derivatives of expression x_{ij} phi_{ij}
        block_j = (dphis[j, :, :].T * x[j]).T
        block_j[:, 2:] += np.diag(phis[j])

        # p, T derivatives
        idx = (j - 1) * ncomp  # start of row block
        jac[idx : idx + ncomp, :2] = block_j[:, :2] - block_0[:, :2]

        # Derivatives w.r.t. partial fractions.
        # d(x_ij * phi_ij - x_ir * phi_ir)
        # Hence every row-block associated with an independent phase contains -block_0
        jac[idx : idx + ncomp, 2 : 2 + ncomp] = -block_0[:, 2:]
        # Derivatives w.r.t. fractions in independent phase j
        jac[idx : idx + ncomp, 2 + j * ncomp : 2 + (j + 1) * ncomp] = block_j[:, 2:]

    # Adding trivial columns for derivatives w.r.t. phase fractions and saturations.
    return np.hstack((jac[:, :2], np.zeros((ncomp * nip, 2 * nip)), jac[:, 2:]))


@nb.njit(nb.f8[:](nb.f8, nb.f8[:], nb.f8[:]), fastmath=NUMBA_FAST_MATH, cache=True)
def first_order_constraint_res(
    phi_target: float, w: np.ndarray, phis: np.ndarray
) -> np.ndarray:
    r"""Assembles the constraint of a first-order thermodynamic function

    .. math::

        \frac{\sum_j w_j \phi_j}{\hat{\phi}} - 1 = 0,

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


@nb.njit(
    nb.f8[:, :](nb.f8[:], nb.f8[:], nb.f8[:, :], nb.i1),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def first_order_constraint_jac(
    w: np.ndarray, phis: np.ndarray, dphis: np.ndarray, w_flag: Literal[0, 1]
) -> np.ndarray:
    """Assembles the Jacobian of the first order constraint given by
    :func:`first_order_constraint_res`.

    The derivatives are assembled such that the first two columns contain the
    derivatives w.r.t. two common dependencies of the phase-related quantities
    (like enthalpy, which depends on pressure and temperature for every phase enthalpy).
    The next columns contain the derivatives w.r.t. independent fractions.
    The last columns contain the derivatives w.r.t. the dependencies per phase, for each
    phase (like partial fractions per phase) in phase-major order.

    Important:
        This is the only (partial) Jacobian which does not include *all* derivatives
        by default, because the weights can either be phase fractions or saturations.

    Parameters:
        w: ``shape=(num_phases,)``

            Phase fractions/saturations.
        phis: ``shape=(num_phases,)``

            Phase-related partial value of constrained function.
        dphis: ``shape=(num_phases, 2 + num_diffs)``

            Derivatives of phase-related partial value of constrained function.
            The derivatives must be such that the first two columns contain the
            derivatives w.r.t. to the same variables (like pressure and temperature).
            The remaining columns can be phase-related variables (like partial
            fractions), but they must be equal in number per phase (``num_diffs``).
        w_flag: ``{0,1}``

            Specifies whether ``w`` denotes the saturations or phase fractions, and
            hence the columns of the respective derivatives in the Jacobian.

            0 for saturations, 1 for fractions. The columns of the saturation
            derivatives come before the fraction derivatives.

    Returns:
        The Jacobian of shape ``(1, 2 + 2 * (num_phases - 1) + num_phases *
        num_components)``.

    """
    nphase = w.shape[0]
    # Number of derivatives excluding p and T derivatives is equal to number of partial
    # fractions (components).
    ncomp = dphis.shape[1] - 2
    nip = nphase - 1  # number of independent phases

    # Allocate correct number of derivatives.
    jac = np.zeros(2 + nip + nphase * ncomp, dtype=np.float64)

    # Derivatives w.r.t. p and T
    jac[:2] = (dphis[:, :2].T * w).T.sum(axis=0)
    # Derivatives w.r.t weights. Keep in mind that w_0 = 1 - w_1 - w_2 ...
    jac[2 : 2 + nip] = phis[1:] - phis[0]
    # Derivatives w.r.t. partial fractions per phase.
    jac[2 + nip :] = (dphis[:, 2:].T * w).T.reshape((nphase * ncomp,))

    # Including zero columns for the other phase-related variables.
    u = np.zeros(nip, dtype=np.float64)
    if w_flag == 0:  # w represents saturations
        jac = np.hstack((jac[: 2 + nip], u, jac[2 + nip :]))
    elif w_flag == 1:  # w represents phase fractions
        jac = np.hstack((jac[:2], u, jac[2:]))
    else:
        raise ValueError(f"w_flag expected to be 0 or 1, got {w_flag}.")

    # Reshaping because this is expected to be a row in a larger Jacobian.
    return jac.reshape((1, jac.shape[0]))


@nb.njit(
    nb.f8[:](nb.f8, nb.f8[:], nb.f8[:]),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def volume_constraint_res(
    v_target: float, s: np.ndarray, rhos: np.ndarray
) -> np.ndarray:
    r"""Analogous to :func:`first_order_constraint_res`, but with a different scaling
    based on the relation :math:`v\cdot\rho = 1`, to avoid some error."""
    return np.ones((1,), dtype=np.float64) * (v_target * (s * rhos).sum() - 1.0)


@nb.njit(
    nb.f8[:](nb.f8[:], nb.f8[:], nb.f8[:]),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def phase_mass_constraints_res(
    s: np.ndarray, y: np.ndarray, rhos: np.ndarray
) -> np.ndarray:
    r"""Assembles the residual of the phase mass constraints

    .. math::

        s_j \rho_j - y_j (\sum_k s_k \rho_k) = 0,

    for each independent phase ``j``.

    Parameters:
        s: ``shape=(num_phases,)``

            Phase saturations.
        y: ``shape=(num_phases,)``

            Phase fractions.
        rhos: ``shape=(num_phases,)``

            Phase densities.

    Returns:
        An array with shape ``(num_phases - 1,)``, containing the residuals of above
        equations.

    """
    rho = (s * rhos).sum()
    # First phase is dependent.
    return (rho * y / rhos - s)[1:]
    # return (s * rhos / rho - y)[1:]


@nb.njit(
    nb.f8[:, :](nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:, :]),
    fastmath=NUMBA_FAST_MATH,
    cache=True,
)
def phase_mass_constraints_jac(
    s: np.ndarray, y: np.ndarray, rhos: np.ndarray, drhos: np.ndarray
) -> np.ndarray:
    """Assembles the Jacobian of the phase mass constraints given by
    :func:`phase_mass_constraints_res`.

    Parameters:
        s: ``shape=(num_phases,)``

            Phase saturations.
        y: ``shape=(num_phases,)``

            Phase fractions.
        rhos: ``shape=(num_phases,)``

            Phase densities.
        drhos: ``shape=(num_phases, 2 + num_diffs)``

            Derivatives of phase densities, with pressure and temperature derivatives
            in the first two columns, and derivatives w.r.t. to other dependencies
            in the remaining columns (like partial fractions per component in phase).

    Returns:
        The Jacobian with shape ``(num_phases - 1, 2 + 2 * (num_phases - 1) + num_phases
        * num_components)``.

    """
    nphase = s.shape[0]
    # Number of derivatives excluding p and T derivatives is equal to number of partial
    # fractions (components).
    ncomp = drhos.shape[1] - 2
    nip = nphase - 1  # number of independent phases

    # Allocating Jacobian
    jac = np.zeros((nip, 2 + 2 * nip + nphase * ncomp), dtype=np.float64)

    # overall density sum s_j rho_j
    rho = (s * rhos).sum()

    for j in range(nip):
        j1 = j + 1  # Skipping the reference phase.
        # Every phase mass constraint can be seen as a weighted first-order constraint.
        # y_j / rho_j * (sum_k s_k* rho_k) - s_j = 0
        w = y[j1] / rhos[j1]
        # NOTE outer multiplication with w holds because below function considers s_j
        # constant and makes only the Jacobian of (sum_k s_k * rho_k)
        jac[j] = first_order_constraint_jac(s, rhos, drhos, 0) * w
        # The derivative w.r.t. s_j has an additional term not covered by above.
        jac[j, 2 + j] -= 1.0

        # The derivative w.r.t. y_j.
        jac[j, 2 + nip + j] = rho / rhos[j1]

        # Now we require also the product rule for the term (y_j / rho_j) * rho.
        # (y_j / rho_j) * d rho is covered above, we need + rho * d (y_j / rho_j).
        # And we need this only w.r.t. to the dependencies of rho_j, which are p,T,x_ij
        outer = -rho * y[j1] / rhos[j1] ** 2
        d = outer * drhos[j1]
        # Contribution to p,T, and the partial fractions respectively.
        jac[j, :2] += d[:2]
        jac[j, 2 + 2 * nip + j * ncomp : 2 + 2 * nip + j1 * ncomp] = d[2:]

    # drho_dpT = (drhos[:, :2].T * s).T.sum(axis=0)

    # # derivative of s_j rho_j / (sum_i s_i rho_i) - y_j, for j > 0,
    # # with s_0 = 1 - s_1 - s_2 ...

    # # derivatives w.r.t. to phase fractions yield identity * -1
    # jac[:, 2 + nip : 2 + 2 * nip] = -np.eye(nip)

    # for j_mat in range(nip):
    #     j = j_mat + 1

    #     # outer derivative of rho_j * dpTx(1 / rho_mix)
    #     outer_j = -rhos[j] / rho**2

    #     # Derivatives w.r.t. saturations of sat_j * rho_j / (sum_i s_i * rho_i)
    #     # With s_0 = 1 - sum_(i > 0) s_i it holds for k > 0
    #     # ds_k (s_j rho_j / (sum_i s_i * rho_i)) =
    #     # s_j * (- rho_j / (sum_i s_i * rho_i)^2 * (rho_k - rho_0))
    #     jac[j_mat, 2 : 2 + nip] = s[j] * outer_j * (rhos[1:] - rhos[0])
    #     # + delta_kj * rho_j / (sum_i s_i * rho_i) with delta_kj = ds_k s_j
    #     jac[j_mat, 2 + j_mat] += rhos[j] / rho

    #     # Derivatives w.r.t. p, T
    #     # With s_0 = 1 - sum_(i > 0) s_i and rho = sum_i s_i * rho_i
    #     # dpt (rho_j(p, T) / rho) =
    #     # dpt(rho_j(p,T)) / rho
    #     # - rho_j / (sum_i s_i * rho_i)^2 * dpt(rho)
    #     jac[j_mat, :2] = s[j] * (drhos[j, :2] / rho + outer_j * drho_dpT)

    #     # Derivatives w.r.t. x_ik
    #     # for all phases k, and j > 0
    #     # dx_ik (rho_j(x_ij) / rho_mix) =
    #     # delta_kj * (dx_ik(rho_j(x_ij)) / rho_mix)
    #     # - rho_j * (1 / rho_mix^2) * (dx_ik(sum_l s_l rho_l(x_il)))
    #     for k in range(nphase):
    #         idx = 2 + 2 * nip + k * ncomp  # starting index of column block
    #         jac[j_mat, idx : idx + ncomp] = s[j] * outer_j * s[k] * drhos[k, 2:]
    #         if k == j:
    #             jac[j_mat, idx : idx + ncomp] += s[j] * drhos[k, 2:] / rho

    return jac
