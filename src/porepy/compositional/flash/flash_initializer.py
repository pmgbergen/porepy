"""Module containing functionality to provide initial guesses for the equilibrium
problem."""

from __future__ import annotations

import logging
import time
from functools import partial
from typing import Callable, Literal, Optional, Sequence, TypeAlias, cast

import numba
import numba.typed
import numpy as np

import porepy as pp

from .._core import NUMBA_CACHE, NUMBA_FAST_MATH, NUMBA_PARALLEL, R_IDEAL_MOL
from ..utils import _compute_saturations, compute_saturations, normalize_rows
from .solvers._core import SOLVER_PARAMETERS_TYPE, _cfunc
from .uniflash_equations import (
    assemble_generic_arg,
    assemble_vectorized_generic_arg,
    first_order_constraint_jac,
    first_order_constraint_res,
    parse_generic_arg,
    parse_vectorized_generic_arg,
    phase_mass_constraints_jac,
    phase_mass_constraints_res,
    volume_constraint_res,
)

logger = logging.getLogger(__name__)

_SUPPORTED_FLASH_TYPES: TypeAlias = Literal["p-T", "p-h", "v-h"]
"""Type alias for supported values of flash types."""


# region Rachford-Rice equations


@numba.njit(
    numba.f8[:](numba.f8[:], numba.f8[:, :]), fastmath=NUMBA_FAST_MATH, cache=True
)
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


@numba.njit(numba.f8(numba.f8[:], numba.f8[:]), fastmath=NUMBA_FAST_MATH, cache=True)
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


@numba.njit(
    numba.f8(numba.f8[:], numba.f8[:], numba.f8[:, :]),
    cache=NUMBA_CACHE,  # NOTE cache is dependent on internal function
)
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


# endregion
# region General routines and helper methods


@_cfunc(numba.f8[:, :](numba.f8, numba.f8, numba.f8[:, :]), cache=True)
def get_K_values_template_func(p: float, T: float, x: np.ndarray) -> np.ndarray:
    """Template c-function for K-value computations.

    Parameters:
        p: Pressure.
        T: Temperature.
        x: 2D array containing row-wise extended partial fractions per phase.

    Returns:
        K-values w.r.t. to the reference phase (first row in ``x``).

    """
    return x * p * T


@_cfunc(numba.f8[:](numba.f8[:], SOLVER_PARAMETERS_TYPE), cache=True)
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


@numba.njit(
    numba.f8[:](
        numba.typeof(get_K_values_template_func),
        numba.f8[:],
        SOLVER_PARAMETERS_TYPE,
        numba.types.unicode_type,
        numba.i1,
    ),
    cache=True,
)
def fractions_from_rr(
    get_K_values: Callable[[float, float, np.ndarray], np.ndarray],
    X_gen: np.ndarray,
    params: dict[str, float],
    flash_type: str,
    use_wilson: Literal[0, 1],
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
            See :func:`~porepy.compositional.flash.uniflash_equations.parse_generic_arg`
        use_wilson: Flag to use the Wilson correlation for the first K-value guess.

    Returns:
        A generic flash argument with updated fractions.

    """
    # Parsing parameters and generic arg.
    nphase = int(params["num_phases"])
    ncomp = int(params["num_components"])
    N1 = int(params["N1"])
    npnc = (nphase, ncomp)
    s, x, y, z, p, T, s1, s2, x_p = parse_generic_arg(X_gen, npnc, flash_type)

    omegas = np.empty(ncomp)
    T_crits = np.empty(ncomp)
    p_crits = np.empty(ncomp)
    for i in range(ncomp):
        T_crits[i] = params[f"_T_crit_{i}"]
        p_crits[i] = params[f"_p_crit_{i}"]
        omegas[i] = params[f"_omega_{i}"]

    # Pseudo-critical quantities.
    T_pc = np.sum(z * T_crits)
    p_pc = np.sum(z * p_crits)

    if use_wilson != 0:
        K = np.empty((nphase - 1, ncomp), dtype=np.float64)
        for j in range(nphase - 1):
            K[j, :] = (
                np.exp(5.37 * (1 + omegas) * (1 - T_crits / T)) * p_crits / p + 1e-10
            )
    else:
        K = get_K_values(p, T, x)

    # Starting iterations using Rachford Rice.
    for n in range(N1):
        # Solving RR for phase fractions.
        if nphase == 2:
            # Only one independent phase assumed.
            K_ = K[0]
            if ncomp == 2:
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
                # K_min = np.min(K_)
                # K_max = np.max(K_)
                # y_1 = 1 / (1 - K_max)
                # y_2 = 1 / (1 - K_min)
                # if y_1 <= y_2:
                #     y_feasible = y_1 < _y < y_2
                # else:
                #     y_feasible = y_2 < _y < y_1

                # if y_feasible & negative:
                #     y_ = 0.0
                # elif y_feasible & exceeds:
                #     y_ = 1.0

                # If all K-values are smaller than 1 and gas fraction is negative,
                # the liquid phase is clearly saturated.
                # Vice versa, if fraction above 1 and K-.
                if negative & np.all(K_ < 1.0):
                    y_ = 0.0
                elif exceeds & np.all(K_ > 1.0):
                    y_ = 1.0

                # Assert corrections did what they have to do.
                assert 0.0 <= y_ <= 1.0, "y fraction estimate outside bound [0, 1]."
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
            K = get_K_values(p, T, x)

    return assemble_generic_arg(s, x, y, z, p, T, s1, s2, x_p, flash_type)


@numba.njit(
    numba.f8[:, :](
        numba.typeof(get_K_values_template_func),
        numba.f8[:, :],
        SOLVER_PARAMETERS_TYPE,
    ),
    parallel=NUMBA_PARALLEL,
    cache=True,
)
def rachford_rice_initializer(
    get_K_values: Callable[[float, float, np.ndarray], np.ndarray],
    X_gen: np.ndarray,
    params: dict[str, float],
) -> np.ndarray:
    """p-T initializer as a parallelized loop over all rows in the vectorized generic
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
    for f in numba.prange(X_gen.shape[0]):
        X_gen[f] = fractions_from_rr(get_K_values, X_gen[f], params, "p-T", 1)
    return X_gen


@numba.njit(
    numba.f8[:, :](
        numba.typeof(get_K_values_template_func),
        numba.typeof(update_state_template_func),
        numba.types.unicode_type,
        numba.f8[:, :],
        SOLVER_PARAMETERS_TYPE,
    ),
    parallel=NUMBA_PARALLEL,
    cache=True,
)
def nested_initializer(
    get_K_values: Callable[[float, float, np.ndarray], np.ndarray],
    update_state_func: Callable[[np.ndarray, dict[str, float]], np.ndarray],
    flash_type: str,
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
    # tol = params['tolerance']
    for f in numba.prange(X_gen.shape[0]):
        xf = X_gen[f]
        for _ in range(N3):
            xf = update_state_func(xf, params)
            xf = fractions_from_rr(get_K_values, xf, params, flash_type, 0)

            # abort if residual already small enough
            # if np.linalg.norm(F_ph(xf)) <= tol:
            #     break

        X_gen[f] = xf
    return X_gen


# endregion


class FlashInitializer:
    """Container for compiled flash initialization methods providing an initial guess
    for the equilibrium problem.

    The base class uses heuristics and Rachford-Rice equations to provide initial values
    for fractions, pressure and temperature, depending on the flash type.

    Important:
        If pressure and temperature should be guessed, they must be passed as zeros
        in the generic flash argument.

    Parameters:
        fluid: The fluid for which the flash is compiled. Supports currently only
            2-phase fluids.
        params: Initialization parameters (see :attr:`params`).

    """

    def __init__(
        self,
        fluid: pp.Fluid[pp.FluidComponent, pp.Phase[pp.FluidComponent]],
        params: Optional[dict[str, float]] = None,
    ) -> None:
        ncomp = fluid.num_components
        nphase = fluid.num_phases

        assert nphase == 2, "Supports only 2-phase mixtures."
        assert ncomp >= 2, "Must have at least two components."

        self._npnc: tuple[int, int] = (nphase, ncomp)
        """Tuple containing the number of phases and components in the fluid."""

        # data used in initializers
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
        self._phasestates: Sequence[pp.compositional.PhysicalState] = [
            phase.state for phase in fluid.phases
        ]
        """A sequence containing the physical phase state per phase."""
        self._gas_phase_index: Optional[int] = fluid.gas_phase_index
        """The index of the gas phase. None if gas not existent."""

        eos = fluid.reference_phase.eos
        assert isinstance(eos, pp.compositional.EoSCompiler)
        self._eos: pp.compositional.EoSCompiler = eos
        """Compiled EoS of the reference phase, assuming all phases have the same EoS.
        """

        self._initializers: dict[
            str,
            Callable[[np.ndarray, dict[str, float]], np.ndarray],
        ] = {}
        """Storage of initialization routines.
        
        Initialization routines take a generic argument and a parameter dictionary as
        arguments, and return the updated generic argument.

        The generic argument can be vectorized (flash per row).

        """

        default_params: dict[str, float] = {
            "N1": 3.0,
            "N2": 1.0,
            "N3": 5.0,
            "tolerance": 1e-6,
        }
        if params is None:
            params = default_params
        else:
            default_params.update(params)
            params = default_params

        self.params = params
        """Initialization parameters passed at instantiation.
        
        Default parameters are:

        - ``'N1'``: Number of loop for fraction guess.
        - ``'N2'``: Number of loops used for the update of other state functions.
        - ``'N3'``: Number of alternations between fraction and state function update.
        - ``'tolerance'``: Criterion for early stopping of initialization.

        """

        self._nb_params: dict[str, float]
        """Numba-type dictionary, to be filled with :attr:`params` and passed to the
        flash initialization methods"""

    def __getitem__(
        self, key: _SUPPORTED_FLASH_TYPES
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Shortcut for accessing flash initial guess methods for flash types denoted by
        ``key``."""
        # This will raise a key error on time.
        _ = self._initializers[key]

        if not hasattr(self, "_nb_params"):
            # Creation of numba-typed dict upon first call.
            d = numba.typed.Dict.empty(
                key_type=numba.types.unicode_type, value_type=numba.types.float64
            )
            self._nb_params = cast(dict[str, float], d)
            self._nb_params["num_phases"] = float(self._npnc[0])
            self._nb_params["num_components"] = float(self._npnc[1])
            self._nb_params["gas_phase_index"] = float(
                -1 if self._gas_phase_index is None else self._gas_phase_index
            )

            # Adding also some component parameters which are required
            for i in range(self._npnc[1]):
                self._nb_params[f"_T_crit_{i}"] = float(self._Tcrits[i])
                self._nb_params[f"_p_crit_{i}"] = float(self._pcrits[i])
                self._nb_params[f"_v_crit_{i}"] = float(self._vcrits[i])
                self._nb_params[f"_omega_{i}"] = float(self._omegas[i])

        def initializer(x: np.ndarray) -> np.ndarray:
            """Wrapper for initialization routine, updating parameters and feeding
            them to the initialization method."""
            params = self._nb_params
            for k, v in self.params.items():
                params[str(k)] = float(v)
            return self._initializers[key](x, params)

        return initializer

    def compile(self, *args: _SUPPORTED_FLASH_TYPES) -> None:
        """Triggers the compilation of initialization routines.

        Parameters:
            *args: Specify subset of flash types which should be compiled to safe time.
                Due to some internal structures, the p-T initializer is always compiled.

        """

        # If not specified, compile all.
        if not args:
            args = ("p-T", "p-h", "v-h")

        if not self._eos.is_compiled:
            self._eos.compile()

        # Setting outer scope variables to avoid referencing self in JIT functions.
        nphase, ncomp = self._npnc
        phasestates = np.array(
            [
                # Depending on the environment, the enum value is sometimes already
                # evaluated, sometimes not... (pytest)
                state if isinstance(state, int) else state.value
                for state in self._phasestates
            ],
            dtype=np.int8,
        )
        npnc = self._npnc

        prearg_val_c = self._eos.funcs["prearg_val"]
        prearg_jac_c = self._eos.funcs["prearg_jac"]
        phi_c = self._eos.funcs["phi"]
        h_c = self._eos.funcs["h"]
        d_h_c = self._eos.funcs["dh"]
        rho_c = self._eos.funcs["rho"]
        d_rho_c = self._eos.funcs["drho"]

        logger.info(f"Compiling {args} flash initialization routines ..")
        start = time.time()

        @numba.njit(numba.f8[:, :](numba.f8, numba.f8, numba.f8[:, :]))
        def get_K_values(p: float, T: float, x: np.ndarray) -> np.ndarray:
            """See :func:`get_K_values_template_func`."""
            nphase, ncomp = x.shape
            K = np.empty((nphase - 1, ncomp), dtype=np.float64)
            xn = normalize_rows(x)
            pre_0 = prearg_val_c(phasestates[0], p, T, xn[0])
            phi_0 = phi_c(pre_0, p, T, xn)
            for j in range(1, nphase):
                pre_j = prearg_val_c(phasestates[j], p, T, xn[j])
                phi_j = phi_c(pre_j, p, T, xn[j])
                # Binding K-values away from zero
                K[j - 1, :] = phi_0 / phi_j + 1e-10
            return K

        self._initializers["p-T"] = partial(rachford_rice_initializer, get_K_values)

        if "p-h" in args:
            logger.debug("Compiling p-h flash initialization ..")

            @numba.njit(numba.f8[:](numba.f8[:], SOLVER_PARAMETERS_TYPE))
            def update_T_guess(
                X_gen: np.ndarray, params: dict[str, float]
            ) -> np.ndarray:
                """See :func:`update_state_template_func`."""
                # Parsing parameters.
                nphase = int(params["num_phases"])
                ncomp = int(params["num_components"])
                N2 = int(params["N2"])
                tol = params["tolerance"]
                gas_phase_idx = int(params["gas_phase_index"])
                npnc = (nphase, ncomp)
                # state 2 is target enthalpy
                s, x, y, z, p, T, s1, s2, x_p = parse_generic_arg(X_gen, npnc, "p-h")

                # If T has not been initialized at all (zero value), compute
                # pseudo-critical value as starting point
                if T == 0.0:
                    T_crits = np.empty(ncomp)
                    for i in range(ncomp):
                        T_crits[i] = params[f"_T_crit_{i}"]

                    T_pc = (T_crits * z).sum()
                    X_gen = assemble_generic_arg(
                        s, x, y, z, p, T_pc, s1, s2, x_p, "p-h"
                    )
                    X_gen = fractions_from_rr(get_K_values, X_gen, params, "p-h", 1)
                    s, x, y, z, p, T, s1, s2, x_p = parse_generic_arg(
                        X_gen, npnc, "p-h"
                    )

                xn = normalize_rows(x)
                hs = np.empty(nphase, dtype=np.float64)
                dh_dTs = np.empty(nphase, dtype=np.float64)

                for _ in range(N2):
                    for j in range(nphase):
                        pre_res_j = prearg_val_c(phasestates[j], p, T, xn[j])
                        pre_jac_j = prearg_jac_c(phasestates[j], p, T, xn[j])
                        hs[j] = h_c(pre_res_j, p, T, xn[j])
                        dh_dTs[j] = d_h_c(pre_res_j, pre_jac_j, p, T, xn[j])[1]

                    h_mix = (hs * y).sum()
                    h_constr_res = 1 - h_mix / s2
                    if np.abs(h_constr_res) < tol:
                        break
                    else:
                        dT_h_constr = -(dh_dTs * y).sum() / s2
                        dT = 0 - h_constr_res / dT_h_constr  # Newton iteration

                        # corrections to unfeasible updates because of decoupling
                        if np.abs(dT) > T:
                            dT = 0.1 * T * np.sign(dT)
                        dT *= 1 - np.abs(dT) / T
                        # Correction if gas phase is present and mixture enthalpy is too
                        # low to avoid overshooting T update
                        if gas_phase_idx >= 0:
                            if h_mix < s2 and y[gas_phase_idx] > 1e-3:
                                dT *= 0.4
                        T += dT

                return assemble_generic_arg(s, x, y, z, p, T, s1, s2, x_p, "p-h")

            self._initializers["p-h"] = partial(
                nested_initializer, get_K_values, update_T_guess, "p-h"
            )

        if "v-h" in args:
            logger.debug("Compiling v-h flash initialization ..")

            @numba.njit(numba.f8[:](numba.f8[:], SOLVER_PARAMETERS_TYPE))
            def update_pT_guess(
                X_gen: np.ndarray, params: dict[str, float]
            ) -> np.ndarray:
                """Helper function to update p-T guess for v-h flash by solving respective
                equations using Newton and some corrections."""

                # Parsing parameters
                N2 = int(params["N2"])
                tol = params["tolerance"]
                gas_phase_idx = int(params["gas_phase_index"])

                # Local system size.
                M = 2 + nphase - 1

                res = np.empty(M)
                jac = np.zeros((M, M))

                # s1 and s2 are target volume and enthalpy respectively
                s, x, y, z, p, T, s1, s2, x_p = parse_generic_arg(X_gen, npnc, "v-h")

                # Assume no guess, fetch later if otherwise.
                y_g = 0.0

                # If no p or T value are provided at all, create initial guess using
                # pseudo-critical values
                if p == 0.0 or T == 0.0:
                    T_crits = np.empty(ncomp)
                    v_crits = np.empty(ncomp)
                    for i in range(ncomp):
                        T_crits[i] = params[f"_T_crit_{i}"]
                        v_crits[i] = params[f"_v_crit_{i}"]
                    # pseudo_critical T_guess
                    T = (z * T_crits).sum()

                    # pseudo-critical pressure guess
                    v_pc = 0.0
                    for i in range(ncomp):
                        v_pc += v_crits[i] * z[i] ** 2
                        for k in range(i + 1, ncomp):
                            v_pc += (
                                z[i]
                                * z[k]
                                / 8
                                * (np.cbrt(v_crits[i]) + np.cbrt(v_crits[k])) ** 3
                            )

                    R = v_pc / s1
                    if R > 1:  # liquid-like gas
                        Z = 0.2
                        # T correction for liquid-like guess
                        T = T / np.sqrt(R)
                    else:  # gas-like
                        Z = 0.7

                    p = Z * T * R_IDEAL_MOL / s1

                    # Make first fraction guess based on pseudo-critical values.
                    xf = assemble_generic_arg(s, x, y, z, p, T, s1, s2, x_p, "v-h")
                    xf = fractions_from_rr(get_K_values, xf, params, "v-h", 1)
                    s, x, y, z, p, T, s1, s2, x_p = parse_generic_arg(
                        X_gen, npnc, "v-h"
                    )

                    # Correct pressure if no gas phase
                    if gas_phase_idx >= 0:
                        y_g = y[gas_phase_idx]

                    if y_g < 1e-3:
                        p *= 0.7
                        # T *= 1.1
                        # Refine fraction guess based on corrected pressure.
                        xf = assemble_generic_arg(s, x, y, z, p, T, s1, s2, x_p, "v-h")
                        xf = fractions_from_rr(get_K_values, xf, params, "v-h", 0)
                        s, x, y, z, p, T, s1, s2, x_p = parse_generic_arg(
                            X_gen, npnc, "v-h"
                        )

                xn = normalize_rows(x)
                if gas_phase_idx >= 0:
                    y_g = y[gas_phase_idx]

                rhos = np.empty(nphase)
                hs = np.empty(nphase)
                dhs = np.empty((nphase, 2 + ncomp))
                drhos = np.empty((nphase, 2 + ncomp))

                for _ in range(N2):
                    # Assembling volume and enthalpy constraints with derivatives for s-p-T

                    for j in range(nphase):
                        pre_val_j = prearg_val_c(phasestates[j], p, T, xn[j])
                        pre_jac_j = prearg_jac_c(phasestates[j], p, T, xn[j])
                        rhos[j] = rho_c(pre_val_j, p, T, xn[j])
                        hs[j] = h_c(pre_jac_j, p, T, xn[j])
                        dhs[j] = d_h_c(pre_val_j, pre_jac_j, p, T, xn[j])
                        drhos[j] = d_rho_c(pre_val_j, pre_jac_j, p, T, xn[j])

                    # Saturations are only used locally, hence we refer to sat, not s
                    # which is in the generic arg.
                    sat = _compute_saturations(y, rhos, 1e-10)
                    v_mix = 1.0 / (sat * rhos).sum()
                    h_mix = (y * hs).sum()

                    res[0] = first_order_constraint_res(s2, y, hs)[0] / s2
                    res[1] = volume_constraint_res(s1, sat, rhos)
                    res[2:] = phase_mass_constraints_res(sat, y, rhos)

                    jac[0] = first_order_constraint_jac(y, hs, dhs, 1)[0, :M] / s2
                    jac[1] = first_order_constraint_jac(sat, rhos, drhos, 0)[0, :M] * s1
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
                            # volume contraction only by positive p update, not negative T
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

                return assemble_generic_arg(s, x, y, z, p, T, s1, s2, x_p, "v-h")

            def vh_init(X_gen: np.ndarray, params: dict[str, float]):
                X_gen = nested_initializer(
                    get_K_values, update_pT_guess, "v-h", X_gen, params
                )
                # Performing final saturation update, after guessing fractions and p,T
                s, x, y, z, p, T, s1, s2, x_p = parse_vectorized_generic_arg(
                    X_gen, npnc, "v-h"
                )
                rhos = np.empty(y.shape)
                for j in range(nphase):
                    x_j = x[j, :, :]
                    xn = normalize_rows(x_j.T).T
                    pre = self._eos.gufuncs["prearg_val"](phasestates[j], p, T, xn)
                    rhos[j] = self._eos.gufuncs["rho"](pre, p, T, xn)
                s = compute_saturations(y, rhos)

                return assemble_vectorized_generic_arg(
                    s, x, y, z, p, T, s1, s2, x_p, "v-h"
                )

            self._initializers["v-h"] = vh_init

        logger.info(
            "Flash initialization routines compiled"
            + " (elapsed time: %.5f (s))." % (time.time() - start)
        )
