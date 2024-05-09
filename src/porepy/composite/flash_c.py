"""Module containing implementation of the unified flash using (parallel) compiled
functions created with numba.

The flash system, including a non-parametric interior point method, is assembled and
compiled using :func:`numba.njit`, to enable an efficient solution of the equilibrium
problem.

The compiled functions are such that they can be used to solve multiple flash problems
in parallel.

Parallelization is achieved by applying Newton in parallel for multiple input.
The intended use is for larg compositional flow problems, where an efficient solution
to the local equilibrium problem is required.

References:
    [1]: `Ben Gharbia et al. (2021) <https://doi.org/10.1051/m2an/2021075>`_
    [2]: `Vu et al. (2021) <https://doi.org/10.1016/j.matcom.2021.07.015>`_


"""

from __future__ import annotations

import logging
import time
from typing import Callable, Literal, Optional, Sequence

import numba
import numpy as np

from ._core import NUMBA_CACHE, R_IDEAL
from .base import Mixture
from .composite_utils import safe_sum
from .eos_compiler import EoSCompiler
from .flash import Flash
from .npipm_c import (
    convert_param_dict,
    initialize_npipm_nu,
    linear_solver,
    parallel_solver,
)
from .states import FluidState
from .utils_c import (
    _compute_saturations,
    _extend_fractional_derivatives,
    insert_pT,
    insert_sat,
    insert_xy,
    normalize_rows,
    parse_pT,
    parse_sat,
    parse_target_state,
    parse_xyz,
)

__all__ = ["CompiledUnifiedFlash"]


logger = logging.getLogger(__name__)


# region Helper methods


@numba.njit("float64[:](float64[:],float64[:,:])", fastmath=True, cache=True)
def _rr_poles(y: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Parameters:
        y: Phase fractions, assuming the first one belongs to the reference phase.
        K: Matrix of K-values per independent phase (row) per component (column)

    Returns:
        A vector of length ``num_comp`` containing the denominators in the RR-equation
        related to K-values per component.
        Each demoninator is given by :math:`1 + \\sum_{j\\neq r} y_j (K_{ji} - 1)`.

    """
    # tensordot is the fastes option for non-contigous arrays,
    # but currently unsupported by numba TODO
    # return 1 + np.tensordot(K.T - 1, y[1:], axes=1)
    return 1 + (K.T - 1) @ y[1:]  # K-values given for each independent phase


@numba.njit("float64(float64[:],float64[:])", fastmath=True, cache=True)
def _rr_binary_vle_inversion(z: np.ndarray, K: np.ndarray) -> float:
    """Inverts the Rachford-Rice equation for the binary 2-phase case.

    Parameters:
        z: ``shape=(num_comp,)``

            Vector of feed fractions.
        K: ``shape=(num_comp,)``

            Matrix of K-values per per component between vapor and liquid phase.

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
    "float64(float64[:],float64[:],float64[:,:])",
    cache=NUMBA_CACHE,  # NOTE cache is dependent on internal function
)
def _rr_potential(z: np.ndarray, y: np.ndarray, K: np.ndarray) -> float:
    """Calculates the potential according to [1] for the j-th Rachford-Rice equation.

    With :math:`n_c` components, :math:`n_p` phases and :math:`R` the reference phase,
    the potential is given by

    .. math::

        F = \\sum\\limits_{i} -(z_i ln(1 - (\\sum\\limits_{j\\neq R}(1 - K_{ij})y_j)))

    References:
        [1] `Okuno and Sepehrnoori (2010) <https://doi.org/10.2118/117752-PA>`_

    Parameters:
        z: ``len=n_c``

            Vector of feed fractions.
        y: ``len=n_p``

            Vector of molar phase fractions.
        K: ``shape=(n_p, n_c)``

            Matrix of K-values per independent phase (row) per component (column).

    Returns:
        The value of the potential based on above formula.

    """
    return np.sum(-z * np.log(np.abs(_rr_poles(y, K))))
    # F = [-np.log(np.abs(_rr_pole(i, y, K))) * z[i] for i in range(len(z))]
    # return np.sum(F)


# endregion

# region General flash equation independent of flash type and EoS


@numba.njit("float64[:](float64[:,:],float64[:],float64[:])", fastmath=True, cache=True)
def mass_conservation_res(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    r"""Assembles the residual of the mass conservation equations.

    For each component ``i``, except reference component, it holds

    ... math::

        z\left[i\right] - \sum_j y\left[j\right] x\left[j, i\right] = 0

    Number of phases and components is determined from the chape of ``x``.

    NJIT-ed function with signature
    ``(float64[:,:], float64[:], float64[:]) -> float64[:]``.

    Note:
        See :func:`parse_xyz` for obtaining the properly formatted arguments ``x,y,z``.

    Parameters:
        x: ``shape=(num_phase, num_comp)``

            Phase compositions
        y: ``shape=(num_phase,)``

            Molar phase fractions.
        z: ``shape=(num_comp,)``

            Overall fractions per component.

    Returns:
        An array with ``shape=(num_comp - 1,)`` containg the residual of the mass
        conservation equation (left-hand side of above equation) for each component,
        except the first one (in ``z``).

    """
    # tensordot is the fastes option for non-contigous arrays,
    # but currently unsupported by numba TODO
    # return (z - np.tensordot(y, x, axes=1))[1:]
    return (z - np.dot(y, x))[1:]


@numba.njit("float64[:,:](float64[:,:],float64[:])", fastmath=True, cache=True)
def mass_conservation_jac(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Returns the Jacobian of the residual described in
    :func:`mass_conservation_res`

    The Jacobian is of shape ``(num_comp - 1, num_phase - 1 + num_phase * num_comp)``.
    The derivatives (columns) are taken w.r.t. to each independent molar fraction,
    except feed fractions.

    The order of derivatives w.r.t. phase compositions is in phase-major order.

    NJIT-ed function with signature
    ``(float64[:,:], float64[:]) -> float64[:,:]``.

    Note:
        The Jacobian does not depend on the overall fractions ``z``, since they are
        assumed given and constant, hence only relevant for residual.

    """
    nphase, ncomp = x.shape

    # must fill with zeros, since slightly sparse and below fill-up does not cover
    # elements which are zero
    jac = np.zeros((ncomp - 1, nphase - 1 + nphase * ncomp), dtype=np.float64)

    for i in range(ncomp - 1):
        # (1 - sum_j y_j) x_ir + y_j x_ij is there, per phase
        # hence d mass_i / d y_j = x_ij - x_ir
        jac[i, : nphase - 1] = x[1:, i + 1] - x[0, i + 1]  # i + 1 to skip ref component

        # d.r.t. w.r.t x_ij is always y_j for all j per mass conv.
        jac[i, nphase + i :: nphase] = y  # nphase -1 + i + 1 to skip ref component

    # -1 because of z - z(x,y) = 0
    # and above is dz(x,y) / dyx
    return (-1) * jac


@numba.njit("float64[:](float64[:,:],float64[:])", fastmath=True, cache=True)
def complementary_conditions_res(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""Assembles the residual of the complementary conditions.

    For each phase ``j`` it holds

    ... math::

        y\left[j\right] \cdot \left(1 - \sum_i x\left[j, i\right]\right) = 0

    Number of phases and components is determined from the chape of ``x``.

    NJIT-ed function with signature
    ``(float64[:,:], float64[:]) -> float64[:]``.

    Note:
        See :func:`parse_xyz` for obtaining the properly formatted arguments ``x,y``.

    Parameters:
        x: ``shape=(num_phase, num_comp)``

            Phase compositions
        y: ``shape=(num_phase,)``

            Molar phase fractions.

    Returns:
        An array with ``shape=(num_phase,)`` containg the residual of the complementary
        condition per phase.

    """
    return y * (1 - np.sum(x, axis=1))


@numba.njit("float64[:,:](float64[:,:],float64[:])", fastmath=True, cache=True)
def complementary_conditions_jac(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Returns the Jacobian of the residual described in
    :func:`complementary_conditions_res`

    The Jacobian is of shape ``(num_phase, num_phase - 1 + num_phase * num_comp)``.
    The derivatives (columns) are taken w.r.t. to each independent molar fraction,
    except feed fractions.

    The order of derivatives w.r.t. phase compositions is in phase-major order.

    NJIT-ed function with signature
    ``(float64[:,:], float64[:]) -> float64[:,:]``.

    """
    nphase, ncomp = x.shape
    # must fill with zeros, since matrix sparsely populated.
    d_ccs = np.zeros((nphase, nphase - 1 + nphase * ncomp), dtype=np.float64)

    unities = 1 - np.sum(x, axis=1)

    # first complementary condition is w.r.t. to reference phase
    # (1 - sum_j y_j) * (1 - sum_i x_i0)
    d_ccs[0, : nphase - 1] = (-1) * unities[0]
    d_ccs[0, nphase - 1 : nphase - 1 + ncomp] = y[0] * (-1)
    for j in range(1, nphase):
        # for the other phases, its slight easier since y_j * (1 - sum_i x_ij)
        d_ccs[j, j - 1] = unities[j]
        d_ccs[j, nphase - 1 + j * ncomp : nphase - 1 + (j + 1) * ncomp] = y[j] * (-1)

    return d_ccs


# endregion


class CompiledUnifiedFlash(Flash):
    """A class providing efficient unified flash calculations using numba-compiled
    functions.

    It uses the no-python mode of numba to produce highly efficient, compiled code.

    Flash equations are represented by callable residuals and Jacobians. Various
    flash types are assembled in a modular way by combining required, compiled equations
    into a solvable system.

    Since each system depends on the modelled phases and components, significant
    parts of the equilibrium problem must be compiled on the fly.

    This is a one-time action once the modelling process is completed.

    The supported flash types are than available until destruction.

    Supported flash types/specifications:

    1. ``'p-T'``: state definition in terms of pressure and temperature
    2. ``'p-h'``: state definition in terms of pressure and specific mixture enthalpy
    3. ``'v-h'``: state definition in terms of specific volume and enthalpy of the
       mixture

    Supported mixtures:

    1. non-reactive
    2. only 1 gas and 1 liquid phase
    3. arbitrary many components

    Multiple flash problems can be solved in parallel by passing vectorized state
    definitions.

    Parameters:
        mixture: A mixture model containing modelled components and phases.
        eos_compiler: An EoS compiler instance required to create a
            :class:`~porepy.composite.flash_compiler.FlashCompiler`.

    Raises:
        AssertionError: If not at least 2 components are present.
        AssertionError: If not 2 phases are modelled.

    """

    def __init__(
        self,
        mixture: Mixture,
        eos_compiler: EoSCompiler,
    ) -> None:
        super().__init__(mixture)

        assert self.npnc[0] == 2, "Supports only 2-phase mixtures."
        assert self.npnc[1] >= 2, "Must have at least two components."
        assert set(self.nc_per_phase) == set(
            [self.npnc[1]]
        ), "Supports only unified mixtures (all components in all phases)."

        # data used in initializers
        self._pcrits: list[float] = [comp.p_crit for comp in mixture.components]
        """A list containing critical pressures per component in ``mixture``."""
        self._Tcrits: list[float] = [comp.T_crit for comp in mixture.components]
        """A list containing critical temperatures per component in ``mixture``."""
        self._vcrits: list[float] = [comp.V_crit for comp in mixture.components]
        """A list containing critical volumes per component in ``mixture``."""
        self._omegas: list[float] = [comp.omega for comp in mixture.components]
        """A list containing acentric factors per component in ``mixture``."""
        self._phasetypes: np.ndarray = np.array(
            [phase.type for phase in mixture.phases], dtype=np.int32
        )
        """An array containing the phase types per phase in ``mixture``."""

        self.eos_compiler: EoSCompiler = eos_compiler
        """Assembler and compiler of EoS-related expressions equation.
        passed at instantiation."""

        self.residuals: dict[
            Literal["p-T", "p-h", "v-h"], Callable[[np.ndarray], np.ndarray]
        ] = dict()
        """Contains per flash configuration the respective residuals as a callable."""

        self.jacobians: dict[
            Literal["p-T", "p-h", "v-h"], Callable[[np.ndarray], np.ndarray]
        ] = dict()
        """Contains per flash configuration the respective Jacobian as a callable."""

        self.initializers: dict[
            Literal["p-T", "p-h", "v-h"], Callable[[np.ndarray], np.ndarray]
        ] = dict()
        """Contains per flash configuration the initialization procedure."""

        self.npipm_parameters: dict[str, float] = {
            "eta": 0.5,
            "u1": 1.0,
            "u2": 1.0,
        }
        """A dictionary containing per parameter name (str, key) the respective
        parameter for the NPIPM:

        - ``'eta': 0.5`` linear decline in slack variable
        - ``'u1': 1.`` penalty for violating complementarity
        - ``'u2': 1.`` penalty for violating negativitiy of fractions

        Values can be set directly by modifying the values of this dictionary.

        """

        self.armijo_parameters: dict[str, float] = {
            "kappa": 0.4,
            "rho": 0.99,
            "j_max": 50,
        }
        """A dictionary containing per parameter name (str, key) the respective
        parameter for the Armijo line-search:

        - ``'kappa': 0.4``
        - ``'rho_0': 0.99``
        - ``'j_max': 50`` (maximal number of Armijo iterations)

        Values can be set directly by modifying the values of this dictionary.

        """

        self.initialization_parameters: dict[str, float | int] = {
            "N1": 3,
            "N2": 1,
            "N3": 5,
            "eps": 1e-3,
        }
        """Numbers of iterations for initialization procedures and other configurations

        - ``'N1'``: Int, default is 3. Iterations for fractions guess.
        - ``'N2'``: Int, default is 1. Iterations for state constraint (p/T update).
        - ``'N3'``: int, default is 5. Alterations between fractions guess and  p/T
          update.
        - ``'eps'``: Float, default is 1e-3.
          If not None, performs checks of the flash residual.
          If norm of residual reaches this tolerance, initialization is stopped.
          Used only for flashes other than p-T to unecessarily expensive initialization.

        """

        self._solver_params: dict = convert_param_dict(
            {
                "f_dim": 1,
                "num_phase": self.npnc[0],
                "num_comp": self.npnc[1],
                "tol": self.tolerance,
                "max_iter": self.max_iter,
                "rho": self.armijo_parameters["rho"],
                "kappa": self.armijo_parameters["kappa"],
                "j_max": self.armijo_parameters["j_max"],
                "u1": self.npipm_parameters["u1"],
                "u2": self.npipm_parameters["u2"],
                "eta": self.npipm_parameters["eta"],
            }
        )
        """Typed numba dicitionary to pass parameters to the solver.
        Compiled once, updated once every flash.

        NOTE: This is a numba experimental feature.

        """

    def _parse_and_complete_results(
        self,
        results: np.ndarray,
        flash_type: Literal["p-T", "p-h", "v-h"],
        fluid_state: FluidState,
    ) -> FluidState:
        """Helper function to fill a fluid state with the results from the flash.

        Modifies and returns the passed state structur containing flash
        specifications.

        Also, fills up secondary expressions for respective flash type.

        Sequences of quantities associated with phases, components or derivatives are
        stored as 2D arrays for convenience (row-wise per phase/component/derivative).

        """
        nphase, ncomp = self.npnc

        # Parsing phase compositions and molar phsae fractions
        y: list[np.ndarray] = list()
        x: list[np.ndarray] = list()
        for j in range(nphase):
            # values for molar phase fractions of independent phases
            if j < nphase - 1:
                y.append(results[:, -(1 + nphase * ncomp + nphase - 1) + j])
            # composition of phase j
            x_j = list()
            for i in range(ncomp):
                x_j.append(results[:, -(1 + (nphase - j) * ncomp) + i])
            x.append(np.array(x_j))

        fluid_state.y = np.vstack([1 - safe_sum(y), np.array(y)])

        # If T is unknown, it is always the last unknown before molar fractions
        if "T" not in flash_type:
            fluid_state.T = results[:, -(1 + ncomp * nphase + nphase - 1 + 1)]

        # If v is a defined value, we fetch pressure and saturations
        if "v" in flash_type:
            # If T is additionally unknown to p, p is the second last quantity before
            # molar fractions
            if "T" not in flash_type:
                p_pos = 1 + ncomp * nphase + nphase - 1 + 2
            else:
                p_pos = 1 + ncomp * nphase + nphase - 1 + 1

            fluid_state.p = results[:, -p_pos]

            # saturations are stored before pressure (for independent phases)
            s: list[np.ndarray] = list()
            for j in range(nphase - 1):
                s.append(results[:, -(p_pos + nphase - 1 + j)])
            fluid_state.sat = np.vstack([1 - safe_sum(s), np.array(s)])

        # Computing states for each phase after filling p, T and x
        fluid_state.phases = list()
        for j in range(nphase):
            fluid_state.phases.append(
                self.eos_compiler.compute_phase_state(
                    self._phasetypes[j], fluid_state.p, fluid_state.T, x[j]
                )
            )

        # if v not defined, evaluate saturations based on rho and y
        if "v" not in flash_type:
            fluid_state.evaluate_saturations()
        # Convert sequence to 2D array (sequence of row vectors)
        if not isinstance(fluid_state.sat, np.ndarray):
            fluid_state.sat = np.array(fluid_state.sat)
        # evaluate extensive properties of the fluid mixture
        fluid_state.evaluate_extensive_state()

        return fluid_state

    def _update_solver_params(self, f_dim: int) -> None:
        """Helper function to update the numba-typed solver parameter dict."""
        self._solver_params["f_dim"] = float(f_dim)
        self._solver_params["tol"] = self.tolerance
        self._solver_params["max_iter"] = float(self.max_iter)
        self._solver_params["rho"] = self.armijo_parameters["rho"]
        self._solver_params["kappa"] = self.armijo_parameters["kappa"]
        self._solver_params["j_max"] = float(self.armijo_parameters["j_max"])
        self._solver_params["u1"] = self.npipm_parameters["u1"]
        self._solver_params["u2"] = self.npipm_parameters["u2"]
        self._solver_params["eta"] = self.npipm_parameters["eta"]

    def compile(self, precompile_solvers: bool = False) -> None:
        """Triggers the assembly and compilation of equilibrium equations, including
        the NPIPM approach.

        The order of equations is always as follows:

        1. ``num_comp -1`` mass constraints
        2. ``(num_phase -1) * num_comp`` isofugacity constraints
        3. state constraints (1 for each)
        4. ``num_phase`` complementarity conditions
        5. 1 NPIPM slack equation

        Important:
            This takes a considerable amount of time.
            The compilation is therefore separated from the instantiation of this class.

        Parameters:
            precompile_solvers: ``default=False``

                Highly invasive flag to hack into numba and pre-compile solvers for
                compiled flash systems.

        """

        nphase, ncomp = self.npnc
        npnc = self.npnc
        phasetypes = self._phasetypes

        if 1 in phasetypes:
            # NOTE only 1 expected (first)
            gas_index = phasetypes.tolist().index(1)
        else:
            gas_index = -1

        ## dimension of flash systems, excluding NPIPM
        # number of equations for the pT system
        # ncomp - 1 mass constraints
        # (nphase - 1) * ncomp fugacity constraints (w.r.t. ref phase formulated)
        # nphase complementary conditions
        pT_dim = ncomp - 1 + (nphase - 1) * ncomp + nphase
        # p-h flash: additional var T, additional equ enthalpy constraint
        ph_dim = pT_dim + 1
        # v-h flash: additional vars p, s_j j!= ref
        # additional equations volume constraint and density constraints
        vh_dim = ph_dim + 1 + (nphase - 1)

        logger.info("Compiling flash equations ..")
        logger.debug("Compiling flash equations: EoS functions")

        prearg_val_c = self.eos_compiler.funcs.get("prearg_val", None)
        if prearg_val_c is None:
            prearg_val_c = self.eos_compiler.get_prearg_for_values()
        prearg_jac_c = self.eos_compiler.funcs.get("prearg_jac", None)
        if prearg_jac_c is None:
            prearg_jac_c = self.eos_compiler.get_prearg_for_derivatives()
        phi_c = self.eos_compiler.funcs.get("phi", None)
        if phi_c is None:
            phi_c = self.eos_compiler.get_fugacity_function()
        d_phi_c = self.eos_compiler.funcs.get("d_phi", None)
        if d_phi_c is None:
            d_phi_c = self.eos_compiler.get_dpTX_fugacity_function()
        h_c = self.eos_compiler.funcs.get("h", None)
        if h_c is None:
            h_c = self.eos_compiler.get_enthalpy_function()
        d_h_c = self.eos_compiler.funcs.get("d_h", None)
        if d_h_c is None:
            d_h_c = self.eos_compiler.get_dpTX_enthalpy_function()
        rho_c = self.eos_compiler.funcs.get("rho", None)
        if rho_c is None:
            rho_c = self.eos_compiler.get_density_function()
        d_rho_c = self.eos_compiler.funcs.get("d_rho", None)
        if d_rho_c is None:
            d_rho_c = self.eos_compiler.get_dpTX_density_function()

        @numba.njit("float64[:,:](float64,float64,float64[:,:])")
        def get_prearg_res(p: float, T: float, xn: np.ndarray) -> np.ndarray:
            """Helper function to compute the prearguments for the residual for all
            phases"""

            p_0 = prearg_val_c(phasetypes[0], p, T, xn[0])
            prearg = np.empty((nphase, p_0.shape[0]), dtype=np.float64)
            prearg[0] = p_0
            for j in range(1, nphase):
                prearg[j] = prearg_val_c(phasetypes[j], p, T, xn[j])
            return prearg

        @numba.njit("float64[:,:](float64,float64,float64[:,:])")
        def get_prearg_jac(p: float, T: float, xn: np.ndarray) -> np.ndarray:
            """Helper function to compute the prearguments for the Jacobian for all
            phases"""
            p_0 = prearg_jac_c(phasetypes[0], p, T, xn[0])
            prearg = np.empty((nphase, p_0.shape[0]), dtype=np.float64)
            prearg[0] = p_0
            for j in range(1, nphase):
                prearg[j] = prearg_jac_c(phasetypes[j], p, T, xn[j])
            return prearg

        logger.debug("Compiling flash equations: isofugacity constraints")

        @numba.njit(
            "float64[:](float64[:,:], float64, float64, float64[:,:], float64[:,:])"
        )
        def isofug_constr_c(
            prearg: np.ndarray,
            p: float,
            T: float,
            X: np.ndarray,
            Xn: np.ndarray,
        ):
            """Helper function to assemble the isofugacity constraint.

            Formulation is always w.r.t. the reference phase r, assumed to be r=0.

            """
            isofug = np.empty(ncomp * (nphase - 1), dtype=np.float64)

            phi_r = phi_c(prearg[0], p, T, Xn[0])

            for j in range(1, nphase):
                phi_j = phi_c(prearg[j], p, T, Xn[j])
                # isofugacity constraint between phase j and phase r
                # NOTE fugacities are evaluated with normalized fractions
                isofug[(j - 1) * ncomp : j * ncomp] = X[j] * phi_j - X[0] * phi_r

            return isofug

        @numba.njit(
            "float64[:,:](float64[:],float64[:],float64,float64,float64[:],float64[:])",
        )
        def d_isofug_block_j(
            prearg_res_j: np.ndarray,
            prearg_jac_j: np.ndarray,
            p: float,
            T: float,
            X: np.ndarray,
            Xn: np.ndarray,
        ):
            """Helper function to construct a block representing the derivative
            of x_ij * phi_ij for all i as a matrix, with i row index.
            This is constructed for a given phase j.
            """

            phi_j = phi_c(prearg_res_j, p, T, Xn)
            d_phi_j = d_phi_c(prearg_res_j, prearg_jac_j, p, T, Xn)
            # NOTE phi depends on normalized fractions
            # extending derivatives from normalized fractions to extended ones
            for i in range(ncomp):
                d_phi_j[i] = _extend_fractional_derivatives(d_phi_j[i], X)

            # product rule: x * dphi
            d_xphi_j = (d_phi_j.T * X).T
            # + phi * dx  (minding the first two columns which contain the dp dT)
            d_xphi_j[:, 2:] += np.diag(phi_j)

            return d_xphi_j

        @numba.njit(
            "float64[:,:](float64[:,:],float64[:, :],"
            + "float64,float64,float64[:,:],float64[:,:])"
        )
        def d_isofug_constr_c(
            prearg_res: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            X: np.ndarray,
            Xn: np.ndarray,
        ):
            """Helper function to assemble the derivative of the isofugacity constraints

            Formulation is always w.r.t. the reference phase r, assumed to be zero 0.

            Important:
                The derivative is taken w.r.t. to A, B, Z (among others).
                An forward expansion must be done after a call to this function.

            """
            d_iso = np.zeros((ncomp * (nphase - 1), 2 + ncomp * nphase))

            # creating derivative parts involving the reference phase
            d_xphi_r = d_isofug_block_j(prearg_res[0], prearg_jac[0], p, T, X[0], Xn[0])

            for j in range(1, nphase):
                # construct the same as above for other phases
                d_xphi_j = d_isofug_block_j(
                    prearg_res[1], prearg_jac[1], p, T, X[j], Xn[j]
                )

                # p, T derivative
                d_iso[(j - 1) * ncomp : j * ncomp, :2] = (
                    d_xphi_j[:, :2] - d_xphi_r[:, :2]
                )
                # remember: d(x_ij * phi_ij - x_ir * phi_ir)
                # hence every row-block contains (-1)* d_xphi_r
                # derivative w.r.t. fractions in reference phase
                d_iso[(j - 1) * ncomp : j * ncomp, 2 : 2 + ncomp] = -d_xphi_r[:, 2:]
                # derivatives w.r.t. fractions in independent phase j
                d_iso[
                    (j - 1) * ncomp : j * ncomp, 2 + j * ncomp : 2 + (j + 1) * ncomp
                ] = d_xphi_j[:, 2:]

            return d_iso

        logger.debug("Compiling flash equations: enthalpy constraint")

        @numba.njit(
            "float64(float64[:,:],float64,float64,float64,float64[:],float64[:,:])"
        )
        def h_constr_res_c(
            prearg: np.ndarray,
            p: float,
            h: float,
            T: float,
            y: np.ndarray,
            xn: np.ndarray,
        ) -> float:
            """Helper function to evaluate the normalized residual of the enthalpy
            constraint.

            Note that ``h`` is the target value.

            """

            h_constr_res = h
            for j in range(xn.shape[0]):
                h_constr_res -= y[j] * h_c(prearg[j], p, T, xn[j])

            # for better conditioning, normalize enthalpy constraint
            h_constr_res /= h

            return h_constr_res

        @numba.njit(
            "float64[:]"
            + "(float64[:,:],float64[:,:],float64,float64,float64,"
            + "float64[:],float64[:,:],float64[:,:])"
        )
        def h_constr_jac_c(
            prearg_res: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            h: float,
            T: float,
            y: np.ndarray,
            x: np.ndarray,
            xn: np.ndarray,
        ) -> np.ndarray:
            """Function to assemble the gradient of the enthalpy constraint w.r.t.
            temperature, molar phase fractions and extended phase compositions."""
            # gradient of sum_j y_j h_j(p, T, x_j)  w.r.t. p, T, y, x
            h_constr_jac = np.zeros(2 + nphase - 1 + nphase * ncomp)

            # treatment of reference phase enthalpy
            # enthalpy and its gradient of the reference phase
            h_0 = h_c(prearg_res[0], p, T, xn[0])
            # gradient of h_0 w.r.t to extended fraction
            d_h_0 = _extend_fractional_derivatives(
                d_h_c(prearg_res[0], prearg_jac[0], p, T, xn[0]), x[0]
            )
            # contribution to p- and T-derivative of reference phase
            h_constr_jac[:2] = y[0] * d_h_0[:2]
            # y_0 = 1 - y_1 - y_2 ..., contribution is below
            # derivative w.r.t. composition in reference phase
            h_constr_jac[2 + nphase - 1 : 2 + nphase - 1 + ncomp] = y[0] * d_h_0[2:]

            for j in range(1, nphase):
                h_j = h_c(prearg_res[j], p, T, xn[j])
                d_h_j = _extend_fractional_derivatives(
                    d_h_c(prearg_res[j], prearg_jac[j], p, T, xn[j]), x[j]
                )
                # contribution to p- and T-derivative of phase j
                h_constr_jac[:2] += y[j] * d_h_j[:2]

                # derivative w.r.t. y_j
                h_constr_jac[1 + j] = h_j - h_0  # because y_0 = 1 - y_1 - y_2 ...

                # derivative w.r.t. composition of phase j
                h_constr_jac[
                    2 + nphase - 1 + j * ncomp : 2 + nphase - 1 + (j + 1) * ncomp
                ] = (y[j] * d_h_j[2:])

            # constraints is h_target - h_mix
            h_constr_jac *= -1.0

            # for better conditioning
            h_constr_jac /= h

            return h_constr_jac

        logger.debug("Compiling flash equations: volume constraint")

        @numba.njit(
            "float64[:]"
            + "(float64[:,:],float64,float64,float64,"
            + "float64[:],float64[:],float64[:,:])"
        )
        def v_constr_res_c(
            prearg: np.ndarray,
            v: float,
            p: float,
            T: float,
            sat: np.ndarray,
            y: np.ndarray,
            xn: np.ndarray,
        ) -> np.ndarray:
            """Helper function to evaluate the residual of the volume constraint,
            including the phase fraction relations."""
            rho_j = np.empty(nphase, dtype=np.float64)
            for j in range(nphase):
                rho_j[j] = rho_c(prearg[j], p, T, xn[j])
            rho_mix = (sat * rho_j).sum()

            res = np.empty(nphase, dtype=np.float64)
            # volume constraint
            res[0] = v * rho_mix - 1
            # nphase - 1 phase fraction relations
            res[1:] = (y - sat * rho_j / rho_mix)[1:]

            return res

        @numba.njit(
            "float64[:,:]"
            + "(float64[:,:],float64[:,:],float64,float64,float64,"
            + "float64[:],float64[:,:],float64[:,:])"
        )
        def v_constr_jac_c(
            prearg_res: np.ndarray,
            prearg_jac: np.ndarray,
            v: float,
            p: float,
            T: float,
            sat: np.ndarray,
            x: np.ndarray,
            xn: np.ndarray,
        ) -> np.ndarray:
            """Helper function to compute the Jacobian of the volume constraint and
            phase fraction relations.
            Returns derivatives w.r.t. sat, p, T, y, and x."""

            rho_j = np.empty(nphase, dtype=np.float64)
            d_rho_j = np.empty((nphase, 2 + ncomp), dtype=np.float64)
            dpT_rho_mix = np.zeros(2, dtype=np.float64)

            for j in range(nphase):
                rho_j[j] = rho_c(prearg_res[j], p, T, xn[j])
                d_rho_j[j] = _extend_fractional_derivatives(
                    d_rho_c(prearg_res[j], prearg_jac[j], p, T, xn[j]), x[j]
                )
                dpT_rho_mix += sat[j] * d_rho_j[j, :2]

            # rho_mix = sum_i s_i * rho_i
            rho_mix = (sat * rho_j).sum()

            # 1 volume constraint, nphase-1 phase fraction relations, all derivatives
            jac = np.zeros((nphase, 2 * nphase + ncomp * nphase), dtype=np.float64)

            # derivatives of volume constraint w.r.t. independent s_j
            # s_r = 1 - sum_j!=r s_j
            # and v * (sum_i s_i * rho_i) - 1 = 0
            jac[0, : nphase - 1] = rho_j[1:] - rho_j[0]
            # derivatives of volume constraint w.r.t. p and T
            jac[0, nphase - 1 : nphase + 1] = dpT_rho_mix
            # derivatives of v constr w.r.t. x_r
            jac[0, 2 * nphase : 2 * nphase + ncomp] = sat[0] * d_rho_j[0, 2:]

            for j in range(1, nphase):
                # derivatives volume constr w.r.t. x_j for independent phases
                jac[0, 2 * nphase + j * ncomp : 2 * nphase + (j + 1) * ncomp] = (
                    sat[j] * d_rho_j[j, 2:]
                )

                # outer derivative of rho_j * dpTx(1 / rho_mix)
                outer_j = -rho_j[j] / rho_mix**2

                # derivatives of phase fraction relations for each independent phase.
                # y_j - sat_j * rho_j / (sum_i s_i * rho_i)
                # First, derivatives w.r.t. saturations
                # With s_0 = 1 - sum_(i > 0) s_i it holds for k > 0
                # ds_k (s_j rho_j / (sum_i s_i * rho_i)) =
                # delta_kj * rho_j / (sum_i s_i * rho_i)
                # + s_j * (- rho_j / (sum_i s_i * rho_i)^2 * (rho_k - rho_0))
                jac[j, : nphase - 1] = sat[j] * outer_j * (rho_j[1:] - rho_j[0])
                jac[j, j - 1] += rho_j[j] / rho_mix

                # derivatives of phase fraction relations w.r.t. p, T
                # With s_0 = 1 - sum_(i > 0) s_i and rho_mix = sum_i s_i * rho_i
                # dpt (rho_j(p, T) / rho_mix) =
                # dpt(rho_j(p,T)) / rho_mix
                # + rho_j * (-1 / rho_mix^2 * dpt(rho_mix))
                jac[j, nphase - 1 : nphase + 1] = sat[j] * (
                    d_rho_j[j, :2] / rho_mix + outer_j * dpT_rho_mix
                )

                # derivatives of phase fraction relation w.r.t. x_ik
                # for all phases k, and j > 0
                # dx_ik (rho_j(x_ij) / rho_mix) =
                # delta_kj * (dx_ik(rho_j(x_ij)) / rho_mix)
                # + rho_j * (-1 / rho_mix^2 * (
                #   dx_ik(sum_l s_l rho_l(x_il)))
                # )
                for k in range(nphase):
                    jac[j, 2 * nphase + k * ncomp : 2 * nphase + (k + 1) * ncomp] = (
                        sat[j] * outer_j * sat[k] * d_rho_j[k, 2:]
                    )
                    if k == j:
                        jac[
                            j, 2 * nphase + k * ncomp : 2 * nphase + (k + 1) * ncomp
                        ] += (sat[j] * d_rho_j[k, 2:] / rho_mix)

            # volume constraint is scaled with target volume
            jac[0] *= v

            # multiply fraction relations with -1 because y_j (-) s_j rho_j / rho_mix
            jac[1:] *= -1
            # derivatives of phase fraction relations w.r.t. independent y_j
            jac[1:, nphase + 1 : 2 * nphase] = np.eye(nphase - 1)

            return jac

        logger.debug("Compiling flash equations: p-T flash")

        @numba.njit("float64[:](float64[:])")
        def F_pT(X_gen: np.ndarray) -> np.ndarray:
            x, y, z = parse_xyz(X_gen, npnc)
            p, T = parse_pT(X_gen, npnc)

            # declare residual array of proper dimension
            res = np.empty(pT_dim, dtype=np.float64)

            res[: ncomp - 1] = mass_conservation_res(x, y, z)
            # last nphase equations are always complementary conditions
            res[-nphase:] = complementary_conditions_res(x, y)

            # EoS specific computations
            xn = normalize_rows(x)
            prearg = get_prearg_res(p, T, xn)

            res[ncomp - 1 : ncomp - 1 + ncomp * (nphase - 1)] = isofug_constr_c(
                prearg, p, T, x, xn
            )

            return res

        @numba.njit("float64[:,:](float64[:])")
        def DF_pT(X_gen: np.ndarray) -> np.ndarray:
            x, y, _ = parse_xyz(X_gen, npnc)
            p, T = parse_pT(X_gen, npnc)

            # declare Jacobian of proper dimension
            jac = np.zeros((pT_dim, pT_dim), dtype=np.float64)

            jac[: ncomp - 1] = mass_conservation_jac(x, y)
            # last nphase equations are always complementary conditions
            jac[-nphase:] = complementary_conditions_jac(x, y)

            # EoS specific computations
            xn = normalize_rows(x)
            prearg_res = get_prearg_res(p, T, xn)
            prearg_jac = get_prearg_jac(p, T, xn)

            jac[ncomp - 1 : ncomp - 1 + ncomp * (nphase - 1), nphase - 1 :] = (
                d_isofug_constr_c(prearg_res, prearg_jac, p, T, x, xn)[:, 2:]
            )

            return jac

        logger.debug("Compiling flash equations: p-h flash")

        @numba.njit("float64[:](float64[:])")
        def F_ph(X_gen: np.ndarray) -> np.ndarray:
            x, y, z = parse_xyz(X_gen, npnc)
            h, p = parse_target_state(X_gen, npnc)
            _, T = parse_pT(X_gen, npnc)

            # declare residual array of proper dimension
            res = np.empty(ph_dim, dtype=np.float64)

            res[: ncomp - 1] = mass_conservation_res(x, y, z)

            # complementarity always last for NPIPM to work
            res[-nphase:] = complementary_conditions_res(x, y)

            # EoS specific computations
            xn = normalize_rows(x)
            prearg = get_prearg_res(p, T, xn)

            res[ncomp - 1 : ncomp - 1 + ncomp * (nphase - 1)] = isofug_constr_c(
                prearg, p, T, x, xn
            )
            # state constraints always after isofug and befor complementary cond.
            res[-(nphase + 1)] = h_constr_res_c(prearg, p, h, T, y, xn) / T**2

            return res

        @numba.njit("float64[:,:](float64[:])")
        def DF_ph(X_gen: np.ndarray) -> np.ndarray:
            x, y, _ = parse_xyz(X_gen, npnc)
            h, p = parse_target_state(X_gen, npnc)
            _, T = parse_pT(X_gen, npnc)

            # declare Jacobian of proper dimension
            jac = np.zeros((ph_dim, ph_dim), dtype=np.float64)

            jac[: ncomp - 1, 1:] = mass_conservation_jac(x, y)
            # last nphase equations are always complementary conditions
            jac[-nphase:, 1:] = complementary_conditions_jac(x, y)

            # EoS specific computations
            xn = normalize_rows(x)
            prearg_res = get_prearg_res(p, T, xn)
            prearg_jac = get_prearg_jac(p, T, xn)

            d_iso = d_isofug_constr_c(prearg_res, prearg_jac, p, T, x, xn)

            # derivatives w.r.t. T
            jac[ncomp - 1 : ncomp - 1 + ncomp * (nphase - 1), 0] = d_iso[:, 1]
            # derivatives w.r.t. fractions
            jac[ncomp - 1 : ncomp - 1 + ncomp * (nphase - 1), nphase:] = d_iso[:, 2:]

            d_h_constr = h_constr_jac_c(prearg_res, prearg_jac, p, h, T, y, x, xn)

            d_h_constr /= T**2
            d_h_constr[1] += 2.0 / T**3 * h_constr_res_c(prearg_res, p, h, T, y, xn)
            jac[-(nphase + 1)] = d_h_constr[1:]  # exclude dp

            return jac

        logger.debug("Compiling flash equations: v-h flash")

        @numba.njit("float64[:](float64[:])")
        def F_vh(X_gen: np.ndarray) -> np.ndarray:
            x, y, z = parse_xyz(X_gen, npnc)
            v, h = parse_target_state(X_gen, npnc)
            p, T = parse_pT(X_gen, npnc)
            sat = parse_sat(X_gen, npnc)

            # declare residual array of proper dimension
            res = np.empty(vh_dim, dtype=np.float64)

            res[: ncomp - 1] = mass_conservation_res(x, y, z)

            # complementarity always last for NPIPM to work
            res[-nphase:] = complementary_conditions_res(x, y)

            # EoS specific computations
            xn = normalize_rows(x)
            prearg = get_prearg_res(p, T, xn)

            res[ncomp - 1 : ncomp - 1 + ncomp * (nphase - 1)] = isofug_constr_c(
                prearg, p, T, x, xn
            )

            # state constraints always after isofug and befor complementary cond.
            # h constraint
            res[ncomp - 1 + ncomp * (nphase - 1)] = (
                h_constr_res_c(prearg, p, h, T, y, xn) / T**2
            )
            # v constraint including closure for saturations (rho y_j = rho_j s_j)
            res[ncomp + ncomp * (nphase - 1) : -nphase] = v_constr_res_c(
                prearg, v, p, T, sat, y, xn
            )

            return res

        @numba.njit("float64[:,:](float64[:])")
        def DF_vh(X_gen: np.ndarray) -> np.ndarray:
            x, y, _ = parse_xyz(X_gen, npnc)
            v, h = parse_target_state(X_gen, npnc)
            p, T = parse_pT(X_gen, npnc)
            sat = parse_sat(X_gen, npnc)

            # declare Jacobian of proper dimension
            jac = np.zeros((vh_dim, vh_dim), dtype=np.float64)

            jac[: ncomp - 1, nphase + 1 :] = mass_conservation_jac(x, y)
            # last nphase equations are always complementary conditions
            jac[-nphase:, nphase + 1 :] = complementary_conditions_jac(x, y)

            # EoS specific computations
            xn = normalize_rows(x)
            prearg_res = get_prearg_res(p, T, xn)
            prearg_jac = get_prearg_jac(p, T, xn)

            # isofugacity constraints
            d_iso = d_isofug_constr_c(prearg_res, prearg_jac, p, T, x, xn)
            # derivatives w.r.t. p, T
            jac[
                ncomp - 1 : ncomp - 1 + ncomp * (nphase - 1), nphase - 1 : nphase + 1
            ] = d_iso[:, :2]
            # derivatives w.r.t. x
            jac[ncomp - 1 : ncomp - 1 + ncomp * (nphase - 1), 2 * nphase :] = d_iso[
                :, 2:
            ]

            # enthalpy constraint
            d_h_constr = h_constr_jac_c(prearg_res, prearg_jac, p, h, T, y, x, xn)
            d_h_constr /= T**2
            d_h_constr[1] += 2.0 / T**3 * h_constr_res_c(prearg_res, p, h, T, y, xn)
            jac[ncomp - 1 + ncomp * (nphase - 1), nphase - 1 :] = d_h_constr

            # volume constraint
            jac[ncomp + ncomp * (nphase - 1) : -nphase] = v_constr_jac_c(
                prearg_res, prearg_jac, v, p, T, sat, x, xn
            )

            return jac

        self.residuals.update(
            {
                "p-T": F_pT,
                "p-h": F_ph,
                "v-h": F_vh,
            }
        )

        self.jacobians.update(
            {
                "p-T": DF_pT,
                "p-h": DF_ph,
                "v-h": DF_vh,
            }
        )

        p_crits = np.array(self._pcrits)
        T_crits = np.array(self._Tcrits)
        v_crits = np.array(self._vcrits)
        omegas = np.array(self._omegas)

        logger.debug("Compiling flash initialization: p-T")

        @numba.njit("float64[:](float64[:],int32,int32)")
        def guess_fractions(
            X_gen: np.ndarray, N1: int, guess_K_vals: int
        ) -> np.ndarray:
            """Guessing fractions for a single flash configuration"""
            x, y, z = parse_xyz(X_gen, npnc)
            p, T = parse_pT(X_gen, npnc)

            # pseudo-critical quantities
            T_pc = np.sum(z * T_crits)
            p_pc = np.sum(z * p_crits)

            # storage of K-values (first phase assumed reference phase)
            K = np.zeros((nphase - 1, ncomp))
            K_tol = 1e-10  # tolerance to bind K-values away from 0

            if guess_K_vals != 0:
                for j in range(nphase - 1):
                    K[j, :] = (
                        np.exp(5.37 * (1 + omegas) * (1 - T_crits / T)) * p_crits / p
                        + K_tol
                    )
            else:
                xn = normalize_rows(x)
                prearg = get_prearg_res(p, T, xn)
                # fugacity coefficients reference phase
                phi_r = phi_c(prearg[0], p, T, xn[0])
                for j in range(1, nphase):
                    phi_j = phi_c(prearg[j], p, T, xn[j])
                    K_jr = phi_r / phi_j + K_tol
                    K[j - 1, :] = K_jr

            # starting iterations using Rachford Rice
            for n in range(N1):
                # solving RR for molar phase fractions
                if nphase == 2:
                    # only one independent phase assumed
                    K_ = K[0]
                    if ncomp == 2:
                        y_ = _rr_binary_vle_inversion(z, K_)
                    else:  # TODO  efficient BRENT method (scipy.optimize.brentq)
                        raise NotImplementedError(
                            "Multicomponent RR solution not implemented."
                        )

                    # copy the original value s.t. different corrections
                    # do not interfer with eachother
                    # _y = float(y_)
                    negative = y_ < 0.0
                    exceeds = y_ > 1.0
                    invalid = exceeds | negative

                    # correction of invalid gas phase values
                    if invalid:
                        # assuming gas saturated for correction using RR potential
                        y_test = np.array([0.0, 1.0], dtype=np.float64)
                        rr_pot = _rr_potential(z, y_test, K)
                        # checking if y is feasible
                        # for more information see Equation 10 in
                        # `Okuno et al. (2010) <https://doi.org/10.2118/117752-PA>`_
                        t_i = _rr_poles(y_test, K)
                        cond_1 = t_i - z >= 0.0
                        # tests holds for arbitrary number of phases
                        # reflected by implementation, despite nph == 2
                        cond_2 = K * z - t_i <= 0.0
                        gas_feasible = np.all(cond_1) & np.all(cond_2)

                        if rr_pot > 0.0:
                            y_ = 0.0
                        elif (rr_pot < 0.0) & gas_feasible:
                            y_ = 1.0

                        # clearly liquid
                        if (T < T_pc) & (p > p_pc):
                            y_ = 0.0
                        # clearly gas
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
                        # the liquid phase is clearly saturated
                        # Vice versa, if fraction above 1 and K-values greater than 1.
                        # the gas phase is clearly saturated
                        if negative & np.all(K_ < 1.0):
                            y_ = 0.0
                        elif exceeds & np.all(K_ > 1.0):
                            y_ = 1.0

                        # assert corrections did what they have to do
                        assert (
                            0.0 <= y_ <= 1.0
                        ), "y fraction estimate outside bound [0, 1]."
                    y[1] = y_
                    y[0] = 1.0 - y_
                else:
                    raise NotImplementedError(
                        "Fractions guess for more than 2 phases not implemented."
                    )

                # resolve compositions
                t = _rr_poles(y, K)
                x[0] = z / t  # fraction in reference phase
                x[1:] = K * x[0]  # fraction in indp. phases

                # update K-values if another iteration comes
                if n < N1 - 1:
                    xn = normalize_rows(x)
                    prearg = get_prearg_res(p, T, xn)
                    # fugacity coefficients reference phase
                    phi_r = phi_c(prearg[0], p, T, xn[0])
                    for j in range(1, nphase):
                        phi_j = phi_c(prearg[j], p, T, xn[j])
                        K_jr = phi_r / phi_j + K_tol
                        K[j - 1, :] = K_jr

            return insert_xy(X_gen, x, y, npnc)

        @numba.njit("float64[:,:](float64[:,:],int32,int32)", parallel=True)
        def pT_initializer(X_gen: np.ndarray, N1: int, guess_K_vals: int) -> np.ndarray:
            """p-T initializer as a parallelized loop over all flash configurations."""
            nf = X_gen.shape[0]
            for f in numba.prange(nf):
                # for f in range(nf):
                X_gen[f] = guess_fractions(X_gen[f], N1, guess_K_vals)
            return X_gen

        logger.debug("Compiling flash initialization: p-h")

        @numba.njit("float64[:](float64[:],int32,float64)")
        def update_T_guess(X_gen: np.ndarray, N2: int, eps: float) -> np.ndarray:
            """Updating T guess by iterating on h-constr w.r.t. T using Newton and some
            corrections"""
            x, y, _ = parse_xyz(X_gen, npnc)
            p, T = parse_pT(X_gen, npnc)
            h, _ = parse_target_state(X_gen, npnc)
            xn = normalize_rows(x)
            h_j = np.empty(nphase, dtype=np.float64)
            dT_h_j = np.empty(nphase, dtype=np.float64)

            for _ in range(N2):
                prearg_res = get_prearg_res(p, T, xn)
                prearg_jac = get_prearg_jac(p, T, xn)
                for j in range(nphase):
                    h_j[j] = h_c(prearg_res[j], p, T, xn[j])
                    dT_h_j[j] = d_h_c(prearg_res[j], prearg_jac[j], p, T, xn[j])[1]

                # h_constr_res = h_constr_res_c(prearg_res, p, h, T, y, xn)
                h_mix = (h_j * y).sum()
                h_constr_res = 1 - h_mix / h
                if np.abs(h_constr_res) < eps:
                    break
                else:
                    dT_h_constr = -(dT_h_j * y).sum() / h
                    dT = 0 - h_constr_res / dT_h_constr  # Newton iteration

                    # corrections to unfeasible updates because of decoupling
                    if np.abs(dT) > T:
                        dT = 0.1 * T * np.sign(dT)
                    dT *= 1 - np.abs(dT) / T
                    # correction if gas phase is present and mixture enthalpy is too low
                    # to avoid overshooting T update
                    if h_mix < h and y[gas_index] > 1e-3:
                        dT *= 0.4
                    T += dT

            return insert_pT(X_gen, p, T, npnc)

        @numba.njit(
            "float64[:,:](float64[:,:],int32,int32,int32,float64)",
            parallel=True,
        )
        def ph_initializer(
            X_gen: np.ndarray, N1: int, N2: int, N3: int, eps: float
        ) -> np.ndarray:
            """p-h initializer as a parallelized loop over all configurations"""
            nf = X_gen.shape[0]
            for f in numba.prange(nf):
                xf = X_gen[f]
                _, _, z = parse_xyz(xf, npnc)
                p, _ = parse_pT(xf, npnc)
                # pseudo-critical T approximation as start
                T_pc = (T_crits * z).sum()
                xf = insert_pT(xf, p, T_pc, npnc)
                xf = guess_fractions(xf, N1, 1)

                for _ in range(N3):
                    xf = update_T_guess(xf, N2, eps)
                    xf = guess_fractions(xf, N1, 0)

                    # abort if residual already small enough
                    if np.linalg.norm(F_ph(xf)) <= eps:
                        break

                X_gen[f] = xf
            return X_gen

        logger.debug("Compiling flash initialization: v-h")

        @numba.njit("float64[:](float64[:],int32,float64)")
        def update_pT_guess(X_gen: np.ndarray, N2: int, eps: float) -> np.ndarray:
            """Helper function to update p-T guess for v-h flash by solving respective
            equations using Newton and some corrections."""

            res = np.empty(nphase + 1)
            jac = np.zeros((nphase + 1, nphase + 1))

            x, y, _ = parse_xyz(X_gen, npnc)
            xn = normalize_rows(x)
            v, h = parse_target_state(X_gen, npnc)
            if gas_index >= 0:
                y_g = y[gas_index]
            else:
                y_g = 0.0

            p, T = parse_pT(X_gen, npnc)

            rho_j = np.empty(nphase)
            h_j = np.empty(nphase)

            for _ in range(N2):
                # Assembling volume and enthalpy constraints with derivatives for s-p-T

                prearg_res = get_prearg_res(p, T, xn)
                prearg_jac = get_prearg_jac(p, T, xn)

                for j in range(nphase):
                    rho_j[j] = rho_c(prearg_res[j], p, T, xn[j])
                    h_j[j] = h_c(prearg_res[j], p, T, xn[j])

                sat = _compute_saturations(y, rho_j, 1e-10)
                v_mix = 1.0 / (sat * rho_j).sum()
                h_mix = (y * h_j).sum()

                res[0] = h_constr_res_c(prearg_res, p, h, T, y, xn)
                res[1:] = v_constr_res_c(prearg_res, v, p, T, sat, y, xn)

                jac[0, nphase - 1 :] = h_constr_jac_c(
                    prearg_res, prearg_jac, p, h, T, y, x, xn
                )[:2]
                jac[1:] = v_constr_jac_c(prearg_res, prearg_jac, v, p, T, sat, x, xn)[
                    :, : nphase + 1
                ]

                if np.linalg.norm(res) <= eps:
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
                    if y_g > 1e-3 and v_mix > v:
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
                    if y_g >= 1.0 and v_mix > v:
                        p_ = p * (2 - fp)
                    # correction for liquid-like mixtures, h is very sensitive to p
                    # because h = u + pv, v small (liquid)
                    # then cancel the update
                    if y_g < 1e-1 and h_mix < h and p_ > p:
                        p_ = p
                    #     if p_ > 0.:
                    #         p_ *= 1.1
                    #     else:  # if for some reason negative, kick up
                    #         p_ = p * 1.1

                    p = p_
                    T = T_

            return insert_pT(X_gen, p, T, npnc)

        @numba.njit(
            "float64[:,:](float64[:,:],int32,int32,int32,float64)",
            parallel=True,
        )
        def vh_initializer(
            X_gen: np.ndarray, N1: int, N2: int, N3: int, eps: float
        ) -> np.ndarray:
            """v-h initializer as a parallelized loop ovr all configurations."""
            nf = X_gen.shape[0]
            for f in numba.prange(nf):
                xf = X_gen[f]
                _, _, z = parse_xyz(xf, npnc)
                v, _ = parse_target_state(xf, npnc)
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

                R = v_pc / v
                if R > 1:  # liquid-like gas
                    Z = 0.2
                    # T correction for liquid-like guess
                    T = T / np.sqrt(R)
                else:  # gas-like
                    Z = 0.7

                p = Z * T * R_IDEAL / v

                xf = insert_pT(xf, p, T, npnc)
                xf = guess_fractions(xf, 3, 1)

                if gas_index >= 0:
                    _, y, _ = parse_xyz(xf, npnc)
                    y_g = y[gas_index]
                else:
                    y_g = 0.0
                if y_g < 1e-3:  # correction if no gas present
                    p *= 0.7
                    # T *= 1.1
                    xf = insert_pT(xf, p, T, npnc)

                # refine fraction guess
                xf = guess_fractions(xf, N1, 0)

                for _ in range(N3):
                    # p-T update
                    xf = update_pT_guess(xf, N2, eps)
                    xf = guess_fractions(xf, N1, 0)

                    # abort if residual already small enough
                    res = F_vh(xf)
                    if np.linalg.norm(res) <= eps:
                        break

                # final saturation update
                x, y, _ = parse_xyz(xf, npnc)
                xn = normalize_rows(x)
                p, T = parse_pT(xf, npnc)
                rho = np.empty(nphase, dtype=np.float64)
                for j in range(nphase):
                    rho[j] = rho_c(
                        prearg_val_c(phasetypes[j], p, T, xn[j]), p, T, xn[j]
                    )
                sat = _compute_saturations(y, rho, 1e-10)
                X_gen[f] = insert_sat(xf, sat[1:], npnc)

            return X_gen

        self.initializers.update(
            {
                "p-T": pT_initializer,
                "p-h": ph_initializer,
                "v-h": vh_initializer,
            }
        )

        if precompile_solvers:
            logger.debug("Compiling solvers ..")

            # pre compile for p-T flash
            gen_arg_dim = ncomp + 1 + pT_dim
            X = np.ones((1, gen_arg_dim))
            self._update_solver_params(pT_dim + 1)
            linear_solver._compile_for_args(X, F_pT, DF_pT, self._solver_params)
            parallel_solver._compile_for_args(X, F_pT, DF_pT, self._solver_params)

            # pre compile for p-h flash
            gen_arg_dim = ncomp + 1 + ph_dim
            X = np.ones((1, gen_arg_dim))
            self._update_solver_params(ph_dim + 1)
            linear_solver._compile_for_args(X, F_ph, DF_ph, self._solver_params)
            parallel_solver._compile_for_args(X, F_ph, DF_ph, self._solver_params)

            # pre-compile for v-h flash
            gen_arg_dim = ncomp + 1 + vh_dim
            X = np.ones((1, gen_arg_dim))
            self._update_solver_params(vh_dim + 1)
            linear_solver._compile_for_args(X, F_vh, DF_vh, self._solver_params)
            parallel_solver._compile_for_args(X, F_vh, DF_vh, self._solver_params)

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
        """Performes the flash for given feed fractions and state definition.

        Assumes the first fraction

        Supported equilibrium definitions:

        - p-T
        - p-h
        - v-h

        Supported parameters:

        - ``'mode'``: Mode of solving the equilibrium problems for multiple state
          definitions given by vectorized input.

          - ``'linear'``: A classical loop over state defintions (row-wise).
          - ``'parallel'``: A parallelized loop, intended for larger amounts of
            problems.

            Defaults to ``'linear'``.

        Raises:
            NotImplementedError: If an unsupported combination or insufficient number of
                of thermodynamic states is passed.

        """
        mode = parameters.get("mode", "linear")
        assert mode in ["linear", "parallel"], f"Unsupported mode {mode}."

        logger.debug("Flash: Parsing input ..")

        nphase, ncomp = self.npnc
        fluid_state, flash_type, f_dim, NF = self.parse_flash_input(
            z, p, T, h, v, initial_state
        )

        assert flash_type in [
            "p-T",
            "p-h",
            "v-h",
        ], f"Unsupported flash type {flash_type}"

        # Because of NPIPM, we have an additiona slack variable
        f_dim += 1
        # the generic argument has additional ncomp - 1 feed fractions and 2 states
        gen_arg_dim = ncomp + 1 + f_dim
        # vectorized, generic flash argument
        X0 = np.zeros((NF, gen_arg_dim))

        # Filling the feed fractions into X0
        i = 0
        for i_, _ in enumerate(fluid_state.z):
            if i_ != self._ref_component_idx:
                X0[:, i] = fluid_state.z[i_]
                i += 1
        # Filling the fixed state values in X0, getting initialization args
        if flash_type == "p-T":
            X0[:, ncomp - 1] = fluid_state.p
            X0[:, ncomp] = fluid_state.T
            init_args = (self.initialization_parameters["N1"], 1)
        elif flash_type == "p-h":
            # has a reverse order of states, because of parse_pT
            X0[:, ncomp - 1] = fluid_state.h
            X0[:, ncomp] = fluid_state.p
            init_args = (
                self.initialization_parameters["N1"],
                self.initialization_parameters["N2"],
                self.initialization_parameters["N3"],
                self.initialization_parameters["eps"],
            )
        elif flash_type == "v-h":
            X0[:, ncomp - 1] = fluid_state.v
            X0[:, ncomp] = fluid_state.h
            init_args = (
                self.initialization_parameters["N1"],
                self.initialization_parameters["N2"],
                self.initialization_parameters["N3"],
                self.initialization_parameters["eps"],
            )
        else:
            assert False, "Missing logic"

        logger.info(f"{flash_type} Flash: Initialization ..")
        if initial_state is None:
            start = time.time()
            # exclude NPIPM variable (last column) from initialization
            X0[:, :-1] = self.initializers[flash_type](X0[:, :-1], *init_args)
            end = time.time()
            init_time = end - start
            logger.debug(f"Flash initialized (elapsed time: {init_time} (s)).")
        else:
            init_time = 0.0
            # parsing phase compositions and molar fractions
            idx = 0
            for j in range(nphase):
                # values for molar phase fractions except for reference phase
                if j != self._ref_phase_idx:
                    X0[:, -(1 + nphase * ncomp + nphase - 1) + idx] = fluid_state.y[j]
                    idx += 1
                # composition of phase j
                for i in range(ncomp):
                    X0[:, -(1 + (nphase - j) * ncomp) + i] = fluid_state.phases[j].x[i]

            # If T is unknown, get provided guess for T
            if "T" not in flash_type:
                X0[:, -(1 + ncomp * nphase + nphase - 1 + 1)] = fluid_state.T
            # If v is given, get provided guess for p and saturations
            if "v" in flash_type:
                # If T is additionally unknown to p, p is the second last quantity
                # before molar fractions
                if "T" not in flash_type:
                    p_pos = 1 + ncomp * nphase + nphase - 1 + 2
                else:
                    p_pos = 1 + ncomp * nphase + nphase - 1 + 1
                X0[:, -p_pos] = fluid_state.p
                # parsing saturation values except for reference phase
                idx = 0
                for j in range(nphase - 1):
                    if j != self._ref_phase_idx:
                        X0[:, -(p_pos + nphase - 1) + idx] = fluid_state.sat[j]
                        idx += 1

        logger.debug(f"{flash_type}Flash: Initializing NPIPM ..")
        X0 = initialize_npipm_nu(X0, self.npnc)

        F = self.residuals[flash_type]
        DF = self.jacobians[flash_type]
        self._update_solver_params(f_dim)

        logger.info(f"{flash_type} Flash: Solving ..")
        start = time.time()
        if mode == "linear":
            results, success, num_iter = linear_solver(X0, F, DF, self._solver_params)
        else:  # parallel, by logic only this
            results, success, num_iter = parallel_solver(X0, F, DF, self._solver_params)
        end = time.time()
        minim_time = end - start
        logger.info(f"Flashed (elapsed time: {minim_time} (s)).")

        self.last_flash_stats = {
            "type": flash_type,
            "init_time": init_time,
            "minim_time": minim_time,
            "num_flash": NF,
            "num_max_iter": int(np.sum(success == 1)),
            "num_failure": int(np.sum(success == 2) + np.sum(success == 3)),
            "num_diverged": int(np.sum(success == 4)),
        }

        return (
            self._parse_and_complete_results(results, flash_type, fluid_state),
            success,
            num_iter,
        )
