"""This module contains a compiled versions of flash-system assemblers.

The functions assembled and NJIT-compiled here, for a given mixture,
provide an efficient evaluation of the residual of flash equations and an efficient
assembly of the Jacobian.

The functions are such that they can be used to solve multiple flash problems in
parallel.

The expressions and compiled functions are generated using a combination of ``sympy``
and ``numba``.

"""
from __future__ import annotations

import abc
from typing import Callable, Literal

import numba
import numpy as np


@numba.njit(
    "Tuple((float64[:,:], float64[:], float64[:]))(float64[:], UniTuple(int32, 2))",
    fastmath=True,
    cache=True,
)
def parse_xyz(
    X_gen: np.ndarray, mpmc: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper function to parse the fractions from a generic argument.

    NJIT-ed function with signature
    ``(float64[:], UniTuple(int32, 2)) -> Tuple(float64[:,:], float64[:], float64[:])``.

    The feed fractions are always the first ``num_comp - 1`` entries
    (feed per component except reference component).

    The phase compositions are always the last ``num_phase * num_comp`` entries,
    orderered per phase per component (phase-major order),
    with phase and component order as given by the mixture model.

    the ``num_phase - 1`` entries before the phase compositions, are always
    the independent molar phase fractions.

    Important:
        This method also computes the feed fraction of the reference component and the
        molar fraction of the reference phase.
        Hence it returns 2 values not found in ``X_gen``.

        The computed values are always the first ones in the respective vector.

    Parameters:
        X_gen: Generic argument for a flash system.
        mpmc: 2-tuple containing information about number of phases and components
            (``num_phase`` and ``num_comp``).
            This information is required for pre-compilation of a mixture-independent
            function.

    Returns:
        A 3-tuple containing

        1. Phase compositions as a matrix with shape ``(num_phase, num_comp)``
        2. Molar phase fractions as an array with shape ``(num_phase,)``
        3. Feed fractions as an array with shape ``(num_comp,)``

    """
    nphase, ncomp = mpmc
    # feed fraction per component, except reference component
    Z = np.empty(ncomp, dtype=np.float64)
    Z[1:] = X_gen[: ncomp - 1]
    Z[0] = 1.0 - np.sum(Z[1:])
    # phase compositions
    X = X_gen[-ncomp * nphase :].copy()  # must copy to be able to reshape
    # matrix:
    # rows have compositions per phase,
    # columns have compositions related to a component
    X = X.reshape((nphase, ncomp))
    # phase fractions, -1 because fraction of ref phase is eliminated and not part of
    # the generic argument
    Y = np.empty(nphase, dtype=np.float64)
    Y[1:] = X_gen[-(ncomp * nphase + nphase - 1) : -ncomp * nphase]
    Y[0] = 1.0 - np.sum(Y[1:])

    return X, Y, Z


@numba.njit(
    "float64[:](float64[:], UniTuple(int32, 2))",
    fastmath=True,
    cache=True,
)
def parse_pT(X_gen: np.ndarray, mpmc: tuple[int, int]) -> np.ndarray:
    """Helper function extracing pressure and temperature from a generic
    argument.

    NJIT-ed function with signature
    ``(float64[:], UniTuple(int32, 2)) -> float64[:]``.

    Pressure and temperature are the last two values before the independent molar phase
    fractions (``num_phase - 1``) and the phase compositions (``num_phase * num_comp``).

    Parameters:
        X_gen: Generic argument for a flash system.
        mpmc: 2-tuple containing information about number of phases and components
            (``num_phase`` and ``num_comp``).
            This information is required for pre-compilation of a mixture-independent
            function.

    Returns:
        An array with shape ``(2,)``.

    """
    nphase, ncomp = mpmc
    return X_gen[-(ncomp * nphase + nphase - 1) - 2 : -(ncomp * nphase + nphase - 1)]


@numba.njit("float64[:,:](float64[:,:])", fastmath=True, cache=True)
def normalize_fractions(X: np.ndarray) -> np.ndarray:
    """Takes a matrix of phase compositions (rows - phase, columns - component)
    and normalizes the fractions.

    Meaning it divides each matrix element by the sum of respective row.

    NJIT-ed function with signature ``(float64[:,:]) -> float64[:,:]``.

    Parameters:
        X: ``shape=(num_phases, num_components)``

            A matrix of phase compositions, containing per row the (extended)
            fractions per component.

    Returns:
        A normalized version of ``X``, with the normalization performed row-wise
        (phase-wise).

    """
    return (X.T / X.sum(axis=1)).T


@numba.njit(
    "float64[:](float64[:,:], float64[:], float64[:])",
    fastmath=True,
    cache=True,
)
def mass_conservation_res(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    r"""Assembles the residual of the mass conservation equations.

    For each component ``i``, except reference component, it holds

    ... math::

        z\left[i\right] - \sum_j y\left[j\right] x\left[j, i\right] = 0

    Number of phases and components is determined from the chape of ``x``.

    NJIT-ed function with signature
    ``(float64[:,:], float64[:], float64[:]) -> float64[:]``.

    Important:
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
    res = np.empty(z.shape[0] - 1, dtype=np.float64)

    for i in range(1, z.shape[0]):
        res[i - 1] = z[i] - np.sum(y * x[:, i])

    return res


@numba.njit(
    "float64[:,:](float64[:,:], float64[:])",
    fastmath=True,
    cache=True,
)
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
    # and above is d z(x,y) / d[y, x]
    return (-1) * jac


@numba.njit("float64[:](float64[:,:], float64[:])", fastmath=True, cache=True)
def complementary_conditions_res(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""Assembles the residual of the complementary conditions.

    For each phase ``j`` it holds

    ... math::

        y\left[j\right] \cdot \left(1 - \sum_i x\left[j, i\right]\right) = 0

    Number of phases and components is determined from the chape of ``x``.

    NJIT-ed function with signature
    ``(float64[:,:], float64[:]) -> float64[:]``.

    Important:
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
    nphase = y.shape[0]
    ccs = np.empty(nphase, dtype=np.float64)
    for j in range(nphase):
        ccs[j] = y[j] * (1.0 - np.sum(x[j]))

    return ccs


@numba.njit("float64[:,:](float64[:,:], float64[:])", fastmath=True, cache=True)
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


class EoSCompiler(abc.ABC):
    """Abstract base class for EoS specific compilation using numba.

    The :class:`FlashCompiler` needs functions computing

    - fugacity coefficients
    - enthalpies
    - densities
    - the derivatives of above w.r.t. pressure, temperature and phase compositions

    Respective functions must be assembled and compiled by a child class with a specific
    EoS.

    The compiled functions are expected to have the following signature:

    ``(prearg: np.ndarray, phase_index: int, p: float, T: float, xn: numpy.ndarray)``

    1. ``prearg``: A 2-dimensional array containing the results of the pre-computation
       using the function returned by :meth:`get_pre_arg_computation`.
    2. ``phase_index``, the index of the phase, for which the quantities should be
       computed (``j = 0 ... num_phase - 1``), assuming 0 is the reference phase
    3. ``p``: The pressure value.
    4. ``T``: The temperature value.
    5 ``xn``: An array with ``shape=(num_comp,)`` containing the normalized fractions
      per component in phase ``phase_index``.

    The purpose of the ``prearg`` is efficiency.
    Many EoS have computions of some coterms or compressibility factors f.e.,
    which must only be computed once for all remaining thermodynamic quantities.

    The function for the ``prearg`` computation must have the signature:

    ``(p: float, T: float, XN: np.ndarray)``

    where ``XN`` contains **all** normalized compositions,
    stored row-wose per phase as a matrix.

    There are two ``prearg`` computations: One for the residual, one for the Jacobian
    of the flash system.

    The ``prearg`` for the Jacobian will be fed to the functions representing
    derivatives of thermodynamic quantities
    (e.g. derivative fugacity coefficients w.r.t. p, T, X).

    The ``prearg`` for the residual will be passed to thermodynamic quantities without
    derivation.

    """

    @abc.abstractmethod
    def get_pre_arg_computation_res(
        self,
    ) -> Callable[[float, float, np.ndarray], np.ndarray]:
        """Abstract function for obtaining the compiled computation of the pre-argument
        for the residual.

        Returns:
            A NJIT-ed function taking values for pressure, temperature and normalized
            phase compositions for all phases as a matrix, and returning any matrix,
            which will be passed to other functions computing thermodynamic quantities.

        """
        pass

    @abc.abstractmethod
    def get_pre_arg_computation_jac(
        self,
    ) -> Callable[[float, float, np.ndarray], np.ndarray]:
        """Abstract function for obtaining the compiled computation of the pre-argument
        for the Jacobian.

        Returns:
            A NJIT-ed function taking values for pressure, temperature and normalized
            phase compositions for all phases as a matrix, and returning any matrix,
            which will be passed to other functions computing derivatives of
            thermodynamic quantities.

        """
        pass

    @abc.abstractmethod
    def get_fugacity_computation(
        self,
    ) -> Callable[[np.ndarray, int, float, float, np.ndarray], np.ndarray]:
        """Abstract assembler for compiled computations of the fugacity coefficients.

        Returns:
            A NJIT-ed function taking
            - the pre-argument for the residual,
            - the phase index,
            - pressure value,
            - temperature value,
            - an array normalized fractions of componentn in phase ``phase_index``

            and returning an array of fugacity coefficients with ``shape=(num_comp,)``.

        """
        pass

    @abc.abstractmethod
    def get_dpTX_fugacity_computation(
        self,
    ) -> Callable[[np.ndarray, int, float, float, np.ndarray], np.ndarray]:
        """Abstract assembler for compiled computations of the derivative of fugacity
        coefficients.

        The functions should return the derivative fugacities for each component
        row-wise in a matrix.
        It must contain the derivatives w.r.t. pressure, temperature and each fraction
        in a specified phase.
        I.e. the return value must be an array with ``shape=(num_comp, 2 + num_comp)``.

        Returns:
            A NJIT-ed function taking
            - the pre-argument for the Jacobian,
            - the phase index,
            - pressure value,
            - temperature value,
            - an array normalized fractions of componentn in phase ``phase_index``

            and returning an array of derivatives offugacity coefficients with
            ``shape=(num_comp, 2 + num_comp)``., where the columns indicate the
            derivatives w.r.t. to pressure, temperature and fractions.

        """
        pass


class FlashCompiler:
    """Class implementing NJIT-compiled representation of the equilibrium equations
    using numba.

    It uses the no-python mode of numba to produce highly efficient, compiled code.

    Intended for parallel solution of the local equilibrium problem in compositional
    flow.

    Parameters:
        mpmc: A 2-tuple containing the number of phases and number of components
        eos_compiler: A compilter class providing NJIT-ed functions for fugacity values
            among others.

    """

    def __init__(
        self,
        mpmc: tuple[int, int],
        eos_compiler: EoSCompiler,
    ) -> None:
        ### declaration of available residuals and Jacobians

        self.residuals: dict[
            Literal["p-T", "p-h", "v-h"], Callable[[np.ndarray], np.ndarray]
        ] = dict()
        """Contains per flash configuration the respective residuals as a callable."""

        self.jacobians: dict[
            Literal["p-T", "p-h", "v-h"], Callable[[np.ndarray], np.ndarray]
        ] = dict()
        """Contains per flash configuration the respective Jacobian as a callable."""

        ### Compilation of residuals and Jacobians

        nphase, ncomp = mpmc

        # number of equations for the pT system
        # ncomp - 1 mass constraints
        # (nphase - 1) * ncomp fugacity constraints (w.r.t. ref phase formulated)
        # nphase complementary conditions
        pT_dim = ncomp - 1 + (nphase - 1) * ncomp + nphase

        prearg_res_c = eos_compiler.get_pre_arg_computation_res()
        prearg_jac_c = eos_compiler.get_pre_arg_computation_jac()
        phi_c = eos_compiler.get_fugacity_computation()
        d_phi_c = eos_compiler.get_dpTX_fugacity_computation()

        @numba.njit("float64[:](float64[:,:], float64, float64, float64[:,:])")
        def isofug_constr_c(
            prearg_res: np.ndarray,
            p: float,
            T: float,
            Xn: np.ndarray,
        ):
            """Helper function to assemble the isofugacity constraint.

            Formulation is always w.r.t. the reference phase r, assumed to be r=0.

            """
            isofug = np.empty(ncomp * (nphase - 1), dtype=np.float64)

            phi_r = phi_c(prearg_res, 0, p, T, Xn[0])

            for j in range(1, nphase):
                phi_j = phi_c(prearg_res, j, p, T, Xn[j])
                # isofugacity constraint between phase j and phase r
                isofug[(j - 1) * ncomp : j * ncomp] = Xn[j] * phi_j - Xn[0] * phi_r

            return isofug

        @numba.njit(
            "float64[:,:](float64[:,:], float64[:,:], int32, float64, float64, float64[:])"
        )
        def d_isofug_block_j(
            prearg_res: np.ndarray,
            prearg_jac: np.ndarray,
            j: int,
            p: float,
            T: float,
            Xn: np.ndarray,
        ):
            """Helper function to construct a block representing the derivative
            of x_ij * phi_ij for all i as a matrix, with i row index.
            This is constructed for a given phase j.
            """
            # derivatives w.r.t. p, T, all compositions, A, B, Z
            dx_phi_j = np.zeros((ncomp, 2 + ncomp))

            phi_j = phi_c(prearg_res, j, p, T, Xn)
            d_phi_j = d_phi_c(prearg_jac, j, p, T, Xn)

            # product rule d(x * phi) = dx * phi + x * dphi
            # dx is is identity
            dx_phi_j[:, 2 : 2 + ncomp] = np.diag(phi_j)
            d_xphi_j = dx_phi_j + (d_phi_j.T * Xn).T

            return d_xphi_j

        @numba.njit(
            "float64[:,:](float64[:,:], float64[:, :], float64, float64, float64[:,:])"
        )
        def d_isofug_constr_c(
            prearg_res: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
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
            d_xphi_r = d_isofug_block_j(prearg_res, prearg_jac, 0, p, T, Xn[0])

            for j in range(1, nphase):
                # construct the same as above for other phases
                d_xphi_j = d_isofug_block_j(prearg_res, prearg_jac, 1, p, T, Xn[j])

                # filling in the relevant blocks
                # remember: d(x_ij * phi_ij - x_ir * phi_ir)
                # hence every row-block contains (-1)* d_xphi_r
                # p, T derivative
                d_iso[(j - 1) * ncomp : j * ncomp, :2] = (
                    d_xphi_j[:, :2] - d_xphi_r[:, :2]
                )
                # derivative w.r.t. fractions in reference phase
                d_iso[(j - 1) * ncomp : j * ncomp, 2 : 2 + ncomp] = (-1) * d_xphi_r[
                    :, 2:
                ]
                # derivatives w.r.t. fractions in independent phase j
                d_iso[
                    (j - 1) * ncomp : j * ncomp, 2 + j * ncomp : 2 + (j + 1) * ncomp
                ] = d_xphi_j[:, 2:]

            return d_iso

        @numba.njit("float64[:](float64[:])")
        def F_pT(X_gen: np.ndarray) -> np.ndarray:
            x, y, z = parse_xyz(X_gen, (nphase, ncomp))
            p, T = parse_pT(X_gen, (nphase, ncomp))

            # declare residual array of proper dimension
            res = np.empty(pT_dim, dtype=np.float64)

            res[: ncomp - 1] = mass_conservation_res(x, y, z)
            # last nphase equations are always complementary conditions
            res[-nphase:] = complementary_conditions_res(x, y)

            # EoS specific computations
            xn = normalize_fractions(x)
            prearg = prearg_res_c(p, T, xn)

            res[ncomp - 1 : ncomp - 1 + ncomp * (nphase - 1)] = isofug_constr_c(
                prearg, p, T, xn
            )

            return res

        @numba.njit("float64[:,:](float64[:])")
        def DF_pT(X_gen: np.ndarray) -> np.ndarray:
            x, y, _ = parse_xyz(X_gen, (nphase, ncomp))
            p, T = parse_pT(X_gen, (nphase, ncomp))

            # declare Jacobian or proper dimension
            jac = np.empty((pT_dim, pT_dim), dtype=np.float64)

            jac[: ncomp - 1] = mass_conservation_jac(x, y)
            # last nphase equations are always complementary conditions
            jac[-nphase:] = complementary_conditions_jac(x, y)

            # EoS specific computations
            xn = normalize_fractions(x)
            prearg_res = prearg_res_c(p, T, xn)
            prearg_jac = prearg_jac_c(p, T, xn)

            jac[
                ncomp - 1 : ncomp - 1 + ncomp * (nphase - 1), nphase - 1 :
            ] = d_isofug_constr_c(prearg_res, prearg_jac, p, T, xn)[:, 2:]

            return jac

        self.residuals.update(
            {
                "p-T": F_pT,
            }
        )

        self.jacobians.update(
            {
                "p-T": DF_pT,
            }
        )
