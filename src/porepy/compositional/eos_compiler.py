"""Module containing the abstract base class for compiling EoS-related functions used
in the evaluation of thermodynamic properties."""

from __future__ import annotations

import abc
import logging
from typing import Callable, Literal, Sequence, TypeAlias, TypedDict, cast

import numba
import numpy as np

from ._core import NUMBA_PARALLEL, PhysicalState
from .base import Component, EquationOfState
from .states import PhaseProperties
from .utils import normalize_rows

__all__ = [
    "EoSCompiler",
]

logger = logging.getLogger(__name__)


ScalarFunction: TypeAlias = Callable[..., float]
"""Type alias for scalar functions returning a floar. Used to type scalar thermodynamic
properties"""


VectorFunction: TypeAlias = Callable[..., np.ndarray]
"""Type alias for vector functions returning a numpy array. Used to type derivative functions
of thermodynamic properties."""


class PropertyFunctionDict(TypedDict):
    """Typed dictionary defining which property functions are expected to be available in
    :attr:`EoSCompiler.funcs` of an :class:`EoSCompiler`-instance.

    """

    prearg_val: VectorFunction
    """Provided by :meth:`EoSCompiler.get_prearg_for_values`."""
    prearg_jac: VectorFunction
    """Provided by :meth:`EoSCompiler.get_prearg_for_derivatives`."""
    h: ScalarFunction
    """Provided by :meth:`EoSCompiler.get_enthalpy_function`."""
    d_h: VectorFunction
    """Provided by :meth:`EoSCompiler.get_enthalpy_derivative_function`."""
    rho: ScalarFunction
    """Provided by :meth:`EoSCompiler.get_density_function`."""
    d_rho: VectorFunction
    """Provided by :meth:`EoSCompiler.get_density_derivative_function`."""
    v: ScalarFunction
    """Provided by :meth:`EoSCompiler.get_volume_function`."""
    d_v: VectorFunction
    """Provided by :meth:`EoSCompiler.get_volume_derivative_function`."""
    mu: ScalarFunction
    """Provided by :meth:`EoSCompiler.get_viscosity_function`."""
    d_mu: VectorFunction
    """Provided by :meth:`EoSCompiler.get_viscosity_derivative_function`."""
    kappa: ScalarFunction
    """Provided by :meth:`EoSCompiler.get_conductivity_function`."""
    d_kappa: VectorFunction
    """Provided by :meth:`EoSCompiler.get_conductivity_derivative_function`."""
    phi: VectorFunction
    """Provided by :meth:`EoSCompiler.get_fugacity_function`."""
    d_phi: VectorFunction
    """Provided by :meth:`EoSCompiler.get_fugacity_derivative_function`."""


def _compile_vectorized_prearg(
    func_c: VectorFunction,
) -> VectorFunction:
    """Helper function implementing the parallelized, compiled computation of
    pre-argument functions ``func_c``, which is called element-wise."""

    @numba.njit(
        "float64[:,:](int32,float64[:],float64[:],float64[:,:])",
        parallel=NUMBA_PARALLEL,
    )
    def inner(phasetype: int, p: np.ndarray, T: np.ndarray, xn: np.ndarray):
        # the dimension of the prearg is unknown, run the first one to get it
        _, N = xn.shape
        pre_arg_0 = func_c(phasetype, p[0], T[0], xn[:, 0])
        pre_arg_all = np.empty((N, pre_arg_0.shape[0]))
        pre_arg_all[0] = pre_arg_0
        for i in numba.prange(1, N):
            pre_arg_all[i] = func_c(phasetype, p[i], T[i], xn[:, i])
        return pre_arg_all

    return inner


def _compile_vectorized_phi(
    phi_c: VectorFunction,
) -> VectorFunction:
    """Helper function implementing the parallelized, compiled computation of
    fugacity coefficients given by ``phi_c``.

    The resulting 2D array will contain row-wise the fugacity coefficients per
    component, analogous to the scalar case given by
    :meth:`EoSCompiler.get_fugacity_derivative_function`.

    The second dimension will reflect the vectorized input.

    """

    @numba.njit(
        "float64[:,:](float64[:,:],float64[:],float64[:],float64[:,:])",
        parallel=NUMBA_PARALLEL,
    )
    def inner(prearg: np.ndarray, p: np.ndarray, T: np.ndarray, xn: np.ndarray):
        ncomp, N = xn.shape
        phis = np.empty((N, ncomp))
        for i in numba.prange(N):
            phis[i] = phi_c(prearg[i], p[i], T[i], xn[:, i])
        # phis per component row-wise
        return phis.T

    return inner


def _compile_vectorized_d_phi(
    d_phi_c: VectorFunction,
) -> VectorFunction:
    """Helper function implementing the parallelized, compiled computation of
    fugacity coefficient derivatives given by ``phi_c``.

    The resulting 3D array has the following structure:

    - First dimension reflects the components
    - Second dimension reflects the derivatives (pressure, temperature, dx per fraction)
    - Third dimension reflects the vectorized values

    This is for consistency reasons with the scalar case.

    """

    @numba.njit(
        "float64[:,:,:](float64[:,:],float64[:,:],float64[:],float64[:],float64[:,:])",
        parallel=NUMBA_PARALLEL,
    )
    def inner(
        prearg_res: np.ndarray,
        prearg_jac: np.ndarray,
        p: np.ndarray,
        T: np.ndarray,
        xn: np.ndarray,
    ):
        ncomp, N = xn.shape
        ndiffs = ncomp + 2
        d_phis = np.empty((ncomp, ndiffs, N))
        for i in numba.prange(N):
            d_phis[:, :, i] = d_phi_c(
                prearg_res[i], prearg_jac[i], p[i], T[i], xn[:, i]
            )
        return d_phis

    return inner


def _compile_vectorized_property(
    func_c: ScalarFunction,
) -> VectorFunction:
    """Helper function implementing the parallelized, compiled computation of
    properties given by ``func_c`` element-wise."""

    @numba.njit(
        "float64[:](float64[:,:],float64[:],float64[:],float64[:,:])",
        parallel=NUMBA_PARALLEL,
    )
    def inner(prearg: np.ndarray, p: np.ndarray, T: np.ndarray, xn: np.ndarray):
        _, N = xn.shape
        vals = np.empty(N)
        for i in numba.prange(N):
            vals[i] = func_c(prearg[i], p[i], T[i], xn[:, i])
        return vals

    return inner


def _compile_vectorized_derivatives(
    func_c: VectorFunction,
) -> VectorFunction:
    """Helper function implementing the parallelized, compiled computation of
    property derivatives given by ``func_c`` element-wise.

    The resulting 2D array has structure:

    - First dimension per derivative (pressure, temperature, dx per fraction)
    - Second dimension per element in vectorized input

    """

    @numba.njit(
        "float64[:,:](float64[:,:],float64[:,:],float64[:],float64[:],float64[:,:])",
        parallel=NUMBA_PARALLEL,
    )
    def inner(
        prearg_val: np.ndarray,
        prearg_jac: np.ndarray,
        p: np.ndarray,
        T: np.ndarray,
        xn: np.ndarray,
    ):
        ncomp, N = xn.shape
        ndiffs = ncomp + 2  # derivatives w.r.t. p and T included

        # derivatives are stored row-wise
        vals = np.empty((ndiffs, N))
        for i in numba.prange(N):
            vals[:, i] = func_c(prearg_val[i], prearg_jac[i], p[i], T[i], xn[:, i])
        return vals

    return inner


class EoSCompiler(EquationOfState):
    """Abstract base class for EoS compilation using numba.

    This class needs functions computing

    - fugacity coefficients
    - enthalpies
    - densities
    - the derivatives of above w.r.t. pressure, temperature and phase compositions

    Respective functions must be assembled and compiled by a child class with a specific
    EoS.

    The compiled functions are expected to have a specific signature (see below).

    1. One or two pre-arguments (vectors)
    2. An arbitrary number of scalar arguments (like pressure and temperature value)
    3. A vector argument representing a family of fractions (like partial fractions in a phase)

    The purpose of the ``prearg`` is efficiency.
    Many EoS have computions of some co-terms or compressibility factors f.e.,
    which must only be computed once for all remaining thermodynamic quantities.

    The function for the ``prearg`` computation must have the signature:

    1. an integer representing a phase :attr:`~porepy.compositional.base.Phase.state`
    2. An arbitrary number of scalar arguments (like pressure and temperature value)
    3. A vector argument representing a family of fractions (like partial fractions in a phase)

    There are two ``prearg`` computations: One for property values, one for the
    derivatives.

    The ``prearg`` for the derivatives will be fed to the functions representing
    derivatives of thermodynamic quantities **additionally** to the ``prearg`` for residuals.

    I.e., the signature of functions representing derivatives is expected to be

    ``(prearg_val: np.ndarray, prearg_jac: np.ndarray, ..., x: np.ndarray)``,

    whereas the signature of functions representing values only is expected to be

    ``(prearg_val: np.ndarray, ..., x: np.ndarray)``

    Parameters:
        components: Sequence of components for which the EoS should be compiled.
            The class :class:`~porepy.compositional.base.Component` is used as a storage
            for physical properties, the only relevant information for the EoS.

    """

    def __init__(self, components: Sequence[Component]) -> None:
        super().__init__(components)

        # Ignoring mypy error because functions are compiled at later stage.
        # An empty dict will alert the user that something is missing.
        self.funcs: PropertyFunctionDict = {}  # type:ignore[typeddict-item]
        """Dictionary for storing functions which are compiled in various
        ``get_*`` methods.

        Accessed during :meth:`compile` to create vectorized functions, which
        in return are stored in :attr:`gufuncs`.

        See documentation of type class for more information.

        """

        self.gufuncs: dict[
            Literal[
                "prearg_val",
                "prearg_jac",
                "phi",
                "d_phi",
                "h",
                "d_h",
                "v",
                "d_v",
                "rho",
                "d_rho",
                "mu",
                "d_mu",
                "kappa",
                "d_kappa",
            ],
            VectorFunction,
        ] = {}
        """Storage of generalized and vectorized versions of the functions found in
        :attr:`funcs`.

        To be used for efficient evaluation of properties after the flash converged.

        """

        self._is_compiled: bool = False
        """"Boolean to keep track of whether (costly) compilation process has been performed.

        Should be required only once.

        """

    @property
    def is_compiled(self) -> bool:
        """Returns true, if :meth:`compile` has already been called, False otherwise."""
        return self._is_compiled

    # TODO what is more efficient, just one pre-arg having everything?
    # Or splitting for computations for residuals, since it does not need derivatives?
    # 1. Armijo line search evaluated often, need only residual
    # 2. On the other hand, residual pre-arg is evaluated twice, for residual and jac
    @abc.abstractmethod
    def get_prearg_for_values(self) -> VectorFunction:
        """Abstract function for obtaining the compiled computation of the pre-argument
        for the evaluation of thermodynamic properties.

        Returns:
            A NJIT-ed function with signature
            ``(phasetype: int, *args: float, x: np.ndarray)``
            returning a 1D array.

        """
        pass

    @abc.abstractmethod
    def get_prearg_for_derivatives(self) -> VectorFunction:
        """Abstract function for obtaining the compiled computation of the pre-argument
        for the evaluation of derivatives of thermodynamic properties.

        Returns:
            A NJIT-ed function with signature
            ``(phasetype: int, *args: float, x: np.ndarray)``
            returning a 1D array.

        """
        pass

    @abc.abstractmethod
    def get_fugacity_function(self) -> VectorFunction:
        """Abstract assembler for compiled computations of the fugacity coefficients.

        Returns:
            A NJIT-ed function with signature
            ``(prearg_val: np.ndarray, *args: float, x: np.ndarray)``
            and returning an array of fugacity coefficients with ``shape=(num_comp,)``.

        """
        pass

    @abc.abstractmethod
    def get_fugacity_derivative_function(self) -> VectorFunction:
        """Abstract assembler for compiled computations of the derivative of fugacity
        coefficients.

        The functions should return the derivative fugacities for each component
        row-wise in a matrix.
        It must contain the derivatives w.r.t. to the arguments and each fraction.
        I.e. the return value must be an array with ``shape=(num_comp, m + num_comp)``.

        Returns:
            A NJIT-ed function with signature
            ``(prearg_val: np.ndarray, prearg_jac: np.ndarray, *args: float, x: np.ndarray)``
            and returning an array of derivatives of fugacity coefficients with
            ``shape=(num_comp, 2 + num_comp)``., where containing the derivatives w.r.t. the
            arguments and fractions.

        """
        pass

    @abc.abstractmethod
    def get_enthalpy_function(self) -> ScalarFunction:
        """Abstract assembler for compiled computations of the specific molar enthalpy.

        Returns:
            A NJIT-ed function with signature
            ``(prearg_val: np.ndarray, *args: float, x: np.ndarray)``
            and returning an enthalpy value.

        """
        pass

    @abc.abstractmethod
    def get_enthalpy_derivative_function(self) -> VectorFunction:
        """Abstract assembler for compiled computations of the derivative of the
        enthalpy function for a phase.

        Returns:
            A NJIT-ed function with signature
            ``(prearg_val: np.ndarray, prearg_jac: np.ndarray, *args: float, x: np.ndarray)``
            and returning an array of derivatives of the enthalpy with
            ``shape=(2 + num_comp,)``., containing the derivatives w.r.t. the arguments and
            fractions.

        """
        pass

    @abc.abstractmethod
    def get_density_function(self) -> ScalarFunction:
        """Abstract assembler for compiled computations of the density.

        Returns:
            A NJIT-ed function with signature
            ``(prearg_val: np.ndarray, *args: float, x: np.ndarray)``
            and returning a density value.

        """
        pass

    @abc.abstractmethod
    def get_density_derivative_function(self) -> VectorFunction:
        """Abstract assembler for compiled computations of the derivative of the
        density function for a phase.

        Returns:
            A NJIT-ed function with signature
            ``(prearg_val: np.ndarray, prearg_jac: np.ndarray, *args: float, x: np.ndarray)``
            and returning an array of derivatives of the density with
            ``shape=(2 + num_comp,)``., containing the derivatives w.r.t. the arguments and
            fractions.

        """
        pass

    @abc.abstractmethod
    def get_viscosity_function(self) -> ScalarFunction:
        """Abstract assembler for compiled computations of the dynamic molar viscosity.

        Returns:
            A NJIT-ed function with signature
            ``(prearg_val: np.ndarray, *args: float, x: np.ndarray)``
            and returning a viscosity value.

        """
        pass

    @abc.abstractmethod
    def get_viscosity_derivative_function(self) -> VectorFunction:
        """Abstract assembler for compiled computations of the derivative of the
        viscosity function for a phase.

        Returns:
            A NJIT-ed function with signature
            ``(prearg_val: np.ndarray, prearg_jac: np.ndarray, *args: float, x: np.ndarray)``
            and returning an array of derivatives of the viscosity with
            ``shape=(2 + num_comp,)``., containing the derivatives w.r.t. the arguments and
            fractions.

        """
        pass

    @abc.abstractmethod
    def get_conductivity_function(self) -> ScalarFunction:
        """Abstract assembler for compiled computations of the thermal conductivity.

        Returns:
            A NJIT-ed with signature
            ``(prearg_val: np.ndarray, *args: float, x: np.ndarray)``
            and returning a conductivity value.

        """
        pass

    @abc.abstractmethod
    def get_conductivity_derivative_function(self) -> VectorFunction:
        """Abstract assembler for compiled computations of the derivative of the
        conductivity function for a phase.

        Returns:
            A NJIT-ed function taking with signature
            ``(prearg_val: np.ndarray, prearg_jac: np.ndarray, *args: float, x: np.ndarray)``
            and returning an array of derivatives of the conductivity with
            ``shape=(2 + num_comp,)``., containing the derivatives w.r.t. the arguments and
            fractions.

        """
        pass

    def get_volume_function(self) -> ScalarFunction:
        """Assembler for compiled computations of the specific molar volume.

        The volume is computed as the reciprocal of the return value of
        :meth:`get_density_function`.

        Note:
            This function is compiled faster, if the density function has already been
            compiled and stored in :attr:`funcs`.

        Returns:
            A NJIT-ed function with signature
            ``(prearg_val: np.ndarray, *args: float, x: np.ndarray)``
            and returning a density value.

        """
        rho_c = self.funcs.get("rho", None)
        if rho_c is None:
            rho_c = self.get_density_function()

        rho_c = cast(ScalarFunction, rho_c)

        @numba.njit("float64(float64[:],float64,float64,float64[:])")
        def v_c(prearg: np.ndarray, p: float, T: float, xn: np.ndarray) -> float:
            rho = rho_c(prearg, p, T, xn)
            if rho > 0.0:
                return 1.0 / rho
            else:
                return 0.0

        return v_c

    def get_volume_derivative_function(self) -> VectorFunction:
        """Assembler for compiled computations of the derivative of the
        volume function for a phase.

        Volume is expressed as the reciprocal of density.
        Hence the computations utilize :meth:`get_density_function`,
        :meth:`get_density_derivative_function` and the chain-rule to compute the derivatives.

        Note:
            This function is compiled faster, if the density function and its
            deritvative have already been compiled and stored in :attr:`funcs`.

        Returns:
            A NJIT-ed function with signature
            ``(prearg_val: np.ndarray, prearg_jac: np.ndarray, *args: float, x: np.ndarray)``
            and returning an array of derivatives of the volume with
            ``shape=(2 + num_comp,)``., containing the derivatives w.r.t. the arguments and
            fractions.

        """
        rho_c = self.funcs.get("rho", None)
        if rho_c is None:
            rho_c = self.get_density_function()

        drho_c = self.funcs.get("d_rho", None)
        if drho_c is None:
            drho_c = self.get_density_derivative_function()

        rho_c = cast(ScalarFunction, rho_c)
        drho_c = cast(VectorFunction, drho_c)

        @numba.njit("float64[:](float64[:],float64[:],float64,float64,float64[:])")
        def dv_c(
            prearg_res: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
        ) -> np.ndarray:
            rho = rho_c(prearg_res, p, T, xn)
            drho = drho_c(prearg_res, prearg_jac, p, T, xn)
            if rho > 0.0:
                # chain rule: dv = d(1 / rho) = - 1 / rho**2 * drho
                return -drho / rho**2
            else:
                return np.zeros_like(drho)

        return dv_c

    def compile(self) -> None:
        """Compiles vectorized functions for properties, depending on pre-arguments,
        pressure, temperature and fractions.

        Accesses :attr:`funcs` to find functions for element-wise computations.
        If not found, calls various ``get_*`` methods to create them and stores them
        in :attr:`funcs`.

        Important:
            This function takes possibly long to complete. It compiles all scalar, and
            vectorized computations of properties.

            Hence it checks whether it has already been run or not.

        """

        if self.is_compiled:
            return

        logger.info("Compiling property functions ..")

        # region Element-wise computations
        self.funcs["prearg_val"] = self.get_prearg_for_values()
        logger.debug("Compiling property functions 1/14")
        self.funcs["prearg_jac"] = self.get_prearg_for_derivatives()
        logger.debug("Compiling property functions 2/14")
        self.funcs["phi"] = self.get_fugacity_function()
        logger.debug("Compiling property functions 3/14")
        self.funcs["d_phi"] = self.get_fugacity_derivative_function()
        logger.debug("Compiling property functions 4/14")
        self.funcs["h"] = self.get_enthalpy_function()
        logger.debug("Compiling property functions 5/14")
        self.funcs["d_h"] = self.get_enthalpy_derivative_function()
        logger.debug("Compiling property functions 6/14")
        self.funcs["rho"] = self.get_density_function()
        logger.debug("Compiling property functions 7/14")
        self.funcs["d_rho"] = self.get_density_derivative_function()
        logger.debug("Compiling property functions 8/14")
        self.funcs["v"] = self.get_volume_function()
        logger.debug("Compiling property functions 9/14")
        self.funcs["d_v"] = self.get_volume_derivative_function()
        logger.debug("Compiling property functions 10/14")
        self.funcs["mu"] = self.get_viscosity_function()
        logger.debug("Compiling property functions 11/14")
        self.funcs["d_mu"] = self.get_viscosity_derivative_function()
        logger.debug("Compiling property functions 12/14")
        self.funcs["kappa"] = self.get_conductivity_function()
        logger.debug("Compiling property functions 13/14")
        self.funcs["d_kappa"] = self.get_conductivity_derivative_function()
        logger.debug("Compiling property functions 14/14")
        # endregion

        logger.info("Compiling vectorized functions ..")

        # region vectorized computations
        self.gufuncs["prearg_val"] = _compile_vectorized_prearg(
            self.funcs["prearg_val"]
        )
        logger.debug("Compiling vectorized functions 1/14")
        self.gufuncs["prearg_jac"] = _compile_vectorized_prearg(
            self.funcs["prearg_jac"]
        )
        logger.debug("Compiling vectorized functions 2/14")
        self.gufuncs["phi"] = _compile_vectorized_phi(self.funcs["phi"])
        logger.debug("Compiling vectorized functions 3/14")
        self.gufuncs["d_phi"] = _compile_vectorized_d_phi(self.funcs["d_phi"])
        logger.debug("Compiling vectorized functions 4/14")
        self.gufuncs["h"] = _compile_vectorized_property(self.funcs["h"])
        logger.debug("Compiling vectorized functions 5/14")
        self.gufuncs["d_h"] = _compile_vectorized_derivatives(self.funcs["d_h"])
        logger.debug("Compiling vectorized functions 6/14")
        self.gufuncs["rho"] = _compile_vectorized_property(self.funcs["rho"])
        logger.debug("Compiling vectorized functions 7/14")
        self.gufuncs["d_rho"] = _compile_vectorized_derivatives(self.funcs["d_rho"])
        logger.debug("Compiling vectorized functions 8/14")
        self.gufuncs["v"] = _compile_vectorized_property(self.funcs["v"])
        logger.debug("Compiling vectorized functions 9/14")
        self.gufuncs["d_v"] = _compile_vectorized_derivatives(self.funcs["d_v"])
        logger.debug("Compiling vectorized functions 10/14")
        self.gufuncs["mu"] = _compile_vectorized_property(self.funcs["mu"])
        logger.debug("Compiling vectorized functions 11/14")
        self.gufuncs["d_mu"] = _compile_vectorized_derivatives(self.funcs["d_mu"])
        logger.debug("Compiling vectorized functions 12/14")
        self.gufuncs["kappa"] = _compile_vectorized_property(self.funcs["kappa"])
        logger.debug("Compiling vectorized functions 13/14")
        self.gufuncs["d_kappa"] = _compile_vectorized_derivatives(self.funcs["d_kappa"])
        logger.debug("Compiling vectorized functions 14/14")
        # endregion

        self._is_compiled = True

    def compute_phase_properties(
        self,
        phase_state: PhysicalState,
        *thermodynamic_input: np.ndarray,
    ) -> PhaseProperties:
        """This method must only be called after the vectorized computations have been
        compiled (see :meth:`compile`).

        Note:
            The returned derivatives include derivatives w.r.t. (physical) partial
            fractions, not extended fractions.

        Important:
            The last element of ``thermodynamic_input`` is expected to be a family of
            (extended) partial fractions belonging to a phase, i.e. a 2D array.
            They will be normalized before calling the compiled property functions

        Parameters:
            phasetype: Type of phase (passed to pre-arg computation).
            p: ``shape=(N,)``

                Pressure values.
            T: ``shape=(N,)``

                Temperature values.
            x: ``shape=(num_comp, N)``

                Partial fractions per component (row-wise). Note that extended partial
                fractions must be normalized before passing them as arguments.

        Returns:
            A complete datastructure containing values for thermodynamic phase
            properties and their derivatives.

            All sequential data fields are storred as arrays, with the sequence
            reflected in the first dimension.

        """

        prearg_val = self.gufuncs["prearg_val"](phase_state.value, *thermodynamic_input)
        prearg_jac = self.gufuncs["prearg_jac"](phase_state.value, *thermodynamic_input)

        x = thermodynamic_input[-1]
        assert x.ndim == 2, (
            "Last thermodynamic input expected to be a 2D array (fractions)"
        )
        x_norm = normalize_rows(x.T).T

        thermodynamic_input = tuple([_ for _ in thermodynamic_input[:-1]] + [x_norm])

        state = PhaseProperties(
            state=phase_state,
            x=x,
            h=self.gufuncs["h"](prearg_val, *thermodynamic_input),
            rho=self.gufuncs["rho"](prearg_val, *thermodynamic_input),
            # shape = (num_comp, num_vals), sequence per component
            phis=self.gufuncs["phi"](prearg_val, *thermodynamic_input),
            # shape = (num_diffs, num_vals), sequence per derivative
            dh=self.gufuncs["d_h"](prearg_val, prearg_jac, *thermodynamic_input),
            # shape = (num_diffs, num_vals), sequence per derivative
            drho=self.gufuncs["d_rho"](prearg_val, prearg_jac, *thermodynamic_input),
            # shape = (num_comp, num_diffs, num_vals)
            dphis=self.gufuncs["d_phi"](prearg_val, prearg_jac, *thermodynamic_input),
            mu=self.gufuncs["mu"](prearg_val, *thermodynamic_input),
            dmu=self.gufuncs["d_mu"](prearg_val, prearg_jac, *thermodynamic_input),
            kappa=self.gufuncs["kappa"](prearg_val, *thermodynamic_input),
            dkappa=self.gufuncs["d_kappa"](
                prearg_val, prearg_jac, *thermodynamic_input
            ),
        )

        return state
