"""Module containing the abstract base class for compiling EoS-related functions used
in the evaluation of thermodynamic properties."""

from __future__ import annotations

import abc
import logging
from functools import partial
from typing import Callable, Literal, Sequence, TypeAlias, TypedDict, cast

import numba as nb
import numpy as np

from ._core import NUMBA_PARALLEL, PhysicalState, cfunc, typeof
from .base import Component, EquationOfState
from .states import PhaseProperties
from .utils import normalize_rows

__all__ = [
    "EoSCompiler",
]

logger = logging.getLogger(__name__)


ScalarFunction: TypeAlias = Callable[..., float]
"""Type alias for scalar functions returning a float. Used to type scalar thermodynamic
properties"""


VectorFunction: TypeAlias = Callable[..., np.ndarray]
"""Type alias for vector functions returning a numpy array. Used to type derivative functions
of thermodynamic properties."""


class PropertyFunctionDict(TypedDict, total=False):
    """Typed dictionary defining which property functions are expected to be available
    in :attr:`EoSCompiler.funcs`."""

    prearg_val: VectorFunction
    """Provided by :meth:`EoSCompiler.get_prearg_for_values`."""
    prearg_jac: VectorFunction
    """Provided by :meth:`EoSCompiler.get_prearg_for_derivatives`."""
    h: ScalarFunction
    """Provided by :meth:`EoSCompiler.get_enthalpy_function`."""
    dh: VectorFunction
    """Provided by :meth:`EoSCompiler.get_enthalpy_derivative_function`."""
    rho: ScalarFunction
    """Provided by :meth:`EoSCompiler.get_density_function`."""
    drho: VectorFunction
    """Provided by :meth:`EoSCompiler.get_density_derivative_function`."""
    v: ScalarFunction
    """Provided by :meth:`EoSCompiler.get_volume_function`."""
    dv: VectorFunction
    """Provided by :meth:`EoSCompiler.get_volume_derivative_function`."""
    mu: ScalarFunction
    """Provided by :meth:`EoSCompiler.get_viscosity_function`."""
    dmu: VectorFunction
    """Provided by :meth:`EoSCompiler.get_viscosity_derivative_function`."""
    kappa: ScalarFunction
    """Provided by :meth:`EoSCompiler.get_conductivity_function`."""
    dkappa: VectorFunction
    """Provided by :meth:`EoSCompiler.get_conductivity_derivative_function`."""
    phis: VectorFunction
    """Provided by :meth:`EoSCompiler.get_fugacity_function`."""
    dphis: VectorFunction
    """Provided by :meth:`EoSCompiler.get_fugacity_derivative_function`."""


# NOTE The template functions need some non-trivial body and return value to compile.
# They are completely meaningless and need to return something corresponding to the
# annotated return type. They also use every argument somehow, in case mypy ever
# complains.
@cfunc(nb.f8[:](nb.i1, nb.f8, nb.f8, nb.f8[:]), cache=True)
def prearg_template_func(
    phase_State: int, p: float, T: float, xn: np.ndarray
) -> np.ndarray:
    """Template c-func for the pre-argument, both for property values and derivative
    values.

    Parameters:
        phase_State: See :class:`~porepy.compositional._core.PhysicalState`.
        p: Pressure value.
        T: Temperature value.
        xn: 1D array containing normalized fractions.

    Returns:
        Some 1D array.

    """
    if phase_State:
        return p * T * xn
    else:
        return xn


@cfunc(nb.f8(nb.f8[:], nb.f8, nb.f8, nb.f8[:]), cache=True)
def property_template_func(
    prearg_val: np.ndarray, p: float, T: float, xn: np.ndarray
) -> float:
    """Template c-func for a thermodynamic property.

    Used for numba type infering.

    Parameters:
        prearg_val: 1D array representing the pre-argument for property values.
        p: Pressure value.
        T: Temperature value.
        xn: 1D array representing normalized partial fractions.

    Returns:
        Some scalar value.

    """
    return prearg_val[0] * p * T * xn[0]


@cfunc(nb.f8[:](nb.f8[:], nb.f8[:], nb.f8, nb.f8, nb.f8[:]), cache=True)
def property_derivative_template_func(
    prearg_val: np.ndarray, prearg_jac: np.ndarray, p: float, T: float, xn: np.ndarray
) -> np.ndarray:
    """Template c-func for a thermodynamic property derivatives.

    Used for numba type infering.

    Parameters:
        prearg_val: 1D array representing the pre-argument for property value functions.
        prearg_jac: 1D array representing the pre-argument for property derivative
            functions.
        p: Pressure value.
        T: Temperature value.
        xn: 1D array representing normalized partial fractions.

    Returns:
        Some 1D array containing derivatives w.r.t. p, T and each fraction, i.e.
        ``(2 + xn.shape[0],)``.

    """
    return prearg_val[0] * prearg_jac[0] * p * T * xn


@cfunc(nb.f8[:](nb.f8[:], nb.f8, nb.f8, nb.f8[:]), cache=True)
def fugacity_coeff_template_func(
    prearg_val: np.ndarray, p: float, T: float, xn: np.ndarray
) -> np.ndarray:
    """Template c-func for fugacity coefficients.

    Used for numba type infering.

    The difference to :func:`property_template_func` and
    :func:`property_derivative_template_func` is that while still taking only one
    pre-argument (for values), it returns a vector (fugacities per component)

    Parameters:
        prearg_val: 1D array representing the pre-argument for property value functions.
        p: Pressure value.
        T: Temperature value.
        xn: 1D array representing normalized partial fractions.

    Returns:
        An array with the shape of ``xn``.

    """
    return prearg_val[0] * p * T * xn


@cfunc(nb.f8[:, :](nb.f8[:], nb.f8[:], nb.f8, nb.f8, nb.f8[:]), cache=True)
def fugacity_coeff_derivative_template_func(
    prearg_val: np.ndarray, prearg_jac: np.ndarray, p: float, T: float, xn: np.ndarray
) -> np.ndarray:
    """Template c-func for derivatives offugacity coefficients.

    Used for numba type infering.

    The difference to :func:`property_derivative_template_func` is that this returns
    derivatives for each fugacity coefficient, hence a 2D array.

    Parameters:
        prearg_val: 1D array representing the pre-argument for property value functions.
        prearg_jac: 1D array representing the pre-argument for property derivative
            functions.
        p: Pressure value.
        T: Temperature value.
        xn: 1D array representing normalized partial fractions.

    Returns:
        An array with the shape ``(xn.shape[0], 2 + xn.shape[0])``.

    """
    ncomp = xn.shape[0]
    return np.zeros((ncomp, 2 + ncomp)) * prearg_val[0] * prearg_jac[0] * p * T


# NOTE Every parallelized evaluation requires an own method because of the exact
# signatures. This ensures maximum efficiency when importing and executing the code
# continuously.
@nb.njit(
    nb.f8[:, :](typeof(prearg_template_func), nb.i1, nb.f8[:], nb.f8[:], nb.f8[:, :]),
    parallel=NUMBA_PARALLEL,
    cache=True,
)
def _evaluate_vectorized_prearg_func(
    prearg_func: Callable[[int, float, float, np.ndarray], np.ndarray],
    phase_state: int,
    p: np.ndarray,
    T: np.ndarray,
    xn: np.ndarray,
) -> np.ndarray:
    """Parallelized evaluation of some pre-argument function.

    Parameters:
        property_diffs_func: Property derivative function to be evaluated.
            See :func:`prearg_template_func`.
        phase_State: See :class:`~porepy.compositional._core.PhysicalState`.
        p: ``shape=(N,)``

            Pressure values.
        T: ``shape=(N,)``

            Temperature values.
        xn: ``shape(N, num_components)``

            (Normalized) partial fractions.

    Returns:
        An array of shape ``(N, M)``, where each row represents an evaluation of
        ``prearg_func``, evaluated with rows of the parameters.

    """
    N = p.shape[0]
    prearg_0 = prearg_func(phase_state, p[0], T[0], xn[0])
    prearg = np.empty((N, prearg_0.shape[0]))
    prearg[0] = prearg_0
    for i in nb.prange(1, N):
        prearg[i] = prearg_func(phase_state, p[i], T[i], xn[i])
    return prearg


@nb.njit(
    nb.f8[:](
        typeof(property_template_func), nb.f8[:, :], nb.f8[:], nb.f8[:], nb.f8[:, :]
    ),
    parallel=NUMBA_PARALLEL,
    cache=True,
)
def _evaluate_vectorized_property_func(
    property_func: Callable[[np.ndarray, float, float, np.ndarray], float],
    prearg: np.ndarray,
    p: np.ndarray,
    T: np.ndarray,
    xn: np.ndarray,
) -> np.ndarray:
    """Parallelized evaluation of a scalar function given by ``property_func``.

    Intended use is for evaluation of thermodynamic properties.

    Parameters:
        property_func: Property function to be evaluated.
        prearg: ``shape=(N, M)``

            Matrix containing pre-arguments row-wise.
        p: ``shape=(N,)``

            Pressure values.
        T: ``shape=(N,)``

            Temperature values.
        xn: ``shape(N, num_components)``

            (Normalized) partial fractions.

    Returns:
        An array of shape ``(N,)``, where each row represents an evaluation of
        ``property_func``, evaluated with rows of the parameters.

    """
    N = p.shape[0]
    vals = np.empty(N)
    for i in nb.prange(N):
        vals[i] = property_func(prearg[i], p[i], T[i], xn[i])
    return vals


@nb.njit(
    nb.f8[:, :](
        typeof(property_derivative_template_func),
        nb.f8[:, :],
        nb.f8[:, :],
        nb.f8[:],
        nb.f8[:],
        nb.f8[:, :],
    ),
    parallel=NUMBA_PARALLEL,
    cache=True,
)
def _evaluate_vectorized_property_derivatives_func(
    property_diffs_func: Callable[
        [np.ndarray, np.ndarray, float, float, np.ndarray], np.ndarray
    ],
    prearg_val: np.ndarray,
    prearg_jac: np.ndarray,
    p: np.ndarray,
    T: np.ndarray,
    xn: np.ndarray,
) -> np.ndarray:
    """Parallelized evaluation of a vector-valued function ``property_diffs_func``,
    representing the derivatives of some thermodynamic property..

    Intended use is for evaluation of thermodynamic properties.

    See also:
        :func:`_evaluate_vectorized_property_func`

    Parameters:
        property_diffs_func: Property derivative function to be evaluated.
        prearg_val: ``shape=(N, M1)``

            Matrix containing pre-arguments for the property function row-wise.
        prearg_jac: ``shape=(N, M2)``

            Matrix containing pre-arguments for the derivative function row-wise.
        p: ``shape=(N,)``

            Pressure values.
        T: ``shape=(N,)``

            Temperature values.
        xn: ``shape(N, num_components)``

            (Normalized) partial fractions.

    Returns:
        An array of shape ``(N,)``, where each row represents an evaluation of
        ``property_func``, evaluated with rows of the parameters.

    """
    N = p.shape[0]
    num_comp = xn.shape[1]
    diffs = np.empty((2 + num_comp, N))
    for i in nb.prange(N):
        diffs[:, i] = property_diffs_func(
            prearg_val[i], prearg_jac[i], p[i], T[i], xn[i]
        )
    return diffs


@nb.njit(
    nb.f8[:, :](
        typeof(fugacity_coeff_template_func),
        nb.f8[:, :],
        nb.f8[:],
        nb.f8[:],
        nb.f8[:, :],
    ),
    parallel=NUMBA_PARALLEL,
    cache=True,
)
def _evaluate_vectorized_fug_coeff_func(
    fug_coeff_func: Callable[[np.ndarray, float, float, np.ndarray], np.ndarray],
    prearg: np.ndarray,
    p: np.ndarray,
    T: np.ndarray,
    xn: np.ndarray,
) -> np.ndarray:
    """Parallelized evaluation of a vector function given by ``fug_coeff_func``.

    Intended use is for evaluation of fugacity coefficients.

    Parameters:
        fug_coeff_func: Fugacity coefficient function to be evaluated.
        prearg: ``shape=(N, M)``

            Matrix containing pre-arguments row-wise.
        p: ``shape=(N,)``

            Pressure values.
        T: ``shape=(N,)``

            Temperature values.
        xn: ``shape(N, num_components)``

            (Normalized) partial fractions.

    Returns:
        An array of shape ``(num_components, N)``, where each **column** represents an
        evaluation of ``fug_coeff_func``, evaluated with **rows** of the parameters.

        Note, that it is intended to return the coefficients like this, and not
        transposed (looping over component means looping over first axis of return
        value).

    """
    N, ncomp = xn.shape
    phis = np.empty((ncomp, N))
    for i in nb.prange(N):
        phis[:, i] = fug_coeff_func(prearg[i], p[i], T[i], xn[i])
    return phis


@nb.njit(
    nb.f8[:, :, :](
        typeof(fugacity_coeff_derivative_template_func),
        nb.f8[:, :],
        nb.f8[:, :],
        nb.f8[:],
        nb.f8[:],
        nb.f8[:, :],
    ),
    parallel=NUMBA_PARALLEL,
    cache=True,
)
def _evaluate_vectorized_fug_coeff_diff_func(
    fug_coeff_diff_func: Callable[
        [np.ndarray, np.ndarray, float, float, np.ndarray], np.ndarray
    ],
    prearg_val: np.ndarray,
    prearg_jac: np.ndarray,
    p: np.ndarray,
    T: np.ndarray,
    xn: np.ndarray,
) -> np.ndarray:
    """Parallelized evaluation of a matrix-valued function ``fug_coeff_diff_func``,
    representing the derivatives of the fugacity coefficients per component.

    Intended use is for evaluation of thermodynamic properties.

    See also:
        :func:`_evaluate_vectorized_fug_coeff_func`

    Parameters:
        fug_coeff_diff_func: Fugacity coefficient derivative function to be evaluated.
        prearg_val: ``shape=(N, M1)``

            Matrix containing pre-arguments for the property function row-wise.
        prearg_jac: ``shape=(N, M2)``

            Matrix containing pre-arguments for the derivative function row-wise.
        p: ``shape=(N,)``

            Pressure values.
        T: ``shape=(N,)``

            Temperature values.
        xn: ``shape(N, num_components)``

            (Normalized) partial fractions.

    Returns:
        An array of shape ``(num_components, 2 + num_components, N)``, where the first
        two dimensions represent the derivatives of the coefficients (per component).

    """
    n, ncomp = xn.shape
    dphis = np.empty((ncomp, 2 + ncomp, n))
    for i in nb.prange(n):
        dphis[:, :, i] = fug_coeff_diff_func(
            prearg_val[i], prearg_jac[i], p[i], T[i], xn[i]
        )
    return dphis


PropertyFunctionNames: TypeAlias = Literal[
    "prearg_val",
    "prearg_jac",
    "phis",
    "dphis",
    "h",
    "dh",
    "v",
    "dv",
    "rho",
    "drho",
    "mu",
    "dmu",
    "kappa",
    "dkappa",
]
"""Type alias for names/keys of property functions stored in :attr:`EoSCompiler.funcs`
and :attr:`EoSCompiler.gufuncs`."""


class EoSCompiler(EquationOfState):
    """Abstract base class for EoS compilation using numba.

    This class needs functions computing

    - fugacity coefficients
    - enthalpies
    - densities
    - the derivatives w.r.t. pressure, temperature and partial fractions (array)

    Respective functions must be assembled and compiled by a child class with a specific
    EoS.

    The compiled functions are expected to have a specific signature (see below).

    1. One or two pre-arguments (vectors), for property or derivative function
       respectively.
    2. Two scalar arguments representing pressure and temperature.
    3. A vector argument representing partial fractions.

    The purpose of the ``prearg`` is efficiency.
    Many EoS have computions of some co-terms or compressibility factors f.e.,
    which must only be computed once for all remaining thermodynamic quantities.

    The function for the ``prearg`` computation must have the signature:

    1. an integer representing a phase :attr:`~porepy.compositional.base.Phase.state`
    2. Two scalar arguments representing pressure and temperature.
    3. A vector argument representing partial fractions.

    There are two ``prearg`` computations: One for property values, one for the
    derivatives.

    The ``prearg`` for the derivatives will be fed to the functions representing
    derivatives of thermodynamic quantities **additionally** to the ``prearg`` for
    residuals.

    Example:
        The signature of functions representing derivatives is expected to be

        .. code:: python3

            def f(
                prearg_val: np.ndarray,
                prearg_jac: np.ndarray,
                p: float,
                T: float,
                x: np.ndarray
            ) -> np.ndarray:
                ...

    Parameters:
        components: Sequence of components for which the EoS should be compiled.
            The class :class:`~porepy.compositional.base.Component` is used as a storage
            for physical properties, the only relevant information for the EoS.

    """

    def __init__(self, components: Sequence[Component]) -> None:
        super().__init__(components)

        self.funcs: PropertyFunctionDict = {}
        """Dictionary for storing functions which are compiled in various
        ``get_*`` methods.

        Accessed during :meth:`compile` to create vectorized functions, which
        in return are stored in :attr:`gufuncs`.

        See documentation of type class for more information.

        """

        self.gufuncs: dict[PropertyFunctionNames, VectorFunction] = {}
        """Storage of vectorized versions of the functions found in :attr:`funcs`.

        To be used for efficient evaluation of properties after the flash converged.

        """

        self._is_compiled: bool = False
        """"Boolean to keep track of whether (costly) compilation process has been
        performed. Should be required only once.

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

        @nb.njit(nb.f8(nb.f8[:], nb.f8, nb.f8, nb.f8[:]))
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

        drho_c = self.funcs.get("drho", None)
        if drho_c is None:
            drho_c = self.get_density_derivative_function()

        rho_c = cast(ScalarFunction, rho_c)
        drho_c = cast(VectorFunction, drho_c)

        @nb.njit(nb.f8[:](nb.f8[:], nb.f8[:], nb.f8, nb.f8, nb.f8[:]))
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
        self.funcs["phis"] = self.get_fugacity_function()
        logger.debug("Compiling property functions 3/14")
        self.funcs["dphis"] = self.get_fugacity_derivative_function()
        logger.debug("Compiling property functions 4/14")
        self.funcs["h"] = self.get_enthalpy_function()
        logger.debug("Compiling property functions 5/14")
        self.funcs["dh"] = self.get_enthalpy_derivative_function()
        logger.debug("Compiling property functions 6/14")
        self.funcs["rho"] = self.get_density_function()
        logger.debug("Compiling property functions 7/14")
        self.funcs["drho"] = self.get_density_derivative_function()
        logger.debug("Compiling property functions 8/14")
        self.funcs["v"] = self.get_volume_function()
        logger.debug("Compiling property functions 9/14")
        self.funcs["dv"] = self.get_volume_derivative_function()
        logger.debug("Compiling property functions 10/14")
        self.funcs["mu"] = self.get_viscosity_function()
        logger.debug("Compiling property functions 11/14")
        self.funcs["dmu"] = self.get_viscosity_derivative_function()
        logger.debug("Compiling property functions 12/14")
        self.funcs["kappa"] = self.get_conductivity_function()
        logger.debug("Compiling property functions 13/14")
        self.funcs["dkappa"] = self.get_conductivity_derivative_function()
        logger.debug("Compiling property functions 14/14")
        # endregion

        logger.info("Assembling vectorized functions ..")

        # Constructint vectorized computations for fast evaluation of properties.
        k: PropertyFunctionNames
        dk: PropertyFunctionNames
        # Awkward definition of keys is for mypy.
        keys: list[PropertyFunctionNames] = ["prearg_val", "prearg_jac"]
        for k in keys:
            self.gufuncs[k] = partial(_evaluate_vectorized_prearg_func, self.funcs[k])

        self.gufuncs["phis"] = partial(
            _evaluate_vectorized_fug_coeff_func, self.funcs["phis"]
        )
        self.gufuncs["dphis"] = partial(
            _evaluate_vectorized_fug_coeff_diff_func, self.funcs["dphis"]
        )

        keys = ["h", "rho", "v", "mu", "kappa"]
        for k in keys:
            self.gufuncs[k] = partial(_evaluate_vectorized_property_func, self.funcs[k])
            dk = cast(PropertyFunctionNames, f"d{k}")
            self.gufuncs[dk] = partial(
                _evaluate_vectorized_property_derivatives_func, self.funcs[dk]
            )

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
            phase_State: See :class:`~porepy.compositional._core.PhysicalState`.
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

        """

        x = thermodynamic_input[-1]
        assert x.ndim == 2, (
            "Last thermodynamic input expected to be  a 2D array (fractions)."
        )

        # NOTE: The vectorized functions expect fractions column-wise, while the
        # remainig framework expects them row-wise.
        # This is because the remaining framework iterates usually over components and
        # the primary axis of 2D arrays is the zero-th (rows).
        # The parallel evaluation on the other hand, takes all arguments row-wise, hence
        # the transpose here.
        x_norm = normalize_rows(x.T)

        thermodynamic_input = tuple([_ for _ in thermodynamic_input[:-1]] + [x_norm])

        prearg_val = self.gufuncs["prearg_val"](phase_state.value, *thermodynamic_input)
        prearg_jac = self.gufuncs["prearg_jac"](phase_state.value, *thermodynamic_input)

        props = {}
        k: PropertyFunctionNames
        dk: PropertyFunctionNames

        keys: list[PropertyFunctionNames] = ["h", "rho", "phis", "mu", "kappa"]
        for k in keys:
            props[k] = self.gufuncs[k](prearg_val, *thermodynamic_input)
            dk = cast(PropertyFunctionNames, f"d{k}")
            props[dk] = self.gufuncs[dk](prearg_val, prearg_jac, *thermodynamic_input)

        return PhaseProperties(
            state=phase_state, x=x, **cast(dict[str, np.ndarray], props)
        )
