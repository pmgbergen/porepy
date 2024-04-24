"""Module containing the abstract base class for compiling EoS-related functions used
in the evaluation of thermodynamic properties."""

from __future__ import annotations

import abc
import logging
from typing import Callable, Optional, Sequence

import numba
import numpy as np

from .base import AbstractEoS, Component
from .states import PhaseState
from .utils_c import extend_fractional_derivatives, normalize_rows

__all__ = [
    "EoSCompiler",
]

logger = logging.getLogger(__name__)


def _compile_vectorized_prearg(
    func_c: Callable[[int, float, float, np.ndarray], np.ndarray]
) -> Callable[[int, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Helper function implementing the parallelized, compiled computation of
    pre-argument functions ``func_c``, which is called element-wise."""

    @numba.njit("float64[:,:](int32,float64[:],float64[:],float64[:,:])", parallel=True)
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


def _compile_vectorized_fugacity_coeffs(
    phi_c: Callable[[np.ndarray, float, float, np.ndarray], np.ndarray]
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Helper function implementing the parallelized, compiled computation of
    fugacity coefficients given by ``phi_c``.

    The resulting 2D array will contain row-wise the fugacity coefficients per
    component, analogous to the scalar case given by
    :meth:`EoSCompiler.get_dpTX_fugacity_function`.

    The second dimension will reflect the vectorized input.

    """

    @numba.njit(
        "float64[:,:](float64[:,:],float64[:],float64[:],float64[:,:])", parallel=True
    )
    def inner(prearg: np.ndarray, p: np.ndarray, T: np.ndarray, xn: np.ndarray):
        ncomp, N = xn.shape
        phis = np.empty((N, ncomp))
        for i in numba.prange(N):
            phis[i] = phi_c(prearg[i], p[i], T[i], xn[:, i])
        # phis per component row-wise
        return phis.T

    return inner


def _compile_vectorized_fugacity_coeff_derivatives(
    d_phi_c: Callable[[np.ndarray, float, float, np.ndarray], np.ndarray]
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
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
        parallel=True,
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
    func_c: Callable[[np.ndarray, float, float, np.ndarray], float]
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Helper function implementing the parallelized, compiled computation of
    properties given by ``func_c`` element-wise."""

    @numba.njit(
        "float64[:](float64[:,:],float64[:],float64[:],float64[:,:])", parallel=True
    )
    def inner(prearg: np.ndarray, p: np.ndarray, T: np.ndarray, xn: np.ndarray):
        _, N = xn.shape
        vals = np.empty(N)
        for i in numba.prange(N):
            vals[i] = func_c(prearg[i], p[i], T[i], xn[:, i])
        return vals

    return inner


def _compile_vectorized_property_derivatives(
    func_c: Callable[[np.ndarray, np.ndarray, float, float, np.ndarray], np.ndarray]
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Helper function implementing the parallelized, compiled computation of
    property derivatives given by ``func_c`` element-wise.

    The resulting 2D array has structure:

    - First dimension per derivative (pressure, temperature, dx per fraction)
    - Second dimension per element in vectorized input

    """

    @numba.njit(
        "float64[:,:](float64[:,:],float64[:,:],float64[:],float64[:],float64[:,:])",
        parallel=True,
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


class EoSCompiler(AbstractEoS):
    """Abstract base class for EoS specific compilation using numba.

    The :class:`EoSCompiler` needs functions computing

    - fugacity coefficients
    - enthalpies
    - volumes
    - the derivatives of above w.r.t. pressure, temperature and phase compositions

    Respective functions must be assembled and compiled by a child class with a specific
    EoS.

    Important:
        This class fixes the thermodynamic input for computing phase properties to
        pressure, temperature and partial fractions.

    The compiled functions are expected to have a specific signature (see below).

    1. One or two pre-arguments (vectors)
    2. ``p``: The pressure value.
    3. ``T``: The temperature value.
    4. ``xn``: An array with ``shape=(num_comp,)`` containing the normalized fractions
       per component of a phase.

    The purpose of the ``prearg`` is efficiency.
    Many EoS have computions of some coterms or compressibility factors f.e.,
    which must only be computed once for all remaining thermodynamic quantities.

    The function for the ``prearg`` computation must have the signature:

    ``(phasetype: int, p: float, T: float, xn: np.ndarray)``

    where ``xn`` contains normalized fractions,

    There are two ``prearg`` computations: One for property values, one for the
    derivatives.

    The ``prearg`` for the derivatives will be fed to the functions representing
    derivatives of thermodynamic quantities
    (e.g. derivative fugacity coefficients w.r.t. p, T, X),
    **additionally** to the ``prearg`` for residuals.

    I.e., the signature of functions representing derivatives is expected to be

    ``(prearg_res: np.ndarray, prearg_jac: np.ndarray,
    p: float, T: float, xn: np.ndarray)``,

    whereas the signature of functions representing values only is expected to be

    ``(prearg: np.ndarray, p: float, T: float, xn: np.ndarray)``

    Important:
        Functions compiled in the abstract methods can be stored in :attr:`funcs`
        to be re-used in the compilation of generalized ufuncs for fast evaluation of
        properties. See :meth:`compile` and :attr:`gufuncs`.

        If stored, the functions will be accessed to compile efficient, vectorized
        computations of thermodynamic quantities, otherwise the respective
        ``get_*``-method will be called again.

    Parameters:
        components: Sequence of components for which the EoS should be compiled.
            The class :class:`~porepy.composite.base.Component` is used as a storage for
            physical properties, the only relevant information for the EoS.

    """

    def __init__(self, components: Sequence[Component]) -> None:
        super().__init__(components)

        self.funcs: dict[str, Optional[Callable]] = {
            "prearg_val": None,
            "prearg_jac": None,
            "phi": None,
            "d_phi": None,
            "h": None,
            "d_h": None,
            "v": None,
            "d_v": None,
            "rho": None,
            "d_rho": None,
        }
        """Dictionary for storing functions which are compiled in various
        ``get_*`` methods.

        Accessed during :meth:`compile` to create vectorized functions, which
        in return are stored in :attr:`gufuncs`.

        Keywords for storage are:

        - ``'prearg_res'``: Function compiled by :meth:`get_pre_arg_function_res`
        - ``'prearg_jac'``: Function compiled by :meth:`get_pre_arg_function_jac`
        - ``'phi'``: Function compiled by :meth:`get_fugacity_function`
        - ``'d_phi'``: Function compiled by :meth:`get_dpTX_fugacity_function`
        - ``'h'``: Function compiled by :meth:`get_enthalpy_function`
        - ``'d_h'``: Function compiled by :meth:`get_dpTX_enthalpy_function`
        - ``'v'``: Function compiled by :meth:`get_volume_function`
        - ``'d_v'``: Function compiled by :meth:`get_dpTX_volume_function`
        - ``'rho'``: Function compiled by :meth:`get_density_function`
        - ``'d_rho'``: Function compiled by :meth:`get_dpTX_density_function`

        """

        self.gufuncs: dict[
            str,
            Optional[
                Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
                | Callable[
                    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                    np.ndarray,
                ]
            ],
        ] = {
            "prearg_val": None,
            "prearg_res": None,
            "phi": None,
            "d_phi": None,
            "h": None,
            "d_h": None,
            "v": None,
            "d_v": None,
            "rho": None,
            "d_rho": None,
        }
        """Storage of generalized functions for computing thermodynamic properties.

        The functions are created when calling :meth:`compile`.

        The purpose of these functions is a fast evaluation of properties which

        1. are secondary in the flash (evaluated after convergence)
        2. which need to be evaluated for multiple values states for example on a grid
           in flow and transport.

        Important:
            Every generalized function has the following signature:

            1. pre-argument for values
            2. (for derivatives only) pre-argument for derivatives
            3. 1D-array for pressure values
            4. 1D-array for temperature values
            5. 2D- array containing **row-wise** fractions per component

            The number of rows in each argument must be the same.

        """

    # TODO what is more efficient, just one pre-arg having everything?
    # Or splitting for computations for residuals, since it does not need derivatives?
    # 1. Armijo line search evaluated often, need only residual
    # 2. On the other hand, residual pre-arg is evaluated twice, for residual and jac
    @abc.abstractmethod
    def get_prearg_for_values(
        self,
    ) -> Callable[[int, float, float, np.ndarray], np.ndarray]:
        """Abstract function for obtaining the compiled computation of the pre-argument
        for the evaluation of thermodynamic properties.

        Returns:
            A NJIT-ed function with signature
            ``(phasetype: int, p: float, T: float, xn: np.ndarray)``
            returning a 1D array.

        """
        pass

    @abc.abstractmethod
    def get_prearg_for_derivatives(
        self,
    ) -> Callable[[int, float, float, np.ndarray], np.ndarray]:
        """Abstract function for obtaining the compiled computation of the pre-argument
        for the evaluation of derivatives of thermodynamic properties.

        Returns:
            A NJIT-ed function with signature
            ``(phasetype: int, p: float, T: float, xn: np.ndarray)``
            returning a 1D array.

        """
        pass

    @abc.abstractmethod
    def get_fugacity_function(
        self,
    ) -> Callable[[np.ndarray, float, float, np.ndarray], np.ndarray]:
        """Abstract assembler for compiled computations of the fugacity coefficients.

        Returns:
            A NJIT-ed function taking

            - pre-argument for values (1D-array),
            - pressure value,
            - temperature value,
            - an 1D-array of normalized fractions of components in a phase,

            and returning an array of fugacity coefficients with ``shape=(num_comp,)``.

        """
        pass

    @abc.abstractmethod
    def get_dpTX_fugacity_function(
        self,
    ) -> Callable[[np.ndarray, np.ndarray, float, float, np.ndarray], np.ndarray]:
        """Abstract assembler for compiled computations of the derivative of fugacity
        coefficients.

        The functions should return the derivative fugacities for each component
        row-wise in a matrix.
        It must contain the derivatives w.r.t. pressure, temperature and each fraction
        in a specified phase.
        I.e. the return value must be an array with ``shape=(num_comp, 2 + num_comp)``.

        Returns:
            A NJIT-ed function taking

            - pre-argument for values (1D-array),
            - pre-argument for derivatives (1D-array),
            - pressure value,
            - temperature value,
            - an 1D-array of normalized fractions of components of a phase,

            and returning an array of derivatives of fugacity coefficients with
            ``shape=(num_comp, 2 + num_comp)``., where containing the derivatives w.r.t.
            pressure, temperature and fractions (columns), for each component (row).

        """
        pass

    @abc.abstractmethod
    def get_enthalpy_function(
        self,
    ) -> Callable[[np.ndarray, float, float, np.ndarray], float]:
        """Abstract assembler for compiled computations of the specific molar enthalpy.

        Returns:
            A NJIT-ed function taking

            - pre-argument for values (1D-array),
            - pressure value,
            - temperature value,
            - an 1D-array of normalized fractions of components of a phase,

            and returning an enthalpy value.

        """
        pass

    @abc.abstractmethod
    def get_dpTX_enthalpy_function(
        self,
    ) -> Callable[[np.ndarray, np.ndarray, float, float, np.ndarray], np.ndarray]:
        """Abstract assembler for compiled computations of the derivative of the
        enthalpy function for a phase.

        Returns:
            A NJIT-ed function taking

            - pre-argument for values (1D-array),
            - pre-argument for derivatives (1D-array),
            - pressure value,
            - temperature value,
            - an 1D-array of normalized fractions of components of a phase,

            and returning an array of derivatives of the enthalpy with
            ``shape=(2 + num_comp,)``., containing the derivatives w.r.t.
            pressure, temperature and fractions.

        """
        pass

    @abc.abstractmethod
    def get_volume_function(
        self,
    ) -> Callable[[np.ndarray, float, float, np.ndarray], float]:
        """Abstract assembler for compiled computations of the specific molar volume.

        Returns:
            A NJIT-ed function taking

            - pre-argument for values (1D-array),
            - pressure value,
            - temperature value,
            - an 1D-array of normalized fractions of components of a phase,

            and returning a volume value.

        """
        pass

    @abc.abstractmethod
    def get_dpTX_volume_function(
        self,
    ) -> Callable[[np.ndarray, np.ndarray, float, float, np.ndarray], np.ndarray]:
        """Abstract assembler for compiled computations of the derivative of the
        volume function for a phase.

        Returns:
            A NJIT-ed function taking

            - pre-argument for values (1D-array),
            - pre-argument for derivatives (1D-array),
            - pressure value,
            - temperature value,
            - an 1D-array of normalized fractions of components of a phase,

            and returning an array of derivatives of the volume with
            ``shape=(2 + num_comp,)``., containing the derivatives w.r.t.
            pressure, temperature and fractions.

        """
        pass

    @abc.abstractmethod
    def get_viscosity_function(
        self,
    ) -> Callable[[np.ndarray, float, float, np.ndarray], float]:
        """Abstract assembler for compiled computations of the dynamic molar viscosity.

        Returns:
            A NJIT-ed function taking

            - pre-argument for values (1D-array),
            - pressure value,
            - temperature value,
            - an 1D-array of normalized fractions of components of a phase,

            and returning a viscosity value.

        """
        pass

    @abc.abstractmethod
    def get_dpTX_viscosity_function(
        self,
    ) -> Callable[[np.ndarray, np.ndarray, float, float, np.ndarray], np.ndarray]:
        """Abstract assembler for compiled computations of the derivative of the
        viscosity function for a phase.

        Returns:
            A NJIT-ed function taking

            - pre-argument for values (1D-array),
            - pre-argument for derivatives (1D-array),
            - pressure value,
            - temperature value,
            - an 1D-array of normalized fractions of components of a phase,

            and returning an array of derivatives of the viscosity with
            ``shape=(2 + num_comp,)``., containing the derivatives w.r.t.
            pressure, temperature and fractions.

        """
        pass

    @abc.abstractmethod
    def get_conductivity_function(
        self,
    ) -> Callable[[np.ndarray, float, float, np.ndarray], float]:
        """Abstract assembler for compiled computations of the thermal conductivity.

        Returns:
            A NJIT-ed function taking

            - pre-argument for values (1D-array),
            - pressure value,
            - temperature value,
            - an 1D-array of normalized fractions of components of a phase,

            and returning a conductivity value.

        """
        pass

    @abc.abstractmethod
    def get_dpTX_conductivity_function(
        self,
    ) -> Callable[[np.ndarray, np.ndarray, float, float, np.ndarray], np.ndarray]:
        """Abstract assembler for compiled computations of the derivative of the
        conductivity function for a phase.

        Returns:
            A NJIT-ed function taking

            - pre-argument for values (1D-array),
            - pre-argument for derivatives (1D-array),
            - pressure value,
            - temperature value,
            - an 1D-array of normalized fractions of components of a phase,

            and returning an array of derivatives of the conductivity with
            ``shape=(2 + num_comp,)``., containing the derivatives w.r.t.
            pressure, temperature and fractions.

        """
        pass

    def get_density_function(
        self,
    ) -> Callable[[np.ndarray, float, float, np.ndarray], float]:
        """Assembler for compiled computations of the specific molar density.

        The density is computed as the reciprocal of the return value of
        :meth:`get_volume_function`.

        Note:
            This function is compiled faster, if the volume function has already been
            compiled and stored in :attr:`funcs`.

        Returns:
            A NJIT-ed function taking

            - pre-argument for values (1D-array),
            - pressure value,
            - temperature value,
            - an 1D-array of normalized fractions of components of a phase,

            and returning a volume value.

        """
        v_c = self.funcs.get("v", None)
        if v_c is None:
            v_c = self.get_volume_function()

        @numba.njit("float64(float64[:],float64,float64,float64[:])")
        def rho_c(prearg: np.ndarray, p: float, T: float, xn: np.ndarray) -> np.ndarray:
            return 1.0 / v_c(prearg, p, T, xn)

        return rho_c

    def get_dpTX_density_function(
        self,
    ) -> Callable[[np.ndarray, np.ndarray, float, float, np.ndarray], np.ndarray]:
        """Assembler for compiled computations of the derivative of the
        density function for a phase.

        Density is expressed as the reciprocal of volume.
        Hence the computations utilize :meth:`get_volume_function`,
        :meth:`get_dpTX_volume_function` and the chain-rule to compute the derivatives.

        Note:
            This function is compiled faster, if the volume function and its deritvative
            have already been compiled and stored in :attr:`funcs`.

        Returns:
            A NJIT-ed function taking

            - pre-argument for values (1D-array),
            - pre-argument for derivatives (1D-array),
            - pressure value,
            - temperature value,
            - an 1D-array of normalized fractions of components of a phase,

            and returning an array of derivatives of the density with
            ``shape=(2 + num_comp,)``., containing the derivatives w.r.t.
            pressure, temperature and fractions.

        """
        v_c = self.funcs.get("v", None)
        if v_c is None:
            v_c = self.get_volume_function()

        dv_c = self.funcs.get("d_v", None)
        if dv_c is None:
            dv_c = self.get_dpTX_volume_function()

        @numba.njit("float64[:](float64[:],float64[:],float64,float64,float64[:])")
        def drho_c(
            prearg_res: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
        ) -> np.ndarray:
            v = v_c(prearg_res, p, T, xn)
            dv = dv_c(prearg_res, prearg_jac, p, T, xn)
            # chain rule: drho = d(1 / v) = - 1 / v**2 * dv
            return -dv / v**2

        return drho_c

    def compile(self) -> None:
        """Compiles vectorized functions for properties, depending on pre-arguments,
        pressure, temperature and fractions.

        Accesses :attr:`funcs` to find functions for element-wise computations.
        If not found, calls various ``get_*`` methods to create them and stores them
        in :attr:`funcs`.

        Important:
            This function takes long to complete. It compiles all scalar, and vectorized
            computations of properties.

        """

        logger.info("Compiling property functions ..")

        # region Element-wise computations
        prearg_val_c = self.funcs.get("prearg_val", None)
        if prearg_val_c is None:
            prearg_val_c = self.get_prearg_for_values()
            self.funcs["prearg_val"] = prearg_val_c
        logger.debug("Compiling property functions 1/14")

        prearg_jac_c = self.funcs.get("prearg_jac", None)
        if prearg_jac_c is None:
            prearg_jac_c = self.get_prearg_for_derivatives()
            self.funcs["prearg_jac"] = prearg_jac_c
        logger.debug("Compiling property functions 2/14")

        phi_c = self.funcs.get("phi", None)
        if phi_c is None:
            phi_c = self.get_fugacity_function()
            self.funcs["phi"] = phi_c
        logger.debug("Compiling property functions 3/14")

        d_phi_c = self.funcs.get("d_phi", None)
        if d_phi_c is None:
            d_phi_c = self.get_dpTX_fugacity_function()
            self.funcs["d_phi"] = d_phi_c
        logger.debug("Compiling property functions 4/14")

        h_c = self.funcs.get("h", None)
        if h_c is None:
            h_c = self.get_enthalpy_function()
            self.funcs["h"] = h_c
        logger.debug("Compiling property functions 5/14")

        d_h_c = self.funcs.get("d_h", None)
        if d_h_c is None:
            d_h_c = self.get_dpTX_enthalpy_function()
            self.funcs["d_h"] = d_h_c
        logger.debug("Compiling property functions 6/14")

        v_c = self.funcs.get("v", None)
        if v_c is None:
            v_c = self.get_volume_function()
            self.funcs["v"] = v_c
        logger.debug("Compiling property functions 7/14")

        d_v_c = self.funcs.get("d_v", None)
        if d_v_c is None:
            d_v_c = self.get_dpTX_volume_function()
            self.funcs["d_v"] = d_v_c
        logger.debug("Compiling property functions 8/14")

        rho_c = self.funcs.get("rho", None)
        if rho_c is None:
            rho_c = self.get_density_function()
            self.funcs["rho"] = rho_c
        logger.debug("Compiling property functions 9/14")

        d_rho_c = self.funcs.get("d_rho", None)
        if d_rho_c is None:
            d_rho_c = self.get_dpTX_density_function()
            self.funcs["d_rho"] = d_rho_c
        logger.debug("Compiling property functions 10/14")

        mu_c = self.funcs.get("mu", None)
        if mu_c is None:
            mu_c = self.get_viscosity_function()
            self.funcs["mu"] = mu_c
        logger.debug("Compiling property functions 11/14")

        d_mu_c = self.funcs.get("d_mu", None)
        if d_mu_c is None:
            d_mu_c = self.get_dpTX_viscosity_function()
            self.funcs["d_mu"] = d_mu_c
        logger.debug("Compiling property functions 12/14")

        kappa_c = self.funcs.get("kappa", None)
        if kappa_c is None:
            kappa_c = self.get_conductivity_function()
            self.funcs["kappa"] = kappa_c
        logger.debug("Compiling property functions 13/14")

        d_kappa_c = self.funcs.get("d_kappa", None)
        if d_kappa_c is None:
            d_kappa_c = self.get_dpTX_conductivity_function()
            self.funcs["d_kappa"] = d_kappa_c
        logger.debug("Compiling property functions 14/14")
        # endregion

        logger.info("Compiling vectorized functions ..")

        # region vectorized computations
        prearg_val_v = _compile_vectorized_prearg(prearg_val_c)
        logger.debug("Compiling vectorized functions 1/14")
        prearg_jac_v = _compile_vectorized_prearg(prearg_jac_c)
        logger.debug("Compiling vectorized functions 2/14")
        phi_v = _compile_vectorized_fugacity_coeffs(phi_c)
        logger.debug("Compiling vectorized functions 3/14")
        d_phi_v = _compile_vectorized_fugacity_coeff_derivatives(d_phi_c)
        logger.debug("Compiling vectorized functions 4/14")
        h_v = _compile_vectorized_property(h_c)
        logger.debug("Compiling vectorized functions 5/14")
        d_h_v = _compile_vectorized_property_derivatives(d_h_c)
        logger.debug("Compiling vectorized functions 6/14")
        v_v = _compile_vectorized_property(v_c)
        logger.debug("Compiling vectorized functions 7/14")
        d_v_v = _compile_vectorized_property_derivatives(d_v_c)
        logger.debug("Compiling vectorized functions 8/14")
        rho_v = _compile_vectorized_property(rho_c)
        logger.debug("Compiling vectorized functions 9/14")
        d_rho_v = _compile_vectorized_property_derivatives(d_rho_c)
        logger.debug("Compiling vectorized functions 10/14")
        mu_v = _compile_vectorized_property(mu_c)
        logger.debug("Compiling vectorized functions 11/14")
        d_mu_v = _compile_vectorized_property_derivatives(d_mu_c)
        logger.debug("Compiling vectorized functions 12/14")
        kappa_v = _compile_vectorized_property(kappa_c)
        logger.debug("Compiling vectorized functions 13/14")
        d_kappa_v = _compile_vectorized_property_derivatives(d_kappa_c)
        logger.debug("Compiling vectorized functions 14/14")

        self.gufuncs.update(
            {
                "prearg_val": prearg_val_v,
                "prearg_jac": prearg_jac_v,
                "phi": phi_v,
                "d_phi": d_phi_v,
                "h": h_v,
                "d_h": d_h_v,
                "v": v_v,
                "d_v": d_v_v,
                "rho": rho_v,
                "d_rho": d_rho_v,
                "mu": mu_v,
                "d_mu": d_mu_v,
                "kappa": kappa_v,
                "d_kappa": d_kappa_v,
            }
        )
        # endregion

    def compute_phase_state(
        self,
        phasetype: int,
        p: np.ndarray,
        T: np.ndarray,
        x: Sequence[np.ndarray],
    ) -> PhaseState:
        """This method must only be called after the vectorized computations have been
        compiled (see :meth:`compile`).

        Parameters:
            phasetype: Type of phase (passed to pre-arg computation).
            p: ``shape=(N,)``

                Pressure values.
            T: ``shape=(N,)``

                Temperature values.
            x: ``shape=(num_comp, N)``

                Extended fractions per component (row-wise).
                They will be normalized before computing properties.

                Derivatives of properties w.r.t. to normalized fractions will be
                extended to derivatives w.r.t. to extended fractions.

        Returns:
            A complete datastructure containing values for thermodynamic phase
            properties and their derivatives.

            All sequential data fields are storred as arrays, with the sequence
            reflected in the first dimension.

        """

        # normalization of fractions for computing properties
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        xn = normalize_rows(x.T).T

        ncomp, _ = x.shape

        prearg_val = self.gufuncs["prearg_val"](phasetype, p, T, xn)
        prearg_jac = self.gufuncs["prearg_jac"](phasetype, p, T, xn)

        state = PhaseState(
            phasetype=phasetype,
            x=x,
            h=self.gufuncs["h"](prearg_val, p, T, xn),
            v=self.gufuncs["v"](prearg_val, p, T, xn),
            # shape = (num_comp, num_vals), sequence per component
            phis=self.gufuncs["phi"](prearg_val, p, T, xn),
            # shape = (num_diffs, num_vals), sequence per derivative
            dh=self.gufuncs["d_h"](prearg_val, prearg_jac, p, T, xn),
            # shape = (num_diffs, num_vals), sequence per derivative
            dv=self.gufuncs["d_v"](prearg_val, prearg_jac, p, T, xn),
            # shape = (num_comp, num_diffs, num_vals)
            dphis=self.gufuncs["d_phi"](prearg_val, prearg_jac, p, T, xn),
            mu=self.gufuncs["mu"](prearg_val, p, T, xn),
            dmu=self.gufuncs["d_mu"](prearg_val, prearg_jac, p, T, xn),
            kappa=self.gufuncs["kappa"](prearg_val, p, T, xn),
            dkappa=self.gufuncs["d_kappa"](prearg_val, prearg_jac, p, T, xn),
        )

        # Extending derivatives to extended fractions
        state.dh = extend_fractional_derivatives(state.dh, x)
        state.dv = extend_fractional_derivatives(state.dv, x)
        state.dmu = extend_fractional_derivatives(state.dmu, x)
        state.dkappa = extend_fractional_derivatives(state.dkappa, x)
        for i in range(ncomp):
            # TODO check if this is indexed correctly
            state.dphis[i] = extend_fractional_derivatives(state.dphis[i], x)

        return state
