"""DEPRACATED.

This module contains a class implementing the Peng-Robinson EoS for either
a liquid- or gas-like phase."""

from __future__ import annotations

import abc
import logging
import numbers
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp

from .._core import R_IDEAL_MOL
from ..base import Component
from ..utils import safe_sum

__all__ = [
    "PhaseProperties_cubic",
    "PengRobinson",
    "A_CRIT",
    "B_CRIT",
    "Z_CRIT",
    "critical_line",
    "widom_line",
]

logger = logging.getLogger(__name__)

DeprecationWarning("The module porepy.compositional.peng_robinson.eos is deprecated.")

NumericType = Union[pp.number, np.ndarray, pp.ad.AdArray]
"""DEPRECATED: Delete with finalization of flash and eos update to numba. TODO"""


def _sqrt(a):
    return a ** (1 / 2)


def _power(a, b):
    return a**b


def _cbrt(a):
    return a ** (1 / 3)


def truncexp(var):
    if isinstance(var, pp.ad.AdArray):
        trunc = var > 700
        val = np.exp(var.val, where=(~trunc))
        val[trunc] = np.exp(700)
        der = var._diagvec_mul_jac(val)
        return pp.ad.AdArray(val, der)
    else:
        trunc = var > 700
        val = np.exp(var, where=(~trunc))
        val[trunc] = np.exp(700)
        return val


def trunclog(var, eps):
    if isinstance(var, pp.ad.AdArray):
        trunc_val = np.maximum(var.val, eps)
        val = np.log(trunc_val)
        der = var._diagvec_mul_jac(1 / trunc_val)
        return pp.ad.AdArray(val, der)
    else:
        return np.log(np.maximum(var, eps))


@dataclass
class ThermodynamicState:
    """Data class for storing the thermodynamic state of a mixture..

    The name of the attributes ``p, T, X`` is designed such that they can be used
    as keyword arguments for

    1. :meth:`~porepy.compositional.phase.AbstractEoS.compute`,
    2. :meth:`~porepy.compositional.phase.Phase.compute_properties` and
    3. :meth:`~porepy.compositional.mixture.BasicMixture.compute_properties`,

    and should not be meddled with (especially capital ``X``).

    Important:
        Upon inheritance, always provide default values for :meth:`initialize` to work.

    """

    p: NumericType = 0.0
    """Pressure."""

    T: NumericType = 0.0
    """Temperature."""

    h: NumericType = 0.0
    """Specific molar enthalpy of the mixture."""

    v: NumericType = 0.0
    """Molar volume of the mixture."""

    rho: NumericType = 0.0
    """Molar density of the mixture.

    As of now, density is always considered a secondary expression and never an
    independent variable.

    """

    z: list[NumericType] = field(default_factory=lambda: [])
    """Feed fractions per component. The first fraction is always the feed fraction of
    the reference component."""

    y: list[NumericType] = field(default_factory=lambda: [])
    """Phase fractions per phase. The first fraction is always the phase fraction of the
    reference phase."""

    s: list[NumericType] = field(default_factory=lambda: [])
    """Volume fractions (saturations) per phase. The first fraction is always the phase
    saturation of the reference phase."""

    X: list[list[NumericType]] = field(default_factory=lambda: [[]])
    """Phase compositions per phase (outer list) per component in phase (inner list)."""

    def __str__(self) -> str:
        """Returns a string representation of the stored state values."""
        vals = self.values()
        nc = len(self.z)
        np = len(self.y)

        msg = f"Thermodynamic state with {nc} components and {np} phases:\n"
        msg += f"\nIntensive state:\n\tPressure: {vals.p}\n\tTemperature: {vals.T}"
        for i, z in enumerate(vals.z):
            msg += f"\n\tFeed fraction {i}: {z}"
        for j, y in enumerate(vals.y):
            msg += f"\n\tPhase fraction {j}: {y}"
        for j, s in enumerate(vals.s):
            msg += f"\n\tPhase saturation {j}: {s}"
        for j in range(np):
            msg += f"\n\tComposition phase {j}:"
            for i in range(nc):
                msg += f"\n\t\t Component {i}: {vals.X[j][i]}"
        msg += (
            f"\nExtensive state:\n\tSpec. Enthalpy: {vals.h}"
            + f"\n\tMol. Density: {vals.rho}\n\tMol. Volume: {vals.v}"
        )

        return msg

    def diff(self, other: ThermodynamicState) -> ThermodynamicState:
        """Returns a state containing the absolute difference between this instance
        and another state.

        The difference is calculated per state function and fraction and uses only
        values (no derivatives, if any is given as an AD-array).

        Parameters:
            other: The other thermodynamic state.

        Returns:
            A new data class instance containing absolute difference values.

        """
        sv = self.values()
        ov = other.values()

        p = np.abs(sv.p - ov.p)
        T = np.abs(sv.T - ov.T)
        h = np.abs(sv.h - ov.h)
        v = np.abs(sv.v - ov.v)
        rho = np.abs(sv.rho - ov.rho)
        z = [np.abs(sz - oz) for sz, oz in zip(sv.z, ov.z)]
        y = [np.abs(sy - oy) for sy, oy in zip(sv.y, ov.y)]
        s = [np.abs(ss - os) for ss, os in zip(sv.s, ov.s)]
        X = [[np.abs(sx - ox) for sx, ox in zip(Xs, Xo)] for Xs, Xo in zip(sv.X, ov.X)]

        return ThermodynamicState(p=p, T=T, h=h, v=v, rho=rho, z=z, y=y, s=s, X=X)

    def values(self) -> ThermodynamicState:
        """Returns a derivative-free state in case any state function is stored as
        an :class:`~porepy.numerics.ad.forward_mode.AdArray`."""

        p = self.p.val if isinstance(self.p, pp.ad.AdArray) else self.p
        T = self.T.val if isinstance(self.T, pp.ad.AdArray) else self.T
        h = self.h.val if isinstance(self.h, pp.ad.AdArray) else self.h
        v = self.v.val if isinstance(self.v, pp.ad.AdArray) else self.v
        rho = self.rho.val if isinstance(self.rho, pp.ad.AdArray) else self.rho
        z = [z.val if isinstance(z, pp.ad.AdArray) else z for z in self.z]
        y = [y.val if isinstance(y, pp.ad.AdArray) else y for y in self.y]
        s = [s.val if isinstance(s, pp.ad.AdArray) else s for s in self.s]
        X = [
            [x.val if isinstance(x, pp.ad.AdArray) else x for x in x_j]
            for x_j in self.X
        ]

        return ThermodynamicState(p=p, T=T, h=h, v=v, rho=rho, z=z, y=y, s=s, X=X)

    @classmethod
    def initialize(
        cls,
        num_comp: int = 1,
        num_phases: int = 1,
        num_vals: int = 1,
        as_ad: bool = False,
        is_independent: Optional[
            list[Literal["p", "T", "z_r", "z_i", "s_r", "s_i", "y_r"]]
        ] = None,
        values_from: Optional[ThermodynamicState] = None,
    ) -> ThermodynamicState:
        """Initializes a thermodynamic state with zero values, based on given
        configurations.

        If the AD format with derivatives is requested, the order of derivatives is
        as follows:

        1. (optional) pressure
        2. (optional) temperature
        3. (optional) feed fractions as ordered in :attr:`z`
        4. (optional) phase saturations as ordered in :attr:`vf`
        5. phase fractions as ordered in :attr:`y`
        6. phase compositions as ordered in :attr:`X`

           .. math::

               (x_{00},\\dots,x_{0, num_comp},\\dots, x_{num_phases, num_comp})

        Note:
            The default arguments are such that a derivative-free state for a p-T flash
            (fixed pressure, temperature, feed) with eliminated reference phase fraction
            is created.

        Parameters:
            num_comp: ``default=1``

                Number of components. Must be at least 1.
            num_phases: ``default=1``

                Number of phases. Must be at least 1.

                Important:
                    For the case of 1 phase, the phase fraction will never have a
                    derivative.

            num_vals: ``default=1``

                Number of values per state function. States can be vectorized using
                numpy.
            as_ad: ``default=False``

                If True, the values are initialized as
                :class:`~porepy.numerics.ad.forward_mode.AdArray` instances, with
                proper derivatives in csr format.

                If False, the values are initialized as numpy arrays with length
                ``num_vals``.
            is_independent: ``default=None``

                Some additional states can be marked as independent, meaning they are
                considered as variables and have unity as its derivative
                (hence increasing the whole Jacobian, if Ad arrays are requested)

                States which can be marked as independent include

                - ``'p'``: pressure
                - ``'T'``: temperature
                - ``'z_r'``: reference component (feed) fraction
                - ``'z_i'``: feed fractions of other components
                - ``'s_r'``: reference phase saturation
                - ``'s_i'``: saturations of other phases
                - ``'y_r'``: reference phase fraction

                Phase compositions :attr:`X` and other phase fractions are **always**
                considered independent.
            values_from: ``default=None``

                If another state structure is passed, copy the values.
                Assumes the other state structure has values **only**
                (no derivatives in form of AD-arrays).

        Raises:
            ValueError: If an unsupported state is requested in ``is_independent``.
            AssertionError: If ``num_comp,num_vals < 1`` or ``num_phases<2``.

        Returns:
            A state data structure with above configurations

        """

        assert num_phases >= 1, "Number of phases must be at least 1."
        assert num_comp >= 1, "Number of components must be at least 1."
        assert num_vals >= 1, "Number of values per state must be at least 1."

        indp = num_phases - 1  # number of independent phases

        if values_from:
            p = values_from.p
            T = values_from.T
            h = values_from.h
            v = values_from.v
            rho = values_from.rho
            z = [values_from.z[i] for i in range(num_comp)]
            y = [values_from.y[j] for j in range(num_phases)]
            s = [values_from.s[j] for j in range(num_phases)]
            X = [
                [values_from.X[j][i] for i in range(num_comp)]
                for j in range(num_phases)
            ]
        else:
            vec = np.zeros(num_vals)  # default zero values
            # default state
            p = vec.copy()
            T = vec.copy()
            h = vec.copy()
            v = vec.copy()
            rho = vec.copy()
            z = [vec.copy() for _ in range(num_comp)]
            y = [vec.copy() for _ in range(num_phases)]
            s = [vec.copy() for _ in range(num_phases)]
            X = [[vec.copy() for _ in range(num_comp)] for _ in range(num_phases)]

        # update state with derivatives if requested
        if as_ad:
            # identity derivative per independent state
            id_block = sps.identity(num_vals, dtype=float, format="lil")
            # determining the number of column blocks per independent state
            # defaults to phase compositions and independent phase fractions
            N_default = num_comp * num_phases + indp

            # The default global matrices are always created
            jac_glob_d = sps.lil_matrix((num_vals, N_default * num_vals))
            y_jacs: list[sps.lil_matrix] = list()
            X_jacs: list[list[sps.lil_matrix]] = list()
            # number of columns belonging to independent phases
            n_p = (indp) * num_vals
            # dependent phase composition
            X_jacs.append(list())
            for i in range(num_comp):
                jac_x_0i = jac_glob_d.copy()
                jac_x_0i[:, n_p + i * num_vals : n_p + (i + 1) * num_vals] = id_block
                X_jacs[-1].append(jac_x_0i)
            # update the column number based on reference phase composition vals
            n_p += num_comp * num_vals
            for j in range(indp):
                jac_y_j = jac_glob_d.copy()
                jac_y_j[:, j * num_vals : (j + 1) * num_vals] = id_block
                y_jacs.append(jac_y_j)
                X_jacs.append(list())
                for i in range(num_comp):
                    jac_x_ji = jac_glob_d.copy()
                    jac_x_ji[:, n_p + i * num_vals : n_p + (i + 1) * num_vals] = (
                        id_block
                    )
                    X_jacs[-1].append(jac_x_ji)

            # reference phase fraction is dependent by unity
            if len(y_jacs) > 0:
                y = [pp.ad.AdArray(y[0], -1 * safe_sum(y_jacs))] + [
                    pp.ad.AdArray(y[j + 1], y_jacs[j]) for j in range(indp)
                ]
            X = [
                [pp.ad.AdArray(X[j][i], X_jacs[j][i]) for i in range(num_comp)]
                for j in range(num_phases)
            ]

            if is_independent:
                # Number of blocks with new independent vars
                N = N_default
                # make unique
                is_independent = list(set(is_independent))
                for i in is_independent:
                    # adding blocks per independent feed fraction if requested
                    if i == "z_i":
                        N += num_comp - 1
                    # Adding blocks per independent phase saturation
                    elif i == "s_i":
                        N += num_phases - 1
                    # adding other blocks per independent state
                    elif i in ["p", "T", "y_r", "z_r", "s_r"]:
                        N += 1
                    else:
                        raise ValueError(f"Independent state {i} not supported.")

                # number of column blocks to pre-append to existing states
                N_new = N - N_default
                pre_block = sps.lil_matrix((num_vals, num_vals * N_new))

                # update derivatives of independent phases
                for j in range(1, num_phases):
                    y[j].jac = sps.hstack([pre_block, y[j].jac])
                # update derivative of reference phase
                if indp > 0:
                    y[0].jac = sps.hstack([pre_block, y[0].jac])
                else:
                    y[0] = pp.ad.AdArray(y[0], pre_block.copy())

                # update derivatives of phase compositions
                for j in range(num_phases):
                    for i in range(num_comp):
                        X[j][i].jac = sps.hstack([pre_block, X[j][i].jac])

                # Global Jacobian for new, independent states
                jac_glob = sps.lil_matrix((num_vals, N * num_vals))
                # occupied column indices, counted from right to left
                occupied = (num_comp * num_phases + indp) * num_vals

                # update derivative of reference phase fraction if requested
                if "y_r" in is_independent and not indp:
                    jac_y_r = jac_glob.copy()
                    jac_y_r[:, -(occupied + num_vals) : -occupied] = id_block
                    y[0] = pp.ad.AdArray(y[0], jac_y_r)
                    occupied += num_vals  # update occupied

                # construct derivatives w.r.t to saturations of independent phases
                jac_s_0_dep = None
                if "s_i" in is_independent and indp:
                    jac_s_0_dep = jac_glob.copy()
                    for j in range(indp):
                        jac_s_i = jac_glob.copy()
                        jac_s_i[:, -(occupied + num_vals) : -occupied] = id_block
                        jac_s_0_dep = jac_s_0_dep - jac_s_i
                        s[indp - j] = pp.ad.AdArray(s[indp - j], jac_s_i)
                        occupied += num_vals  # update occupied
                if "s_r" in is_independent and not indp:
                    jac_s_0 = jac_glob.copy()
                    jac_s_0[:, -(occupied + num_vals) : -occupied] = id_block
                    s[0] = pp.ad.AdArray(s[0], jac_s_i)
                    occupied += num_vals  # update occupied
                # eliminate reference saturation by unity
                elif jac_s_0_dep is not None:
                    s[0] = pp.ad.AdArray(s[0], jac_s_0_dep)

                # construct derivatives w.r.t. feed fractions
                if "z_i" in is_independent:
                    for i in range(num_comp - 1):
                        jac_z_i = jac_glob.copy()
                        jac_z_i[:, -(occupied + num_vals) : -occupied] = id_block
                        z[num_comp - 1 - i] = pp.ad.AdArray(
                            z[num_comp - 1 - i], jac_z_i
                        )
                        occupied += num_vals

                # construct derivative w.r.t. reference feed fraction
                if "z_r" in is_independent:
                    jac_z_r = jac_glob.copy()
                    jac_z_r[:, -(occupied + num_vals) : -occupied] = id_block
                    z[0] = pp.ad.AdArray(z[0], jac_z_r)
                    occupied += num_vals

                # construct derivatives for states which are not given as list
                # in reverse order, right to left
                modified_quantities: list[NumericType] = list()
                for key, quantity in zip(["T", "p"], [T, p]):
                    # modify quantity if requested
                    if key in is_independent:
                        jac_q = jac_glob.copy()
                        jac_q[:, -(occupied + num_vals) : -occupied] = id_block
                        quantity = pp.ad.AdArray(quantity, jac_q)
                        occupied += num_vals
                    modified_quantities.append(quantity)
                T, p = modified_quantities

        return cls(p=p, T=T, h=h, v=v, rho=rho, z=z, y=y, s=s, X=X)


@dataclass(frozen=True)
class PhaseProperties:
    """Basic data class for general phase properties, relevant for this framework.

    Use this dataclass to extend the list of relevant phase properties for a specific
    equation of state.

    """

    rho: NumericType
    """Molar density ``[mol / m^3]``."""

    rho_mass: NumericType
    """Mass density ``[kg / m^3]``."""

    v: NumericType
    """Molar volume ``[m^3 / mol]``."""

    h_ideal: NumericType
    """Specific ideal enthalpy ``[J / mol / K]``, which is a sum of ideal enthalpies of
    components weighed with their fraction. """

    h_dep: NumericType
    """Specific departure enthalpy ``[J / mol / K]``."""

    h: NumericType
    """Specific enthalpy ``[J / mol / K]``, a sum of :attr:`h_ideal` and :attr:`h_dep`.
    """

    phis: list[NumericType]
    """Fugacity coefficients per component, ordered as compositional fractions."""

    kappa: NumericType
    """Thermal conductivity ``[W / m / K]``."""

    mu: NumericType
    """Dynamic molar viscosity ``[mol / m / s]``."""


class VanDerWaals:
    """A  class providing functions representing mixing rules according to
    Van der Waals.

    This class is purely a container class to provide a namespace.

    """

    @staticmethod
    def cohesion(
        X: list[NumericType],
        a: list[NumericType],
        dT_a: list[NumericType],
        bip: list[list[NumericType]],
        dT_bip: list[list[NumericType]],
    ) -> tuple[NumericType, NumericType]:
        """
        Note:
            The reason why the cohesion and its temperature-derivative are returned
            together is because of efficiency and the similarity of the code.

        Parameters:
            X: A list of fractions.
            a: A list of component cohesion values, with the same length and order as
                ``X``.
            dT_a: A list of temperature-derivatives of component cohesion values,
                with the same length as ``X``.
            bip: A nested list or matrix-like structure, such that ``bip[i][j]`` is the
                binary interaction parameter between components ``i`` and ``j``,
                where the indices run over the enumeration of ``X`` and ``a``.

                The matrix-like structure is expected to be symmetric.
            dT_bip: Same as ``bip``, holding only the temperature-derivative of the
                binary interaction parameters.

        Returns:
            The mixture cohesion and its temperature-derivative,
            according to Van der Waals.

        """
        nc = len(X)  # number of components

        a_parts: list[NumericType] = []
        dT_a_parts: list[NumericType] = []

        # mixture matrix is symmetric, sum over all entries in upper triangle
        # multiply off-diagonal elements with 2
        for i in range(nc):
            a_parts.append(X[i] ** 2 * a[i])
            dT_a_parts.append(X[i] ** 2 * dT_a[i])
            for j in range(i + 1, nc):
                x_ij = X[i] * X[j]
                a_ij_ = _sqrt(a[i] * a[j])
                delta_ij = 1 - bip[i][j]

                a_ij = a_ij_ * delta_ij
                dT_a_ij = (
                    _power(a[i] * a[j], -1 / 2)
                    / 2
                    * (dT_a[i] * a[j] + a[i] * dT_a[j])
                    * delta_ij
                    - a_ij_ * dT_bip[i][j]
                )

                # off-diagonal elements appear always twice due to symmetry
                a_parts.append(2.0 * x_ij * a_ij)
                dT_a_parts.append(2.0 * x_ij * dT_a_ij)

        return safe_sum(a_parts), safe_sum(dT_a_parts)

    @staticmethod
    def dXi_cohesion(
        X: list[NumericType], a: list[NumericType], bip: list[list[NumericType]], i: int
    ) -> NumericType:
        """
        Parameters:
            X: A list of fractions.
            a: A list of component cohesion values, with the same length and order as
                ``X``.
            bip: A nested list or matrix-like structure, such that ``bip[i][j]`` is the
                binary interaction parameter between components ``i`` and ``j``,
                where the indices run over the enumeration of ``X`` and ``a``.

                The matrix-like structure is expected to be symmetric.
            i: An index for ``X``.

        Returns:
            The derivative of the mixture cohesion w.r.t to ``X[i]``
        """
        return 2.0 * safe_sum(
            [X[j] * _sqrt(a[i] * a[j]) * (1 - bip[i][j]) for j in range(len(X))]
        )

    @staticmethod
    def covolume(X: list[NumericType], b: list[NumericType]) -> NumericType:
        """
        Parameters:
            X: A list of fractions.
            b: A list of component covolume values, with the same length and order as
                ``X``.

        Returns:
            The mixture covolume according to Van der Waals.

        """
        return safe_sum([x_i * b_i for x_i, b_i in zip(X, b)])


class AbstractEoS(abc.ABC):
    """Abstract class representing an equation of state.

    Child classes have to implement the method :meth:`compute` which must return
    relevant :class:`PhaseProperties` using a specific EoS.

    The purpose of this class is the abstraction of the property computations, as well
    as providing the necessary information about the supercritical state and the
    extended state in the unified setting.

    Parameters:
        gaslike: A bool indicating if the EoS is represents a gas-like state.

            Since in general there can only be one gas-phase, this flag must be
            provided.
        *args: In case of inheritance.
        **kwargs: In case of inheritance.

    """

    def __init__(self, gaslike: bool, *args, **kwargs) -> None:
        super().__init__()

        self._components: list[Component] = list()
        """Private container for components with species data. See :meth:`components`.
        """

        self.is_supercritical: np.ndarray = np.array([], dtype=bool)
        """A boolean array flagging if the mixture became super-critical.

        In vectorized computations, the results are stored component-wise.

        Important:
            It is unclear, what the meaning of super-critical phases is using this EoS
            and values in this phase region should be used with suspicion.

        """

        self.is_extended: np.ndarray = np.array([], dtype=bool)
        """A boolean array flagging if an extended state is computed in the unified
        setting.

        In vectorized computations, the results are stored component-wise.

        Important:
            Extended states are un-physical in general.

        """

        self.gaslike: bool = bool(gaslike)
        """Flag passed at instantiation indicating if gas state or not."""

    @property
    def components(self) -> list[Component]:
        """A list of (compatible) components, which hold relevant chemical data.

        A setter is provided, which concrete EoS can overwrite to perform one-time
        computations, if any.

        Important:
            This attribute is set by a the setter of :meth:`Phase.components`,
            and should not be meddled with.

            The order in this list is crucial for computations involving fractions.

            Order of passed fractions must coincide with the order of components
            passed here.

            **This is a design choice**.
            Alternatively, component-related parameters and
            functions can be passed during instantiation, which would render the
            signature of the constructor quite hideous.

        Parameters:
            components: A list of component for the EoS containing species data.

        """
        return self._components

    @components.setter
    def components(self, components: list[Component]) -> None:
        # deep copy
        self._components = [c for c in components]

    @abc.abstractmethod
    def compute(
        self,
        p: NumericType,
        T: NumericType,
        X: list[NumericType],
        **kwargs,
    ) -> PhaseProperties:
        """Abstract method for computing all thermodynamic properties based on the
        passed, thermodynamic state.

        Warning:
            ``p``, ``T``, ``*X`` have a union type, meaning the results will be of
            the same. When mixing numpy arrays, porepy's Ad arrays and numbers,
            the user must make sure there will be no compatibility issues.

            This method is not supposed to be used with AD Operator instances.

        Important:
            This method must update :attr:`is_supercritical` and :attr:`is_extended`.

        Parameters:
            p: Pressure
            T: Temperature
            X: ``len=num_components``

                (Normalized) Fractions to be used in the computation.
            **kwargs: Any options necessary for specific computations can be passed as
                keyword arguments.

        Returns:
            A dataclass containing the basic phase properties. The basic properties
            include those, which are required for the reminder of the framework to
            function as intended.

        """
        raise NotImplementedError("Call to abstract method.")

    def get_h_ideal(
        self, p: NumericType, T: NumericType, X: list[NumericType]
    ) -> NumericType:
        """
        Parameters:
            p: Pressure.
            T: Temperature.
            X: ``len=num_components``

                (Normalized) Fraction per component to be used in the computation,
                ordered as in :attr:`components`.

        Returns:
            The ideal part of the enthalpy, which is a sum of ideal component enthalpies
            weighed with their fraction.

        """
        return safe_sum([x * comp.h_ideal(p, T) for x, comp in zip(X, self.components)])

    def get_rho_mass(self, rho_mol: NumericType, X: list[NumericType]) -> NumericType:
        """
        Parameters:
            rho_mol: Molar density resulting from :meth:`compute`.
            X: ``len=num_components``

                (Normalized) Fraction per component to be used in the computation,
                ordered as in :attr:`components`.

        Returns:
            The mass density, which is the molar density multiplied with the sum of
            fractions weighed with component molar masses.

        """
        return rho_mol * safe_sum(
            [x * comp.molar_mass for x, comp in zip(X, self.components)]
        )


A_CRIT: float = (
    1
    / 512
    * (
        -59
        + 3 * np.cbrt(276231 - 192512 * np.sqrt(2))
        + 3 * np.cbrt(276231 + 192512 * np.sqrt(2))
    )
)
"""Critical, non-dimensional cohesion value in the Peng-Robinson EoS,
~ 0.457235529."""

B_CRIT: float = (
    1
    / 32
    * (-1 - 3 * np.cbrt(16 * np.sqrt(2) - 13) + 3 * np.cbrt(16 * np.sqrt(2) + 13))
)
"""Critical, non-dimensional covolume in the Peng-Robinson EoS, ~ 0.077796073."""

Z_CRIT: float = (
    1 / 32 * (11 + np.cbrt(16 * np.sqrt(2) - 13) - np.cbrt(16 * np.sqrt(2) + 13))
)
"""Critical compressibility factor in the Peng-Robinson EoS, ~ 0.307401308."""


def coef0(A: Any, B: Any) -> Any:
    r"""Coefficient for the zeroth monomial of the characteristic equation

    :math:`Z^3 + c_2 Z^2 + c_1 Z + c_0 = 0`.

    For any input type supporting Python's overload of ``+,-,*,**``.

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.

    Returns:
        The result of :math:`B^3 + B^2 - AB`.

    """
    return B**3 + B**2 - A * B


def coef1(A: Any, B: Any) -> Any:
    r"""Coefficient for the first monomial of the characteristic equation

    :math:`Z^3 + c_2 Z^2 + c_1 Z + c_0 = 0`.

    For any input type supporting Python's overload of ``+,-,*,**``.

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.

    Returns:
        The result of :math:`A - 2 B - 3 B^2`.

    """
    return A - 2.0 * B - 3.0 * B**2


def coef2(A: Any, B: Any) -> Any:
    r"""Coefficient for the second monomial of the characteristic equation

    :math:`Z^3 + c_2 Z^2 + c_1 Z + c_0 = 0`.

    For any input type supporting Python's overload of ``-``.

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.

    Returns:
        The result of :math:`B - 1`.

    """
    return B - 1


def red_coef0(A: Any, B: Any) -> Any:
    r"""Zeroth coefficient of the reduced characteristic equation

    :math:`Z^3 + c_{r1} Z + c_{r0} = 0`.

    Uses :func:`coef0` - :func:`coef2` to compute the expressions in terms of
    ``A`` and ``B``

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.

    Returns:
        The result of

        .. math::

            c_2^3(A, B)\frac{2}{27} - c_2(A, B) c_1(A, B)\frac{1}{3} + c_0(A, B)

    """
    c2 = coef2(A, B)
    return c2**3 * (2.0 / 27.0) - c2 * coef1(A, B) * (1.0 / 3.0) + coef0(A, B)


def red_coef1(A: Any, B: Any) -> Any:
    r"""First coefficient of the reduced characteristic equation

    :math:`Z^3 + c_{r1} Z + c_{r0} = 0`.

    Uses :func:`coef0` - :func:`coef2` to compute the expressions in terms of
    ``A`` and ``B``

    Parameters:
        A: Non-dimensional cohesion.
        B: Non-dimensional covolume.

    Returns:
        The result of

        .. math::

            c_1(A, B) - c_2^2(A, B)\frac{1}{3}

    """
    return coef1(A, B) - coef2(A, B) ** 2 * (1.0 / 3.0)


def discr(rc0: Any, rc1: Any) -> Any:
    """Discriminant of the characeteristic polynomial based on the reduced coefficient.

    Parameters:
        rc0: Zeroth reduced coefficient (see :func:`red_coef0`)
        rc1: First reduced coefficient (see :func:`red_coef1`)

    Returns:
        The result of

        .. math::

            c_{r0}^2\\frac{1}{4} - c_{r1}^3\\frac{1}{27}

    """
    return rc0**2 * (1.0 / 4.0) + rc1**3 * (1.0 / 27.0)


def critical_line(A: Any) -> Any:
    r"""Returns the critical line parametrized as ``B(A)``.

    .. math::

        \frac{B_{crit}}{A_{crit}} A

    """
    return (B_CRIT / A_CRIT) * A


def widom_line(A: Any) -> Any:
    r"""Returns the Widom-line parametrized as ``B(A)`` in the A-B space:

    .. math::

        B_{crit} + 0.8 \cdot 0.3381965009398633 \cdot \left(A - A_{crit}\right)

    """
    return B_CRIT + 0.8 * 0.3381965009398633 * (A - A_CRIT)


B_CRIT_LINE_POINTS: tuple[np.ndarray, np.ndarray] = (
    np.array([0.0, B_CRIT], dtype=np.float64),
    np.array([A_CRIT, B_CRIT], dtype=np.float64),
)
r"""Two 2D points characterizing the line ``B=B_CRIT`` in the A-B space, namely

.. math::

    (0, B_{crit}),~(A_{crit},B_{crit})

See :data:`B_CRIT`, data:`A_CRIT`.

"""


S_CRIT_LINE_POINTS: tuple[np.ndarray, np.ndarray] = (
    np.zeros(2, dtype=np.float64),
    np.array([A_CRIT, B_CRIT], dtype=np.float64),
)
r"""Two 2D points characterizing the super-critical line in the A-B space, namely

.. math::

    (0,0),~(A_{crit},B_{crit})

See :data:`B_CRIT`, data:`A_CRIT`.

"""


W_LINE_POINTS: tuple[np.ndarray, np.ndarray] = (
    np.array([0.0, widom_line(0)], dtype=np.float64),
    np.array([A_CRIT, widom_line(A_CRIT)], dtype=np.float64),
)
r"""Two 2D points characterizing the Widom-line for water.

The points are created by using :func:`widom_line` for :math:`A\in\{0, A_{crit}\}`.

See :data:`~porepy.compositional.peng_robinson.eos.A_CRIT`.

"""


def _point_to_line_distance(point: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Auxiliary function to compute the normal distance between a point and a line
    represented by two points (rows in a matrix ``line``)."""

    d = np.sqrt((line[1, 0] - line[0, 0]) ** 2 + (line[1, 1] - line[0, 1]) ** 2)
    n = np.abs(
        (line[1, 0] - line[0, 0]) * (line[0, 1] - point[1])
        - (line[0, 0] - point[0]) * (line[1, 1] - line[0, 1])
    )
    return n / d


def point_to_line_distance(p: np.ndarray, lp1: np.ndarray, lp2: np.ndarray) -> float:
    """Computes the distance between a 2-D point and a line spanned by two points.

    NJIT-ed function with signature ``(float64[:], float64[:], float64[:]) -> float64``.

    Parameters:
        p: ``shape=(2,n)``

            Point(s) in 2D space.
        lp1: ``shape=(2,)``

            First point spanning the line.
        lp2: ``shape=(2,)``

            Second point spanning the line.

    Returns:
        Normal distance between ``p`` and the spanned line.

    """

    d = np.sqrt((lp2[0] - lp1[0]) ** 2 + (lp2[1] - lp1[1]) ** 2)
    n = np.abs(
        (lp2[0] - lp1[0]) * (lp1[1] - p[1]) - (lp1[0] - p[0]) * (lp2[1] - lp1[1])
    )
    return n / d


def root_smoother(
    Z_L: pp.ad.AdArray, Z_I: pp.ad.AdArray, Z_G: pp.ad.AdArray, s: float
) -> tuple[pp.ad.AdArray, pp.ad.AdArray]:
    """Smoothing procedure on boundaries of three-root-region.

    Smooths the value and Jacobian rows of the liquid and gas root close to
    phase boundaries.

    See Also:
        `Vu et al. (2021), Section 6.
        <https://doi.org/10.1016/j.matcom.2021.07.015>`_

    Parameters:
        Z_L: Liquid-like root.
        Z_I: Intermediate root.
        Z_G: Gas-like root.
        s: smoothing factor.

    Returns:
        A tuple containing the smoothed liquid and gas root as AD arrays

    """
    # proximity:
    # If close to 1, intermediate root is close to gas root.
    # If close to 0, intermediate root is close to liquid root.
    # values bound by [0,1]
    proximity = (Z_I - Z_L) / (Z_G - Z_L)
    if isinstance(proximity, pp.ad.AdArray):
        proximity = proximity.val

    # average intermediate and gas root for gas root smoothing
    W_G = (Z_I + Z_G) / 2
    # analogously for liquid root
    W_L = (Z_I + Z_L) / 2

    v_G = _gas_smoother(proximity, s)
    v_L = _liquid_smoother(proximity, s)

    return (
        Z_L * (1 - v_L) + W_L * v_L,  # weights are arrays, Z possibly AD -> * right
        Z_G * (1 - v_G) + W_G * v_G,
    )


def _gas_smoother(proximity: np.ndarray, s: float) -> np.ndarray:
    """Smoothing function for three-root-region where the intermediate root comes
    close to the gas root.

    Parameters:
        proximity: An array containing the proximity between the intermediate
            root and the liquid root, relative to the difference in liquid and
            gas root
        s: smoothing factor

    Returns:
        The smoothing weight for the gas root.

    """
    # smoother starts with zero values
    smoother = np.zeros(proximity.shape[0])
    # values around smoothing parameter are constructed according to Vu
    upper_bound = proximity < 1 - s
    lower_bound = (1 - 2 * s) < proximity
    bound = upper_bound & lower_bound

    bound_smoother = (proximity[bound] - (1 - 2 * s)) / s
    bound_smoother = bound_smoother**2 * (3 - 2 * bound_smoother)

    smoother[bound] = bound_smoother
    # where proximity is close to one, set value of one
    smoother[proximity >= 1 - s] = 1.0

    return smoother


def _liquid_smoother(proximity: np.ndarray, s: float) -> np.ndarray:
    """Smoothing function for three-root-region where the intermediate root comes
    close to the liquid root.

    Parameters:
        proximity: An array containing the proximity between the intermediate
            root and the liquid root, relative to the difference in liquid and
            gas root
        s: smoothing factor

    Returns:
        The smoothing weight for the liquid root

    """
    # smoother starts with zero values
    smoother = np.zeros(proximity.shape[0])
    # values around smoothing parameter are constructed according to Vu
    upper_bound = proximity < 2 * s
    lower_bound = s < proximity
    bound = upper_bound & lower_bound

    bound_smoother = (proximity[bound] - s) / s
    bound_smoother = (-1) * bound_smoother**2 * (3 - 2 * bound_smoother) + 1

    smoother[bound] = bound_smoother
    # where proximity is close to zero, set value of one
    smoother[proximity <= s] = 1.0

    return smoother


@dataclass(frozen=True)
class PhaseProperties_cubic(PhaseProperties):
    """Extended phase properties resulting from computations using a cubic EoS.

    The properties here are not necessarily required by the general framework.

    """

    a: NumericType
    """Cohesion term ``a`` in the cubic EoS."""

    dT_a: NumericType
    """The derivative of the cohesion w.r.t. the temperature."""

    b: NumericType
    """Covolume term ``b`` in the cubic EoS."""

    A: NumericType
    """Non-dimensional cohesion (see :attr:`a`)."""

    B: NumericType
    """Non-dimensional covolume (see :attr:`b`)"""

    Z: NumericType
    """Compressibility factor."""


class PengRobinson(AbstractEoS):
    """A class implementing thermodynamic properties resulting from the Peng-Robinson
    equation of state

        ``p = R * T / (v - b) - a / (b**2 + 2 * v * b - b**2)``.

    Note:
        1. The various methods providing thermodynamic quantities are AD-compatible.
           They can be wrapped into AD-Functions and are able to take Ad-Arrays as
           input.
        2. Calling any thermodynamic property with molar fractions ``X`` as arguments
           raises errors if the number of passed fractions does not match the
           number of modelled components in :attr:`components`.

           The order of fractions arguments must correspond to the order in
           :attr:`components`.
        3. As of now, supercritical phases are not really supported, just indicated.
           Be aware of the limitations of the Peng-Robinson model!

    For a list of computed properties, see dataclass :class:`PhaseProperties_cubic`.

    References:
        [1]: `Peng, Robinson (1976) <https://doi.org/10.1021/i160057a011>`_
        [2]: `Ben Gharbia et al. (2021) <https://doi.org/10.1051/m2an/2021075>`_
        [3]: `Vu et al. (2021) <https://doi.org/10.1016/j.matcom.2021.07.015>`_
        [4]: `Zhu, Okuno (2014) <http://dx.doi.org/10.1016/j.fluid.2014.07.003>`_
        [5]: `Zhu, Okuno (2015) <https://onepetro.org/spersc/proceedings/15RSS/
             1-15RSS/D011S001R002/183434>`_
        [6]: `Connolly et al. (2021) <https://doi.org/10.1016/j.ces.2020.116150>`_

    Parameters:
        gaslike: A bool indicating if if the gas-like root of the cubic polynomial
            should be used for computations.

            If False, the liquid-like root is used.
        mixingrule: ``default='VdW'``

            Name of the mixing rule to be applied.
        smoothing_factor: ``default=1e-2``

            A small number to determine proximity between 3-root and double-root case.

            See also `Vu et al. (2021), Section 6. <https://doi.org/10.1016/
            j.matcom.2021.07.015>`_ .
        eps: ``default=1e-14``

            A small number defining the numerical zero.

            Used for the computation of roots to distinguish between phase-regions.
        *args: Placeholder in case of inheritance.
        **kwargs: Placeholder in case of inheritance.

    """

    A_CRIT: float = (
        1
        / 512
        * (
            -59
            + 3 * np.cbrt(276231 - 192512 * np.sqrt(2))
            + 3 * np.cbrt(276231 + 192512 * np.sqrt(2))
        )
    )
    """Critical, non-dimensional cohesion value in the Peng-Robinson EoS,
    ~ 0.457235529."""

    B_CRIT: float = (
        1
        / 32
        * (-1 - 3 * np.cbrt(16 * np.sqrt(2) - 13) + 3 * np.cbrt(16 * np.sqrt(2) + 13))
    )
    """Critical, non-dimensional covolume in the Peng-Robinson EoS, ~ 0.077796073."""

    Z_CRIT: float = (
        1 / 32 * (11 + np.cbrt(16 * np.sqrt(2) - 13) - np.cbrt(16 * np.sqrt(2) + 13))
    )
    """Critical compressibility factor in the Peng-Robinson EoS, ~ 0.307401308."""

    def __init__(
        self,
        gaslike: bool,
        *args,
        mixingrule: Literal["VdW"] = "VdW",
        smoothing_factor: float = 1e-2,
        eps: float = 1e-14,
        **kwargs,
    ) -> None:
        super().__init__(gaslike)

        self._a_crit_vals: tuple[float] = []
        """Critical cohesion values per component, using EoS specific critical value.

        Computed in setter for :meth:`components`.

        """

        self._b_vals: tuple[float] = []
        """Critical covolume values per component, using EoS-specific critical value.

        Computed in setter for :meth:`components`.

        """

        self._a_cor_vals: tuple[float] = []
        """Cohesion correction weights, appearing in the linearized alpha-correction of
        the cohesion.

        Computed in setter for :meth:`components`.

        """

        self._bip_vals: np.ndarray = np.array([])
        """A matrix  in strictly upper-triangle form containing the constant
        binary interaction parameters between components at indices ``[i][j]``.

        The values are loaded once during the setter for :meth:`components`.

        """

        self._custom_bips: dict[
            str, Callable[[NumericType], tuple[NumericType, NumericType]]
        ] = dict()
        """A map between indices for :attr:`_bip_vals` (upper triangle) and
        custom implementation of BIPs (temperature-dependent), if available.

        See :attr:`~porepy.compositional.peng_robinson.pr_components.Component_PR.bip_map`.

        """

        self._widom_points: np.ndarray = np.array(
            [[0, self.Widom_line(0)], [self.A_CRIT, self.Widom_line(self.A_CRIT)]]
        )
        """The Widom line for water characterized by two points at ``A=0`` and
        ``A=A_criet``"""

        self._B_crit_points: np.ndarray = np.array(
            [[0, self.B_CRIT], [self.A_CRIT, self.B_CRIT]]
        )
        """Two points (rows) characterizing the line ``B=B_crit``."""

        self._critline_points: np.ndarray = np.array(
            [[0, 0], [self.A_CRIT, self.B_CRIT]]
        )
        """The critical line characterized by two points ``(0, 0)`` and
        ``(A_crit, B_crit)``"""

        self.gaslike: bool = bool(gaslike)
        """Flag passed at instantiation denoting the state of matter, liquid or gas."""

        assert mixingrule in ["VdW"], f"Unknown mixing rule {mixingrule}."
        self.mixingrule: str = mixingrule
        """The mixing rule passed at instantiation."""

        self.eps: float = eps
        """Passed at instantiation."""

        self.smooth_3: float = smoothing_factor
        """Passed at instantiation."""
        self.smooth_e = 1e-4

        self.regions: list[np.ndarray] = [np.zeros(1, dtype=bool)] * 4
        """A list of root-region indicates.

        Contains boolean arrays indicating where which root case is registered
        during the last computation.

        - 0: Array containing True where a triple-root was computed.
        - 1: Array containing True where a single real root was computed.
        - 2: Array indicating where 2 distinct real roots were computed.
        - 3: Array indicating where 3 distinct real roots were computed.

        """

    def _num_frac_check(self, X) -> None:
        """Auxiliary method to check the number of passed fractions.
        Raises an error if number does not match the number of present components."""
        if len(X) != len(self.components):
            raise ValueError(
                f"{len(X)} fractions given but "
                + f"{len(self.components)} components present."
            )

    @property
    def components(self) -> list:
        """The child class setter calculates EoS specific values per set component.

        Values like critical cohesion and covolume values, corrective parameters and
        binary interaction parameters are obtained only once and stored during in the
        setter method.

        The setter additionally checks for the availability of BIPs and throws
        respective warnings. Unavailable BIPs are set to zero.

        Todo:
            Should it be a warning or an error?

        Warning:
            If two components ``i`` and ``j``, with ``i < j``, have both BIPs
            BIPs implemented for each other in
            :attr:`~porepy.compositional.peng_robinson.pr_components.Component_PR.bip_map`,
            the first implementation of ``i`` is taken and a warning is emitted.

        Parameters:
            components: A list of Peng-Robinson compatible components.

        Raises:
            RuntimeError: If two different components implement models for BIP for each
                other (it is impossible to decide which one to use).

        """
        return AbstractEoS.components.fget(self)

    @components.setter
    def components(self, components: list) -> None:

        a_crits: list[float] = list()
        bs: list[float] = list()
        a_cors: list[float] = list()

        nc = len(components)

        # computing constant parameters
        for comp in components:
            a_crits.append(self._a_crit(comp.p_crit, comp.T_crit))
            bs.append(self._b_crit(comp.p_crit, comp.T_crit))
            a_cors.append(self._a_cor(comp.omega))

        self._a_cor_vals = tuple(a_cors)
        self._b_vals = tuple(bs)
        self._a_crit_vals = tuple(a_crits)

        # prepare storage for constant bips
        self._bip_vals = np.zeros((nc, nc))

        # storing missing bips and custom bip callables
        bip_callables: dict[
            str, Callable[[NumericType], tuple[NumericType, NumericType]]
        ] = dict()

        for i in range(nc):
            comp_i = components[i]
            # storage for custom BIPs with other component, if custom available.
            customs_for_i = []
            # check if component i has custom BIPs implemented
            if hasattr(comp_i, "bip_map"):
                for j in range(0, nc):
                    comp_j = components[j]
                    bip_c = comp_i.bip_map.get(comp_j.CASr_number, None)
                    # if found, store and mark to check it with the other component
                    if bip_c is not None:
                        customs_for_i.append(j)
                        bip_callables.update({(i, j): bip_c})

            for j in range(i + 1, nc):  # strictly upper triangle matrix
                comp_j = components[j]

                # use custom models if implemented
                if hasattr(comp_j, "bip_map"):
                    bip_c = comp_j.bip_map.get(comp_i.CASr_number, None)
                    # check if component i has already implemented a bip
                    # If not, check if j has it implemented, store and continue
                    # if it has, warn if double implementation detected and continue
                    if j in customs_for_i:
                        if bip_c is not None:
                            logger.warn(
                                "Detected double-implementation of BIP for components"
                                + f"{comp_i.name} and {comp_j.name}."
                                + "\nA fix to the model components if recommended."
                            )
                        continue
                    else:
                        # store callables, if implemented, and continue
                        if bip_c is not None:
                            assert (i, j) not in bip_callables.keys()
                            bip_callables.update({(i, j): bip_c})
                            continue
                        # at this point, the code will continue and try load a database
                        # bip.
                        # This happens if i and j have custom BIPs, but not for each
                        # other

                # try to load BIPs from database, if custom has not been found so far
                bip = load_bip(comp_i.CASr_number, comp_j.CASr_number)
                # TODO: This needs some more thought. How to handle missing BIPs?
                if bip == 0.0:
                    logger.info(
                        "Loaded a BIP with zero value for"
                        + f" components {comp_i.name} and {comp_j.name}."
                    )
                self._bip_vals[i, j] = bip

        # If custom implementations for bips were found, store them.
        # Otherwise store an empty dict.
        if bip_callables:
            self._custom_bips = bip_callables
        else:
            self._custom_bips = dict()

        AbstractEoS.components.fset(self, components)

    def compute(
        self,
        p: NumericType,
        T: NumericType,
        X: list[NumericType],
        apply_smoother: bool = False,
        Z_as_AD: bool = True,
        **kwargs,
    ) -> PhaseProperties_cubic:
        """Computes all thermodynamic properties based on the passed state.

        Warning:
            ``p``, ``T``, ``X`` have a union type, meaning the results will be of
            the same. When mixing numpy arrays, porepy's Ad arrays and numbers,
            the user must make sure there will be no compatibility issues.

            This method is not supposed to be used with AD Operator instances.

        Parameters:
            p: Pressure
            T: Temperature
            X: ``len=num_components``

                Fraction per component to be used in the computation,
                ordered as in :meth:`components`.
            apply_smoother: ``default=False``

                If True, a smoothing procedure is applied in the three-root-region,
                where the intermediate root approaches one of the other roots
                (see [3]).

                This is to be used **within** iterative procedures for numerical
                reasons. Once convergence is reached, the true roots should be computed
                without smoothing.
            **kwargs: Placeholder in case of inheritance.

        Raises:
            ValueError: If a mismatch between number passed fractions and modelled
                components is detected.

        Returns:
            A dataclass containing thermodynamic properties resulting from a cubic EoS.

        """
        # sanity check
        self._num_frac_check(X)
        # binary interaction parameters
        bip, dT_bip = self._compute_bips(T)
        # cohesion and covolume, and derivative
        a, dT_a, a_comps, _ = self._compute_cohesion_terms(T, X, bip, dT_bip)
        b = self._compute_mixture_covolume(X)
        # compute non-dimensional quantities
        A = self._A(a, p, T)
        B = self._B(b, p, T)
        # root
        Z, Z_other = self._Z(A, B, apply_smoother=apply_smoother, Z_as_AD=Z_as_AD)

        # volume and density
        v = self._v(p, T, Z)
        rho = v ** (-1)  # self._rho(p, T, Z)
        rho_mass = self.get_rho_mass(rho, X)
        # departure enthalpy
        dT_A = dT_a / (R_IDEAL_MOL * T) ** 2 * p - 2 * a / (R_IDEAL_MOL**2 * T**3) * p
        h_dep = self._h_dep(T, Z, A, dT_A, B)
        h_ideal = self.get_h_ideal(p, T, X)
        h = h_ideal + h_dep

        # Fugacity extensions as per Ben Gharbia 2021
        # dxi_a = list()
        # if np.any(self.is_extended):
        #     extend_phi: bool = True
        #     rho_ext = self._v(p, T, Z_other) **(-1)
        #     G = self._Z_polynom(Z, A, B)
        #     Gamma_ext = G * Z / ((B - Z) * (B**2 - Z**2 - 2 * Z * B))
        #     dxi_Z_other = list()
        #     for i in range(len(self.components)):
        #         dxi_a_ = self._dXi_a(X, a_comps, bip, i)
        #         dxi_a.append(dxi_a_)
        #         dxi_Z_other_ = self._dxi_Z(T, rho_ext, a, b, self._b_vals[i], dxi_a[i])
        #         dxi_Z_other.append(dxi_Z_other_)
        #     dZ_other = safe_sum([x * dz for x, dz in zip(X, dxi_Z_other)])
        # else:
        #     extend_phi: bool = False
        #     for i in range(len(self.components)):
        #         dxi_a_ = self._dXi_a(X, a_comps, bip, i)
        #         dxi_a.append(dxi_a_)
        # fugacity per present component
        phis: list[NumericType] = list()
        for i in range(len(self.components)):
            b_i = self._b_vals[i]
            B_i = self._B(b_i, p, T)
            A_i = self._A(self._dXi_a(X, a_comps, bip, i), p, T)
            # A_i = self._A(dxi_a[i], p, T)
            phi_i = self._phi_i(Z, A_i, A, B_i, B)

            # if extend_phi:
            #     w1 = (B - B_i + dZ_other - dxi_Z_other[i]) / Z / 2
            #     w2 = (B - B_i) / B
            #     ext_i = Gamma_ext * (w1 + w2)

            #     ext_i = ext_i[self.is_extended]
            #     phi_i[self.is_extended] = phi_i[self.is_extended] + ext_i

            phis.append(phi_i)

        # these two are open TODO
        kappa = self._kappa(p, T, Z)
        mu = self._mu(p, T, Z)

        return PhaseProperties_cubic(
            a=a,
            dT_a=dT_a,
            b=b,
            A=A,
            B=B,
            Z=Z,
            rho=rho,
            rho_mass=rho_mass,
            v=v,
            h_ideal=h_ideal,
            h_dep=h_dep,
            h=h,
            phis=phis,
            kappa=kappa,
            mu=mu,
        )

    def _compute_bips(
        self, T: NumericType
    ) -> tuple[list[list[NumericType]], list[list[NumericType]]]:
        """
        Parameters:
            T: Temperature.

        Returns:
            Two nested lists in upper-triangle form containing the binary interaction
            parameters. The first structure contains the parameters, the second
            their temperature-derivatives.

            The lists are such that indexing by ``[i][j]`` gives the interaction
            parameter between components ``i`` and ``j``.

            The lower triangle and the main diagonal of this matrix-like structure
            are filled with zeros, since the interaction parameters are symmetrical.

        """
        # construct first output from the constant bips and their trivial derivative
        bips: list[list[NumericType]] = (self._bip_vals + self._bip_vals.T).tolist()
        dT_bips: list[list[NumericType]] = np.zeros(self._bip_vals.shape).tolist()

        # if any custom bips are to be used, update the respective entries
        if self._custom_bips:
            for idx, bip_c in self._custom_bips.items():
                i, j = idx
                bip, dT_bip = bip_c(T)

                # values are symmetric
                bips[i][j] = bip
                bips[j][i] = bip
                dT_bips[i][j] = dT_bip
                dT_bips[j][i] = dT_bip

        return bips, dT_bips

    def _compute_component_cohesions(
        self, T: NumericType
    ) -> tuple[list[NumericType], list[NumericType]]:
        """
        Parameters:
            T: Temperature

        Returns:
            A 2-tuple containing

            1. temperature-dependent cohesion values per component.
            2. the temperature-derivative of the the cohesion per component.

        """
        a: list[NumericType] = []
        dT_a: list[NumericType] = []

        for i, comp in enumerate(self.components):
            a_cor_i = self._a_cor_vals[i]
            a_crit_i = self._a_crit_vals[i]
            T_r_i = T / comp.T_crit

            # check if special model for alpha was implemented.
            if hasattr(comp, "alpha"):
                alpha_i = comp.alpha(T)
            else:
                alpha_i = self._a_alpha(a_cor_i, T_r_i)

            a_i = a_crit_i * _power(alpha_i, 2)
            # outer derivative
            dT_a_i = 2 * a_crit_i * alpha_i
            # inner derivative
            dT_a_i *= (-a_cor_i / (2 * comp.T_crit)) * _power(T_r_i, -1 / 2)

            a.append(a_i)
            dT_a.append(dT_a_i)

        return a, dT_a

    def _compute_cohesion_terms(
        self,
        T: NumericType,
        X: list[NumericType],
        bip: list[list[NumericType]],
        dT_bip: list[list[NumericType]],
    ) -> tuple[NumericType, NumericType, NumericType, NumericType]:
        """
        Parameters:
            T: Temperature.
            X: ``len=num_components``

                Fraction per component to be used in the computation,
                ordered as in :attr:`components`.
            bip: A nested list or matrix-like structure, such that ``bip[i][j]`` is the
                binary interaction parameter between components ``i`` and ``j``,
                where the indices run over the enumeration of ``X`` and ``a``.

                The matrix-like structure can be an upper diagonal matrix, since the
                interaction parameters are symmetric.
            dT_bip: Same as ``bip``, holding only the temperature-derivative of the
                binary interaction parameters.

        Returns:
            A 4-tuple containing

            - The cohesion ``a`` of the mixture for given thermodynamic state,
              using the assigned mixing rule,
            - its temperature-derivative,
            - cohesion values per component,
            - their temperature, derivatives

        """
        a_c, dT_a_c = self._compute_component_cohesions(T)
        if self.mixingrule == "VdW":
            a_mix, dT_a_mix = VanDerWaals.cohesion(X, a_c, dT_a_c, bip, dT_bip)
            return a_mix, dT_a_mix, a_c, dT_a_c
        else:
            raise ValueError(f"Unknown mixing rule {self.mixingrule}.")

    def _compute_mixture_covolume(self, X: list[NumericType]) -> NumericType:
        """
        Parameters:
            X: ``len=num_components``

                Fraction per component to be used in the computation,
                ordered as in :attr:`components`.

        Returns:
            The covolume ``b`` for given thermodynamic state, using the assigned
            mixing rule.

        """
        if self.mixingrule == "VdW":
            return VanDerWaals.covolume(X, self._b_vals)
        else:
            raise ValueError(f"Unknown mixing rule {self.mixingrule}.")

    # formulae -------------------------------------------------------------------------

    @classmethod
    def _b_crit(cls, p_crit: float, T_crit: float) -> float:
        """
        .. math::

            a_{crit} = A_{crit} * \\frac{R^2 T_{crit}^2}{p_{crit}}

        Parameters:
            p_crit: Critical pressure of a component.
            T_crit: Critical temperature of a component.

        Returns:
            The component-specific critical covolume.

        """
        return cls.B_CRIT * (R_IDEAL_MOL * T_crit) / p_crit

    @classmethod
    def _a_crit(cls, p_crit: float, T_crit: float) -> float:
        """
        .. math::

            a_{crit} = A_{crit} * \\frac{R T_{crit}}{p_{crit}}

        Parameters:
            p_crit: Critical pressure of a component.
            T_crit: Critical temperature of a component.

        Returns:
            The component-specific critical cohesion.

        """
        return cls.A_CRIT * (R_IDEAL_MOL**2 * T_crit**2) / p_crit

    @staticmethod
    def _a_cor(omega: float) -> float:
        """
        References:
            `Zhu et al. (2014), Appendix A
            <https://doi.org/10.1016/j.fluid.2014.07.003>`_

        Parameters:
            omega: Acentric factor for a component.

        Returns:
            Returns the cohesion correction parameter depending on a component's
            acentric factor.

        """
        if omega < 0.491:
            return 0.37464 + 1.54226 * omega - 0.26992 * omega**2
        else:
            return (
                0.379642 + 1.48503 * omega - 0.164423 * omega**2 + 0.016666 * omega**3
            )

    @staticmethod
    def _a_alpha(a_cor: float, T_r: NumericType) -> NumericType:
        """
        Parameters:
            a_cor: Acentric factor-dependent weight in the linearized correction of
                the cohesion for a component.
            T_r: Reduced temperature for a component (divided by the component's
                critical temperature.).

        Returns:
            The root of the linearized correction for the cohesion term.

        """
        return 1 + a_cor * (1 - _sqrt(T_r))

    def _dXi_a(
        self,
        X: list[NumericType],
        a: list[NumericType],
        bip: list[list[NumericType]],
        i: int,
    ) -> NumericType:
        """Auxiliary method to compute parts of the fugacity coefficients."""
        if self.mixingrule == "VdW":
            return VanDerWaals.dXi_cohesion(X, a, bip, i)
        else:
            raise ValueError(f"Unknown mixing rule {self.mixingrule}.")

    @staticmethod
    def _A(a: NumericType, p: NumericType, T: NumericType) -> NumericType:
        """Auxiliary method implementing formula for non-dimensional cohesion."""
        if isinstance(T, pp.ad.AdArray):
            return T ** (-2) * a * p / R_IDEAL_MOL**2
        else:
            return a * p / (R_IDEAL_MOL**2 * T**2)

    @staticmethod
    def _B(b: NumericType, p: NumericType, T: NumericType) -> NumericType:
        """Auxiliary method implementing formula for non-dimensional covolume."""
        if isinstance(T, pp.ad.AdArray):
            return T ** (-1) * b * p / R_IDEAL_MOL
        else:
            return b * p / (R_IDEAL_MOL * T)

    # TODO
    def _kappa(self, p: NumericType, T: NumericType, Z: NumericType) -> NumericType:
        """
        Parameters:
            p: Pressure.
            T: Temperature.
            Z: Compressibility factor.

        Returns:
            The thermal conductivity.

        """
        return 1.0

    # TODO
    def _mu(self, p: NumericType, T: NumericType, Z: NumericType) -> NumericType:
        """
        Parameters:
            p: Pressure.
            T: Temperature.
            Z: Compressibility factor.

        Returns:
            The dynamic viscosity.

        """
        return 1.0

    @staticmethod
    def _rho(p: NumericType, T: NumericType, Z: NumericType) -> NumericType:
        """Auxiliary function implementing the formula for density."""
        return Z ** (-1) * p / (T * R_IDEAL_MOL)

    @staticmethod
    def _v(p: NumericType, T: NumericType, Z: NumericType) -> NumericType:
        """Auxiliary function implementing the formula for volume."""
        if isinstance(T, pp.ad.AdArray):
            return T * Z * R_IDEAL_MOL / p
        else:
            return Z * T / p * R_IDEAL_MOL

    @staticmethod
    def _log_ZB_0(Z: NumericType, B: NumericType) -> NumericType:
        return trunclog((B - Z) * (-1), 1e-6)

    @staticmethod
    def _log_ZB_1(Z: NumericType, B: NumericType) -> NumericType:
        return trunclog(((1 + np.sqrt(2)) * B + Z) / ((1 - np.sqrt(2)) * B + Z), 1e-6)

    def _h_dep(
        self,
        T: NumericType,
        Z: NumericType,
        A: NumericType,
        dT_A: NumericType,
        B: NumericType,
    ) -> NumericType:
        """Auxiliary function for computing the enthalpy departure function."""
        if isinstance(T, pp.ad.AdArray):
            i = T * (Z - 1) * R_IDEAL_MOL
        else:
            i = (Z - 1) * T * R_IDEAL_MOL
        return (
            1
            / np.sqrt(8)
            * (dT_A * T**2 * R_IDEAL_MOL + A * T * R_IDEAL_MOL)
            / B
            * self._log_ZB_1(Z, B)
            + i
        )

    def _g_dep(
        self, T: NumericType, A: NumericType, B: NumericType, Z: NumericType
    ) -> NumericType:
        """Auxiliary function for computing the Gibbs departure function."""
        return (
            (self._log_ZB_0(Z, B) - self._log_ZB_1(Z, B) * A / B / np.sqrt(8))
            * T
            * R_IDEAL_MOL
        )

    @staticmethod
    def _g_ideal(X: list[NumericType]) -> NumericType:
        """Auxiliary function to compute the ideal part of the Gibbs energy."""
        return safe_sum([x * trunclog(x, 1e-6) for x in X])

    def _phi_i(
        self,
        Z: NumericType,
        A_i: NumericType,
        A: NumericType,
        B_i: NumericType,
        B: NumericType,
    ) -> NumericType:
        """Auxiliary method implementing the formula for the fugacity coefficient."""
        log_phi_i = (
            B ** (-1) * (Z - 1) * B_i
            - self._log_ZB_0(Z, B)
            - A / (B * np.sqrt(8)) * (A_i / A - B ** (-1) * B_i) * self._log_ZB_1(Z, B)
        )
        return truncexp(log_phi_i)

    @staticmethod
    def _dxi_Z(
        T: NumericType,
        rho: NumericType,
        a: NumericType,
        b: NumericType,
        b_i: NumericType,
        dxi_a: NumericType,
    ) -> NumericType:
        """Auxiliary function implementing the derivative of the compressibility factor
        w.r.t. molar fraction ``x_i``."""
        d = 1 + 2 * b * rho - (b * rho) ** 2
        return (1 - b * rho) ** (-2) * rho * b_i + (
            a * rho * (2 * rho * b_i + 2 * b * rho**2 * b_i) / (d**2 * T * R_IDEAL_MOL)
            - dxi_a * rho / (d * T * R_IDEAL_MOL)
        )

    @staticmethod
    def _Z_polynom(Z: NumericType, A: NumericType, B: NumericType) -> NumericType:
        """Auxiliary method implementing the compressibility polynomial."""
        return (
            (B**3 + B**2 - A * B) + (B - 1) * Z**2 + (A - 2 * B - 3 * B**2) * Z + Z**3
        )

    @classmethod
    def Widom_line(cls, A: NumericType) -> NumericType:
        """Returns the Widom-line ``B(A)``"""
        return cls.B_CRIT + 0.8 * 0.3381965009398633 * (A - cls.A_CRIT)

    @classmethod
    def critical_line(cls, A: NumericType) -> NumericType:
        """Returns the critical line ``B_crit / A_crit * A``"""
        return cls.B_CRIT / cls.A_CRIT * A

    @staticmethod
    def extended_root_sub(B: NumericType, Z: NumericType) -> NumericType:
        """Auxiliary function implementing the formula for the extended, subcritical
        root proposed in Gharbia et al. (2021)

        Parameters:
            B: Dimensionless covolume.
            Z: The real root.

        """
        return (1 - B - Z) / 2

    @staticmethod
    def extended_root_gas_sc(B: NumericType, Z: NumericType) -> NumericType:
        """Auxiliary function implementing the formula for the extended, supercritical
        gas root.

        Parameters:
            B: Dimensionless covolume.
            Z: The real root.

        """
        return (1 - B - Z) / 2 + B  # * 2 + self.B_CRIT

    @staticmethod
    def extended_root_liquid_sc(B: NumericType, Z: NumericType) -> NumericType:
        """Auxiliary function implementing the formula for the extended, supercritical
        liquid root

        Parameters:
            B: Dimensionless covolume.
            Z: The real root.

        """
        return (B - Z) / 2 + Z

    def _Z(
        self,
        A: NumericType,
        B: NumericType,
        apply_smoother: bool = False,
        asymmetric_extension: bool = True,
        use_widom_line: bool = True,
        Z_as_AD: bool = True,
    ) -> tuple[NumericType, NumericType]:
        """Auxiliary method to compute the compressibility factor based on Cardano
        formulas.

        Parameters:
            A: Non-dimensional cohesion.
            B: Non-dimensional covolume.
            apply_smoother: ``default=False``

                Flag to apply smoothing procedure.
            asymmetric_extension: ``default=True``

                If True, creates an asymmetric extension above the critical line for the
                liquid-like root, since violations of the lower ``B``-bound where
                observed there.
            use_widom_line: ``default=True``

                If True, uses a linear approximation of the Widom line to separate
                between liquid-like and gas-like root in the supercritical region.
                Otherwise a simple comparison of root size is used.

        Returns:
            Returns two roots. The first one corresponds to the assigned phase label
            :attr:`gaslike`. The second root is the other root.

        """
        shape = None  # will remain None if input is not ad
        # to determine the number of values
        n_a = None
        n_b = None

        # Avoid sparse efficiency warnings and make indexable
        if isinstance(A, pp.ad.AdArray):
            shape = A.jac.shape
            n_a = len(A.val)
            A.jac = A.jac.tolil()
        elif isinstance(A, numbers.Real):
            A = np.array([A])
            n_a = 1
        else:
            n_a = len(A)
        if isinstance(B, pp.ad.AdArray):
            shape = B.jac.shape
            n_b = len(B.val)
            B.jac = B.jac.tolil()
        elif isinstance(B, numbers.Real):
            B = np.array([B])
            n_b = 1
        else:
            n_b = len(B)

        n = np.max((n_a, n_b))  # determine the number of vectorized values

        # the coefficients of the compressibility polynomial
        c0 = _power(B, 3) + _power(B, 2) - A * B
        c1 = A - 2 * B - 3 * _power(B, 2)
        c2 = B - 1

        # the coefficients of the reduced polynomial (elimination of 2-monomial)
        r = c1 - _power(c2, 2) / 3
        q = 2 / 27 * _power(c2, 3) - c2 * c1 / 3 + c0

        # discriminant to determine the number of roots
        delta = _power(q, 2) / 4 + _power(r, 3) / 27

        if shape:
            Z_L = pp.ad.AdArray(np.zeros(n), sps.lil_matrix(shape))
            Z_G = pp.ad.AdArray(np.zeros(n), sps.lil_matrix(shape))
        else:
            Z_L = np.zeros(n)
            Z_G = np.zeros(n)

        # an indicater where the root is extended
        self.is_extended = np.zeros(n, dtype=bool)

        ### CLASSIFYING REGIONS
        # NOTE: The logical comparisons are a bit awkward for compatibility reasons with
        # AD-arrays
        # identify super-critical line
        self.is_supercritical = B >= self.critical_line(A)
        # identify approximated sub pseudo-critical line (approximates Widom line)
        widom_line = B <= self.Widom_line(A)

        # At A,B=0 we have 2 real roots, one with multiplicity 2
        zero_point = (
            (A >= -self.eps) & (A <= self.eps) & (B >= -self.eps) & (B <= self.eps)
        )
        # The critical point is known to be a triple-point
        critical_point = (
            (A >= self.A_CRIT - self.eps)
            & (A <= self.A_CRIT + self.eps)
            & (B >= self.B_CRIT - self.eps)
            & (B <= self.B_CRIT + self.eps)
        )
        # subcritical triangle in the acbc rectangle
        # Area where the Gharbia analysis and extension holds
        gharbia_ext = (~self.is_supercritical) & (B < self.B_CRIT)
        # extra regions for extensions
        liq_ext_supc = self.is_supercritical & (~widom_line)
        gas_ext_supc = (~gharbia_ext) & widom_line

        # discriminant of zero indicates triple or two real roots with multiplicity
        degenerate_region = (delta >= -self.eps) & (delta <= self.eps)

        double_root_region = degenerate_region & ((r < -self.eps) | (r > self.eps))
        triple_root_region = degenerate_region & ((r >= -self.eps) & (r <= self.eps))

        one_root_region = delta > self.eps
        three_root_region = delta < -self.eps

        # sanity check that every cell/case is covered
        assert np.all(
            one_root_region
            | double_root_region
            | three_root_region
            | triple_root_region
        ), "Uncovered cells/rows detected in PR root computation."

        # sanity check that the regions are mutually exclusive
        # this array must have 1 in every entry for the test to pass
        trues_per_row = np.vstack(
            [one_root_region, triple_root_region, double_root_region, three_root_region]
        ).sum(axis=0)
        trues_check = np.ones(n, dtype=trues_per_row.dtype)
        assert np.all(
            trues_check == trues_per_row
        ), "Regions with different root scenarios overlap."

        ### COMPUTATIONS IN THE ONE-ROOT-REGION
        # Missing real root is replaced with conjugated imaginary roots
        self.regions[1] = one_root_region
        if np.any(one_root_region):
            r_ = r[one_root_region]
            q_ = q[one_root_region]
            delta_ = delta[one_root_region]
            c2_ = c2[one_root_region]
            A_ = A[one_root_region]
            B_ = B[one_root_region]

            # delta has only positive values in this case by logic
            t_1 = -q_ / 2 + _sqrt(delta_)
            t_2 = -q_ / 2 - _sqrt(delta_)
            t = t_1.copy()
            t2_greater = pp.ad.abs(t_2) > pp.ad.abs(t_1)
            t[t2_greater] = t_2[t2_greater]

            # principal cubic root if t is negative
            im_cube = t < 0.0
            if np.any(im_cube):
                t[im_cube] = t[im_cube] * (-1)
                u = _cbrt(t)
                u[im_cube] = u[im_cube] * (-1)
            else:
                u = _cbrt(t)

            # TODO In rare, un-physical areas of A,B, u can become zero,
            # causing infinity here, e.g.
            # A = 0.3620392380873223
            # B = -0.4204815080014268
            # this should never happen in physical simulations,
            # but I note it here nevertheless - VL
            real_part = u - r_ / (u * 3)
            z_1 = real_part - c2_ / 3  # Only real root, always greater than B

            # real part of the conjugate imaginary roots
            # used for extension of vanished roots
            w = self.extended_root_sub(B_, z_1)

            # using asymmetric, supercritical extension
            gas_ext_supc = gas_ext_supc[one_root_region]
            liq_ext_supc = liq_ext_supc[one_root_region]
            smoothing_distance = self.smooth_e
            if np.any(gas_ext_supc) and asymmetric_extension:
                w_g = self.extended_root_gas_sc(B_, z_1)[gas_ext_supc]
                w_sub = w[gas_ext_supc]

                # compute normal distance of extended gas root in supercritical area
                # to line B = B_CRIT
                a = (
                    A_.val[gas_ext_supc]
                    if isinstance(A_, pp.ad.AdArray)
                    else A_[gas_ext_supc]
                )
                b = (
                    B_.val[gas_ext_supc]
                    if isinstance(B_, pp.ad.AdArray)
                    else B_[gas_ext_supc]
                )
                ab = np.array([a, b])

                d_g = _point_to_line_distance(ab, self._B_crit_points)

                # smoothing towards subcritical region (Gharbia extension)
                smooth = (d_g < smoothing_distance) & (b >= self.B_CRIT)
                d = d_g / smoothing_distance  # normalize distance
                w_g[smooth] = (w_sub * (1 - d) + w_g * d)[smooth]
                w[gas_ext_supc] = w_g
            if np.any(liq_ext_supc) and asymmetric_extension:
                # Extended root in the supercritical region
                w_l = self.extended_root_liquid_sc(B_, z_1)[liq_ext_supc]
                w_g = self.extended_root_gas_sc(B_, z_1)[liq_ext_supc]
                w_sub = w[liq_ext_supc]

                # compute normal distance of extended liquid root to critical line and
                # normal line and chose the smaller one
                a = (
                    A_.val[liq_ext_supc]
                    if isinstance(A_, pp.ad.AdArray)
                    else A_[liq_ext_supc]
                )
                b = (
                    B_.val[liq_ext_supc]
                    if isinstance(B_, pp.ad.AdArray)
                    else B_[liq_ext_supc]
                )
                ab = np.array([a, b])
                d_w = _point_to_line_distance(ab, self._widom_points)
                d_s = _point_to_line_distance(ab, self._critline_points)

                # smoothing towards supercritical gas extension
                # Smoothing using a convex combination of extended gas root
                smooth = (d_w < smoothing_distance) & (b >= self.B_CRIT)
                d = d_w / smoothing_distance  # normalize distance
                w_l[smooth] = (w_g * (1 - d) + w_l * d)[smooth]
                # smoothing towards subcritical Ben Gharbia extension
                smooth = (d_s < smoothing_distance) & (b < self.B_CRIT)
                d = d_s / smoothing_distance
                w_l[smooth] = (w_sub * (1 - d) + w_l * d)[smooth]

                w[liq_ext_supc] = w_l

            # if use_widom_line:
            #     extension_is_bigger = widom_line[one_root_region]
            # else:
            extension_is_bigger = (z_1 < w) & gharbia_ext[one_root_region]

            # assign the smaller values to w
            z_1_small = z_1[extension_is_bigger]
            z_1[extension_is_bigger] = w[extension_is_bigger]
            w[extension_is_bigger] = z_1_small

            Z_L[one_root_region] = w
            Z_G[one_root_region] = z_1

            extension_is_bigger = widom_line[one_root_region] & (
                ~gharbia_ext[one_root_region]
            )

            # assign the smaller values to w
            z_1_small = z_1[extension_is_bigger]
            z_1[extension_is_bigger] = w[extension_is_bigger]
            w[extension_is_bigger] = z_1_small

            Z_L[one_root_region] = w
            Z_G[one_root_region] = z_1

            # Store flag where the extended root was used
            if self.gaslike:
                self.is_extended[one_root_region] = extension_is_bigger
            else:
                self.is_extended[one_root_region] = ~extension_is_bigger

        ### COMPUTATIONS IN THE THREE-ROOT-REGION
        # compute all three roots, label them (smallest=liquid, biggest=gas)
        # optionally smooth them
        self.regions[3] = three_root_region
        if np.any(three_root_region):
            r_ = r[three_root_region]
            q_ = q[three_root_region]
            c2_ = c2[three_root_region]

            # compute roots in three-root-region using Cardano formula,
            # Casus Irreducibilis
            t_2 = pp.ad.arccos(-q_ / 2 * _sqrt(-27 * _power(r_, -3))) / 3
            t_1 = _sqrt(-4 / 3 * r_)

            z3 = t_1 * pp.ad.cos(t_2) - c2_ / 3
            z2 = -t_1 * pp.ad.cos(t_2 + np.pi / 3) - c2_ / 3
            z1 = -t_1 * pp.ad.cos(t_2 - np.pi / 3) - c2_ / 3

            # Smoothing procedure only valid in the sub-critical area, where the three
            # roots are positive and bound from below by B
            smoothable = gharbia_ext[three_root_region]
            if apply_smoother and np.any(smoothable):
                z1_s, z3_s = root_smoother(z1, z2, z3, self.smooth_3)

                z3[smoothable] = z3_s[smoothable]
                z1[smoothable] = z1_s[smoothable]

            # assert roots are ordered by size
            assert np.all(z1 <= z3), "Roots in three-root-region improperly ordered."

            Z_L[three_root_region] = z1
            Z_G[three_root_region] = z3

        # we put computations in the triple and double root region at the end
        # as corrective features.

        ### COMPUTATIONS IN TRIPLE ROOT REGION
        # The critical point is known to be a triple root
        # Use logical or to include unknown triple points, but that should not happen
        # NOTE In this case, the roots are unavoidably equal
        region = triple_root_region | critical_point
        self.regions[0] = region
        if np.any(region):
            c2_ = c2[region]

            z = -c2_ / 3

            assert np.all(
                z > B[region]
            ), "Triple-roots violating the lower physical bound B detected."

            Z_L[region] = z
            Z_G[region] = z

        ### COMPUTATIONS IN DOUBLE ROOT REGION
        # The point A,B = 0 is known to be such a point
        region = double_root_region | zero_point
        self.regions[2] = region
        if np.any(region):
            r_ = r[region]
            q_ = q[region]
            c2_ = c2[region]

            u = 3 / 2 * q_ / r_

            z_1 = 2 * u - c2_ / 3
            z_23 = -u - c2_ / 3

            # to avoid indexing issues
            if isinstance(z_1, numbers.Real):
                z_1 = np.array([z_1])
                z_23 = np.array([z_23])

            # choose bigger root as gas like
            # theoretically they should strictly be different, otherwise it would be
            # the three root case
            double_is_bigger = z_23 > z_1

            # exchange values such that z_1 is the bigger root
            z_1_small = z_1[double_is_bigger]
            z_1[double_is_bigger] = z_23[double_is_bigger]
            z_23[double_is_bigger] = z_1_small

            Z_L[region] = z_23
            Z_G[region] = z_1

        # convert Jacobians to csr
        if isinstance(Z_L, pp.ad.AdArray):
            Z_L.jac = Z_L.jac.tocsr()
            if not Z_as_AD:
                Z_L = Z_L.val
        if isinstance(Z_G, pp.ad.AdArray):
            Z_G.jac = Z_G.jac.tocsr()
            if not Z_as_AD:
                Z_G = Z_G.val

        if self.gaslike:
            return Z_G, Z_L
        else:
            return Z_L, Z_G


class _PengRobinson(AbstractEoS):
    """A class implementing thermodynamic properties resulting from the Peng-Robinson
    equation of state

        ``p = R * T / (v - b) - a / (b**2 + 2 * v * b - b**2)``.

    Note:
        1. The various methods providing thermodynamic quantities are AD-compatible.
           They can be wrapped into AD-Functions and are able to take Ad-Arrays as
           input.
        2. Calling any thermodynamic property with molar fractions ``X`` as arguments
           raises errors if the number of passed fractions does not match the
           number of modelled components in :attr:`components`.

           The order of fractions arguments must correspond to the order in
           :attr:`components`.
        3. As of now, supercritical phases are not really supported, just indicated.
           Be aware of the limitations of the Peng-Robinson model!

    For a list of computed properties, see dataclass :class:`PhaseProperties_cubic`.

    References:
        [1]: `Peng, Robinson (1976) <https://doi.org/10.1021/i160057a011>`_
        [2]: `Ben Gharbia et al. (2021) <https://doi.org/10.1051/m2an/2021075>`_
        [3]: `Vu et al. (2021) <https://doi.org/10.1016/j.matcom.2021.07.015>`_
        [4]: `Zhu, Okuno (2014) <http://dx.doi.org/10.1016/j.fluid.2014.07.003>`_
        [5]: `Zhu, Okuno (2015) <https://onepetro.org/spersc/proceedings/15RSS/
             1-15RSS/D011S001R002/183434>`_
        [6]: `Connolly et al. (2021) <https://doi.org/10.1016/j.ces.2020.116150>`_

    Parameters:
        gaslike: A bool indicating if if the gas-like root of the cubic polynomial
            should be used for computations.

            If False, the liquid-like root is used.
        mixingrule: ``default='VdW'``

            Name of the mixing rule to be applied.
        smoothing_factor: ``default=1e-2``

            A small number to determine proximity between 3-root and double-root case.

            See also `Vu et al. (2021), Section 6. <https://doi.org/10.1016/
            j.matcom.2021.07.015>`_ .
        eps: ``default=1e-14``

            A small number defining the numerical zero.

            Used for the computation of roots to distinguish between phase-regions.
        *args: Placeholder in case of inheritance.
        **kwargs: Placeholder in case of inheritance.

    """

    def __init__(
        self,
        gaslike: bool,
        *args,
        mixingrule: Literal["VdW"] = "VdW",
        smoothing_factor: float = 1e-2,
        eps: float = 1e-14,
        **kwargs,
    ) -> None:
        super().__init__(gaslike)

        self._a_crit_vals: tuple[float] = []
        """Critical cohesion values per component, using EoS specific critical value.

        Computed in setter for :meth:`components`.

        """

        self._b_vals: tuple[float] = []
        """Critical covolume values per component, using EoS-specific critical value.

        Computed in setter for :meth:`components`.

        """

        self._a_cor_vals: tuple[float] = []
        """Cohesion correction weights, appearing in the linearized alpha-correction of
        the cohesion.

        Computed in setter for :meth:`components`.

        """

        self._bip_vals: np.ndarray = np.array([])
        """A matrix  in strictly upper-triangle form containing the constant
        binary interaction parameters between components at indices ``[i][j]``.

        The values are loaded once during the setter for :meth:`components`.

        """

        self._custom_bips: dict[
            str, Callable[[NumericType], tuple[NumericType, NumericType]]
        ] = dict()
        """A map between indices for :attr:`_bip_vals` (upper triangle) and
        custom implementation of BIPs (temperature-dependent), if available.

        See :attr:`~porepy.compositional.peng_robinson.pr_components.Component_PR.bip_map`.

        """

        self.gaslike: bool = bool(gaslike)
        """Flag passed at instantiation denoting the state of matter, liquid or gas."""

        assert mixingrule in ["VdW"], f"Unknown mixing rule {mixingrule}."
        self.mixingrule: str = mixingrule
        """The mixing rule passed at instantiation."""

        self.eps: float = eps
        """Passed at instantiation."""

        self.smooth_3: float = smoothing_factor
        """Passed at instantiation."""

        self.smooth_e = 1e-3

        self.regions: list[np.ndarray] = [np.zeros(1, dtype=bool)] * 4
        """A list of root-region indicates.

        Contains boolean arrays indicating where which root case is registered
        during the last computation.

        - 0: Array containing True where a triple-root was computed.
        - 1: Array containing True where a single real root was computed.
        - 2: Array indicating where 2 distinct real roots were computed.
        - 3: Array indicating where 3 distinct real roots were computed.

        """

    def _num_frac_check(self, X) -> None:
        """Auxiliary method to check the number of passed fractions.
        Raises an error if number does not match the number of present components."""
        if len(X) != len(self.components):
            raise ValueError(
                f"{len(X)} fractions given but "
                + f"{len(self.components)} components present."
            )

    @property
    def components(self) -> list[pp.compositional.Component]:
        """The child class setter calculates EoS specific values per set component.

        Values like critical cohesion and covolume values, corrective parameters and
        binary interaction parameters are obtained only once and stored during in the
        setter method.

        The setter additionally checks for the availability of BIPs and throws
        respective warnings. Unavailable BIPs are set to zero.

        Todo:
            Should it be a warning or an error?

        Warning:
            If two components ``i`` and ``j``, with ``i < j``, have both BIPs
            BIPs implemented for each other in
            :attr:`~porepy.compositional.peng_robinson.pr_components.Component_PR.bip_map`,
            the first implementation of ``i`` is taken and a warning is emitted.

        Parameters:
            components: A list of Peng-Robinson compatible components.

        Raises:
            RuntimeError: If two different components implement models for BIP for each
                other (it is impossible to decide which one to use).

        """
        return AbstractEoS.components.fget(self)

    @components.setter
    def components(self, components: list[pp.compositional.Component]) -> None:
        a_crits: list[float] = list()
        bs: list[float] = list()
        a_cors: list[float] = list()

        nc = len(components)

        # computing constant parameters
        for comp in components:
            a_crits.append(self.a_crit(comp.p_crit, comp.T_crit))
            bs.append(self.b_crit(comp.p_crit, comp.T_crit))
            a_cors.append(self.a_correction_weight(comp.omega))

        self._a_cor_vals = tuple(a_cors)
        self._b_vals = tuple(bs)
        self._a_crit_vals = tuple(a_crits)

        # prepare storage for constant bips
        self._bip_vals = np.zeros((nc, nc))

        # storing missing bips and custom bip callables
        bip_callables: dict[
            str, Callable[[NumericType], tuple[NumericType, NumericType]]
        ] = dict()

        for i in range(nc):
            comp_i = components[i]
            # storage for custom BIPs with other component, if custom available.
            customs_for_i = []
            # check if component i has custom BIPs implemented
            if hasattr(comp_i, "bip_map"):
                for j in range(0, nc):
                    comp_j = components[j]
                    bip_c = comp_i.bip_map.get(comp_j.CASr_number, None)
                    # if found, store and mark to check it with the other component
                    if bip_c is not None:
                        customs_for_i.append(j)
                        bip_callables.update({(i, j): bip_c})

            for j in range(i + 1, nc):  # strictly upper triangle matrix
                comp_j = components[j]

                # use custom models if implemented
                if hasattr(comp_j, "bip_map"):
                    bip_c = comp_j.bip_map.get(comp_i.CASr_number, None)
                    # check if component i has already implemented a bip
                    # If not, check if j has it implemented, store and continue
                    # if it has, warn if double implementation detected and continue
                    if j in customs_for_i:
                        if bip_c is not None:
                            logger.warn(
                                "Detected double-implementation of BIP for components"
                                + f"{comp_i.name} and {comp_j.name}."
                                + "\nA fix to the model components if recommended."
                            )
                        continue
                    else:
                        # store callables, if implemented, and continue
                        if bip_c is not None:
                            assert (i, j) not in bip_callables.keys()
                            bip_callables.update({(i, j): bip_c})
                            continue
                        # at this point, the code will continue and try load a database
                        # bip.
                        # This happens if i and j have custom BIPs, but not for each
                        # other

                # try to load BIPs from database, if custom has not been found so far
                bip = load_bip(comp_i.CASr_number, comp_j.CASr_number)
                # TODO: This needs some more thought. How to handle missing BIPs?
                if bip == 0.0:
                    logger.warn(
                        "Loaded a BIP with zero value for"
                        + f" components {comp_i.name} and {comp_j.name}."
                    )
                self._bip_vals[i, j] = bip

        # If custom implementations for bips were found, store them.
        # Otherwise store an empty dict.
        if bip_callables:
            self._custom_bips = bip_callables
        else:
            self._custom_bips = dict()

        AbstractEoS.components.fset(self, components)

    def compute(
        self,
        p: NumericType,
        T: NumericType,
        X: list[NumericType],
        apply_smoother: bool = False,
        Z_as_AD: bool = True,
        **kwargs,
    ) -> PhaseProperties_cubic:
        """Computes all thermodynamic properties based on the passed state.

        Warning:
            ``p``, ``T``, ``X`` have a union type, meaning the results will be of
            the same. When mixing numpy arrays, porepy's Ad arrays and numbers,
            the user must make sure there will be no compatibility issues.

            This method is not supposed to be used with AD Operator instances.

        Parameters:
            p: Pressure
            T: Temperature
            X: ``len=num_components``

                Fraction per component to be used in the computation,
                ordered as in :meth:`components`.
            apply_smoother: ``default=False``

                If True, a smoothing procedure is applied in the three-root-region,
                where the intermediate root approaches one of the other roots
                (see [3]).

                This is to be used **within** iterative procedures for numerical
                reasons. Once convergence is reached, the true roots should be computed
                without smoothing.
            **kwargs: Placeholder in case of inheritance.

        Raises:
            ValueError: If a mismatch between number passed fractions and modelled
                components is detected.

        Returns:
            A dataclass containing thermodynamic properties resulting from a cubic EoS.

        """
        # sanity check
        self._num_frac_check(X)
        # binary interaction parameters
        bip, dT_bip = self.compute_bips(T)
        # cohesion and covolume, and derivative
        a, dT_a, a_comps, _ = self._compute_cohesion_terms(T, X, bip, dT_bip)
        b = self._compute_mixture_covolume(X)
        # compute non-dimensional quantities
        A = self.A(a, p, T)
        B = self.B(b, p, T)
        # root
        Z, Z_other = self._Z(A, B, apply_smoother=apply_smoother, Z_as_AD=Z_as_AD)

        # volume and density
        v = self._v(p, T, Z)
        rho = v ** (-1)  # self._rho(p, T, Z)
        rho_mass = self.get_rho_mass(rho, X)
        # departure enthalpy
        dT_A = dT_a / (R_IDEAL_MOL * T) ** 2 * p - 2 * a / (R_IDEAL_MOL**2 * T**3) * p
        h_dep = self._h_dep(T, Z, A, dT_A, B)
        h_ideal = self.get_h_ideal(p, T, X)
        h = h_ideal + h_dep

        # Fugacity extensions as per Ben Gharbia 2021
        # dxi_a = list()
        # if np.any(self.is_extended):
        #     extend_phi: bool = True
        #     rho_ext = self._v(p, T, Z_other) **(-1)
        #     G = self._Z_polynom(Z, A, B)
        #     Gamma_ext = G * Z / ((B - Z) * (B**2 - Z**2 - 2 * Z * B))
        #     dxi_Z_other = list()
        #     for i in range(len(self.components)):
        #         dxi_a_ = self._dXi_a(X, a_comps, bip, i)
        #         dxi_a.append(dxi_a_)
        #         dxi_Z_other_ = self._dxi_Z(T, rho_ext, a, b, self._b_vals[i], dxi_a[i])
        #         dxi_Z_other.append(dxi_Z_other_)
        #     dZ_other = safe_sum([x * dz for x, dz in zip(X, dxi_Z_other)])
        # else:
        #     extend_phi: bool = False
        #     for i in range(len(self.components)):
        #         dxi_a_ = self._dXi_a(X, a_comps, bip, i)
        #         dxi_a.append(dxi_a_)
        # fugacity per present component
        phis: list[NumericType] = list()
        for i in range(len(self.components)):
            b_i = self._b_vals[i]
            B_i = self.B(b_i, p, T)
            A_i = self.A(self._dXi_a(X, a_comps, bip, i), p, T)
            # A_i = self._A(dxi_a[i], p, T)
            phi_i = self._phi_i(Z, A_i, A, B_i, B)

            # if extend_phi:
            #     w1 = (B - B_i + dZ_other - dxi_Z_other[i]) / Z / 2
            #     w2 = (B - B_i) / B
            #     ext_i = Gamma_ext * (w1 + w2)

            #     ext_i = ext_i[self.is_extended]
            #     phi_i[self.is_extended] = phi_i[self.is_extended] + ext_i

            phis.append(phi_i)

        # these two are open TODO
        kappa = self._kappa(p, T, Z)
        mu = self._mu(p, T, Z)

        return PhaseProperties_cubic(
            a=a,
            dT_a=dT_a,
            b=b,
            A=A,
            B=B,
            Z=Z,
            rho=rho,
            rho_mass=rho_mass,
            v=v,
            h_ideal=h_ideal,
            h_dep=h_dep,
            h=h,
            phis=phis,
            kappa=kappa,
            mu=mu,
        )

    def compute_bips(
        self, T: NumericType
    ) -> tuple[list[list[NumericType]], list[list[NumericType]]]:
        """
        Parameters:
            T: Temperature.

        Returns:
            Two nested lists in upper-triangle form containing the binary interaction
            parameters. The first structure contains the parameters, the second
            their temperature-derivatives.

            The lists are such that indexing by ``[i][j]`` gives the interaction
            parameter between components ``i`` and ``j``.

            The lower triangle and the main diagonal of this matrix-like structure
            are filled with zeros, since the interaction parameters are symmetrical.

        """
        # construct first output from the constant bips and their trivial derivative
        bips: list[list[NumericType]] = (self._bip_vals + self._bip_vals.T).tolist()
        dT_bips: list[list[NumericType]] = np.zeros(self._bip_vals.shape).tolist()

        # if any custom bips are to be used, update the respective entries
        if self._custom_bips:
            for idx, bip_c in self._custom_bips.items():
                i, j = idx
                bip, dT_bip = bip_c(T)

                # values are symmetric
                bips[i][j] = bip
                bips[j][i] = bip
                dT_bips[i][j] = dT_bip
                dT_bips[j][i] = dT_bip

        return bips, dT_bips

    def _compute_component_cohesions(
        self, T: NumericType
    ) -> tuple[list[NumericType], list[NumericType]]:
        """
        Parameters:
            T: Temperature

        Returns:
            A 2-tuple containing

            1. temperature-dependent cohesion values per component.
            2. the temperature-derivative of the the cohesion per component.

        """
        a: list[NumericType] = []
        dT_a: list[NumericType] = []

        for i, comp in enumerate(self.components):
            a_cor_i = self._a_cor_vals[i]
            a_crit_i = self._a_crit_vals[i]
            T_r_i = T / comp.T_crit

            # check if special model for alpha was implemented.
            if hasattr(comp, "alpha"):
                alpha_i = comp.alpha(T)
            else:
                alpha_i = self.a_correction(a_cor_i, T_r_i)

            a_i = a_crit_i * _power(alpha_i, 2)
            # outer derivative
            dT_a_i = 2 * a_crit_i * alpha_i
            # inner derivative
            dT_a_i *= (-a_cor_i / (2 * comp.T_crit)) * _power(T_r_i, -1 / 2)

            a.append(a_i)
            dT_a.append(dT_a_i)

        return a, dT_a

    def _compute_cohesion_terms(
        self,
        T: NumericType,
        X: list[NumericType],
        bip: list[list[NumericType]],
        dT_bip: list[list[NumericType]],
    ) -> tuple[NumericType, NumericType, NumericType, NumericType]:
        """
        Parameters:
            T: Temperature.
            X: ``len=num_components``

                Fraction per component to be used in the computation,
                ordered as in :attr:`components`.
            bip: A nested list or matrix-like structure, such that ``bip[i][j]`` is the
                binary interaction parameter between components ``i`` and ``j``,
                where the indices run over the enumeration of ``X`` and ``a``.

                The matrix-like structure can be an upper diagonal matrix, since the
                interaction parameters are symmetric.
            dT_bip: Same as ``bip``, holding only the temperature-derivative of the
                binary interaction parameters.

        Returns:
            A 4-tuple containing

            - The cohesion ``a`` of the mixture for given thermodynamic state,
              using the assigned mixing rule,
            - its temperature-derivative,
            - cohesion values per component,
            - their temperature, derivatives

        """
        a_c, dT_a_c = self._compute_component_cohesions(T)
        if self.mixingrule == "VdW":
            a_mix, dT_a_mix = VanDerWaals.cohesion(X, a_c, dT_a_c, bip, dT_bip)
            return a_mix, dT_a_mix, a_c, dT_a_c
        else:
            raise ValueError(f"Unknown mixing rule {self.mixingrule}.")

    def _compute_mixture_covolume(self, X: list[NumericType]) -> NumericType:
        """
        Parameters:
            X: ``len=num_components``

                Fraction per component to be used in the computation,
                ordered as in :attr:`components`.

        Returns:
            The covolume ``b`` for given thermodynamic state, using the assigned
            mixing rule.

        """
        if self.mixingrule == "VdW":
            return VanDerWaals.covolume(X, self._b_vals)
        else:
            raise ValueError(f"Unknown mixing rule {self.mixingrule}.")

    # formulae -------------------------------------------------------------------------

    @staticmethod
    def b_crit(p_crit: float, T_crit: float) -> float:
        """
        .. math::

            a_{crit} = A_{crit} * \\frac{R^2 T_{crit}^2}{p_{crit}}

        Parameters:
            p_crit: Critical pressure of a component.
            T_crit: Critical temperature of a component.

        Returns:
            The component-specific critical covolume.

        """
        return B_CRIT * (R_IDEAL_MOL * T_crit) / p_crit

    @staticmethod
    def a_crit(p_crit: float, T_crit: float) -> float:
        """
        .. math::

            a_{crit} = A_{crit} * \\frac{R T_{crit}}{p_{crit}}

        Parameters:
            p_crit: Critical pressure of a component.
            T_crit: Critical temperature of a component.

        Returns:
            The component-specific critical cohesion.

        """
        return A_CRIT * (R_IDEAL_MOL**2 * T_crit**2) / p_crit

    @staticmethod
    def a_correction_weight(omega: float) -> float:
        """
        References:
            `Zhu et al. (2014), Appendix A
            <https://doi.org/10.1016/j.fluid.2014.07.003>`_

        Parameters:
            omega: Acentric factor for a component.

        Returns:
            Returns the cohesion correction parameter depending on a component's
            acentric factor.

        """
        if omega < 0.491:
            return 0.37464 + 1.54226 * omega - 0.26992 * omega**2
        else:
            return (
                0.379642 + 1.48503 * omega - 0.164423 * omega**2 + 0.016666 * omega**3
            )

    @staticmethod
    def a_correction(kappa: float, T_r: NumericType) -> NumericType:
        """
        Parameters:
            kappa: Acentric factor-dependent weight in the linearized correction of
                the cohesion for a component.
            T_r: Reduced temperature for a component (divided by the component's
                critical temperature.).

        Returns:
            The root of the linearized correction for the cohesion term.

        """
        return 1 + kappa * (1 - _sqrt(T_r))

    def _dXi_a(
        self,
        X: list[NumericType],
        a: list[NumericType],
        bip: list[list[NumericType]],
        i: int,
    ) -> NumericType:
        """Auxiliary method to compute parts of the fugacity coefficients."""
        if self.mixingrule == "VdW":
            return VanDerWaals.dXi_cohesion(X, a, bip, i)
        else:
            raise ValueError(f"Unknown mixing rule {self.mixingrule}.")

    @staticmethod
    def A(a: NumericType, p: NumericType, T: NumericType) -> NumericType:
        """Auxiliary method implementing formula for non-dimensional cohesion."""
        if isinstance(T, pp.ad.AdArray):
            return T ** (-2) * a * p / R_IDEAL_MOL**2
        else:
            return a * p / (R_IDEAL_MOL**2 * T**2)

    @staticmethod
    def B(b: NumericType, p: NumericType, T: NumericType) -> NumericType:
        """Auxiliary method implementing formula for non-dimensional covolume."""
        if isinstance(T, pp.ad.AdArray):
            return T ** (-1) * b * p / R_IDEAL_MOL
        else:
            return b * p / (R_IDEAL_MOL * T)

    # TODO
    def _kappa(self, p: NumericType, T: NumericType, Z: NumericType) -> NumericType:
        """
        Parameters:
            p: Pressure.
            T: Temperature.
            Z: Compressibility factor.

        Returns:
            The thermal conductivity.

        """
        return 1.0

    # TODO
    def _mu(self, p: NumericType, T: NumericType, Z: NumericType) -> NumericType:
        """
        Parameters:
            p: Pressure.
            T: Temperature.
            Z: Compressibility factor.

        Returns:
            The dynamic viscosity.

        """
        return 1.0

    @staticmethod
    def _rho(p: NumericType, T: NumericType, Z: NumericType) -> NumericType:
        """Auxiliary function implementing the formula for density."""
        return Z ** (-1) * p / (T * R_IDEAL_MOL)

    @staticmethod
    def _v(p: NumericType, T: NumericType, Z: NumericType) -> NumericType:
        """Auxiliary function implementing the formula for volume."""
        if isinstance(T, pp.ad.AdArray):
            return T * Z * R_IDEAL_MOL / p
        else:
            return Z * T / p * R_IDEAL_MOL

    @staticmethod
    def _log_ZB_0(Z: NumericType, B: NumericType) -> NumericType:
        return trunclog((B - Z) * (-1), 1e-6)

    @staticmethod
    def _log_ZB_1(Z: NumericType, B: NumericType) -> NumericType:
        return trunclog(((1 + np.sqrt(2)) * B + Z) / ((1 - np.sqrt(2)) * B + Z), 1e-6)

    def _h_dep(
        self,
        T: NumericType,
        Z: NumericType,
        A: NumericType,
        dT_A: NumericType,
        B: NumericType,
    ) -> NumericType:
        """Auxiliary function for computing the enthalpy departure function."""
        if isinstance(T, pp.ad.AdArray):
            i = T * (Z - 1) * R_IDEAL_MOL
        else:
            i = (Z - 1) * T * R_IDEAL_MOL
        return (
            1
            / np.sqrt(8)
            * (dT_A * T**2 * R_IDEAL_MOL + A * T * R_IDEAL_MOL)
            / B
            * self._log_ZB_1(Z, B)
            + i
        )

    def _g_dep(
        self, T: NumericType, A: NumericType, B: NumericType, Z: NumericType
    ) -> NumericType:
        """Auxiliary function for computing the Gibbs departure function."""
        return (
            (self._log_ZB_0(Z, B) - self._log_ZB_1(Z, B) * A / B / np.sqrt(8))
            * T
            * R_IDEAL_MOL
        )

    @staticmethod
    def _g_ideal(X: list[NumericType]) -> NumericType:
        """Auxiliary function to compute the ideal part of the Gibbs energy."""
        return safe_sum([x * trunclog(x, 1e-6) for x in X])

    def _phi_i(
        self,
        Z: NumericType,
        A_i: NumericType,
        A: NumericType,
        B_i: NumericType,
        B: NumericType,
    ) -> NumericType:
        """Auxiliary method implementing the formula for the fugacity coefficient."""
        log_phi_i = (
            B ** (-1) * (Z - 1) * B_i
            - self._log_ZB_0(Z, B)
            - A / (B * np.sqrt(8)) * (A_i / A - B ** (-1) * B_i) * self._log_ZB_1(Z, B)
        )
        return truncexp(log_phi_i)

    @staticmethod
    def _dxi_Z(
        T: NumericType,
        rho: NumericType,
        a: NumericType,
        b: NumericType,
        b_i: NumericType,
        dxi_a: NumericType,
    ) -> NumericType:
        """Auxiliary function implementing the derivative of the compressibility factor
        w.r.t. molar fraction ``x_i``."""
        d = 1 + 2 * b * rho - (b * rho) ** 2
        return (1 - b * rho) ** (-2) * rho * b_i + (
            a * rho * (2 * rho * b_i + 2 * b * rho**2 * b_i) / (d**2 * T * R_IDEAL_MOL)
            - dxi_a * rho / (d * T * R_IDEAL_MOL)
        )

    @staticmethod
    def _Z_polynom(Z: NumericType, A: NumericType, B: NumericType) -> NumericType:
        """Auxiliary method implementing the compressibility polynomial."""
        return (
            (B**3 + B**2 - A * B) + (B - 1) * Z**2 + (A - 2 * B - 3 * B**2) * Z + Z**3
        )

    @staticmethod
    def extended_root_sub(B: NumericType, Z: NumericType) -> NumericType:
        """Auxiliary function implementing the formula for the extended, subcritical
        root proposed in Gharbia et al. (2021)

        Parameters:
            B: Dimensionless covolume.
            Z: The real root.

        """
        return (1 - B - Z) / 2

    @staticmethod
    def extended_root_gas_sc(B: NumericType, Z: NumericType) -> NumericType:
        """Auxiliary function implementing the formula for the extended, supercritical
        gas root.

        Parameters:
            B: Dimensionless covolume.
            Z: The real root.

        """
        return (1 - B - Z) / 2 + B  # * 2 + self.B_CRIT

    @staticmethod
    def extended_root_liquid_sc(B: NumericType, Z: NumericType) -> NumericType:
        """Auxiliary function implementing the formula for the extended, supercritical
        liquid root

        Parameters:
            B: Dimensionless covolume.
            Z: The real root.

        """
        return (B - Z) / 2 + Z

    def _Z(
        self,
        A: NumericType,
        B: NumericType,
        apply_smoother: bool = False,
        asymmetric_extension: bool = True,
        use_widom_line: bool = True,
        Z_as_AD: bool = True,
    ) -> tuple[NumericType, NumericType]:
        """Auxiliary method to compute the compressibility factor based on Cardano
        formulas.

        Parameters:
            A: Non-dimensional cohesion.
            B: Non-dimensional covolume.
            apply_smoother: ``default=False``

                Flag to apply smoothing procedure.
            asymmetric_extension: ``default=True``

                If True, creates an asymmetric extension above the critical line for the
                liquid-like root, since violations of the lower ``B``-bound where
                observed there.
            use_widom_line: ``default=True``

                If True, uses a linear approximation of the Widom line to separate
                between liquid-like and gas-like root in the supercritical region.
                Otherwise a simple comparison of root size is used.

        Returns:
            Returns two roots. The first one corresponds to the assigned phase label
            :attr:`gaslike`. The second root is the other root.

        """
        shape = None  # will remain None if input is not ad
        # to determine the number of values
        n_a = None
        n_b = None

        # Avoid sparse efficiency warnings and make indexable
        if isinstance(A, pp.ad.AdArray):
            shape = A.jac.shape
            n_a = len(A.val)
            A.jac = A.jac.tolil()
        elif isinstance(A, numbers.Real):
            A = np.array([A])
            n_a = 1
        else:
            n_a = len(A)
        if isinstance(B, pp.ad.AdArray):
            shape = B.jac.shape
            n_b = len(B.val)
            B.jac = B.jac.tolil()
        elif isinstance(B, numbers.Real):
            B = np.array([B])
            n_b = 1
        else:
            n_b = len(B)

        n = np.max((n_a, n_b))  # determine the number of vectorized values

        # the coefficients of the compressibility polynomial
        c0 = _power(B, 3) + _power(B, 2) - A * B
        c1 = A - 2 * B - 3 * _power(B, 2)
        c2 = B - 1

        # the coefficients of the reduced polynomial (elimination of 2-monomial)
        r = c1 - _power(c2, 2) / 3
        q = 2 / 27 * _power(c2, 3) - c2 * c1 / 3 + c0

        # discriminant to determine the number of roots
        delta = _power(q, 2) / 4 + _power(r, 3) / 27

        if shape:
            Z_L = pp.ad.AdArray(np.zeros(n), sps.lil_matrix(shape))
            Z_G = pp.ad.AdArray(np.zeros(n), sps.lil_matrix(shape))
        else:
            Z_L = np.zeros(n)
            Z_G = np.zeros(n)

        # an indicater where the root is extended
        self.is_extended = np.zeros(n, dtype=bool)

        ### CLASSIFYING REGIONS
        # NOTE: The logical comparisons are a bit awkward for compatibility reasons with
        # AD-arrays
        # identify super-critical line
        self.is_supercritical = B >= critical_line(A)
        # identify approximated sub pseudo-critical line (approximates Widom line)
        below_widom = B <= widom_line(A)

        # At A,B=0 we have 2 real roots, one with multiplicity 2
        zero_point = (
            (A >= -self.eps) & (A <= self.eps) & (B >= -self.eps) & (B <= self.eps)
        )
        # The critical point is known to be a triple-point
        critical_point = (
            (A >= A_CRIT - self.eps)
            & (A <= A_CRIT + self.eps)
            & (B >= B_CRIT - self.eps)
            & (B <= B_CRIT + self.eps)
        )
        # subcritical triangle in the acbc rectangle
        # Area where the Gharbia analysis and extension holds
        gharbia_ext = (~self.is_supercritical) & (B < B_CRIT)
        # extra regions for extensions
        liq_ext_supc = self.is_supercritical & (~below_widom)
        gas_ext_supc = (~gharbia_ext) & below_widom

        # discriminant of zero indicates triple or two real roots with multiplicity
        degenerate_region = (delta >= -self.eps) & (delta <= self.eps)

        double_root_region = degenerate_region & ((r < -self.eps) | (r > self.eps))
        triple_root_region = degenerate_region & ((r >= -self.eps) & (r <= self.eps))

        one_root_region = delta > self.eps
        three_root_region = delta < -self.eps

        # sanity check that every cell/case is covered
        assert np.all(
            one_root_region
            | double_root_region
            | three_root_region
            | triple_root_region
        ), "Uncovered cells/rows detected in PR root computation."

        # sanity check that the regions are mutually exclusive
        # this array must have 1 in every entry for the test to pass
        trues_per_row = np.vstack(
            [one_root_region, triple_root_region, double_root_region, three_root_region]
        ).sum(axis=0)
        trues_check = np.ones(n, dtype=trues_per_row.dtype)
        assert np.all(
            trues_check == trues_per_row
        ), "Regions with different root scenarios overlap."

        ### COMPUTATIONS IN THE ONE-ROOT-REGION
        # Missing real root is replaced with conjugated imaginary roots
        self.regions[1] = one_root_region
        if np.any(one_root_region):
            r_ = r[one_root_region]
            q_ = q[one_root_region]
            delta_ = delta[one_root_region]
            c2_ = c2[one_root_region]
            A_ = A[one_root_region]
            B_ = B[one_root_region]

            # delta has only positive values in this case by logic
            t = -q_ / 2 + _sqrt(delta_)

            # t_1 might be negative, in this case we must choose the real cubic root
            # by extracting cbrt(-1), where -1 is the real cubic root.
            im_cube = t < 0.0
            if np.any(im_cube):
                t[im_cube] = t[im_cube] * (-1)
                u = _cbrt(t)
                u[im_cube] = u[im_cube] * (-1)
            else:
                u = _cbrt(t)

            # TODO In rare, un-physical areas of A,B, u can become zero,
            # causing infinity here, e.g.
            # A = 0.3620392380873223
            # B = -0.4204815080014268
            # this should never happen in physical simulations,
            # but I note it here nevertheless - VL
            real_part = u - r_ / (u * 3)
            z_1 = real_part - c2_ / 3  # Only real root, always greater than B

            # real part of the conjugate imaginary roots
            # used for extension of vanished roots
            w = self.extended_root_sub(B_, z_1)

            # using asymmetric, supercritical extension
            gas_ext_supc = gas_ext_supc[one_root_region]
            liq_ext_supc = liq_ext_supc[one_root_region]
            if np.any(gas_ext_supc) and asymmetric_extension:
                w_g = self.extended_root_gas_sc(B_, z_1)[gas_ext_supc]
                w_sub = w[gas_ext_supc]

                # compute normal distance of extended gas root in supercritical area
                # to line B = B_CRIT
                a = (
                    A_.val[gas_ext_supc]
                    if isinstance(A_, pp.ad.AdArray)
                    else A_[gas_ext_supc]
                )
                b = (
                    B_.val[gas_ext_supc]
                    if isinstance(B_, pp.ad.AdArray)
                    else B_[gas_ext_supc]
                )
                ab = np.array([a, b])

                d_g = point_to_line_distance(
                    ab, B_CRIT_LINE_POINTS[0], B_CRIT_LINE_POINTS[1]
                )

                # smoothing towards subcritical region (Gharbia extension)
                smooth = (d_g < self.smooth_e) & (b >= B_CRIT)
                d = d_g / self.smooth_e  # normalize distance
                w_g[smooth] = (w_sub * (1 - d) + w_g * d)[smooth]
                w[gas_ext_supc] = w_g
            if np.any(liq_ext_supc) and asymmetric_extension:
                # Extended root in the supercritical region
                w_l = self.extended_root_liquid_sc(B_, z_1)[liq_ext_supc]
                w_g = self.extended_root_gas_sc(B_, z_1)[liq_ext_supc]
                w_sub = w[liq_ext_supc]

                # compute normal distance of extended liquid root to critical line and
                # normal line and chose the smaller one
                a = (
                    A_.val[liq_ext_supc]
                    if isinstance(A_, pp.ad.AdArray)
                    else A_[liq_ext_supc]
                )
                b = (
                    B_.val[liq_ext_supc]
                    if isinstance(B_, pp.ad.AdArray)
                    else B_[liq_ext_supc]
                )
                ab = np.array([a, b])
                d_w = point_to_line_distance(ab, W_LINE_POINTS[0], W_LINE_POINTS[1])
                d_s = point_to_line_distance(
                    ab, S_CRIT_LINE_POINTS[0], S_CRIT_LINE_POINTS[1]
                )

                # smoothing towards supercritical gas extension
                # Smoothing using a convex combination of extended gas root
                smooth = (d_w < self.smooth_e) & (b >= B_CRIT)
                d = d_w / self.smooth_e  # normalize distance
                w_l[smooth] = (w_g * (1 - d) + w_l * d)[smooth]
                # smoothing towards subcritical Ben Gharbia extension
                smooth = (d_s < self.smooth_e) & (b < B_CRIT)
                d = d_s / self.smooth_e
                w_l[smooth] = (w_sub * (1 - d) + w_l * d)[smooth]

                w[liq_ext_supc] = w_l

            if use_widom_line:
                extension_is_bigger = below_widom[one_root_region]
            else:
                extension_is_bigger = z_1 < w

            # assign the smaller values to w
            z_1_small = z_1[extension_is_bigger]
            z_1[extension_is_bigger] = w[extension_is_bigger]
            w[extension_is_bigger] = z_1_small

            Z_L[one_root_region] = w
            Z_G[one_root_region] = z_1

            # Store flag where the extended root was used
            if self.gaslike:
                self.is_extended[one_root_region] = extension_is_bigger
            else:
                self.is_extended[one_root_region] = ~extension_is_bigger

        ### COMPUTATIONS IN THE THREE-ROOT-REGION
        # compute all three roots, label them (smallest=liquid, biggest=gas)
        # optionally smooth them
        self.regions[3] = three_root_region
        if np.any(three_root_region):
            r_ = r[three_root_region]
            q_ = q[three_root_region]
            c2_ = c2[three_root_region]

            # compute roots in three-root-region using Cardano formula,
            # Casus Irreducibilis
            t_2 = pp.ad.arccos(-q_ / 2 * _sqrt(-27 * _power(r_, -3))) / 3
            t_1 = _sqrt(-4 / 3 * r_)

            z3 = t_1 * pp.ad.cos(t_2) - c2_ / 3
            z2 = -t_1 * pp.ad.cos(t_2 + np.pi / 3) - c2_ / 3
            z1 = -t_1 * pp.ad.cos(t_2 - np.pi / 3) - c2_ / 3

            # Smoothing procedure only valid in the sub-critical area, where the three
            # roots are positive and bound from below by B
            smoothable = gharbia_ext[three_root_region]
            if apply_smoother and np.any(smoothable):
                z1_s, z3_s = root_smoother(z1, z2, z3, self.smooth_3)

                z3[smoothable] = z3_s[smoothable]
                z1[smoothable] = z1_s[smoothable]

            # assert roots are ordered by size
            assert np.all(z1 <= z3), "Roots in three-root-region improperly ordered."

            Z_L[three_root_region] = z1
            Z_G[three_root_region] = z3

        # we put computations in the triple and double root region at the end
        # as corrective features.

        ### COMPUTATIONS IN TRIPLE ROOT REGION
        # The critical point is known to be a triple root
        # Use logical or to include unknown triple points, but that should not happen
        # NOTE In this case, the roots are unavoidably equal
        region = triple_root_region | critical_point
        self.regions[0] = region
        if np.any(region):
            c2_ = c2[region]

            z = -c2_ / 3

            assert np.all(
                z > B[region]
            ), "Triple-roots violating the lower physical bound B detected."

            Z_L[region] = z
            Z_G[region] = z

        ### COMPUTATIONS IN DOUBLE ROOT REGION
        # The point A,B = 0 is known to be such a point
        region = double_root_region | zero_point
        self.regions[2] = region
        if np.any(region):
            r_ = r[region]
            q_ = q[region]
            c2_ = c2[region]

            u = 3 / 2 * q_ / r_

            z_1 = 2 * u - c2_ / 3
            z_23 = -u - c2_ / 3

            # to avoid indexing issues
            if isinstance(z_1, numbers.Real):
                z_1 = np.array([z_1])
                z_23 = np.array([z_23])

            # choose bigger root as gas like
            # theoretically they should strictly be different, otherwise it would be
            # the three root case
            double_is_bigger = z_23 > z_1

            # exchange values such that z_1 is the bigger root
            z_1_small = z_1[double_is_bigger]
            z_1[double_is_bigger] = z_23[double_is_bigger]
            z_23[double_is_bigger] = z_1_small

            Z_L[region] = z_23
            Z_G[region] = z_1

        # convert Jacobians to csr
        if isinstance(Z_L, pp.ad.AdArray):
            Z_L.jac = Z_L.jac.tocsr()
            if not Z_as_AD:
                Z_L = Z_L.val
        if isinstance(Z_G, pp.ad.AdArray):
            Z_G.jac = Z_G.jac.tocsr()
            if not Z_as_AD:
                Z_G = Z_G.val

        if self.gaslike:
            return Z_G, Z_L
        else:
            return Z_L, Z_G
