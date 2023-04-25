"""This module contains a class implementing the Peng-Robinson EoS for either
a liquid- or gas-like phase."""
from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.ad.operator_functions import NumericType

from .._core import R_IDEAL, MPa_kJ_SCALE
from .pr_component import PR_Component
from .pr_mixing import VdW_a, VdW_b, VdW_dT_a, VdW_dXi_a
from .pr_utils import A_CRIT, B_CRIT

__all__ = ["PR_EoS"]


def root_smoother(
    Z_L: pp.ad.Ad_array, Z_I: pp.ad.Ad_array, Z_G: pp.ad.Ad_array, s: float
) -> tuple[pp.ad.Ad_array, pp.ad.Ad_array]:
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
    proximity: np.ndarray = (Z_I.val - Z_L.val) / (Z_G.val - Z_L.val)

    # average intermediate and gas root for gas root smoothing
    W_G = (Z_I + Z_G) / 2
    # analogously for liquid root
    W_L = (Z_I + Z_L) / 2

    v_G = _gas_smoother(proximity, s)
    v_L = _liquid_smoother(proximity, s)

    # smoothing values with convex combination
    Z_G_val = (1 - v_G) * Z_G.val + v_G * W_G.val
    Z_L_val = (1 - v_L) * Z_L.val + v_L * W_L.val

    # smoothing jacobian with component-wise product
    # betweem matrix row and vector component
    Z_G_jac = Z_G.diagvec_mul_jac((1 - v_G)) + W_G.diagvec_mul_jac(v_G)
    Z_L_jac = Z_L.diagvec_mul_jac((1 - v_L)) + W_L.diagvec_mul_jac(v_L)

    # store in AD array format and return
    smooth_Z_L = pp.ad.Ad_array(Z_L_val, Z_L_jac.tocsr())
    smooth_Z_G = pp.ad.Ad_array(Z_G_val, Z_G_jac.tocsr())

    return smooth_Z_L, smooth_Z_G


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
    bound = np.logical_and(upper_bound, lower_bound)

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
    bound = np.logical_and(upper_bound, lower_bound)

    bound_smoother = (proximity[bound] - s) / s
    bound_smoother = (-1) * bound_smoother**2 * (3 - 2 * bound_smoother) + 1

    smoother[bound] = bound_smoother
    # where proximity is close to zero, set value of one
    smoother[proximity <= s] = 1.0

    return smoother


class PR_EoS:
    """A class implementing thermodynamic properties resulting from the Peng-Robinson
    equation of state

        ``p = R * T / (v - b) - a / (b**2 + 2 * v * b - b**2)``.

    Important:
        1. The method :meth:`compute` must be called first, by passing a
           thermodynamic state. The EoS is AD compatible, meaning the thermodynamic
           state can be passed in form of
           :class:`~porepy.numerics.ad.forward_mode.Ad_array` instances.

           Once this computation is done, the resulting thermodynamic properties can
           be accessed using respective *attributes* of this class.

           This is for performance reasons, since the computation of cohesion, covolume
           and compressibility factor can be done once for all properties.

        2. If one for some reason whishes to obtain the thermodynamic property
           as AD :class:`~porepy.numerics.ad.operators.Operator` instances, one
           can do so by wrapping various ``get_*`` methods into instances of
           i:class:`~porepy.numerics.ad.operator_functions.Function`.

           All methods are compatible with PorePy's
           :class:`~porepy.numerics.ad.forward_mode.Ad_array`.

           The compressibility factor ``Z`` must be provided in some form in this case.

           Be aware that in this case, every call to
           :meth:`~porepy.numerics.ad.operators.Operator.evaluate`
           will evaluate terms like cohesion everytime a thermodynamic property is
           evaluated. This is expensive.

        2. Calling any thermodynamic property with molar fractions ``*X`` as arguments
           raises errors if the number of passed fractions does not match the
           number of modelled components in :data:`components`.

           The order of fractions arguments must correspond to the order in
           :data:`components`.

    The list of properties which are computed during a call to :meth:`compute` includes:

    - :data:`a`
    - :data:`dT_a`
    - :data:`b`
    - :data:`A`
    - :data:`B`
    - :data:`Z`
    - :data:`rho`
    - :data:`h_dep`
    - :data:`phi`
    - :data:`kappa`
    - :data:`mu`

    Note:
        As of now, supercritical and solid phase are not supported.

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
        transition_smoothing: ``default=1e-4``

            A small number to determine proximity between 3-root and double-root case.

            See also `Vu et al. (2021), Section 6. <https://doi.org/10.1016/
            j.matcom.2021.07.015>`_ .
        eps: ``default=1e-16``

            A small number defining the numerical zero.

            Used for the computation of roots to distinguish between phase-regions.

    """

    def __init__(
        self,
        gaslike: bool,
        mixingrule: Literal["VdW"] = "VdW",
        smoothing_factor: float = 1e-4,
        eps: float = 1e-14,
    ) -> None:

        self.components: list[PR_Component] = []
        """A list containing all modelled components.

        Important:
            The order in this list is crutial for thermodynamic properties which
            have component fractions as argument.

            In all cases, the order in this list and the order of arguments ought to be
            the same.

        """

        self.is_supercritical: np.ndarray = np.array([], dtype=bool)
        """A boolean array flagging if the mixture became super-critical.

        In vectorized computations, the results are stored component-wise.

        Important:
            It is unclear, what the meaning of super-critical phases is using this EoS
            and values in this phase region should be used with suspicion.

        """

        self.is_extended: np.ndarray = np.array([], dtype=bool)
        """A boolean array flagging if where extended roots were used in the
        one-root-region.

        In vectorized computations, the results are stored component-wise.

        Important:
            Extended roots are no roots of the original compressibility polynomial.

        """

        self.gaslike: bool = bool(gaslike)
        """Flag passed at instantiation denoting the state of matter, liquid or gas."""

        assert mixingrule in ["VdW"], f"Unknown mixing rule {mixingrule}."
        self._mixingrule: str = mixingrule
        """The mixing rule passed at instantiation."""

        self.eps: float = eps
        """Passed at instantiation."""

        self.smoothing_factor: float = smoothing_factor
        """Passed at instantiation."""

        self.a: NumericType = 0.0
        """The cohesion using the assigned mixing rule."""

        self.dT_a: NumericType = 0.0
        """The derivative of the cohesion w.r.t. the temperatur."""

        self.b: NumericType = 0.0
        """The covolume using the assigned mixing rule."""

        self.A: NumericType = 0.0
        """The non-dimensional cohesion using the assigned mixing rule."""

        self.B: NumericType = 0.0
        """The non-dimensional cohesion using the assigned mixing rule."""

        self.Z: NumericType = 0.0
        """The compressibility factor of the EoS corresponding to the assigned label:
        gas- or liquid like."""

        self.rho: NumericType = 0.0
        """The molar density ``[mol / V]``."""

        self.h_dep: NumericType = 0.0
        """Specific departure enthalpy ``[kJ / mol / K]``."""

        self.phi: dict[PR_Component, NumericType] = dict()
        """Fugacity coefficients per present component (key)."""

        self.kappa: NumericType = 0.0
        """Thermal conductivity ``[W / m / K]``."""

        self.mu: NumericType = 0.0
        """Dynamic viscosity ``[mol / m / s]``."""

    def _num_frac_check(self, X) -> None:
        """Auxiliary method to check the number of passed fractions.
        Raises an error if number does not match the number of present components."""
        if len(X) != len(self.components):
            raise TypeError(
                f"{len(X)} fractions given but "
                + f"{len(self.components)} components present."
            )

    def compute(
        self,
        p: NumericType,
        T: NumericType,
        *X: NumericType,
        apply_smoother: bool = False,
    ) -> None:
        """Computes all thermodynamic properties based on the passed state.

        The results can be accessed using the respective instance properties.

        Warning:
            ``p``, ``T``, ``*X`` have a union type, meaning the results will be of
            the same. When mixing numpy arrays, porepy's Ad arrays and numbers,
            the user must make sure there will be no compatibility issues.

            This method is not supposed to be used with AD Operator instances.

        Parameters:
            p: Pressure
            T: Temperature
            *X: ``len=num_components``

                Fractions to be usd in the computation.
            apply_smoother: ``default=False``

                If True, a smoothing procedure is applied in the three-root-region,
                where the intermediate root approaches one of the other roots
                (see [3]).

                This is to be used **within** iterative procedures for numerical
                reasons. Once convergence is reached, the true roots should be computed
                without smoothing.

        """
        # sanity check
        self._num_frac_check(X)

        # cohesion and covolume, and derivative
        self.a = self.get_a(T, *X)
        self.b = self.get_b(*X)
        self.dT_a = self.get_dT_a(T, *X)
        # compute non-dimensional quantities
        self.A = (self.a * p) / (R_IDEAL**2 * T * T)
        self.B = (self.b * p) / (R_IDEAL * T)
        # root
        self.Z = self._Z(self.A, self.B, apply_smoother)
        # density
        self.rho = self.get_rho(p, T, self.Z)
        # departure enthalpy
        self.h_dep = self._h_dep(T, self.Z, self.a, self.dT_a, self.b, self.B)
        # fugacity per present component
        for i, comp_i in enumerate(self.components):
            B_i = comp_i.covolume * p / (R_IDEAL * T)
            A_i = self._get_a_i(T, list(X), i) * p / (R_IDEAL**2 * T * T)

            self.phi.update({comp_i: self._phi_i(self.Z, A_i, self.A, B_i, self.B)})

        # these two are open TODO
        self.kappa = self.get_kappa(p, T, self.Z)
        self.mu = self.get_mu(p, T, self.Z)

    def get_a(self, T: NumericType, *X: NumericType) -> NumericType:
        """Returns the cohesion term ``a`` using the assigned mixing rule.

        Parameters:
            p: Pressure
            T: Temperature
            *X: ``len=num_components``

                Fractions to be usd in the computation.

                For the mixing rule to be valid, the length must be equal to the number
                of components present in :data:`components`.

        Returns:
            The cohesion ``a`` for given thermodynamic state.

        """
        if self._mixingrule == "VdW":
            return VdW_a(T, list(X), self.components)
        else:
            raise ValueError(f"Unknown mixing rule {self._mixingrule}.")

    def get_dT_a(self, T: NumericType, *X: NumericType) -> NumericType:
        """Returns the derivative of the cohesion term ``a`` w.r.t. the temperature,
        using the assigned mixing rule.

        Parameters:
            T: Temperature
            *X: ``len=num_components``

                Fractions to be usd in the computation.

        Returns:
            The temperature-derivative of cohesion ``a`` for given thermodynamic state.

        """
        if self._mixingrule == "VdW":
            return VdW_dT_a(T, list(X), self.components)
        else:
            raise ValueError(f"Unknown mixing rule {self._mixingrule}.")

    def get_b(self, *X: NumericType) -> NumericType:
        """Returns the covolume term ``b`` using the assigned mixing rule.

        Parameters:
            *X: ``len=num_components``

                Fractions to be usd in the computation.

                For the mixing rule to be valid, the length must be equal to the number
                of components present in :data:`components`.

        Returns:
            The covolume ``b`` for given thermodynamic state.

        """
        if self._mixingrule == "VdW":
            return VdW_b(list(X), self.components)
        else:
            raise ValueError(f"Unknown mixing rule {self._mixingrule}.")

    def get_A(self, p: NumericType, T: NumericType, *X: NumericType) -> NumericType:
        """Returns the non-dimensional cohesion term ``A`` in the EoS

            ``A = (a * p) / (R_IDEAL**2 * T**2)``.

        Parameters:
            p: Pressure
            T: Temperature
            *X: ``len=num_components``

                Fractions to be usd in the computation.

                For the mixing rule to be valid, the length must be equal to the number
                of components present in :data:`components`.

        Returns:
            The non-dimensional cohesion ``A`` for given thermodynamic state.

        """
        return (self.get_a(T, *X) * p) / (R_IDEAL**2 * T * T)

    def get_B(self, p: NumericType, T: NumericType, *X: NumericType) -> NumericType:
        """Returns the non-dimensional covolume term ``B`` in the EoS

            `` B = (b * p) / (R_IDEAL * T)``.

        Parameters:
            p: Pressure
            T: Temperature
            *X: ``len=num_components``

                Fractions to be usd in the computation.

                For the mixing rule to be valid, the length must be equal to the number
                of components present in :data:`components`.

        Returns:
            The non-dimensional cohesion ``B`` for given thermodynamic state.

        """
        return (self.get_b(*X) * p) / (R_IDEAL * T)

    def get_Z(self, p: NumericType, T: NumericType, *X: NumericType) -> NumericType:
        """Computes the compressibility factor from scratch.

        Parameters:
            p: Pressure
            T: Temperature
            *X: ``len=num_components``

                Fractions to be usd in the computation.

        Returns:
            The compressibility factor ``Z`` for given thermodynamic state.

        """
        # get necessary properties
        A = self.get_A(p, T, *X)  # cohesion
        B = self.get_B(p, T, *X)  # covolume

        return self._Z(A, B)

    def _Z(
        self, A: pp.ad.Ad_array, B: pp.ad.Ad_array, apply_smoother: bool = False
    ) -> pp.ad.Ad_array:
        """Auxiliary method to compute the compressibility factor based on Cardano
        formulas.

        Warning:
            Due to the nature of the operations here, the input must be wrapped in
            Ad-arrays with vectors and matrices as val and jac, i.e. must be indexable.

        Parameters:
            A: Non-dimensional cohesion.
            B: Non-dimensional covolume.
            apply_smoother: ``default=False``

                Flag to apply smoothing procedure.

        Returns:
            The root ``Z`` corresponding to the assigned phase label.

        """
        # NOTE A and B must in theory be strictly positive
        # Numerically they can be NAN or non-positive.
        # this must be checked.
        # Non-positivity can happen due to numerical imprecision
        # (e.g. slightly negative fractions)
        # For now we try to work with it.

        # # Positivity check
        # assert np.all(A.val > 0.), "Cohesion A must be strictly positive"
        # assert np.all(B.val > 0.), "Covolume B must be strictly positive"

        # NAN check
        A_nan_vals = np.any(np.isnan(A.val))
        B_nan_vals = np.any(np.isnan(B.val))
        if A_nan_vals or B_nan_vals:
            raise ValueError(
                f"Discovered NANs in A ({A_nan_vals}) or B ({B_nan_vals})."
            )

        # the coefficients of the compressibility polynomial
        c0 = pp.ad.power(B, 3) + pp.ad.power(B, 2) - A * B
        c1 = A - 2 * B - 3 * pp.ad.power(B, 2)
        c2 = B - 1

        # the coefficients of the reduced polynomial (elimination of 2-monomial)
        r = c1 - pp.ad.power(c2, 2) / 3
        q = 2 / 27 * pp.ad.power(c2, 3) - c2 * c1 / 3 + c0

        # discriminant to determine the number of roots
        delta = pp.ad.power(q, 2) / 4 + pp.ad.power(r, 3) / 27

        # convert to more efficient format for region-wise slicing
        c2.jac = c2.jac.tolil()
        r.jac = r.jac.tolil()
        q.jac = q.jac.tolil()
        delta.jac = delta.jac.tolil()

        # prepare storage for root
        nc = len(delta.val)
        shape = delta.jac.shape

        Z_L = pp.ad.Ad_array(np.zeros(nc), sps.lil_matrix(shape))
        Z_G = pp.ad.Ad_array(np.zeros(nc), sps.lil_matrix(shape))

        # an indicater where the root is extended
        self.is_extended = np.zeros(nc, dtype=bool)

        ### CLASSIFYING REGIONS
        # identify super-critical line
        self.is_supercritical = B.val >= B_CRIT / A_CRIT * A.val
        # At A,B=0 we have 2 real roots, one with multiplicity 2
        zero_point = np.logical_and(
            np.isclose(A.val, 0, atol=self.eps), np.isclose(B.val, 0, atol=self.eps)
        )
        # The critical point is known to be a triple-point
        critical_point = np.logical_and(
            np.isclose(A.val, A_CRIT, rtol=0, atol=self.eps),
            np.isclose(B.val, B_CRIT, rtol=0, atol=self.eps),
        )
        # rectangle with upper right corner at (Ac,Bc) and lower left corner at 0
        # with tolerance
        acbc_rect = np.logical_and(
            np.logical_and(self.eps < A.val, A.val < A_CRIT - self.eps),
            np.logical_and(self.eps < B.val, B.val < B_CRIT - self.eps),
        )
        # subcritical triangle in the acbc rectangle
        subc_triang = np.logical_and(np.logical_not(self.is_supercritical), acbc_rect)

        # discriminant of zero indicates triple or two real roots with multiplicity
        degenerate_region = np.isclose(delta.val, 0.0, atol=self.eps)

        double_root_region = np.logical_and(degenerate_region, np.abs(r.val) > self.eps)
        triple_root_region = np.logical_and(
            degenerate_region, np.isclose(r.val, 0.0, atol=self.eps)
        )

        one_root_region = delta.val > self.eps
        three_root_region = delta.val < -self.eps

        # sanity check that every cell/case is covered
        assert np.all(
            np.logical_or.reduce(
                [
                    one_root_region,
                    triple_root_region,
                    double_root_region,
                    three_root_region,
                ]
            )
        ), "Uncovered cells/rows detected in PR root computation."

        # sanity check that the regions are mutually exclusive
        # this array must have 1 in every entry for the test to pass
        trues_per_row = np.vstack(
            [one_root_region, triple_root_region, double_root_region, three_root_region]
        ).sum(axis=0)
        trues_check = np.ones(nc, dtype=trues_per_row.dtype)
        assert np.all(
            trues_check == trues_per_row
        ), "Regions with different root scenarios overlap."

        ### COMPUTATIONS IN THE ONE-ROOT-REGION
        # Missing real root is replaced with conjugated imaginary roots
        if np.any(one_root_region):
            B_ = pp.ad.Ad_array(B.val[one_root_region], B.jac[one_root_region])
            B_ = pp.ad.Ad_array(r.val[one_root_region], r.jac[one_root_region])
            q_ = pp.ad.Ad_array(q.val[one_root_region], q.jac[one_root_region])
            delta_ = pp.ad.Ad_array(
                delta.val[one_root_region], delta.jac[one_root_region]
            )
            c2_ = pp.ad.Ad_array(c2.val[one_root_region], c2.jac[one_root_region])

            # delta has only positive values in this case by logic
            t = -q_ / 2 + pp.ad.sqrt(delta_)

            # t_1 might be negative, in this case we must choose the real cubic root
            # by extracting cbrt(-1), where -1 is the real cubic root.
            im_cube = t.val < 0.0
            if np.any(im_cube):
                t.val[im_cube] *= -1
                t.jac[im_cube] *= -1

                u = pp.ad.cbrt(t)

                u.val[im_cube] *= -1
                u.jac[im_cube] *= -1
            else:
                u = pp.ad.cbrt(t)

            # TODO In rare, un-physical areas of A,B, u can become zero,
            # causing infinity here, e.g.
            # A = 0.3620392380873223
            # B = -0.4204815080014268
            # this should never happen in physical simulations,
            # but I note it here nevertheless - VL
            real_part = u - B_ / (u * 3)
            z_1 = real_part - c2_ / 3  # Only real root, always greater than B

            # real part of the conjugate imaginary roots
            # used for extension of vanished roots
            w = (1 - B_ - z_1) / 2
            # w = -real_part / 2 - c2_1 / 3

            extension_is_bigger = z_1.val < w.val
            extension_is_smaller = np.logical_not(extension_is_bigger)

            nc_e = np.count_nonzero(one_root_region)
            small_root_val = np.zeros(nc_e)
            small_root_jac = sps.lil_matrix((nc_e, shape[1]))
            big_root_val = np.zeros(nc_e)
            big_root_jac = sps.lil_matrix((nc_e, shape[1]))

            # storring roots as they are
            big_root_val[extension_is_bigger] = w.val[extension_is_bigger]
            big_root_jac[extension_is_bigger] = w.jac[extension_is_bigger]
            big_root_val[extension_is_smaller] = z_1.val[extension_is_smaller]
            big_root_jac[extension_is_smaller] = z_1.jac[extension_is_smaller]

            small_root_val[extension_is_bigger] = z_1.val[extension_is_bigger]
            small_root_jac[extension_is_bigger] = z_1.jac[extension_is_bigger]
            small_root_val[extension_is_smaller] = w.val[extension_is_smaller]
            small_root_jac[extension_is_smaller] = w.jac[extension_is_smaller]

            Z_L.val[one_root_region] = small_root_val
            Z_L.jac[one_root_region] = small_root_jac

            Z_G.val[one_root_region] = big_root_val
            Z_G.jac[one_root_region] = big_root_jac

            # Store flag where the extended root was used
            if self.gaslike:
                self.is_extended[one_root_region] = extension_is_bigger
            else:
                self.is_extended[one_root_region] = extension_is_smaller

        ### COMPUTATIONS IN THE THREE-ROOT-REGION
        # compute all three roots, label them (smallest=liquid, biggest=gas)
        # optionally smooth them
        if np.any(three_root_region):
            B_ = pp.ad.Ad_array(B.val[three_root_region], B.jac[three_root_region])
            r_ = pp.ad.Ad_array(r.val[three_root_region], r.jac[three_root_region])
            q_ = pp.ad.Ad_array(q.val[three_root_region], q.jac[three_root_region])
            c2_ = pp.ad.Ad_array(c2.val[three_root_region], c2.jac[three_root_region])

            # compute roots in three-root-region using Cardano formula,
            # Casus Irreducibilis
            t_2 = pp.ad.arccos(-q_ / 2 * pp.ad.sqrt(-27 * pp.ad.power(r_, -3))) / 3
            t_1 = pp.ad.sqrt(-4 / 3 * r_)

            z3 = t_1 * pp.ad.cos(t_2) - c2_ / 3
            z2 = -t_1 * pp.ad.cos(t_2 + np.pi / 3) - c2_ / 3
            z1 = -t_1 * pp.ad.cos(t_2 - np.pi / 3) - c2_ / 3

            # Smoothing procedure only valid in the sub-critical area, where the three
            # roots are positive and bound from below by B
            smoothable = subc_triang[three_root_region]
            if apply_smoother and np.any(smoothable):
                z1_s, z3_s = root_smoother(z1, z2, z3, self.smoothing_factor)

                z3.val[smoothable] = z3_s.val[smoothable]
                z3.jac[smoothable] = z3_s.jac[smoothable]

                z1.val[smoothable] = z1_s.val[smoothable]
                z1.jac[smoothable] = z1_s.jac[smoothable]

            # assert roots are ordered by size
            assert np.all(z1.val <= z2.val) and np.all(
                z2.val <= z3.val
            ), "Roots in three-root-region improperly ordered."

            Z_L.val[three_root_region] = z1.val
            Z_L.jac[three_root_region] = z1.jac

            Z_G.val[three_root_region] = z3.val
            Z_G.jac[three_root_region] = z3.jac

        # we put computations in the triple and double root region at the end
        # as corrective features.

        ### COMPUTATIONS IN TRIPLE ROOT REGION
        # The critical point is known to be a triple root
        # Use logical or to include unknown triple points, but that should not happen
        # NOTE In the critical point, the roots are unavoidably equal
        region = np.logical_or(triple_root_region, critical_point)
        if np.any(region):
            c2_ = pp.ad.Ad_array(c2.val[region], c2.jac[region])

            z = -c2_ / 3

            assert np.all(
                B.val[region] < z.val
            ), "Triple-roots violating the lower physical bound B detected."

            Z_L.val[region] = z.val
            Z_L.jac[region] = z.jac

            Z_G.val[region] = z.val
            Z_G.jac[region] = z.jac

        ### COMPUTATIONS IN DOUBLE ROOT REGION
        # The point A,B = 0 is known to be such a point
        region = np.logical_or(double_root_region, zero_point)
        if np.any(region):
            B_ = pp.ad.Ad_array(B.val[region], B.jac[region])
            r_ = pp.ad.Ad_array(r.val[region], r.jac[region])
            q_ = pp.ad.Ad_array(q.val[region], q.jac[region])
            c2_ = pp.ad.Ad_array(c2.val[region], c2.jac[region])

            u = 3 / 2 * q_ / r_

            z_1 = 2 * u - c2_ / 3
            z_23 = -u - c2_ / 3

            # choose bigger root as gas like
            # theoretically they should strictly be different, otherwise it would be
            # the three root case
            double_is_bigger = z_23.val > z_1.val
            double_is_smaller = np.logical_not(double_is_bigger)

            # allocate storage for roots in this region
            nc_d = np.count_nonzero(region)
            small_root_val = np.zeros(nc_d)
            small_root_jac = sps.lil_matrix((nc_d, shape[1]))
            big_root_val = np.zeros(nc_d)
            big_root_jac = sps.lil_matrix((nc_d, shape[1]))

            # storring roots as they are
            big_root_val[double_is_bigger] = z_23.val[double_is_bigger]
            big_root_jac[double_is_bigger] = z_23.jac[double_is_bigger]
            big_root_val[double_is_smaller] = z_1.val[double_is_smaller]
            big_root_jac[double_is_smaller] = z_1.jac[double_is_smaller]

            small_root_val[double_is_bigger] = z_1.val[double_is_bigger]
            small_root_jac[double_is_bigger] = z_1.jac[double_is_bigger]
            small_root_val[double_is_smaller] = z_23.val[double_is_smaller]
            small_root_jac[double_is_smaller] = z_23.jac[double_is_smaller]

            Z_L.val[region] = small_root_val
            Z_L.jac[region] = small_root_jac

            Z_G.val[region] = big_root_val
            Z_G.jac[region] = big_root_jac

        # Correct the smaller root if it violates the lower bound B
        correction = Z_L.val <= B.val
        if np.any(correction):
            Z_L.val[correction] = B.val[correction] + self.eps
            Z_L.jac[correction] = B.jac[correction]  # + self.eps

        # assert physical meaningfulness
        assert np.all(
            Z_L.val > B.val
        ), "Liquid root violates lower physical bound given by covolume B."
        # assert gas root is bigger than liquid root
        assert np.all(
            Z_G.val >= Z_L.val
        ), "Liquid root violates upper physical bound given by gas root."

        # convert Jacobians to csr
        Z_L.jac = Z_L.jac.tocsr()
        Z_G.jac = Z_G.jac.tocsr()

        if self.gaslike:
            return Z_G
        else:
            return Z_L

    def get_rho(self, p: NumericType, T: NumericType, Z: NumericType) -> NumericType:
        """Computes the molar density from scratch.

            ``rho = p / (R_ideal * T * Z)``.

        Parameters:
            p: Pressure.
            T: Temperature.
            Z: Compressibility factor.

        Returns:
            The molar density ``rho`` from above equation.

        """
        # Scaling due to to p being in MPa and R in kJ / K / mol
        return p / (R_IDEAL * T * Z) * MPa_kJ_SCALE

    def get_h_dep(
        self, p: NumericType, T: NumericType, Z: NumericType, *X: NumericType
    ) -> NumericType:
        """Computes the specific departure enthalpy from scratch.

        See Also:
            [5], equation A-9

        Parameters:
            p: Pressure.
            T: Temperature.
            Z: Compressibility factor.

        Returns:
            The specific enthalpy departure value.

        """
        # sanity check
        self._num_frac_check(X)

        a = self.get_a(T, *X)
        dT_a = self.get_dT_a(T, *X)
        b = self.get_b(*X)
        B = self.get_B(p, T, *X)

        return self._h_dep(T, Z, a, dT_a, b, B)

    def _h_dep(
        self,
        T: NumericType,
        Z: NumericType,
        a: NumericType,
        dT_a: NumericType,
        b: NumericType,
        B: NumericType,
    ) -> NumericType:
        """Auxiliary function for computing the departure function."""
        return 1 / np.sqrt(8) * (T * dT_a - a) / b * pp.ad.log(
            (Z + (1 + np.sqrt(2)) * B) / (Z + (1 - np.sqrt(2)) * B)
        ) + R_IDEAL * T * (Z - 1)

    def get_phi(
        self, p: NumericType, T: NumericType, Z: NumericType, i: int, *X: NumericType
    ) -> NumericType:
        """Computes the fugacity coefficient for component ``i`` from scratch.

        See Also:
            [4], equation A-4

        Parameters:
            p: Pressure.
            T: Temperature.
            Z: Compressibility factor.
            i: Index of a component present in :data:`components`.
            *X: Component fractions.

        Returns:
            The specific enthalpy departure value.

        """
        B_i = self.components[i].covolume * p / (R_IDEAL * T)
        B = self.get_B(p, T, *X)
        A = self.get_A(T, *X)
        A_i = self._get_a_i(T, list(X), i) * p / (R_IDEAL**2 * T * T)

        return self._phi_i(Z, A_i, A, B_i, B)

    def _get_a_i(self, T: NumericType, X: list[NumericType], i: int) -> NumericType:
        """Auxiliary method to compute parts of the fugacity coefficients."""
        if self._mixingrule == "VdW":
            return VdW_dXi_a(T, X, self.components, i)
        else:
            raise ValueError(f"Unknown mixing rule {self._mixingrule}.")

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
            B_i / B * (Z - 1)
            - pp.ad.log(Z - B)
            - A
            / (B * np.sqrt(8))
            * (A_i / A - B_i / B)
            * pp.ad.log((Z + (1 + np.sqrt(2)) * B) / (Z + (1 - np.sqrt(2)) * B))
        )

        return pp.ad.exp(log_phi_i)

    # TODO
    def get_kappa(self, p: NumericType, T: NumericType, Z: NumericType) -> NumericType:
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
    def get_mu(self, p: NumericType, T: NumericType, Z: NumericType) -> NumericType:
        """
        Parameters:
            p: Pressure.
            T: Temperature.
            Z: Compressibility factor.

        Returns:
            The dynamic viscosity.

        """
        return 1.0
