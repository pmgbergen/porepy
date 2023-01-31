"""This module contains a class implementing the Peng-Robinson EoS for either
a liquid- or gas-like phase."""
from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.ad.operator_functions import NumericType

from .._composite_utils import R_IDEAL
from .pr_component import PR_Component
from .pr_mixing import VdW_a, VdW_b, VdW_dT_a, VdW_dXi_a
from .pr_utils import A_CRIT, B_CRIT

__all__ = ["PR_EoS"]


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
        transition_smoothing: ``default=0.1``

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
        smoothing_factor: float = 0.1,
        eps: float = 1e-16,
    ) -> None:

        self.components: list[PR_Component] = []
        """A list containing all modelled components.

        Important:
            The order in this list is crutial for thermodynamic properties which
            have component fractions as argument.

            In all cases, the order in this list and the order of arguments ought to be
            the same.

        """

        self._gaslike: bool = bool(gaslike)
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
        """The extended cohesion using the assigned mixing rule."""

        self.B: NumericType = 0.0
        """The extended cohesion using the assigned mixing rule."""

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
        # compute extended quantities
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
            b_i = comp_i.covolume
            a_i = self._get_a_i(T, list(X), i)

            self.phi.update(
                {comp_i: self._phi_i(T, self.Z, a_i, self.a, b_i, self.b, self.B)}
            )

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
        """Returns the extended cohesion term ``A`` in the EoS

            ``A = (a * p) / (R_IDEAL**2 * T**2)``.

        Parameters:
            p: Pressure
            T: Temperature
            *X: ``len=num_components``

                Fractions to be usd in the computation.

                For the mixing rule to be valid, the length must be equal to the number
                of components present in :data:`components`.

        Returns:
            The extended cohesion ``A`` for given thermodynamic state.

        """
        return (self.get_a(T, *X) * p) / (R_IDEAL**2 * T * T)

    def get_B(self, p: NumericType, T: NumericType, *X: NumericType) -> NumericType:
        """Returns the extended covolume term ``B`` in the EoS

            `` B = (b * p) / (R_IDEAL * T)``.

        Parameters:
            p: Pressure
            T: Temperature
            *X: ``len=num_components``

                Fractions to be usd in the computation.

                For the mixing rule to be valid, the length must be equal to the number
                of components present in :data:`components`.

        Returns:
            The extended cohesion ``B`` for given thermodynamic state.

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
        self, A: NumericType, B: NumericType, apply_smoother: bool = False
    ) -> NumericType:
        """Auxiliary method to compute the compressibility factor based on Cardano
        formulas

        Parameters:
            A: Extended cohesion.
            B: Extended covolume.
            apply_smoother: ``default=False``

                Flag to apply smoothing procedure.

        Returns:
            The root ``Z`` corresponding to the assigned phase label.

        """
        # For compatibility reasons we must broadcast A and B into Ad_arrays with 0-jacs
        # If the number of rows mismatch and numpy fails to broadcast them in the next
        # step, this is up to the user.
        if isinstance(A, np.ndarray):
            A = pp.ad.Ad_array(A)
        elif isinstance(A, pp.ad.Ad_array):
            pass  # keep as is
        else:
            # try to convert to float and wrap into array
            A = pp.ad.Ad_array(np.array([float(A)]))

        if isinstance(B, np.ndarray):
            B = pp.ad.Ad_array(B)
        elif isinstance(B, pp.ad.Ad_array):
            pass  # keep as is
        else:
            # try to convert to float and wrap into array
            B = pp.ad.Ad_array(np.array([float(B)]))

        assert isinstance(A, pp.ad.Ad_array) and isinstance(
            B, pp.ad.Ad_array
        ), "Type mismatch for A and B in root computation"

        # If this fails, everything else will faile
        try:
            broad_cast_test = A + B
        except Exception as err:
            raise TypeError(
                "Failed to broadcast 'A' and 'B' during root computation."
            ) from err

        nc = len(broad_cast_test.val)
        # If the Jacobian is trivial due to the broadcasting (equal 0)
        # then we simulate a single-column matrix to avoid indexing errors in the
        # following
        revert_broadcast = False
        if broad_cast_test.jac == 0:
            revert_broadcast = True
            shape = (len(broad_cast_test.val), 1)
            A.jac = sps.lil_matrix(shape)
            B.jac = sps.lil_matrix(shape)
        # else we assume its a sparse matrix
        else:
            shape = broad_cast_test.jac.shape

        # prepare storage for roots
        # storage for roots
        Z_L_val = np.zeros(nc)
        Z_G_val = np.zeros(nc)
        Z_L_jac = sps.lil_matrix(shape)
        Z_G_jac = sps.lil_matrix(shape)

        # the coefficients of the compressibility polynomial
        c0 = pp.ad.power(B, 3) + pp.ad.power(B, 2) - A * B
        c1 = A - 2 * B - 3 * pp.ad.power(B, 2)
        c2 = B - 1

        # the coefficients of the reduced polynomial (elimination of 2-monomial)
        r = c1 - pp.ad.power(c2, 2) / 3
        q = 2 / 27 * pp.ad.power(c2, 3) - c2 * c1 / 3 + c0

        # discriminant to determine the number of roots
        delta = pp.ad.power(q, 2) / 4 + pp.ad.power(r, 3) / 27

        # convert to more efficient format
        c2.jac = c2.jac.tolil()
        r.jac = r.jac.tolil()
        q.jac = q.jac.tolil()
        delta.jac = delta.jac.tolil()

        ### CLASSIFYING REGIONS
        # identify super-critical region
        is_super_critical = B.val > B_CRIT / A_CRIT * A.val

        # as of now, this is not covered
        if np.any(is_super_critical):
            raise NotImplementedError("Super-critical roots not available yet.")

        # discriminant of zero indicates one or two real roots with multiplicity
        degenerate_region = np.isclose(delta.val, 0.0, atol=self.eps)
        double_root_region = np.logical_and(degenerate_region, np.abs(r.val) > self.eps)
        triple_root_region = np.logical_and(
            degenerate_region, np.isclose(r.val, 0.0, atol=self.eps)
        )

        # ensure we are not in the uncovered two-real-root case or triple-root case
        if np.any(double_root_region):
            raise NotImplementedError("Case with two distinct real roots encountered.")
        if np.any(triple_root_region):
            raise NotImplementedError("Case with real triple-root encountered.")

        # physically covered cases
        one_root_region = delta.val > self.eps  # can only happen if p is positive
        three_root_region = delta.val < -self.eps  # can only happen if p is negative

        # sanity check that every cell/case covered (with above exclusions)
        if np.any(np.logical_not(np.logical_or(one_root_region, three_root_region))):
            raise NotImplementedError(
                "Uncovered cells/cases detected in PR root computation."
            )

        ### COMPUTATIONS IN THE ONE-ROOT-REGION
        # Missing real root is replaced with conjugated imaginary roots
        if np.any(one_root_region):
            B_1 = pp.ad.Ad_array(B.val[one_root_region], B.jac[one_root_region])
            r_1 = pp.ad.Ad_array(r.val[one_root_region], r.jac[one_root_region])
            q_1 = pp.ad.Ad_array(q.val[one_root_region], q.jac[one_root_region])
            delta_1 = pp.ad.Ad_array(
                delta.val[one_root_region], delta.jac[one_root_region]
            )
            c2_1 = pp.ad.Ad_array(c2.val[one_root_region], c2.jac[one_root_region])

            # delta has only positive values in this case by logic
            t_1 = -q_1 / 2 + pp.ad.sqrt(delta_1)

            # t_1 might be negative, in this case we must choose the real cubic root
            # by extracting cbrt(-1), where -1 is the real cubic root.
            im_cube = t_1.val < 0.0
            if np.any(im_cube):
                t_1.val[im_cube] *= -1
                t_1.jac[im_cube] *= -1

                u_1 = pp.ad.cbrt(t_1)

                u_1.val[im_cube] *= -1
                u_1.jac[im_cube] *= -1
            else:
                u_1 = pp.ad.cbrt(t_1)

            ## Relevant roots
            # only real root, Cardano formula, positive discriminant
            z_1 = u_1 - r_1 / (u_1 * 3) - c2_1 / 3
            # real part of the conjugate imaginary roots
            # used for extension of vanished roots
            w_1 = (1 - B_1 - z_1) / 2

            ## simplified labeling, Vu et al. (2021), equ. 4.24
            gas_region = w_1.val < z_1.val
            liquid_region = z_1.val < w_1.val

            # assert the roots are not overlapping,
            # this should not happen,
            # except in cases where the polynomial has "almost" a triple root
            assert not np.any(
                np.isclose(w_1.val - z_1.val, 0.0, atol=self.eps)
            ), "Triple root proximity detected in one-root-region."
            # assert the whole one-root-region is covered
            assert np.all(
                np.logical_or(gas_region, liquid_region)
            ), "Phase labeling does not cover whole one-root-region."
            # assert mutual exclusion to check sanity
            assert np.all(
                np.logical_not(np.logical_and(gas_region, liquid_region))
            ), "Labeled subregions in one-root-region overlap."

            # store values in one-root-region
            nc_1 = np.count_nonzero(one_root_region)
            Z_L_val_1 = np.zeros(nc_1)
            Z_G_val_1 = np.zeros(nc_1)
            Z_L_jac_1 = sps.csr_matrix((nc_1, shape[1]))
            Z_G_jac_1 = sps.csr_matrix((nc_1, shape[1]))

            # store gas root where actual gas, use extension where liquid
            Z_G_val_1[gas_region] = z_1.val[gas_region]
            Z_G_val_1[liquid_region] = w_1.val[liquid_region]
            Z_G_jac_1[gas_region] = z_1.jac[gas_region]
            Z_G_jac_1[liquid_region] = w_1.jac[liquid_region]
            # store liquid where actual liquid, use extension where gas
            Z_L_val_1[liquid_region] = z_1.val[liquid_region]
            Z_L_val_1[gas_region] = w_1.val[gas_region]
            Z_L_jac_1[liquid_region] = z_1.jac[liquid_region]
            Z_L_jac_1[gas_region] = w_1.jac[gas_region]

            # store values in global root structure
            Z_L_val[one_root_region] = Z_L_val_1
            Z_L_jac[one_root_region] = Z_L_jac_1
            Z_G_val[one_root_region] = Z_G_val_1
            Z_G_jac[one_root_region] = Z_G_jac_1

        ### COMPUTATIONS IN THE THREE-ROOT-REGION
        if np.any(three_root_region):
            r_3 = pp.ad.Ad_array(r.val[three_root_region], r.jac[three_root_region])
            q_3 = pp.ad.Ad_array(q.val[three_root_region], q.jac[three_root_region])
            c2_3 = pp.ad.Ad_array(c2.val[three_root_region], c2.jac[three_root_region])

            # compute roots in three-root-region using Cardano formula,
            # Casus Irreducibilis
            t_2 = pp.ad.arccos(-q_3 / 2 * pp.ad.sqrt(-27 * pp.ad.power(r_3, -3))) / 3
            t_1 = pp.ad.sqrt(-4 / 3 * r_3)

            z3_3 = t_1 * pp.ad.cos(t_2) - c2_3 / 3
            z2_3 = -t_1 * pp.ad.cos(t_2 + np.pi / 3) - c2_3 / 3
            z1_3 = -t_1 * pp.ad.cos(t_2 - np.pi / 3) - c2_3 / 3

            # assert roots are ordered by size
            assert np.all(z1_3.val <= z2_3.val) and np.all(
                z2_3.val <= z3_3.val
            ), "Roots in three-root-region improperly ordered."

            ## Smoothing of roots close to double-real-root case
            # this happens when the phase changes, at the phase border the polynomial
            # can have a real double root.
            if apply_smoother:
                Z_L_3, Z_G_3 = self._smoother(z1_3, z2_3, z3_3)
            else:
                Z_L_3, Z_G_3 = (z1_3, z3_3)

            ## Labeling in the three-root-region follows topological patterns
            # biggest root belongs to gas phase
            # smallest root belongs to liquid phase
            Z_L_val[three_root_region] = Z_L_3.val
            Z_L_jac[three_root_region] = Z_L_3.jac
            Z_G_val[three_root_region] = Z_G_3.val
            Z_G_jac[three_root_region] = Z_G_3.jac

        if self._gaslike:
            Z = pp.ad.Ad_array(Z_G_val, Z_G_jac.tocsr())
        else:
            Z = pp.ad.Ad_array(Z_L_val, Z_L_jac.tocsr())

        # revert broadcasting if it caused an unecessary jac
        if revert_broadcast:
            return Z.val
        else:
            return Z

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
        return p / (R_IDEAL * T * Z)

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
        return 1 / (b * np.sqrt(8)) * (T * dT_a - a) * pp.ad.log(
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
        b_i = self.components[i].covolume
        b = self.get_b(*X)
        B = self.get_B(p, T, *X)
        a = self.get_a(T, *X)
        a_i = self._get_a_i(T, list(X), i)

        return self._phi_i(T, Z, a_i, a, b_i, b, B)

    def _get_a_i(self, T: NumericType, X: list[NumericType], i: int) -> NumericType:
        """Auxiliary method to compute parts of the fugacity coefficients."""
        if self._mixingrule == "VdW":
            return VdW_dXi_a(T, X, self.components, i)
        else:
            raise ValueError(f"Unknown mixing rule {self._mixingrule}.")

    def _phi_i(
        self,
        T: NumericType,
        Z: NumericType,
        a_i: NumericType,
        a: NumericType,
        b_i: NumericType,
        b: NumericType,
        B: NumericType,
    ) -> NumericType:
        """Auxiliary method implementing the formula for the fugacity coefficient."""

        log_phi_i = (
            b_i / b * (Z - 1)
            - pp.ad.log(Z - B)
            - a
            / (b * R_IDEAL * T * np.sqrt(8))
            * (a_i / a - b_i / b)
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

    ### Smoothing method ---------------------------------------------------------------

    def _smoother(
        self, Z_L: pp.ad.Ad_array, Z_I: pp.ad.Ad_array, Z_G: pp.ad.Ad_array
    ) -> tuple[pp.ad.Ad_array, pp.ad.Ad_array]:
        """Smoothing procedure on boundaries of three-root-region.

        Smooths the value and Jacobian rows of the liquid and gas root close to
        phase boundaries.

        See Also:
            `Vu et al. (2021), Section 6.
            <https://doi.org/10.1016/j.matcom.2021.07.015>`_

        Parameters:
            Z_L: liquid-like root.
            Z_I: intermediate root.
            Z_G: gas-like root.

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

        v_G = self._gas_smoother(proximity)
        v_L = self._liquid_smoother(proximity)

        # smoothing values with convex combination
        Z_G_val = (1 - v_G) * Z_G.val + v_G * W_G.val
        Z_L_val = (1 - v_L) * Z_L.val + v_L * W_L.val
        # smoothing jacobian with component-wise product
        # betweem matrix row and vector component
        Z_G_jac = Z_G.jac.multiply((1 - v_G)[:, np.newaxis]) + W_G.jac.multiply(
            v_G[:, np.newaxis]
        )
        Z_L_jac = Z_L.jac.multiply((1 - v_L)[:, np.newaxis]) + W_L.jac.multiply(
            v_L[:, np.newaxis]
        )

        # store in AD array format and return
        smooth_Z_L = pp.ad.Ad_array(Z_L_val, Z_L_jac.tocsr())
        smooth_Z_G = pp.ad.Ad_array(Z_G_val, Z_G_jac.tocsr())

        return smooth_Z_L, smooth_Z_G

    def _gas_smoother(self, proximity: np.ndarray) -> np.ndarray:
        """Smoothing function for three-root-region where the intermediate root comes
        close to the gas root."""
        # smoother starts with zero values
        smoother = np.zeros(proximity.shape[0])
        # values around smoothing parameter are constructed according to Vu
        upper_bound = proximity < 1 - self.smoothing_factor
        lower_bound = (1 - 2 * self.smoothing_factor) < proximity
        bound = np.logical_and(upper_bound, lower_bound)
        bound_smoother = (
            proximity[bound] - (1 - 2 * self.smoothing_factor)
        ) / self.smoothing_factor
        bound_smoother = bound_smoother**2 * (3 - 2 * bound_smoother)
        smoother[bound] = bound_smoother
        # where proximity is close to one, set value of one
        smoother[proximity >= 1 - self.smoothing_factor] = 1.0

        return smoother

    def _liquid_smoother(self, proximity: np.ndarray) -> np.ndarray:
        """Smoothing function for three-root-region where the intermediate root comes
        close to the liquid root."""
        # smoother starts with zero values
        smoother = np.zeros(proximity.shape[0])
        # values around smoothing parameter are constructed according to Vu
        upper_bound = proximity < 2 * self.smoothing_factor
        lower_bound = self.smoothing_factor < proximity
        bound = np.logical_and(upper_bound, lower_bound)
        bound_smoother = (
            proximity[bound] + (1 - 2 * self.smoothing_factor)
        ) / self.smoothing_factor
        bound_smoother = (-1) * bound_smoother**2 * (3 - 2 * bound_smoother) + 1
        smoother[bound] = bound_smoother
        # where proximity is close to zero, set value of one
        smoother[proximity <= self.smoothing_factor] = 1.0

        return smoother
