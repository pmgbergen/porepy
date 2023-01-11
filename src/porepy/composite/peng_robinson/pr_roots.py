"""This module contains a class providing AD representations of (extended) roots of the
Peng-Robinson EoS.

The extension is performed according [1] and transferred into PorePy's AD framework.

References:
    [1]: `Ben Gharbia et al. (2021) <https://doi.org/10.1051/m2an/2021075>`_

"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sps

import porepy as pp

from .pr_utils import A_CRIT, B_CRIT, Leaf, _power


class PR_Roots:
    """A class for computing and providing access to roots of the Peng-Robinson EoS.

    The characteristic polynomial of the PR EoS is given by

        ``Z**3 + (B - 1) * Z**2 + (A - 2 * B - 3 * B**2) * Z + (B**3 + B**2 - A * B)``,

    where

        ``Z = p * v / (R * T)``,
        ``A = p * a / (R * T)**2``,
        ``B = P * b / (R * T)``.

    Note:
        The
        :class:`~porepy.composite.peng_robinson.pr_composition.PR_Composition` object
        whose
        :meth:`~porepy.composite.peng_robinson.pr_composition.PR_Composition.A` and
        :meth:`~porepy.composite.peng_robinson.pr_composition.PR_Composition.B`
        are passed during instantiation must have defined attraction and co-volume
        parameters
        (see
        :meth:`~porepy.composite.peng_robinson.pr_composition.PR_Composition.initialize`
        ).

    The roots are computed and labeled using :meth:`compute_roots`.

    Parameters:
        ad_system: The AD system for which the PR Composition class is created.
        A: An operator representing ``A`` in the PR EoS.
        B: An operator representing ``B`` in the PR EoS
        eps: ``default=1e-16``

            A small number defining the numerical zero. Used for the computation of
            roots to distinguish between phase-regions. Defaults to ``1e-16``.

    """

    def __init__(
        self,
        ad_system: pp.ad.ADSystem,
        A: pp.ad.Operator,
        B: pp.ad.Operator,
        eps: float = 1e-16,
    ) -> None:

        self.ad_system: pp.ad.ADSystem = ad_system
        """The AD system passed at instantiation."""

        self.A = A
        """``A`` passed at instantiation."""

        self.B = B
        """``B`` passed at instantiation."""

        self.is_super_critical: np.ndarray = np.array([], dtype=bool)
        """A boolean array indicating which cell is super-critical,
        after the roots of the EoS were computed."""

        self.transition_smoothing: float = 0.1
        """A small number to determine proximity between 3-root case and
        double-root case.

        Smoothing is applied according to below reference.

        See also:
            `Vu et al. (2021), Section 6.
            <https://doi.org/10.1016/j.matcom.2021.07.015>`_

        """

        self.liquid_root: Leaf = Leaf("PR Liquid Root")
        """An AD leaf-operator returning the AD array representing the liquid root
        computed using :meth:`compute_roots`."""

        self.gas_root: Leaf = Leaf("PR Gas Root")
        """An AD leaf-operator returning the AD array representing the gas root
        computed using :meth:`compute_roots`."""

        # self._z_l: pp.ad.Ad_array = pp.ad.Ad_array()
        # """AD representation of liquid root."""
        # self._z_g: pp.ad.Ad_array = pp.ad.Ad_array()
        # """AD representation of gaseous root."""

        self._eps = eps
        """``eps`` passed at instantiation."""

    ### EoS roots ----------------------------------------------------------------------

    # @property
    # def liquid_root(self) -> pp.ad.Ad_array:
    #     """An AD array representing (cell-wise) the extended root of the characteristic
    #     polynomial associated with the **liquid** phase.

    #     The AD framework provides information about the derivatives with respect to the
    #     thermodynamic state, i.e. the dependencies of the attraction and co-volume.

    #     """
    #     return self._z_l

    # @property
    # def gas_root(self) -> pp.ad.Ad_array:
    #     """AD representing (cell-wise) of the **gaseous** phase
    #     (see :meth:`liquid_root`)."""
    #     return self._z_g

    ### EoS parameters -----------------------------------------------------------------

    @property
    def c2(self) -> pp.ad.Operator:
        """An operator representing the coefficient of the monomial ``Z**2`` in the
        characteristic polynomial."""
        return self.B - 1

    @property
    def c1(self) -> pp.ad.Operator:
        """An operator representing the coefficient of the monomial ``Z`` in the
        characteristic polynomial."""
        return self.A - 3 * self.B * self.B - 2 * self.B

    @property
    def c0(self) -> pp.ad.Operator:
        """An operator representing the coefficient of the monomial ``Z**0`` in the
        characteristic polynomial."""
        return _power(self.B, pp.ad.Scalar(3)) + self.B * self.B - self.A * self.B

    @property
    def p(self) -> pp.ad.Operator:
        """An operator representing the coefficient of the monomial ``Z`` of the
        **reduced** characteristic polynomial."""
        return self.c1 - self.c2 * self.c2 / 3

    @property
    def q(self) -> pp.ad.Operator:
        """An operator representing the coefficient of the monomial ``Z**0`` of the
        **reduced** characteristic polynomial."""
        return (
            2 / 27 * _power(self.c2, pp.ad.Scalar(3)) - self.c2 * self.c1 / 3 + self.c0
        )

    @property
    def delta(self) -> pp.ad.Operator:
        """An AD representation of the discriminant of the characteristic polynomial,
        based on the current thermodynamic state.

        The sign of the discriminant can be used to distinguish between 1 and 2-phase
        regions:

        - ``> 0``: single-phase region (1 real root)
        - ``< 0``: 2-phase region (3 distinct real roots)
        - ``= 0``: special region (1 or 2 distinct real roots with respective
          multiplicity)

        This holds for the values of the AD array.

        Note:
            The definition of this discriminant deviates from the standard definition,
            including the sign convention.
            This is due to the discriminant of the reduced polynomial being used.

        Warning:
            The physical meaning of the case with two distinct real roots is
            unclear as of now.

        """
        return self.q * self.q / 4 + _power(self.p, pp.ad.Scalar(3)) / 27

    ### root computation ---------------------------------------------------------------

    def compute_roots(self) -> None:
        """Computes the roots of the characteristic polynomial and assigns phase labels.

        The roots depend on ``A`` and ``B`` of the Peng-Robinson EoS, hence on the
        thermodynamic state. They must be re-computed after any change in pressure,
        temperature and composition.
        This holds especially in iterative schemes.

        The results can be accessed (by reference) using :meth:`liquid_root` and
        :meth:`gas_root`.

        """
        # evaluate necessary parameters
        A = self.A.evaluate(self.ad_system.dof_manager)
        B = self.B.evaluate(self.ad_system.dof_manager)
        p = self.p.evaluate(self.ad_system.dof_manager)
        q = self.q.evaluate(self.ad_system.dof_manager)
        c2 = self.c2.evaluate(self.ad_system.dof_manager)
        delta = self.delta.evaluate(self.ad_system.dof_manager)

        nc = len(A.val)
        shape = A.jac.shape
        # storage for roots
        Z_L_val = np.zeros(nc)
        Z_G_val = np.zeros(nc)
        Z_L_jac = sps.csr_matrix(shape)
        Z_G_jac = sps.csr_matrix(shape)

        # identify super-critical region
        self.is_super_critical = B.val > B_CRIT / A_CRIT * A.val
        # sub_critical = np.logical_not(self.is_super_critical)

        # as of now, this is not covered
        if np.any(self.is_super_critical):
            raise NotImplementedError("Super-critical roots not available yet.")

        # discriminant of zero indicates one or two real roots with multiplicity
        degenerate_region = np.isclose(delta.val, 0.0, atol=self._eps)
        double_root_region = np.logical_and(
            degenerate_region, np.abs(p.val) > self._eps
        )
        triple_root_region = np.logical_and(
            degenerate_region, np.isclose(p.val, 0.0, atol=self._eps)
        )

        # ensure we are not in the uncovered two-real-root case
        if np.any(double_root_region):
            raise NotImplementedError("Case with two distinct real roots encountered.")
        # check for single triple root. Not sure what to do in this case...
        if np.any(triple_root_region):
            raise NotImplementedError("Case with real triple-root encountered.")

        # physically covered cases
        one_root_region = delta.val > self._eps  # can only happen if p is positive
        three_root_region = delta.val < -self._eps  # can only happen if p is negative

        # sanity check that every cell/case covered (with above exclusions)
        if np.any(np.logical_not(np.logical_or(one_root_region, three_root_region))):
            raise NotImplementedError(
                "Uncovered cells/cases detected in PR root computation."
            )

        ### compute the one real root and the extended root from
        # conjugated imaginary roots
        if np.any(one_root_region):
            B_1 = pp.ad.Ad_array(B.val[one_root_region], B.jac[one_root_region])
            p_1 = pp.ad.Ad_array(p.val[one_root_region], p.jac[one_root_region])
            q_1 = pp.ad.Ad_array(q.val[one_root_region], q.jac[one_root_region])
            delta_1 = pp.ad.Ad_array(
                delta.val[one_root_region], delta.jac[one_root_region]
            )
            c2_1 = pp.ad.Ad_array(c2.val[one_root_region], c2.jac[one_root_region])
            # delta has only positive values in this case by logic
            t_1 = -q_1 / 2 + pp.ad.sqrt(delta_1)

            # t_1 should only be positive, since delta positive and greater than q
            # assert above, because cubic root is imaginary otherwise
            assert np.all(
                t_1.val > 0.0
            ), "Real root in one-root-region has imaginary parts."
            u_1 = pp.ad.cbrt(t_1)

            ## Relevant roots
            # only real root, Cardano formula, positive discriminant
            z_1 = u_1 - p_1 / (u_1 * 3) - c2_1 / 3
            # real part of the conjugate imaginary roots
            # used for extension of vanished roots
            r_1 = (1 - B_1 - z_1) / 2

            ## PHASE LABELING in one-root-region
            # this has to hold, otherwise the labeling polynomial can't have 3 distinct
            # roots according to Gharbia et al. (2021)

            # assert np.all(
            #     B_1.val < B_CRIT
            # ), "Co-volume exceeds critical value for labeling."

            # A_1 = pp.ad.Ad_array(A.val[one_root_region], A.jac[one_root_region])
            # liquid_region, gas_region = self._get_labeled_regions(
            #     A_1,
            #     B_1,
            #     one_real_root_region
            # )

            ## simplified labeling, Vu et al. (2021), equ. 4.24
            gas_region = r_1.val < z_1.val
            liquid_region = z_1.val < r_1.val

            # assert the roots are not overlapping,
            # this should not happen,
            # except in cases where the polynomial has "almost" a triple root
            assert not np.any(
                np.isclose(r_1.val - z_1.val, 0.0, atol=self._eps)
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
            Z_G_val_1[liquid_region] = r_1.val[liquid_region]
            Z_G_jac_1[gas_region] = z_1.jac[gas_region]
            Z_G_jac_1[liquid_region] = r_1.jac[liquid_region]
            # store liquid where actual liquid, use extension where gas
            Z_L_val_1[liquid_region] = z_1.val[liquid_region]
            Z_L_val_1[gas_region] = r_1.val[gas_region]
            Z_L_jac_1[liquid_region] = z_1.jac[liquid_region]
            Z_L_jac_1[gas_region] = r_1.jac[gas_region]

            # store values in global root structure
            Z_L_val[one_root_region] = Z_L_val_1
            Z_L_jac[one_root_region] = Z_L_jac_1
            Z_G_val[one_root_region] = Z_G_val_1
            Z_G_jac[one_root_region] = Z_G_jac_1

        ### compute the two relevant roots in the three root region
        if np.any(three_root_region):
            p_3 = pp.ad.Ad_array(p.val[three_root_region], p.jac[three_root_region])
            q_3 = pp.ad.Ad_array(q.val[three_root_region], q.jac[three_root_region])
            c2_3 = pp.ad.Ad_array(c2.val[three_root_region], c2.jac[three_root_region])

            # compute roots in three-root-region using Cardano formula,
            # Casus Irreducibilis
            t_2 = pp.ad.arccos(-q_3 / 2 * pp.ad.sqrt(-27 * pp.ad.power(p_3, -3))) / 3
            t_1 = pp.ad.sqrt(-4 / 3 * p_3)

            z3_3 = t_1 * pp.ad.cos(t_2) - c2_3 / 3
            z2_3 = -t_1 * pp.ad.cos(t_2 + np.pi / 3) - c2_3 / 3
            z1_3 = -t_1 * pp.ad.cos(t_2 - np.pi / 3) - c2_3 / 3

            # assert roots are ordered by size
            assert np.all(
                z1_3.val <= z2_3.val <= z3_3.val
            ), "Roots in three-root-region improperly ordered."

            ## Smoothing of roots close to double-real-root case
            # this happens when the phase changes, at the phase border the polynomial
            # can have a real double root.
            Z_L_3, Z_G_3 = self._smoother(z1_3, z2_3, z3_3)

            ## Labeling in the three-root-region follows topological patterns
            # biggest root belongs to gas phase
            # smallest root belongs to liquid phase
            Z_L_val[three_root_region] = Z_L_3.val
            Z_L_jac[three_root_region] = Z_L_3.jac
            Z_G_val[three_root_region] = Z_G_3.val
            Z_G_jac[three_root_region] = Z_G_3.jac

        ### storing results for access
        # self._z_l = pp.ad.Ad_array(Z_L_val, Z_L_jac)
        # self._z_g = pp.ad.Ad_array(Z_G_val, Z_G_jac)
        self.liquid_root.value = pp.ad.Ad_array(Z_L_val, Z_L_jac)
        self.gas_root.value = pp.ad.Ad_array(Z_G_val, Z_G_jac)

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
        smooth_Z_L = pp.ad.Ad_array(Z_G_val, Z_G_jac.tocsr())
        smooth_Z_G = pp.ad.Ad_array(Z_L_val, Z_L_jac.tocsr())

        return smooth_Z_L, smooth_Z_G

    def _gas_smoother(self, proximity: np.ndarray) -> np.ndarray:
        """Smoothing function for three-root-region where the intermediate root comes
        close to the gas root."""
        # smoother starts with zero values
        smoother = np.zeros(proximity.shape[0])
        # values around smoothing parameter are constructed according to Vu
        upper_bound = proximity < 1 - self.transition_smoothing
        lower_bound = (1 - 2 * self.transition_smoothing) < proximity
        bound = np.logical_and(upper_bound, lower_bound)
        bound_smoother = (
            proximity[bound] - (1 - 2 * self.transition_smoothing)
        ) / self.transition_smoothing
        bound_smoother = bound_smoother**2 * (3 - 2 * bound_smoother)
        smoother[bound] = bound_smoother
        # where proximity is close to one, set value of one
        smoother[proximity >= 1 - self.transition_smoothing] = 1.0

        return smoother

    def _liquid_smoother(self, proximity: np.ndarray) -> np.ndarray:
        """Smoothing function for three-root-region where the intermediate root comes
        close to the liquid root."""
        # smoother starts with zero values
        smoother = np.zeros(proximity.shape[0])
        # values around smoothing parameter are constructed according to Vu
        upper_bound = proximity < 2 * self.transition_smoothing
        lower_bound = self.transition_smoothing < proximity
        bound = np.logical_and(upper_bound, lower_bound)
        bound_smoother = (
            proximity[bound] + (1 - 2 * self.transition_smoothing)
        ) / self.transition_smoothing
        bound_smoother = (-1) * bound_smoother**2 * (3 - 2 * bound_smoother) + 1
        smoother[bound] = bound_smoother
        # where proximity is close to zero, set value of one
        smoother[proximity <= self.transition_smoothing] = 1.0

        return smoother

    def _get_labeled_regions(
        self,
        A_1: pp.ad.Ad_array,
        B_1: pp.ad.Ad_array,
        one_root_region: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns labeled region indicators based on a cubic polynomial.

        A cubic polynomial in A with coefficients dependent on B
        (here denoted as labeling polynomial) has always three roots in the
        sub-critical area.
        Intermediate and largest root are used to determine the phase label.
        Gharbia et al. (2021)

        Parameters:
            A_1: ``A`` in the EoS reduced to the one-real-root-region.
            B_1: ``B`` in the EoS reduced to the one-real-root-region.
            one_root_region: ``dtype=bool``

                A boolean array indicating the one-real-root-region.

        Returns:
            A tuple containing two boolean arrays, where the first one indicates the
            liquid region, the second one the gas region.
            The regions are mutually exclusive.

        """
        # coefficients for the labeling polynomial
        c2_label = (2 * self.B * self.B - 10 * self.B - 1 / 4).evaluate(
            self.ad_system.dof_manager
        )
        c1_label = (
            -4 * _power(self.B, pp.ad.Scalar(4))
            + 28 * _power(self.B, pp.ad.Scalar(3))
            + 22 * self.B * self.B
            + 2 * self.B
        ).evaluate(self.ad_system.dof_manager)
        c0_label = (
            -8 * _power(self.B, pp.ad.Scalar(6))
            - 32 * _power(self.B, pp.ad.Scalar(5))
            - 40 * _power(self.B, pp.ad.Scalar(4))
            - 26 * _power(self.B, pp.ad.Scalar(3))
            - 2 * self.B * self.B
        ).evaluate(self.ad_system.dof_manager)
        # reduce to relevant region
        c2_label = pp.ad.Ad_array(
            c2_label.val[one_root_region], c2_label.jac[one_root_region]
        )
        c1_label = pp.ad.Ad_array(
            c1_label.val[one_root_region], c1_label.jac[one_root_region]
        )
        c0_label = pp.ad.Ad_array(
            c0_label.val[one_root_region], c0_label.jac[one_root_region]
        )

        # reduced coefficients for the labeling polynomial
        p_label: pp.ad.Ad_array = c1_label - c2_label * c2_label / 3
        q_label: pp.ad.Ad_array = (
            2 / 27 * pp.ad.power(c2_label, 3) - c2_label * c1_label / 3 + c0_label
        )
        # discriminant of the labeling polynomial
        delta_label = q_label * q_label / 4 + pp.ad.power(p_label, 3) / 27
        # assert labeling polynomial has three distinct real roots
        # this should always hold according to Gharbia,
        assert np.all(
            delta_label.val < -self._eps
        ), "Labeling polynomial has less than 3 distinct real roots."

        # compute labeling roots using Cardano formula, Casus Irreducibilis
        t_2 = (
            pp.ad.arccos(-q_label / 2 * pp.ad.sqrt(-27 * pp.ad.power(p_label, -3))) / 3
        )
        t_1 = pp.ad.sqrt(-4 / 3 * p_label)
        AL_1 = t_1 * pp.ad.cos(t_2) - c2_label / 3
        AG_1 = -t_1 * pp.ad.cos(t_2 + np.pi / 3) - c2_label / 3
        A0_1 = -t_1 * pp.ad.cos(t_2 - np.pi / 3) - c2_label / 3

        # assert roots are ordered by size
        assert np.all(
            A0_1.val <= AG_1.val <= AL_1.val
        ), "Labeling roots improperly ordered."

        # compute criteria for phase labels
        gas_region = np.logical_and(
            0 < B_1.val < B_CRIT,
            A_CRIT / B_CRIT * B_1.val < A_1.val < AG_1.val,
        )
        liquid_region = np.logical_or(
            np.logical_and(0 < B_1.val < B_CRIT, AL_1.val < A_1.val),
            np.logical_and(B_CRIT < B_1.val, A_CRIT / B_CRIT * B_1.val < A_1.val),
        )

        return liquid_region, gas_region
