"""This module contains a class providing AD representations of (extended) roots of the
Peng-Robinson EoS. The extension is performed according [1] and transferred into PorePy's AD
framework.

References:
    [1]: `Ben Gharbia et al. (2021) <https://doi.org/10.1051/m2an/2021075>`_

"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sps

import porepy as pp

from .pr_utils import A_CRIT, B_CRIT

ad_power = pp.ad.Function(pp.ad.power, "power")
cos = pp.ad.Function(pp.ad.cos, "cos")
arccos = pp.ad.Function(pp.ad.arccos, "arccos")


class PR_Roots:
    """A class for computing and providing access to roots of the Peng-Robinson EoS.

    The characteristic polynomial of the PR EoS is given by

        ``Z**3 + (B - 1) * Z**2 + (A - 2 * B - 3 * B**2) * Z + (B**3 + B**2 - A * B) = 0``,

    where

        ``Z = p * v / (R * T)``,
        ``A = p * a / (R * T)**2``,
        ``B = P * b / (R * T)``.

    Note:
        The :class:`~porepy.composite.peng_robinson.pr_composition.PR_Composition` object whose
        :meth:`~porepy.composite.peng_robinson.pr_composition.PR_Composition.A` and
        :meth:`~porepy.composite.peng_robinson.pr_composition.PR_Composition.B`
        are passed during instantiation must have defined attraction and co-volume parameters
        (see
        :meth:`~porepy.composite.peng_robinson.pr_composition.PR_Composition.initialize`).

    The roots are computed and labeled using :meth:`compute_roots`.

    Parameters:
        ad_system: The AD system for which the PR Composition class is created.
        A: An operator representing ``A`` in the PR EoS.
        B: An operator representing ``B`` in the PR EoS
        eps (optional): a small number defining the numerical zero. Used for the computation of
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
        """A boolean array indicating which cell is super-critical, after the roots of the EoS
        were computed."""

        self._z_l: pp.ad.Ad_array = pp.ad.Ad_array()
        """AD representation of liquid root."""
        self._z_g: pp.ad.Ad_array = pp.ad.Ad_array()
        """AD representation of gaseous root."""

        self._eps = eps
        """``eps`` passed at instantiation."""

    ### EoS roots -----------------------------------------------------------------------------

    @property
    def liquid_root(self) -> pp.ad.Ad_array:
        """An AD array representing (cell-wise) the extended root of the characteristic
        polynomial associated with the **liquid** phase.

        The AD framework provides information about the derivatives with respect to the
        thermodynamic state, i.e. the dependencies of the attraction and co-volume.

        """
        return self._z_l

    @property
    def gas_root(self) -> pp.ad.Ad_array:
        """AD representing (cell-wise) of the **gaseous** phase (see :meth:`liquid_root`)."""
        return self._z_g

    ### EoS parameters ------------------------------------------------------------------------

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
        return ad_power(self.B, pp.ad.Scalar(3)) + self.B * self.B - self.A * self.B

    @property
    def p(self) -> pp.ad.Operator:
        """An operator representing the coefficient of the monomial ``Z`` of the **reduced**
        characteristic polynomial."""
        return self.c1 - self.c2 * self.c2 / 3

    @property
    def q(self) -> pp.ad.Operator:
        """An operator representing the coefficient of the monomial ``Z**0`` of the **reduced**
        characteristic polynomial."""
        return (
            2 / 27 * ad_power(self.c2, pp.ad.Scalar(3))
            - self.c2 * self.c1 / 3
            + self.c0
        )

    @property
    def delta(self) -> pp.ad.Operator:
        """An AD representation of the discriminant of the characteristic polynomial, based
        on the current thermodynamic state.

        The sign of the discriminant can be used to distinguish between 1 and 2-phase regions:

        - ``> 0``: single-phase region (1 real root)
        - ``< 0``: 2-phase region (3 distinct real roots)
        - ``= 0``: special region (1 or 2 distinct real roots with respective multiplicity)

        This holds for the values of the AD array.

        Note:
            The definition of this discriminant deviates from the standard definition,
            including the sign convention. This is due to the discriminant of the reduced
            polynomial being used.

        Warning:
            The physical meaning of the case with two distinct real roots is
            unclear as of now.

        """
        return self.q * self.q / 4 + ad_power(self.p, pp.ad.Scalar(3)) / 27

    ### root computation ----------------------------------------------------------------------

    def compute_roots(self) -> None:
        """Computes the roots of the characteristic polynomial and assigns phase labels.

        The roots depend on ``A`` and ``B`` of the Peng-Robinson EoS, hence on the
        thermodynamic state. They must be re-computed after any change in pressure, temperature
        and composition. This holds especially in iterative schemes.

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
        sub_critical = np.logical_not(self.is_super_critical)

        # as of now, this is not covered
        if np.any(self.is_super_critical):
            raise NotImplementedError("Super-critical roots not available yet.")

        # discriminant of zero indicates one or two real roots with multiplicity
        degenerate_region = np.isclose(delta.val, 0.0, atol=self._eps)
        two_real_root_region = np.logical_and(
            degenerate_region, np.abs(p.val) > self._eps
        )
        triple_root_region = np.logical_and(
            degenerate_region, np.isclose(p.val, 0.0, atol=self._eps)
        )

        # ensure we are not in the uncovered two-real-root case
        if np.any(two_real_root_region):
            raise NotImplementedError("Case with two distinct real roots encountered.")
        # check for single triple root. Not sure what to do in this case...
        if np.any(triple_root_region):
            raise NotImplementedError("Case with real triple-root encountered.")

        # physically covered cases
        one_real_root_region = delta.val > self._eps  # can only happen if p is positive
        three_real_root_region = (
            delta.val < -self._eps
        )  # can only happen if p is negative

        # last sanity check we have every cell/case covered (with above exclusions)
        if np.any(
            np.logical_not(np.logical_or(one_real_root_region, three_real_root_region))
        ):
            raise NotImplementedError(
                "Uncovered cells/cases detected in PR root computation."
            )

        ### compute the one real root and the extended root from the conjugated imaginary roots
        if np.any(one_real_root_region):
            B_1 = pp.ad.Ad_array(
                B.val[one_real_root_region], B.jac[one_real_root_region]
            )
            A_1 = pp.ad.Ad_array(
                A.val[one_real_root_region], A.jac[one_real_root_region]
            )
            p_1 = pp.ad.Ad_array(
                p.val[one_real_root_region], p.jac[one_real_root_region]
            )
            q_1 = pp.ad.Ad_array(
                q.val[one_real_root_region], q.jac[one_real_root_region]
            )
            delta_1 = pp.ad.Ad_array(
                delta.val[one_real_root_region], delta.jac[one_real_root_region]
            )
            c2_1 = pp.ad.Ad_array(
                c2.val[one_real_root_region], c2.jac[one_real_root_region]
            )
            # delta has only positive values in this case by logic
            t_1 = -q_1 / 2 + pp.ad.sqrt(delta_1)

            # t_1 should only be positive, since delta positive and greater than q
            # assert above, because cubic root is imaginary otherwise
            assert np.all(
                t_1.val > 0.0
            ), "Real root in one-root-region has imaginary parts."
            u_1 = pp.ad.cbrt(t_1)

            # only real root, Cardano formula, positive discriminant
            z_1 = u_1 - p_1 / (u_1 * 3) - c2_1 / 3
            # real part of the conjugate imaginary roots
            r_1 = (1 - B_1 - z_1) / 2

            ## PHASE LABELING in one-root-region, Gharbia et al. (2021)
            # A cubic polynomial in A with coefficients dependent on B
            # (here denoted as labeling polynomial) has always three roots in the sub-critical
            # area. Intermediate and largest root are used to determine the phase label.
            assert np.all(
                B_1.val < B_CRIT
            ), "Co-volume exceeds critical value for labeling."

            # coefficients for the labeling polynomial
            c2_label = (2 * self.B * self.B - 10 * self.B - 1 / 4).evaluate(
                self.ad_system.dof_manager
            )
            c1_label = (
                -4 * ad_power(self.B, pp.ad.Scalar(4))
                + 28 * ad_power(self.B, pp.ad.Scalar(3))
                + 22 * self.B * self.B
                + 2 * self.B
            ).evaluate(self.ad_system.dof_manager)
            c0_label = (
                -8 * ad_power(self.B, pp.ad.Scalar(6))
                - 32 * ad_power(self.B, pp.ad.Scalar(5))
                - 40 * ad_power(self.B, pp.ad.Scalar(3))
                - 26 * ad_power(self.B, pp.ad.Scalar(3))
                - 2 * self.B * self.B
            ).evaluate(self.ad_system.dof_manager)
            # reduce to relevant region
            c2_label = pp.ad.Ad_array(
                c2_label.val[one_real_root_region], c2_label.jac[one_real_root_region]
            )
            c1_label = pp.ad.Ad_array(
                c1_label.val[one_real_root_region], c1_label.jac[one_real_root_region]
            )
            c0_label = pp.ad.Ad_array(
                c0_label.val[one_real_root_region], c0_label.jac[one_real_root_region]
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
                pp.ad.arccos(-q_label / 2 * pp.ad.sqrt(-27 / pp.ad.power(p_label, 3)))
                / 3
            )
            t_1 = pp.ad.sqrt(-4 * p_label / 3)
            A0_1 = t_1 * pp.ad.cos(t_2) - c2_label / 3
            AG_1 = -t_1 * pp.ad.cos(t_2 + np.pi / 3) - c2_label / 3
            AL_1 = -t_1 * pp.ad.cos(t_2 - np.pi / 3) - c2_label / 3

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

            # assert the whole one-root-region is covered
            assert np.all(
                np.logical_or(gas_region, liquid_region)
            ), "Phase labeling does not cover whole one-root-region"

        ### compute the two relevant roots in the three root region
        if np.any(three_real_root_region):
            p_3 = p[three_real_root_region]
            q_3 = q[three_real_root_region]
            c2_3 = pp.ad.Ad_array(
                c2.val[three_real_root_region], c2.jac[three_real_root_region]
            )

            # compute roots in three-root-region using Cardano formula, Casus Irreducibilis
            t_2 = pp.ad.arccos(-q_3 / 2 * pp.ad.sqrt(-27 / pp.ad.power(p_3, 3))) / 3
            t_1 = pp.ad.sqrt(-4 * p_3 / 3)

            z1_3 = t_1 * pp.ad.cos(t_2) - c2_3 / 3
            z2_3 = -t_1 * pp.ad.cos(t_2 + np.pi / 3) - c2_3 / 3
            z3_3 = -t_1 * pp.ad.cos(t_2 - np.pi / 3) - c2_3 / 3

            # assert roots are ordered by size
            assert np.all(
                z1_3.val <= z2_3.val <= z3_3.val
            ), "Roots in three-root-region improperly ordered"

            ## Labeling in the three-root-region follows topological patterns
            # biggest root belongs to gas phase
            # smallest root belongs to liquid phase
