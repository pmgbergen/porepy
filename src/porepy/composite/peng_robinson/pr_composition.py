"""This module contains a composition class for the Peng-Robinson equation of state.

As of now, it supports a liquid and a gas phase, and several modelled components.

The formulation is thermodynamically consistent. The PR composition creates and assigns
thermodynamic properties of phases, based on the roots of the cubic polynomial and
added components.

For the equilibrium equations, formulae for fugacity values for each component in each
phase are implemented.

This framework is highly non-linear and active research code.

References:
    [1]: `Ben Gharbia et al. (2021) <https://doi.org/10.1051/m2an/2021075>`_
    [2]: `Vu et al. (2021) <https://doi.org/10.1016/j.matcom.2021.07.015>`_

"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import scipy.sparse as sps

import porepy as pp

from .._composite_utils import R_IDEAL
from ..composition import Composition
from .pr_bip import get_PR_BIP
from .pr_component import PR_Component
from .pr_mixing import VdW_a_ij, dT_VdW_a_ij
from .pr_phase import PR_Phase
from .pr_utils import A_CRIT, B_CRIT, _power

__all__ = ["PR_Composition"]


class PR_Composition(Composition):
    """A composition modelled using the Peng-Robinson equation of state.

        ``p = R * T / (v - b) - a / (b**2 + 2 * v * b - b**2)``.

    Van der Waals mixing rules are applied to ``a`` and ``b``.

    Note:
        - This class currently supports only a liquid and a gaseous phase.
        - The various properties of this class depend on the thermodynamic state.
          They are defined during initialization since they depend on all components.

    """

    def __init__(self, ad_system: Optional[pp.ad.EquationSystem] = None) -> None:
        super().__init__(ad_system)

        self.eps: float = 1e-16
        """A small number defining the numerical zero.

        Used for the computation of roots to distinguish between phase-regions.
        Defaults to ``1e-16``.

        """

        self.transition_smoothing: float = 0.1
        """A small number to determine proximity between 3-root case and
        double-root case.

        Smoothing is applied according to below reference.
        Defaults to 0.1.

        See also:
            `Vu et al. (2021), Section 6.
            <https://doi.org/10.1016/j.matcom.2021.07.015>`_

        """

        self.is_super_critical: np.ndarray = np.array([], dtype=bool)
        """A boolean array indicating which cell is super-critical,
        after the roots of the EoS were computed."""

        ### PRIVATE

        # setting of currently supported phases
        self._phases: list[PR_Phase] = [
            PR_Phase(self.ad_system, name="L"),
            PR_Phase(self.ad_system, name="G"),
        ]

        # cohesion and covolume, assembled during initialization
        # (based on mixing-rule and present components)
        self._a: pp.ad.Operator
        self._dT_a: pp.ad.Operator
        self._b: pp.ad.Operator

        self._A: pp.ad.Operator
        self._B: pp.ad.Operator

        # operators representing the coefficients of the compressibility polynomial
        # and the reduced polynomial
        self._c0: pp.ad.Operator
        self._c1: pp.ad.Operator
        self._c2: pp.ad.Operator
        self._r: pp.ad.Operator
        self._q: pp.ad.Operator
        # discriminant of the polynomial
        self._delta: pp.ad.Operator

    def add_component(self, component: PR_Component | list[PR_Component]) -> None:
        """This child class method checks additionally if BIPs are defined for
        components to be added and components already added.

        For the Peng-Robinson EoS to work as intended, BIPs must be available for any
        combination of two present components and present in
        :data:`~porepy.composite.peng_robinson.pr_bip.PR_BIP_MAP`.

        Parameters:
            component: One or multiple model (PR-) components for this EoS.

        Raises:
            NotImplementedError: If a BIP is not available for any combination of
                modelled components.

        """
        if isinstance(component, PR_Component):
            component = [component]  # type: ignore

        missing_bips: list[tuple[str, str]] = list()

        # check for missing bips between new components and present components
        for comp_new in component:
            for comp_present in self.components:
                # there is no bip between a component and itself
                if comp_new != comp_present:
                    bip, *_ = get_PR_BIP(comp_new.name, comp_present.name)
                    # if bip is not available, add pair to missing bips
                    if bip is None:
                        missing_bips.append((comp_new.name, comp_present.name))

        # check for missing bips between new components
        for comp_1 in component:
            for comp_2 in component:
                # no bip between a component and itself
                if comp_2 != comp_1:
                    bip, *_ = get_PR_BIP(comp_1.name, comp_2.name)
                    if bip is None:
                        missing_bips.append((comp_1.name, comp_2.name))

        # if missing bips detected, raise error
        if missing_bips:
            raise NotImplementedError(
                f"BIPs not available for following component-pairs:\n\t{missing_bips}"
            )
        # if no missing bips, we proceed adding the components using the parent method.
        super().add_component(component)

    def initialize(self) -> None:
        """Creates and assigns thermodynamic properties, as well as equilibrium
        equations using PR.

        Before initializing the p-h and p-T subsystems,
        this method additionally constructs the cohesion and covolume factors in the EoS.

        After that, it performs a super-call to :meth:`~Composition.initialize`.

        As a third and final step it assigns equilibrium equations in the form of
        equality of fugacities.

        Raises:
            AssertionError: If the mixture is empty (no components).

        """
        # assert non-empty mixture
        assert self.num_components >= 1

        ## defining the cohesion value
        self._assign_cohesion()
        ## defining the covolume
        self._assign_covolume()
        ## defining A and B based on above
        self._A = (self.cohesion * self.p) / (R_IDEAL**2 * self.T * self.T)
        self._B = (self.covolume * self.p) / (R_IDEAL * self.T)
        ## creating operator representing the polynomial coefficients
        self._c0 = (
            _power(self.B, pp.ad.Scalar(3))
            + _power(self.B, pp.ad.Scalar(2))
            - self.A * self.B
        )
        self._c1 = self.A - 2 * self.B - 3 * _power(self.B, pp.ad.Scalar(2))
        self._c2 = self.B - 1

        self._r = self._c1 - _power(self._c2, pp.ad.Scalar(2)) / 3
        self._q = (
            2 / 27 * _power(self._c2, pp.ad.Scalar(3))
            - self._c2 * self._c1 / 3
            + self._c0
        )

        self._delta = (
            _power(self._q, pp.ad.Scalar(2)) / 4 + _power(self._r, pp.ad.Scalar(3)) / 27
        )

        # super call to initialize p-h and p-T subsystems
        super().initialize()

    ### EoS parameters -----------------------------------------------------------------

    @property
    def cohesion(self) -> pp.ad.Operator:
        """An operator representing ``a`` in the Peng-Robinson EoS using the component
        feed fraction and Van der Waals mixing rule."""
        return self._a

    @property
    def dT_cohesion(self) -> pp.ad.Operator:
        """An operator representing the derivative of :meth:`cohesion` w.r.t.
        temperature."""
        return self._dT_a

    @property
    def covolume(self) -> pp.ad.Operator:
        """An operator representing ``b`` in the Peng-Robinson EoS using the component
        feed fraction and Van der Waals mixing rule."""
        return self._b

    @property
    def A(self) -> pp.ad.Operator:
        """An operator representing ``A`` in the characteristic polynomial of the EoS.

        It is based on :meth:`cohesion` and its representation."""
        return self._A

    @property
    def B(self) -> pp.ad.Operator:
        """An operator representing ``B`` in the characteristic polynomial of the EoS.

        It is based on :meth:`covolume` and its representation."""
        return self._B

    ### Subsystem assembly method ------------------------------------------------------

    def linearize_subsystem(
        self,
        flash_type: Literal["isenthalpic", "isothermal"],
        other_vars: Optional[list[str]] = None,
        other_eqns: Optional[list[str]] = None,
        state: Optional[np.ndarray] = None,
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Before the system is linearized by a super call to the parent method,
        the PR mixture computes the EoS Roots for (iterative) updates.

        Warning:
            If the isothermal system is assembled, the roots of the EoS must be
            computed beforehand using :meth:`compute_roots`.

            In the isenthalpic system they are computed here due to the
            temperature-dependency of the roots.

            This is for performance reasons, since the roots must be evaluated only
            once in the isothermal flash.

        """

        if flash_type == "isenthalpic":
            self.compute_roots(state=state)
        return super().linearize_subsystem(flash_type, other_vars, other_eqns, state)

    ### Model equations ----------------------------------------------------------------

    def _assign_cohesion(self) -> None:
        """Creates the cohesion parameter for a mixture according to PR,
        as well as its derivative w.r.t. temperature."""
        components: list[PR_Component] = [c for c in self.components]  # type: ignore

        # First we sum over the diagonal elements of the mixture matrix,
        # starting with the first component
        comp_0 = components[0]
        a = VdW_a_ij(self.T, comp_0, comp_0) * comp_0.fraction * comp_0.fraction
        dT_a = dT_VdW_a_ij(self.T, comp_0, comp_0) * comp_0.fraction * comp_0.fraction

        if len(components) > 1:
            # add remaining diagonal elements
            for comp in components[1:]:
                a += VdW_a_ij(self.T, comp, comp) * comp.fraction * comp.fraction
                dT_a += dT_VdW_a_ij(self.T, comp, comp) * comp.fraction * comp.fraction

            # adding off-diagonal elements, including BIPs
            for comp_i in components:
                for comp_j in components:
                    if comp_i != comp_j:
                        # computing the cohesion between components i and j
                        a += (
                            VdW_a_ij(self.T, comp_i, comp_j)
                            * comp_i.fraction
                            * comp_j.fraction
                        )
                        # computing the derivative w.r.t temperature of cohesion terms
                        dT_a += (
                            dT_VdW_a_ij(self.T, comp_i, comp_j)
                            * comp_i.fraction
                            * comp_j.fraction
                        )

        # store cohesion parameters
        self._a = a
        self._dT_a = dT_a

    def _assign_covolume(self) -> None:
        """Creates the covolume of the mixture according to van der Waals- mixing rule."""
        components: list[PR_Component] = [c for c in self.components]  # type: ignore

        b = components[0].fraction * components[0].covolume

        if len(components) > 1:
            for comp in components[1:]:
                b += comp.fraction * comp.covolume

        self._b = b

    ### root computation ---------------------------------------------------------------

    def compute_roots(
        self,
        state: Optional[np.ndarray] = None,
        apply_smoother: bool = False,
    ) -> None:
        """Computes the roots of the characteristic polynomial and assigns phase labels.

        The roots depend on ``A`` and ``B`` of the Peng-Robinson EoS, hence on the
        thermodynamic state. They must be re-computed after any change in pressure,
        temperature and composition.
        This holds especially in iterative schemes.

        The results are stored in respective phase as ``Z``.

        Parameters:
            state: ``default=None``

                An optional (global) state vector for the AD system to evaluate
                ``A`` and ``B`` w.r.t to it.
            apply_smoother: ``default=False``

                If True, a smoothing procedure is applied in the three-root-region,
                where the intermediate root approaches one of the other roots.

                This is to be used **within** iterative procedures for numerical
                reasons. Once convergence is reached, the true roots should be computed
                without smoothing.

        """
        # evaluate necessary parameters
        A = self.A.evaluate(self.ad_system, state)
        B = self.B.evaluate(self.ad_system, state)
        r = self._r.evaluate(self.ad_system, state)
        q = self._q.evaluate(self.ad_system, state)
        c2 = self._c2.evaluate(self.ad_system, state)
        delta = self._delta.evaluate(self.ad_system, state)

        nc = len(A.val)
        shape = A.jac.shape
        # storage for roots
        Z_L_val = np.zeros(nc)
        Z_G_val = np.zeros(nc)
        Z_L_jac = sps.lil_matrix(shape)
        Z_G_jac = sps.lil_matrix(shape)

        # identify super-critical region
        self.is_super_critical = B.val > B_CRIT / A_CRIT * A.val
        # sub_critical = np.logical_not(self.is_super_critical)

        # as of now, this is not covered
        if np.any(self.is_super_critical):
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

        ### compute the one real root and the extended root from
        # conjugated imaginary roots
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

            ## PHASE LABELING in one-root-region
            # This has to hold, otherwise the polynomial can't have 3 distinct roots.

            # assert np.all(
            #     B_1.val < B_CRIT
            # ), "Co-volume exceeds critical value for labeling."

            # A_1 = pp.ad.Ad_array(A.val[one_root_region], A.jac[one_root_region])
            # # TODO pass state to labeling method
            # liquid_region, gas_region = self._get_labeled_regions(
            #     A_1,
            #     B_1,
            #     one_real_root_region
            # )

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

        ### compute the two relevant roots in the three root region
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

        ### storing results for access
        self._phases[0].Z.value = pp.ad.Ad_array(Z_L_val, Z_L_jac.tocsr())
        self._phases[1].Z.value = pp.ad.Ad_array(Z_G_val, Z_G_jac.tocsr())

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

        A cubic polynomial in ``A`` with coefficients dependent on ``B``
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
        c2_label = (2 * self.B * self.B - 10 * self.B - 1 / 4).evaluate(self.ad_system)
        c1_label = (
            -4 * _power(self.B, pp.ad.Scalar(4))
            + 28 * _power(self.B, pp.ad.Scalar(3))
            + 22 * self.B * self.B
            + 2 * self.B
        ).evaluate(self.ad_system)
        c0_label = (
            -8 * _power(self.B, pp.ad.Scalar(6))
            - 32 * _power(self.B, pp.ad.Scalar(5))
            - 40 * _power(self.B, pp.ad.Scalar(4))
            - 26 * _power(self.B, pp.ad.Scalar(3))
            - 2 * self.B * self.B
        ).evaluate(self.ad_system)
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
            delta_label.val < -self.eps
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
