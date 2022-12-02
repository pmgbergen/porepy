"""This module contains a class providing AD representations of (extended) roots of the
Peng-Robinson EoS. The extension is performed according [1] and transferred into PorePy's AD
framework.

References:
    [1]: `Ben Gharbia et al. (2021) <https://doi.org/10.1051/m2an/2021075>`_

"""
from __future__ import annotations

import porepy as pp
import numpy as np


ad_power = pp.ad.Function(pp.ad.power, "power")


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
            roots to distinguish between phase-regions.

    """

    def __init__(
        self,
        ad_system: pp.ad.ADSystem,
        A: pp.ad.Operator,
        B: pp.ad.Operator,
        eps: float = 1e-12,
    ) -> None:

        self.ad_system: pp.ad.ADSystem = ad_system
        """The AD system passed at instantiation."""

        self.A = A
        """``A`` passed at instantiation."""

        self.B = B
        """``B`` passed at instantiation."""

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
    def Z_crit(self) -> float:
        """Critical compressibility factor for the Peng-Robinson EoS, ~ 0.307401308."""
        return 1 / 32 * (
            11
            + np.cbrt(16 * np.sqrt(2) - 13)
            - np.cbrt(16 * np.sqrt(2) + 13)
        )
    
    @property
    def A_crit(self) -> float:
        """Critical value for ``A`` in the Peng-Robinson EoS, ~ 0.457235529."""
        return 1 / 512 * (
            -59
            + 3 * np.cbrt(276231 - 192512 * np.sqrt(2))
            + 3 * np.cbrt(276231 + 192512 * np.sqrt(2))
        )

    @property
    def B_crit(self) -> float:
        """Critical value for ``B`` in the Peng-Robinson EoS, ~ 0.077796073."""
        return 1 / 32 * (
            -1
            - 3 * np.cbrt(16 * np.sqrt(2) - 13)
            + 3 * np.cbrt(16 * np.sqrt(2) + 13)
        )

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
        return ad_power(self.B, 3) + self.B * self.B - self.A * self.B

    @property
    def p(self) -> pp.ad.Operator:
        """An operator representing the coefficient of the monomial ``Z`` of the **reduced**
        characteristic polynomial."""
        return (3 * self.c1 - self.c2 * self.c2) / 3

    @property
    def q(self) -> pp.ad.Operator:
        """An operator representing the coefficient of the monomial ``Z**0`` of the **reduced**
        characteristic polynomial."""
        return (2 * ad_power(self.c2, 3) - 9 * self.c2 * self.c1 + 27 * self.c0) / 27

    @property
    def delta(self) -> np.ndarray:
        """An array representing the discriminant of the characteristic polynomial, based
        on the current thermodynamic state.

        The sign of the discriminant can be used to distinguish between 1 and 2-phase regions:

        - ``< 0``: single-phase region (1 real root)
        - ``> 0``: 2-phase region (3 distinct real roots)
        - ``= 0``: special region (1 or 2 distinct real roots with respective multiplicity)

        Warning:
            The physical meaning of the case with two distinct real roots is
            unclear as of now.

        """
        return -(
            (
                self.q * self.q / 4
                + ad_power(self.p, 3) / 27
            )
            .evaluate(self.ad_system.dof_manager)
            .val
        )

    ### root computation ----------------------------------------------------------------------

    def compute_roots(self) -> None:
        """Computes the roots of the characteristic polynomial and assigns phase labels.
        
        The roots depend on ``A`` and ``B`` of the Peng-Robinson EoS, hence on the
        thermodynamic state. They must be re-computed after any change in pressure, temperature
        and composition.

        The results can be accessed using :meth:`liquid_root` and :meth:`gas_root`.

        """
        A = self.A.evaluate(self.ad_system.dof_manager)
        B = self.B.evaluate(self.ad_system.dof_manager)
        p = self.p.evaluate(self.ad_system.dof_manager)
        delta = self.delta

        super_crit_region = B.val > self.B_crit / self.A_crit * A.val

        if np.any(super_crit_region):
            raise NotImplementedError("Super-critical roots not available yet.")

        if np.any(np.logical_and(
            np.isclose(delta, 0., atol=self._eps),
            np.abs(p.val) > self._eps
        )):
            raise NotImplementedError("Case with two distinct real roots encountered.")
