"""A module containing implementations of various mixing rules to be used in combination
with the Peng-Robinson framework for mutliphase-multicomponent mixtures."""

from __future__ import annotations

import sympy as sm

import porepy as pp
from porepy.numerics.ad.operator_functions import NumericType

from ..composite_utils import safe_sum

__all__ = ["VanDerWaals"]


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
                a_ij_ = pp.ad.sqrt(a[i] * a[j])
                delta_ij = 1 - bip[i][j]

                a_ij = a_ij_ * delta_ij
                dT_a_ij = (
                    pp.ad.power(a[i] * a[j], -1 / 2)
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
    def cohesion_s(
        X: list[sm.Expr], a: list[sm.Expr], bip: list[list[sm.Expr]]
    ) -> NumericType:
        """Symbolic implementation for the mixing rule for the cohesion term."""

        nc = len(X)  # number of components

        a_parts: list[sm.Expr] = []

        # mixture matrix is symmetric, sum over all entries in upper triangle
        # multiply off-diagonal elements with 2
        for i in range(nc):
            a_parts.append(X[i] ** 2 * a[i])
            for j in range(i + 1, nc):
                x_ij = X[i] * X[j]
                a_ij_ = sm.sqrt(a[i] * a[j])
                delta_ij = 1 - bip[i][j]

                a_ij = a_ij_ * delta_ij

                # off-diagonal elements appear always twice due to symmetry
                a_parts.append(2.0 * x_ij * a_ij)

        return safe_sum(a_parts)

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
            [X[j] * pp.ad.sqrt(a[i] * a[j]) * (1 - bip[i][j]) for j in range(len(X))]
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
