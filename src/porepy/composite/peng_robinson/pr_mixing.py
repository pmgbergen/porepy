"""A module containing implementations of various mixing rules to be used in combination
with the Peng-Robinson framework for mutliphase-multicomponent mixtures."""

from __future__ import annotations

import porepy as pp
from porepy.numerics.ad.operator_functions import NumericType

from .pr_bip import get_PR_BIP
from .pr_component import PR_Component


def VdW_a(
    T: NumericType, x: list[NumericType], components: list[PR_Component]
) -> NumericType:
    """Compute the cohesion term ``a`` for cubic EoS using Van der Waals mixing rule

        ``a = sum_i sum_j x_i * x_j * a_ij``,

    where
        ``a_ij  = sqrt(a_i * a_j) * (1 - delta_ij)``.

    Parameters:
        T: Temperature.
        x: List of fractions.
        components: List of compatible components, corresponding to the order in ``x``.

    Returns:
        The term ``a`` from above equations.

    """
    num_components = len(components)
    assert (
        len(x) == num_components
    ), "VdW mixing rule: Mismatch in numbers of components and fractions."

    a = 0.0

    # mixture matrix is symmetric, sum over all entries in upper triangle
    # multiply off-diagonal elements with 2
    for i in range(num_components):
        for j in range(i, num_components):
            a_ij = x[i] * x[j] * VdW_a_ij(T, components[i], components[j])
            if i != j:
                a += 2 * a_ij
            else:
                a += a_ij

    return a


def VdW_a_ij(T: NumericType, comp_i: PR_Component, comp_j: PR_Component) -> NumericType:
    """Computes the mixed cohesion value ``a_ij`` using the Van der Waals mixing rule.

    It holds

        ``a_ij  = sqrt(a_i * a_j) * (1 - delta_ij)``,

    where ``delta_ij`` is the binary interaction parameter for ``i!=j``,
    and 0 for ``i==j``.

    Parameters:
        T: Temperature.
        comp_i: The first (Peng-Robinson) component.
        comp_j: The second one.

    Returns:
        The term ``a_ij`` in above equation.

    """
    # for different components, the expression is more complex
    if comp_i != comp_j:
        # first part without BIP
        a_ij = pp.ad.sqrt(comp_i.cohesion(T) * comp_j.cohesion(T))
        # get bip
        bip, _, order = get_PR_BIP(comp_i.name, comp_j.name)
        # assert there is a bip, to appease mypy
        assert bip is not None

        # call to BIP and multiply with a_ij
        if order:
            a_ij *= 1 - bip(T, comp_i, comp_j)
        else:
            a_ij *= 1 - bip(T, comp_j, comp_i)
    # for same components, the expression can be simplified
    else:
        a_ij = comp_i.cohesion(T)

    return a_ij


def VdW_dT_a(
    T: NumericType, x: list[NumericType], components: list[PR_Component]
) -> NumericType:
    """Compute the temperature-derivative of the cohesion term ``a`` for cubic EoS using
    Van der Waals mixing rule (see :func:`VdW_a`).

    Parameters:
        T: Temperature.
        x: List of fractions.
        components: List of compatible components, corresponding to the order in ``x``.

    Returns:
        The term ``dT_a`` for the cubic EoS.

    """
    num_components = len(components)
    assert (
        len(x) == num_components
    ), "VdW mixing rule: Mismatch in numbers of components and fractions."

    a = 0.0

    # mixture matrix is symmetric, sum over all entries in upper triangle
    # multiply off-diagonal elements with 2
    for i in range(num_components):
        for j in range(i, num_components):
            a_ij = x[i] * x[j] * VdW_dT_a_ij(T, components[i], components[j])
            if i != j:
                a += 2 * a_ij
            else:
                a += a_ij

    return a


def VdW_dXi_a(
    T: NumericType, x: list[NumericType], components: list[PR_Component], i: int
) -> NumericType:
    """Returns the derivative of ``a`` w.r.t. to the fraction of component ``i``.

    TODO:
        Is this really this derivative or something else?

    Parameters:
        T: Temperature.
        x: List of fractions.
        components: List of compatible components, corresponding to the order in ``x``.
        i: Index for which component the derivative should be calculated.

    Returns:
        The term ``dXi_a`` for the cubic EoS.

    """
    num_components = len(components)
    assert (
        len(x) == num_components
    ), "VdW mixing rule: Mismatch in numbers of components and fractions."

    dXi_a = 0.0
    comp_i = components[i]

    for j in range(num_components):
        dXi_a += x[j] * VdW_a_ij(T, comp_i, components[j])

    return 2 * dXi_a


def VdW_dT_a_ij(
    T: NumericType, comp_i: PR_Component, comp_j: PR_Component
) -> NumericType:
    """Returns an operator representing the derivative of :meth:`VdW_a_ij` w.r.t. to the
    temperature.

    See :meth:`VdW_a_ij` for more.

    """
    # the expression for two different components
    if comp_i != comp_j:
        # the derivative of a_ij
        dT_a_ij = (
            pp.ad.power(comp_i.cohesion(T) * comp_j.cohesion(T), -1 / 2)
            / 2
            * (
                comp_i.dT_cohesion(T) * comp_j.cohesion(T)
                + comp_i.cohesion(T) * comp_j.dT_cohesion(T)
            )
        )

        bip, dT_bip, order = get_PR_BIP(comp_i.name, comp_j.name)
        # assert there is a bip, to appease mypy
        assert bip is not None

        # multiplying with BIP
        if order:
            dT_a_ij *= 1 - bip(T, comp_i, comp_j)

            # if the derivative of the BIP is not trivial, add respective part
            if dT_bip:
                dT_a_ij -= pp.ad.sqrt(comp_i.cohesion(T) * comp_j.cohesion(T)) * dT_bip(
                    T, comp_i, comp_j
                )
        else:
            dT_a_ij *= 1 - bip(T, comp_j, comp_i)

            # if the derivative of the BIP is not trivial, add respective part
            if dT_bip:
                dT_a_ij -= pp.ad.sqrt(comp_i.cohesion(T) * comp_j.cohesion(T)) * dT_bip(
                    T, comp_j, comp_i
                )
    # if the components are the same, the expression simplifies
    else:
        dT_a_ij = comp_i.dT_cohesion(T)

    return dT_a_ij


def VdW_b(x: list[NumericType], components: list[PR_Component]) -> NumericType:
    """Compute the covolume term ``b`` for cubic EoS using Van der Waals mixing rule

        ``b = sum_i x_i * b_i``.

    Parameters:
        x: List of fractions.
        components: List of compatible components, corresponding to the order in ``x``.

    Returns:
        The term ``b`` from above equations.

    """
    num_components = len(components)
    assert (
        len(x) == num_components
    ), "VdW mixing rule: Mismatch in numbers of components and fractions."

    b = 0.0

    for i in range(num_components):
        b += x[i] * components[i].covolume

    return b
