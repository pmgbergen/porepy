"""A module containing implementations of various mixing rules to be used in combination
with the Peng-Robinson framework for mutliphase-multicomponent mixtures."""

from __future__ import annotations

import porepy as pp

from .pr_bip import get_PR_BIP
from .pr_component import PR_Component
from .pr_utils import _power, _sqrt

_div_sqrt = pp.ad.Scalar(-1 / 2)


def VdW_a_ij(
    T: pp.ad.MixedDimensionalVariable, comp_i: PR_Component, comp_j: PR_Component
) -> pp.ad.Operator:
    """Computes the mixed cohesion value ``a_ij`` using the Van der Waals mixing rule.

    It holds

        ``a_ij  = sqrt(a_i * a_j) * (1 - delta_ij)``,

    where ``delta_ij`` is the binary interaction parameter for ``i!=j``,
    and 0 for ``i==j``.

    Parameters:
        T: The temperature variable of the mixture.
        comp_i: The first (Peng-Robinson) component.
        comp_j: The second one.

    Returns:
        An AD operator representing ``a_ij``.

    """
    # for different components, the expression is more complex
    if comp_i != comp_j:
        # first part without BIP
        a_ij = _sqrt(comp_i.cohesion(T) * comp_j.cohesion(T))
        # get bip
        bip, _, order = get_PR_BIP(comp_i.name, comp_j.name)
        # assert there is a bip, to appease mypy
        # this check is performed in add_component anyways
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


def dT_VdW_a_ij(
    T: pp.ad.MixedDimensionalVariable, comp_i: PR_Component, comp_j: PR_Component
) -> pp.ad.Operator:
    """Returns an operator representing the derivative of :meth:`VdW_a_ij` w.r.t. to the
    temperature.

    See :meth:`VdW_a_ij` for more.

    """
    # the expression for two different components
    if comp_i != comp_j:
        # the derivative of a_ij
        dT_a_ij = (
            _power(comp_i.cohesion(T), _div_sqrt)
            * comp_i.dT_cohesion(T)
            * _sqrt(comp_j.cohesion(T))
            + _sqrt(comp_i.cohesion(T))
            * _power(comp_j.cohesion(T), _div_sqrt)
            * comp_j.dT_cohesion(T)
        ) / 2

        bip, dT_bip, order = get_PR_BIP(comp_i.name, comp_j.name)
        # assert there is a bip, to appease mypy
        # this check is performed in add_component anyways
        assert bip is not None

        # multiplying with BIP
        if order:
            dT_a_ij *= 1 - bip(T, comp_i, comp_j)

            # if the derivative of the BIP is not trivial, add respective part
            if dT_bip:
                dT_a_ij -= _sqrt(comp_i.cohesion(T) * comp_j.cohesion(T)) * dT_bip(
                    T, comp_i, comp_j
                )
        else:
            dT_a_ij *= 1 - bip(T, comp_j, comp_i)

            # if the derivative of the BIP is not trivial, add respective part
            if dT_bip:
                dT_a_ij -= _sqrt(comp_i.cohesion(T) * comp_j.cohesion(T)) * dT_bip(
                    T, comp_j, comp_i
                )
    # if the components are the same, the expression simplifies
    else:
        dT_a_ij = comp_i.dT_cohesion(T)

    return dT_a_ij
