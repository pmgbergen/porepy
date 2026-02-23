"""Module testing assembly of common flash equations as well as generic argument
parsing."""

from itertools import product

import numpy as np
import pytest

import porepy as pp
import porepy.compositional as pc
import porepy.compositional.flash as pf
from porepy.applications.test_utils.derivative_testing import (
    assert_order_at_least,
    get_EOC_taylor,
)


def _dummy_property(p: float, T: float, x: np.ndarray, power: int = 2) -> float:
    """A dummy phase property as assumed by the flash framework.

    It takes a pressure and temperature value, as well as partial fractions and returns
    a float.

    """
    return p**power + T**power + (x**power).sum()


def _dummy_property_derivative(
    p: float, T: float, x: np.ndarray, power: int = 2
) -> float:
    """A dummy phase property derivative as assumed by the flash framework.

    Contains the analytical derivative of :func:`_dummy_property`.

    It takes a pressure and temperature value, as well as partial fractions and returns
    an array with shape ``(2 + x.size,)``

    """
    n = x.shape[0]
    assert x.shape == (n,)
    d = np.array(
        [power * p ** (power - 1), power * T ** (power - 1)]
        + [power * x_ ** (power - 1) for x_ in x]
    )
    assert d.shape == (2 + n,)
    return d


@pytest.mark.parametrize(
    "spec", [spec for spec in pc.FlashSpec if spec != pc.FlashSpec.none]
)
@pytest.mark.parametrize("with_params", [True, False])
@pytest.mark.parametrize("nphase", [1, 2, 5])
@pytest.mark.parametrize("ncomp", [1, 2, 5])
@pytest.mark.parametrize("vectorized", [True, False])
def test_assembly_and_parsing_of_generic_flash_argument(
    vectorized: bool,
    ncomp: int,
    nphase: int,
    spec: pc.FlashSpec,
    with_params: bool,
) -> None:
    if vectorized:
        N = 10
        parser = pf.parse_vectorized_generic_arg
        assembler = pf.assemble_vectorized_generic_arg
    else:
        N = 0
        parser = pf.parse_generic_arg
        assembler = pf.assemble_generic_arg

    if with_params:
        dp = np.random.randint(1, 10)
        params = np.random.random((dp, N) if vectorized else (dp,))
    else:
        params = np.zeros((0, N) if vectorized else (0,))

    # Make sure all values distinct.
    d = pf.dim_gen_arg(ncomp, nphase, spec)
    non_params = np.random.choice(
        np.arange(0, 100000),
        replace=False,
        size=(d, N) if vectorized else (d,),
    )

    if vectorized:
        Xgen = np.vstack([params, non_params]).transpose()
    else:
        Xgen = np.hstack([params, non_params])

    x, y, z, p, T, state1, state2, pars = parser(Xgen, ncomp, nphase, spec)

    if vectorized:
        assert x.shape == (nphase, ncomp, N), "Partial fractions of unexpected shape."
        assert y.shape == (nphase, N), "Phase fractions of unexpected shape."
        assert z.shape == (ncomp, N), "Overall compositions of unexpected shape."
        assert p.shape == (N,), "Pressure of unexpected shape."
        assert T.shape == (N,), "Temperature of unexpected shape."
        assert state1.shape == (N,), "State value 1 of unexpected shape."
        assert state2.shape == (N,), "State value 2 of unexpected shape."
    else:
        assert x.shape == (nphase, ncomp), "Partial fractions of unexpected shape."
        assert y.shape == (nphase,), "Phase fractions of unexpected shape."
        assert z.shape == (ncomp,), "Overall compositions of unexpected shape."
        assert isinstance(p, float), "Pressure expected to be float."
        assert isinstance(T, float), "Temperature expected to be float."
        assert isinstance(state1, float), "State value 1 expected to be float."
        assert isinstance(state2, float), "State value 2 expected to be float."
    assert pars.shape == params.shape, "Parsed parameters of unexpected shape."
    assert np.all(pars == params), "Expecting parameters to not change."

    # Expecting to be unequal in any case.
    assert np.all(p != T), "Expecting pressure and temperature to be distinct."
    assert np.all(state1 != state2), "Expecting state values to be distinct."

    if spec in [pc.FlashSpec.pT, pc.FlashSpec.vT]:
        assert "T" == spec.name[1], "Expecting character T in isothermal spec."
        assert np.all(T == state2), (
            "State value 2 and temperature expected to be equal in isothermal spec."
        )
    else:
        assert np.all(T != state2), (
            "State value 2 and temperature expected to be distinct in non-isothermal "
            "spec."
        )
    if "p" == spec.name[0]:
        assert np.all(p == state1), (
            "State value 1 and pressure expected to be equal in isobaric spec."
        )
    elif "v" == spec.name[0]:
        assert np.all(p != state1), (
            "State value 1 and pressure expected to be distinct in isochoric spec."
        )
        assert np.all(T != state1), (
            "State value 1 and temperature expected to be distinct in isochoric spec."
        )
    else:
        assert False, "Uncovered specification."

    # For non-isobaric and non-isothermal specifications, all values must be distinct
    if spec > pc.FlashSpec.vT:
        if vectorized:
            vals = set([*p, *T, *state1, *state2])
            s = 4 * N
        else:
            vals = set([p, T, state1, state2])
            s = 4
        assert len(vals) == s, (
            "State values, pressure and temperature expected to be distinct for "
            "isochoric, non-isothermal spec."
        )

    Xgen2 = assembler(x, y, z, p, T, state1, state2, pars, spec)
    assert np.all(Xgen == Xgen2), (
        "Parsed and re-assembled generic arg expected to be equal to original arg,"
    )

    # Sanity check that the values remain the same and there is no accidental
    # cancelation of errors.
    x2, y2, z2, p2, T2, state12, state22, pars2 = parser(
        Xgen2.copy(), ncomp, nphase, spec
    )
    assert np.all(x2 == x)
    assert np.all(y2 == y)
    assert np.all(z == z)
    assert np.all(p2 == p)
    assert np.all(T2 == T)
    assert np.all(state12 == state1)
    assert np.all(state22 == state2)
    assert np.all(pars2 == pars)

    Xgen3 = assembler(x2, y2, z2, p2, T2, state12, state22, pars2, spec)
    assert np.all(Xgen == Xgen3)


def test_parsing_with_no_flash_spec() -> None:
    """Expected to fail."""
    ncomp = np.random.randint(1, 10)
    nphase = np.random.randint(1, 10)

    Xgen = np.random.random((ncomp * nphase,))

    with pytest.raises(ValueError):
        _ = pf.dim_gen_arg(ncomp, nphase, pc.FlashSpec.none)
    with pytest.raises(ValueError):
        _ = pf.parse_generic_arg(Xgen, ncomp, nphase, pc.FlashSpec.none)


@pytest.mark.parametrize(
    "spec",
    [spec for spec in pf.FlashSpec if spec != pf.FlashSpec.none],
)
@pytest.mark.parametrize("with_params", [True, False])
@pytest.mark.parametrize("with_init", [True, False])
@pytest.mark.parametrize("nphase", [1, 2, 5])
@pytest.mark.parametrize("ncomp", [1, 2, 5])
@pytest.mark.parametrize("N", [1, 10])
def test_generic_arg_from_result_struture(
    N: int,
    ncomp: int,
    nphase: int,
    with_params: bool,
    with_init: bool,
    spec: pc.FlashSpec,
) -> None:
    """Tests the assembly of the generic argument using a flash results structure."""

    z = np.random.random((ncomp, N))
    p = np.random.random((N,))
    T = np.random.random((N,))
    h = np.random.random((N,))
    u = np.random.random((N,))
    v = np.random.random((N,))
    y = np.random.random((nphase, N))
    x = np.random.random((nphase, ncomp, N))

    match spec:
        case pc.FlashSpec.pT:
            state1 = p.copy()
            state2 = T.copy()
        case pc.FlashSpec.ph:
            state1 = p.copy()
            state2 = h.copy()
        case pc.FlashSpec.vT:
            state1 = v.copy()
            state2 = T.copy()
        case pc.FlashSpec.vh:
            state1 = v.copy()
            state2 = h.copy()
        case pc.FlashSpec.vu:
            state1 = v.copy()
            state2 = u.copy()
        case _:
            assert False, "Missing test logic."

    if with_params:
        params = np.random.random((np.random.randint(1, 10), N))
    else:
        params = np.random.random((0, N))

    results = pf.FlashResults(
        specification=spec,
        size=N,
        p=p,
        T=T,
        z=z,
        y=y,
        h=h,
        u=u,
        rho=1.0 / v,
        phases=[pp.compositional.PhaseProperties(x=x[j, :, :]) for j in range(nphase)],
    )

    XgenA = pf.assemble_vectorized_generic_arg(
        x, y, z, p, T, state1, state2, params, spec
    )
    XgenB = pf.generic_arg_from_flash_results(
        results, ncomp, nphase, with_init, params if with_params else None
    )

    # If all values are used, they must be identical.
    # NOTE: The checks for volume/state1 must be done with allclose because v=1/rho
    # in the FlashResults structure.
    if with_init:
        assert np.allclose(XgenA, XgenB, rtol=0.0, atol=1e-15)
    # If not all values are used, the ones associated with degrees of freedom must be 0.
    else:
        xb, yb, zb, pb, Tb, st1b, st2b, paramsb = pf.parse_vectorized_generic_arg(
            XgenB, ncomp, nphase, spec
        )

        assert np.all(xb == 0.0)
        # The parsing assembles the reference entities by unity of fractions
        assert np.all(yb[0] == 1)
        if nphase > 1:
            assert np.all(yb[1:] == 0)

        if ncomp == 1:
            assert np.all(zb == 1)
        else:
            assert np.all(zb[1:] == z[1:])
            assert np.all(zb[0] == 1 - z[1:].sum(axis=0))

        assert np.allclose(st1b, state1, rtol=0.0, atol=1e-15)
        assert np.all(st2b == state2)
        assert np.all(paramsb == params)

        if pc.FlashSpec.none < spec < pc.FlashSpec.vT:
            assert np.all(p == pb)
            assert np.all(p == st1b)
            if spec == pc.FlashSpec.pT:
                assert np.all(T == Tb)
                assert np.all(T == st2b)
            elif spec == pc.FlashSpec.ph:
                assert np.all(h == st2b)
            else:
                assert False, "Missing test logic"
        elif spec >= pc.FlashSpec.vT:
            assert np.allclose(v, st1b, rtol=0.0, atol=1e-15)
            assert np.all(pb == 0)
            if spec == pc.FlashSpec.vT:
                assert np.all(T == Tb)
                assert np.all(T == st2b)
            elif spec == pc.FlashSpec.vh:
                assert np.all(Tb == 0)
                assert np.all(h == st2b)
            elif spec == pc.FlashSpec.vu:
                assert np.all(Tb == 0)
                assert np.all(u == st2b)
            else:
                assert False, "Missing test logic"


# NOTE: The directions for the derivatives in the Taylor approximation test implemented
# in the tests below assume a certain knowledge about the structure of the generic
# flash argument. If that changes, the tests need adaption.


def _d_from_npnc(nphase: int, ncomp: int, spec: pc.FlashSpec) -> np.ndarray:
    """Directions for"""
    dim = pf.dim_gen_arg(ncomp, nphase, spec)
    nf = ncomp * nphase + nphase - 1
    return np.hstack((np.zeros((nf, dim - nf)), np.eye(nf)))


# Mass constraint for 1 component is always a dependent equation and never assembled.
@pytest.mark.parametrize(
    ["nphase", "ncomp", "d"],
    [
        (nphase, ncomp, d)
        for nphase, ncomp in product([1, 2, 5], [2, 5])
        for d in _d_from_npnc(nphase, ncomp, pc.FlashSpec.pT)
    ],
)
def test_mass_constraints(ncomp: int, nphase: int, d: np.ndarray) -> None:
    """Tests if the mass conservation equation is correctly implemented and its
    Jacobian function allows the Taylor approximation to be of second order."""
    spec = pc.FlashSpec.pT
    dim = pf.dim_gen_arg(ncomp, nphase, spec)
    nf = ncomp * nphase + nphase - 1

    def func(xgen):
        x, y, z, *_ = pf.parse_generic_arg(xgen, ncomp, nphase, spec)
        res = pf.mass_constraint_res(x, y, z)
        assert res.shape == (ncomp - 1,), "Residual of unexpected shape."
        return res

    def dfunc(xgen):
        x, y, *_ = pf.parse_generic_arg(xgen, ncomp, nphase, spec)
        jac = pf.mass_constraint_jac(x, y)
        assert jac.shape == (ncomp - 1, nf + 2), "Jacobian of unexpected shape."
        assert np.all(jac[:, :2] == 0), (
            "Jacobian has non-trivial derivatives for p and T."
        )
        return np.hstack((np.zeros((ncomp - 1, dim - nf)), jac[:, -nf:]))

    # Whatever z and x are, if y or x is zero we expect values -z
    z = np.random.random((ncomp,))
    y = np.zeros(nphase)
    x = np.random.random((nphase, ncomp))
    res = pf.mass_constraint_res(x, y, z)
    assert np.all(res == -z[1:]), "Unexpected residual values."
    y = np.random.random((nphase,))
    x = np.zeros((nphase, ncomp))
    res = pf.mass_constraint_res(x, y, z)
    assert np.all(res == -z[1:]), "Unexpected residual values."
    # If x = 1 and y = 1/nphase (homogenous mass distribution), result should be 1 - z
    y = np.ones(nphase) / nphase
    x = np.ones((nphase, ncomp))
    res = pf.mass_constraint_res(x, y, z)
    assert np.all(res == 1.0 - z[1:]), "Unexpected residual values."

    # If only 1 component, the mass conservation equations should be empty.
    assert pf.mass_constraint_res(x, y, np.ones(1)).shape == (0,), (
        "Unexpected residual shape for 1 component."
    )

    Xgen = np.random.random((dim,))
    h = np.logspace(0, -9, 10)

    orders = get_EOC_taylor(func, dfunc, Xgen, d, h)
    assert_order_at_least(orders, 2.0, tol=1e-3)


@pytest.mark.parametrize(
    ["nphase", "ncomp", "d"],
    [
        (nphase, ncomp, d)
        for nphase, ncomp in product([1, 2, 5], [1, 2, 5])
        for d in _d_from_npnc(nphase, ncomp, pc.FlashSpec.pT)
    ],
)
def test_complementary_conditions(ncomp: int, nphase: int, d: np.ndarray) -> None:
    """Tests if the complementary conditions are correctly implemented and its
    Jacobian function allows the Taylor approximation to be of second order.

    We test the smooth version without the inequality constraints. I.e., just the
    multiplication of fractions and unity of partial fractions per phase.

    """
    spec = pc.FlashSpec.pT
    dim = pf.dim_gen_arg(ncomp, nphase, spec)
    nf = ncomp * nphase + nphase - 1

    def func(xgen):
        x, y, *_ = pf.parse_generic_arg(xgen, ncomp, nphase, spec)
        res = pf.complementary_conditions_res(x, y)
        assert res.shape == (nphase,), "Residual of unexpected shape."
        return res

    def dfunc(xgen):
        x, y, *_ = pf.parse_generic_arg(xgen, ncomp, nphase, spec)
        jac = pf.complementary_conditions_jac(x, y)
        assert jac.shape == (nphase, nf + 2), "Jacobian of unexpected shape."
        assert np.all(jac[:, :2] == 0), (
            "Jacobian has non-trivial derivatives for p and T."
        )
        return np.hstack((np.zeros((nphase, dim - nf)), jac[:, -nf:]))

    # If y is zero, the complementary conditions are zero.
    y = np.zeros(nphase)
    x = np.random.random((nphase, ncomp))
    res = pf.complementary_conditions_res(x, y)
    assert np.all(res == 0.0), "Unexpected residual values."
    # If x are homogenous, the unity of fractions leads to zero.
    y = np.random.random((nphase,))
    x = np.ones((nphase, ncomp)) / ncomp
    res = pf.complementary_conditions_res(x, y)
    assert np.all(res == 0.0), "Unexpected residual values."
    # If x = 0, we should get y
    y = np.random.random((nphase,))
    x = np.zeros((nphase, ncomp))
    res = pf.complementary_conditions_res(x, y)
    assert np.all(res == y), "Unexpected residual values."
    # If y = 1, we should get the unity of fractions
    y = np.ones(nphase)
    x = np.random.random((nphase, ncomp))
    res = pf.complementary_conditions_res(x, y)
    assert np.all(res == 1 - x.sum(axis=1)), "Unexpected residual values."

    Xgen = np.random.random((dim,))
    h = np.logspace(0, -10, 11)

    orders = get_EOC_taylor(func, dfunc, Xgen, d, h)
    assert_order_at_least(orders, 2.0, tol=1e-3)


@pytest.mark.parametrize(
    ["nphase", "ncomp", "d"],
    [
        (nphase, ncomp, d)
        for nphase, ncomp in product([1, 2, 5], [1, 2, 5])
        for d in _d_from_npnc(nphase, ncomp, pc.FlashSpec.vh)
    ],
)
def test_first_order_constraint(ncomp: int, nphase: int, d: np.ndarray) -> None:
    """Tests if the first-order constraint is correctly implemented and its
    Jacobian function allows the Taylor approximation to be of second order."""
    spec = pc.FlashSpec.vh
    dim = pf.dim_gen_arg(ncomp, nphase, spec)
    nf = ncomp * nphase + nphase - 1 + 2

    # Target value of the constraint.
    phi = np.random.rand()

    def func(xgen):
        x, y, _, p, T, *_ = pf.parse_generic_arg(xgen, ncomp, nphase, spec)
        phis = np.array([_dummy_property(p, T, x_) for x_ in x])
        res = pf.first_order_constraint_res(phi, y, phis)
        assert res.shape == (1,), "Residual of unexpected shape."
        return res

    def dfunc(xgen):
        x, y, _, p, T, *_ = pf.parse_generic_arg(xgen, ncomp, nphase, spec)
        phis = np.array([_dummy_property(p, T, x_) for x_ in x])
        dphis = np.array([_dummy_property_derivative(p, T, x_) for x_ in x])
        jac = pf.first_order_constraint_jac(y, phis, dphis)
        assert jac.shape == (1, nf), "Jacobian of unexpected shape."
        return np.hstack((np.zeros((1, dim - nf)), jac))

    # If weights are zero, or the phis are zero, we expect -phi
    w = np.zeros(nphase)
    phis = np.random.random((nphase,))
    res = pf.first_order_constraint_res(phi, w, phis)
    assert np.all(res == -phi), "Unexpected residual values."
    res = pf.first_order_constraint_res(phi, phis, w)
    assert np.all(res == -phi), "Unexpected residual values."
    # If target value is zero, we expect the dot product of the weights and phis
    w = np.random.random((nphase,))
    res = pf.first_order_constraint_res(0.0, w, phis)
    assert np.allclose(res, np.dot(w, phis), rtol=0.0), "Unexpected residual values."
    res = pf.first_order_constraint_res(0.0, phis, w)
    assert np.allclose(res, np.dot(w, phis), rtol=0.0), "Unexpected residual values."
    # If w=1 and phis = phi / nphase, we expect zero residual
    w = np.ones(nphase)
    phis = np.ones(nphase) * phi / nphase
    res = pf.first_order_constraint_res(phi, w, phis)
    assert np.allclose(res, 0, rtol=0.0), "Unexpected residual values."
    res = pf.first_order_constraint_res(phi, phis, w)
    assert np.allclose(res, 0, rtol=0.0), "Unexpected residual values."

    Xgen = np.random.random((dim,))
    h = np.logspace(0, -9, 10)

    orders = get_EOC_taylor(func, dfunc, Xgen, d, h)
    assert_order_at_least(orders, 2.0, tol=1e-2)


# Isofugacity constraints make no sense for 1 phase.
@pytest.mark.parametrize(
    ["nphase", "ncomp", "d"],
    [
        (nphase, ncomp, d)
        for nphase, ncomp in product([2, 5], [1, 2, 5])
        # Isofugacity constraints do not depend on saturations or phase fractions, hence
        # remove the directions.
        for d in np.vstack(
            (
                _d_from_npnc(nphase, ncomp, pc.FlashSpec.vh)[:2],
                _d_from_npnc(nphase, ncomp, pc.FlashSpec.vh)[2 + 2 * (nphase - 1) :],
            )
        )
    ],
)
def test_isofugacity_constraints(ncomp: int, nphase: int, d: np.ndarray) -> None:
    """Tests if the isofugacity constraints constraints are correctly implemented and
    its Jacobian function allows the Taylor approximation to be of second order."""
    spec = pc.FlashSpec.vh
    dim = pf.dim_gen_arg(ncomp, nphase, spec)
    nf = ncomp * nphase + nphase - 1 + 2

    def func(xgen):
        x, _, _, p, T, *_ = pf.parse_generic_arg(xgen, ncomp, nphase, spec)
        phis = np.array(
            [[_dummy_property(p, T, x_, power=i + 1) for i in range(ncomp)] for x_ in x]
        )
        assert phis.shape == x.shape
        res = pf.isofugacity_constraints_res(x, phis)
        assert res.shape == ((nphase - 1) * ncomp,), "Residual of unexpected shape."
        return res

    def dfunc(xgen):
        x, _, _, p, T, *_ = pf.parse_generic_arg(xgen, ncomp, nphase, spec)
        phis = np.array(
            [[_dummy_property(p, T, x_, power=i + 1) for i in range(ncomp)] for x_ in x]
        )
        dphis = np.array(
            [
                [
                    _dummy_property_derivative(p, T, x_, power=i + 1)
                    for i in range(ncomp)
                ]
                for x_ in x
            ]
        )
        assert phis.shape == x.shape
        assert dphis.shape == (nphase, ncomp, 2 + ncomp)
        jac = pf.isofugacity_constraints_jac(x, dphis)
        assert jac.shape == ((nphase - 1) * ncomp, nf), "Jacobian of unexpected shape."
        return np.hstack((np.zeros(((nphase - 1) * ncomp, dim - nf)), jac))

    # If partial fractions in independent phases are zero, or their phis, we get only
    # the part containing the reference phase -x_r phi_r
    # x = np.array([np.ones(ncomp)] + [np.zeros(ncomp)] * (nphase - 1))
    # phis = np.random.random((nphase, ncomp))
    # res = flash.isofugacity_constraints_res(x, phis)
    # assert np.all(res == -np.hstack([x[0] * phis[0]] * (nphase - 1))), (
    #     "Unexpected residual values."
    # )
    # res = flash.isofugacity_constraints_res(phis, x)
    # assert np.all(res == -np.hstack([x[0] * phis[0]] * (nphase - 1))), (
    #     "Unexpected residual values."
    # )
    # # Vice versa if the reference phase partial fractions or fugacities are zero, we
    # # get a stack of x_j * phi_j
    # x = np.array([np.zeros(ncomp)] + [np.ones(ncomp)] * (nphase - 1))
    # phis = np.random.random((nphase, ncomp))
    # res = flash.isofugacity_constraints_res(x, phis)
    # res = flash.isofugacity_constraints_res(x, phis)
    # assert np.all(res == np.hstack([x[j] * phis[j] for j in range(1, nphase)])), (
    #     "Unexpected residual values."
    # )
    # res = flash.isofugacity_constraints_res(phis, x)
    # assert np.all(res == np.hstack([x[j] * phis[j] for j in range(1, nphase)])), (
    #     "Unexpected residual values."
    # )
    # # If all x are zero, or all phis, the result is a zero array.
    # x = np.zeros((nphase, ncomp))
    # phis = np.random.random((nphase, ncomp))
    # res = flash.isofugacity_constraints_res(x, phis)
    # assert np.all(res == 0.0), "Unexpected residual values."
    # res = flash.isofugacity_constraints_res(phis, x)
    # assert np.all(res == 0.0), "Unexpected residual values."

    # # If only 1 phase, the isofugacity constraints return a zero array.
    # x = np.random.random((1, ncomp))
    # phis = np.random.random((1, ncomp))
    # res = flash.isofugacity_constraints_res(x, phis)
    # assert res.shape == (0,), "Unexpected residual shape for 1 phase."

    Xgen = np.random.random((dim,))
    h = np.logspace(0, -9, 10)

    orders = get_EOC_taylor(func, dfunc, Xgen, d, h)
    assert_order_at_least(orders, 2.0, tol=2e-2, asymptotic=6)
