"""Module testing assembly of common flash equations as well as generic argument
parsing."""

import pytest

# import os

# os.environ["NUMBA_DISABLE_JIT"] = "1"

import numpy as np
import porepy as pp

import porepy.compositional.flash as flash
from tests.compositional.peng_robinson.test_cubic_polynomial import (
    get_EOC_taylor,
    assert_order_at_least,
)


def _dummy_property(p: float, T: float, x: np.ndarray) -> float:
    """A dummy phase property as assumed by the flash framework.

    It takes a pressure and temperature value, as well as partial fractions and returns
    a float.

    """
    return p**2 + T**2 + (x**2).sum()


def _dummy_property_derivative(p: float, T: float, x: np.ndarray) -> float:
    """A dummy phase property derivative as assumed by the flash framework.

    Contains the analytical derivative of :func:`_dummy_property`.

    It takes a pressure and temperature value, as well as partial fractions and returns
    an array with shape ``(2 + x.size,)``

    """
    n = x.shape[0]
    assert x.shape == (n,)
    d = np.array([2 * p, 2 * T] + [2 * x_ for x_ in x])
    assert d.shape == (2 + n,)
    return d


@pytest.mark.parametrize(
    "spec", [spec for spec in flash.FlashSpec if spec != flash.FlashSpec.none]
)
@pytest.mark.parametrize("with_params", [True, False])
@pytest.mark.parametrize("nphase", [1, 2, 5])
@pytest.mark.parametrize("ncomp", [1, 2, 5])
@pytest.mark.parametrize("vectorized", [True, False])
def test_assembly_and_parsing_of_generic_flash_argument(
    vectorized: bool,
    ncomp: int,
    nphase: int,
    spec: flash.FlashSpec,
    with_params: bool,
) -> None:
    if vectorized:
        N = 10
        parser = flash.parse_vectorized_generic_arg
        assembler = flash.assemble_vectorized_generic_arg
    else:
        N = 0
        parser = flash.parse_generic_arg
        assembler = flash.assemble_generic_arg

    if with_params:
        dp = np.random.randint(1, 10)
        params = np.random.random((dp, N) if vectorized else (dp,))
    else:
        params = np.zeros((0, N) if vectorized else (0,))

    # Make sure all values distinct.
    d = flash.dim_gen_arg(ncomp, nphase, spec)
    non_params = np.random.choice(
        np.arange(0, 100000),
        replace=False,
        size=(d, N) if vectorized else (d,),
    )

    if vectorized:
        Xgen = np.vstack([params, non_params]).transpose()
    else:
        Xgen = np.hstack([params, non_params])

    sat, x, y, z, p, T, state1, state2, pars = parser(Xgen, ncomp, nphase, spec)

    if vectorized:
        assert sat.shape == (nphase, N), "Saturations of unexpected shape."
        assert x.shape == (nphase, ncomp, N), "Partial fractions of unexpected shape."
        assert y.shape == (nphase, N), "Phase fractions of unexpected shape."
        assert z.shape == (ncomp, N), "Overall compositions of unexpected shape."
        assert p.shape == (N,), "Pressure of unexpected shape."
        assert T.shape == (N,), "Temperature of unexpected shape."
        assert state1.shape == (N,), "State value 1 of unexpected shape."
        assert state2.shape == (N,), "State value 2 of unexpected shape."
    else:
        assert sat.shape == (nphase,), "Saturations of unexpected shape."
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

    if spec in [flash.FlashSpec.pT, flash.FlashSpec.vT]:
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
    if spec > flash.FlashSpec.vT:
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

    Xgen2 = assembler(sat, x, y, z, p, T, state1, state2, pars, spec)
    assert np.all(Xgen == Xgen2), (
        "Parsed and re-assembled generic arg expected to be equal to original arg,"
    )

    # Sanity check that the values remain the same and there is no accidental
    # cancelation of errors.
    sat2, x2, y2, z2, p2, T2, state12, state22, pars2 = parser(
        Xgen2.copy(), ncomp, nphase, spec
    )
    assert np.all(sat2 == sat)
    assert np.all(x2 == x)
    assert np.all(y2 == y)
    assert np.all(z == z)
    assert np.all(p2 == p)
    assert np.all(T2 == T)
    assert np.all(state12 == state1)
    assert np.all(state22 == state2)
    assert np.all(pars2 == pars)

    Xgen3 = assembler(sat2, x2, y2, z2, p2, T2, state12, state22, pars2, spec)
    assert np.all(Xgen == Xgen3)


def test_parsing_with_no_flash_spec() -> None:
    """Expected to fail."""
    ncomp = np.random.randint(1, 10)
    nphase = np.random.randint(1, 10)

    Xgen = np.random.random((ncomp * nphase,))

    with pytest.raises(ValueError):
        _ = flash.dim_gen_arg(ncomp, nphase, flash.FlashSpec.none)
    with pytest.raises(ValueError):
        _ = flash.parse_generic_arg(Xgen, ncomp, nphase, flash.FlashSpec.none)


@pytest.mark.parametrize(
    "spec",
    [  # NOTE: add more onces isoenergetic definitions u are supported by FluidProperty
        flash.FlashSpec.pT,
        flash.FlashSpec.ph,
        flash.FlashSpec.vh,
    ],
)
@pytest.mark.parametrize("with_params", [True, False])
@pytest.mark.parametrize("with_init", [True, False])
@pytest.mark.parametrize("nphase", [1, 2, 5])
@pytest.mark.parametrize("ncomp", [1, 2, 5])
@pytest.mark.parametrize("N", [1, 10])
def test_generic_arg_from_result_struture(
    N: int, ncomp: int, nphase: int, with_params: bool, with_init, spec: flash.FlashSpec
) -> None:
    """Tests the assembly of the generic argument using a flash results structure."""

    z = np.random.random((ncomp, N))
    p = np.random.random((N,))
    T = np.random.random((N,))
    h = np.random.random((N,))
    rho = np.random.random((N,))
    y = np.random.random((nphase, N))
    sat = np.random.random((nphase, N))
    x = np.random.random((nphase, ncomp, N))

    match spec:
        case flash.FlashSpec.pT:
            state1 = p.copy()
            state2 = T.copy()
        case flash.FlashSpec.ph:
            state1 = p.copy()
            state2 = h.copy()
        case flash.FlashSpec.vh:
            state1 = 1 / rho.copy()
            state2 = h.copy()

    if with_params:
        params = np.random.random((np.random.randint(1, 10), N))
    else:
        params = np.random.random((0, N))

    results = flash.FlashResults(
        specification=spec,
        size=N,
        p=p,
        T=T,
        z=z,
        y=y,
        sat=sat,
        h=h,
        rho=rho,
        phases=[pp.compositional.PhaseProperties(x=x[j, :, :]) for j in range(nphase)],
    )

    XgenA = flash.assemble_vectorized_generic_arg(
        sat, x, y, z, p, T, state1, state2, params, spec
    )
    XgenB = flash.generic_arg_from_flash_results(
        results, ncomp, nphase, with_init, params if with_params else None
    )

    # If all values are used, they must be identical.
    if with_init:
        assert np.all(XgenA == XgenB)
    # If not all values are used, the ones associated with degrees of freedom must be 0.
    else:
        satb, xb, yb, zb, pb, Tb, st1b, st2b, paramsb = (
            flash.parse_vectorized_generic_arg(XgenB, ncomp, nphase, spec)
        )

        assert np.all(xb == 0.0)
        # The parsing assembles the reference entities by unity of fractions
        assert np.all(yb[0] == 1)
        assert np.all(satb[0] == 1)
        if nphase > 1:
            assert np.all(satb[1:] == 0)
            assert np.all(yb[1:] == 0)

        if ncomp == 1:
            assert np.all(zb == 1)
        else:
            assert np.all(zb[1:] == z[1:])
            assert np.all(zb[0] == 1 - z[1:].sum(axis=0))

        assert np.all(st1b == state1)
        assert np.all(st2b == state2)
        assert np.all(paramsb == params)

        if flash.FlashSpec.none < spec < flash.FlashSpec.vT:
            assert np.all(p == pb)
            assert np.all(p == st1b)
            if spec == flash.FlashSpec.pT:
                assert np.all(T == Tb)
                assert np.all(T == st2b)
            elif spec == flash.FlashSpec.ph:
                assert np.all(h == st2b)
            else:
                assert False, "Missing test logic"
        elif spec >= flash.FlashSpec.vT:
            assert np.all(1 / rho == st1b)
            assert np.all(pb == 0)
            if spec == flash.FlashSpec.vT:
                assert np.all(T == Tb)
                assert np.all(T == st2b)
            elif spec == flash.FlashSpec.vh:
                assert np.all(Tb == 0)
                assert np.all(h == st2b)
            else:
                assert False, "Missing test logic"


# NOTE: The directions for the derivatives in the Taylor approximation test implemented
# in the tests below assume a certain knowledge about the structure of the generic
# flash argument. If that changes, the tests need adaption.


@pytest.mark.parametrize("nphase", [1, 2, 5])
@pytest.mark.parametrize("ncomp", [2, 5])
def test_mass_conservation(ncomp: int, nphase: int) -> None:
    """Tests if the mass conservation equation is correctly implemented and its
    Jacobian function allows the Taylor approximation to be of second order."""
    spec = flash.FlashSpec.pT
    dim = flash.dim_gen_arg(ncomp, nphase, spec)
    nf = ncomp * nphase + nphase - 1
    directions = np.hstack((np.zeros((nf, dim - nf)), np.eye(nf)))

    def func(*x):
        xgen = np.array(x)
        _, x, y, z, *_ = flash.parse_generic_arg(xgen, ncomp, nphase, spec)
        res = flash.mass_conservation_res(x, y, z)
        assert res.shape == (ncomp - 1,), "Residual of unexpected shape."
        return res

    def dfunc(*x):
        xgen = np.array(x)
        _, x, y, *_ = flash.parse_generic_arg(xgen, ncomp, nphase, spec)
        jac = flash.mass_conservation_jac(x, y)
        assert jac.shape == (ncomp - 1, nf + 2 + nphase - 1), (
            "Jacobian of unexpected shape."
        )
        assert np.all(jac[:, : 2 + nphase - 1] == 0), (
            "Jacobian has non-trivial derivatives for p, T and sat."
        )
        return np.hstack((np.zeros((ncomp - 1, dim - nf)), jac[:, -nf:]))

    # Whatever z and x are, if y or x is zero we expect values -z
    z = np.random.random((ncomp,))
    y = np.zeros(nphase)
    x = np.random.random((nphase, ncomp))
    res = flash.mass_conservation_res(x, y, z)
    assert np.all(res == -z[1:]), "Unexpected residual values."
    y = np.random.random((nphase,))
    x = np.zeros((nphase, ncomp))
    res = flash.mass_conservation_res(x, y, z)
    assert np.all(res == -z[1:]), "Unexpected residual values."
    # If x = 1 and y = 1/nphase (homogenous mass distribution), result should be 1 - z
    y = np.ones(nphase) / nphase
    x = np.ones((nphase, ncomp))
    res = flash.mass_conservation_res(x, y, z)
    assert np.all(res == 1.0 - z[1:]), "Unexpected residual values."

    # If only 1 component, the mass conservation equations should be empty.
    assert flash.mass_conservation_res(x, y, np.ones(1)).shape == (0,), (
        "Unexpacted residual shape for 1 component."
    )

    Xgen = np.random.random((dim,))
    h = np.logspace(0, -10, 11)

    for d in directions:
        orders = get_EOC_taylor(func, dfunc, Xgen, d, h)
        assert_order_at_least(orders, 2.0, tol=1e-3)


@pytest.mark.parametrize("nphase", [1, 2, 5])
@pytest.mark.parametrize("ncomp", [1, 2, 5])
def test_complementary_conditions(ncomp: int, nphase: int) -> None:
    """Tests if the complementary conditions are correctly implemented and its
    Jacobian function allows the Taylor approximation to be of second order.

    We test the smooth version without the inequality constraints. I.e., just the
    multiplication of fractions and unity of partial fractions per phase.

    """
    spec = flash.FlashSpec.pT
    dim = flash.dim_gen_arg(ncomp, nphase, spec)
    nf = ncomp * nphase + nphase - 1
    directions = np.hstack((np.zeros((nf, dim - nf)), np.eye(nf)))

    def func(*x):
        xgen = np.array(x)
        _, x, y, *_ = flash.parse_generic_arg(xgen, ncomp, nphase, spec)
        res = flash.complementary_conditions_res(x, y)
        assert res.shape == (nphase,), "Residual of unexpected shape."
        return res

    def dfunc(*x):
        xgen = np.array(x)
        _, x, y, *_ = flash.parse_generic_arg(xgen, ncomp, nphase, spec)
        jac = flash.complementary_conditions_jac(x, y)
        assert jac.shape == (nphase, nf + 2 + nphase - 1), (
            "Jacobian of unexpected shape."
        )
        assert np.all(jac[:, : 2 + nphase - 1] == 0), (
            "Jacobian has non-trivial derivatives for p, T and sat."
        )
        return np.hstack((np.zeros((nphase, dim - nf)), jac[:, -nf:]))

    # If y is zero, the complementary conditions are zero.
    y = np.zeros(nphase)
    x = np.random.random((nphase, ncomp))
    res = flash.complementary_conditions_res(x, y)
    assert np.all(res == 0.0), "Unexpected residual values."
    # If x are homogenous, the unity of fractions leads to zero.
    y = np.random.random((nphase,))
    x = np.ones((nphase, ncomp)) / ncomp
    res = flash.complementary_conditions_res(x, y)
    assert np.all(res == 0.0), "Unexpected residual values."
    # If x = 0, we should get y
    y = np.random.random((nphase,))
    x = np.zeros((nphase, ncomp))
    res = flash.complementary_conditions_res(x, y)
    assert np.all(res == y), "Unexpected residual values."
    # If y = 1, we should get the unity of fractions
    y = np.ones(nphase)
    x = np.random.random((nphase, ncomp))
    res = flash.complementary_conditions_res(x, y)
    assert np.all(res == 1 - x.sum(axis=1)), "Unexpected residual values."

    Xgen = np.random.random((dim,))
    h = np.logspace(0, -10, 11)

    for d in directions:
        orders = get_EOC_taylor(func, dfunc, Xgen, d, h)
        assert_order_at_least(orders, 2.0, tol=1e-3)


@pytest.mark.parametrize("nphase", [1, 2, 5])
@pytest.mark.parametrize("ncomp", [1, 2, 5])
@pytest.mark.parametrize("w_flag", [True, False])
def test_first_order_constraint(w_flag: bool, ncomp: int, nphase: int) -> None:
    """Tests if the first-order constraint is correctly implemented and its
    Jacobian function allows the Taylor approximation to be of second order."""
    spec = flash.FlashSpec.vh
    dim = flash.dim_gen_arg(ncomp, nphase, spec)
    nf = ncomp * nphase + 2 * (nphase - 1) + 2
    directions = np.hstack((np.zeros((nf, dim - nf)), np.eye(nf)))

    # Target value of the constraint.
    phi = np.random.rand()

    def func(*x):
        xgen = np.array(x)
        sat, x, y, _, p, T, *_ = flash.parse_generic_arg(xgen, ncomp, nphase, spec)
        phis = np.array([_dummy_property(p, T, x_) for x_ in x])
        res = flash.first_order_constraint_res(phi, y if w_flag else sat, phis)
        assert res.shape == (1,), "Residual of unexpected shape."
        return res

    def dfunc(*x):
        xgen = np.array(x)
        sat, x, y, _, p, T, *_ = flash.parse_generic_arg(xgen, ncomp, nphase, spec)
        phis = np.array([_dummy_property(p, T, x_) for x_ in x])
        dphis = np.array([_dummy_property_derivative(p, T, x_) for x_ in x])
        jac = flash.first_order_constraint_jac(
            y if w_flag else sat, phis, dphis, w_flag
        )
        assert jac.shape == (1, nf), "Jacobian of unexpected shape."
        if w_flag:
            assert np.all(jac[:, 2 : 2 + nphase - 1] == 0), (
                "Jacobian has non-trivial derivatives for sat."
            )
        else:
            assert np.all(jac[:, 2 + nphase - 1 : 2 + 2 * (nphase - 1)] == 0), (
                "Jacobian has non-trivial derivatives for y."
            )
        return np.hstack((np.zeros((1, dim - nf)), jac))

    # If weights are zero, or the phis are zero, we expect -phi
    w = np.zeros(nphase)
    phis = np.random.random((nphase,))
    res = flash.first_order_constraint_res(phi, w, phis)
    assert np.all(res == -phi), "Unexpected residual values."
    res = flash.first_order_constraint_res(phi, phis, w)
    assert np.all(res == -phi), "Unexpected residual values."
    # If target value is zero, we expect the dot product of the weights and phis
    w = np.random.random((nphase,))
    res = flash.first_order_constraint_res(0.0, w, phis)
    assert np.allclose(res, np.dot(w, phis), rtol=0.0), "Unexpected residual values."
    res = flash.first_order_constraint_res(0.0, phis, w)
    assert np.allclose(res, np.dot(w, phis), rtol=0.0), "Unexpected residual values."
    # If w=1 and phis = phi / nphase, we expect zero residual
    w = np.ones(nphase)
    phis = np.ones(nphase) * phi / nphase
    res = flash.first_order_constraint_res(phi, w, phis)
    assert np.allclose(res, 0, rtol=0.0), "Unexpected residual values."
    res = flash.first_order_constraint_res(phi, phis, w)
    assert np.allclose(res, 0, rtol=0.0), "Unexpected residual values."

    Xgen = np.random.random((dim,))
    h = np.logspace(0, -10, 11)

    for d in directions:
        orders = get_EOC_taylor(func, dfunc, Xgen, d, h)
        assert_order_at_least(orders, 2.0, tol=1e-2)


@pytest.mark.parametrize("nphase", [1, 2, 5])
@pytest.mark.parametrize("ncomp", [1, 2, 5])
def test_phase_mass_constraint(ncomp: int, nphase: int) -> None:
    """Tests if the phase mass constraints are correctly implemented and its
    Jacobian function allows the Taylor approximation to be of second order."""
    spec = flash.FlashSpec.vh
    dim = flash.dim_gen_arg(ncomp, nphase, spec)
    nf = ncomp * nphase + 2 * (nphase - 1) + 2
    directions = np.hstack((np.zeros((nf, dim - nf)), np.eye(nf)))

    def func(*x):
        xgen = np.array(x)
        sat, x, y, _, p, T, *_ = flash.parse_generic_arg(xgen, ncomp, nphase, spec)
        phis = np.array([_dummy_property(p, T, x_) for x_ in x])
        res = flash.phase_mass_constraints_res(sat, y, phis)
        assert res.shape == (nphase - 1,), "Residual of unexpected shape."
        return res

    def dfunc(*x):
        xgen = np.array(x)
        sat, x, y, _, p, T, *_ = flash.parse_generic_arg(xgen, ncomp, nphase, spec)
        phis = np.array([_dummy_property(p, T, x_) for x_ in x])
        dphis = np.array([_dummy_property_derivative(p, T, x_) for x_ in x])
        jac = flash.phase_mass_constraints_jac(sat, y, phis, dphis)
        assert jac.shape == (nphase - 1, nf), "Jacobian of unexpected shape."
        return np.hstack((np.zeros((nphase - 1, dim - nf)), jac))

    # If saturations are zero, the constraint is zero.
    y = np.random.random((nphase,))
    rhos = np.random.random((nphase,))
    sat = np.zeros(nphase)
    res = flash.phase_mass_constraints_res(sat, y, rhos)
    assert np.all(res == 0.0), "Unexpected residual values."
    # If fractions are zero, the constraint is -s
    res = flash.phase_mass_constraints_res(y, sat, rhos)
    assert np.all(res == -y[1:]), "Unexpected residual values."
    # For individual zero entries, the residual is never zero if one saturation vanishes
    # but elementwise zero where the fraction vanishes.
    for j in range(nphase - 1):
        y = np.random.random((nphase,)) + 1
        rhos = np.random.random((nphase,)) + 1
        sat = np.random.random((nphase,)) + 1
        sat[j + 1] = 0.0
        res = flash.phase_mass_constraints_res(sat, y, rhos)
        assert np.all(res != 0.0), "Unexpected residual values."
        res = flash.phase_mass_constraints_res(y, sat, rhos)
        assert np.all(res[j] == -y[j + 1]), "Unexpected residual values."

    Xgen = np.random.random((dim,))
    h = np.logspace(0, -10, 11)

    for d in directions:
        orders = get_EOC_taylor(func, dfunc, Xgen, d, h)
        assert_order_at_least(orders, 2.0, tol=1e-2, asymptotic=7)
