"""Test module for the implementation of the Peng-Robinson EoS.

Note:
    The EoS is implemented using numba-compilation.
    Due to the heavy overhead, it does not test the compiled code, but the Python code.

    This is regulated locally in the file by disabling numba JIT

"""

_NO_JIT = True
"""Flag to disable numba JIT for this module by setting an environment flag.
The flag is deleted at the end of the module.

NOTE: This still may interfer with other tests if pytest does some fancy parallelization
or networking.

"""

from __future__ import annotations
import os

import numpy as np
import pytest

if _NO_JIT:
    os.environ["NUMBA_DISABLE_JIT"] = "1"

import porepy.composite as ppc
from porepy.composite import peng_robinson as ppcpr

# NOTE Fixtures created here are expensive because of compilation
# broadening the scope to have the value cached.

@pytest.fixture(scope='module')
def components() -> tuple[ppcpr.H2O, ppcpr.CO2]:
    """A 2-component mixture with Water and CO2"""
    chems = ["H2O", "CO2"]
    species = ppc.load_species(chems)
    components = [
        ppcpr.H2O.from_species(species[0]),
        ppcpr.CO2.from_species(species[1]),
    ]

    return tuple(components)


@pytest.fixture(scope="module")
def eos(components) -> ppcpr.PengRobinsonCompiler:
    """The fixture for this series of tests is a 2-component mixture with water and CO2."""

    eos = ppcpr.PengRobinsonCompiler(list(components))
    eos.compile()
    return eos


@pytest.fixture(scope='module')
def mixture(components, eos) -> ppc.Mixture:
    """Returns the 2-phase, 2-component mixture class used in this series of tests."""

    phases = [ppc.Phase(eos, 0, 'L'), ppc.Phase(eos, 1, 'G')]
    for p in phases:
        p.components = components

    mixture = ppc.Mixture(components, phases)

    return mixture


@pytest.fixture(scope='module')
def flash(mixture, eos) -> ppc.CompiledUnifiedFlash:
    """Compiled, unified flash instance based on the compiled PR EoS and the
    mixture"""

    flash_= ppc.CompiledUnifiedFlash(mixture, eos)
    flash_.compile()

    return flash_


# @pytest.mark.skip("EoS compilation takes too much time as of now")
def test_compressibility_factor_from_eos(eos: ppcpr.PengRobinsonCompiler):
    """The only (known) double root for the characteristic polynomial
    (2 compressibility factors) is the point where cohesion and co-volume are zero."""
    ncomp = eos._nc
    tol = 1e-12

    # If fractions x are zero, p and T don't matter
    p = 1.0
    T = 1.0
    x = np.array([0.0] * ncomp)

    A_c = eos._cfuncs["A"]
    d_A_c = eos._cfuncs["d_A"]
    B_c = eos._cfuncs["B"]
    d_B_c = eos._cfuncs["d_B"]
    Z_c = eos._cfuncs["Z"]
    d_Z_c = eos._cfuncs["d_Z"]

    # Zero composition leads to zero A and B
    A = A_c(p, T, x)
    B = B_c(p, T, x)
    d_A = d_A_c(p, T, x)
    d_B = d_B_c(p, T, x)
    assert np.abs(A) < tol
    assert np.abs(B) < tol

    z_liq = ppcpr.eos_c.Z_double_l_c(A, B)
    z_gas = ppcpr.eos_c.Z_double_g_c(A, B)
    d_z_liq = ppcpr.eos_c.d_Z_double_l_c(A, B)
    d_z_gas = ppcpr.eos_c.d_Z_double_g_c(A, B)

    # The tailored computations from EoS should give the same result as the formulas
    # dependent on A, B
    assert np.abs(Z_c(p, T, x, True, tol, 0.0, 0.0) - z_gas) < tol
    assert np.abs(Z_c(p, T, x, False, tol, 0.0, 0.0) - z_liq) < tol
    # for testing derivatives, extend dAB to dptx by chain rule
    d_test = d_z_gas[0] * d_A + d_z_gas[1] * d_B
    assert np.linalg.norm(d_Z_c(p, T, x, True, tol, 0.0, 0.0) - d_test) < tol
    d_test = d_z_liq[0] * d_A + d_z_liq[1] * d_B
    assert np.linalg.norm(d_Z_c(p, T, x, False, tol, 0.0, 0.0) - d_test) < tol


def test_compressibility_factor_double_root():
    """The only, easily computable double root is ``A=B=0``.
    (Other double roots are given by vapor- and liquid-saturated lines, which cannot
    be obtained trivially).

    Testing the formula and the general compuation.

    """

    tol = 1e-12

    A = 0.0
    B = 0.0

    # If A and B are zero, this should give the double root case
    root_case = ppcpr.eos_c._get_root_case(A, B, tol)
    assert root_case == 2

    # liquid-like and gas-like root in the double root case should both solve the
    # polynomial exactly
    z_liq = ppcpr.eos_c.Z_double_l_c(A, B)
    z_gas = ppcpr.eos_c.Z_double_g_c(A, B)
    d_z_liq = ppcpr.eos_c.d_Z_double_l_c(A, B)
    d_z_gas = ppcpr.eos_c.d_Z_double_g_c(A, B)
    assert np.abs(ppcpr.eos_c._characteristic_residual(z_gas, A, B)) < tol
    assert np.abs(ppcpr.eos_c._characteristic_residual(z_liq, A, B)) < tol

    # The general calculation of the compressibility factor should give the
    # same result as the formulas
    assert np.abs(ppcpr.eos_c._Z_gen(A, B, True, tol, 0.0, 0.0) - z_gas) < tol
    assert np.abs(ppcpr.eos_c._Z_gen(A, B, False, tol, 0.0, 0.0) - z_liq) < tol
    assert np.linalg.norm(ppcpr.eos_c._d_Z_gen(A, B, True, tol, 0.0, 0.0) - d_z_gas) < tol
    assert np.linalg.norm(ppcpr.eos_c._d_Z_gen(A, B, False, tol, 0.0, 0.0) - d_z_liq) < tol


def test_compressibility_factor_triple_root():
    """The only triple root is at the critical point ``A_CRIT, B_CRIT`` of the
    Peng-Robinson EoS.

    Testing the formula and the general compuation.

    """

    tol = 1e-12

    # critical point in the AB space
    A = ppcpr.A_CRIT
    B = ppcpr.B_CRIT

    # triple root in this case
    z = ppcpr.eos_c.Z_triple_c(A, B)
    d_z = ppcpr.eos_c.d_Z_triple_c(A, B)
    assert np.abs(ppcpr.eos_c._characteristic_residual(z, A, B)) < tol

    nroot = ppcpr.eos_c._get_root_case(A, B, tol)
    # 0 is the code for triple root
    assert nroot == 0

    # Assert general compuations also give the same result, for liquid and gas-like
    assert np.abs(ppcpr.eos_c._Z_gen(A, B, True, tol, 0.0, 0.0,) - z) < tol
    assert np.abs(ppcpr.eos_c._Z_gen(A, B, False, tol, 0.0, 0.0,) - z) < tol
    assert np.linalg.norm(ppcpr.eos_c._d_Z_gen(A, B, True, tol, 0.0, 0.0) - d_z) < tol
    assert np.linalg.norm(ppcpr.eos_c._d_Z_gen(A, B, False, tol, 0.0, 0.0) - d_z) < tol


def test_compressibility_factors_are_roots():
    """Randomized computation of compressibility factors.
    If the result is a non-extended factor, it must be an actual root of the
    characteristic polynomial.

    This test also tests the vectorized computation of the factor."""

    tol = 1e-14
    steps = 0.001

    # testing the vectorized computation with arbitrary A, B pairs
    # If the root is not extended, it should be an actual root of the polynomial
    A, B = np.meshgrid(np.arange(0, 1, steps), np.arange(0, 1, steps))
    A = A.flatten()
    B = B.flatten()
    Z_liq = ppcpr.eos_c.compressibility_factor(A, B, False, tol, 0.0, 0.0)
    not_extended_liq = ppcpr.eos_c.is_real_root(A, B, False, tol)
    residual = ppcpr.eos_c.characteristic_residual(
        Z_liq[not_extended_liq], A[not_extended_liq], B[not_extended_liq]
    )
    assert np.all(np.abs(residual) < tol)

    Z_gas = ppcpr.eos_c.compressibility_factor(A, B, True, tol, 0.0, 0.0)
    not_extended_gas = ppcpr.eos_c.is_real_root(A, B, True, tol)
    residual = ppcpr.eos_c.characteristic_residual(
        Z_gas[not_extended_gas], A[not_extended_gas], B[not_extended_gas]
    )
    assert np.all(np.abs(residual) < tol)

    # where neither liquid root nor gas root are extended (2-phase), the liquid root
    # must be smaller than the gas root.
    # NOTE This may in future also be desirable for extended roots
    not_exteded = not_extended_liq & not_extended_gas
    assert np.all(Z_liq[not_exteded] <= Z_gas[not_exteded])


# @pytest.mark.skip("FLash compilation takes too much time as of now")
@pytest.mark.parametrize(
    ['flash_type', 'X0', 'var_idx_delta'],
    [
        (
            'p-T',
            np.array(  # p-T
                [
                    0.01,  # z_co2
                    9683544.303797469,  # p
                    450.0,  # T
                    0.003224682234577669,  # y_g
                    0.9927348925274755,  # x_h2o_l
                    0.007265107472524585,  # x_co2_l
                    0.14462263564478728,  # x_h2o_g
                    0.8553773643552127,  # x_co2_g
                ]
            ),
            [
                ('y_g', 3, 0.01),
                ('x_h2o_l', 4, 0.01),
                ('x_co2_l', 5, 0.01),
                ('x_h2o_g', 6, 0.01),
                ('x_co2_g', 7, 0.01)
            ]
        ),
        (
            'p-h',
            np.array(  # p-h
                [
                    0.01,  # z_co2
                    -26944.248743227625,  # h
                    6789473.684210526,  # p
                    499.99989708191384,  # T
                    0.007915211517683818,  # y_g
                    0.9942155515034707,  # x_h2o_l
                    0.005784448496529226,  # x_co2_l
                    0.46162695648652396,  # x_h2o_g
                    0.5383730435134759,  # x_co2_g
                ]
            ),
            [
                ('T', 3, 1),
                ('y_g', 4, 0.01),
                ('x_h2o_l', 5, 0.01),
                ('x_co2_l', 6, 0.01),
                ('x_h2o_g', 7, 0.01),
                ('x_co2_g', 8, 0.01)
            ]
        ),
        (
            'v-h',
            np.array(  # v-h
                [
                    0.01,  # z_co2
                    3.267067077646246e-05,  # v
                    -18911.557739855507,  # h
                    0.0,  # s_g
                    15000000.37937989,  # p
                    575.0000000014545,  # T
                    0.0,  # y_g
                    0.99,  # x_h2o_l
                    0.01,  # x_co2_l
                    0.7441053921417927,  # x_h2o_g
                    0.16487463968601462,  # x_co2_g
                ]
            ),
            [
                ("s_g", 3, 0.01),
                ("p", 4, 100),
                ("T", 5, 1),
                ("y", 6, 0.01),
                ("xh20_l", 7, 0.01),
                ("xco2_l", 8, 0.01),
                ("xh2o_g", 9, 0.01),
                ("xco2_g", 10, 0.01),
            ]
        ),
    ],
)
def test_flash_system_using_pr_eos(
    flash_type, X0, var_idx_delta, flash: ppc.CompiledUnifiedFlash
):
    """Computes the Flash equations (Jacobian and residual) and checks that the
    Taylor expansion for each unknown is approximately of second order.
    
    Taylor expansion is calculated for each flash type, around an argument ``X0``,
    which contains the solution of the flash.

    ``var_idx_delta`` contains the name, the index and the delta for the expansion,
    for each variable in respective flash type to be tested.

    """

    tol = 1e-14  # numerical zero for errors

    N = 10  # Number of steps n * delta in Taylor expansion for each unknown

    minimal_order = 1.65  # Lower bound for order to be reached by Taylor expansion
    # NOTE the theoretical order is 2, but this lower bound is required because N
    # is used for each variable and for some of them this is far away from X0

    F = flash.residuals[flash_type]
    DF = flash.jacobians[flash_type]

    F0 = F(X0)
    DF0 = DF(X0)

    for var, i, delta in var_idx_delta:
    
        delta_err = []
        deltas = []

        for k in range(1, N + 1):
            delta_x = np.zeros_like(X0)
            delta_x[i] = k * delta  # +3 because of gen arg
            deltas.append(k * delta)

            F_k = F(X0 + delta_x)
            F_t = F0 + DF0 @ delta_x[3:]

            err = np.linalg.norm(F_k - F_t)
            delta_err.append(err)

        log_err = np.diff(np.log(np.array(delta_err)))
        log_dx = np.diff(np.log(np.array(deltas)))

        # Estimated order of convergence using the mean of delta_err / delta_dx for each
        # step
        eoc = log_err / log_dx
        eoc_mean = np.mean(eoc)
        if np.isnan(eoc_mean):  # this can happen if the error is numerically zero
            assert np.all(np.abs(np.array(delta_err)) < tol)
        else:
            assert (
                eoc_mean >= minimal_order
            ), f"Failure to reach EOC goal for variable {var} for flash {flash_type}"

        assert np.mean(eoc)

if _NO_JIT:
    del os.environ["NUMBA_DISABLE_JIT"]
