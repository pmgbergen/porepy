"""Test module for the implementation of the Peng-Robinson EoS.

Note:
    The EoS is implemented using numba-compilation.
    Due to the heavy overhead, it does not test the compiled code, but the Python code.

    This is regulated locally in the file by disabling numba JIT

"""

from __future__ import annotations

# Disabling JIT, might be in conflict with other tests.
import os

os.environ["NUMBA_DISABLE_JIT"] = "1"

import numpy as np
import pytest

import porepy.composite as ppc
from porepy.composite import peng_robinson as ppcpr


@pytest.fixture(scope="module")
def eos() -> ppcpr.PengRobinsonCompiler:
    """The fixture for this series of tests is a 2-component mixture with water and CO2."""

    chems = ["H2O", "CO2"]
    species = ppc.load_species(chems)
    components = [
        ppcpr.H2O.from_species(species[0]),
        ppcpr.CO2.from_species(species[1]),
    ]

    eos = ppcpr.PengRobinsonCompiler(components)
    eos.compile()
    return eos


# @pytest.mark.skip("EoS compilation takes too much time as of now")
def test_compressibility_from_eos(eos: ppcpr.PengRobinsonCompiler):
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
    assert np.abs(Z_c(1, p, T, x, tol, 0.0, 0.0) - z_gas) < tol
    assert np.abs(Z_c(0, p, T, x, tol, 0.0, 0.0) - z_liq) < tol
    # for testing derivatives, extend dAB to dptx by chain rule
    d_test = d_z_gas[0] * d_A + d_z_gas[1] * d_B
    assert np.linalg.norm(d_Z_c(1, p, T, x, tol, 0.0, 0.0) - d_test) < tol
    d_test = d_z_liq[0] * d_A + d_z_liq[1] * d_B
    assert np.linalg.norm(d_Z_c(0, p, T, x, tol, 0.0, 0.0) - d_test) < tol


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
    assert np.abs(ppcpr.eos_c.Z_c(1, A, B, tol, 0.0, 0.0) - z_gas) < tol
    assert np.abs(ppcpr.eos_c.Z_c(0, A, B, tol, 0.0, 0.0) - z_liq) < tol
    assert np.linalg.norm(ppcpr.eos_c.d_Z_c(1, A, B, tol, 0.0, 0.0) - d_z_gas) < tol
    assert np.linalg.norm(ppcpr.eos_c.d_Z_c(0, A, B, tol, 0.0, 0.0) - d_z_liq) < tol


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
    assert (
        np.abs(
            ppcpr.eos_c.Z_c(
                1,
                A,
                B,
                tol,
                0.0,
                0.0,
            )
            - z
        )
        < tol
    )
    assert (
        np.abs(
            ppcpr.eos_c.Z_c(
                0,
                A,
                B,
                tol,
                0.0,
                0.0,
            )
            - z
        )
        < tol
    )
    assert np.linalg.norm(ppcpr.eos_c.d_Z_c(1, A, B, tol, 0.0, 0.0) - d_z) < tol
    assert np.linalg.norm(ppcpr.eos_c.d_Z_c(0, A, B, tol, 0.0, 0.0) - d_z) < tol


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
    Z_liq = ppcpr.eos_c.Z_u(0, A, B, tol, 0.0, 0.0)
    not_extended = ppcpr.eos_c.is_real_root(np.zeros_like(A, dtype=np.int8), A, B, tol)
    residual = ppcpr.eos_c.characteristic_residual(
        Z_liq[not_extended], A[not_extended], B[not_extended]
    )
    assert np.all(np.abs(residual) < tol)

    Z_gas = ppcpr.eos_c.Z_u(1, A, B, tol, 0.0, 0.0)
    not_extended = ppcpr.eos_c.is_real_root(np.ones_like(A, dtype=np.int8), A, B, tol)
    residual = ppcpr.eos_c.characteristic_residual(
        Z_gas[not_extended], A[not_extended], B[not_extended]
    )
    assert np.all(np.abs(residual) < tol)


del os.environ["NUMBA_DISABLE_JIT"]
