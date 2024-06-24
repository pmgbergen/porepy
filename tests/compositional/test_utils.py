"""Module testing utility funtions of the ``compositional`` sub-package."""

from __future__ import annotations

import numpy as np
import pytest

import porepy.compositional as composit

# NOTE to disable numba compilation and debug tests
import os
os.environ["NUMBA_DISABLE_JIT"] = "1"


@pytest.mark.parametrize("vectorized", [0, 1, 3])
@pytest.mark.parametrize("nphase", [1, 2, 3])
def test_compute_saturations(nphase: int, vectorized: int):
    """Tests :meth:`~porepy.compositional.utils.compute_saturations` for number of
    phases ``nphase``.

    Tests the correctness for saturated and unsaturated cases, as well as the vectorized
    computations, with ``vectorized`` indicating the length of the vectorized
    computations.

    """
    # shape of input and output formats
    if vectorized > 0:
        shape = (nphase, vectorized)
    else:
        shape = (nphase,)

    # Testing is always performed with homogenous densities: If all densities are the
    # same, the resulting saturation values should be equal to the fraction values
    rho = np.ones(shape)

    # Inconsistent shapes between fractions and densities gives an value error
    with pytest.raises(ValueError):
        composit.compute_saturations(np.ones((nphase + 1, vectorized)), rho)
    with pytest.raises(ValueError):
        composit.compute_saturations(rho / nphase, np.ones((nphase, vectorized + 1)))

    # special case, single-phase is always saturated
    if nphase == 1:
        y = np.ones(shape)
        sat = composit.compute_saturations(y, rho)
        assert sat.shape == shape
        assert np.all(sat == 1.0)
    # for multi-phase, tests include saturated cases for each phase
    if nphase > 1:
        # non-saturated case, homogenous distribution of mass
        y = np.ones(shape)

        # More than 1 phase saturated gives a value error
        with pytest.raises(ValueError):
            composit.compute_saturations(y, rho)

        # homogenous distribution of mass accross phases, with equal densities
        # should lead to saturations equal to y
        y = y / nphase
        sat = composit.compute_saturations(y, rho)
        assert sat.shape == shape
        assert np.allclose(y, sat, rtol=0.0, atol=1e-14)

        # testing cases where 1 phase vanishes

        for j in range(nphase):
            y = np.ones(shape)
            y[j] = 0.0
            # homogenous distribution of mass accross non-vanished phases
            y = y / (nphase - 1)

            sat = composit.compute_saturations(y, rho)
            assert sat.shape == shape
            assert np.allclose(y, sat, rtol=0.0, atol=1e-14)


def test_chainrule_fractional_derivatives():
    """Tests the expansion of fractional derivatives to derivatives w.r.t. extended
    fractions."""

    # derivatives w.r.t. to fractions are in the last 3 rows.
    # They are such that the i-th derivative contains non-zero external derivatives,
    # but the other contain the internal derivative of the normalization.
    # I.e., the result of sending df through the computations should give exactly
    # the Jacobian of the normalization x_i = y_i / sum_j y_j
    # dy_k x_i = delta_ki 1 / sum_j y_j - y_i / (sum_j y_j)^2
    # in the last 3 rows
    df = np.vstack([np.ones((2, 3)) * 20., np.eye(3)])
    
    x_ext = np.array([0.1, 0.2, 0.3])

    s = x_ext.sum()
    jac = np.eye(3) / s - np.outer(x_ext, np.ones(3)) / (s**2)

    x_ext_v = np.vstack([x_ext] * 3).T
    df_ext = composit.chainrule_fractional_derivatives(df, x_ext_v)

    assert df_ext.shape == df.shape
    assert np.allclose(df_ext[2:], jac.T, rtol=0.,  atol=1e-14)
    # other derivatives are left untouched
    assert np.allclose(df_ext[:2], 20, rtol=0, atol=1e-14)

    # slicing columns should mimic non-vectorized computations. assert shape-consistency
    df_0 = df[:, 0]
    assert df_0.shape == (5,)
    df_ext_0 = composit.chainrule_fractional_derivatives(df_0, x_ext)
    assert df_ext_0.shape == df_0.shape
    assert np.allclose(df_ext_0[2:], jac[0], rtol=0.,  atol=1e-14)
    assert np.allclose(df_ext_0[:2], 20., rtol=0, atol=1e-14)
