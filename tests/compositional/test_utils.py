"""Module testing utility funtions of the ``compositional`` sub-package."""

from __future__ import annotations

import numpy as np
import pytest

import porepy.compositional as composit


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

    # This tests mimics the values of a function, depending on 2 state functions and
    # 3 component fractions, evaluated on 3 cells.
    # The global Jacobian has hence 2 + 3 rows and 3 columns.
    # The derivatives w.r.t. to fractions are in the last 3 rows.
    # In each cell, the derivative w.r.t fractions is 1 for only 1 component, hence
    # that row-block is the identity.
    # Assuming that we want to expand the derivatives to extended fractions,
    # which are obtained by normalization of (normal) fractions, this identity block
    # shoud turn into the Jacobian of the normalization function
    # x_i = y_i / sum_j y_j
    # dy_k x_i = delta_ki 1 / sum_j y_j - y_i / (sum_j y_j)^2

    # Construct the derivatives of the functions, with value 20 for the 2 state
    # functions. Those values should not be changed by the tested method
    df = np.vstack([np.ones((2, 3)) * 20.0, np.eye(3)])

    # Set some arbitrary values for the extended fractions for 3 components
    # We use the same value on all 3 cells, in order to obtain the Jacobian of the
    # normalization
    x_ext = np.array([0.1, 0.2, 0.3])
    x_ext_v = np.vstack([x_ext] * 3).T

    # Analytical derivative with respect to the extended fractions.
    s = x_ext.sum()
    jac = np.eye(3) / s - np.outer(x_ext, np.ones(3)) / (s**2)

    # Compute the chainrule with the method
    df_ext = composit.chainrule_fractional_derivatives(df, x_ext_v)

    # The jacobian should be un-altered in terms of shape and the first two rows, which
    # are considered to be derivatives w.r.t to some other independent variables
    assert df_ext.shape == df.shape
    assert np.allclose(df_ext[:2], 20, rtol=0, atol=1e-14)

    # The derivatives w.r.t. extended fractions should match the analytical solution
    assert np.allclose(df_ext[2:], jac.T, rtol=0.0, atol=1e-14)

    # slicing columns should mimic non-vectorized computations
    # This tests the consistency of the method, when passing arguments as 1D arrays
    df_0 = df[:, 0]
    assert df_0.shape == (5,)
    df_ext_0 = composit.chainrule_fractional_derivatives(df_0, x_ext)
    assert df_ext_0.shape == df_0.shape
    assert np.allclose(df_ext_0[:2], 20.0, rtol=0, atol=1e-14)
    assert np.allclose(df_ext_0[2:], jac[0], rtol=0.0, atol=1e-14)
