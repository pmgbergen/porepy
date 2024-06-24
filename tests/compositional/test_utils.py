"""Module testing utility funtions of the ``compositional`` sub-package."""

from __future__ import annotations

import numpy as np
import pytest

import porepy.compositional as composit

# NOTE to disable numba compilation and debug tests
# import os
# os.environ["NUMBA_DISABLE_JIT"] = "1"


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
