""" Test of interpolation functions
"""
import numpy as np
import pytest

import porepy as pp


class _LinearFunc:
    def __init__(self, nd, *args):
        self.coeff = np.random.rand(nd, 1)
        self.tol = 1e-10
        self.diff_tol = self.tol

    def truth(self, x):
        return np.sum(self.coeff * x, axis=0)

    def func(self):
        def f(*args):
            val = 0
            for (coeff, x) in zip(self.coeff, args):
                val += coeff * x
            return val

        return f

    def diff(self, x, axis):
        return self.coeff[axis] * np.ones(x.shape[1])


class _QuadraticFunc:
    # Quadratic polynomical of given dimension
    def __init__(self, nd, h):
        self.coeff = np.random.rand(nd, 1)

        # Error estimate for a quadratic function
        self.tol = 2 * np.max(self.coeff) * np.max(h) / 4
        # Interpolation error unclear at the moment.
        # Assign large value - this will in effect invalidate
        # the test for this class of functions
        self.diff_tol = 4

    def truth(self, x):
        return np.sum(self.coeff * np.power(x, 2), axis=0)

    def func(self):
        def f(*args):
            val = 0
            for (coeff, x) in zip(self.coeff, args):
                val += coeff * x**2
            return val

        return f

    def diff(self, x, axis):
        return 2 * self.coeff[axis] * x[axis]


@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("factory", [_LinearFunc, _QuadraticFunc])
def test_interpolation(dim, factory):
    low = np.random.rand(dim)
    high = 2 + np.random.rand(dim)
    low = np.zeros(dim)
    high = 2 * np.ones(dim)
    num_pt = np.array([4 + d for d in range(dim)])
    max_size = (high - low) / (num_pt - 1)
    function = factory(dim, max_size)

    # We have occationally had issues with these tests failing on random data, however,
    # subsequent testing (on huge arrays) failed to uncover any issues. Fix the random
    # seed so that if errors arise, they should be reproducible.
    # TODO: We should rather use hypothesis for this part.
    np.random.seed(42)

    # Create interpolation tables
    table = pp.InterpolationTable(low, high, num_pt, function.func())
    adaptive_table = pp.AdaptiveInterpolationTable(
        dx = (high - low) / (num_pt - 1), base_point = low, function = function.func(), dim=1
    )
    
    coord = np.vstack(table._coord)
    
    adaptive_vals = adaptive_table.interpolate(coord)
    table_vals = table.interpolate(coord)
    known_vals = function.truth(coord)
    
    assert np.allclose(table_vals, adaptive_vals, atol=1e-10)
    assert np.allclose(known_vals, adaptive_vals, atol=1e-10)

    # Create a new adaptive table, so that we can evaluate it on a rather sparse point
    # set and check that we have not used all the quadrature points.
    adaptive_table = pp.AdaptiveInterpolationTable(
        dx = (high - low) / (num_pt - 1), base_point = low, function = function.func(), dim=1
    )

    npt = 10

    pts = low.reshape((-1, 1)) + np.random.rand(dim, npt) * (high - low).reshape(
        (-1, 1)
    )

    adaptive_vals = adaptive_table.interpolate(pts)
    table_vals = table.interpolate(pts                                 )

    # Check that the adaptive table has used fewer points than the full version
    # Worst case scenario is that all evaluation points trigger a new
    # evaluation, or that all points have been evaluated.
    num_quad_pts = adaptive_table._values.size
    assert num_quad_pts <= np.minimum(npt * np.power(2, dim), table._values.shape[-1])

    known_vals = function.truth(pts)

    # Function values should be reproduced with sufficient accuracy
    assert np.max(np.abs(table_vals - known_vals)) < function.tol
    # The two interpolation approaches should be identical
    assert np.allclose(table_vals, adaptive_vals, atol=1e-10)

    for d in range(dim):
        diff = table.diff(pts, axis=d)
        adaptive_diff = adaptive_table.diff(pts, axis=d)

        # It should not be necessary to compute new function values
        assert adaptive_table._values.size == num_quad_pts

        known_diff = function.diff(pts, d)
        # Note that for the time being, this test is not active for the
        # Quadractic function (due to diff_tol)
        assert np.max(np.abs(diff - known_diff)) < function.diff_tol
        assert np.allclose(diff, adaptive_diff, atol=1e-10)

test_interpolation(4, _QuadraticFunc)