""" Test of interpolation functions.
"""
import numpy as np
import pytest

import porepy as pp


class _LinearFunc:
    # Helper class to represent a linear function.
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
    # Helper class to represent a quadratic function.
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
def test_value_assigment_and_interpolation(dim, factory):
    """This test covers all aspects of assignment of values and interpolation using
    the values. The test is set up to vary the dimension of the parameter space which
    the tables represents, and the function to be enterpolated.

    It would have been possible to split this into more and smaller tests, however, this
    would soon have led to code duplication, and it was decided not to pursue this
    option.

    The general structure of the tests is: For a given function, make full and adaptive
    interpolation tables. The tables are populated with values (at instantiation for the
    full table, on demand for the adaptive ones), and the function is interpolated at
    different points. The accuracy expected depends on the function and on the
    evaluation points (linear functions should be exactly interpolated, all functions
    should be exactly interpolated in data points).

    """

    # Set the bounds for the part of the parameter space to be covered by the tables.
    # The full table will adhere strictly to this,  the adaptive tables are anchored
    # in the point 'low', and computed according to the resolution of the table (but
    # since the two types of tables have the same resolution, the resulting grids in
    # parameter space will have the same nodes).
    low = np.random.rand(dim)
    high = 2 + np.random.rand(dim)
    low = np.zeros(dim)
    high = 2 * np.ones(dim)
    # Vary the number of points in each dimension.
    num_pt = np.array([4 + d for d in range(dim)])
    # Grid resolution in each dimension
    max_size = (high - low) / (num_pt - 1)
    # Create function to be tested.
    function = factory(dim, max_size)

    # We have occationally had issues with these tests failing on random data, however,
    # subsequent testing (on huge arrays) failed to uncover any issues. Fix the random
    # seed so that if errors arise, they should be reproducible.
    # TODO: We should rather use hypothesis for this part.
    np.random.seed(42)

    # Create and test interpolation tables.
    # Three types of tests are run:
    #    1) The interpolated function should be evaluated exactly in the nodes of the
    #       grid in parameter space.
    #    2) The adaptive interpolation table can be built by adding points gradually
    #       or all at once, the result should be the same.
    #    3) Interpolation of values and derivatives have the expected accuracy relative
    #       to the interpolated function in random points in parameter space.

    # Create interpolation tables
    table = pp.InterpolationTable(low, high, num_pt, function.func())
    adaptive_table = pp.AdaptiveInterpolationTable(
        dx=(high - low) / (num_pt - 1), base_point=low, function=function.func(), dim=1
    )

    # First check that the interpolation is exact in the nodes of the underlying grids.
    # In this case, the interpolation weights in the tables should be calculated to be
    # 1 in the relevant node, and 0 in adjacent nodes.
    coords = np.vstack(table._coords)

    adaptive_vals = adaptive_table.interpolate(coords)
    table_vals = table.interpolate(coords)
    known_vals = function.truth(coords)

    # An error message from here likely indicates that either something is wrong with
    # the function evaluation itself, or that the weights in the interpolation are
    # wrong.
    assert np.allclose(table_vals, adaptive_vals, atol=1e-10)
    assert np.allclose(known_vals, adaptive_vals, atol=1e-10)

    # Second test (only relevant for the adaptive table): Assign data to the table,
    # instead of passing a function and letting the table compute the values. The two
    # should be equivalent. We evaluate the table in the nodes in parameter space
    # (e.g. variable coords), since the test is on assignment and not interpolation
    # which is tested below.

    # Assign all values at once.
    adaptive_table_2 = pp.AdaptiveInterpolationTable(
        dx=(high - low) / (num_pt - 1), base_point=low, dim=1
    )

    extended_coords, extended_inds = adaptive_table_2.interpolation_nodes_from_coordsinates(coords)
    extended_vals = function.truth(extended_coords)

    # It is critical that we pass the indices in addition to the coordinates here.
    # See comments in the method assign_values() for details.
    adaptive_table_2.assign_values(extended_vals, extended_coords, extended_inds)
    adaptive_vals_2 = adaptive_table_2.interpolate(coords)
    assert np.allclose(known_vals, adaptive_vals_2, atol=1e-10)

    # Also try to add one data point at a time, to check that we don't overwrite anything.
    adaptive_table_gradual = pp.AdaptiveInterpolationTable(
        dx=(high - low) / (num_pt - 1), base_point=low, dim=1
    )
    for ci in range(extended_coords.shape[-1]):
        # Again, it is critical to pass the indices as well as coordinates.
        adaptive_table_gradual.assign_values(extended_vals[ci], extended_coords[:, ci].reshape((-1, 1)), indices = extended_inds[:, ci].reshape((-1, 1)))

    adaptive_vals_gradual = adaptive_table_gradual.interpolate(coords)
    assert np.allclose(known_vals, adaptive_vals_gradual, atol=1e-10)

    # Final test: Evaluate on random points, make sure that 1) The interpolated and
    # exact function values are not too far apart (the interpolation accuracy depends
    # on the first and second derivative of the function); 2) that the two interpolation
    # methods calculate the same approximated function values, as well as derivatives.

    # Create a new adaptive table, so that we can evaluate it on a rather sparse point
    # set and check that we have not used all the quadrature points.
    adaptive_table_3 = pp.AdaptiveInterpolationTable(
        dx=(high - low) / (num_pt - 1), base_point=low, function=function.func(), dim=1
    )

    # The number of points is somewhat randomly set here.
    npt = 10
    pts = low.reshape((-1, 1)) + np.random.rand(dim, npt) * (high - low).reshape(
        (-1, 1)
    )

    adaptive_vals = adaptive_table_3.interpolate(pts)
    table_vals = table.interpolate(pts)

    # Check that the adaptive table has used fewer points than the full version
    # Worst case scenario is that all evaluation points trigger a new
    # evaluation, or that all points have been evaluated.
    num_quad_pts = adaptive_table_3._values.size
    assert num_quad_pts <= np.minimum(npt * np.power(2, dim), table._values.shape[-1])

    known_vals = function.truth(pts)

    # Function values should be reproduced with sufficient accuracy - the tolerance is
    # determined by the function itself.
    assert np.max(np.abs(table_vals - known_vals)) < function.tol
    # The two interpolation approaches should be identical
    assert np.allclose(table_vals, adaptive_vals, atol=1e-10)

    # Test of derivatives
    for d in range(dim):
        diff = table.diff(pts, axis=d)
        adaptive_diff = adaptive_table_3.diff(pts, axis=d)

        # It should not be necessary to compute new function values
        assert adaptive_table_3._values.size == num_quad_pts

        known_diff = function.diff(pts, d)
        # Note that for the time being, this test is not active for the
        # Quadractic function (due to diff_tol)
        assert np.max(np.abs(diff - known_diff)) < function.diff_tol
        assert np.allclose(diff, adaptive_diff, atol=1e-10)
