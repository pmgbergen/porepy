"""This test covers all aspects of assignment of values and interpolation using
the values. The test is set up to vary the dimension of the parameter space which
the tables represents, and the function to be enterpolated.

The general structure of the tests is: For a given function, make full and adaptive
interpolation tables. The tables are populated with values (at instantiation for the
full table, on demand for the adaptive ones), and the function is interpolated at
different points. The accuracy expected depends on the function and on the
evaluation points (linear functions should be exactly interpolated, all functions
should be exactly interpolated in data points).

"""

import numpy as np
import pytest

from porepy.utils import interpolation_tables


class _LinearFunc:
    # Helper class to represent a linear function.
    def __init__(self, nd, h):
        self.coeff = np.random.rand(nd, 1)

        # Tolerances tuned to the accuracy that can be expected from linear
        # interpolation of this function.
        self.tol = 1e-10
        self.diff_tol = self.tol

    def truth(self, x):
        return np.sum(self.coeff * x, axis=0)

    def func(self):
        def f(*args):
            val = 0
            for coeff, x in zip(self.coeff, args):
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
        self.tol = 2 * np.max(self.coeff * h) / 4
        # The error in a finite difference approximation of the derivative of a
        # quadratic function is bounded by self.coeff * h
        self.diff_tol = np.max(self.coeff * h)

    def truth(self, x):
        return np.sum(self.coeff * np.power(x, 2), axis=0)

    def func(self):
        def f(*args):
            val = 0
            for coeff, x in zip(self.coeff, args):
                val += coeff * x**2
            return val

        return f

    def diff(self, x, axis):
        return 2 * self.coeff[axis] * x[axis]


def _generate_interpolation_tables(dim, factory, adaptive_has_function, random_seed=42):
    """For a given dimension of the parameter space and a function factory,
    generate a dense and an adaptive interpolation table. Also return the function
    to be interpolated.
    """

    # We have occationally had issues with these tests failing on random data, however,
    # subsequent testing (on huge arrays) failed to uncover any issues. Fix the random
    # seed so that if errors arise, they should be reproducible.
    np.random.seed(random_seed)

    # Set the bounds for the part of the parameter space to be covered by the tables.
    # The full table will adhere strictly to this,  the adaptive tables are anchored
    # in the point 'low', and computed according to the resolution of the table (but
    # since the two types of tables have the same resolution, the resulting grids in
    # parameter space will have the same nodes).
    low = np.random.rand(dim)
    high = 2 + np.random.rand(dim)

    # Vary the number of points in each dimension.
    num_pt = np.array([4 + d for d in range(dim)])
    # Grid resolution in each dimension
    max_size = (high - low) / (num_pt - 1)
    # Create function to be tested.
    function = factory(dim, max_size)

    # Create interpolation tables
    table = interpolation_tables.InterpolationTable(low, high, num_pt, function.func())

    # Give the adaptive table the function to be interpolated, if requested.
    function_arg = function.func() if adaptive_has_function else None
    adaptive_table = interpolation_tables.AdaptiveInterpolationTable(
        dx=(high - low) / (num_pt - 1), base_point=low, function=function_arg, dim=1
    )

    return table, adaptive_table, function

    # Create and test interpolation tables.
    # Three types of tests are run:
    #    1) The interpolated function should be evaluated exactly in the nodes of the
    #       grid in parameter space.
    #    2) The adaptive interpolation table can be built by adding points gradually
    #       or all at once, the result should be the same.
    #    3) Interpolation of values and derivatives have the expected accuracy relative
    #       to the interpolated function in random points in parameter space.


@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("factory", [_LinearFunc, _QuadraticFunc])
@pytest.mark.parametrize("random_seed", [2, 3, 4])
def test_interpolation_in_quadrature_points(dim, factory, random_seed):
    """Test interpolation in the quadrature points themselves.

    For this test, the adaptive table has access to the function to be interpolated.
    """
    table, adaptive_table, function = _generate_interpolation_tables(
        dim, factory, adaptive_has_function=True, random_seed=random_seed
    )

    # First check that the interpolation is exact in the quadrature points.
    # In this case, the interpolation weights in the tables should be calculated to be
    # 1 in the relevant node, and 0 in adjacent nodes.
    coords = np.vstack(table._coord)

    adaptive_vals = adaptive_table.interpolate(coords)
    table_vals = table.interpolate(coords)
    known_vals = function.truth(coords)

    # An error message from here likely indicates that either something is wrong with
    # the function evaluation itself, or that the weights in the interpolation are
    # wrong.
    # The function values in the quadrature nodes should be interpolated exactly,
    # thus use a strict tolerance in the comparison.
    assert np.allclose(table_vals, adaptive_vals, atol=1e-10)
    assert np.allclose(known_vals, adaptive_vals, atol=1e-10)


@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("factory", [_LinearFunc, _QuadraticFunc])
def test_adaptive_interpolation_table_assign_externally_computed_values(dim, factory):
    """Test that the adaptive interpolation table can be built by adding points.

    This is only relevant for the adaptive table.

    """
    # Don't parametrize the random seed here, this has been done in the previous test.
    table, adaptive_table, function = _generate_interpolation_tables(
        dim, factory, adaptive_has_function=False
    )
    # Fetch quadrature points from the full table.
    coords = np.vstack(table._coord)
    known_vals = function.truth(coords)
    # Find which quadrature points are needed to interpolate the function. This forms
    # an extended set of coordinates, which we will use to build the adaptive table.
    (
        extended_coords,
        extended_inds,
    ) = adaptive_table.quadrature_points_from_coordinates(coords)
    extended_vals = function.truth(extended_coords)

    # It is critical that we pass the indices in addition to the coordinates here.
    # See comments in the method AdaptiveInterpolationTable.assign_values() for details.
    adaptive_table.assign_values(extended_vals, extended_coords, extended_inds)

    # The adaptive table can be interpolated in coords (but not in extended_coords).
    adaptive_vals = adaptive_table.interpolate(coords)

    # The function values in the quadrature nodes should be interpolated exactly,
    # thus use a strict tolerance in the comparison.
    assert np.allclose(known_vals, adaptive_vals, atol=1e-10)


@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("factory", [_LinearFunc, _QuadraticFunc])
def test_adaptive_interpolation_assign_values_one_by_one(dim, factory):
    """Test that the adaptive interpolation table can be built by adding points."""
    # See function test_adaptive_interpolation_table_assign_externally_computed_values
    # for a description of the test.
    table, adaptive_table, function = _generate_interpolation_tables(
        dim, factory, adaptive_has_function=False
    )
    coords = np.vstack(table._coord)
    known_vals = function.truth(coords)
    (
        extended_coords,
        extended_inds,
    ) = adaptive_table.quadrature_points_from_coordinates(coords)
    extended_vals = function.truth(extended_coords)

    for ci in range(extended_coords.shape[-1]):
        # Again, it is critical to pass the indices as well as coordinates.
        adaptive_table.assign_values(
            extended_vals[ci],
            extended_coords[:, ci].reshape((-1, 1)),
            indices=extended_inds[:, ci].reshape((-1, 1)),
        )

    adaptive_vals_gradual = adaptive_table.interpolate(coords)

    # The function values in the quadrature nodes should be interpolated exactly,
    # thus use a strict tolerance in the comparison.
    assert np.allclose(known_vals, adaptive_vals_gradual, atol=1e-10)


@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("factory", [_LinearFunc, _QuadraticFunc])
@pytest.mark.parametrize("random_seed", [2, 3, 4])
def test_interpolation_tables_random_points(dim, factory, random_seed):
    """Evaluate on random points, make sure that
    1) The interpolated and exact function values are not too far apart (the
    interpolation accuracy depends on the first and second derivative of the
    function)
    2) that the two interpolation methods calculate the same approximated function
    values, as well as derivatives.

    """
    # Create a new adaptive table, so that we can evaluate it on a rather sparse point
    # set and check that we have not used all the quadrature points.
    # Here we do vary the random seed.
    table, adaptive_table, function = _generate_interpolation_tables(
        dim, factory, adaptive_has_function=True, random_seed=random_seed
    )

    # The number of points is somewhat randomly set here.
    low = table._low
    high = table._high

    npt = 10
    pts = low.reshape((-1, 1)) + np.random.rand(dim, npt) * (high - low).reshape(
        (-1, 1)
    )

    adaptive_vals = adaptive_table.interpolate(pts)
    table_vals = table.interpolate(pts)

    # Check that the adaptive table has used fewer points than the full version
    # Worst case scenario is that all evaluation points trigger a new
    # evaluation, or that all points have been evaluated.
    num_quad_pts = adaptive_table._values.size
    assert num_quad_pts <= np.minimum(npt * np.power(2, dim), table._values.shape[-1])

    known_vals = function.truth(pts)

    # Function values should be reproduced with sufficient accuracy - the tolerance is
    # known by the function itself.
    assert np.max(np.abs(table_vals - known_vals)) < function.tol

    # The two interpolation approaches should be identical
    assert np.allclose(table_vals, adaptive_vals, atol=1e-10)

    # Test of derivatives
    for d in range(dim):
        diff = table.gradient(pts, axis=d)
        adaptive_diff = adaptive_table.gradient(pts, axis=d)

        # It should not be necessary to compute new function values
        assert adaptive_table._values.size == num_quad_pts

        known_diff = function.diff(pts, d)
        # Compare with the known derivative, with a tolerance that is known by the
        # function itself.
        assert np.max(np.abs(diff - known_diff)) < function.diff_tol
        # The adaptive and the full table should be equivalent.
        assert np.allclose(diff, adaptive_diff, atol=1e-10)
