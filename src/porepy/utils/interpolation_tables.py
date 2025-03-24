"""The module contains interpolation tables, intended for use in function
evaluations. Specifically, the motivation is to facilitate the parametrization
framework described in

    Operator-based linearization approach for modeling of multiphase
    multi-component flow in porous media by Denis Voskov (JCP 2017)

The module contains two classes:

    InterpolationTable: Interpolation based on pre-computation of function values.
        Essentially this is a cumbersome implementation of scipy interpolation
        functionality, and the latter is likely to be preferred.

    AdaptiveInterpolationTable: Interpolation where function values are computed
        and stored on demand. Can give significant computational speedup in cases
        where function evaluations are costly and only part of the parameter space
        is accessed during simulation.

Both classes use piecewise linear interpolation of functions, and piecewise
constant approximations of derivatives.

"""

from __future__ import annotations

import itertools
from typing import Callable, Iterator, Optional

import numpy as np

import porepy as pp


class InterpolationTable:
    """Interpolation table based on pre-computation of function values.

    Function values are interpolated on a Cartesian mesh.
    The interpolation is done using piecewise linears (for function values) and
    constants (for derivatives). The domain of interpolation is an Nd box.

    The implementation may not be efficient, consider using functions from
    scipy.interpolate instead.

    Parameters:
        low: Minimum values of the domain boundary per dimension.
        high: Maximum values of the domain boundary per dimension.
        npt: Number of quadrature points (including endpoints of intervals) per
            dimension.
        function: Function to represent in the table. Should be vectorized (if necessary
            with a for-loop wrapper) so that multiple coordinates can be evaluated at
            the same time.
        dim: Dimension of the range of the function.

    """

    def __init__(
        self,
        low: np.ndarray,
        high: np.ndarray,
        npt: np.ndarray,
        function: Callable[[np.ndarray], np.ndarray],
        dim: int = 1,
    ) -> None:
        self.dim = dim
        # Dimension of the function to interpolate

        self._param_dim = low.size
        # Dimension of the parameter space

        self._set_sizes(low, high, npt, dim)
        # Data processing is left to a separate function.

        # Evaluate the function values in all coordinate points.
        self._table_values = np.zeros((self.dim, self._coord[0].size))
        for i, c in enumerate(zip(*self._coord)):
            self._table_values[:, i] = function(*c)

    def _set_sizes(
        self, low: np.ndarray, high: np.ndarray, npt: np.ndarray, dim: int
    ) -> None:
        """Helper method to set the size of the interpolation grid."""

        self._low = low
        self._high = high
        self._npt = npt

        # The base point for a standard table is in the same as the low coordinate.
        self._base_point = self._low

        # Define the quadrature points along each coordinate access.
        self._pt_on_axes: list[np.ndarray] = [
            np.linspace(low[i], high[i], npt[i]) for i in range(self._param_dim)
        ]

        # define the mesh size along each axis
        self._h = (high - low) / (npt - 1)

        # Set the strides necessary to advance to the next point in each dimension.
        # Refers to indices in self._coord. Is unity for first dimension, then
        # number of quadrature points in the first dimension etc.
        tmp = np.hstack((1, self._npt))
        self._strides = np.cumprod(tmp)[: self._param_dim].reshape((-1, 1))

        # Prepare table of function values. This will be filled in by __init__
        # Create interpolation grid.
        # The indexing, together with the Fortran-style raveling is necessary
        # to get a reasonable ordering of the expanded coefficients.
        coord_table = np.meshgrid(*self._pt_on_axes, indexing="ij")
        # Ravel into an array
        self._coord: list[np.ndarray] = [c.ravel("F") for c in coord_table]

    def interpolate(self, x: np.ndarray) -> np.ndarray:
        """Perform interpolation on a Cartesian grid by a piecewise linear
        approximation.

        Args:
            x (np.ndarray): Points to evaluate the function. Size dimension of
                parameter space times number of points.

        Returns:
            np.ndarray: Function values.

        """

        # allocate value array
        values: np.ndarray
        if x.ndim > 1:
            values = np.zeros((self.dim, x.shape[1]))
        else:
            # in case a single point in the function domain was passed as a row vector
            x = x.reshape((-1, 1))
            values = np.zeros((self.dim, 1))

        # Get indices of the base vertexes of the hypercubes where the function
        # is to be evaluated. The base vertex is in the (generalized) lower-left
        # corner of the cube.
        base_ind = self._find_base_vertex(x)
        # Get weights in each dimension for the interpolation between the higher
        # (right) and base (left) coordinate.
        right_weight, left_weight = self._right_left_weights(x, base_ind)

        # Loop over all vertexes in the hypercube. Evaluate the function in the
        # vertex with the relevant weight.
        # We need a linear index here, since eval_ind will be used to access the array
        # of computed values.
        for i, (incr, eval_ind) in enumerate(
            self._generate_indices(base_ind, linear=True)
        ):
            # Incr is 0 for dimensions to be evaluated in the left (base)
            # coordinate, 1 for others.
            # eval_ind is the linear index for this vertex.

            # Compute weight for this vertex
            weight = np.prod(
                right_weight * incr + left_weight * (1 - incr), axis=0
            )  # Not sure about self.dim > 1.

            # Add this part of the function evaluation
            # Handle special case: If the evaluation point is on the generalized
            # rightmost axis compared to what is covered in the table, there may be
            # indices in eval_ind that are outside the range of the value array (e.g.,
            # outside the grid). These can be ignored, but we do check that their
            # weights are computed to zero.
            inside_grid = eval_ind < self._values.shape[1]
            assert np.all(weight[np.logical_not(inside_grid)] < 1e-10)
            values[:, inside_grid] += (
                weight[inside_grid] * self._values[:, eval_ind[inside_grid]]
            )

        return values

    def gradient(self, x: np.ndarray, axis: int) -> np.ndarray:
        """Perform differentiation on a Cartesian grid by a piecewise constant
        approximation.

        Args:
            x (np.ndarray): Points to evaluate the function. Size dimension of
                parameter space times number of points.
            axis (int): Axis to differentiate along.

        Returns:
            np.ndarray: Function values.

        """

        # Allocate value array.
        # This is consistent with the derivative w.r.t. to only one axis,
        values: np.ndarray
        if len(x.shape) > 1:
            values = np.zeros((self.dim, x.shape[1]))
        else:
            # in case a single point in the function domain was passed as a row vector
            x = x.reshape((-1, 1))
            values = np.zeros((self.dim, 1))

        # Get indices of the base vertexes of the hypercubes where the function
        # is to be evaluated. The base vertex is in the (generalized) lower-left
        # corner of the cube.
        base_ind = self._find_base_vertex(x)
        # Get weights in each dimension for the interpolation between the higher
        # (right) and base (left) coordinate.
        right_weight, left_weight = self._right_left_weights(x, base_ind)

        # Loop over all vertexes in the hypercube. Evaluate the function in the
        # vertex with the relevant weight, and use a first order difference
        # to evaluate the derivative.
        # We need a linear index here, since it will be used to access the value array.
        for incr, eval_ind in self._generate_indices(base_ind, linear=True):
            # Incr is 0 for dimensions to be evaluated in the left (base)
            # coordinate, 1 for others.
            # eval_ind is the linear index for this vertex.

            # Compute weight for each dimension
            weight_ind = right_weight * incr + left_weight * (1 - incr)
            # The axis to differentiate has +1 when evaluated to the right,
            # -1 for the right
            weight_ind[axis] = 2 * incr[axis] - 1

            # Compute weights
            weight = np.prod(weight_ind, axis=0)

            # Add contribution from this vertex
            values += weight * self._values[:, eval_ind]

        return values / self._h[axis]

    @property
    def _values(self) -> np.ndarray:
        # Use a property decorator for self._values. This is not needed in this class,
        # but is convenient in the subclass AdaptiveInterpolationTable.
        return self._table_values

    def _find_base_vertex(self, coord: np.ndarray) -> np.ndarray:
        """Helper function to get the base (generalized lower-left) vertex of a
        hypercube.

        """

        ind = list()
        # performing Cartesian search per axis of the interpolation grid.
        for x_i, h_i, low_i, high_i in zip(coord, self._h, self._low, self._high):
            # checking if any point is out of bound
            if np.any(x_i < low_i) or np.any(high_i < x_i):
                raise ValueError(
                    f"Point(s) outside coordinate range [{self._low}, {self._high}]"
                )
            # cartesian search for uniform grid, floor division by mesh size
            ind.append(((x_i - low_i) // h_i).astype(int))

        return np.array(ind)

    def _generate_indices(
        self, base_ind: np.ndarray, linear: bool
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Iterator for linear indices that form the vertexes of a hypercube with a
        given base vertex, and the dimension-wise increments in indices from the base.
        """

        # IMPLEMENTATION NOTE: We could have used np.unravel_index here. That may be
        # faster, if this ever turns out to be a bottleneck.
        # Need to use _param_dim (dimension of parameter space) here, not dim.
        for increment in itertools.product(range(2), repeat=self._param_dim):
            incr = np.asarray(increment).reshape((-1, 1))
            eval_ind = self._index_from_base_and_increment(base_ind, incr, linear)
            assert isinstance(eval_ind, np.ndarray)
            yield incr, eval_ind

    def _index_from_base_and_increment(
        self, base_ind: np.ndarray, incr: np.ndarray, linear: bool
    ) -> np.ndarray:
        """For a given base index and increment in each dimension, get the full index
        from a base (e.g., lower-left corner of a hypercube) and an increment.

        NOTE: The parameter 'linear' is ignored since standard interpolation tables
        always work with a linear index (the ordering of the points in the underlying
        grid and that used in the array of values is the same).
        """
        vertex_ind = base_ind + incr
        eval_ind = np.sum(vertex_ind * self._strides, axis=0).ravel()
        return eval_ind

    def _right_left_weights(
        self, x: np.ndarray, base_ind: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """For each dimension, find the interpolation weights to the right
        and left sides.
        """
        right_weight = np.array(
            [
                (x[i] - (self._pt_on_axes[i][base_ind[i]])) / self._h[i]
                for i in range(self._param_dim)
            ]
        )

        # Check that we have found the right interval. Accept a small error to
        # account for floating point errors.
        tol = 1e-13
        assert np.all(right_weight >= -tol) and np.all(right_weight <= 1 + tol)

        left_weight = 1 - right_weight

        return right_weight, left_weight

    def __repr__(self) -> str:
        """String representation"""
        s = f"Interpolation table in {self._param_dim} dimensions. \n"
        s += f"Lower bounds: {self._low}.\n"
        s += f"Upper bounds: {self._high}.\n"
        s += f"Number of quadrature points in each dimension: {self._npt}.\n"

        s += f"Minimum function value: {np.min(self._values)}.\n"
        s += f"Maximum function value: {np.max(self._values)}."

        return s


class AdaptiveInterpolationTable(InterpolationTable):
    """Interpolation table based on adaptive computation of function values.

    Function values are interpolated on a Cartesian mesh.
    The interpolation is done using piecewise linears (for function values) and
    constants (for derivatives). The domain of interpolation is an Nd box.

    The function values are computed on demand, then stored. This can give
    substantial computational savings in cases where only a part of the
    parameter space is accessed, and the computation of function values
    is costly.

    Parameters:
        dx: Grid resolution in each direction.
        base_point: A point in the underlying grid. Used to fix the location of the
            grid lines.
        function (Callable): Function to represent in the table. Should be
            vectorized (if necessary with a for-loop wrapper) so that multiple
            coordinates can be evaluated at the same time. If not provided, table values
            must be assigned via the method 'assign_values'.
        dim (int): Dimension of the range of the function. Values above one
            have not been much tested, use with care.

    """

    def __init__(
        self,
        dx: np.ndarray,
        base_point: Optional[np.ndarray] = None,
        function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        dim: int = 1,
    ) -> None:
        self.dim: int = dim
        # Dimension of the field to interpolate

        self._param_dim = dx.size
        # Dimension of the parameter space

        # IMPLEMENTATION NOTE: The sparse array which stores the actual data represents
        # coordinates by (possibly negative) integer indices, while the parameter space
        # for interpolation is defined on real numbers. This requires mappings points in
        # parameter space to quadrature points in an underlying Cartesian grid, and
        # further identifying these points with indices in the data table.

        self._table = pp.array_operations.SparseNdArray(self._param_dim)
        # Construct grid for interpolation

        self._pt: np.ndarray = np.zeros((self._param_dim, 0))
        # self._pt store coordinates of quadrature points in parameter space.
        # Note that this is different from super()._pt_on_axes, which gives the
        # 1d-coordinates of the quadrature points along each coordinate axis.
        # It is also different from super()._coord, which gives Nd-coordinates, but as
        # structured and dense data.

        self._h: np.ndarray = dx.reshape((-1, 1))
        # Set resolution of the Cartesian grid in parameter space.

        if base_point is None:
            base_point = np.zeros(dim)
        self._base_point = base_point.reshape((-1, 1))
        # The base point is the point in parameter space which corresponds to the index
        # (0, 0, ..., 0) in the sparse array.

        self._function = function
        # The function to interpolate

    @property
    def _values(self):
        # Use a property decorator to tie self._values to the value array of the
        # underlying sparse array.
        return self._table._values

    def interpolate(self, x: np.ndarray) -> np.ndarray:
        """Perform interpolation on a Cartesian grid by a piecewise linear
        approximation.

        If the table has a function, function values in the quadrature points will be
        computed as needed. If the values are computed externally and fed through the
        method assign_values(), the user is responsible that all relevant quadrature
        points have been assigned values.

        Args:
            x (np.ndarray): Points to evaluate the function. Size dimension of
                parameter space times number of points.

        Returns:
            np.ndarray: Function values.

        """
        # Fill any missing function values if a function is available.
        # NOTE: If the table is populated with the method 'assign_values', the user is
        # responsible for assigning all necessary values.
        if self._function is not None:
            self._fill_values(x)

        # Use parent method for interpolation.
        # NOTE: If the table values were set using assign_values(), without feeding the
        # indices of the quadrature points (argument indices), this sometimes lead to
        # strange errors: If the points to be evaluated are at or very close to
        # quadrature points (in the underlying Cartesian grids), rounding errors
        # in the method _find_base_vertex() sometimes lead the points to be associated
        # with the wrong Cartesian indices. This may lead to the interpolation (which
        # partly works on indices, not coordinates) requesting values at indices that
        # are not known to the table, leading to an error, typically from the method
        # _right_left_weights() that negative weights were computed, or alternatively
        # that the table does not contain the necessary information.
        # The solution to such problems is to use the method
        # quadrature_points_from_coordinates() to get quadrature point coordinates and
        # indices, and feed both these to assign_indices().
        return super().interpolate(x)

    def gradient(self, x: np.ndarray, axis: int) -> np.ndarray:
        """Perform differentiation on a Cartesian grid by a piecewise constant
        approximation.

        Args:
            x (np.ndarray): Points to evaluate the function. Size dimension of
                parameter space times number of points.
            axis (int): Axis to differentiate along.

        Returns:
            np.ndarray: Function values.

        """
        # Fill any missing function values if a function is available.
        # NOTE: If the table is populated with the method 'assign_values', the user is
        # responsible for assigning all necessary values.
        if self._function is not None:
            self._fill_values(x)

        # Use standard method for differentiation.
        # NOTE: See comments in self.interpolation() regarding possible sources of
        # unexpected errors from this function, and how to circumwent it.
        return super().gradient(x, axis)

    def assign_values(
        self,
        val: np.ndarray,
        coord: np.ndarray,
        indices: Optional[np.ndarray] = None,
    ) -> None:
        """Assign externally computed values to the table.

        The user is responsible that the coordinates are nodes in the Cartesian grid
        underlying this interpolation tabel.

        This method can be used, e.g., if the table values are calculated by an
        external function which it is not desirable to pass to the interpolation table.
        One use case is when the function is an external library (e.g., a thermodynamic
        flash calculation) which computes several properties at the same time. In this
        case it is better to collect all properties externally and pass it to the
        respective interpolation tables for the individual properties.

        If the indices in the underlying Cartesian grid corresponding to the coordinate
        points are known (e.g., the points were found by the method
        quadrature_points_from_coordinates), it is strongly recommended that these are
        also provided in assignment.

        Args:
            val: Values to assign.
            coord: Coordinates of points to assign values to.
            indices: Indices of the coordinates in the underlying Grid.

        """

        if indices is None:
            # Get the corresponding indices in the underlying Cartesian grids.
            inds = self._find_base_vertex(coord)
        else:
            inds = indices

        ind_list = [i for i in inds.T]

        # Add values and indices to the table.
        column_permutation = self._table.add(ind_list, val, additive=False)

        # Add the coordinates to the table. The table may have permuted the node
        # coordinates during unification of the indices, the coordinates should be
        # likewise permuted.
        self._pt = np.hstack((self._pt, coord[:, column_permutation]))

    def quadrature_points_from_coordinates(
        self, x: np.ndarray, remove_known_points: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Obtain the coordinates of quadrature points that are necessary to evalute
        the table in specified points.

        If the table values are calculated by an external computation (i.e., not through
        self._function), this method can be used to determine which evaluations are
        needed before interpolation is invoked.

        The method also returns indices of the quadrature points in the Cartesian grid
        underlying the interpolation table. If the quadrature points are used to compute
        function values which later are fed to this interpolation table using the
        method 'assign_values()', it is strongly recommended that the indices are also
        passed to the assignment function.

        Args:
            x: Coordinates of evaluation points.
            remove_known_points: If True, only quadrature points that have not
                previously been assigned values in this table are returned.

        Returns:
            Coordinates of the quadrature points.
            Indices of the quadrature points in the underlying Cartesian grid.

        """
        # The lower-left corner of each hypercube.
        base_ind = self._find_base_vertex(x, safeguarding=True)

        # Loop over all vertexes in the hypercube, store the index. In this case we do
        # not want a linear index, since we will compare the multiindex with the
        # coordinates in the sparse array.
        ind = []
        for _, eval_ind in self._generate_indices(base_ind, linear=False):
            ind.append(eval_ind)

        # Uniquify indices to avoid computing values for the same vertex twice.
        unique_ind = np.unique(np.hstack(ind), axis=1)

        coord = self._base_point + self._h * unique_ind

        # Remove points where function values already exists if requested.
        # This avoids recomputation of known values.
        if remove_known_points:
            _, _, exists, _ = pp.array_operations.intersect_sets(coord, self._pt)

            coord = coord[:, np.logical_not(exists)]
            unique_ind = unique_ind[:, np.logical_not(exists)]

        return coord, unique_ind

    def _fill_values(self, x: np.ndarray) -> None:
        """Find points in the interpolation grid that will be used for function
        evaluation. Compute function values as needed.

        Args:
            x: Array of coordinates to be evaluated.

        """

        if self._function is None:
            raise ValueError(
                "No function to evaluate - should values be added instead?"
            )

        # Get hold of the quadrature points needed to interpolate in the given
        # coordinates.
        coord, unique_ind = self.quadrature_points_from_coordinates(x)

        # Find which values have been computed before.
        _, _, precomputed, _ = pp.array_operations.intersect_sets(
            unique_ind, self._table._coords
        )

        # Find which vertexes have not been used before.
        indices_to_compute = np.where(np.logical_not(precomputed))[0]

        # Compute and store values
        if indices_to_compute.size > 0:
            new_values = np.array(
                [self._function(*coord[:, i]) for i in indices_to_compute]
            ).T

            # In the sparse array we use the integer indices, referring to the
            # underlying Cartesian grid of this table.
            self._table.add([unique_ind[:, i] for i in indices_to_compute], new_values)
            # In this table, we need to store the actual coordinates.
            self._pt = np.hstack((self._pt, coord))

    def _index_from_base_and_increment(
        self, base_ind: np.ndarray, incr: np.ndarray, linear: bool
    ) -> np.ndarray:
        """For a given base index and increment in each dimension, get the full index
        from a base (e.g., lower-left corner of a hypercube) and an increment.

        For adaptive interpolation tables, with unstructured storage of data, we
        sometimes need the linear index, sometimes the multi-dimensional one.

        Args:
            base_ind: Indices of the lower-left corners of the hypercubes on which
                the interpolation is based.
            incr: Increment to be added from the base index.
            linear: If True, the returned index is linear; if False, a multiindex is
                returned.

        Returns:
            np.ndarray: Array of indices on the specified format.

        """
        if linear:
            # Add the base ind and the increment, and then find the corresponding value
            # in the coordinate set of the table.
            _, _, is_mem, ind_in_table = pp.array_operations.intersect_sets(
                base_ind + incr, self._table._coords
            )

            # Safeguarding.
            assert np.all(is_mem)

            return np.ravel(ind_in_table)
        else:
            # A multi-index is wanted. Simply add the increment to the base index.
            return np.asarray(base_ind + incr)

    def _right_left_weights(
        self, x: np.ndarray, base_ind: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """For each dimension, find the interpolation weights to the right
        and left sides.
        """

        # First do a mapping from the structured ordering underlying this table, and
        # reflected in base_ind, to the unstructured ordering in self._table.
        _, _, ismem, ind_in_table = pp.array_operations.intersect_sets(
            base_ind, self._table._coords
        )
        raveled_ind = np.ravel(ind_in_table)
        # Sanity checks.
        assert np.all(ismem)
        assert all([len(s) == 1 for s in ind_in_table])

        # Find the weights on the right array.
        right_weight = np.array(
            [
                (x[i] - (self._pt[i, raveled_ind])) / self._h[i]
                for i in range(self._param_dim)
            ]
        )
        # Check that we have found the right interval. Accept a small error to
        # account for floating point errors.
        tol = 1e-13
        assert np.all(right_weight >= -tol) and np.all(right_weight <= 1 + tol)

        left_weight = 1 - right_weight

        return right_weight, left_weight

    def _find_base_vertex(self, coord: np.ndarray, safeguarding=False) -> np.ndarray:
        """Helper function to get the base (generalized lower-left) vertex of a
        hypercube.

        Upon request, the method will also identify cases where the choice of lower-left
        vertex (thus which hypercube to use in interpolation) may be unclear due to
        rounding errors, as may happen if a quadrature point is very close to a node
        in the underlying Cartesian grid. For such cases, additional points will be
        added. This makes subsequent interpolation more robust, to the price of
        requesting more data points, and also removing the one-to-one relation between
        coordinate and index. This functionality, triggered by the parameter
        safeguarding should only be invoked from the method
        quadrature_points_from_coordinates() (but there it can be critical to use it).

        Args:
            coord: Coordinates for which the base vertex is sought.
            safeguarding: If True, additional vertexes may be added to avoid
                vulnerabilities with respect to rounding errors.


        """
        # The below safeguarding is motivated by the following potential behavior:
        # For a given point to be interpolated, the lower left coordinate is found by
        # floor division after adjusting for the origin and resolution of the Cartesian
        # grid. If the point to be interpolated is at a grid line in the Cartesian grid
        # used in the interpolation table, rounding errors may lead to the wrong index
        # being identified (say, the number 5 is calculated as 4.9999999, and the index
        # 4 is identified). Safeguarding will add both 4 and 5 to the list.
        #
        # For a 2d parameter space, (5, 5) may be rounded to (4, 4) - in which case all
        # points (4, 4), (4, 5), (5, 4), (5, 5) are added - this is necessary to ensure
        # that when we interpolate, the necessary quadrature points are available in
        # the table.
        #
        # If instead (5, 5) is rounded to (5, 4), both (5, 4) and (5, 5) will be added
        # (but not (4, 4) and (4, 5)).

        ind = list()

        # Keep track of indices that may be impacted by rounding errors.
        rounding_error_danger = np.zeros(coord.shape, dtype=bool)

        # Perform Cartesian search per dimension of the interpolation grid.
        # IMPLEMENTATION NOTE: If we want non-uniform grids in parameter space, here is
        # the place to modify the code.
        for i, (x_i, h_i, base_i) in enumerate(zip(coord, self._h, self._base_point)):
            # Cartesian search for uniform grid, floor division by mesh size
            # Subtract the base to find the origin in the underlying Cartesian grid.
            floored_ind = ((x_i - base_i) // h_i).astype(int)

            if safeguarding:
                # Find the actual coordinate in the Cartesian grid
                exact = (x_i - base_i) / h_i
                # Since the index was identified by floor division, the danger is that
                # rounding errors caused identification of an index one too low.
                # Find points that almost was on the next grid line.
                # The threshold is set arbitrary here, the current value should be
                # much looser than what is necessary to identify rounding errors, but we
                # are safeguarding, so better safe than sorry.
                rounding_error_danger[i, exact - floored_ind > 0.999] = True

            # Store data for this dimension.
            ind.append(floored_ind)

        # Merge hte arrays
        full_ind = np.array(ind)

        if safeguarding:
            # Add additional points for safeguarding purposes.

            # Identify columns (indices) that were impacted by rounding errors.
            columns = np.any(rounding_error_danger, axis=0)

            # Find which rows (dimensions) were impacted by rounding errors
            rows_with_repeats = np.where(
                np.any(rounding_error_danger[:, columns], axis=1)
            )[0]

            # Only add indices if there were points that could have been rounded
            # wrongly.
            if np.any(rows_with_repeats):
                # Storage of extra indices.
                extra_ind = []

                # Loop over all combinations of dimensions that could have been
                # afflicted by rounding errors.
                # Referring to the example at the begining of this method, if (5, 5) was
                # rounded to (4, 4), we need to add 1 to (4, 4) on both the first and
                # second axis, as well as their combination. If instead (5, 5) became
                # (5, 4), we only add on the second axis.
                # We achieve this by invoking itertools.combinations on correctly
                # sized arrays.
                for combination_length in range(1, rows_with_repeats.size + 1):
                    for active_rows in itertools.combinations(
                        rows_with_repeats, combination_length
                    ):
                        # Find which columns should be fixed for this combination of
                        # rows.
                        # The list is needed since itertools.combinations returns a
                        # tuple that is not fit for array indexing.
                        active_columns = np.all(
                            np.atleast_2d(rounding_error_danger[list(active_rows)]),
                            axis=0,
                        )

                        # Do a copy (!) and increase the index.
                        tmp_ind = full_ind[:, active_columns].copy()
                        tmp_ind[list(active_rows)] += 1
                        extra_ind.append(tmp_ind)

                # Collect data, prepare for return.
                full_ind = np.hstack((full_ind, np.hstack(extra_ind)))

        return full_ind

    def __repr__(self) -> str:
        """String representation"""
        s = f"Adaptive interpolation table in {self._param_dim} dimensions.\n"
        s += f"The table stores function values in {self._pt.shape[1]} points.\n"
        if self._pt.size != 0 and self._values.size != 0:
            s += "The minimum coordinates in each dimension (possibly combining multiple "
            s += "coordinates) are: \n \n \t"
            for dim in range(self._param_dim):
                s += f"{dim}: {self._pt[dim].min()}, "
            s = s[:-2] + "\n \n"
            s += "The maximum coordinates in each dimension (possibly combining multiple "
            s += "coordinates) are: \n \n \t"
            for dim in range(self._param_dim):
                s += f"{dim}: {self._pt[dim].max()}, "
            s = s[:-2] + "\n \n"
            s += f"Minimum function value: {np.min(self._values)}\n"
            s += f"Maximum function value: {np.max(self._values)}"
        else:
            s += "The table is currently empty."

        return s
