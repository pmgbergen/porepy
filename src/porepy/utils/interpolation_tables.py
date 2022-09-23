""" The module contains interpolation tables, intended for use in function
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
import warnings
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

    Args:
        low: Minimum values of the domain boundary per dimension.
        high: Maximum values of the domain boundary per dimension.
        npt: Number of interpolation points (including endpoints of intervals) per
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
        function: Callable[[np.ndarray], np.ndarray] = None,
        dim: int = 1,
    ) -> None:
        # Data processing is left to a separate function.
        self._set_sizes(low, high, npt, dim)
        
        # Evaluate the function values in all coordinate points.

        self._table_values = np.zeros((self.dim, self._coord[0].size))
        for i, c in enumerate(zip(*self._coord)):
            self._table_values[:, i] = function(*c)
                
      

    def _set_sizes(
        self, low: np.ndarray, high: np.ndarray, npt: np.ndarray, dim: int
    ) -> None:
        """Helper method to set the size of the interpolation grid. Separated
        from __init__ to be used with inheritance.
        """
        self.dim = dim
        self._param_dim = low.size

        self._low = low
        self._high = high
        self._npt = npt

        # The base point for a standard table is in the same as the low coordinate.
        self._base_point = self._low

        # Define the interpolation points along each coordinate access.
        self._pt: list[np.ndarray] = [
            np.linspace(low[i], high[i], npt[i]) for i in range(self._param_dim)
        ]

        # define the mesh size along each axis
        self._h = (high - low) / (npt - 1)

        # Set the strides necessary to advance to the next point in each dimension.
        # Refers to indices in self._coord. Is unity for first dimension, then
        # number of interpolation points in the first dimension etc.
        tmp = np.hstack((1, self._npt))
        self._strides = np.cumprod(tmp)[: self._param_dim].reshape((-1, 1))

        # Prepare table of function values. This will be filled in by __init__
        # Create interpolation grid.
        # The indexing, together with the Fortran-style raveling is necessary
        # to get a reasonable ordering of the expanded coefficients.
        coord_table = np.meshgrid(*self._pt, indexing="ij")
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
        for i, (incr, eval_ind) in enumerate(self._generate_indices(base_ind, linear=True)):
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
            values[:, inside_grid] += weight[inside_grid] * self._values[:, eval_ind[inside_grid]]

        return values

    def diff(self, x: np.ndarray, axis: int) -> np.ndarray:
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
        """ Iterator for linear indices that form the vertexes of a hypercube with a
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

    def _index_from_base_and_increment(self, base_ind: np.ndarray, incr: np.ndarray, linear: bool) -> np.ndarray:
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
        """ For each dimension, find the interpolation weights to the right
        and left sides.
        """
        right_weight = np.array(
            [
                (x[i] - (self._pt[i][base_ind[i]] ))  / self._h[i]
                for i in range(self._param_dim)
            ]
        )
        # Check that we have found the right interval
        assert np.all(right_weight >= 0) and np.all(right_weight <= 1)

        left_weight = 1 - right_weight

        return right_weight, left_weight


class AdaptiveInterpolationTable(InterpolationTable):
    """Interpolation table based on adaptive computation of function values.

    Function values are interpolated on a Cartesian mesh.
    The interpolation is done using piecewise linears (for function values) and
    constants (for derivatives). The domain of interpolation is an Nd box.

    The function values are computed on demand, then stored. This can give
    substantial computational savings in cases where only a part of the
    parameter space is accessed, and the computation of function values
    is costly.

    Args:
        dx: Grid resolution in each direction. 
        base_point: A point in the underlying grid. Used to fix the location of the
            grid lines.
        function (Callable): Function to represent in the table. Should be
            vectorized (if necessary with a for-loop wrapper) so that multiple
            coordinates can be evaluated at the same time.
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
        self._param_dim = dx.shape
        # Construct grid for interpolation
        self._table = pp.array_operations.SparseNdArray(param_dim)

        self._pt = np.zeros((self._param_dim, 0))

        self._h = dx
        if base_point is None:
            base_point = np.zeros(dim)
        self._base_point = base_point

        # Store function
        if function is None:
            self._has_function = False
        else:
            self._function = function
            self._has_function = True

    @property
    def _values(self):
        # Use a property decorator to tie self._values to the value array of the
        # underlying sparse array.
        return self._table._values


    def interpolate(self, x: np.ndarray) -> np.ndarray:
        """Perform interpolation on a Cartesian grid by a piecewise linear
        approximation. Compute and store the necessary function values.

        Args:
            x (np.ndarray): Points to evaluate the function. Size dimension of
                parameter space times number of points.

        Returns:
            np.ndarray: Function values.

        """
        # Fill any missing function values.
        self._fill_values(x)
        # Use standard method for interpolation.
        return super().interpolate(x)

    def diff(self, x: np.ndarray, axis: int) -> np.ndarray:
        """Perform differentiation on a Cartesian grid by a piecewise constant
        approximation.

        Args:
            x (np.ndarray): Points to evaluate the function. Size dimension of
                parameter space times number of points.
            axis (int): Axis to differentiate along.

        Returns:
            np.ndarray: Function values.

        """
        # Fill any missing function values.
        self._fill_values(x)
        # Use standard method for differentiation.
        return super().diff(x, axis)

    def assign_values(self, coord: list[np.ndarray], val: np.ndarray) -> None:
        """Assign values to the table.

        Args:
            coord (list[np.ndarray]): Coordinates of points to assign values to.
            val (np.ndarray): Values to assign.

        """
        self._table.add(coord, val, additive=False)


    def _fill_values(self, x: np.ndarray) -> None:
        # Find points in the interpolation grid that will be used for function
        # evaluation. Compute function values as needed.

        if self._function is None:
            raise ValueError("No function to evaluate - should values be added instead?")

        # The lower-left corner of each hypercube.
        base_ind = self._find_base_vertex(x)

        # Loop over all vertexes in the hypercube, store the index. In this case we do
        # not want a linear index, since we will compare the multiindex with the
        # coordinates in the sparse array.
        ind = []
        for _, eval_ind in self._generate_indices(base_ind, linear=False):
            ind.append(eval_ind)

        # Uniquify indices to avoid computing values for the same vertex twice.
        unique_ind = np.unique(np.hstack(ind), axis=1)
        # Find which values have been computed before.
        _, _, precomputed, _= pp.array_operations.intersect_sets(unique_ind, self._table._coords)
        
        # Find which vertexes have not been used before.
        indices_to_compute = unique_ind[:, np.logical_not(precomputed)].T
        
        coord = [self._base_point + self._h * v for v in indices_to_compute]

        # Compute and store values
        if len(coord) > 0:
            new_values = np.vstack(np.array([self._function(*c) for c in coord])).T
            # In the spares array we use the integer indices, referring to the
            # underlying Cartesian grid of this table.
            self._table.add([a for a in indices_to_compute], new_values)
            # In this table, we need to store the actual coordinates.
            self._pt = np.hstack((self._pt, np.vstack(coord).T))


    def _index_from_base_and_increment(self, base_ind: np.ndarray, incr: np.ndarray, linear: bool) -> np.ndarray:
        """For a given base index and increment in each dimension, get the full index
        from a base (e.g., lower-left corner of a hypercube) and an increment.
        
        For adaptive interpolation tables, with unstructured storage of data, we
        sometimes need the linear index, sometimes the multi-dimensional one.
        
        """        
        if linear:
            # Add the base ind and the increment, and then find the corresponding value
            # in the coordinate set of the table.
            _, _, is_mem, ind_in_table = pp.array_operations.intersect_sets(base_ind + incr, self._table._coords)
            
            # Safeguarding.
            assert np.all(is_mem)
            
            return np.ravel(ind_in_table)
        else:
            # A multi-index is wanted. Simply add the increment to the base index.
            return np.asarray(base_ind + incr)
    
    def _right_left_weights(
        self, x: np.ndarray, base_ind: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """ For each dimension, find the interpolation weights to the right
        and left sides.
        """
        
        # First do a mapping from the structured ordering underlying this table, and
        # reflected in base_ind, to the unstructured ordering in self._table.
        _, _, ismem, ind_in_table = pp.array_operations.intersect_sets(base_ind, self._table._coords)
        raveled_ind = np.ravel(ind_in_table)
        # Sanity checks.
        assert np.all(ismem)
        assert all([len(s) == 1 for s in ind_in_table])
        
        # Find the weights on the right array.
        right_weight = np.array(
            [
                (x[i] - (self._pt[i,raveled_ind] ))  / self._h[i]
                for i in range(self._param_dim)
            ]
        )

        # Check that we have found the right interval
        assert np.all(right_weight >= 0) and np.all(right_weight <= 1)

        left_weight = 1 - right_weight

        return right_weight, left_weight

    def _find_base_vertex(self, coord: np.ndarray) -> np.ndarray:
        """Helper function to get the base (generalized lower-left) vertex of a
        hypercube.

        """

        ind = list()
        # performing Cartesian search per axis of the interpolation grid.
        for x_i, h_i, base_i in zip(coord, self._h, self._base_point):
            # cartesian search for uniform grid, floor division by mesh size
            ind.append(((x_i - base_i) // h_i).astype(int))

        return np.array(ind)