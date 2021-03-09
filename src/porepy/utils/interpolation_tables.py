""" The module contains interpolation tables, intended for use in function
evalutaions. Specifically, the motivation is to facilitate the parametrization
framework described in

    Operator-based linearization approach for modeling of multiphase
    multi-component flow in porous media by Denis Voskov (JCP 2017)

The module contains two classes:

    InterpolationTable: Interpolation based on precomputation of function values.
        Essentially this is a cumbersome implementation of scipy interpolation
        functionality, and the latter is likely to be preferred.

    AdaptiveInterpolationTable: Interpolation where function values are computed
        and stored on demand. Can give significant computational speedup in cases
        where function evaluations are costly and only part of the parameter space
        is accessed during simulation.

Both classes use piecewise linear interpolation of functions, and piecewise
constant approximations of derivatives.

"""
import itertools
from typing import Callable, Iterator, List, Tuple

import numpy as np

import porepy as pp

module_sections = ["parameters"]


class InterpolationTable:
    """Interpolation table based on precomputation of function values.

    Function values are interpolated on a Cartesian mesh.
    The interpolation is done using piecewise linears (for function values) and
    constants (for derivatives). The domain of interpolation is an Nd box.

    The implementation may not be efficient, consider using functions from
    scipy.interpolate instead.

    """

    @pp.time_logger(sections=module_sections)
    def __init__(
        self,
        low: np.ndarray,
        high: np.ndarray,
        npt: np.ndarray,
        function: Callable[[np.ndarray], np.ndarray],
        dim: int = 1,
    ) -> None:
        """Constructor for the interpolation Table.

        Parameters:
            low (np.ndarray): Minimum values of the domain boundary per dimension.
            high (np.ndarray): Maximum values of the domain boundary per dimension.
            npt (np.ndarray): Number of interpolation points (including endpoints
                of intervals) per dimension.
            function (Callable): Function to represent in the table. Should be
                vectorized (if necessary with a for-loop wrapper) so that multiple
                coordinates can be evaluated at the same time.
            dim (int): Dimension of the range of the function. Values above one
                have not been much tested, use with care.

        """

        # Data processing is left to a separate function.
        self._set_sizes(low, high, npt, dim)

        # Evaluate the function values in all coordinate points.
        for i, c in enumerate(zip(*self._coord)):
            self._values[:, i] = function(*c)

    @pp.time_logger(sections=module_sections)
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

        # Define the interpolation points along each coordinate access.
        self._pt: List[np.ndarray] = [
            np.linspace(low[i], high[i], npt[i]) for i in range(self._param_dim)
        ]

        # Create interpolation grid.
        # The indexing, together with the Fortran-style raveling is necessary
        # to get a reasonable ordering of the expanded coefficients.
        coord_table = np.meshgrid(*self._pt, indexing="ij")
        # Ravel into an array
        self._coord: List[np.ndarray] = [c.ravel("F") for c in coord_table]

        # Set the strides necessary to advance to the next point in each dimension.
        # Refers to indices in self._coord. Is unity for first dimension, then
        # number of interpolation points in the first dimension etc.
        tmp = np.hstack((1, self._npt))
        self._strides = np.cumprod(tmp)[: self._param_dim].reshape((-1, 1))

        # Prepare table of function values. This will be filled in by __init__
        sz = np.prod(npt)
        self._values: np.ndarray = np.zeros((dim, sz))  # type: ignore

    @pp.time_logger(sections=module_sections)
    def interpolate(self, x: np.ndarray) -> np.ndarray:
        """Perform interpolation on a Cartesian grid by a piecwise linear
        approximation.

        Parameters:
            x (np.ndarray): Points to evaluate the function. Size dimension of
                parameter space times number of points.

        Returns:
            np.ndarray: Function values.

        """

        # Get indices of the base vertexes of the hypercubes where the function
        # is to be evaluated. The base vertex is in the (generalized) lower-left
        # corner of the cube.
        base_ind = self._find_base_vertex(x)
        # Get weights in each dimension for the interpolation between the higher
        # (right) and base (left) coordinate.
        right_weight, left_weight = self._right_left_weights(x, base_ind)

        # Loop over all vertexes in the hypercube. Evaluate the function in the
        # vertex with the relevant weight.
        for i, (incr, eval_ind) in enumerate(self._generate_indices(base_ind)):
            # Incr is 0 for dimensions to be evaluated in the left (base)
            # coordinate, 1 for others.
            # eval_ind is the linear index for this vertex.

            # Compute weight for this vertex
            weight = np.prod(
                right_weight * incr + left_weight * (1 - incr), axis=0
            )  # Not sure about self.dim > 1.

            # Add this part of the function evaluation and store it
            new_val = weight * self._values[:, eval_ind]
            if i == 0:
                # create array if this is the first iteration.
                values: np.ndarray = new_val
            else:
                values += new_val

        return values

    @pp.time_logger(sections=module_sections)
    def diff(self, x: np.ndarray, axis: int) -> np.ndarray:
        """Perform differentiation on a Cartesian grid by a piecwise constant
        approximation.

        Parameters:
            x (np.ndarray): Points to evaluate the function. Size dimension of
                parameter space times number of points.
            axis (int): Axis to differentiate along.

        Returns:
            np.ndarray: Function values.

        """
        # Get indices of the base vertexes of the hypercubes where the function
        # is to be evaluated. The base vertex is in the (generalized) lower-left
        # corner of the cube.
        base_ind = self._find_base_vertex(x)
        # Get weights in each dimension for the interpolation between the higher
        # (right) and base (left) coordinate.
        right_weight, left_weight = self._right_left_weights(x, base_ind)

        # placeholder variable for the function evaulation
        values = None

        # Loop over all vertexes in the hypercube. Evaluate the function in the
        # vertex with the relevant weight, and use a first order difference
        # to evaluate the derivative.
        for incr, eval_ind in self._generate_indices(base_ind):
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
            new_val = weight * self._values[:, eval_ind]
            if values is None:
                values = new_val
            else:
                values += new_val

        # Denominator in the finite difference approximation.
        denominator = (
            self._pt[axis][base_ind[axis] + 1] - self._pt[axis][base_ind[axis]]
        )
        #        breakpoint()
        return values / denominator

    @pp.time_logger(sections=module_sections)
    def _find_base_vertex(self, coord: np.ndarray) -> np.ndarray:
        # Helper function to get the base (generalized lower-left) vertex of a
        # hypecube.

        ind = np.zeros(coord.shape, dtype=int)

        # For each dimension, find the first index where coordinate in the
        # interpolation grid is higher than that of the point to be evaluated.
        # https://stackoverflow.com/questions/16243955/numpy-first-occurrence-of-value-greater-than-existing-value
        for i in range(ind.shape[1]):
            ind[:, i] = [np.argmax(pt > c) for (c, pt) in zip(coord[:, i], self._pt)]

        # Adjust to get the last interpolation point with a lower coordinate
        ind -= 1
        # If a point is outside the interpolation grid, the argmax will return 0
        # (either because the point is to the left, or it is to the right, which case
        # the entire array to argmax is zero, thus the first value is 0)
        if np.any(ind < 0):
            raise ValueError(
                f"Point outside coordinate range [{self._low}, {self._high}]"
            )

        return ind

    @pp.time_logger(sections=module_sections)
    def _generate_indices(
        self, base_ind: np.ndarray
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        # Iterator for linear indices that form the vertexes of a hypercube with a
        # given base vertex, and the dimension-wise increments in indices from the
        # base.

        # IMPLEMENTATION NOTE: Could have used np.unravel_index here. That may be
        # faster, if this ever turns out to be a bottleneck.
        for increment in itertools.product(range(2), repeat=self._param_dim):
            incr = np.asarray(increment).reshape((-1, 1))
            vertex_ind = base_ind + incr
            eval_ind = np.sum(vertex_ind * self._strides, axis=0).ravel()
            assert isinstance(eval_ind, np.ndarray)
            yield incr, eval_ind

    @pp.time_logger(sections=module_sections)
    def _right_left_weights(
        self, x: np.ndarray, base_ind: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # For each dimension, find the interpolation weights to the right
        # and left sides

        # Find corners of interpolation hypercube
        base_ind = self._find_base_vertex(x)

        right_weight = np.array(
            [
                (x[i] - self._pt[i][base_ind[i]])
                / (self._pt[i][base_ind[i] + 1] - self._pt[i][base_ind[i]])
                for i in range(self._param_dim)
            ]
        )
        # Check that we have bound the right interval
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

    """

    @pp.time_logger(sections=module_sections)
    def __init__(
        self,
        low: np.ndarray,
        high: np.ndarray,
        npt: np.ndarray,
        function: Callable[[np.ndarray], np.ndarray],
        dim: int = 1,
    ) -> None:
        """Constructor for the interpolation Table.

        Parameters:
            low (np.ndarray): Minimum values of the domain boundary per dimension.
            high (np.ndarray): Maximum values of the domain boundary per dimension.
            npt (np.ndarray): Number of interpolation points (including endpoints
                of intervals) per dimension.
            function (Callable): Function to represent in the table. Should be
                vectorized (if necessary with a for-loop wrapper) so that multiple
                coordinates can be evaluated at the same time.
            dim (int): Dimension of the range of the function. Values above one
                have not been much tested, use with care.

        """
        # Construct grid for interpolation
        self._set_sizes(low, high, npt, dim)

        # Store function
        self._function = function

        # Keep track of which grid points have had their values computed.
        num_pt = np.prod(npt)
        self._has_value: np.ndarray = np.zeros(num_pt, dtype=bool)  # type: ignore

    @pp.time_logger(sections=module_sections)
    def interpolate(self, x):
        """Perform interpolation on a Cartesian grid by a piecwise linear
        approximation. Compute and store the necessary function values.

        Parameters:
            x (np.ndarray): Points to evaluate the function. Size dimension of
                parameter space times number of points.

        Returns:
            np.ndarray: Function values.

        """
        # Fill any missing function values.
        self._fill_values(x)
        # Use standard method for interpolation.
        return super().interpolate(x)

    @pp.time_logger(sections=module_sections)
    def diff(self, x, axis):
        """Perform differentiation on a Cartesian grid by a piecwise constant
        approximation.

        Parameters:
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

    @pp.time_logger(sections=module_sections)
    def _fill_values(self, x: np.ndarray):
        # Find points in the interpolation grid that will be used for function
        # evaluation. Compute function values as needed.

        # The lower-left corner of each hypercube.
        base_ind = self._find_base_vertex(x)

        # Loop over all vertexes in the hypercube, store linear index.
        ind = []
        for _, eval_ind in self._generate_indices(base_ind):
            ind.append(eval_ind)

        # Uniquify indices to avoid computing values for the same vertex twice.
        unique_ind = np.unique(np.hstack(ind))
        # Find which vertexes has not been used before.
        values_needed = unique_ind[np.logical_not(self._has_value[unique_ind])]

        # Compute and store function values.
        for ind in values_needed:
            coord = np.array([self._coord[d][ind] for d in range(x.shape[0])])
            self._values[:, ind] = self._function(*coord)

        # Update list of funcions with known values.
        self._has_value[values_needed] = 1
