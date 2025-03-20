"""The module contains various functions and classes useful for operations on numpy
ndarrays and various functions with set operations.

"""

from __future__ import annotations

from itertools import product

import numpy as np
from scipy.spatial import KDTree

from typing import Any, Tuple


class SparseNdArray:
    """Data structure for representing N-dimensional sparse arrays.

    The underlying data format is a generalization of the COO-format (in the
    scipy.sparse sense). The dimension is fixed upon initialization, but no bounds are
    set on the maximum and minimum coordinates. The coordinates are restricted to
    integers, thus, if the data to be stored lives a grid with non-integer 'grid-lines',
    the data should be mapped prior to feeding it to the data. To do this, it is
    recommended to consider the class AdaptiveInterpolationTable.

    Args:
        dims (int): Number of coordinate axes of the table.
        value_dim: Dimension of the values to be represented in the array. Defaults to 1.

    """

    # IMPLEMENTATION NOTE: It would be feasible to allow for non-integer coordinates,
    # e.g., to move the mapping from AdaptiveInterpolationTable to this class. However,
    # the current solution seems like a reasonable compromise; specifically, desicions
    # on what to do with accuracy of point errors should be left to clients of this
    # class.
    #
    # IMPLEMENTATION NOTE: Storage of vector data can easily be accommodated by making
    # self._values a 2d-array, but this has not been prioritized.
    #
    # IMPLEMENTATION NOTE: Depending on how this ends up being used, it could be useful
    # to implement a method 'sanity_check' which can check that the array coordinates
    # and values are within specified bounds.

    def __init__(self, dim: int, value_dim: int = 1) -> None:
        self.dim: int = dim

        # Data structure for the coordinates (an ndim x npt array), and values.
        # Initialize as empty arrays.
        self._coords: np.ndarray = np.ndarray((self.dim, 0), dtype=int)
        # Data structure for the values.
        self._values: np.ndarray = np.ndarray((value_dim, 0), dtype=float)

    def add(
        self, coords: list[np.ndarray], values: np.ndarray, additive: bool = False
    ) -> np.ndarray:
        """Add one or several new data points to the interpolation table.

        Values of new coordinates are added to the array. Coordinates that were already
        represented in the table have their values increased or overwritten, see
        parameter additive.

        Args:
            coords: List of coordinates, each corresponding to an indivdiual data point.
            values: New values. Each element corresponds to an item in the list of new
                coordinates.
            additive: If True, values associated with duplicate coordinates (either
                between new and existing coordinates, or within the new coordinates) are
                added. If False, existing values will be overwritten by the new value
                (if there are duplicates in new coordinates the last of this coordinates
                are used). Defaults to False.

        Returns:
            np.ndarray: Permutation vector applied before the coordinates and data were
                added to storage.

        """
        # IMPLEMENTATION NOTE: We could have passed coordinates as a 2d np.ndarray,
        # but the list of 1d arrays (each representing a coordinate) is better fit to
        # the expected use of this method. This could be revised at a later point.

        # Shortcut for empty coordinate array.
        if len(coords) == 0:
            return np.array([], dtype=int)

        values = np.atleast_2d(values)

        # The main idea is to do a search for the new coordinates in the array of
        # existing coordinates. This becomes simpler if we first remove duplicates in
        # the list of new coordinates.

        # Convert the coordinates to a numpy 2d array, fit for uniquification.
        coord_array = np.array([c for c in coords]).T
        # Uniquifying will also do a sort of the arrays. This will permute the order of
        # the coordinates when they are eventually stored in the main coordinate array,
        # but that should not be a problem, since the data is essentially unstructured.
        unique_coords, unique_2_all, all_2_unique, counts = np.unique(
            coord_array,
            axis=1,
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )
        all_2_unique = all_2_unique.ravel()

        # Next, consolidate the values for duplicate coordinates.
        if additive:
            # In additive mode, use numpy bincount. We need to split the arrays here,
            # since bincount only works with 1d weights.
            unique_values = np.vstack(
                [
                    np.bincount(all_2_unique, weights=values[i])
                    for i in range(values.shape[0])
                ]
            )
        else:
            # We need to overwrite (the right) data for duplicate points.
            if np.all(counts == 1):
                # No duplicates, simply map the coordinates.
                unique_values = values[:, all_2_unique]
            else:
                # We have duplicates. Find the last occurrence of each coordinate,
                # assign this to the new values.
                unique_values = np.zeros(
                    (self._values.shape[0], counts.size), dtype=float
                )
                for i in range(unique_coords.shape[1]):
                    unique_values[:, i] = values[:, np.where(all_2_unique == i)[0][-1]]

        # Next, find which of the new coordinates are already present in the coordinate
        # array.
        _, ind, is_mem, _ = intersect_sets(unique_coords, self._coords)

        # For the coordinates previously added, either add or overwrite the values.
        # There is no need to worry about duplicate coordinates in the new values.
        if additive:
            self._values[:, ind] += unique_values[:, is_mem]
        else:
            self._values[:, ind] = unique_values[:, is_mem]

        # Append the new coordinates and values to the storage arrays.
        new_coord = unique_coords[:, np.logical_not(is_mem)]
        if new_coord.ndim == 1:
            # Some formatting may be necessary here.
            new_coord = new_coord.reshape((-1, 1))

        self._coords = np.hstack((self._coords, new_coord))
        self._values = np.hstack(
            (self._values, unique_values[:, np.logical_not(is_mem)])
        )
        return unique_2_all[np.logical_not(is_mem)]

    def get(self, coords: list[np.ndarray]) -> np.ndarray:
        """Retrieve values from the sparse array.

        Args:
            coords (list[np.ndarray]): List of coordinates, each corresponding to an
                indivdiual data point.
        Raises:
            ValueError: If a coordinate is not present in this array.

        Returns:
            np.ndarray: Values associated with the items in coords.

        """
        # IMPLEMENTATION NOTE: We could have passed coordinates as a 2d np.ndarray,
        # but the list of 1d arrays (each representing a coordinate) is better fit to
        # the expected use of this method. This could be revised at a later point.

        # Convert coordinates to numpy nd-array.
        coord_array = np.array([c for c in coords]).T
        # Make an intersection between the sought-after coordinates and the coordinates
        # of this array.
        _, _, is_mem, ind_list = intersect_sets(coord_array, self._coords)

        # Sanity check, we should not ask for data not present in the array.
        if np.any(np.logical_not(is_mem)):
            raise ValueError("Inquiry on unassigned coordinate.")

        # The mapping from coord_array to self._coords is a nested list, but, since we
        # have taken care not to add duplicates from self._coord (see method add), we
        # known ind_list contains one element per column in coord_array. Thus, we can
        # simply ravel the ind_list to get an array of indices.
        ind = np.ravel(ind_list)

        # Return values.
        return self._values[:, ind]

    def __repr__(self) -> str:
        # String representation.
        s = f"Sparse array of dimension {self.dim}.\n"
        s += f"The values in the array are {self._values.shape[0]}-dimensional."
        s += f"There are {self._coords.shape[1]} coordinates present.\n"
        s += f"The minimum coodinate is {np.min(self._coords, axis=1)}."
        s += f"The maximum coodinate is {np.max(self._coords, axis=1)}.\n"
        s += "The array values are in the range "
        s += f"[{np.min(self._values)}, {np.max(self._values)}]"
        return s


def intersect_sets(
    a: np.ndarray, b: np.ndarray, tol: float = 1e-10
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Intersect two sets to find common elements up to a tolerance.

    Args:
        a (np.ndarray, shape nd x n_pts): The first set to be intersected. The columns
            of a form set members.
        b (np.ndarray, shape nd x n_pts): The second set to be intersected. The columns
            of b form set members.
        tol (float, optional): Tolerance used in comparison of set members. Defaults to
            1e-10.

    Returns:
        ia (np.ndarray, dtype int): Column indices of a, so that a[:, ia] is found in
            b. Information of which elements in a and b that are equal can be obtained
            from the return intersection.
        ib (np.ndarray, dtype int): Column indices of b, so that b[:, ib] is found in
            a. Information of which elements in a and b that are equal can be obtained
            from the return intersection.
        a_in_b (np.ndarray, dtype int): Element i is true if a[:, i]
            is found in b. To get the reverse, b_in_a, switch the order of the
            arguments.
        intersection (list of list of int): The outer list has one element for each
            column in a. Each element in the outer list is itself a list which contains
            column indices of all elements in b that corresponds to this element of a.
            If there is no such elements in b, the inner list is empty.

    """
    # The implementation relies on scipy.spatial's KDTree implementation. The idea is to
    # make a KD tree representation of each of the point sets and identify proximate
    # points in the two trees.

    a = np.atleast_2d(a)
    b = np.atleast_2d(b)

    # KD tree representations. We need to transpose to be compatible with the data
    # structure of KDTree.
    a_tree = KDTree(a.T)
    b_tree = KDTree(b.T)

    # Find points in b that are close to elements in a. The intersection is structured
    # as a list of length a.shape[1]. Each list item is itself list that contains the
    # indices to items in array b that that corresponds to this item in a (so, if
    # intersection[2] == [3], a[:, 2] and b[:, 3] are identical up to the given tolerance).
    intersection = a_tree.query_ball_tree(b_tree, tol)

    # Get the indices.
    if len(intersection) > 0:
        ib = np.hstack([np.asarray(i) for i in intersection]).astype(int)
        ia = np.array(
            [i for i in range(len(intersection)) if len(intersection[i]) > 0]
        ).astype(int)
    else:
        # Special treatment if no intersections are found (the above hstack raised an
        # error without this handling).
        ia = np.array([], dtype=int)
        ib = np.array([], dtype=int)

    # Uniquify ia and ib - this will also sort them.
    ia_unique = np.unique(ia)
    ib_unique = np.unique(ib)

    # Make boolean of whether elements in a are also in b.
    a_in_b = np.zeros(a.shape[-1], dtype=bool)
    a_in_b[ia] = True

    return ia_unique, ib_unique, a_in_b, intersection


def unique_rows(
    data: np.ndarray[Any, np.dtype[np.float64]],
) -> Tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.int64]],
    np.ndarray[Any, np.dtype[np.int64]],
]:
    r"""Function similar to Matlab's unique(...,'rows')

    See also function unique_columns in this module; this is likely slower, but
    is understandable, documented, and has a tolerance option.

    Copied from
    http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array/
    (summary pretty far down on the page)
    Note: I have no idea what happens here

    Parameters:
        data ``shape=(n, m)``.

            Array to be uniquified along its first array.

    Returns:
        Uniquified array.

        np.ndarray: ``shape=(n, r), r \leq m``. Unique rows.

        np.ndarray: ``shape=(r)`` Indexes that map from the unique to the original
            array.

        np.ndarray: ``shape=(m)`` Indexes that map from the original to the unique
            array.

    """
    b = np.ascontiguousarray(data).view(
        np.dtype((np.void, data.dtype.itemsize * data.shape[1]))
    )
    _, ia = np.unique(b, return_index=True)
    _, ic = np.unique(b, return_inverse=True)
    # Ravel index data to get 1d output
    return data[ia], ia.ravel(), ic.ravel()


def ismember_rows(
    a: np.ndarray[Any, np.dtype[np.int64]],
    b: np.ndarray[Any, np.dtype[np.int64]],
    sort: float = True,
) -> Tuple[np.ndarray[Any, np.dtype[np.bool_]], np.ndarray[Any, np.dtype[np.int64]]]:
    """
    Find *columns* of a that are also members of *columns* of b.

    The function mimics Matlab's function ismember(..., 'rows').

    TODO: Rename function, this is confusing!

    Parameters:
        a (np.array): Each column in a will search for an equal in b.
        b (np.array): Array in which we will look for a twin
        sort (boolean, optional): If true, the arrays will be sorted before
            seraching, increasing the chances for a match. Defaults to True.

    Returns:
        np.array (boolean): For each column in a, true if there is a
            corresponding column in b.
        np.array (int): Indexes so that b[:, ind] is also found in a.

    Examples:
        >>> a = np.array([[1, 3, 3, 1, 7], [3, 3, 2, 3, 0]])
        >>> b = np.array([[3, 1, 3, 5, 3], [3, 3, 2, 1, 2]])
        >>> ismember_rows(a, b)
        (array([ True,  True,  True,  True, False], dtype=bool), [1, 0, 2, 1])

        >>> a = np.array([[1, 3, 3, 1, 7], [3, 3, 2, 3, 0]])
        >>> b = np.array([[3, 1, 2, 5, 1], [3, 3, 3, 1, 2]])
        >>> ismember_rows(a, b, sort=False)
        (array([ True,  True, False,  True, False], dtype=bool), [1, 0, 1])

    """
    # IMPLEMENTATION NOTE: A serious attempt was made (June 2022) to speed up
    # this calculation by using functionality from scipy.spatial. This did not
    # work: The Scipy functions scaled poorly with the size of arrays a and b,
    # and lead to memory overflow for large arrays.

    # Sort if required, but not if the input is 1d
    if sort and a.ndim > 1:
        sa = np.sort(a, axis=0)
        sb = np.sort(b, axis=0)
    else:
        sa = a
        sb = b

    b = np.atleast_1d(b)
    a = np.atleast_1d(a)
    num_a = a.shape[-1]

    # stack the arrays
    c = np.hstack((sa, sb))
    # Uniquify c. We don't care about the unique array itself, but rather
    # the indices which maps the unique array back to the original c
    if a.ndim > 1:
        _, ind = np.unique(c, axis=1, return_inverse=True)
        ind = ind.ravel()
    else:
        _, ind = np.unique(c, return_inverse=True)

    # Indices in a and b referring to the unique array.
    # Elements in ind_a that are also in ind_b will correspond to rows
    # in a that are also found in b
    ind_a = ind[:num_a]
    ind_b = ind[num_a:]

    # Find common members
    ismem_a = np.isin(ind_a, ind_b)

    # Found this trick on
    # https://stackoverflow.com/questions/8251541/numpy-for-every-element-in-one-array-find-the-index-in-another-array
    # See answer by Joe Kington
    sort_ind = np.argsort(ind_b)
    ypos = np.searchsorted(ind_b[sort_ind], ind_a[ismem_a])
    ia = sort_ind[ypos]

    # Done
    return ismem_a, ia


def unique_columns_tol(
    mat: np.ndarray,
    tol: float = 1e-8,
) -> Tuple[
    np.ndarray,
    np.ndarray[Any, np.dtype[np.int64]],
    np.ndarray[Any, np.dtype[np.int64]],
]:
    """
    For an array, remove columns that are closer than a given tolerance.

    To uniquify a point set, consider using the function uniquify_point_set
    instead.

    Resembles Matlab's uniquetol function, as applied to columns. To rather
    work on rows, use a transpose.

    Parameters:
        mat (np.ndarray, nd x n_pts): Columns to be uniquified.
        tol (double, optional): Tolerance for when columns are considered equal.
            Should be seen in connection with distance between the points in
            the points (due to rounding errors). Defaults to 1e-8.

    Returns:
        np.ndarray: Unique columns.
        new_2_old: Index of which points that are preserved
        old_2_new: Index of the representation of old points in the reduced
            list.

    Example:
        >>> p_un, n2o, o2n = unique_columns(np.array([[1, 0, 1], [1, 0, 1]]))
        >>> p_un
        array([[1, 0], [1, 0]])
        >>> n2o
        array([0, 1])
        >>> o2n
        array([0, 1, 0])

    """
    import numba

    # Treat 1d array as 2d
    mat = np.atleast_2d(mat)

    # Some special cases
    if mat.shape[1] == 0:
        # Empty arrays gets empty return
        return mat, np.array([], dtype=int), np.array([], dtype=int)
    elif mat.shape[1] == 1:
        # Array with a single column needs no processing
        return mat, np.array([0], dtype=int), np.array([0], dtype=int)

    # If the matrix is integers, and the tolerance less than 1/2, we can use
    # numpy's unique function
    if issubclass(mat.dtype.type, np.int_) and tol < 0.5:
        un_ar, new_2_old, old_2_new = np.unique(
            mat, return_index=True, return_inverse=True, axis=1
        )
        # Ravel index arrays to get 1d output
        return un_ar, new_2_old.ravel(), old_2_new.ravel()

    @numba.jit("Tuple((b1[:],i8[:],i8[:]))(f8[:, :],f8)", nopython=True, cache=True)
    def _numba_distance(mat, tol):
        """Helper function for numba acceleration of unique_columns_tol.

        IMPLEMENTATION NOTE: Calling this function many times (it is unclear
        what this really means, but likely >=100s of thousands of times) may
        lead to enhanced memory consumption and significant reductions in
        performance. This could be related to this GH issue

            https://github.com/numba/numba/issues/1361

        However, it is not clear this is really the error. No solution is known
        at the time of writing, the only viable options seem to be algorithmic
        modifications that reduce the number of calls to this function.

        """
        num_cols = mat.shape[0]
        keep = np.zeros(num_cols, dtype=np.bool_)
        keep[0] = True
        keep_counter = 1

        # Map from old points to the unique subspace. Defaults to map to itself.
        old_2_new = np.arange(num_cols)

        # Loop over all points, check if it is already represented in the kept list
        for i in range(1, num_cols):
            d = np.sum((mat[i] - mat[keep]) ** 2, axis=1)
            condition = d < tol**2

            if np.any(condition):
                # We will not keep this point
                old_2_new[i] = np.argmin(d)
            else:
                # We have found a new point
                keep[i] = True
                old_2_new[i] = keep_counter
                keep_counter += 1

        # Finally find which elements we kept
        new_2_old = np.nonzero(keep)[0]

        return keep, new_2_old, old_2_new

    mat_t = np.atleast_2d(mat.T).astype(float)

    # IMPLEMENTATION NOTE: It could pay off to make a pure Python implementation
    # to be used for small arrays, however, attempts on making this work in
    # practice failed.

    keep, new_2_old, old_2_new = _numba_distance(mat_t, tol)
    return mat[:, keep], new_2_old, old_2_new


def uniquify_point_set(
    points: np.ndarray[Any, np.dtype[np.float64]], tol: float = 1e-8
) -> Tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.int64]],
    np.ndarray[Any, np.dtype[np.int64]],
]:
    """Uniquify a set of points so that no two sets of points are closer than a
    distance tol from each other.

    This function partially overlaps the function unique_columns_tol,
    but the latter is more general, as it provides fast treatment of integer
    arrays.

    FIXME: It should be possible to unify the two implementations, however,
    more experience is needed before doing so.

    Parameters:
        points (np.ndarray, nd x n_pts): Columns to be uniquified.
        tol (double, optional): Tolerance for when columns are considered equal.
            Should be seen in connection with distance between the points in
            the point set (due to rounding errors). Defaults to 1e-8.

    Returns:
        np.ndarray: Unique columns.
        new_2_old: Index of which points that are preserved
        old_2_new: Index of the representation of old points in the reduced
            list.

    """
    # The implementation uses Scipy's KDTree implementation to efficiently get
    # the distance between points.
    num_p = points.shape[1]
    # Transpose needed to comply with KDTree.
    tree = KDTree(points.T)

    # Get all pairs of points closer than the tolerance.
    pairs = tree.query_pairs(tol, output_type="ndarray")

    if pairs.size == 0:
        # No points were found, we can return
        return points, np.arange(num_p), np.arange(num_p)

    # Process information to arrive at a unique point set. This is technical,
    # since we need to deal with cases where more than two points coincide.
    # As an example: if the points p1, p2 and p3 coincide, they will be
    # identified either by the pairs {(i1, i2), (i1, i3)}, by
    # {(i1, i2), (i2, i3)}, or by {(i1, i3), (i2, i3)}). To be clear,
    # more than three points can coincide - such configurations will
    # include more point combinations, but will not introduce additional
    # complications.

    # Sort the index pairs of identical points for simpler identification.
    # NOTE: pairs, as returned by KDTree, is a num_pairs x 2 array, thus
    # sorting the pairs should be along axis 1.
    pair_arr = np.sort(pairs, axis=1)
    # Sort the pairs along axis=1. The lexsort will make the sorting first
    # according to pair_arr[:, 0] (the point with the lowest index in each
    # pair), and then according to the second index (pair_arr[:, 1]). The
    # result will be a lexicographically ordered array.
    # Also note the transport back to a 2 x num_pairs array.
    sorted_arr = pair_arr[np.lexsort((pair_arr[:, 1], pair_arr[:, 0]))].T

    # Find points that are both in the first and second row. Referring to the
    # example with three intersecting points, this will identify triplets
    # expressed as pairs {(i1, i2), (i2, i3)}.
    duplicate = np.isin(sorted_arr[0], sorted_arr[1])
    # Array with duplicates of the type {(i1, i2), (i1, i3)} removed.
    reduced_arr = sorted_arr[:, np.logical_not(duplicate)]

    # Also identify points that are not involved in any pairs, these should be
    # included in the unique set. Append these to the point array.
    not_in_pairs = np.setdiff1d(np.arange(points.shape[1]), pair_arr.ravel())
    reduced_arr = np.hstack((reduced_arr, np.tile(not_in_pairs, (2, 1))))

    # The array can still contain pairs of type {(i1, i2), (i1, i3)} and
    # {(i1, i3), (i1, i3)}, again referring to the example with three identical
    # points.
    # These can be identified by a unique on the first row.
    ia = np.unique(reduced_arr[0])

    # Make a mapping from all points to the reduced set.
    ib = np.arange(num_p)
    _, inv_map = np.unique(reduced_arr[0], return_inverse=True)
    ib[reduced_arr[0]] = inv_map
    ib[reduced_arr[1]] = ib[reduced_arr[0]]

    # Uniquify points.
    upoints = points[:, ia]

    # Done.
    return upoints, ia, ib


def accum(accmap: np.ndarray, a: np.ndarray):
    """An accumulation function similar to Matlab's `accumarray` function.

    Parameters:

        accmap: This is the "accumulation map". It maps input (i.e. indices into
            `a`) to their destination in the output array. The first `a.ndim`
            dimensions of `accmap` must be the same as `a.shape`. That is,
            `accmap.shape[:a.ndim]` must equal `a.shape`. For example, if `a`
            has shape (15,4), then `accmap.shape[:2]` must equal (15,4). In this
            case `accmap[i,j]` gives the index into the output array where
            element (i,j) of `a` is to be accumulated. If the output is, say,
            a 2D, then `accmap` must have shape (15,4,2). The value in the
            last dimension give indices into the output array. If the output is
            1D, then the shape of `accmap` can be either (15,4) or (15,4,1)

        a: The input data to be accumulated.

    Returns: The accumulated results.

    Examples:

        >>> from numpy import array, prod
        >>> a = array([[1,2,3],[4,-1,6],[-1,8,9]])
        >>> a
        array([[ 1,  2,  3],
            [ 4, -1,  6],
            [-1,  8,  9]])
        >>> # Sum the diagonals.
        >>> accmap = array([[0,1,2],[2,0,1],[1,2,0]])
        >>> s = accum(accmap, a)
        array([9, 7, 15])

    """
    return np.bincount(accmap.ravel(), weights=a.ravel())


def expand_indices_nd(ind: np.ndarray, nd: int, order="F") -> np.ndarray:
    """Expand indices from scalar to vector form.

    Examples:
    >>> i = np.array([0, 1, 3])
    >>> expand_indices_nd(i, 2)
    (array([0, 1, 2, 3, 6, 7]))

    >>> expand_indices_nd(i, 3, "C")
    (array([0, 3, 9, 1, 4, 10, 2, 5, 11])

    Parameters:
        ind: Indices to be expanded.
        nd: Dimension of the vector.
        order: Order of the expansion. "F" for Fortran, "C" for C. Default is "F".

    Returns:
        np.ndarray: Expanded indices.

    """
    if nd == 1:
        return ind
    dim_inds = np.arange(nd)
    dim_inds = dim_inds[:, np.newaxis]  # Prepare for broadcasting
    new_ind = nd * ind + dim_inds
    new_ind = new_ind.ravel(order)
    return new_ind


def expand_indices_incr(ind, dim, increment):
    # Convenience method for duplicating a list, with a certain increment

    # Duplicate rows
    ind_nd = np.tile(ind, (dim, 1))
    # Add same increment to each row (0*incr, 1*incr etc.)
    ind_incr = ind_nd + increment * np.array([np.arange(dim)]).transpose()
    # Back to row vector
    ind_new = ind_incr.reshape(-1, order="F")
    return ind_new


def mcolon(lo, hi):
    """Expansion of np.arange(a, b) for arrays a and b.

    The code is equivalent to the following (less efficient) loop:
    arr = np.empty(0)
    for l, h in zip(lo, hi):
        arr = np.hstack((arr, np.arange(l, h, 1)))

    Parameters:
        lo (np.ndarray, int): Lower bounds of the arrays to be created.
        hi (np.ndarray, int): Upper bounds of the arrays to be created. The
            elements in hi will *not* be included in the resulting array.

        lo and hi should either have 1 or n elements. If their size are both
        larger than one, they should have the same length.

    Examples:
        >>> mcolon(np.array([0, 0, 0]), np.array([2, 4, 3]))
        array([0, 1, 0, 1, 2, 3, 0, 1, 2])

        >>> mcolon(np.array([0, 1]), np.array([2]))
        array([0, 1, 1])

        >>> mcolon(np.array([0, 1, 1, 1]), np.array([1, 3, 3, 3]))
        array([0, 1, 2, 1, 2, 1, 2])

    Acknowledgements:
        The functions are a python translation of the corresponding matlab
        functions found in the Matlab Reservoir Simulation Toolbox (MRST) developed
        by SINTEF ICT, see www.sintef.no/projectweb/mrst/ .

    """
    # IMPLEMENTATION NOTE: The below code uses clever tricks to arrive at the correct
    # result (credit for the cleverness goes to SINTEF). The provided comments should
    # explain the logic behind the code to some extent, but if you really want to
    # understand what is going on, it is probably wise to go through the logic by hand
    # for a few examples.

    if lo.size == 1:
        lo = lo * np.ones(hi.size, dtype=np.int32)
    if hi.size == 1:
        hi = hi * np.ones(lo.size, dtype=np.int32)
    if lo.size != hi.size:
        raise ValueError(
            "Low and high should have same number of elements, or a single item "
        )

    # Find the positive differences. If there are none, return an empty array.
    pos_diff = hi >= lo + 1
    if not any(pos_diff):
        return np.array([], dtype=np.int32)

    # We only need the lows and highs where there is a positive difference.
    lo = lo[pos_diff].astype(int)
    # This changes hi from a non-inclusive to an inclusive upper bound of the range.
    hi = (hi[pos_diff] - 1).astype(int)
    # The number of elements to be generated for each interval (each pair of lo and hi).
    # We have to add 1 to include the upper bound.
    num_elements_in_interval = hi - lo + 1
    # Create an array with size equal to the total number of elements to be generated.
    # Initialize the array with ones, so that, if we do a cumulative sum, the indices
    # will be increasing by one. This will work, provided we can also set the elements
    # corresponding to the start of an interval (a lo value in a lo-hi pair) to its
    # correct value.
    tot_num_elements = np.sum(num_elements_in_interval)
    x = np.ones(tot_num_elements, dtype=int)

    # Set the first element to the first lo value.
    x[0] = lo[0]
    # For the elements corresponding to the start of interval 1, 2, ..., we cannot just
    # set the value to lo[1:], since this will be overriden by the cumulative sum.
    # Instead, set the value to lo[1:] - hi[0:-1], that is, to the difference between
    # the end of the previous interval and the start of the current interval. Under a
    # cumulative sum, this will give the correct value for the start of interval i,
    # provided that interval i-1 ends at the correct value, and, since we set the first
    # element to lo[0], with ones up to lo[1], this all works out by induction. Kudos to
    # whomever came up with this!
    x[np.cumsum(num_elements_in_interval[0:-1])] = lo[1:] - hi[0:-1]
    # Finally, a cumulative sum will give the correct values for all elements. Use x to
    # store the result to avoid allocating a new array (this can give substantial
    # savings for large arrays).
    return np.cumsum(x, out=x)
