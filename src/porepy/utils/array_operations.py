"""The module contains various functions and classes useful for operations on numpy
ndarrays and various functions with set operations.

"""

from __future__ import annotations

from typing import Any, Literal, Tuple

import numpy as np
from numba import njit
from scipy.spatial import KDTree


class SparseNdArray:
    """Data structure for representing N-dimensional sparse arrays.

    The underlying data format is a generalization of the COO-format (in the
    scipy.sparse sense). The dimension is fixed upon initialization, but no bounds are
    set on the maximum and minimum coordinates. The coordinates are restricted to
    integers, thus, if the data to be stored lives a grid with non-integer 'grid-lines',
    the data should be mapped prior to feeding it to the data. To do this, it is
    recommended to consider the class AdaptiveInterpolationTable.

    Parameters:
        dims: Number of coordinate axes of the table.
        value_dim: Dimension of the values to be represented in the array. Defaults to
            1.

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
    # intersection[2] == [3], a[:, 2] and b[:, 3] are identical up to the given
    # tolerance).
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


def unique_rows(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finds unique rows in a 2D NumPy array using np.unique with axis=0.

    Parameters:
        data: A 2D NumPy array of shape `(n, m)`.

    Returns:
        Tuple containing:
            An array of the unique rows.
            Indices of first occurrences of unique rows.
            Inverse indices mapping original rows to unique rows.

    Example:
        >>> arr = np.array([[1, 2], [3, 4], [1, 2], [5, 6]])
        >>> unique_rows(arr)
        (array([[1, 2], [3, 4], [5, 6]]), array([0, 1, 3]), array([0, 1, 0, 2]))

    """
    error_message = "Use `unique_vectors_tol` for comparing floats."
    assert np.issubdtype(data.dtype, np.integer), error_message

    unique, idx, inverse = np.unique(
        data, axis=0, return_index=True, return_inverse=True
    )
    return unique, idx, inverse


def unique_columns(
    data: np.ndarray, preserve_order: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finds unique columns in a 2D numpy array.

    Parameters:
        data: The 2D numpy array that we want to find unique columns in.
        preserve_order: If `True`, guarantees to preserve the order, so each unique
            vector is the first encountered. That is for the input
            `[[5, 1, 5, 1], [1, 5, 1, 5]]`, the output will be `[[5, 1], [1, 5]]`,
            and not `[[1, 5], [5, 1]]`. Otherwise, the ordering is controlled by numpy
            and is not guaranteed.

    Returns:
        Tuple containing:
            An array of the unique columns.
            An array of the mapping between the new and old collection of columns.
            An array of the mapping between the old and new collection of columns.

    """
    if data.shape[1] == 0:
        return (
            np.empty_like(data),
            np.empty(shape=data.shape[1], dtype=np.int64),
            np.empty(shape=data.shape[1], dtype=np.int64),
        )

    error_message = "Use `unique_vectors_tol` for comparing floats."
    assert np.issubdtype(data.dtype, np.integer), error_message
    un_ar, new_2_old, old_2_new = np.unique(
        data, axis=1, return_index=True, return_inverse=True
    )
    # Ravel index arrays to get 1d output
    new_2_old = new_2_old.ravel()
    old_2_new = old_2_new.ravel()

    if not preserve_order:
        return un_ar, new_2_old, old_2_new

    # Sorting to preserve the order.
    ordering = np.argsort(new_2_old)
    un_ar = un_ar[:, ordering]
    new_2_old = new_2_old[ordering]

    # Constructing the reverse mapping for old_2_new.
    # This is equivalent to:
    # tmp = np.zeros_like(old_2_new)
    # for k, v in enumerate(ordering):
    #     tmp[old_2_new == v] = k
    # old_2_new = tmp
    lookup = np.zeros(ordering.max() + 1, dtype=int)
    lookup[ordering] = np.arange(len(ordering))
    old_2_new = lookup[old_2_new]

    return un_ar, new_2_old, old_2_new


def ismember_columns(
    a: np.ndarray[Any, np.dtype[np.int64]],
    b: np.ndarray[Any, np.dtype[np.int64]],
    sort: bool = True,
) -> Tuple[np.ndarray[Any, np.dtype[np.bool_]], np.ndarray[Any, np.dtype[np.int64]]]:
    """
    Find *columns* of a that are also members of *columns* of b.

    The function mimics Matlab's function ismember(..., 'rows').

    Parameters:
        a: `shape=(nd, num_vectors_a)`, each column in a will search for an equal in b.
        b: `shape=(nd, num_vectors_b)`, array in which we will look for a twin.
        sort: If `True`, treats permuted columns as twins. E.g., will treat `[5, 1]`
            and `[1, 5]` as unique. Defaults to `True`.

    Returns:
        np.array: `shape=(nd, num_vectors_a)`, `True` if there is a twin column in b.
        np.array: `shape=(nd, num_unique)`, Indices so that b[:, ind] only consists of
            columns also found in a, see examples.

    Examples:
        ```
        >>> a = np.array([[1, 3, 3, 1, 7],
                          [3, 3, 2, 3, 0],])
        >>> b = np.array([[3, 1, 3, 5, 3],
                          [3, 3, 2, 1, 2],])
        >>> ib, ic = ismember_columns(a, b)
        >>> ib
        array([ True,  True,  True,  True, False])
        >>> ic
        array([1, 0, 2, 1])
        >>> a[:, ib]
        array([[1, 3, 3, 1],
               [3, 3, 2, 3],])
        >>> b[:, ic]
        array([[1, 3, 3, 1],
               [3, 3, 2, 3],])

        >>> a = np.array([[1, 3, 3, 1, 7],
                          [3, 3, 2, 3, 0]])
        >>> b = np.array([[3, 1, 2, 5, 1],
                          [3, 3, 3, 1, 2]])
        >>> ismember_columns(a, b, sort=False)
        (array([ True,  True, False,  True, False], dtype=bool), [1, 0, 1])
        ```

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

    num_a = sa.shape[-1]

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


def expand_indices_nd(
    ind: np.ndarray, nd: int, order: Literal["F", "C"] = "F"
) -> np.ndarray:
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


def expand_indices_add_increment(x: np.ndarray, n: int, increment: int) -> np.ndarray:
    """Expend array values and consequently add the increment to each next repetition.
    Using Fortran order.

    Examples:
        >>> i = np.array([0, 1, 3])
        >>> repeat_array_add_increment(i, 3, 200)
        (array([0, 200, 400, 1, 201, 401, 3, 203, 403]))

    Parameters:
        x: Array to be repeated with the increment.
        n: How many times to repeat the array with the increment.
        increment: The number to add to each next repetition.

    Returns:
        The repeated array in Fortran order.

    """
    # Duplicate rows
    ind_nd = np.tile(x, (n, 1))
    # Add same increment to each row (0*incr, 1*incr etc.)
    ind_incr = ind_nd + increment * np.array([np.arange(n)]).transpose()
    # Back to row vector
    ind_new = ind_incr.reshape(-1, order="F")
    return ind_new


def expand_index_pointers(lo, hi):
    """Expansion of np.arange(a, b) for arrays a and b.

    The code is equivalent to the following (less efficient) loop:
    arr = np.empty(0)
    for l, h in zip(lo, hi):
        arr = np.hstack((arr, np.arange(l, h, 1)))

    The function mimics Matlab's function mcolon(...).

    Parameters:
        lo (np.ndarray, int): Lower bounds of the arrays to be created.
        hi (np.ndarray, int): Upper bounds of the arrays to be created. The
            elements in hi will *not* be included in the resulting array.

        lo and hi should either have 1 or n elements. If their size are both
        larger than one, they should have the same length.

    Examples:
        >>> expand_index_pointers(np.array([0, 0, 0]), np.array([2, 4, 3]))
        array([0, 1, 0, 1, 2, 3, 0, 1, 2])

        >>> expand_index_pointers(np.array([0, 1]), np.array([2]))
        array([0, 1, 1])

        >>> expand_index_pointers(np.array([0, 1, 1, 1]), np.array([1, 3, 3, 3]))
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


@njit(cache=True)
def _unique_vectors_in_cluster(
    vectors: np.ndarray,
    sorted_idx: np.ndarray,
    cluster_start: int,
    cluster_size: int,
    tol: float,
):
    """This is an inner function of `unique_vectors_tol`, but moved outside due to
    numba. Finds unique vectors in a specified cluster.

    Parameters:
        vectors: `shape=(nd, n_pts)`, vectors to be uniquified.
        sorted_idx: `shape=(n_pts,)`, sorted index of the vectors norms.
        cluster_start: Index of the cluster start.
        cluster_size: Number of vectors in the cluster.
        tol: Tolerance for when vectors are considered equal.

    Returns:
        np.ndarray: Unique vectors within the cluster.
        new_2_old: Index of which vectors that are preserved within the cluster.
        old_2_new: Index of the representation of old vectors in the reduced list.

    """
    unique_cols = np.zeros((vectors.shape[0], cluster_size), dtype=vectors.dtype)
    new_2_old = np.full(shape=cluster_size, fill_value=vectors.shape[1], dtype=np.int64)
    old_2_new = np.zeros(cluster_size, dtype=np.int64)
    # Number of found unique vectors in this cluster.
    keep = 0
    # Slice of the result corresponding to the found unique vectors.
    unique_cols_keep = unique_cols[:, :keep]

    # Iterating over the cluster.
    for i in range(cluster_start, cluster_start + cluster_size):
        col = vectors[:, sorted_idx[i]]
        # Comparing against known unique vectors in this cluster.
        within_tol = np.sum((col[:, None] - unique_cols_keep) ** 2, axis=0) < tol**2

        if not np.any(within_tol):
            # This is a new vector.
            unique_cols[:, keep] = col
            old_2_new[i - cluster_start] = keep
            new_2_old[keep] = sorted_idx[i]
            # Increasing the number of known unique vectors and reslicing.
            keep += 1
            unique_cols_keep = unique_cols[:, :keep]
        else:
            # This is not a new vector. Finding the index of its unique twin.
            idx = np.argmax(within_tol)
            old_2_new[i - cluster_start] = idx

            if sorted_idx[i] < new_2_old[idx]:
                # New vector goes before its stored twin. We replace the twin with it.
                new_2_old[idx] = sorted_idx[i]
                unique_cols[:, idx] = col

    return unique_cols_keep, new_2_old[:keep], old_2_new


@njit(cache=True)
def unique_vectors_tol(vectors: np.ndarray, tol: float):
    """Uniquify a set of vectors so that no two sets of vectors are closer than a
    distance tol from each other.

    The result order is guaranteed, each unique vector is the first encountered. That is
    for the input `[[5, 1, 5, 1], [1, 5, 1, 5]]`, the output will be `[[5, 1], [1, 5]]`,
    and not `[[1, 5], [5, 1]]`.

    For integers and `tol < 0.5`, this is equivalent to
    `unique_columns(vectors, preserve_order=True)`.

    Implementation note: It groups the vectors into clusters based on similar norms.
    Then it filters unique vectors in each cluster. This avoids heavy comparison of
    all to all.

    Parameters:
        vectors: `shape=(nd, n_pts)`, vectors to be uniquified.
        tol: Tolerance for when columns are considered equal.
            Should be seen in connection with distance between the vectors in
            the vector set (due to rounding errors).

    Returns:
        np.ndarray: `shape=(nd, n_unique)`, unique vectors.
        new_2_old: `shape=(n_unique,)`, index of which vectors that are preserved.
        old_2_new: `shape=(n_pts,)`, index of the representation of old vectors in
            the reduced list.

    """
    if vectors.shape[1] == 0:
        return (
            np.empty_like(vectors),
            np.empty(shape=vectors.shape[1], dtype=np.int64),
            np.empty(shape=vectors.shape[1], dtype=np.int64),
        )

    # Finding norms of each vector and sorting them.
    vector_norms = np.sqrt(np.sum(vectors**2, axis=0))
    sorted_idx = np.argsort(vector_norms)

    # Counting vectors with close norms.
    close_norms_count = np.zeros(vectors.shape[1], dtype=np.int64)
    # Index corresponds to the current cluster of vectors with close norms.
    cluster_idx = 0
    # Norm of a vector in the current cluster.
    cluster_norm = vector_norms[sorted_idx[0]]

    # Iterating through sorted norms.
    for current_norm in vector_norms[sorted_idx]:
        # Reverse triangle inequality: abs(||x|| - ||y||) <= ||x - y||.
        # If they don't fall into one cluster, they are certainly not close within tol.
        if abs(cluster_norm - current_norm) > tol:
            # Norms are not close. Moving to the next cluster.
            cluster_idx += 1
            cluster_norm = current_norm

        close_norms_count[cluster_idx] += 1

    # Keeping only nonzero clusters.
    close_norms_count = close_norms_count[: cluster_idx + 1]
    # Current number of unique vectors.
    num_unique = 0
    # Index of the cluster start.
    cluster_start = 0

    # Initializing result arrays.
    unique_pts = np.zeros_like(vectors)
    new_2_old = np.zeros(vectors.shape[1], dtype=np.int64)
    old_2_new = np.zeros(vectors.shape[1], dtype=np.int64)

    # Iterating through the clusters of vectors with close norms.
    for cluster_size in close_norms_count:
        # Compare vectors and find unique ones. O(m^2) for m vectors in a cluster.
        unique_pts_inner, new_2_old_inner, old_2_new_inner = _unique_vectors_in_cluster(
            vectors=vectors,
            sorted_idx=sorted_idx,
            cluster_start=cluster_start,
            cluster_size=cluster_size,
            tol=tol,
        )
        # Number of unique vectors in the cluster.
        unique_size_inner = unique_pts_inner.shape[1]

        # Updating the global result arrays.
        unique_pts[:, num_unique : num_unique + unique_size_inner] = unique_pts_inner
        new_2_old[num_unique : num_unique + unique_size_inner] = new_2_old_inner
        old_2_new[
            sorted_idx[np.arange(cluster_start, cluster_start + cluster_size)]
        ] = old_2_new_inner + num_unique

        # Updating the counters.
        num_unique += unique_size_inner
        cluster_start += cluster_size

    # Slicing result arrays to remove unused space.
    unique_pts = unique_pts[:, :num_unique]
    new_2_old = new_2_old[:num_unique]

    # Reordering to preserve the first-encountered order.
    ordering = np.argsort(new_2_old)
    unique_pts = unique_pts[:, ordering]
    new_2_old = new_2_old[ordering]

    # Constructing the reverse mapping for old_2_new.
    # This is equivalent to:
    # tmp = np.zeros_like(old_2_new)
    # for k, v in enumerate(ordering):
    #     tmp[old_2_new == v] = k
    # old_2_new = tmp
    lookup = np.zeros(ordering.max() + 1, dtype=np.int64)
    lookup[ordering] = np.arange(len(ordering))
    old_2_new = lookup[old_2_new]

    return unique_pts, new_2_old, old_2_new
