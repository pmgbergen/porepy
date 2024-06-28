"""The module contains various functions and classes useful for operations on numpy
ndarrays.

"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree


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
