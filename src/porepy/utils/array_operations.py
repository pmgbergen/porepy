"""The module contains various functions and classes useful for operations on numpy
ndarrays.

"""
from scipy.spatial import KDTree
import numpy as np


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
    ib = np.hstack([np.asarray(i) for i in intersection]).astype(int)
    ia = np.array(
        [i for i in range(len(intersection)) if len(intersection[i]) > 0]
    ).astype(int)

    # Uniquify ia and ib - this will also sort them.
    ia_unique = np.unique(ia)
    ib_unique = np.unique(ib)

    # Make boolean of whether elements in a are also in b.
    a_in_b = np.zeros(a.shape[-1], dtype=bool)
    a_in_b[ia] = True

    return ia_unique, ib_unique, a_in_b, intersection
