"""Test helpers for arrays and matrices."""

import numpy as np
import scipy.sparse as sps
import porepy as pp


def compare_arrays(
    a: np.ndarray, b: np.ndarray, tol: float = 1e-4, sort: bool = True
) -> bool:
    """Compare two arrays and check that they are equal up to a column permutation.

    One use case is to compare coordinate arrays.

    Parameters:
        a: First array to be compared.
        b: Second array to be compared.
        tol: Tolerance used in comparison.
        sort: Sort arrays columnwise before comparing

    Returns:
        True if there is a permutation ind so that all(a[:, ind] == b).

    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)

    if not np.all(a.shape == b.shape):
        return False

    if sort:
        a = np.sort(a, axis=0)
        b = np.sort(b, axis=0)

    for i in range(a.shape[1]):
        dist = np.sum((b - a[:, i].reshape((-1, 1))) ** 2, axis=0)
        if dist.min() > tol:
            return False
    for i in range(b.shape[1]):
        dist = np.sum((a - b[:, i].reshape((-1, 1))) ** 2, axis=0)
        if dist.min() > tol:
            return False
    return True


def compare_matrices(m1: sps.spmatrix, m2: sps.spmatrix, tol: float = 1e-10) -> bool:
    """Compare two matrices.

    Parameters:
        m1: First matrix to be compared.
        m2: Second matrix to be compared.
        tol: Tolerance used in comparison.

    Returns:
        True if the matrices are identical up to a small tolerance.

    """
    if m1.shape != m2.shape:
        # Matrices with different shape are unequal, unless the number of
        # rows and columns is zero in at least one dimension.
        if m1.shape[0] == 0 and m2.shape[0] == 0:
            return True
        elif m1.shape[1] == 0 and m2.shape[1] == 0:
            return True
        return False
    d = m1 - m2

    if d.data.size > 0:
        if np.max(np.abs(d.data)) > tol:
            return False
    return True


def projection_matrix_from_matrix_slicer(
    slicers: pp.matrix_operations.MatrixSlicer
    | list[pp.matrix_operations.MatrixSlicer],
    dim: int,
) -> sps.coo_matrix:
    """Recover a projection matrix from a one or multiple matrix slicers.

    For a set of slicers {P_1, P_2, ..., P_n}, obtain the matrix P that corresponds to

        P = P_1 + P_2 + ... + P_n.

    Parameters:
        slicers: Basis vector(s) as one or more operators.
        dim: Dimension of the domain space of the slicer.

    Returns:
        Matrix representation of the basis vector.

    """
    # Always deal with a list of slicers.
    if isinstance(slicers, pp.matrix_operations.MatrixSlicer):
        slicers = [slicers]

    # Initialize result.
    result = None

    for slicer in slicers:
        if result is None:
            result = slicer @ np.eye(dim)
        else:
            result += slicer @ np.eye(dim)

    return sps.coo_matrix(result)
