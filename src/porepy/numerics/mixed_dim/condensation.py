"""
The static condensation or Schur complement proceedure 
It can e.g. be used to improve condition numbers when solving linear
systems by removing the 0d fracture intersection cells.
"""


import numpy as np
import scipy.sparse as sps
import copy

from porepy.grids import grid_bucket


def solve_static_condensation(A, rhs, gb, dim=0, condensation_inverter=sps.linalg.inv,
                              system_inverter=sps.linalg.spsolve):
    """
    A call to this function uses a static condensation to solve a linear
    problem without the degrees of freedom related to grids of dimension dim.

    Input:
        A (sps.csr_matrix): Original matrix and right hand side of the problem to be
            solved.
        rhs (np.array): Original matrix and right hand side of the problem to be
            solved.
        dim: The dimension one wishes to get rid of. No tests for dim>0.
        condensation_inverter: The inverter of the (small) system solved 
            to perform the static condensation.
        system_inverter: Inverter for solving the problem after static 
            condensation has been performed.
    Returns:
        x: The solution vector corresponding to the initial system, i.e.,
            with all dofs.
        x_reduced: The solution vector for the reduced system, i.e.,
            corresponding to the master dofs only.
        original_to_kept_dofs: Mapping from the full to the reduced set of 
            degrees of freedom.
        eliminated_dofs: Mapping from the full to the removed set of degrees of
            freedom (i.e., which of the initial correspond to grids of dimension
            dim).
    """
    to_be_eliminated = dofs_of_dimension(gb, A, dim)

    a_reduced, rhs_reduced, Condensation_matrix, original_to_kept_dofs = eliminate_dofs(
        A, rhs, to_be_eliminated, condensation_inverter)

    eliminated_dofs = np.nonzero(to_be_eliminated)[0]

    x_reduced = system_inverter(a_reduced, rhs_reduced)

    x = np.zeros(A.shape[0])
    x[original_to_kept_dofs] = x_reduced
    x[to_be_eliminated] = Condensation_matrix * x_reduced

    return x, x_reduced, original_to_kept_dofs, eliminated_dofs


def dofs_of_dimension(gb, A, dim=0):
    """
    Extracts the global dof numbers corresponding to a given dimension.
    Returns a boolean mask extracting the dofs to be eliminated.
    """
    original_ndof = A.shape[1]  # bytt ut med info fra gb
    dofs = np.empty(gb.size(), dtype=int)
    for _, d in gb:
        dofs[d['node_number']] = d['dof']
    dofs = np.r_[0, np.cumsum(dofs)]

    to_be_eliminated = np.zeros(original_ndof, dtype=bool)

    for g, d in gb:
        i = d['node_number']
        if g.dim == dim:
            to_be_eliminated[slice(dofs[i], dofs[i + 1])] = \
                np.ones(d['dof'], dtype=bool)

    return to_be_eliminated


def eliminate_dofs(A, rhs, to_be_eliminated, inverter=sps.linalg.inv):
    """
    Splits the system matrix A into four blocks according to which dofs
    are to be eliminated (the "slaves"). The right hand side is split 
    into two parts. For the computation of the blocks, the diagonal 
    part corresponding to the slaves has to be inverted, hence the option
    to choose inverter. This system will usually be quite small.

    Input:
    A, rhs: original matrix and right hand side of the problem to be
        solved.
        to_be_eliminated: boolean mask specifying which degrees of freedom 
        should be eliminated from the system.

    Returns:
        A_reduced (scipy.sparse.csr_matrix): The system matrix for the reduced system, i.e.,
            corresponding to the master dofs only.
        rhs_reduced: The right hand side for the reduced system.
        Condensation_matrix: The matrix used for back-computation of the
            unknowns of the slaves once the reduced system has been solved.
        to_be_kept: Indices of the masters.

    """
    to_be_kept = np.invert(to_be_eliminated)
    # Get indexes of the masters:
    to_be_kept = np.nonzero(to_be_kept)[0]
    # and slaves:
    to_be_eliminated = np.nonzero(to_be_eliminated)[0]

    # Masters and slaves:
    A_mm = A[to_be_kept, :]
    A_mm = A_mm[:, to_be_kept]

    A_ms = A[:, to_be_eliminated]
    A_ms = A_ms[to_be_kept, :]

    A_sm = A[:, to_be_kept]
    A_sm = A_sm[to_be_eliminated, :]

    A_ss = A[:, to_be_eliminated]
    A_ss = A_ss[to_be_eliminated, :]

    A_ss_inv = inverter(A_ss)
    A_ms_A_ss_inv = A_ms * A_ss_inv

    # Needed for broadcasting
    if A_ss.size == 1:
        A_ms_A_ss_inv = A_ms_A_ss_inv[:, np.newaxis]

    sparse_product = sps.csr_matrix(A_ms_A_ss_inv * A_sm)

    A_reduced = A_mm - sparse_product
    rhs_reduced = rhs[to_be_kept][:, np.newaxis] - \
        A_ms_A_ss_inv * rhs[to_be_eliminated, np.newaxis]
    Condensation_matrix = sps.csr_matrix(- A_ss_inv * A_sm)

    return A_reduced, rhs_reduced, Condensation_matrix, to_be_kept
