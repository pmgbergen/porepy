""" Utility functions for the tests.

Access: from test import test_utils.
"""

import numpy as np
import scipy.sparse as sps
import os

import porepy as pp


def permute_matrix_vector(A, rhs, block_dof, full_dof, grids, variables):
    """ Permute the matrix and rhs from assembler order to a specified order.

    Args:
        A: global solution matrix as returned by Assembler.assemble_matrix_rhs.
        rhs: global rhs vector as returned by Assembler.assemble_matrix_rhs.
        block_dof: Map coupling a (grid, variable) pair to an block index of A, as
            returned by Assembler.assemble_matrix_rhs.
        full_dof: Number of DOFs for each pair in block_dof, as returned by
            Assembler.assemble_matrix_rhs.

    Returns:
        sps.bmat(A.size): Permuted matrix.
        np.ndarray(b.size): Permuted rhs vector.
    """
    sz = len(block_dof)
    mat = np.empty((sz, sz), dtype=np.object)
    b = np.empty(sz, dtype=np.object)
    dof = np.empty(sz, dtype=np.object)
    # Initialize dof vector
    dof[0] = np.arange(full_dof[0])
    for i in range(1, sz):
        dof[i] = dof[i - 1][-1] + 1 + np.arange(full_dof[i])

    for row in range(sz):
        # Assembler index 0
        i = block_dof[(grids[row], variables[row])]
        b[row] = rhs[dof[i]]
        for col in range(sz):
            # Assembler index 1
            j = block_dof[(grids[col], variables[col])]
            # Put the A block indexed by i and j in mat of running indexes row and col
            mat[row, col] = A[dof[i]][:, dof[j]]

    return sps.bmat(mat, format="csr"), np.concatenate(tuple(b))


def setup_flow_assembler(gb, method, data_key=None, coupler=None):
    """ Setup a standard assembler for the flow problem for a given grid bucket.

    The assembler will be set up with primary variable name 'pressure' on the
    GridBucket nodes, and mortar_flux for the mortar variables.

    Parameters:
        gb: GridBucket.
        method (EllipticDiscretization).
        data_key (str, optional): Keyword used to identify data dictionary for
            node and edge discretization.
        Coupler (EllipticInterfaceLaw): Defaults to RobinCoulping.

    Returns:
        Assembler, ready to discretize and assemble problem.

    """

    if data_key is None:
        data_key = "flow"
    if coupler is None:
        coupler = pp.RobinCoupling(data_key, method)

    if isinstance(method, pp.MVEM) or isinstance(method, pp.RT0):
        mixed_form = True
    else:
        mixed_form = False

    for g, d in gb:
        if mixed_form:
            d[pp.PRIMARY_VARIABLES] = {"pressure": {"cells": 1, "faces": 1}}
        else:
            d[pp.PRIMARY_VARIABLES] = {"pressure": {"cells": 1}}
        d[pp.DISCRETIZATION] = {"pressure": {"diffusive": method}}
    for e, d in gb.edges():
        g1, g2 = gb.nodes_of_edge(e)
        d[pp.PRIMARY_VARIABLES] = {"mortar_flux": {"cells": 1}}
        d[pp.COUPLING_DISCRETIZATION] = {
            "lambda": {
                g1: ("pressure", "diffusive"),
                g2: ("pressure", "diffusive"),
                e: ("mortar_flux", coupler),
            }
        }
        d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

    assembler = pp.Assembler(gb)
    return assembler


def solve_and_distribute_pressure(gb, assembler):
    """ Given an assembler, assemble and solve the pressure equation, and distribute
    the result.

    Parameters:
        GridBucket: Of problem to be solved
        assembler (Assembler):
    """
    assembler.discretize()
    A, b = assembler.assemble_matrix_rhs()
    p = np.linalg.solve(A.A, b)
    assembler.distribute_variable(p)


def compare_arrays(a, b, tol=1e-4, sort=True):
    """ Compare two arrays and check that they are equal up to a column permutation.

    Typical usage is to compare coordinate arrays.

    Parameters:
        a, b (np.array): Arrays to be compared. W
        tol (double, optional): Tolerance used in comparison.
        sort (boolean, defaults to True): Sort arrays columnwise before comparing

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


def delete_file(file_name):
    """ Delete a file if it exist. Cleanup after tests.
    """
    if os.path.exists(file_name):
        os.remove(file_name)
