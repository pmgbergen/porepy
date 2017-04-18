# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 09:04:16 2016

@author: eke001
"""
from __future__ import division
import numpy as np
import scipy.sparse as sps
import numba
import sys

from utils import matrix_compression
from utils import mcolon


class SubcellTopology(object):
    """
    Class to represent data of subcell topology (interaction regions) for
    mpsa/mpfa.

    Attributes:
        g - the grid

        Subcell topology, all cells seen apart:
        nno - node numbers
        fno - face numbers
        cno - cell numbers
        subfno - subface numbers. Has exactly two duplicates for internal faces
        subhfno - subface numbers. No duplicates
        num_cno - cno.max() + 1
        num_subfno - subfno.max() + 1

        Subcell topology, after cells sharing a face have been joined. Use
        terminology _unique, since subfno is unique after
        nno_unique - node numbers after pairing sub-faces
        fno_unique - face numbers after pairing sub-faces
        cno_unique - cell numbers after pairing sub-faces
        subfno_unique - subface numbers  after pairing sub-faces
        unique_subfno - index of those elements in subfno that are included
            in subfno_unique, that is subfno_unique = subfno[unique_subfno],
            and similarly cno_unique = cno[subfno_unique] etc.
        num_subfno_unique = subfno_unique.max() + 1

    """

    def __init__(self, g):
        """
        Constructor for subcell topology

        Parameters
        ----------
        g grid
        """
        self.g = g

        # Indices of neighboring faces and cells. The indices are sorted to
        # simplify later treatment
        g.cell_faces.sort_indices()
        face_ind, cell_ind = g.cell_faces.nonzero()

        # Number of faces per node
        num_face_nodes = np.diff(g.face_nodes.indptr)

        # Duplicate cell and face indices, so that they can be matched with
        # the nodes
        cells_duplicated = matrix_compression.rldecode(
            cell_ind, num_face_nodes[face_ind])
        faces_duplicated = matrix_compression.rldecode(
            face_ind, num_face_nodes[face_ind])
        M = sps.coo_matrix((np.ones(face_ind.size),
                            (face_ind, np.arange(face_ind.size))),
                           shape=(face_ind.max() + 1, face_ind.size))
        nodes_duplicated = g.face_nodes * M
        nodes_duplicated = nodes_duplicated.indices

        face_nodes_indptr = g.face_nodes.indptr
        face_nodes_indices = g.face_nodes.indices
        face_nodes_data = np.arange(face_nodes_indices.size) + 1
        sub_face_mat = sps.csc_matrix((face_nodes_data, face_nodes_indices,
                                       face_nodes_indptr))
        sub_faces = sub_face_mat * M
        sub_faces = sub_faces.data - 1

        # Sort data
        idx = np.lexsort((sub_faces, faces_duplicated, nodes_duplicated,
                          cells_duplicated))
        self.nno = nodes_duplicated[idx]
        self.cno = cells_duplicated[idx]
        self.fno = faces_duplicated[idx]
        self.subfno = sub_faces[idx].astype(int)
        self.subhfno = np.arange(idx.size, dtype='>i4')
        self.num_subfno = self.subfno.max() + 1
        self.num_cno = self.cno.max() + 1

        # Make subface indices unique, that is, pair the indices from the two
        # adjacent cells
        _, unique_subfno = np.unique(self.subfno, return_index=True)

        # Reduce topology to one field per subface
        self.nno_unique = self.nno[unique_subfno]
        self.fno_unique = self.fno[unique_subfno]
        self.cno_unique = self.cno[unique_subfno]
        self.subfno_unique = self.subfno[unique_subfno]
        self.num_subfno_unique = self.subfno_unique.max() + 1
        self.unique_subfno = unique_subfno

    def pair_over_subfaces(self, other):
        """
        Transfer quantities from a cell-face base (cells sharing a face have
        their own expressions) to a face-based. The quantities should live
        on sub-faces (not faces)

        The normal vector is honored, so that the here and there side get
        different signs when paired up.

        The method is intended used for combining forces, fluxes,
        displacements and pressures, as used in MPSA / MPFA.

        Parameters
        ----------
        other: sps.matrix, size (self.subhfno.size x something)

        Returns
        -------
        sps.matrix, size (self.subfno_unique.size x something)
        """

        sgn = self.g.cell_faces[self.fno, self.cno].A
        pair_over_subfaces = sps.coo_matrix((sgn[0], (self.subfno,
                                                      self.subhfno)))
        return pair_over_subfaces * other

    def pair_over_subfaces_nd(self, other):
        """ nd-version of pair_over_subfaces, see above. """
        nd = self.g.dim
        # For force balance, displacements and stresses on the two sides of the
        # matrices must be paired
        # Operator to create the pairing
        sgn = self.g.cell_faces[self.fno, self.cno].A
        pair_over_subfaces = sps.coo_matrix((sgn[0], (self.subfno,
                                                      self.subhfno)))
        # vector version, to be used on stresses
        pair_over_subfaces_nd = sps.kron(sps.eye(nd), pair_over_subfaces)
        return pair_over_subfaces_nd * other


def compute_dist_face_cell(g, subcell_topology, eta):
    """
    Compute vectors from cell centers continuity points on each sub-face.

    The location of the continuity point is given by

        x_cp = (1-eta) * x_facecenter + eta * x_vertex

    On the boundary, eta is set to zero, thus the continuity point is at the
    face center

    Parameters
    ----------
    g: Grid
    subcell_topology: Of class subcell topology in this module
    eta: [0,1), eta = 0 gives cont. pt. at face midpoint, eta = 1 means at
    the vertex

    Returns
    -------
    sps.csr() matrix representation of vectors. Size g.nf x (g.nc * g.nd)
    """
    _, blocksz = matrix_compression.rlencode(np.vstack((
        subcell_topology.cno, subcell_topology.nno)))
    dims = g.dim

    rows, cols = np.meshgrid(subcell_topology.subhfno, np.arange(dims))
    cols += matrix_compression.rldecode(np.cumsum(blocksz)-blocksz[0], blocksz)

    eta_vec = eta*np.ones(subcell_topology.fno.size)
    # Set eta values to zero at the boundary
    bnd = np.argwhere(np.abs(g.cell_faces).sum(axis=1).A.squeeze()
                            == 1).squeeze()
    eta_vec[bnd] = 0
    cp = g.face_centers[:, subcell_topology.fno] \
        + eta_vec * (g.nodes[:, subcell_topology.nno] -
                      g.face_centers[:, subcell_topology.fno])
    dist = cp - g.cell_centers[:, subcell_topology.cno]
    mat = sps.coo_matrix((dist.ravel(), (rows.ravel(), cols.ravel()))).tocsr()
    return subcell_topology.pair_over_subfaces(mat)


# @profile
def invert_diagonal_blocks(mat, s, method='numba'):
    """
    Invert block diagonal matrix.

    Two implementations are available, either pure numpy, or a speedup using
    numba. The latter is default.

    Parameters
    ----------
    mat: sps.csr matrix to be inverted.
    s: block size. Must be int64 for the numba acceleration to work
    method: Choice of method. Either numba (default), cython or 'python'. 
        If another option is passed, numba is used.

    Returns
    -------
    imat: Inverse matrix
    """

    def invert_diagonal_blocks_python(a, sz):
        """
        Invert block diagonal matrix using pure python code.

        The implementation is slow for large matrices, consider to use the
        numba-accelerated method invert_invert_diagagonal_blocks_numba instead

        Parameters
        ----------
        A sps.crs-matrix, to be inverted
        sz - size of the individual blocks

        Returns
        -------
        inv_a inverse matrix
        """
        v = np.zeros(np.sum(np.square(sz)))
        p1 = 0
        p2 = 0
        for b in range(sz.size):
            n = sz[b]
            n2 = n * n
            i = p1 + np.arange(n+1)
            # Picking out the sub-matrices here takes a lot of time.
            v[p2 + np.arange(n2)] = np.linalg.inv(a[i[0]:i[-1], i[0]:i[-1]].A)
            p1 = p1 + n
            p2 = p2 + n2
        return v

    def invert_diagonal_blocks_cython(a, size):
        import fvdiscr.cythoninvert
        a.sorted_indices()
        ptr = a.indptr
        indices = a.indices
        dat = a.data

        v = fvdiscr.cythoninvert.inv_python(ptr, indices, dat, size)
        return v

    def invert_diagonal_blocks_numba(a, size):
        """
        Invert block diagonal matrix by invoking numba acceleration of a simple
        for-loop based algorithm.

        This approach should be more efficient than the related method
        invert_diagonal_blocks_python for larger problems.

        Parameters
        ----------
        a : sps.csr matrix
        size : Size of individual blocks

        Returns
        -------
        ia: inverse of a
        """

        # Sort matrix storage before pulling indices and data
        a.sorted_indices()
        ptr = a.indptr
        indices = a.indices
        dat = a.data

        # Just in time compilation
        @numba.jit("f8[:](i4[:],i4[:],f8[:],i8[:])", nopython=True,
                   nogil=False)
        def inv_python(indptr, ind, data, sz):
            """
            Invert block matrices by explicitly forming local matrices. The code
            in itself is not efficient, but it is hopefully well suited for
            speeding up with numba.

            It may be possible to restruct the code to further help numba,
            this has not been investigated.

            The computation can easily be parallelized, consider this later.
            """

            # Index of where the rows start for each block.
            # block_row_starts_ind = np.hstack((np.array([0]),
            #                                   np.cumsum(sz[:-1])))
            block_row_starts_ind = np.zeros(sz.size, dtype=np.int32)
            block_row_starts_ind[1:] = np.cumsum(sz[:-1])

            # Number of columns per row. Will change from one column to the
            # next
            num_cols_per_row = indptr[1:] - indptr[0:-1]
            # Index to where the columns start for each row (NOT blocks)
            # row_cols_start_ind = np.hstack((np.zeros(1),
            #                                 np.cumsum(num_cols_per_row)))
            row_cols_start_ind = np.zeros(num_cols_per_row.size + 1,
                                          dtype=np.int32)
            row_cols_start_ind[1:] = np.cumsum(num_cols_per_row)

            # Index to where the (full) data starts. Needed, since the
            # inverse matrix will generally be full
            # full_block_starts_ind = np.hstack((np.array([0]),
            #                                    np.cumsum(np.square(sz))))
            full_block_starts_ind = np.zeros(sz.size + 1, dtype=np.int32)
            full_block_starts_ind[1:] = np.cumsum(np.square(sz))
            # Structure to store the solution
            inv_vals = np.zeros(np.sum(np.square(sz)))

            # Loop over all blocks
            for iter1 in range(sz.size):
                n = sz[iter1]
                loc_mat = np.zeros((n, n))
                # Fill in non-zero elements in local matrix
                for iter2 in range(n):  # Local rows
                    global_row = block_row_starts_ind[iter1] + iter2
                    data_counter = row_cols_start_ind[global_row]

                    # Loop over local columns. Getting the number of columns
                    #  for each row is a bit involved
                    for iter3 in range(num_cols_per_row[iter2 +
                            block_row_starts_ind[iter1]]):
                        loc_col = ind[data_counter] \
                                  - block_row_starts_ind[iter1]
                        loc_mat[iter2, loc_col] = data[data_counter]
                        data_counter += 1

                # Compute inverse. np.linalg.inv is supported by numba (May
                # 2016), it is not clear if this is the best option. To be
                # revised
                inv_mat = np.ravel(np.linalg.inv(loc_mat))
              
                loc_ind = np.arange(full_block_starts_ind[iter1],
                                    full_block_starts_ind[iter1 + 1])
                inv_vals[loc_ind] = inv_mat
                # Update fields
            return inv_vals

        v = inv_python(ptr, indices, dat, size)
        return v

    if method == 'python':
        inv_vals = invert_diagonal_blocks_python(mat, s)
    elif method == 'cython' and (sys.platform == 'linux' 
                                 or sys.platform == 'linux2'):
        inv_vals = invert_diagonal_blocks_cython(mat, s)
    else:
        inv_vals = invert_diagonal_blocks_numba(mat, s)

    ia = block_diag_matrix(inv_vals, s)
    return ia


def block_diag_matrix(vals, sz):
    """
    Construct block diagonal matrix based on matrix elements and block sizes.

    Parameters
    ----------
    vals: matrix values
    sz: size of matrix blocks

    Returns
    -------
    sps.csr matrix
    """
    row, col = block_diag_index(sz)
    # This line recovers starting indices of the rows.
    indptr = np.hstack((np.zeros(1),
                        np.cumsum(matrix_compression\
                                  .rldecode(sz, sz)))).astype('int32')
    return sps.csr_matrix((vals, row, indptr))


def block_diag_index(m, n=None):
    """
    Get row and column indices for block diagonal matrix

    This is intended as the equivalent of the corresponding method in MRST.

    Examples:
    >>> m = np.array([2, 3])
    >>> n = np.array([1, 2])
    >>> i, j = block_diag_index(m, n)
    >>> i, j
    (array([0, 1, 2, 3, 4, 2, 3, 4]), array([0, 0, 1, 1, 1, 2, 2, 2]))
    >>> a = np.array([1, 3])
    >>> i, j = block_diag_index(a)
    >>> i, j
    (array([0, 1, 2, 3, 1, 2, 3, 1, 2, 3]), array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3]))

    Parameters:
        m - ndarray, dimension 1
        n - ndarray, dimension 1, defaults to m

    """
    if n is None:
        n = m

    start = np.hstack((np.zeros(1, dtype='int'), m))
    pos = np.cumsum(start)
    p1 = pos[0:-1]
    p2 = pos[1:]-1
    p1_full = matrix_compression.rldecode(p1, n)
    p2_full = matrix_compression.rldecode(p2, n)

    i = mcolon.mcolon(p1_full, p2_full)
    sumn = np.arange(np.sum(n))
    m_n_full = matrix_compression.rldecode(m, n)
    j = matrix_compression.rldecode(sumn, m_n_full)
    return i, j


def scalar_divergence(g):
    """
    Get divergence operator for a grid.

    The operator is easily accessible from the grid itself, so we keep it
    here for completeness.

    See also vector_divergence(g)

    Parameters
    ----------
    g grid

    Returns
    -------
    divergence operator
    """
    return g.cell_faces.T


def vector_divergence(g):
    """
    Get vector divergence operator for a grid g

    It is assumed that the first column corresponds to the x-equation of face
    0, second column is y-equation etc. (and so on in nd>2). The next column is
    then the x-equation for face 1. Correspondingly, the first row
    represents x-component in first cell etc.

    Parameters
    ----------
    g grid

    Returns
    -------
    vector_div (sparse csr matrix), dimensions: nd * (num_cells, num_faces)
    """
    # Scalar divergence
    scalar_div = g.cell_faces

    # Vector extension, convert to coo-format to avoid odd errors when one
    # grid dimension is 1 (this may return a bsr matrix)
    # The order of arguments to sps.kron is important.
    block_div = sps.kron(scalar_div, sps.eye(g.dim)).tocsr()

    return block_div.transpose()


class ExcludeBoundaries(object):

    def __init__(self, subcell_topology, bound, nd):

        """
        Define mappings to exclude boundary faces with dirichlet and neumann
        conditions

        Parameters
        ----------
        subcell_topology
        bound

        Returns
        -------
        exclude_neumann: Matrix, mapping from all faces to those having flux
                         continuity
        exclude_dirichlet: Matrix, mapping from all faces to those having pressure
                           continuity
        """
        self.nd = nd

        # Short hand notation
        fno = subcell_topology.fno_unique
        num_subfno = subcell_topology.num_subfno_unique

        # Define mappings to exclude boundary values
        col_neu = np.argwhere([not it for it in bound.is_neu[fno]])
        row_neu = np.arange(col_neu.size)
        self.exclude_neu = sps.coo_matrix((np.ones(row_neu.size),
                                           (row_neu, col_neu.ravel(0))),
                                          shape=(row_neu.size,
                                                num_subfno)).tocsr()
        col_dir = np.argwhere([not it for it in bound.is_dir[fno]])
        row_dir = np.arange(col_dir.size)
        self.exclude_dir = sps.coo_matrix((np.ones(row_dir.size),
                                            (row_dir, col_dir.ravel(0))),
                                           shape=(row_dir.size,
                                                  num_subfno)).tocsr()

    def exclude_dirichlet(self, other):
        return self.exclude_dir * other

    def exclude_neumann(self, other):
        return self.exclude_neu * other

    def exclude_neumann_nd(self, other):
        exclude_neumann_nd = sps.kron(sps.eye(self.nd), self.exclude_neu)
        return exclude_neumann_nd * other

    def exclude_dirichlet_nd(self, other):
        exclude_dirichlet_nd = sps.kron(sps.eye(self.nd),
                                        self.exclude_dir)
        return exclude_dirichlet_nd * other

