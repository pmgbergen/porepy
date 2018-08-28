# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 09:04:16 2016

@author: eke001
"""
from __future__ import division
import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.utils import matrix_compression, mcolon
from porepy.grids.grid_bucket import GridBucket


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
            cell_ind, num_face_nodes[face_ind]
        )
        faces_duplicated = matrix_compression.rldecode(
            face_ind, num_face_nodes[face_ind]
        )
        M = sps.coo_matrix(
            (np.ones(face_ind.size), (face_ind, np.arange(face_ind.size))),
            shape=(face_ind.max() + 1, face_ind.size),
        )
        nodes_duplicated = g.face_nodes * M
        nodes_duplicated = nodes_duplicated.indices

        face_nodes_indptr = g.face_nodes.indptr
        face_nodes_indices = g.face_nodes.indices
        face_nodes_data = np.arange(face_nodes_indices.size) + 1
        sub_face_mat = sps.csc_matrix(
            (face_nodes_data, face_nodes_indices, face_nodes_indptr)
        )
        sub_faces = sub_face_mat * M
        sub_faces = sub_faces.data - 1

        # Sort data
        idx = np.lexsort(
            (sub_faces, faces_duplicated, nodes_duplicated, cells_duplicated)
        )
        self.nno = nodes_duplicated[idx]
        self.cno = cells_duplicated[idx]
        self.fno = faces_duplicated[idx]
        self.subfno = sub_faces[idx].astype(int)
        self.subhfno = np.arange(idx.size, dtype=">i4")
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

    def __repr__(self):
        s = "Subcell topology with:\n"
        s += str(self.num_cno) + " cells\n"
        s += str(self.g.num_nodes) + " nodes\n"
        s += str(self.g.num_faces) + " faces\n"
        s += str(self.num_subfno_unique) + " unique subfaces\n"
        s += str(self.fno.size) + " subfaces before pairing face neighbors\n"
        return s

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
        pair_over_subfaces = sps.coo_matrix((sgn[0], (self.subfno, self.subhfno)))
        return pair_over_subfaces * other

    def pair_over_subfaces_nd(self, other):
        """ nd-version of pair_over_subfaces, see above. """
        nd = self.g.dim
        # For force balance, displacements and stresses on the two sides of the
        # matrices must be paired
        # Operator to create the pairing
        sgn = self.g.cell_faces[self.fno, self.cno].A
        pair_over_subfaces = sps.coo_matrix((sgn[0], (self.subfno, self.subhfno)))
        # vector version, to be used on stresses
        pair_over_subfaces_nd = sps.kron(sps.eye(nd), pair_over_subfaces)
        return pair_over_subfaces_nd * other


# ------------------------ End of class SubcellTopology ----------------------


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
    _, blocksz = matrix_compression.rlencode(
        np.vstack((subcell_topology.cno, subcell_topology.nno))
    )
    dims = g.dim

    _, cols = np.meshgrid(subcell_topology.subhfno, np.arange(dims))
    cols += matrix_compression.rldecode(np.cumsum(blocksz) - blocksz[0], blocksz)
    eta_vec = eta * np.ones(subcell_topology.fno.size)
    # Set eta values to zero at the boundary
    bnd = np.in1d(subcell_topology.fno, g.get_all_boundary_faces())
    eta_vec[bnd] = 0
    cp = g.face_centers[:, subcell_topology.fno] + eta_vec * (
        g.nodes[:, subcell_topology.nno] - g.face_centers[:, subcell_topology.fno]
    )
    dist = cp - g.cell_centers[:, subcell_topology.cno]

    ind_ptr = np.hstack((np.arange(0, cols.size, dims), cols.size))
    mat = sps.csr_matrix((dist.ravel("F"), cols.ravel("F"), ind_ptr))
    return subcell_topology.pair_over_subfaces(mat)


def determine_eta(g):
    """ Set default value for the location of continuity point eta in MPFA and
    MSPA.

    The function is intended to give a best estimate of eta in cases where the
    user has not specified a value.

    Parameters:
        g: Grid for discretization

    Returns:
        double. 1/3 if the grid in known to consist of simplicies (it is one of
           TriangleGrid, TetrahedralGrid, or their structured versions). 0 if
           not.
    """

    if "StructuredTriangleGrid" in g.name:
        return 1 / 3
    elif "TriangleGrid" in g.name:
        return 1 / 3
    elif "StructuredTetrahedralGrid" in g.name:
        return 1 / 3
    elif "TetrahedralGrid" in g.name:
        return 1 / 3
    else:
        return 0


# ------------- Methods related to block inversion ----------------------------

# @profile


def invert_diagonal_blocks(mat, s, method=None):
    """
    Invert block diagonal matrix.

    Three implementations are available, either pure numpy, or a speedup using
    numba or cython. If none is specified, the function will try to use numba,
    then cython. The python option will only be invoked if explicitly asked
    for; it will be very slow for general problems.

    Parameters
    ----------
    mat: sps.csr matrix to be inverted.
    s: block size. Must be int64 for the numba acceleration to work
    method: Choice of method. Either numba (default), cython or 'python'.
        Defaults to None, in which case first numba, then cython is tried.

    Returns
    -------
    imat: Inverse matrix

    Raises
    -------
    ImportError: If numba or cython implementation is invoked without numba or
        cython being available on the system.

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
            i = p1 + np.arange(n + 1)
            # Picking out the sub-matrices here takes a lot of time.
            v[p2 + np.arange(n2)] = np.linalg.inv(
                a[i[0] : i[-1], i[0] : i[-1]].A
            ).ravel()
            p1 = p1 + n
            p2 = p2 + n2
        return v

    def invert_diagonal_blocks_cython(a, size):
        """ Invert block diagonal matrix using code wrapped with cython.
        """
        try:
            import porepy.numerics.fv.cythoninvert as cythoninvert
        except:
            ImportError(
                "Compiled Cython module not available. Is cython\
            installed?"
            )

        a.sorted_indices()
        ptr = a.indptr
        indices = a.indices
        dat = a.data

        v = cythoninvert.inv_python(ptr, indices, dat, size)
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
        try:
            import numba
        except:
            raise ImportError("Numba not available on the system")

        # Sort matrix storage before pulling indices and data
        a.sorted_indices()
        ptr = a.indptr
        indices = a.indices
        dat = a.data

        # Just in time compilation
        @numba.jit("f8[:](i4[:],i4[:],f8[:],i8[:])", nopython=True, nogil=False)
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
            row_cols_start_ind = np.zeros(num_cols_per_row.size + 1, dtype=np.int32)
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
                    for _ in range(
                        num_cols_per_row[iter2 + block_row_starts_ind[iter1]]
                    ):
                        loc_col = ind[data_counter] - block_row_starts_ind[iter1]
                        loc_mat[iter2, loc_col] = data[data_counter]
                        data_counter += 1

                # Compute inverse. np.linalg.inv is supported by numba (May
                # 2016), it is not clear if this is the best option. To be
                # revised
                inv_mat = np.ravel(np.linalg.inv(loc_mat))

                loc_ind = np.arange(
                    full_block_starts_ind[iter1], full_block_starts_ind[iter1 + 1]
                )
                inv_vals[loc_ind] = inv_mat
                # Update fields
            return inv_vals

        v = inv_python(ptr, indices, dat, size)
        return v

    # Variable to check if we have tried and failed with numba
    try_cython = False
    if method == "numba" or method is None:
        try:
            inv_vals = invert_diagonal_blocks_numba(mat, s)
        except:
            # This went wrong, fall back on cython
            try_cython = True
    # Variable to check if we should fall back on python
    if method == "cython" or try_cython:
        try:
            inv_vals = invert_diagonal_blocks_cython(mat, s)
        except ImportError as e:
            raise e
    elif method == "python":
        inv_vals = invert_diagonal_blocks_python(mat, s)

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
    row, _ = block_diag_index(sz)
    # This line recovers starting indices of the rows.
    indptr = np.hstack(
        (np.zeros(1), np.cumsum(matrix_compression.rldecode(sz, sz)))
    ).astype("int32")
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

    start = np.hstack((np.zeros(1, dtype="int"), m))
    pos = np.cumsum(start)
    p1 = pos[0:-1]
    p2 = pos[1:] - 1
    p1_full = matrix_compression.rldecode(p1, n)
    p2_full = matrix_compression.rldecode(p2, n)

    i = mcolon.mcolon(p1_full, p2_full + 1)
    sumn = np.arange(np.sum(n))
    m_n_full = matrix_compression.rldecode(m, n)
    j = matrix_compression.rldecode(sumn, m_n_full)
    return i, j


# ------------------- End of methods related to block inversion ---------------


def expand_indices_nd(ind, nd, direction=1):
    """
    Expand indices from scalar to vector form.

    Examples:
    >>> i = np.array([0, 1, 3])
    >>> __expand_indices_nd(i, 2)
    (array([0, 1, 2, 3, 6, 7]))

    >>> __expand_indices_nd(i, 3, 0)
    (array([0, 3, 9, 1, 4, 10, 2, 5, 11])

    Parameters
    ----------
    ind
    nd
    direction

    Returns
    -------

    """
    dim_inds = np.arange(nd)
    dim_inds = dim_inds[:, np.newaxis]  # Prepare for broadcasting
    new_ind = nd * ind + dim_inds
    new_ind = new_ind.ravel(direction)
    return new_ind


def map_hf_2_f(fno, subfno, nd):
    """
    Create mapping from half-faces to faces for vector problems.

    Parameters
    ----------
    fno face numbering in sub-cell topology based on unique subfno
    subfno sub-face numbering
    nd dimension

    Returns
    -------

    """

    hfi = expand_indices_nd(subfno, nd)
    hf = expand_indices_nd(fno, nd)
    hf2f = sps.coo_matrix(
        (np.ones(hf.size), (hf, hfi)), shape=(hf.max() + 1, hfi.max() + 1)
    ).tocsr()
    return hf2f


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


def zero_out_sparse_rows(A, rows, diag=None):
    """
    zeros out given rows from sparse csr matrix. Optionally also set values on
    the diagonal.

    Parameters:
        A: Sparse matrix
        rows (np.ndarray of int): Indices of rows to be eliminated.
        diag (np.ndarray, double, optional): Values to be set to the diagonal
            on the eliminated rows.

    """

    # If matrix is not csr, it will be converted to csr, then the rows will be
    # zeroed, and the matrix converted back.
    flag = False
    if not A.getformat() == "csr":
        mat_format = A.getformat()
        A = A.tocsr()
        flag = True

    ip = A.indptr
    row_indices = mcolon.mcolon(ip[rows], ip[rows + 1])
    A.data[row_indices] = 0
    if diag is not None:
        # now we set the diagonal
        diag_vals = np.zeros(A.shape[1])
        diag_vals[rows] = diag
        A += sps.dia_matrix((diag_vals, 0), shape=A.shape)

    if flag:
        # Convert matrix back
        A = A.astype(mat_format)

    return A


# -----------------------------------------------------------------------------


class ExcludeBoundaries(object):
    """ Wrapper class to store mapping for exclusion of equations that are
    redundant due to the presence of boundary conditions.

    The systems being set up in mpfa (and mpsa) describe continuity of flux and
    potential (respectively stress and displacement) on all sub-faces. For
    boundary faces, one of the two should be excluded (e.g. for a Dirichlet
    boundary condition, there is no concept of continuity of flux/stresses).
    The class contains mappings to eliminate the necessary fields.

    """

    def __init__(self, subcell_topology, bound, nd):
        """
        Define mappings to exclude boundary faces/components with dirichlet and neumann
        conditions

        Parameters
        ----------
        subcell_topology
        bound

        Returns
        -------
        exclude_neumann: Matrix, mapping from all faces/components to those having flux
                         continuity
        exclude_dirichlet: Matrix, mapping from all faces/components to those having pressure
                           continuity
        """
        self.nd = nd
        self.bc_type = bound.bc_type

        # Short hand notation
        fno = subcell_topology.fno_unique
        num_subfno = subcell_topology.num_subfno_unique

        # Define mappings to exclude boundary values

        if self.bc_type == "scalar":

            col_neu = np.argwhere([not it for it in bound.is_neu[fno]])
            row_neu = np.arange(col_neu.size)
            self.exclude_neu = sps.coo_matrix(
                (np.ones(row_neu.size), (row_neu, col_neu.ravel("C"))),
                shape=(row_neu.size, num_subfno),
            ).tocsr()

            col_dir = np.argwhere([not it for it in bound.is_dir[fno]])
            row_dir = np.arange(col_dir.size)
            self.exclude_dir = sps.coo_matrix(
                (np.ones(row_dir.size), (row_dir, col_dir.ravel("C"))),
                shape=(row_dir.size, num_subfno),
            ).tocsr()

        elif self.bc_type == "vectorial":

            # Neumann
            col_neu_x = np.argwhere([not it for it in bound.is_neu[0, fno]])
            row_neu_x = np.arange(col_neu_x.size)
            self.exclude_neu_x = sps.coo_matrix(
                (np.ones(row_neu_x.size), (row_neu_x, col_neu_x.ravel("C"))),
                shape=(row_neu_x.size, num_subfno),
            ).tocsr()

            col_neu_y = np.argwhere([not it for it in bound.is_neu[1, fno]])
            row_neu_y = np.arange(col_neu_y.size)
            self.exclude_neu_y = sps.coo_matrix(
                (np.ones(row_neu_y.size), (row_neu_y, col_neu_y.ravel("C"))),
                shape=(row_neu_y.size, num_subfno),
            ).tocsr()
            col_neu_y += num_subfno
            col_neu = np.append(col_neu_x, [col_neu_y])

            if nd == 3:
                col_neu_z = np.argwhere([not it for it in bound.is_neu[2, fno]])
                row_neu_z = np.arange(col_neu_z.size)
                self.exclude_neu_z = sps.coo_matrix(
                    (np.ones(row_neu_z.size), (row_neu_z, col_neu_z.ravel("C"))),
                    shape=(row_neu_z.size, num_subfno),
                ).tocsr()

                col_neu_z += 2 * num_subfno
                col_neu = np.append(col_neu, [col_neu_z])

            row_neu = np.arange(col_neu.size)
            self.exclude_neu_nd = sps.coo_matrix(
                (np.ones(row_neu.size), (row_neu, col_neu.ravel("C"))),
                shape=(row_neu.size, nd * num_subfno),
            ).tocsr()

            # Dirichlet, same procedure
            col_dir_x = np.argwhere([not it for it in bound.is_dir[0, fno]])
            row_dir_x = np.arange(col_dir_x.size)
            self.exclude_dir_x = sps.coo_matrix(
                (np.ones(row_dir_x.size), (row_dir_x, col_dir_x.ravel("C"))),
                shape=(row_dir_x.size, num_subfno),
            ).tocsr()

            col_dir_y = np.argwhere([not it for it in bound.is_dir[1, fno]])
            row_dir_y = np.arange(col_dir_y.size)
            self.exclude_dir_y = sps.coo_matrix(
                (np.ones(row_dir_y.size), (row_dir_y, col_dir_y.ravel("C"))),
                shape=(row_dir_y.size, num_subfno),
            ).tocsr()

            col_dir_y += num_subfno
            col_dir = np.append(col_dir_x, [col_dir_y])

            if nd == 3:
                col_dir_z = np.argwhere([not it for it in bound.is_dir[2, fno]])
                row_dir_z = np.arange(col_dir_z.size)
                self.exclude_dir_z = sps.coo_matrix(
                    (np.ones(row_dir_z.size), (row_dir_z, col_dir_z.ravel("C"))),
                    shape=(row_dir_z.size, num_subfno),
                ).tocsr()

                col_dir_z += 2 * num_subfno
                col_dir = np.append(col_dir, [col_dir_z])

            row_dir = np.arange(col_dir.size)
            self.exclude_dir_nd = sps.coo_matrix(
                (np.ones(row_dir.size), (row_dir, col_dir.ravel("C"))),
                shape=(row_dir.size, nd * num_subfno),
            ).tocsr()

    def exclude_dirichlet(self, other):
        """ Mapping to exclude faces/components with Dirichlet boundary conditions from
        local systems.

        Parameters:
            other (scipy.sparse matrix): Matrix of local equations for
                continuity of flux and pressure.

        Returns:
            scipy.sparse matrix, with rows corresponding to faces/components with
                Dirichlet conditions eliminated.

        """
        if self.bc_type == "scalar":
            exclude_dir = self.exclude_dir * other

        elif self.bc_type == "vectorial":
            exclude_dir = np.append(
                self.exclude_dir_x * other, [self.exclude_dir_y * other]
            )
            if self.nd == 3:
                exclude_dir = np.append(exclude_dir, [self.exclude_dir_z * other])

        return exclude_dir

    def exclude_dirichlet_x(self, other):

        """ Mapping to exclude components in the x-direction with Dirichlet boundary conditions from
        local systems.
        NOTE: Currently works for boundary faces aligned with the coordinate system.

        Parameters:
            other (scipy.sparse matrix): Matrix of local equations for
                continuity of flux and pressure.

        Returns:
            scipy.sparse matrix, with components in the x-direction corresponding to faces with
                Dirichlet conditions eliminated.

        """

        return self.exclude_dir_x * other

    def exclude_dirichlet_y(self, other):
        """ Mapping to exclude components in the y-direction with Dirichlet boundary conditions from
        local systems.
        NOTE: Currently works for boundary faces aligned with the coordinate system.

        Parameters:
            other (scipy.sparse matrix): Matrix of local equations for
                continuity of flux and pressure.

        Returns:
            scipy.sparse matrix, with components in the y-direction corresponding to faces with
                Dirichlet conditions eliminated.

        """

        return self.exclude_dir_y * other

    def exclude_dirichlet_z(self, other):
        """ Mapping to exclude components in the z-direction with Dirichlet boundary conditions from
        local systems.
        NOTE: Currently works for boundary faces aligned with the coordinate system.

        Parameters:
            other (scipy.sparse matrix): Matrix of local equations for
                continuity of flux and pressure.

        Returns:
            scipy.sparse matrix, with components in the z-direction corresponding to faces with
                Dirichlet conditions eliminated.

        """

        return self.exclude_dir_z * other

    def exclude_neumann(self, other):
        """ Mapping to exclude faces/components with Neumann boundary conditions from
        local systems.

        Parameters:
            other (scipy.sparse matrix): Matrix of local equations for
                continuity of flux and pressure.

        Returns:
            scipy.sparse matrix, with rows corresponding to faces/components with
                Neumann conditions eliminated.

        """
        if self.bc_type == "scalar":
            exclude_neu = self.exclude_neu * other

        elif self.bc_type == "vectorial":
            exclude_neu = np.append(
                self.exclude_neu_x * other, [self.exclude_neu_y * other]
            )
            if self.nd == 3:
                exclude_neu = np.append(exclude_neu, [self.exclude_neu_z * other])

        return exclude_neu

    def exclude_neumann_x(self, other):
        """ Mapping to exclude components in the x-direction with Neumann boundary conditions from
        local systems.
        NOTE: Currently works for boundary faces aligned with the coordinate system.

        Parameters:
            other (scipy.sparse matrix): Matrix of local equations for
                continuity of flux and pressure.

        Returns:
            scipy.sparse matrix, with components in the x-direction corresponding to faces with
                Neumann conditions eliminated.

        """

        return self.exclude_neu_x * other

    def exclude_neumann_y(self, other):
        """ Mapping to exclude components in the y-direction with Neumann boundary conditions from
        local systems.
        NOTE: Currently works for boundary faces aligned with the coordinate system.

        Parameters:
            other (scipy.sparse matrix): Matrix of local equations for
                continuity of flux and pressure.

        Returns:
            scipy.sparse matrix, with components in the y-direction corresponding to faces with
                Neumann conditions eliminated.

        """

        return self.exclude_neu_y * other

    def exclude_neumann_z(self, other):
        """ Mapping to exclude components in the z-direction with Neumann boundary conditions from
        local systems.
        NOTE: Currently works for boundary faces aligned with the coordinate system.

        Parameters:
            other (scipy.sparse matrix): Matrix of local equations for
                continuity of flux and pressure.

        Returns:
            scipy.sparse matrix, with components in the z-direction corresponding to faces with
                Neumann conditions eliminated.

        """

        return self.exclude_neu_z * other

    def exclude_dirichlet_nd(self, other):
        """ Exclusion of Dirichlet conditions for vector equations (elasticity).
        See above method without _nd suffix for description.

        """

        if self.bc_type == "scalar":
            exclude_dirichlet_nd = sps.kron(sps.eye(self.nd), self.exclude_dir)

        elif self.bc_type == "vectorial":
            exclude_dirichlet_nd = self.exclude_dir_nd

        return exclude_dirichlet_nd * other

    def exclude_neumann_nd(self, other):
        """ Exclusion of Neumann conditions for vector equations (elasticity).
        See above method without _nd suffix for description.

        """
        if self.bc_type == "scalar":
            exclude_neumann_nd = sps.kron(sps.eye(self.nd), self.exclude_neu)

        elif self.bc_type == "vectorial":
            exclude_neumann_nd = self.exclude_neu_nd

        return exclude_neumann_nd * other


# -----------------End of class ExcludeBoundaries-----------------------------


def cell_ind_for_partial_update(g, cells=None, faces=None, nodes=None):
    """ Obtain indices of cells and faces needed for a partial update of the
    discretization stencil.

    Implementation note: This function should really be split into three parts,
    one for each of the modes (cell, face, node).

    The subgrid can be specified in terms of cells, faces and nodes to be
    updated. The method will then define a sufficiently large subgrid to
    account for changes in the flux discretization. The idea is that cells are
    used to account for updates in material parameters (or geometry), faces
    when faces are split (fracture growth), while the parameter nodes is mainly
    aimed at a gradual build of the discretization of the entire grid (for
    memory conservation, see comments in mpfa.mpfa()). For more details, see
    the implementations and comments below.

    Cautionary note: The option to combine cells, faces and nodes in one go has
    not been tested. Problems may arise for grid configurations where separate
    regions are close to touching. This is however speculation at the time of
    writing.

    Parameters:
        g (core.grids.grid): grid to be discretized
        cells (np.array, int, optional): Index of cells on which to base the
            subgrid computation. Defaults to None.
        faces (np.array, int, optional): Index of faces on which to base the
            subgrid computation. Defaults to None.
        nodes (np.array, int, optional): Index of faces on which to base the
            subgrid computation. Defaults to None.

    Returns:
        np.array, int: Cell indexes of the subgrid. No guarantee that they form
            a connected grid.
        np.array, int: Indexes of faces to have their discretization updated.

    """

    # Faces that are active, and should have their discretization stencil
    # updated / returned.
    active_faces = np.zeros(g.num_faces, dtype=np.bool)

    # Index of cells to include in the subgrid.
    cell_ind = np.empty(0)

    if cells is not None:
        # To understand the update stencil for a cell-based update, consider
        # the Cartesian 2d configuration below.
        #
        #    _ s s s _
        #    s o o o s
        #    s o x o s
        #    s o o o s
        #    - s s s -
        #
        # The central cell (x) is to be updated. The support of MPFA basis
        # functions dictates that the stencil between the central cell and its
        # primary neighbors (o) must be updated, as must the stencil for the
        # sub-faces between o-cells that shares a vertex with x. Since the
        # flux information is stored face-wise (not sub-face), the whole o-o
        # faces must be recomputed, and this involves the secondary neighbors
        # of x (denoted s). This is most easily realized by defining an overlap
        # of 2. This will also involve some cells and nodes not needed;
        # specifically those marked by -. This requires quite a song and dance,
        # see below; but most of this is necessary to get hold of the active
        # faces anyhow.
        #
        # Note that a simpler option, with a somewhat higher computational cost,
        # would be to define
        #   cell_overlap = partition.overlap(g, cells, num_layers=2)
        # This would however include more cells (all marked - in the
        # illustration, and potentially significantly many more in 3d, in
        # particular for unstructured grids).

        cn = g.cell_nodes()

        # The active faces (to be updated; (o-x and o-o above) are those that
        # share at least one vertex with cells in ind.
        prim_cells = np.zeros(g.num_cells, dtype=np.bool)
        prim_cells[cells] = 1
        # Vertexes of the cells
        active_vertexes = np.zeros(g.num_nodes, dtype=np.bool)
        active_vertexes[np.squeeze(np.where(cn * prim_cells > 0))] = 1

        # Faces of the vertexes, these will be the active faces.
        active_face_ind = np.squeeze(
            np.where(g.face_nodes.transpose() * active_vertexes > 0)
        )
        active_faces[active_face_ind] = 1

        # Secondary vertexes, involved in at least one of the active faces,
        # that is, the faces to be updated. Corresponds to vertexes between o-o
        # above.
        active_vertexes[np.squeeze(np.where(g.face_nodes * active_faces > 0))] = 1

        # Finally, get hold of all cells that shares one of the secondary
        # vertexes.
        cells_overlap = np.squeeze(np.where((cn.transpose() * active_vertexes) > 0))
        # And we have our overlap!
        cell_ind = np.hstack((cell_ind, cells_overlap))

    if faces is not None:
        # The faces argument is intended used when the configuration of the
        # specified faces has changed, e.g. due to the introduction of an
        # external boundary. This requires the recomputation of all faces that
        # share nodes with the specified faces. Since data is not stored on
        # sub-faces. This further requires the inclusion of all cells that
        # share a node with a secondary face.
        #
        #      o o o
        #    o o x o o
        #    o o x o o
        #      o o o
        #
        # To illustrate for the Cartesian configuration above: The face
        # between the two x-cells are specified, and this requires the
        # inclusion of all o-cells.
        #

        cf = g.cell_faces
        # This avoids overwriting data in cell_faces.
        data = np.ones_like(cf.data)
        cf = sps.csc_matrix((data, cf.indices, cf.indptr))

        primary_faces = np.zeros(g.num_faces, dtype=np.bool)
        primary_faces[faces] = 1

        # The active faces are those sharing a vertex with the primary faces
        primary_vertex = np.zeros(g.num_nodes, dtype=np.bool)
        primary_vertex[np.squeeze(np.where((g.face_nodes * primary_faces) > 0))] = 1
        active_face_ind = np.squeeze(
            np.where((g.face_nodes.transpose() * primary_vertex) > 0)
        )
        active_faces[active_face_ind] = 1

        # Find vertexes of the active faces
        active_nodes = np.zeros(g.num_nodes, dtype=np.bool)
        active_nodes[np.squeeze(np.where((g.face_nodes * active_faces) > 0))] = 1

        active_cells = np.zeros(g.num_cells, dtype=np.bool)
        # Primary cells, those that have the faces as a boundary
        cells_overlap = np.squeeze(
            np.where((g.cell_nodes().transpose() * active_nodes) > 0)
        )
        cell_ind = np.hstack((cell_ind, cells_overlap))

    if nodes is not None:
        # Pick out all cells that have the specified nodes as a vertex.
        # The active faces will be those that have all their vertexes included
        # in nodes.
        cn = g.cell_nodes()
        # Introduce active nodes, and make the input nodes active
        # The data type of active_vertex is int (in contrast to similar cases
        # in other parts of this function), since we will use it to count the
        # number of active face_nodes below.
        active_vertexes = np.zeros(g.num_nodes, dtype=np.int)
        active_vertexes[nodes] = 1

        # Find cells that share these nodes
        active_cells = np.squeeze(np.where((cn.transpose() * active_vertexes) > 0))
        # Append the newly found active cells
        cell_ind = np.hstack((cell_ind, active_cells))

        # Multiply face_nodes.transpose() (e.g. node-faces) with the active
        # vertexes to get the number of active nodes perm face
        num_active_face_nodes = np.array(g.face_nodes.transpose() * active_vertexes)
        # Total number of nodes per face
        num_face_nodes = np.array(g.face_nodes.sum(axis=0))
        # Active faces are those where all nodes are active.
        active_face_ind = np.squeeze(
            np.argwhere((num_active_face_nodes == num_face_nodes).ravel("F"))
        )
        active_faces[active_face_ind] = 1

    face_ind = np.squeeze(np.where(active_faces))

    # Do a sort of the indexes to be returned.
    cell_ind.sort()
    face_ind.sort()
    # Return, with data type int
    return cell_ind.astype("int"), face_ind.astype("int")


def map_subgrid_to_grid(g, loc_faces, loc_cells, is_vector):

    num_faces_loc = loc_faces.size
    num_cells_loc = loc_cells.size

    nd = g.dim
    if is_vector:
        face_map = sps.csr_matrix(
            (
                np.ones(num_faces_loc * nd),
                (expand_indices_nd(loc_faces, nd), np.arange(num_faces_loc * nd)),
            ),
            shape=(g.num_faces * nd, num_faces_loc * nd),
        )

        cell_map = sps.csr_matrix(
            (
                np.ones(num_cells_loc * nd),
                (np.arange(num_cells_loc * nd), expand_indices_nd(loc_cells, nd)),
            ),
            shape=(num_cells_loc * nd, g.num_cells * nd),
        )
    else:
        face_map = sps.csr_matrix(
            (np.ones(num_faces_loc), (loc_faces, np.arange(num_faces_loc))),
            shape=(g.num_faces, num_faces_loc),
        )
        cell_map = sps.csr_matrix(
            (np.ones(num_cells_loc), (np.arange(num_cells_loc), loc_cells)),
            shape=(num_cells_loc, g.num_cells),
        )
    return face_map, cell_map


# ------------------------------------------------------------------------------


def compute_discharges(
    gb,
    physics="flow",
    d_name="discharge",
    p_name="pressure",
    lam_name="mortar_solution",
    data=None,
):
    """
    Computes discharges over all faces in the entire grid /grid bucket given
    pressures for all nodes, provided as node properties.

    Parameter:
    gb: grid bucket with the following data fields for all nodes/grids:
        'flux': Internal discretization of fluxes.
        'bound_flux': Discretization of boundary fluxes.
        'pressure': Pressure values for each cell of the grid (overwritten by p_name).
        'bc_val': Boundary condition values.
            and the following edge property field for all connected grids:
        'coupling_flux': Discretization of the coupling fluxes.
    physics (string): defaults to 'flow'. The physic regime
    d_name (string): defaults to 'discharge'. The keyword which the computed
                     discharge will be stored by in the dictionary.
    p_name (string): defaults to 'pressure'. The keyword that the pressure
                     field is stored by in the dictionary
    data (dictionary): defaults to None. If gb is mono-dimensional grid the
                       data dictionary must be given. If gb is a
                       multi-dimensional grid, this variable has no effect

    Returns:
        gb, the same grid bucket with the added field 'discharge' added to all
        node data fields. Note that the fluxes between grids will be added only
        at the gb edge, not at the node fields. The sign of the discharges
        correspond to the directions of the normals, in the edge/coupling case
        those of the higher grid. For edges beteween grids of equal dimension,
        there is an implicit assumption that all normals point from the second
        to the first of the sorted grids (gb.sorted_nodes_of_edge(e)).
    """
    if not isinstance(gb, GridBucket) and not isinstance(gb, pp.GridBucket):
        pa = data["param"]
        if data.get("flux") is not None:
            dis = data["flux"] * data[p_name] + data["bound_flux"] * pa.get_bc_val(
                physics
            )
        else:
            raise ValueError(
                "Discharges can only be computed if a flux-based discretization has been applied"
            )
        data[d_name] = dis
        return

    # Compute fluxes from pressures internal to the subdomain, and for global
    # boundary conditions.
    for g, d in gb:
        if g.dim > 0:
            pa = d["param"]
            if d.get("flux") is not None:
                dis = d["flux"] * d[p_name] + d["bound_flux"] * pa.get_bc_val(physics)
            else:
                raise ValueError(
                    "Discharges can only be computed if a flux-based discretization has been applied"
                )

            d[d_name] = dis

    # Compute fluxes over internal faces, induced by the mortar flux. These
    # are a critical part of what makes MPFA consistent, but will not be
    # present for TPFA.
    # Note that fluxes over faces on the subdomain boundaries are not included,
    # these are already accounted for in the mortar solution.
    for e, d in gb.edges():
        g_h = gb.nodes_of_edge(e)[1]
        # The mapping mortar_to_hat_bc contains is composed of a mapping to
        # faces on the higher-dimensional grid, and computation of the induced
        # fluxes.
        induced_flux = d["mortar_to_hat_bc"] * d[lam_name]
        # Remove contribution directly on the boundary faces.
        induced_flux[g_h.tags["fracture_faces"]] = 0
        gb.node_props(g_h)[d_name] += induced_flux
