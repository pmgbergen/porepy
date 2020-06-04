"""
Various FV specific utility functions.
"""
import numpy as np
import scipy.sparse as sps
from typing import Tuple, Any, Generator, Dict, Optional

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
        self.num_nodes = self.nno.max() + 1

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


def compute_dist_face_cell(g, subcell_topology, eta, return_paired=True):
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
        the vertex. If eta is given as a scalar this value will be applied to
        all subfaces except the boundary faces, where eta=0 will be enforced.
        If the length of eta equals the number of subfaces, eta[i] will be used
        in the computation of the continuity point of the subface s_t.subfno_unique[i].
        Note that eta=0 at the boundary is ONLY enforced for scalar eta.

    Returns
    -------
    sps.csr() matrix representation of vectors. Size g.nf x (g.nc * g.nd)

    Raises:
    -------
    ValueError if the size of eta is not 1 or subcell_topology.num_subfno_unique.
    """
    _, blocksz = matrix_compression.rlencode(
        np.vstack((subcell_topology.cno, subcell_topology.nno))
    )
    dims = g.dim

    _, cols = np.meshgrid(subcell_topology.subhfno, np.arange(dims))
    cols += matrix_compression.rldecode(np.cumsum(blocksz) - blocksz[0], blocksz)
    if np.asarray(eta).size == subcell_topology.num_subfno_unique:
        eta_vec = eta[subcell_topology.subfno]
    elif np.asarray(eta).size == 1:
        eta_vec = eta * np.ones(subcell_topology.fno.size)
        # Set eta values to zero at the boundary
        bnd = np.in1d(subcell_topology.fno, g.get_all_boundary_faces())
        eta_vec[bnd] = 0
    else:
        raise ValueError("size of eta must either be 1 or number of subfaces")
    cp = g.face_centers[:, subcell_topology.fno] + eta_vec * (
        g.nodes[:, subcell_topology.nno] - g.face_centers[:, subcell_topology.fno]
    )
    dist = cp - g.cell_centers[:, subcell_topology.cno]

    ind_ptr = np.hstack((np.arange(0, cols.size, dims), cols.size))
    mat = sps.csr_matrix((dist.ravel("F"), cols.ravel("F"), ind_ptr))
    if return_paired:
        return subcell_topology.pair_over_subfaces(mat)
    else:
        return mat


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


def find_active_indices(
    parameter_dictionary: Dict[str, Any], g: pp.Grid
) -> Tuple[np.ndarray, np.ndarray]:
    """ Process information in parameter dictionary on whether the discretization
    should consider a subgrid. Look for fields in the parameter dictionary called
    specified_cells, specified_faces or specified_nodes. These are then processed by
    the function pp.fvutils.cell-ind_for_partial_update.

    If no relevant information is found, the active indices are all cells and
    faces in the grid.

    Parameters:
        parameter_dictionary (dict): Parameters, potentially containing fields
            "specified_cells", "specified_faces", "specified_nodes".
        g (pp.Grid): Grid to be discretized.

    Returns:
        np.ndarray: Cells to be included in the active grid.
        np.ndarary: Faces to have their discretization updated. NOTE: This may not
            be all faces in the grid.

    """
    # The discretization can be limited to a specified set of cells, faces or nodes
    # If none of these are specified, the entire grid will be discretized
    specified_cells = parameter_dictionary.get("specified_cells", None)
    specified_faces = parameter_dictionary.get("specified_faces", None)
    specified_nodes = parameter_dictionary.get("specified_nodes", None)

    # Find the cells and faces that should be considered for discretization
    if (
        (specified_cells is not None)
        or (specified_faces is not None)
        or (specified_nodes is not None)
    ):
        # Find computational stencil, based on specified cells, faces and nodes.
        active_cells, active_faces = pp.fvutils.cell_ind_for_partial_update(
            g, cells=specified_cells, faces=specified_faces, nodes=specified_nodes
        )
        parameter_dictionary["active_cells"] = active_cells
        parameter_dictionary["active_faces"] = active_faces
    else:
        # All cells and faces in the grid should be updated
        active_cells = np.arange(g.num_cells)
        active_faces = np.arange(g.num_faces)
        parameter_dictionary["active_cells"] = active_cells
        parameter_dictionary["active_faces"] = active_faces

    return active_cells, active_faces


def subproblems(
    g: pp.Grid, max_memory: int, peak_memory_estimate: int
) -> Generator[Any, None, None]:

    if g.dim == 0:
        # nothing realy to do here
        loc_faces = np.ones(g.num_faces, dtype=bool)
        loc_cells = np.ones(g.num_cells, dtype=bool)
        loc2g_cells = sps.eye(g.num_cells, dtype=bool)
        loc2g_face = sps.eye(g.num_faces, dtype=bool)
        return g, loc_faces, loc_cells, loc2g_cells, loc2g_face

    num_part: int = np.ceil(peak_memory_estimate / max_memory).astype(np.int)

    if num_part == 1:
        yield g, np.arange(g.num_faces), np.arange(g.num_cells), np.arange(
            g.num_cells
        ), np.arange(g.num_faces)

    else:
        # Let partitioning module apply the best available method
        part: np.ndarray = pp.partition.partition(g, num_part)

        # Cell-node relation
        cn: sps.csc_matrix = g.cell_nodes()

        # Loop over all partition regions, construct local problemsac, and transfer
        # discretization to the entire active grid
        for p in np.unique(part):
            # Cells in this partitioning
            cells_in_partition: np.ndarray = np.argwhere(part == p).ravel("F")

            # To discretize with as little overlap as possible, we use the
            # keyword nodes to specify the update stencil. Find nodes of the
            # local cells.
            cells_in_partition_boolean = np.zeros(g.num_cells, dtype=np.bool)
            cells_in_partition_boolean[cells_in_partition] = 1

            nodes_in_partition: np.ndarray = np.squeeze(
                np.where((cn * cells_in_partition_boolean) > 0)
            )

            # Find computational stencil, based on the nodes in this partition
            loc_cells, loc_faces = pp.fvutils.cell_ind_for_partial_update(
                g, nodes=nodes_in_partition
            )

            # Extract subgrid, together with mappings between local and active
            # (global, or at least less local) cells
            sub_g, l2g_faces, _ = pp.partition.extract_subgrid(g, loc_cells)
            l2g_cells = sub_g.parent_cell_ind

            yield sub_g, loc_faces, cells_in_partition, l2g_cells, l2g_faces


def remove_nonlocal_contribution(
    raw_ind: np.ndarray, nd: int, *args: sps.spmatrix
) -> None:
    """
    For a set of matrices, zero out rows associated with given faces, adjusting for
    the matrices being related to vector quantities if necessary.

    Example: If raw_ind = np.array([2]), and nd = 2, rows 4 and 5 will be eliminated
        (row 0 and 1 will in this case be associated with face 0, row 2 and 3 with face
         1).

    Args:
        raw_ind (np.ndarray): Face indices to have their rows eliminated.
        nd (int): Spatial dimension. Needed to map face indices to rows.
        *args (sps.spmatrix): Set of matrices. Will be eliminated in place.

    Returns:
        None: DESCRIPTION.

    """
    eliminate_ind = pp.fvutils.expand_indices_nd(raw_ind, nd)
    for mat in args:
        pp.fvutils.zero_out_sparse_rows(mat, eliminate_ind)


# ------------- Methods related to block inversion ----------------------------


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
            raise ImportError(
                """Compiled Cython module not available. Is cython installed?"""
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
        @numba.jit("f8[:](i4[:],i4[:],f8[:],i8[:])", nopython=True, cache=True)
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
        except np.linalg.LinAlgError:
            raise ValueError("Error in inversion of local linear systems")
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


def expand_indices_nd(ind, nd, direction="F"):
    """
    Expand indices from scalar to vector form.

    Examples:
    >>> i = np.array([0, 1, 3])
    >>> __expand_indices_nd(i, 2)
    (array([0, 1, 2, 3, 6, 7]))

    >>> __expand_indices_nd(i, 3, "C")
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


def expand_indices_incr(ind, dim, increment):

    # Convenience method for duplicating a list, with a certain increment

    # Duplicate rows
    ind_nd = np.tile(ind, (dim, 1))
    # Add same increment to each row (0*incr, 1*incr etc.)
    ind_incr = ind_nd + increment * np.array([np.arange(dim)]).transpose()
    # Back to row vector
    ind_new = ind_incr.reshape(-1, order="F")
    return ind_new


def map_hf_2_f(fno=None, subfno=None, nd=None, g=None):
    """
    Create mapping from half-faces to faces for vector problems.
    Either fno, subfno and nd should be given or g (and optinally nd) should be
    given.

    Parameters
    ----------
    EITHER:
       fno (np.ndarray): face numbering in sub-cell topology based on unique subfno
       subfno (np.ndarrary): sub-face numbering
       nd (int): dimension
    OR:
        g (pp.Grid): If a grid is supplied the function will set:
            fno = pp.fvutils.SubcellTopology(g).fno_unique
            subfno = pp.fvutils.SubcellTopology(g).subfno_unique
        nd (int): Optinal, defaults to g.dim. Defines the dimension of the vector.
    Returns
    -------
    """
    if g is not None:
        s_t = SubcellTopology(g)
        fno = s_t.fno_unique
        subfno = s_t.subfno_unique
        if nd is None:
            nd = g.dim
    hfi = expand_indices_nd(subfno, nd)
    hf = expand_indices_nd(fno, nd)
    hf2f = sps.coo_matrix(
        (np.ones(hf.size), (hf, hfi)), shape=(hf.max() + 1, hfi.max() + 1)
    ).tocsr()
    return hf2f


def cell_vector_to_subcell(nd, sub_cell_index, cell_index):

    """
    Create mapping from sub-cells to cells for scalar problems.
    For example, discretization of div_g-term in mpfa with gravity,
    where g is a cell-center vector (dim nd)

    Parameters
        nd: dimension
        sub_cell_index: sub-cell indices
        cell_index: cell indices

    Returns:
        scipy.sparse.csr_matrix (shape num_subcells * nd, num_cells * nd):

    """

    num_cells = cell_index.max() + 1

    rows = sub_cell_index.ravel("F")
    cols = expand_indices_nd(cell_index, nd)
    vals = np.ones(rows.size)
    mat = sps.coo_matrix(
        (vals, (rows, cols)), shape=(sub_cell_index.size, num_cells * nd)
    ).tocsr()

    return mat


def cell_scalar_to_subcell_vector(nd, sub_cell_index, cell_index):

    """
    Create mapping from sub-cells to cells for vector problems.
    For example, discretization of grad_p-term in Biot,
    where p is a cell-center scalar

    Parameters
        nd: dimension
        sub_cell_index: sub-cell indices
        cell_index: cell indices

    Returns:
        scipy.sparse.csr_matrix (shape num_subcells * nd, num_cells):

    """

    num_cells = cell_index.max() + 1

    def build_sc2c_single_dimension(dim):
        rows = np.arange(sub_cell_index[dim].size)
        cols = cell_index
        vals = np.ones(rows.size)
        mat = sps.coo_matrix(
            (vals, (rows, cols)), shape=(sub_cell_index[dim].size, num_cells)
        ).tocsr()
        return mat

    sc2c = build_sc2c_single_dimension(0)
    for i in range(1, nd):
        this_dim = build_sc2c_single_dimension(i)
        sc2c = sps.vstack([sc2c, this_dim])

    return sc2c


def scalar_divergence(g: pp.Grid) -> sps.csr_matrix:
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
    return g.cell_faces.T.tocsr()


def vector_divergence(g: pp.Grid) -> sps.csr_matrix:
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
    block_div = sps.kron(scalar_div, sps.eye(g.dim)).tocsc()

    return block_div.transpose().tocsr()


def scalar_tensor_vector_prod(
    g: pp.Grid, k: pp.SecondOrderTensor, subcell_topology: SubcellTopology
) -> Tuple[sps.csr_matrix, np.ndarray, np.ndarray]:
    """
    Compute product of normal vectors and tensors on a sub-cell level.
    This is essentially defining Darcy's law for each sub-face in terms of
    sub-cell gradients. Thus, we also implicitly define the global ordering
    of sub-cell gradient variables (via the interpretation of the columns in
    nk).
    NOTE: In the local numbering below, in particular in the variables i and j,
    it is tacitly assumed that g.dim == g.nodes.shape[0] ==
    g.face_normals.shape[0] etc. See implementation note in main method.
    Parameters:
        g (pp.Grid): Discretization grid
        k (pp.Second_order_tensor): The permeability tensor
        subcell_topology (fvutils.SubcellTopology): Wrapper class containing
            subcell numbering.
    Returns:
        nk: sub-face wise product of normal vector and permeability tensor.
        cell_node_blocks pairings of node and cell indices, which together
            define a sub-cell.
        sub_cell_ind: index of all subcells
    """

    # Stack cell and nodes, and remove duplicate rows. Since subcell_mapping
    # defines cno and nno (and others) working cell-wise, this will
    # correspond to a unique rows (Matlab-style) from what I understand.
    # This also means that the pairs in cell_node_blocks uniquely defines
    # subcells, and can be used to index gradients etc.
    cell_node_blocks, blocksz = pp.utils.matrix_compression.rlencode(
        np.vstack((subcell_topology.cno, subcell_topology.nno))
    )

    nd = g.dim

    # Duplicates in [cno, nno] corresponds to different faces meeting at the
    # same node. There should be exactly nd of these. This test will fail
    # for pyramids in 3D
    if not np.all(blocksz == nd):
        raise AssertionError()

    # Define row and column indices to be used for normal_vectors * perm.
    # Rows are based on sub-face numbers.
    # Columns have nd elements for each sub-cell (to store a gradient) and
    # is adjusted according to block sizes
    _, j = np.meshgrid(subcell_topology.subhfno, np.arange(nd))
    sum_blocksz = np.cumsum(blocksz)
    j += pp.utils.matrix_compression.rldecode(sum_blocksz - blocksz[0], blocksz)

    # Distribute faces equally on the sub-faces
    num_nodes = np.diff(g.face_nodes.indptr)
    normals = g.face_normals[:, subcell_topology.fno] / num_nodes[subcell_topology.fno]

    # Represent normals and permeability on matrix form
    ind_ptr = np.hstack((np.arange(0, j.size, nd), j.size))
    normals_mat = sps.csr_matrix((normals.ravel("F"), j.ravel("F"), ind_ptr))
    k_mat = sps.csr_matrix(
        (k.values[::, ::, cell_node_blocks[0]].ravel("F"), j.ravel("F"), ind_ptr)
    )

    nk = normals_mat * k_mat

    # Unique sub-cell indexes are pulled from column indices, we only need
    # every nd column (since nd faces of the cell meet at each vertex)
    sub_cell_ind = j[::, 0::nd]
    return nk, cell_node_blocks, sub_cell_ind


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

    if not A.getformat() == "csr":
        raise ValueError("Can only zero out sparse rows for csr matrix")

    ip = A.indptr
    row_indices = mcolon.mcolon(ip[rows], ip[rows + 1])
    A.data[row_indices] = 0
    if diag is not None:
        # now we set the diagonal
        diag_vals = np.zeros(A.shape[1])
        diag_vals[rows] = diag
        A += sps.dia_matrix((diag_vals, 0), shape=A.shape)

    return A


# -----------------------------------------------------------------------------


class ExcludeBoundaries(object):
    """ Wrapper class to store mappings needed in the finite volume discretizations.
    The original use for this class was for exclusion of equations that are
    redundant due to the presence of boundary conditions, hence the name. The use of
    this class has increased to also include linear transformation that can be applied
    to the subfaces before the exclusion operator. This will typically be a rotation matrix,
    so that the boundary conditions can be specified in an arbitary coordinate system.

    The systems being set up in mpfa (and mpsa) describe continuity of flux and
    potential (respectively stress and displacement) on all sub-faces. For the boundary
    faces, unless a Robin condition is specified, only one of the two should be included
    (e.g. for a Dirichlet boundary condition, there is no concept of continuity of
    flux/stresses). This class contains mappings to eliminate the necessary fields in
    order set the correct boundary conditions.

    """

    def __init__(self, subcell_topology, bound, nd):
        """
        Define mappings to exclude boundary subfaces/components with Dirichlet,
        Neumann or Robin conditions. If bound.bc_type=="scalar" we assign one
        component per subface, while if bound.bc_type=="vector" we assign nd
        components per subface.

        Parameters
        ----------
        subcell_topology (pp.SubcellTopology)
        bound (pp.BoundaryCondition / pp.BoundaryConditionVectorial)
        nd (int)

        Attributes:
        ----------
        basis_matrix: sps.csc_matrix, mapping from all subfaces/components to all
            subfaces/components. Will be applied to other before the exclusion
            operator for the functions self.exlude...(other, transform),
            if transform==True.
        robin_weight: sps.csc_matrix, mapping from all subfaces/components to all
            subfaces/components. Gives the weight that is applied to the displacement in
            the Robin condition.
        exclude_neu: sps.csc_matrix, mapping from all subfaces/components to those having
            pressure continuity
        exclude_dir: sps.csc_matrix, mapping from all subfaces/components to those having
            flux continuity
        exclude_neu_dir: sps.csc_matrix, mapping from all subfaces/components to those
            having both pressure and flux continuity (i.e., Robin + internal)
        exclude_neu_rob: sps.csc_matrix, mapping from all subfaces/components to those
            not having Neumann or Robin conditions (i.e., Dirichlet + internal)
        exclude_rob_dir: sps.csc_matrix, mapping from all subfaces/components to those
            not having Robin or Dirichlet conditions (i.e., Neumann + internal)
        exclude_bnd: sps.csc_matrix, mapping from all subfaces/components to internal
            subfaces/components.
        keep_neu: sps.csc_matrix, mapping from all subfaces/components to only Neumann
            subfaces/components.
        keep_rob: sps.csc_matrix, mapping from all subfaces/components to only Robin
            subfaces/components.
        """
        self.nd = nd
        self.bc_type = bound.bc_type

        # Short hand notation
        num_subfno = subcell_topology.num_subfno_unique
        self.num_subfno = num_subfno
        self.any_rob = np.any(bound.is_rob)

        # Define mappings to exclude boundary values
        if self.bc_type == "scalar":
            self.basis_matrix = self._linear_transformation(bound.basis)
            self.robin_weight = self._linear_transformation(bound.robin_weight)

            self.exclude_neu = self._exclude_matrix(bound.is_neu)
            self.exclude_dir = self._exclude_matrix(bound.is_dir)
            self.exclude_rob = self._exclude_matrix(bound.is_rob)
            self.exclude_neu_dir = self._exclude_matrix(bound.is_neu | bound.is_dir)
            self.exclude_neu_rob = self._exclude_matrix(bound.is_neu | bound.is_rob)
            self.exclude_rob_dir = self._exclude_matrix(bound.is_rob | bound.is_dir)
            self.exclude_bnd = self._exclude_matrix(
                bound.is_rob | bound.is_dir | bound.is_neu
            )
            self.keep_neu = self._exclude_matrix(~bound.is_neu)
            self.keep_rob = self._exclude_matrix(~bound.is_rob)

        elif self.bc_type == "vectorial":
            self.basis_matrix = self._linear_transformation(bound.basis)
            self.robin_weight = self._linear_transformation(bound.robin_weight)

            self.exclude_neu = self._exclude_matrix_xyz(bound.is_neu)
            self.exclude_dir = self._exclude_matrix_xyz(bound.is_dir)
            self.exclude_rob = self._exclude_matrix_xyz(bound.is_rob)
            self.exclude_neu_dir = self._exclude_matrix_xyz(bound.is_neu | bound.is_dir)
            self.exclude_neu_rob = self._exclude_matrix_xyz(bound.is_neu | bound.is_rob)
            self.exclude_rob_dir = self._exclude_matrix_xyz(bound.is_rob | bound.is_dir)
            self.exclude_bnd = self._exclude_matrix_xyz(
                bound.is_rob | bound.is_dir | bound.is_neu
            )
            self.keep_rob = self._exclude_matrix_xyz(~bound.is_rob)
            self.keep_neu = self._exclude_matrix_xyz(~bound.is_neu)

    def _linear_transformation(self, loc_trans):
        """
        Creates a global linear transformation matrix from a set of local matrices.
        The global matrix is a mapping from sub-faces to sub-faces as given by loc_trans.
        The loc_trans matrices are given per sub-face and is a mapping from a nd vector
        on each subface to a nd vector on each subface. (If self.bc_type="scalar" nd=1
        is enforced). The loc_trans is a (nd, nd, num_subfno_unique) matrix where
        loc_trans[:, :, i] gives the local transformation matrix to be applied to
        subface i.

        Example:
        --------
        # We have two subfaces in dimension 2.
        self.num_subfno = 4
        self.nd = 2
        # Identity map on first subface and rotate second by np.pi/2 radians
        loc_trans = np.array([[[1, 0], [0, -1]], [[0, 1], [1, 0]]])
        print(sef._linear_transformation(loc_trans))
            [[1, 0, 0, 0],
             [0, 0, 0, -1],
             [0, 1, 0, 0],
             [0, 0, 1, 0]]
        """
        if self.bc_type == "scalar":
            data = loc_trans
            col = np.arange(self.num_subfno)
            row = np.arange(self.num_subfno)
            return sps.coo_matrix(
                (data, (row, col)), shape=(self.num_subfno, self.num_subfno)
            ).tocsr()
        elif self.bc_type == "vectorial":
            data = loc_trans.ravel("C")
            row = np.arange(self.num_subfno * self.nd).reshape((-1, self.num_subfno))
            row = np.tile(row, (1, self.nd)).ravel("C")
            col = np.tile(np.arange(self.num_subfno * self.nd), (1, self.nd)).ravel()

            return sps.coo_matrix(
                (data, (row, col)),
                shape=(self.num_subfno * self.nd, self.num_subfno * self.nd),
            ).tocsr()
        else:
            raise AttributeError("Unknow loc_trans type: " + self.bc_type)

    def _exclude_matrix(self, ids):
        """
        creates an exclusion matrix. This is a mapping from sub-faces to
        all sub-faces except those given by ids.
        Example:
        ids = [0, 2]
        self.num_subfno = 4
        print(sef._exclude_matrix(ids))
            [[0, 1, 0, 0],
              [0, 0, 0, 1]]
        """
        col = np.argwhere([not it for it in ids])
        row = np.arange(col.size)
        return sps.coo_matrix(
            (np.ones(row.size, dtype=np.bool), (row, col.ravel("C"))),
            shape=(row.size, self.num_subfno),
        ).tocsr()

    def _exclude_matrix_xyz(self, ids):
        col_x = np.argwhere([not it for it in ids[0]])

        col_y = np.argwhere([not it for it in ids[1]])
        col_y += self.num_subfno

        col_neu = np.append(col_x, [col_y])

        if self.nd == 3:
            col_z = np.argwhere([not it for it in ids[2]])
            col_z += 2 * self.num_subfno
            col_neu = np.append(col_neu, [col_z])

        row_neu = np.arange(col_neu.size)
        exclude_nd = sps.coo_matrix(
            (np.ones(row_neu.size), (row_neu, col_neu.ravel("C"))),
            shape=(row_neu.size, self.nd * self.num_subfno),
        ).tocsr()

        return exclude_nd

    def exclude_dirichlet(self, other, transform=True):
        """
        Mapping to exclude faces/components with Dirichlet boundary conditions from
        local systems.

        Parameters:
            other (scipy.sparse matrix): Matrix of local equations for
                continuity of flux and pressure.

        Returns:
            scipy.sparse matrix, with rows corresponding to faces/components with
                Dirichlet conditions eliminated.

        """
        exclude_dirichlet = self.exclude_dir
        if transform:
            return exclude_dirichlet * self.basis_matrix * other
        return exclude_dirichlet * other

    def exclude_neumann(self, other, transform=True):
        """
        Mapping to exclude faces/components with Neumann boundary conditions from
        local systems.

        Parameters:
            other (scipy.sparse matrix): Matrix of local equations for
                continuity of flux and pressure.

        Returns:
            scipy.sparse matrix, with rows corresponding to faces/components with
                Neumann conditions eliminated.
        """
        if transform:
            return self.exclude_neu * self.basis_matrix * other
        return self.exclude_neu * other

    def exclude_neumann_robin(self, other, transform=True):
        """
        Mapping to exclude faces/components with Neumann and Robin boundary
        conditions from local systems.

        Parameters:
            other (scipy.sparse matrix): Matrix of local equations for
                continuity of flux and pressure.

        Returns:
            scipy.sparse matrix, with rows corresponding to faces/components with
                Neumann conditions eliminated.
        """
        if transform:
            return self.exclude_neu_rob * self.basis_matrix * other
        else:
            return self.exclude_neu_rob * other

    def exclude_neumann_dirichlet(self, other, transform=True):
        """
        Mapping to exclude faces/components with Neumann and Dirichlet boundary
        conditions from local systems.

        Parameters:
            other (scipy.sparse matrix): Matrix of local equations for
                continuity of flux and pressure.

        Returns:
            scipy.sparse matrix, with rows corresponding to faces/components with
                Neumann conditions eliminated.
        """
        if transform:
            return self.exclude_neu_dir * self.basis_matrix * other
        return self.exclude_neu_dir * other

    def exclude_robin_dirichlet(self, other, transform=True):
        """
        Mapping to exclude faces/components with Robin and Dirichlet boundary
        conditions from local systems.

        Parameters:
            other (scipy.sparse matrix): Matrix of local equations for
                continuity of flux and pressure.

        Returns:
            scipy.sparse matrix, with rows corresponding to faces/components with
                Neumann conditions eliminated.
        """
        if transform:
            return self.exclude_rob_dir * self.basis_matrix * other
        return self.exclude_rob_dir * other

    def exclude_boundary(self, other, transform=False):
        """ Mapping to exclude faces/component with any boundary condition from
        local systems.

        Parameters:
            other (scipy.sparse matrix): Matrix of local equations for
                continuity of flux and pressure.

        Returns:
            scipy.sparse matrix, with rows corresponding to faces/components with
                Neumann conditions eliminated.

        """
        if transform:
            return self.exclude_bnd * self.basis_matrix * other
        return self.exclude_bnd * other

    def keep_robin(self, other, transform=True):
        """
        Mapping to exclude faces/components that is not on the Robin boundary
        conditions from local systems.

        Parameters:
            other (scipy.sparse matrix): Matrix of local equations for
                continuity of flux and pressure.

        Returns:
            scipy.sparse matrix, with rows corresponding to faces/components with
                all but Robin conditions eliminated.
        """
        if transform:
            return self.keep_rob * self.basis_matrix * other
        return self.keep_rob * other

    def keep_neumann(self, other, transform=True):
        """
        Mapping to exclude faces/components that is not on the Neumann boundary
        conditions from local systems.

        Parameters:
            other (scipy.sparse matrix): Matrix of local equations for
                continuity of flux and pressure.

        Returns:
            scipy.sparse matrix, with rows corresponding to faces/components with
                all but Neumann conditions eliminated.
        """
        if transform:
            return self.keep_neu * self.basis_matrix * other
        return self.keep_neu * other


# -----------------End of class ExcludeBoundaries-----------------------------


def cell_ind_for_partial_update(
    g: pp.Grid,
    cells: np.ndarray = None,
    faces: np.ndarray = None,
    nodes: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
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

    face_ind = np.atleast_1d(np.squeeze(np.where(active_faces)))

    # Do a sort of the indexes to be returned.
    cell_ind.sort()
    face_ind.sort()
    # Return, with data type int
    return cell_ind.astype("int"), face_ind.astype("int")


def map_subgrid_to_grid(
    g: pp.Grid,
    loc_faces: np.ndarray,
    loc_cells: np.ndarray,
    is_vector: bool,
    nd: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """ Obtain mappings from the cells and faces of a subgrid back to a larger grid.

    Parameters:
        g (pp.Grid): The larger grid.
        loc_faces (np.ndarray): For each face in the subgrid, the index of the
            corresponding face in the larger grid.
        loc_cells (np.ndarray): For each cell in the subgrid, the index of the
            corresponding cell in the larger grid.
        is_vector (bool): If True, the returned mappings are sized to fit with vector
            variables, with nd elements per cell and face.
        nd (int, optional): Dimension. Defaults to g.dim.

    Retuns:
        sps.csr_matrix, size (g.num_faces, loc_faces.size): Mapping from local to
            global faces. If is_vector is True, the size is multiplied with g.dim.
        sps.csr_matrix, size (loc_cells.size, g.num_cells): Mapping from global to
            local cells. If is_vector is True, the size is multiplied with g.dim.

    """
    if nd is None:
        nd = g.dim

    num_faces_loc = loc_faces.size
    num_cells_loc = loc_cells.size

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


def compute_darcy_flux(
    gb,
    keyword="flow",
    keyword_store=None,
    d_name="darcy_flux",
    p_name="pressure",
    lam_name="mortar_solution",
    data=None,
    from_iterate=False,
):
    """
    Computes darcy_flux over all faces in the entire grid /grid bucket given
    pressures for all nodes, provided as node properties.

    Parameter:
    gb: grid bucket with the following data fields for all nodes/grids:
        'flux': Internal discretization of fluxes.
        'bound_flux': Discretization of boundary fluxes.
        'pressure': Pressure values for each cell of the grid (overwritten by p_name).
        'bc_val': Boundary condition values.
            and the following edge property field for all connected grids:
        'coupling_flux': Discretization of the coupling fluxes.
    keyword (str): defaults to 'flow'. The parameter keyword used to obtain the
        data necessary to compute the fluxes.
    keyword_store (str): defaults to keyword. The parameter keyword determining
        where the data will be stored.
    d_name (str): defaults to 'darcy_flux'. The parameter name which the computed
        darcy_flux will be stored by in the dictionary.
    p_name (str): defaults to 'pressure'. The keyword that the pressure
        field is stored by in the dictionary.
    lam_name (str): defaults to 'mortar_solution'. The keyword that the mortar flux
        field is stored by in the dictionary.
    data (dictionary): defaults to None. If gb is mono-dimensional grid the data
        dictionary must be given. If gb is a multi-dimensional grid, this variable has
        no effect.

    Returns:
        gb, the same grid bucket with the added field 'darcy_flux' added to all
        node data fields. Note that the fluxes between grids will be added only
        at the gb edge, not at the node fields. The signs of the darcy_flux
        correspond to the directions of the normals, in the edge/coupling case
        those of the higher grid. For edges beteween grids of equal dimension,
        there is an implicit assumption that all normals point from the second
        to the first of the sorted grids (gb.sorted_nodes_of_edge(e)).

    """

    def extract_variable(d, var):
        if from_iterate:
            return d[pp.STATE]["previous_iterate"][var]
        else:
            return d[pp.STATE][var]

    def calculate_flux(param_dict, mat_dict, d):
        # Calculate the flux. First contributions from pressure and boundary conditions
        dis = (
            mat_dict["flux"] * extract_variable(d, p_name)
            + mat_dict["bound_flux"] * param_dict["bc_values"]
        )
        # Discretization of vector source terms
        vector_source_discr = mat_dict["vector_source"]
        # Get the actual source term - put to zero if not provided
        vector_source = param_dict.get(
            "vector_source", np.zeros(vector_source_discr.shape[1])
        )
        dis += vector_source_discr * vector_source
        return dis

    if keyword_store is None:
        keyword_store = keyword
    if not isinstance(gb, GridBucket) and not isinstance(gb, pp.GridBucket):
        parameter_dictionary = data[pp.PARAMETERS][keyword]
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][keyword]
        if "flux" in matrix_dictionary:
            dis = calculate_flux(parameter_dictionary, matrix_dictionary, data)
        else:
            raise ValueError(
                """Darcy_Flux can only be computed if a flux-based
                                 discretization has been applied"""
            )
        data[pp.PARAMETERS][keyword_store][d_name] = dis
        return

    # Compute fluxes from pressures internal to the subdomain, and for global
    # boundary conditions.
    for g, d in gb:
        parameter_dictionary = d[pp.PARAMETERS][keyword]
        matrix_dictionary = d[pp.DISCRETIZATION_MATRICES][keyword]
        if "flux" in matrix_dictionary:
            dis = calculate_flux(parameter_dictionary, matrix_dictionary, d)
        else:
            raise ValueError(
                """Darcy_Flux can only be computed if a flux-based
                             discretization has been applied"""
            )

        d[pp.PARAMETERS][keyword_store][d_name] = dis
    # Compute fluxes over internal faces, induced by the mortar flux. These
    # are a critical part of what makes MPFA consistent, but will not be
    # present for TPFA.
    # Note that fluxes over faces on the subdomain boundaries are not included,
    # these are already accounted for in the mortar solution.
    for e, d in gb.edges():
        g_h = gb.nodes_of_edge(e)[1]
        d_h = gb.node_props(g_h)
        # The mapping mortar_to_hat_bc contains is composed of a mapping to
        # faces on the higher-dimensional grid, and computation of the induced
        # fluxes.

        bound_flux = d_h[pp.DISCRETIZATION_MATRICES][keyword]["bound_flux"]
        induced_flux = (
            bound_flux
            * d["mortar_grid"].mortar_to_master_int()
            * extract_variable(d, lam_name)
        )
        # Remove contribution directly on the boundary faces.
        induced_flux[g_h.tags["fracture_faces"]] = 0
        d_h[pp.PARAMETERS][keyword_store][d_name] += induced_flux
        d[pp.PARAMETERS][keyword_store][d_name] = extract_variable(d, lam_name).copy()


def boundary_to_sub_boundary(bound, subcell_topology):
    """
    Convert a boundary condition defined for faces to a boundary condition defined by
    subfaces.

    Parameters:
    -----------
    bound (pp.BoundaryCondition/pp.BoundarConditionVectorial):
        Boundary condition given for faces.
    subcell_topology (pp.fvutils.SubcellTopology):
        The subcell topology defining the finite volume subgrid.

    Returns:
    --------
    pp.BoundarCondition/pp.BoundarConditionVectorial: An instance of the
        BoundaryCondition/BoundaryConditionVectorial class, where all face values of
        bound has been copied to the subfaces as defined by subcell_topology.
    """
    bound = bound.copy()
    bound.is_dir = np.atleast_2d(bound.is_dir)[:, subcell_topology.fno_unique].squeeze()
    bound.is_rob = np.atleast_2d(bound.is_rob)[:, subcell_topology.fno_unique].squeeze()
    bound.is_neu = np.atleast_2d(bound.is_neu)[:, subcell_topology.fno_unique].squeeze()
    bound.is_internal = np.atleast_2d(bound.is_internal)[
        :, subcell_topology.fno_unique
    ].squeeze()
    if bound.robin_weight.ndim == 3:
        bound.robin_weight = bound.robin_weight[:, :, subcell_topology.fno_unique]
        bound.basis = bound.basis[:, :, subcell_topology.fno_unique]
    else:
        bound.robin_weight = bound.robin_weight[subcell_topology.fno_unique]
        bound.basis = bound.basis[subcell_topology.fno_unique]
    bound.num_faces = subcell_topology.num_subfno_unique
    bound.bf = np.where(np.isin(subcell_topology.fno, bound.bf))[0]
    return bound
