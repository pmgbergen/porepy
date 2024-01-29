"""
Various FV specific utility functions.

Implementation note: This could perhaps have been implemneted as a superclass
for MPxA discertizations, however, due to the somewhat intricate inheritance relation
between these methods, the current structure with multiple auxiliary methods emerged.

"""
from __future__ import annotations

from typing import Any, Callable, Generator, Optional

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data


class SubcellTopology:
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

    def __init__(self, sd):
        """
        Constructor for subcell topology

        Args:
            sd: grid
        """
        self.sd = sd

        # Indices of neighboring faces and cells. The indices are sorted to
        # simplify later treatment
        sd.cell_faces.sort_indices()
        face_ind, cell_ind = sd.cell_faces.nonzero()
        # Number of faces per node
        num_face_nodes = np.diff(sd.face_nodes.indptr)

        # Duplicate cell and face indices, so that they can be matched with
        # the nodes
        cells_duplicated = pp.matrix_operations.rldecode(
            cell_ind, num_face_nodes[face_ind]
        )
        faces_duplicated = pp.matrix_operations.rldecode(
            face_ind, num_face_nodes[face_ind]
        )
        M = sps.coo_matrix(
            (np.ones(face_ind.size), (face_ind, np.arange(face_ind.size))),
            shape=(face_ind.max() + 1, face_ind.size),
        )
        nodes_duplicated = sd.face_nodes * M
        nodes_duplicated = nodes_duplicated.indices

        face_nodes_indptr = sd.face_nodes.indptr
        face_nodes_indices = sd.face_nodes.indices
        face_nodes_data = np.arange(face_nodes_indices.size) + 1
        sub_face_mat = sps.csc_matrix(
            (face_nodes_data, face_nodes_indices, face_nodes_indptr)
        )
        sub_faces = sub_face_mat * M
        sub_faces = (sub_faces.data - 1).astype(int)

        # If the grid has periodic faces the topology of the subcells are changed.
        # The left and right faces should be intrepreted as one face topologically.
        # The face_nodes and cell_faces maps in the grid geometry does not consider
        # this. We therefore have to merge the left subfaces with the right subfaces.
        if hasattr(sd, "periodic_face_map"):
            sorted_left = np.sort(sd.periodic_face_map[0])
            sorted_right = np.sort(sd.periodic_face_map[1])
            # It should be straightforward to generalize to the case where the faces
            # are not sorted. You have to first sort sd.periodic_face_map[0] and
            # sd.periodic_face_map[1], then use the two sorted arrays to find the left
            # and right subfaces, then map the subfaces back to the original
            # sd.periodic_face_map.
            if not np.allclose(sorted_left, sd.periodic_face_map[0]):
                raise NotImplementedError(
                    "Can not create subcell topology for periodic faces that are not sorted"
                )
            if not np.allclose(sorted_right, sd.periodic_face_map[1]):
                raise NotImplementedError(
                    "Can not create subcell topology for periodic faces that are not sorted"
                )
            left_subfaces = np.where(
                np.isin(faces_duplicated, sd.periodic_face_map[0])
            )[0]
            right_subfaces = np.where(
                np.isin(faces_duplicated, sd.periodic_face_map[1])
            )[0]
            # We loose the ordering of sd.per map using np.isin. But since we have assumed
            # sd.periodic_face_map[0] and sd.periodic_face_map[1] to be sorted, we can easily
            # retrive the ordering by this trick:
            left_subfaces = left_subfaces[np.argsort(faces_duplicated[left_subfaces])]
            right_subfaces = right_subfaces[
                np.argsort(faces_duplicated[right_subfaces])
            ]

            # The right subface nodes should be equal to the left subface nodes. We
            # also have to change the nodes of any other subface that has a node that
            # is on the rigth boundary.
            for i in range(right_subfaces.size):
                # We loop over each righ subface and find all other nodes that has the
                # same index as the right node. These node indices are swapped with the
                # corresponding left node index.
                nodes_duplicated = np.where(
                    nodes_duplicated == nodes_duplicated[right_subfaces[i]],
                    nodes_duplicated[left_subfaces[i]],
                    nodes_duplicated,
                )
            # Set the right subfaces equal the left subfaces
            sub_faces[right_subfaces] = sub_faces[left_subfaces]

        # Sort data
        idx = np.lexsort(
            (sub_faces, faces_duplicated, nodes_duplicated, cells_duplicated)
        )
        self.nno = nodes_duplicated[idx]
        self.cno = cells_duplicated[idx]
        self.fno = faces_duplicated[idx]
        self.subfno = sub_faces[idx].astype(int)
        self.subhfno = np.arange(idx.size, dtype=">i4")
        self.num_cno = self.cno.max() + 1
        self.num_nodes = self.nno.max() + 1
        # If we have periodic faces, the subface indices might have gaps. E.g., if
        # subface 4 is mapped to subface 1, the index 4 is not included into subfno.
        # The following code will then subtract 1 from all subface indices larger than 4.
        _, Ia, Ic = np.unique(self.subfno, return_index=True, return_inverse=True)
        self.subfno = (
            self.subfno - np.cumsum(np.diff(np.r_[-1, self.subfno[Ia]]) - 1)[Ic]
        )

        # Make subface indices unique, that is, pair the indices from the two
        # adjacent cells
        _, unique_subfno = np.unique(self.subfno, return_index=True)

        self.num_subfno = self.subfno.max() + 1
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
        s += str(self.sd.num_nodes) + " nodes\n"
        s += str(self.sd.num_faces) + " faces\n"
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

        Args:
            other: sps.matrix, size (self.subhfno.size x something)

        Returns
            sps.matrix, size (self.subfno_unique.size x something)
        """

        sgn = self.sd.cell_faces[self.fno, self.cno].A
        pair_over_subfaces = sps.coo_matrix((sgn[0], (self.subfno, self.subhfno)))
        return pair_over_subfaces * other

    def pair_over_subfaces_nd(self, other):
        """nd-version of pair_over_subfaces, see above."""
        nd = self.sd.dim
        # For force balance, displacements and stresses on the two sides of the
        # matrices must be paired
        # Operator to create the pairing
        sgn = self.sd.cell_faces[self.fno, self.cno].A
        pair_over_subfaces = sps.coo_matrix((sgn[0], (self.subfno, self.subhfno)))
        # vector version, to be used on stresses
        pair_over_subfaces_nd = sps.kron(sps.eye(nd), pair_over_subfaces)
        return pair_over_subfaces_nd * other


# ------------------------ End of class SubcellTopology ----------------------


def compute_dist_face_cell(sd, subcell_topology, eta, return_paired=True):
    """
    Compute vectors from cell centers continuity points on each sub-face.

    The location of the continuity point is given by

        x_cp = (1-eta) * x_facecenter + eta * x_vertex

    On the boundary, eta is set to zero, thus the continuity point is at the
    face center

    Args:
        sd: Grid
        subcell_topology: Of class subcell topology in this module
        eta: [0,1), eta = 0 gives cont. pt. at face midpoint, eta = 1 means at
            the vertex. If eta is given as a scalar this value will be applied to
            all subfaces except the boundary faces, where eta=0 will be enforced.
            If the length of eta equals the number of subfaces, eta[i] will be used
            in the computation of the continuity point of the subface s_t.subfno_unique[i].
            Note that eta=0 at the boundary is ONLY enforced for scalar eta.

    Returns
        sps.csr() matrix representation of vectors. Size sd.nf x (sd.nc * sd.nd)

    Raises:
        ValueError if the size of eta is not 1 or subcell_topology.num_subfno_unique.
    """
    _, blocksz = pp.matrix_operations.rlencode(
        np.vstack((subcell_topology.cno, subcell_topology.nno))
    )
    dims = sd.dim

    _, cols = np.meshgrid(subcell_topology.subhfno, np.arange(dims))
    cols += pp.matrix_operations.rldecode(np.cumsum(blocksz) - blocksz[0], blocksz)
    if np.asarray(eta).size == subcell_topology.num_subfno_unique:
        eta_vec = eta[subcell_topology.subfno]
    elif np.asarray(eta).size == 1:
        eta_vec = eta * np.ones(subcell_topology.fno.size)
        # Set eta values to zero at the boundary
        bnd = np.in1d(subcell_topology.fno, sd.get_all_boundary_faces())
        eta_vec[bnd] = 0
    else:
        raise ValueError("size of eta must either be 1 or number of subfaces")
    cp = sd.face_centers[:, subcell_topology.fno] + eta_vec * (
        sd.nodes[:, subcell_topology.nno] - sd.face_centers[:, subcell_topology.fno]
    )
    dist = cp - sd.cell_centers[:, subcell_topology.cno]

    ind_ptr = np.hstack((np.arange(0, cols.size, dims), cols.size))
    mat = sps.csr_matrix((dist.ravel("F"), cols.ravel("F"), ind_ptr))

    if return_paired:
        return subcell_topology.pair_over_subfaces(mat)
    else:
        return mat


def determine_eta(sd: pp.Grid) -> float:
    """Set default value for the location of continuity point eta in MPFA and
    MSPA.

    The function is intended to give a best estimate of eta in cases where the
    user has not specified a value.

    Args:
        sd: Grid for discretization

    Returns:
        double. 1/3 if the grid in known to consist of simplicies (it is one of
           TriangleGrid, TetrahedralGrid, or their structured versions). 0 if
           not.
    """

    if "StructuredTriangleGrid" in sd.name:
        return 1 / 3
    elif "TriangleGrid" in sd.name:
        return 1 / 3
    elif "StructuredTetrahedralGrid" in sd.name:
        return 1 / 3
    elif "TetrahedralGrid" in sd.name:
        return 1 / 3
    else:
        return 0


def find_active_indices(
    parameter_dictionary: dict[str, Any], sd: pp.Grid
) -> tuple[np.ndarray, np.ndarray]:
    """Process information in parameter dictionary on whether the discretization
    should consider a subgrid. Look for fields in the parameter dictionary called
    specified_cells, specified_faces or specified_nodes. These are then processed by
    the function pp.fvutils.cell-ind_for_partial_update.

    If no relevant information is found, the active indices are all cells and
    faces in the grid.

    Args:
        parameter_dictionary (dict): Parameters, potentially containing fields
            "specified_cells", "specified_faces", "specified_nodes".
        sd (pp.Grid): Grid to be discretized.

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
            sd, cells=specified_cells, faces=specified_faces, nodes=specified_nodes
        )
        parameter_dictionary["active_cells"] = active_cells
        parameter_dictionary["active_faces"] = active_faces
    else:
        # All cells and faces in the grid should be updated
        active_cells = np.arange(sd.num_cells)
        active_faces = np.arange(sd.num_faces)
        parameter_dictionary["active_cells"] = active_cells
        parameter_dictionary["active_faces"] = active_faces

    return active_cells, active_faces


def parse_partition_arguments(
    partition_arguments: Optional[dict[str, int]] = None
) -> tuple[int | None, int | None]:
    """Parse arguments related to the splitting of discretization into subproblems.

    Parameters:
        parameter_dictionary (dict): Parameters, potentially containing fields
            "max_memory" and "num_subproblems".

    Returns:
        Values to be used in the partitioning of the grid. One of the values will be
        numerical, the other will be None; it is up to the calling method to use the
        former to define a partitioning.


        int | None: Maximum memory footprint allowed for the discretization. If
            ``partition_arguments`` has a key ``max_memory``, this value will be
            returned. If ``partition_arguments`` is ``None``, the default value of 1e9
            will be returned. If ``partition_arguments`` does not have a key
            ``max_memory``, but has a key ``num_subproblems``, the value will be set to
            ``None``.

        int | None: The number of subproblems to construct. If ``partition_arguments``
            has a key ``num_subproblems``, but no key ``max_memory``, the value will be
            returned. In all other cases, the value will be set to ``None``.

    """

    # Control of the number of subdomanis.
    if partition_arguments is not None:
        if (
            "max_memory" in partition_arguments
            or "num_subproblems" not in partition_arguments
        ):
            # If max_memory is given, use it. If num_subproblems is not given, use
            # default (which is max_memory = 1e9). Cast to int to avoid problems with
            # mypy.
            max_memory = int(partition_arguments.get("max_memory", 1e9))
            # Explicitly set num_subproblems to None, to signal that it should not
            # be used.
            num_subproblems = None
        else:  # Only num_subproblems is given
            num_subproblems = partition_arguments["num_subproblems"]
            # Explicitly set max_memory to None, to signal that it should not be
            # used.
            max_memory = None
    else:
        # No values are given, use default. Cast to int to avoid problems with mypy.
        max_memory = int(1e9)
        # Explicitly set num_subproblems to None, to signal that it should not be
        # used.
        num_subproblems = None

    return max_memory, num_subproblems


def subproblems(
    sd: pp.Grid,
    peak_memory_estimate: int,
    max_memory: Optional[int] = None,
    num_subproblems: Optional[int] = None,
) -> Generator[
    tuple[pp.Grid, np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None
]:
    """Split a grid into subgrids in preparation for discretization with limited memory
    footprint.

    The subgrids are constructed by partititoning the grid; see comments in the code
    for details, including information on overlap between subgrids.

    Parameters:
        sd: Grid to be partitioned.
        peak_memory_estimate: Estimate of the peak memory footprint of the
            discretization.
        max_memory: Maximum memory footprint allowed for the discretization.
        num_subproblems: Number of subproblems to construct.

        At least one of max_memory and num_subproblems must be given. If both are given,
        max_memory will be used.

    Raises:
        ValueError: If neither max_memory nor num_subproblems are given.

    Yields:
        Subgrids and topological information:

        pp.Grid:
            The subgrid to be discretized.

        :obj:`~numpy.ndarray`:
            Indices (in the global grid) of the faces to be discretized. This does not
            include faces that are in the overlap, but not in the subgrid proper.

        :obj:`~numpy.ndarray`:
            Indices (in the global grid) of the cells contained in the subgrid (not
            including the overlap).

        :obj:`~numpy.ndarray`:
            Indices (in the global grid) of all cells in the subgrid, including those
            in the overlap. Represented as a numpy array, so that element i gives the
            global index of the i-th cell in the subgrid.

        :obj:`~numpy.ndarray`:
            Indices (in the global grid) of all faces in the subgrid, including those
            in the overlap. Represented as a numpy array, so that element i gives the
            global index of the i-th face in the subgrid.

    """

    if sd.dim == 0:
        # nothing realy to do here
        loc_faces = np.ones(sd.num_faces, dtype=bool)
        loc_cells = np.ones(sd.num_cells, dtype=bool)
        loc2g_cells = np.ones(sd.num_cells, dtype=bool)
        loc2g_face = np.ones(sd.num_faces, dtype=bool)
        yield sd, loc_faces, loc_cells, loc2g_cells, loc2g_face

    if max_memory is not None:
        num_part: int = np.ceil(peak_memory_estimate / max_memory).astype(int)
    elif num_subproblems is not None:
        num_part = num_subproblems
    else:
        raise ValueError("Either max_memory or num_subproblems must be given")

    if num_part == 1:
        yield sd, np.arange(sd.num_faces), np.arange(sd.num_cells), np.arange(
            sd.num_cells
        ), np.arange(sd.num_faces)

    else:
        # Since MPxA discretizations are based on interaction regions (cells in the dual
        # grid), we need to construct the subgrids with an overlap: If a vertex is part
        # of the subgrid proper, the overlap must be large enough that all cells that
        # share this vertex are included in the discretization stencil. The overlap will
        # mean that certain faces and cells will be discretized multiple times, it is
        # the responsibility of the discretization to handle this.
        #
        # To construct the partitioning, we first define a partitioning with no overlap,
        # identify the extra cells that should be included in the overlap, and then
        # define the subgrid with overlap.

        # Use the partition model to define a partitioning with no overlap. The function
        # called will decide how to construct the partitioning, depending on the grid
        # type, third-party software available etc.
        part: np.ndarray = pp.partition.partition(sd, num_part)

        # Cell-node relation
        cn: sps.csc_matrix = sd.cell_nodes()

        # Loop over all partition regions, construct local grids, and information to map
        # between local and global grids.
        for p in np.unique(part):
            # Cells in this partitioning
            cells_in_partition: np.ndarray = np.argwhere(part == p).ravel("F")

            # To discretize with as little overlap as possible, we use the
            # keyword nodes to specify the update stencil. Find nodes of the
            # local cells.
            cells_in_partition_boolean = np.zeros(sd.num_cells, dtype=bool)
            cells_in_partition_boolean[cells_in_partition] = 1

            # Nodes present in this partition
            nodes_in_partition: np.ndarray = np.squeeze(
                np.where((cn * cells_in_partition_boolean) > 0)
            )

            # Find computational stencil (cells and faces), based on the nodes in this
            # partition
            loc_cells, loc_faces = pp.fvutils.cell_ind_for_partial_update(
                sd, nodes=nodes_in_partition
            )

            # Extract subgrid, together with mappings between local and active
            # (global, or at least less local) cells
            sub_sd, l2g_faces, _ = pp.partition.extract_subgrid(sd, loc_cells)
            l2g_cells = sub_sd.parent_cell_ind

            yield sub_sd, loc_faces, cells_in_partition, l2g_cells, l2g_faces


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
        pp.matrix_operations.zero_rows(mat, eliminate_ind)


def expand_indices_nd(ind: np.ndarray, nd: int, direction="F") -> np.ndarray:
    """
    Expand indices from scalar to vector form.

    Examples:
    >>> i = np.array([0, 1, 3])
    >>> expand_indices_nd(i, 2)
    (array([0, 1, 2, 3, 6, 7]))

    >>> expand_indices_nd(i, 3, "C")
    (array([0, 3, 9, 1, 4, 10, 2, 5, 11])

    Args:
        ind
        nd
        direction

    Returns

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


def adjust_eta_length(
    eta: np.ndarray, sub_sd: pp.Grid, l2g_faces: np.ndarray
) -> np.ndarray:
    """Adjusts length of vector valued eta for problems partitioned into subproblems.

    Eta can either be a scalar or a vector. If a vector valued eta is passed, it will
    have a length equal to the number of subfaces in the entire grid. If the grid is
    partitioned into subgrids, we need to adjust the length of eta to match the subfaces
    of the subgrid.

    Parameters:
        eta: MPFA/MPSA-eta.
        sub_sd: A subgrid of the domain. Eta is adjusted according to the subfaces in
            sub_sd.
        l2g_faces: Indices (in the global grid) of all faces in the subgrid. Represented
            as a numpy array, so that element i gives the global index of the i-th face
            in the subgrid.

    Returns:
        An array of eta values corresponding to a grid that arises from from domain
        partitioning.

    """
    # Use information in the sparse formatting to find the number of nodes per face
    num_nodes_per_face = np.diff(sub_sd.face_nodes.tocsc().indptr)
    # Verify that all faces have equally many nodes
    assert np.unique(num_nodes_per_face).size == 1
    expansion_index = num_nodes_per_face[0]

    indices = expand_indices_nd(l2g_faces, expansion_index)
    loc_eta = np.array([eta[i] for i in indices])
    return loc_eta


def map_hf_2_f(fno=None, subfno=None, nd=None, sd=None):
    """
    Create mapping from half-faces to faces for vector problems.
    Either fno, subfno and nd should be given or g (and optinally nd) should be
    given.

    Args:
    EITHER:
       fno (np.ndarray): face numbering in sub-cell topology based on unique subfno
       subfno (np.ndarrary): sub-face numbering
       nd (int): dimension
    OR:
        g (pp.Grid): If a grid is supplied the function will set:
            fno = pp.fvutils.SubcellTopology(g).fno_unique
            subfno = pp.fvutils.SubcellTopology(g).subfno_unique
        nd (int): Optinal, defaults to sd.dim. Defines the dimension of the vector.

    Returns
    """
    if sd is not None:
        s_t = SubcellTopology(sd)
        fno = s_t.fno_unique
        subfno = s_t.subfno_unique
        if nd is None:
            nd = sd.dim
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

    Args:
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

    Args:
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


def scalar_divergence(sd: pp.Grid) -> sps.csr_matrix:
    """
    Get divergence operator for a grid.

    The operator is easily accessible from the grid itself, so we keep it
    here for completeness.

    See also vector_divergence(g)

    Args:
        sd (pp.Grid): grid

    Returns
        divergence operator
    """
    return sd.cell_faces.T.tocsr()


def vector_divergence(sd: pp.Grid) -> sps.csr_matrix:
    """
    Get vector divergence operator for a grid g

    It is assumed that the first column corresponds to the x-equation of face
    0, second column is y-equation etc. (and so on in nd>2). The next column is
    then the x-equation for face 1. Correspondingly, the first row
    represents x-component in first cell etc.

    Args:
        sd (pp.Grid): grid

    Returns
        vector_div (sparse csr matrix), dimensions: nd * (num_cells, num_faces)
    """
    # Scalar divergence
    scalar_div = sd.cell_faces

    # Vector extension, convert to coo-format to avoid odd errors when one
    # grid dimension is 1 (this may return a bsr matrix)
    # The order of arguments to sps.kron is important.
    block_div = sps.kron(scalar_div, sps.eye(sd.dim)).tocsc()

    return block_div.transpose().tocsr()


def scalar_tensor_vector_prod(
    sd: pp.Grid, k: pp.SecondOrderTensor, subcell_topology: SubcellTopology
) -> tuple[sps.csr_matrix, np.ndarray, np.ndarray]:
    """
    Compute product of normal vectors and tensors on a sub-cell level.
    This is essentially defining Darcy's law for each sub-face in terms of
    sub-cell gradients. Thus, we also implicitly define the global ordering
    of sub-cell gradient variables (via the interpretation of the columns in
    nk).
    NOTE: In the local numbering below, in particular in the variables i and j,
    it is tacitly assumed that sd.dim == sd.nodes.shape[0] ==
    sd.face_normals.shape[0] etc. See implementation note in main method.
    Args:
        sd (pp.Grid): Discretization grid
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
    cell_node_blocks, blocksz = pp.matrix_operations.rlencode(
        np.vstack((subcell_topology.cno, subcell_topology.nno))
    )

    nd = sd.dim

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
    j += pp.matrix_operations.rldecode(sum_blocksz - blocksz[0], blocksz)

    # Distribute faces equally on the sub-faces
    num_nodes = np.diff(sd.face_nodes.indptr)
    normals = sd.face_normals[:, subcell_topology.fno] / num_nodes[subcell_topology.fno]

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


class ExcludeBoundaries:
    """Wrapper class to store mappings needed in the finite volume discretizations.
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

        Args:
            subcell_topology (pp.SubcellTopology)
            bound (pp.BoundaryCondition / pp.BoundaryConditionVectorial)
            nd (int)

        Attributes:
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
        col = np.argwhere(np.logical_not(ids))
        row = np.arange(col.size)
        return sps.coo_matrix(
            (np.ones(row.size, dtype=bool), (row, col.ravel("C"))),
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

        Args:
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

        Args:
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

        Args:
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

        Args:
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

        Args:
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
        """Mapping to exclude faces/component with any boundary condition from
        local systems.

        Args:
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

        Args:
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

        Args:
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


def partial_update_discretization(
    sd: pp.Grid,  # Grid
    data: dict,  # full data dictionary for this grid
    keyword: str,  # keyword for the target discretization
    discretize: Callable,  # Discretization operation
    dim: Optional[int] = None,  # dimension. Used to expand vector quantities
    scalar_cell_right: Optional[list[str]] = None,  # See method documentation
    vector_cell_right: Optional[list[str]] = None,
    scalar_face_right: Optional[list[str]] = None,
    vector_face_right: Optional[list[str]] = None,
    scalar_cell_left: Optional[list[str]] = None,
    vector_cell_left: Optional[list[str]] = None,
    scalar_face_left: Optional[list[str]] = None,
    vector_face_left: Optional[list[str]] = None,
    second_keyword: Optional[str] = None,  # Used for biot discertization
) -> None:
    """Do partial update of discretization scheme.

    This is intended as a helper function for the update_discretization methods of
    fv schemes. In particular, this method allows for a unified implementation of
    update in mpfa, mpsa and biot.

    The implementation is somewhat heavy to cover both mpfa, mpsa and biot.

    Parameters scalar_cell_right, vector_face_left etc. are lists of keys in the
    discretization matrix dictionary. They are used to tell whehter the matrix should
    be considered a cell or face quantity, and scalar of vector. Left and right are
    used to map rows and columns, respectively.
    Together these fields allows for mapping a discretization between grids.

    If a term misses a right or left mapping, it will be ignored.

    """
    # Process input
    if scalar_cell_right is None:
        scalar_cell_right = []
    if vector_cell_right is None:
        vector_cell_right = []
    if scalar_face_right is None:
        scalar_face_right = []
    if vector_face_right is None:
        vector_face_right = []
    if scalar_cell_left is None:
        scalar_cell_left = []
    if vector_cell_left is None:
        vector_cell_left = []
    if scalar_face_left is None:
        scalar_face_left = []
    if vector_face_left is None:
        vector_face_left = []

    if dim is None:
        dim = sd.dim

    update_info = data["update_discretization"]
    # By default, neither cells nor faces have been updated
    update_cells = update_info.get("modified_cells", np.array([], dtype=int))
    update_faces = update_info.get("modified_faces", np.array([], dtype=int))

    # Mappings of cells and faces. Default to identity maps
    cell_map = update_info.get("map_cells", sps.identity(sd.num_cells))
    face_map = update_info.get("map_faces", sps.identity(sd.num_faces))

    # left cell quantities (known example: div_u term in Biot), are a bit special
    # in that they require expanded computational stencils.
    # To see this, consider an update of a single cell. For a left face quantity,
    # this would require update of the neighboring faces, as will be detected by the
    # cell_ind_for_partial_update below. The necessary update to nearby cells would
    # be achieved by the subsequent multiplication with a divergence. For left cell
    # matrices, the latter step is not available, thus the necessary overlap in
    # stencil must be explicitly set.
    if len(vector_cell_left) > 0 or len(scalar_cell_left) > 0:
        update_cells = pp.partition.overlap(sd, update_cells, 1)

        # We will need the non-updated cells as well (but not faces, for similar
        # reasons as outlined above).
        passive_cells = np.setdiff1d(np.arange(sd.num_cells), update_cells)

    do_discretize = False
    # The actual discretization stencil may be larger than the modified cells and
    # faces (if specified).
    _, active_faces = pp.fvutils.cell_ind_for_partial_update(
        sd, cells=update_cells, faces=update_faces
    )
    active_faces = np.unique(active_faces)

    # Find the faces next to the active faces. All these may be updated (depending on
    # the type of discretizations present).
    _, cells, _ = sparse_array_to_row_col_data(sd.cell_faces[active_faces])
    active_cells = np.unique(cells)
    passive_cells = np.setdiff1d(np.arange(sd.num_cells), active_cells)

    param = data[pp.PARAMETERS][keyword]
    if update_cells.size > 0:
        param["specified_cells"] = update_cells
        do_discretize = True
    if update_faces.size > 0:
        param["specified_faces"] = update_faces
        do_discretize = True

    # Loop over all existing discretization matrices. Map rows and columns,
    # according to the left/right, cell/face and scalar/vector specifications.
    # Also eliminate contributions to rows that will also be updated (but not
    # columns, a non-updated row should keep its information about a column
    # to be updated).
    mat_dict = data[pp.DISCRETIZATION_MATRICES][keyword]
    mat_dict_copy = {}
    for key, val in mat_dict.items():
        mat = val

        # First multiplication from the right
        if key in scalar_cell_right:
            mat = mat * cell_map.T
        elif key in vector_cell_right:
            mat = mat * sps.kron(cell_map.T, sps.eye(dim))
        elif key in scalar_face_right:
            mat = mat * face_map.T
        elif key in vector_face_right:
            mat = mat * sps.kron(face_map.T, sps.eye(dim))
        else:
            # If no mapping is provided, we assume the matrix is not part of the
            # target discretization, and ignore it.
            continue

        # Mapping of faces. Enforce csr format to enable elimination of rows below.
        if key in scalar_cell_left:
            mat = cell_map * mat
            # Zero out existing contributions from the active faces. This is necessary
            # due to the expansive computational stencils for MPxA methods.
            pp.fvutils.remove_nonlocal_contribution(active_cells, 1, mat)
        elif key in vector_cell_left:
            # Need a tocsr() here to work with row-based elimination
            mat = (sps.kron(cell_map, sps.eye(dim)) * mat).tocsr()
            pp.fvutils.remove_nonlocal_contribution(active_cells, dim, mat)
        elif key in scalar_face_left:
            mat = face_map * mat
            pp.fvutils.remove_nonlocal_contribution(active_faces, 1, mat)
        elif key in vector_face_left:
            mat = (sps.kron(face_map, sps.eye(dim)) * mat).tocsr()
            pp.fvutils.remove_nonlocal_contribution(active_faces, dim, mat)
        else:
            # If no mapping is provided, we assume the matrix is not part of the
            # target discretization, and ignore it.
            continue

        mat_dict_copy[key] = mat

    # Do the actual discretization
    if do_discretize:
        discretize(sd, data)

    # Define new discretization as a combination of mapped and rediscretized
    for key, val in data[pp.DISCRETIZATION_MATRICES][keyword].items():
        # If the key is present in the matrix dictionary of the second_keyword,
        # we skip it, and handle below.
        # See comment on Biot discretization below
        if (
            second_keyword is not None
            and key in data[pp.DISCRETIZATION_MATRICES][second_keyword].keys()
        ):
            continue

        if (
            key in data[pp.DISCRETIZATION_MATRICES][keyword].keys()
            and key in mat_dict_copy.keys()
        ):
            # By now, the two matrices should have compatible size
            data[pp.DISCRETIZATION_MATRICES][keyword][key] = mat_dict_copy[key] + val

    # The Biot discretization is special, in that it places part of matrices in
    # the mechanics dictionary, a second part in flow. If a second keyword is provided,
    # the corresponding matrices must be processed, and added with the stored, mapped
    # values.
    # Implementation note: we assume that the previous discretizations are all placed
    # under the primary keyword, see Biot for an example of the necessary pre-processing.
    # It could perhaps have been better allow for processing of two keywords in the
    # mapping, but the implementation ended up being as it is.
    if second_keyword is not None:
        for key, val in data[pp.DISCRETIZATION_MATRICES][second_keyword].items():
            if key in mat_dict_copy.keys():
                if key in scalar_cell_left:
                    remove_nonlocal_contribution(passive_cells, 1, val)
                elif key in vector_cell_left:
                    remove_nonlocal_contribution(passive_cells, dim, val)
                data[pp.DISCRETIZATION_MATRICES][second_keyword][key] = (
                    mat_dict_copy[key] + val
                )


def cell_ind_for_partial_update(
    sd: pp.Grid,
    cells: Optional[np.ndarray] = None,
    faces: Optional[np.ndarray] = None,
    nodes: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Obtain indices of cells and faces needed for a partial update of the
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

    Args:
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
    active_faces = np.zeros(sd.num_faces, dtype=bool)

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
        # sub-faces between o-cells that share a vertex with x. Since the
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

        cn = sd.cell_nodes()

        # The active faces (to be updated; (o-x and o-o above) are those that
        # share at least one vertex with cells in ind.
        prim_cells = np.zeros(sd.num_cells, dtype=bool)
        prim_cells[cells] = 1
        # Vertexes of the cells
        active_vertexes = np.zeros(sd.num_nodes, dtype=bool)
        active_vertexes[np.squeeze(np.where(cn * prim_cells > 0))] = 1

        # Faces of the vertexes, these will be the active faces.
        active_face_ind = np.squeeze(
            np.where(sd.face_nodes.transpose() * active_vertexes > 0)
        )
        active_faces[active_face_ind] = 1

        # Secondary vertexes, involved in at least one of the active faces,
        # that is, the faces to be updated. Corresponds to vertexes between o-o
        # above.
        active_vertexes[np.squeeze(np.where(sd.face_nodes * active_faces > 0))] = 1

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
        #      S S S
        #    V o o o V
        #  S o o x o o S
        #  S o o x o o S
        #    V o o o V
        #      S S S
        #
        # To illustrate for the Cartesian configuration above: The face
        # between the two x-cells are specified, and this requires the
        # inclusion of all o and V-cells. The S-cells are superfluous from a
        # computational standpoint, but they are added in the same operation as the Vs.
        # It may be possible to exclude them, but does not seem worth the mental effort.
        #
        # NOTE: The four V-cells are only needed for Biot-discretizations,
        # specifically to correctly deal with the div-u terms. To be precise, an update
        # of a face requires a recomputation of all cells that.
        # NOTE: The actual stencil retured is even bigger than above ()

        cf = sd.cell_faces
        # This avoids overwriting data in cell_faces.
        data = np.ones_like(cf.data)
        cf = sps.csc_matrix((data, cf.indices, cf.indptr))

        primary_faces = np.zeros(sd.num_faces, dtype=bool)
        primary_faces[faces] = 1

        # The active faces are those sharing a vertex with the primary faces
        primary_vertex = np.zeros(sd.num_nodes, dtype=bool)
        primary_vertex[np.squeeze(np.where((sd.face_nodes * primary_faces) > 0))] = 1
        active_face_ind = np.squeeze(
            np.where((sd.face_nodes.transpose() * primary_vertex) > 0)
        )
        active_faces[active_face_ind] = 1

        # Find vertexes of the active faces
        active_nodes = np.zeros(sd.num_nodes, dtype=bool)
        active_nodes[np.squeeze(np.where((sd.face_nodes * active_faces) > 0))] = 1

        cn = sd.cell_nodes()

        # Primary cells, those that share a vertex with the faces
        primary_cells = np.squeeze(np.where((cn.transpose() * active_nodes) > 0))

        # Get all nodes of the primary cells. These are the secondary_nodes
        active_cells = np.zeros(sd.num_cells, dtype=bool)
        active_cells[primary_cells] = 1
        secondary_nodes = np.where(cn * active_cells)[0]
        active_nodes[secondary_nodes] = 1

        # Get the secondary cells. Refering to the above drawing, this will add
        # V and S-cells.
        secondary_cells = np.where(cn.transpose() * active_nodes > 0)[0]

        cell_ind = np.hstack((cell_ind, secondary_cells))

    if nodes is not None:
        # Pick out all cells that have the specified nodes as a vertex.
        # The active faces will be those that have all their vertexes included
        # in nodes.
        cn = sd.cell_nodes()
        # Introduce active nodes, and make the input nodes active
        # The data type of active_vertex is int (in contrast to similar cases
        # in other parts of this function), since we will use it to count the
        # number of active face_nodes below.
        active_vertexes = np.zeros(sd.num_nodes, dtype=int)
        active_vertexes[nodes] = 1

        # Find cells that share these nodes
        active_cells = np.squeeze(np.where((cn.transpose() * active_vertexes) > 0))
        # Append the newly found active cells
        cell_ind = np.hstack((cell_ind, active_cells))

        # Multiply face_nodes.transpose() (e.g. node-faces) with the active
        # vertexes to get the number of active nodes perm face
        num_active_face_nodes = np.array(sd.face_nodes.transpose() * active_vertexes)
        # Total number of nodes per face
        num_face_nodes = np.array(sd.face_nodes.sum(axis=0))
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
    sd: pp.Grid,
    loc_faces: np.ndarray,
    loc_cells: np.ndarray,
    is_vector: bool,
    nd: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Obtain mappings from the cells and faces of a subgrid back to a larger grid.

    Args:
        g (pp.Grid): The larger grid.
        loc_faces (np.ndarray): For each face in the subgrid, the index of the
            corresponding face in the larger grid.
        loc_cells (np.ndarray): For each cell in the subgrid, the index of the
            corresponding cell in the larger grid.
        is_vector (bool): If True, the returned mappings are sized to fit with vector
            variables, with nd elements per cell and face.
        nd (int, optional): Dimension. Defaults to sd.dim.

    Retuns:
        sps.csr_matrix, size (sd.num_faces, loc_faces.size): Mapping from local to
            global faces. If is_vector is True, the size is multiplied with sd.dim.
        sps.csr_matrix, size (loc_cells.size, sd.num_cells): Mapping from global to
            local cells. If is_vector is True, the size is multiplied with sd.dim.

    """
    if nd is None:
        nd = sd.dim

    num_faces_loc = loc_faces.size
    num_cells_loc = loc_cells.size

    if is_vector:
        face_map = sps.csr_matrix(
            (
                np.ones(num_faces_loc * nd),
                (expand_indices_nd(loc_faces, nd), np.arange(num_faces_loc * nd)),
            ),
            shape=(sd.num_faces * nd, num_faces_loc * nd),
        )

        cell_map = sps.csr_matrix(
            (
                np.ones(num_cells_loc * nd),
                (np.arange(num_cells_loc * nd), expand_indices_nd(loc_cells, nd)),
            ),
            shape=(num_cells_loc * nd, sd.num_cells * nd),
        )
    else:
        face_map = sps.csr_matrix(
            (np.ones(num_faces_loc), (loc_faces, np.arange(num_faces_loc))),
            shape=(sd.num_faces, num_faces_loc),
        )
        cell_map = sps.csr_matrix(
            (np.ones(num_cells_loc), (np.arange(num_cells_loc), loc_cells)),
            shape=(num_cells_loc, sd.num_cells),
        )
    return face_map, cell_map


def diagonal_scaling_matrix(mat: sps.spmatrix) -> sps.spmatrix:
    """Helper function to form a diagonal matrix that scales the rows of a matrix.

    Parameters:
        mat: Matrix to be scaled.

    Returns:
        Diagonal matrix with the diagonal elements equal to the row-wise sum of the
        absolute values of the input matrix.

    """

    # Take the row-wise sum of all non-zero elements in the matrix. Work on a copy,
    # since we want to manipulate the matrix elements.
    tmp = mat.copy()
    # Use an absolute value here. For some of the matrices the row sum will be zero
    # on interior faces.
    tmp.data = np.abs(tmp.data)
    # Take a sum here. Intuitively, an average would be better, but calling tmp.mean()
    # would take the average over all elements, most of which are zero (this turned out
    # not to be optimal). We could also find the number of non-zero elements and divide
    # the sum by this, but a sum seems to be good enough.
    scalings = tmp.sum(axis=1).A.ravel()
    # Diagonal scaling matrix
    full_scaling = sps.dia_matrix((1.0 / scalings, 0), shape=mat.shape)

    return full_scaling


def boundary_to_sub_boundary(bound, subcell_topology):
    """
    Convert a boundary condition defined for faces to a boundary condition defined by
    subfaces.

    Args:
    bound (pp.BoundaryCondition/pp.BoundarConditionVectorial):
        Boundary condition given for faces.
    subcell_topology (pp.fvutils.SubcellTopology):
        The subcell topology defining the finite volume subgrid.

    Returns:
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
    bound.num_faces = np.max(subcell_topology.subfno) + 1
    bound.bf = np.where(np.isin(subcell_topology.fno, bound.bf))[0]
    return bound


def append_dofs_of_discretization(sd, d, kw1, kw2, k_dof):
    """
    Appends rows to existing discretizations stored as 'stress' and
    'bound_stress' in the data dictionary on the nodes. Only applies to the
    highest dimension (for now, at least). The newly added faces are found
    from 'new_faces' in the data dictionary.
    Assumes all new rows and columns should be appended, not inserted to
    the "interior" of the discretization matrices.
    sd -     grid object
    d -     corresponding data dictionary
    kw1 -   keyword for the stored discretization in the data dictionary,
            e.g. 'flux'
    kw2 -   keyword for the stored boundary discretization in the data
            dictionary, e.g. 'bound_flux'
    """
    cells = d["new_cells"]
    faces = d["new_faces"]
    n_new_cells = cells.size * k_dof
    n_new_faces = faces.size * k_dof

    # kw1
    new_rows = sps.csr_matrix((n_new_faces, sd.num_cells * k_dof - n_new_cells))
    new_columns = sps.csr_matrix((sd.num_faces * k_dof, n_new_cells))
    d[kw1] = sps.hstack([sps.vstack([d[kw1], new_rows]), new_columns], format="csr")
    # kw2
    new_rows = sps.csr_matrix((n_new_faces, sd.num_faces * k_dof - n_new_faces))
    new_columns = sps.csr_matrix((sd.num_faces * k_dof, n_new_faces))
    d[kw2] = sps.hstack([sps.vstack([d[kw2], new_rows]), new_columns], format="csr")


def partial_discretization(
    sd, data, tensor, bnd, apertures, partial_discr, physics="flow"
):
    """
    Perform a partial (local) multi-point discretization on a grid with
    provided data, tensor and boundary conditions by
        1)  Appending the existing discretization matrices to the right size
        according to the added cells and faces.
        2)  Discretizing on the relevant subgrids by calls to the provided
        partial discretization function (mpfa_partial or mpsa_partial).
        3)  Zeroing out the rows corresponding to the updated faces.
        4)  Inserting the newly computed values to the just deleted rows.
    """
    # Get keywords and affected geometry
    known_physics = ["flow", "mechanics"]
    assert physics in known_physics
    if physics == "flow":
        kw1, kw2 = "flux", "bound_flux"
        dof_multiplier = 1
    elif physics == "mechanics":
        kw1, kw2 = "stress", "bound_stress"
        dof_multiplier = sd.dim
    cells = sd.tags.get("discretize_cells")
    faces = sd.tags.get("discretize_faces")
    nodes = sd.tags.get("discretize_nodes")

    # Update the existing discretization to the right size
    append_dofs_of_discretization(sd, data, kw1, kw2, dof_multiplier)
    trm, bound_flux, affected_faces = partial_discr(
        sd,
        tensor,
        bnd,
        cells=cells,
        faces=faces,
        nodes=nodes,
        apertures=apertures,
        inverter=None,
    )

    # Account for dof offset for mechanical problem
    affected_faces = expand_indices_nd(affected_faces, dof_multiplier)

    pp.matrix_operations.zero_rows(data[kw1], affected_faces)
    pp.matrix_operations.zero_rows(data[kw2], affected_faces)
    data[kw1] += trm
    data[kw2] += bound_flux
