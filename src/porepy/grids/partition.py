"""
Module for partitioning of grids based on various methods.

Intended support is by Cartesian indexing, and METIS-based.

"""
import warnings
import numpy as np
import scipy.sparse as sps
from typing import List, Tuple

import porepy as pp


def partition_metis(g: pp.Grid, num_part: int) -> np.ndarray:
    """
    Partition a grid using metis.

    This function requires that pymetis is installed, as can be done by

        pip install pymetis

    This will install metis itself in addition to the python bindings. There
    are other python bindings for metis as well, but pymetis has behaved well
    so far.

    Parameters:
        g: core.grids.grid: To be partitioned. Only the cell_face attribute is
            used
        num_part (int): Number of partitions.

    Returns:
        np.array (size:g.num_cells): Partition vector, one number in
            [0, num_part) for each cell.

    """
    try:
        import pymetis
    except ImportError:
        warnings.warn("Could not import pymetis. Partitioning by metis will not work.")
        raise ImportError("Cannot partition by pymetis")

    # Connection map between cells
    c2c = g.cell_connection_map()

    # Convert the cells into the format required by pymetis
    adjacency_list = [list(c2c.getrow(i).indices) for i in range(c2c.shape[0])]
    # Call pymetis
    # It seems it is important that num_part is an int, not an np.int.
    part = pymetis.part_graph(int(num_part), adjacency=adjacency_list)

    # The meaning of the first number returned by pymetis is not clear (poor
    # documentation), only return the partitioning.
    return np.array(part[1])


def partition_structured(
    g: pp.TensorGrid, coarse_dims: np.ndarray = None, num_part: int = None
) -> np.ndarray:
    """
    Define a partitioning of a grid based on logical Cartesian indexing.

    The grid should have a field cart_dims, describing the Cartesian dimensions
    of the grid.

    The coarse grid can be specified either by its Cartesian dimensions
    (parameter coarse_dims), or by its total number of partitions (num_part).
    In the latter case, a partitioning will be inferred from the fine-scale
    Cartesian dimensions, in a way that gives roughly the same number of cells
    in each direction.

    Parameters:
        g: core.grids.grid: To be partitioned. Only the cell_face attribute is
            used
        coarse_dims (np.array): Cartesian dimensions of the coarse grids.
        num_part (int): Number of partitions.

    Returns:
        np.array (size:g.num_cells): Partition vector, one number in
            [0, num_part) for each cell.

    Raises:
        Value error if both coarse_dims and num_part are None.

    """

    if (coarse_dims is None) and (num_part is None):
        raise ValueError(
            "Either coarse dimensions or number of coarse cells \
                         must be specified"
        )

    nd = g.dim
    fine_dims = g.cart_dims

    if coarse_dims is None:
        coarse_dims = determine_coarse_dimensions(num_part, fine_dims)

    # Number of fine cells per coarse cell
    fine_per_coarse = np.floor(fine_dims / coarse_dims)

    # First define the coarse index for the individual dimensions.
    ind = []
    for i in range(nd):
        # Fine indexes where the coarse index will increase
        incr_ind = np.arange(0, fine_dims[i], fine_per_coarse[i], dtype="i")

        # If the coarse dimension is not an exact multiple of the fine, there
        # will be an extra cell in this dimension. Remove this.
        if incr_ind.size > coarse_dims[i]:
            incr_ind = incr_ind[:-1]

        # Array for coarse index of fine cells
        loc_ind = np.zeros(fine_dims[i])
        # The index will increase by one
        loc_ind[incr_ind] += 1
        # A cumulative sum now gives the index, but subtract by one to be
        # 0-offset
        ind.append(np.cumsum(loc_ind) - 1)

    # Then combine the indexes. In 2D meshgrid does the job, in 3D it turned
    # out that some acrobatics was necessary to get the right ordering of the
    # cells.
    if nd == 2:
        xi, yi = np.meshgrid(ind[0], ind[1])
        # y-index jumps in steps of the number of coarse x-cells
        glob_dims = (xi + yi * coarse_dims[0]).ravel("C")
    elif nd == 3:
        xi, yi, zi = np.meshgrid(ind[0], ind[1], ind[2])
        # Combine indices, with appropriate jumps in y and z counting
        glob_dims = xi + yi * coarse_dims[0] + zi * np.prod(coarse_dims[:2])
        # This just happened to work, may be logical, but the documentanion of
        # np.meshgrid was hard to comprehend.
        glob_dims = np.swapaxes(np.swapaxes(glob_dims, 1, 2), 0, 1).ravel("C")

    # Return an int
    return glob_dims.astype("int")


def partition_coordinates(
    g: pp.Grid, num_coarse: int, check_connectivity: bool = True
) -> np.ndarray:
    """"
    Brute force partitioning of a grid based on cell center coordinates.

    The intention at the time of implementation is to provide a partitioning
    for general grids that does not rely on METIS being available. However, if
    METIS is available, partition_metis should be prefered.

    The idea is to divide the domain into a coarse Cartesian grid, and then
    assign a coarse partitioning based on the cell center coordinates.

    It is not clear that this will create a connected coarse partitioning for
    all grid cells (it may be possible to construct pathological examples,
    probably involving non-convex cells). We optionally check for connectivity
    and raise an error if this is not fulfilled.

    The method assumes that the cells have a center, that is,
    g.compute_geometry() has been called. If g does not have a field
    cell_centers, compute_geometry() will be called.

    Parameters:
        g (core.grids.grid): Grid to be partitioned.
        num_coarse (int): Target number of coarse cells. The real number of
            coarse cells will be close, but not necessarily equal.
        check_connectivity (boolean, optional): Check if the partitioning form
            connected coarse grids. Defaults to True.

    Returns:
        np.array, size g.num_cells: Partition vector.

    Raises:
        ValueError if the partitioning is found to not form connected subgrids.

    """

    # Compute geometry if necessary
    if not hasattr(g, "cell_centers"):
        g.compute_geometry()

    if g.dim == 0:
        # Nothing really to do here.
        return np.zeros(g.num_cells, dtype=np.int)

    # The division into boxes must be done within the active dimensions of the grid.
    # For 1d and 2d grids, this involves a mapping of the grid into its natural
    # coordinates.
    if g.dim == 1 or g.dim == 2:
        g = g.copy()
        cell_centers, *_, nodes = pp.map_geometry.map_grid(g)
        g.cell_centers = np.vstack((cell_centers, np.zeros(g.num_cells)))
        g.nodes = np.vstack((nodes, np.zeros(g.num_nodes)))

    # Rough computation of the size of the Cartesian coarse grid: Determine the
    # extension of the domain in each direction, transform into integer sizes,
    # and use function to determine coarse dimensions.

    # Use node coordinates to define the boxes
    min_coord = np.min(g.nodes, axis=1)
    max_coord = np.max(g.nodes, axis=1)

    # Drop unused dimensions for 2d (and 1d) grids
    min_coord = min_coord[: g.dim]
    max_coord = max_coord[: g.dim]
    # Cell centers, with the right number of dimensions
    cc = g.cell_centers[: g.dim]

    delta = max_coord - min_coord

    # Estimate of the number of coarse Cartesian cells in each dimensions:
    # Distribute the target number over all dimensions. Then multiply with
    # relative distances.
    # Use ceil to round up, and thus avoid zeros. This may not be perfect, but
    # it should be robust.
    delta_int = np.ceil(np.power(num_coarse, 1 / g.dim) * delta / np.min(delta)).astype(
        "int"
    )

    coarse_dims = determine_coarse_dimensions(num_coarse, delta_int)

    # Effective number of coarse cells, should be close to num_coarse
    nc = coarse_dims.prod()

    # Initialize partition vector
    partition = -np.ones(g.num_cells)

    # Grid resolution of coarse grid (roughly)
    dx = delta / coarse_dims

    #  Loop over all coarse cells, pick out cells that lies within the coarse
    #  box
    for i in range(nc):
        ind = np.array(np.unravel_index(i, coarse_dims))
        # Bounding coordinates
        lower_coord = min_coord + dx * ind
        upper_coord = min_coord + dx * (ind + 1)
        # Find cell centers within the box
        hit = np.logical_and(
            cc >= lower_coord.reshape((-1, 1)), cc < upper_coord.reshape((-1, 1))
        )
        # We have a hit if all coordinates are within the box
        hit_ind = np.argwhere(np.all(hit, axis=0)).ravel(order="C")
        partition[hit_ind] = i

    # Sanity check, all cells should have received a coarse index
    assert partition.min() >= 0

    if check_connectivity:
        for p in np.unique(partition):
            p_ind = np.squeeze(np.argwhere(p == partition))
            if not grid_is_connected(g, p_ind):
                raise ValueError("Partitioning led to unconnected subgrids")

    return partition


def partition(g: pp.Grid, num_coarse: int) -> np.ndarray:
    """
    Wrapper for partition methods, tries to apply best possible algorithm.

    The method will first try to use METIS; if this is not available (or fails
    otherwise), the partition_structured will be applied if the grid is
    Cartesian. The last resort is partitioning based on coordinates.

    See the methods partition_metis(), partition_structured() and
    partition_coordinates() for further details.

    Parameters:
        g (core.grids.grid): Grid to be partitioned.
        num_coarse (int): Target number of coarse cells.

    Returns:
        np.ndarray (int), size g.num_cells: Partition vector.

    """
    try:
        return partition_metis(g, num_coarse)
    except ImportError:
        if isinstance(g, pp.TensorGrid):
            return partition_structured(g, num_part=num_coarse)
        else:
            return partition_coordinates(g, num_coarse)


def determine_coarse_dimensions(target: int, fine_size: np.ndarray) -> np.ndarray:
    """
    For a logically Cartesian grid determine a coarse partitioning based on a
    target number of coarse cells.

    The target size in general will not be a product of the possible grid
    dimensions (it may be a prime, or it may be outside the bounds [1,
    fine_size]. For concreteness, we seek to have roughly the same number of
    cells in each directions (given by the Nd-root of the target). If this
    requires more coarse cells in a dimension than there are fine cells there,
    the coarse size is set equal to the fine, and the remaining cells are
    distributed to the other dimensions.

    Parameters:
        target (int): Target number of coarse cells.
        fine_size (np.ndarray): Number of fine-scale cell in each dimension

    Returns:
        np.ndarray: Coarse dimension sizes.

    Raises:
        ValueError if the while-loop runs more iterations than the number of
            dimensions. This should not happen, in practice it means there is
            bug.

    """

    # The algorithm may be unstable for values outside the relevant bounds
    target = np.maximum(1, np.minimum(target, fine_size.prod()))

    nd = fine_size.size

    # Array to store optimal values. Set the default value to one, this avoids
    # interfering with target_now below.
    optimum = np.ones(nd)
    found = np.zeros(nd, dtype=np.bool)

    # Counter for number of iterations. Should be unnecessary, remove when the
    # code is trusted.
    it_counter = 0

    # Loop until all dimensions have been assigned a number of cells.
    while not np.all(found) and it_counter <= nd:

        it_counter += 1

        # Remaining cells to deal with
        target_now = target / optimum.prod()

        # The simplest option is to take the Nd-root of the target number. This
        # will generally not give integers, and we will therefore settle for the
        # combination of rounding up and down which brings us closest to the
        # target.
        # There should be at least one coarse cell in each dimension, and at
        # maximum as many cells as on the fine scale.
        s_num = np.power(target_now, 1 / (nd - found.sum()))
        s_low = np.maximum(np.ones(nd), np.floor(s_num))
        s_high = np.minimum(fine_size, np.ceil(s_num))

        # Find dimensions where we have hit the ceiling
        hit_ceil = np.squeeze(np.argwhere(np.logical_and(s_high == fine_size, ~found)))
        # These have a bound, and will have their leeway removed
        optimum[hit_ceil] = s_high[hit_ceil]
        found[hit_ceil] = True

        # If the ceiling was hit in some dimension, we have to start over
        # again.
        if np.any(hit_ceil):
            continue

        # There is no room for variations in found cells
        s_low[found] = optimum[found]
        s_high[found] = optimum[found]

        # Array for storing the combinations.
        coarse_size = np.vstack((s_low, s_high))
        # The closest we've been to hit the target size. Set this to an
        # unrealistically high number
        dist = fine_size.prod()

        # Loop over all combinations of rounding up and down, and test if we
        # are closer to the target number.
        for perm in pp.permutations.multinary_permutations(2, nd):
            size_now = np.zeros(nd)
            for i, bit in enumerate(perm):
                size_now[i] = coarse_size[bit, i]
            if np.abs(target - size_now.prod()) < dist:
                dist = target - size_now.prod()
                optimum = size_now

        # All dimensions that may hit the ceiling have been found, and we have
        # the optimum solution. Declare victory and return home.
        found[:] = True

    if it_counter > nd:
        raise ValueError(
            "Maximum number of iterations exceeded. There is a \
                         bug somewhere."
        )

    return optimum.astype("int")


def extract_subgrid(
    g: pp.Grid,
    c: np.ndarray,
    sort: bool = True,
    faces: bool = False,
    is_planar: bool = True,
) -> Tuple[pp.Grid, np.ndarray, np.ndarray]:
    """
    Extract a subgrid based on cell/face indices.

    For simplicity the cell/face indices will be sorted before the subgrid is
    extracted.

    If the parent grid has geometry attributes (cell centers etc.) these are
    copied to the child.

    No checks are done on whether the cells/faces form a connected area. The
    method should work in theory for non-connected cells, the user will then
    have to decide what to do with the resulting grid. This option has however
    not been tested.

    Parameters:
        g (core.grids.Grid): Grid object, parent
        c (np.array, dtype=int): Indices of cells to be extracted
        sort=True (bool): If true c is sorted
        faces=False (bool): If true c are intrepetred as faces, and the
                            exptracted grid will be a lower dimensional grid
                            defined by the these faces
        is_planar: (optional) defaults to True. Only used when extracting faces from a
           3d grid. If True the faces f must be planar. Set to False to use this
           function for extracting a non-planar 2D grid, but use at own risk.

    Returns:
        Grid: Extracted subgrid. Will share (note, *not* copy)
            geometric fields with the parent grid. Also has an additional
            field parent_cell_ind giving correspondance between parent and
            child cells.
        np.ndarray, dtype=int: Index of the extracted faces, ordered so that
            element i is the global index of face i in the subgrid.
        np.ndarray, dtype=int: Index of the extracted nodes, ordered so that
            element i is the global index of node i in the subgrid.

    Raises:
        IndexError if index is as boolean and do not match the array size.

    """
    if np.asarray(c).dtype == "bool":
        # convert to indices.
        # First check for dimension
        if faces and c.size != g.num_faces:
            raise IndexError("boolean index did not match number of faces")
        elif not faces and c.size != g.num_cells:
            raise IndexError("boolean index did not match number of cells")
        c = np.where(c)[0]

    if sort:
        c = np.sort(np.atleast_1d(c))

    if faces:
        return __extract_cells_from_faces(g, c, is_planar)
    # Local cell-face and face-node maps.
    cf_sub, unique_faces = __extract_submatrix(g.cell_faces.tocsc(), c)
    fn_sub, unique_nodes = __extract_submatrix(g.face_nodes.tocsc(), unique_faces)

    # Append information on subgrid extraction to the new grid's history
    name = list(g.name)
    name.append("Extract subgrid")

    # Construct new grid.
    h = pp.Grid(g.dim, g.nodes[:, unique_nodes], fn_sub, cf_sub, name)

    # Copy geometric information if any
    if hasattr(g, "cell_centers"):
        h.cell_centers = g.cell_centers[:, c]
    if hasattr(g, "cell_volumes"):
        h.cell_volumes = g.cell_volumes[c]
    if hasattr(g, "face_centers"):
        h.face_centers = g.face_centers[:, unique_faces]
    if hasattr(g, "face_normals"):
        h.face_normals = g.face_normals[:, unique_faces]
    if hasattr(g, "face_areas"):
        h.face_areas = g.face_areas[unique_faces]

    h.parent_cell_ind = c

    return h, unique_faces, unique_nodes


def __extract_submatrix(mat, ind):
    """ From a matrix, extract the column specified by ind. All zero columns
    are stripped from the sub-matrix. Mappings from global to local row numbers
    are also returned.
    """
    if mat.getformat() != "csc":
        raise ValueError("To coulums from a matrix it must be csc")
    sub_mat = pp.utils.sparse_mat.slice_mat(mat, ind)
    cols = sub_mat.indptr
    data = sub_mat.data
    unique_rows, rows_sub = np.unique(sub_mat.indices, return_inverse=True)
    shape = (unique_rows.size, cols.size - 1)
    return sps.csc_matrix((data, rows_sub, cols), shape), unique_rows


def __extract_cells_from_faces(g, f, is_planar):
    """
    Extracting a lower-dimensional grid from the fraces of the higher
    dimensional grid g. See extract_subgrid.
    """
    if g.dim == 3:
        return __extract_cells_from_faces_3d(g, f, is_planar)
    elif g.dim == 2:
        return __extract_cells_from_faces_2d(g, f)
    elif g.dim == 1:
        return __extract_cells_from_faces_1d(g, f)
    else:
        raise NotImplementedError("can only create a subgrid for dimension 1, 2 and 3")


def __extract_cells_from_faces_1d(g, f):
    assert np.size(f) == 1
    node = np.argwhere(g.face_nodes[:, f])[:, 0]
    h = pp.PointGrid(g.nodes[:, node])
    h.compute_geometry()
    return h, f, node


def __extract_cells_from_faces_2d(g, f):
    # Local cell-face and face-node maps.
    cell_nodes, unique_nodes = __extract_submatrix(g.face_nodes, f)

    cell_faces_indices = cell_nodes.indices
    data = -1 * cell_nodes.data
    _, ix = np.unique(cell_faces_indices, return_index=True)
    data[ix] *= -1

    cell_faces = sps.csc_matrix((data, cell_faces_indices, cell_nodes.indptr))

    num_faces = np.shape(cell_faces)[0]
    num_nodes = np.shape(cell_nodes)[0]
    num_nodes_per_face = np.ones(num_nodes)

    face_node_ind = pp.utils.matrix_compression.rldecode(
        np.arange(num_faces), num_nodes_per_face
    )

    face_nodes = sps.coo_matrix(
        (np.ones(num_nodes, dtype=bool), (np.arange(num_faces), face_node_ind))
    ).tocsc()

    # Append information on subgrid extraction to the new grid's history
    name = list(g.name)
    name.append("Extract subgrid")

    h = pp.Grid(g.dim - 1, g.nodes[:, unique_nodes], face_nodes, cell_faces, name)

    h.compute_geometry()

    assert np.all(np.isclose(g.face_areas[f], h.cell_volumes))
    h.cell_volumes = g.face_areas[f]
    assert np.all(np.isclose(g.face_centers[:, f], h.cell_centers))
    h.cell_centers = g.face_centers[:, f]

    h.parent_face_ind = f
    return h, f, unique_nodes


def __extract_cells_from_faces_3d(g, f, is_planar=True):
    """
    Extract a 2D grid from the faces of a 3D grid. One of the uses of this function
    is to obtain a 2D MortarGrid from the boundary of a 3D grid. The faces f
    is by default assumed to be planar, however, this is mainly because compute_geometry
    does not handle non-planar grids. compute_geometry is used to do a sanity check
    of the extracted grid. if is_planar is set to False, this function should handle
    non-planar grids, however, this has not been tested thouroghly, and it does not
    performe the geometric sanity checks.

    Parameters:
    ----------
    g: (Grid) A 3d grid
    f: (ndarray) faces of the 3d grid to by used as cells in the 2d grid
    is_planar: (optional) defaults to True: If False the faces f must be planar.
        Set to False to use this function for a non-planar 2D grid, but use at own risk

    Returns:
    -------
    (Grid) A 2d grid extracted from the 3d grid g and the faces f.

    Throws:
    ------
    ValueError: If the given faces is not planar and is_planar==True
    """
    # Local cell-face and face-node maps.
    cell_nodes, unique_nodes = __extract_submatrix(g.face_nodes, f)
    if is_planar and not pp.geometry_property_checks.points_are_planar(
        g.nodes[:, unique_nodes]
    ):
        raise ValueError("The faces extracted from a 3D grid must be planar")
    num_cell_nodes = cell_nodes.nnz

    cell_node_ptr = cell_nodes.indptr
    num_nodes_per_cell = cell_node_ptr[1:] - cell_node_ptr[:-1]

    next_node = np.arange(num_cell_nodes) + 1
    next_node[cell_node_ptr[1:] - 1] = cell_node_ptr[:-1]
    face_start = cell_nodes.indices
    face_end = cell_nodes.indices[next_node]
    face_nodes_indices = np.vstack((face_start, face_end))
    face_nodes_sorted = np.sort(face_nodes_indices, axis=0)
    _, IA, IC = np.unique(
        face_nodes_sorted, return_index=True, return_inverse=True, axis=1
    )

    face_nodes_indices = face_nodes_indices[:, IA].ravel("F")
    num_face_nodes = face_nodes_indices.size
    face_nodes_indptr = np.arange(0, num_face_nodes + 1, 2)
    face_nodes = sps.csc_matrix(
        (np.ones(num_face_nodes), face_nodes_indices, face_nodes_indptr)
    )

    cell_idx = pp.utils.matrix_compression.rldecode(
        np.arange(num_face_nodes), num_nodes_per_cell
    )

    data = np.ones(IC.shape)
    _, ix = np.unique(IC, return_index=True)
    data[ix] *= -1

    cell_faces = sps.coo_matrix((data, (IC, cell_idx))).tocsc()

    # Append information on subgrid extraction to the new grid's history
    name = list(g.name)
    name.append("Extract subgrid")

    h = pp.Grid(g.dim - 1, g.nodes[:, unique_nodes], face_nodes, cell_faces, name)

    if is_planar:
        # We could now just copy the corresponding geometric values from g to h, but we
        # run h.compute_geometry() to check if everything went ok.
        h.compute_geometry()
        if not np.all(np.isclose(g.face_areas[f], h.cell_volumes)):
            raise AssertionError(
                """Somethign went wrong in extracting subgrid. Face area of
            higher dim is not equal face centers of lower dim grid"""
            )
        if not np.all(np.isclose(g.face_centers[:, f], h.cell_centers)):
            raise AssertionError(
                """Somethign went wrong in extracting subgrid. Face centers
            of higher dim is not equal cell centers of lower dim grid"""
            )
    h.cell_volumes = g.face_areas[f]
    h.cell_centers = g.face_centers[:, f]

    h.parent_face_ind = f
    return h, f, unique_nodes


def partition_grid(
    g: pp.Grid, ind: np.ndarray
) -> Tuple[List[pp.Grid], List[np.ndarray], List[np.ndarray]]:
    """
    Partition a grid into multiple subgrids based on an index set.

    No tests are made on whether the resulting grids are connected.

    Example:
        >>> g = pp.CartGrid(np.array([10, 10]))
        >>> p = partition_structured(g, num_part=4)
        >>> subg, face_map, node_map = partition_grid(g, p)

    Parameters:
        g (core.grids.grid): Global grid to be partitioned
        ind (np.array): Partition vector, one per cell. Should be 0-offset.

    Returns:
        list: List of grids, each element representing a grid.
        list of np.arrays: Each element contains the global indices of the
            local faces.
        list of np.arrays: Each element contains the global indices of the
            local nodes.

    """

    sub_grid: List[pp.Grid] = []
    face_map_list: List[np.ndarray] = []
    node_map_list: List[np.ndarray] = []
    for i in np.unique(ind):
        ci = np.squeeze(np.argwhere(ind == i))
        sg, fm, nm = extract_subgrid(g, ci)
        sub_grid.append(sg)
        face_map_list.append(fm)
        node_map_list.append(nm)

    return sub_grid, face_map_list, node_map_list


def overlap(
    g: pp.Grid, cell_ind: np.ndarray, num_layers: int, criterion: str = "node"
) -> np.ndarray:
    """
    From a set of cell indices, find an extended set of cells that form an
    overlap (in the domain decomposition sense).

    The cell set is increased by including all cells that share at least one
    node with the existing set. When multiple layers are asked for, this
    process is repeated.

    The definition of neighborship is specified by criterion. Possible options
    are 'face' (each layer will add cells that share a face with the active
    face set), or 'node' (each layer will add cells sharing a vertex with the
    active set).

    Parameters:
        g (core.grids.grid): The grid; the cell-node relation will be used to
            extend the cell set.
        cell_ind (np.array): Cell indices, the initial cell set.
        num_layers (int): Number of overlap layers.
        criterion (str, optional): Which definition of neighborship to apply.
            Should be either 'face' or 'node'. Default is 'node'.

    Returns:
        np.array: Indices of the extended cell set.

    Examples:
        >>> g = pp.CartGrid([5, 5])
        >>> ci = np.array([0, 1, 5, 6])
        >>> overlap(g, ci, 1)
        array([ 0,  1,  2,  5,  6,  7, 10, 11, 12])

    """

    # Boolean storage of cells in the active set; these are the ones that will
    # be in the overlap
    active_cells = np.zeros(g.num_cells, dtype=np.bool)
    # Initialize by the specified cells
    active_cells[cell_ind] = 1

    if criterion.lower().strip() == "node":
        # Construct cell-node map, its transpose will be a node-cell map
        cn = g.cell_nodes()

        # Also introduce active nodes
        active_nodes = np.zeros(g.num_nodes, dtype=np.bool)

        # Gradually increase the size of the cell set
        for _ in range(num_layers):
            # Nodes are found via the mapping
            active_nodes[np.squeeze(np.where((cn * active_cells) > 0))] = 1
            # Map back to new cells
            ci_new = np.squeeze(np.where((cn.transpose() * active_nodes) > 0))
            # Activate new cells.
            active_cells[ci_new] = 1

    elif criterion.lower().strip() == "face":
        # Create a version of g.cell_faces with only positive values for
        # connections, e.g. let go of the divergence property
        cf = g.cell_faces
        # This avoids overwriting data in cell_faces.
        data = np.ones_like(cf.data)
        cf = sps.csc_matrix((data, cf.indices, cf.indptr))

        active_faces = np.zeros(g.num_faces, dtype=np.bool)

        # Gradually increase the size of the cell set
        for _ in range(num_layers):
            # All faces adjacent to an active cell
            active_faces[np.squeeze(np.where((cf * active_cells) > 0))] = 1
            # Map back to active cells, including that on the other side of the
            # newly found faces
            ci_new = np.squeeze(np.where((cf.transpose() * active_faces) > 0))
            # Activate new cells
            active_cells[ci_new] = 1

    # Sort the output, this should not be a disadvantage
    return np.sort(np.squeeze(np.argwhere(active_cells > 0)))


def grid_is_connected(
    g: pp.Grid, cell_ind: np.ndarray = None
) -> Tuple[bool, List[np.ndarray]]:
    """
    Check if a grid is fully connected, as defined by its cell_connection_map().

    The function is intended used in one of two ways:
        1) To test if a subgrid will be connected before it is extracted. In
        this case, the cells to be tested is specified by cell_ind.
        2) To check if an existing grid is composed of a single component. In
        this case, all cells are should be included in the analyzis.

    Parameters:
        g (core.grids.grid): Grid to be tested. Only its cell_faces map is
            used.
        cell_ind (np.array): Index of cells to be included when looking for
            connections. Defaults to all cells in the grid.

    Returns:
        boolean: True if the grid is connected.
        list of np.arrays: Each list item contains a np.array with cell indices
            of a connected component.

    Examples:
        >>> g = pp.CartGrid(np.array([2, 2]))
        >>> p = np.array([0, 1])
        >>> is_con, l = grid_is_connected(g, p)
        >>> is_con
        True

        >>> g = pp.CartGrid(np.array([2, 2]))
        >>> p = np.array([0, 3])
        >>> is_con, l = grid_is_connected(g, p)
        >>> is_con
        False

    """
    import networkx

    # If no cell indices are specified, we use them all.
    if cell_ind is None:
        cell_ind = np.arange(g.num_cells)

    # Get connection map for the full grid
    c2c = g.cell_connection_map()

    # Extract submatrix of the active cell set.
    # To slice the sparse matrix, we first convert to row storage, slice rows,
    # and then convert to columns and slice those as well.
    c2c = c2c.tocsr()[cell_ind, :].tocsc()[:, cell_ind]

    # Represent the connections as a networkx graph and check for connectivity
    graph = networkx.from_scipy_sparse_matrix(c2c)
    is_connected = networkx.is_connected(graph)

    # Get the connected components of the network.
    # networkx gives an generator that produce sets of node indices. Use this
    # to define a list of numpy arrays.
    component_generator = networkx.connected_components(graph)
    components = [np.array(list(i)) for i in component_generator]

    return is_connected, components
