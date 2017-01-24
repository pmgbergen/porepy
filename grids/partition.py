"""
Module for partitioning of grids based on various methods.

Intended support is by Cartesian indexing, and METIS-based.

"""
import pymetis
import numpy as np
import scipy.sparse as sps
import networkx

from core.grids.grid import Grid
from core.grids import structured
from utils import permutations


def partition_metis(g, num_part):
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

    # Connection map between cells
    c2c = g.cell_connection_map()

    # Convert the cells into the format required by pymetis
    adjacency_list = [c2c.getrow(i).indices for i in range(c2c.shape[0])]
    # Call pymetis
    part = pymetis.part_graph(10, adjacency=adjacency_list)

    # The meaning of the first number returned by pymetis is not clear (poor
    # documentation), only return the partitioning.
    return np.array(part[1])


def partition_structured(g, coarse_dims=None, num_part=None):
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
        raise ValueError('Either coarse dimensions or number of coarse cells \
                         must be specified')

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
        incr_ind = np.arange(0, fine_dims[i], fine_per_coarse[i], dtype='i')

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
        glob_dims = (xi + yi * coarse_dims[0]).ravel('C')
    elif nd == 3:
        xi, yi, zi = np.meshgrid(ind[0], ind[1], ind[2])
        # Combine indices, with appropriate jumps in y and z counting
        glob_dims = (xi + yi * coarse_dims[0]
                   + zi * np.prod(coarse_dims[:2]))
        # This just happened to work, may be logical, but the documentanion of
        # np.meshgrid was hard to comprehend.
        glob_dims = np.swapaxes(np.swapaxes(glob_dims, 1, 2), 0, 1).ravel('C')

    # Return an int
    return glob_dims.astype('int')


def determine_coarse_dimensions(target, fine_size):
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
        fine_size (np.ndarray): Number of fine-scale cell in eac dimension

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
        s_num = np.power(target_now, 1/(nd - found.sum()))
        s_low = np.maximum(np.ones(nd), np.floor(s_num))
        s_high = np.minimum(fine_size, np.ceil(s_num))

        # Find dimensions where we have hit the ceiling
        hit_ceil = np.squeeze(np.argwhere(np.logical_and(s_high == fine_size,
                                                         ~found)))
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
        for perm in permutations.multinary_permutations(2, nd):
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
        raise ValueError('Maximum number of iterations exceeded. There is a \
                         bug somewhere.')

    return optimum


def extract_subgrid(g, c, sort=True):
    """
    Extract a subgrid based on cell indices.

    For simplicity the cell indices will be sorted before the subgrid is
    extracted.

    If the parent grid has geometry attributes (cell centers etc.) these are
    copied to the child.

    No checks are done on whether the cells form a connected area. The method
    should work in theory for non-connected cells, the user will then have to
    decide what to do with the resulting grid. This option has however not been
    tested.

    Parameters:
        g (core.grids.Grid): Grid object, parent
        c (np.array, dtype=int): Indices of cells to be extracted

    Returns:
        core.grids.Grid: Extracted subgrid
        np.ndarray, dtype=int: Index of the extracted faces, ordered so that
            element i is the global index of face i in the subgrid.
        np.ndarray, dtype=int: Index of the extracted nodes, ordered so that
            element i is the global index of node i in the subgrid.
    """
    if sort:
        c = np.sort(c)

    # Local cell-face and face-node maps.
    cf_sub, unique_faces = __extract_submatrix(g.cell_faces, c)
    fn_sub, unique_nodes = __extract_submatrix(g.face_nodes, unique_faces)

    # Append information on subgrid extraction to the new grid's history
    name = list(g.name)
    name.append('Extract subgrid')

    # Construct new grid.
    h = Grid(g.dim, g.nodes[:, unique_nodes], fn_sub, cf_sub, name)

    # Copy geometric information if any
    if hasattr(g, 'cell_centers'):
        h.cell_centers = g.cell_centers[:, c]
    if hasattr(g, 'cell_volumes'):
        h.cell_volumes = g.cell_volumes[c]
    if hasattr(g, 'face_centers'):
        h.face_centers = g.face_centers[:, unique_faces]
    if hasattr(g, 'face_normals'):
        h.face_normals = g.face_normals[:, unique_faces]
    if hasattr(g, 'face_areas'):
        h.face_areas = g.face_areas[unique_faces]

    return h, unique_faces, unique_nodes


def __extract_submatrix(mat, ind):
    """ From a matrix, extract the column specified by ind. All zero columns
    are stripped from the sub-matrix. Mappings from global to local row numbers
    are also returned.
    """
    sub_mat = mat[:, ind]
    cols = sub_mat.indptr
    rows = sub_mat.indices
    data = sub_mat.data
    unique_rows, rows_sub = np.unique(sub_mat.indices,
                                      return_inverse=True)
    return sps.csc_matrix((data, rows_sub, cols)), unique_rows


def partition_grid(g, ind):
    """
    Partition a grid into multiple subgrids based on an index set.

    No tests are made on whether the resulting grids are connected.

    Example:
        >>> g = structured.CartGrid(np.array([10, 10]))
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

    sub_grid = []
    face_map_list = []
    node_map_list = []
    for i in np.unique(ind):
        ci = np.squeeze(np.argwhere(ind == i))
        sg, fm, nm = extract_subgrid(g, ci)
        sub_grid.append(sg)
        face_map_list.append(fm)
        node_map_list.append(nm)

    return sub_grid, face_map_list, node_map_list


def overlap(g, cell_ind, num_layers):
    """
    From a set of cell indices, find an extended set of cells that form an
    overlap (in the domain decomposition sense).

    The cell set is increased by including all cells that share at least one
    node with the existing set. When multiple layers are asked for, this
    process is repeated.

    It should be possible to define other rules for overlap, such as based on
    cell-face mappings by changing the connection matrix cn below.

    Parameters:
        g (core.grids.grid): The grid; the cell-node relation will be used to
            extend the cell set.
        cell_ind (np.array): Cell indices, the initial cell set.
        num_layers (int): Number of overlap layers.

    Returns:
        np.array: Indices of the extended cell set.

    Examples:
        >>> g = structured.CartGrid([5, 5])
        >>> ci = np.array([0, 1, 5, 6])
        >>> overlap(g, ci, 1)
        array([ 0,  1,  2,  5,  6,  7, 10, 11, 12])

    """

    # Construct cell-node map, its transpose will be a node-cell map
    cn = g.cell_nodes()

    # Boolean storage of cells and nodes in the active set
    active_cells = np.zeros(g.num_cells, dtype=np.bool)
    active_nodes = np.zeros(g.num_nodes, dtype=np.bool)
    active_cells[cell_ind] = 1

    # Gradually increase the size of the cell set
    for i in range(num_layers):
        # Nodes are found via the mapping 
        active_nodes[np.squeeze(np.where((cn * active_cells) > 0))] = 1
        # Map back to new cells
        ci_new = np.squeeze(np.where((cn.transpose() * active_nodes) > 0))
        # Activate new cells.
        active_cells[ci_new] = 1

    # Sort the output, this should not be a disadvantage
    return np.sort(np.squeeze(np.argwhere(active_cells > 0)))

#----------------------------------------------------------------------------#

def grid_is_connected(g, cell_ind=None):
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
        >>> g = structured.CartGrid(np.array([2, 2]))
        >>> p = np.array([0, 1])
        >>> is_con, l = grid_is_connected(g, p)
        >>> is_con
        True    

        >>> g = structured.CartGrid(np.array([2, 2]))
        >>> p = np.array([0, 3])
        >>> is_con, l = grid_is_connected(g, p)
        >>> is_con
        False

    """

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

