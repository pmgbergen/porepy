"""
Module to increase the dimensions of grids by extrusion in the z-direction.

Both individual grids and mixed-dimensional grid_buckets can be extruded. The
dimension of the highest-dimensional grid should be 2 at most.

The main methods in the module are

    extrude_grid_bucket()
    extrude_grid()

All other functions are helpers.

"""
from collections import namedtuple
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.grids import mortar_grid

module_sections = ["grids", "gridding"]


@pp.time_logger(sections=module_sections)
def extrude_grid_bucket(gb: pp.GridBucket, z: np.ndarray) -> Tuple[pp.GridBucket, Dict]:
    """Extrude a GridBucket by extending all fixed-dimensional grids in the z-direction.

    In practice, the original grid bucket will be 2d, and the result is 3d.

    The returned GridBucket is fully functional, including mortar grids on the gb edges.
    The data dictionaries on nodes and edges are mainly empty. Data can be transferred from
    the original GridBucket via the returned map between old and new grids.

    Parameters:
        gb (pp.GridBukcet): Mixed-dimensional grid to be extruded. Should be 2d.
        z (np.ndarray): z-coordinates of the nodes in the extruded grid. Should be
            either non-negative or non-positive, and be sorted in increasing or
            decreasing order, respectively.

    Returns:
        gb (pp.GridBucket): Mixed-dimensional grid, 3d. The data dictionaries on nodes and
            edges are mostly empty.
        dict: Mapping from individual grids in the old bucket to the corresponding
            extruded grids in the new one. The dictionary values are a namedtuple with
            elements grid (new grid), cell_map and face_map, where the two latter
            describe mapping between the new and old grid, see extrude_grid for details.

    """

    # New GridBucket. to be filled in
    gb_new: pp.GridBucket = pp.GridBucket()

    # Data structure for mapping between old and new grids
    g_map = {}

    # Container for grid information
    Mapping = namedtuple("Mapping", ["grid", "cell_map", "face_map"])

    # Loop over all grids in the old bucket, extrude the grid, save mapping information
    for g, _ in gb:
        g_new, cell_map, face_map = extrude_grid(g, z)

        if hasattr(g, "frac_num"):
            g_new.frac_num = g.frac_num

        gb_new.add_nodes([g_new])

        g_map[g] = Mapping(g_new, cell_map, face_map)

    # Loop over all edges in the old grid, create corresponding edges in the new gb.
    # Also define mortar_grids
    for e, d in gb.edges():

        # grids of the old edge, extruded version of each grid
        gl, gh = gb.nodes_of_edge(e)
        gl_new = g_map[gl].grid
        gh_new = g_map[gh].grid

        # Next, we need the cell-face mapping for the new grid.
        # The idea is to first find the old map, then replace each cell-face relation
        # with the set of cells and faces (exploiting first that the new grids are
        # matching due to the extrusion algorithm, and second that the cell-map and
        # face-map stores indices in increasing layer index, so that the first cell
        # and first face both are in the first layer, thus they match, etc.).
        face_cells_old = d["face_cells"]

        # cells (in low-dim grid) and faces in high-dim grid that define the same
        # geometric quantity
        cells, faces, _ = sps.find(face_cells_old)

        # Cell-map for the low-dimensional grid, face-map for the high-dim
        cell_map = g_map[gl].cell_map
        face_map = g_map[gh].face_map

        # Data structure for the new face-cell map
        rows = np.empty(0, dtype=int)
        cols = np.empty(0, dtype=int)

        # The standard MortarGrid __init__ assumes that when faces are split because of
        # a fracture, the faces are ordered with one side first, then the other. This
        # will not be True for this layered construction. Instead, keep track of all
        # faces that should be moved to the other side.
        face_on_other_side = np.empty(0, dtype=int)

        # Loop over cells in gl would not have been as clean, as each cell is associated
        # with faces on both sides
        # Faces are found from the high-dim grid, cells in the low-dim grid
        for idx in range(faces.size):
            rows = np.hstack((rows, cell_map[cells[idx]]))
            cols = np.hstack((cols, face_map[faces[idx]]))

            # Here, we tacitly assume that the original grid had its faces split in the
            # standard way, that is, all faces on one side have index lower than any
            # face on the other side.
            if faces[idx] > np.median(faces):
                face_on_other_side = np.hstack(
                    (face_on_other_side, face_map[faces[idx]])
                )

        data = np.ones(rows.size, dtype=bool)
        # Create new face-cell map
        face_cells_new = sps.coo_matrix(
            (data, (rows, cols)), shape=(gl_new.num_cells, gh_new.num_faces)
        ).tocsc()

        # Define the new edge
        new_edge: List[pp.Grid] = [gh_new, gl_new]
        # Add to new gb, together with the new face-cell map
        gb_new.add_edge(new_edge, face_cells_new)

        # Create a mortar grid, add to data of new edge
        if len(face_on_other_side) == 0:  # Only one side
            side_g = {
                mortar_grid.MortarSides.LEFT_SIDE: gl_new.copy(),
            }
        else:
            side_g = {
                mortar_grid.MortarSides.LEFT_SIDE: gl_new.copy(),
                mortar_grid.MortarSides.RIGHT_SIDE: gl_new.copy(),
            }

        # Construct mortar grid, with instructions on which faces belong to which side
        mg = pp.MortarGrid(
            gl_new.dim, side_g, face_cells_new, face_duplicate_ind=face_on_other_side
        )

        d_new = gb_new.edge_props((gh_new, gl_new))

        d_new["mortar_grid"] = mg

    return gb_new, g_map


@pp.time_logger(sections=module_sections)
def extrude_grid(g: pp.Grid, z: np.ndarray) -> Tuple[pp.Grid, np.ndarray, np.ndarray]:
    """Increase the dimension of a given grid by 1, by extruding the grid in the
    z-direction.

    The original grid is assumed to be in the xy-plane, that is, any existing non-zero
    z-direction is ignored.

    Both the original and the new grid will have their geometry computed.

    Parameters:
        g (pp.Grid): Original grid to be extruded. Should have dimension 0, 1 or 2.
        z (np.ndarray): z-coordinates of the nodes in the extruded grid. Should be
            either non-negative or non-positive, and be sorted in increasing or
            decreasing order, respectively.

    Returns:
        pp.Grid: A grid of dimension g.dim + 1.
        np.array of np.arrays: Cell mappings, so that element ci gives all indices of
            cells in the extruded grid that comes from cell ci in the original grid.
        np.array of np.arrays: Face mappings, so that element fi gives all indices of
            faces in the extruded grid that comes from face fi in the original grid.

    Raises:
        ValueError: If the z-coordinates for nodes contain both positive and negative
            values.
        ValueError: If a 3d grid is provided for extrusion.

    """
    if not (np.all(z >= 0) or np.all(z <= 0)):
        raise ValueError("Extrusion should be in either positive or negative direction")

    if g.dim == 0:
        return _extrude_0d(g, z)  # type: ignore
    elif g.dim == 1:
        return _extrude_1d(g, z)  # type: ignore
    elif g.dim == 2:
        return _extrude_2d(g, z)
    else:
        raise ValueError("The grid to be extruded should have dimension at most 2")


@pp.time_logger(sections=module_sections)
def _extrude_2d(g: pp.Grid, z: np.ndarray) -> Tuple[pp.Grid, np.ndarray, np.ndarray]:
    """Extrude a 2d grid into 3d by prismatic extension.

    The original grid is assumed to be in the xy-plane, that is, any existing non-zero
    z-direction is ignored.

    Both the original and the new grid will have their geometry computed.

    Parameters:
        g (pp.Grid): Original grid to be extruded. Should have dimension 2.
        z (np.ndarray): z-coordinates of the nodes in the extruded grid. Should be
            either non-negative or non-positive, and be sorted in increasing or
            decreasing order, respectively.

    Returns:
        pp.Grid: A grid of dimension 3.
        np.array of np.arrays: Cell mappings, so that element ci gives all indices of
            cells in the extruded grid that comes from cell ci in the original grid.
        np.array of np.arrays: Face mappings, so that element fi gives all indices of
            faces in the extruded grid that comes from face fi in the original grid.

    """

    g.compute_geometry()

    negative_extrusion = np.all(z <= 0)

    ## Bookkeeping of the number of grid items

    # Number of nodes in the z-direction
    num_node_layers = z.size
    # Number of cell layers, one less than the nodes
    num_cell_layers = num_node_layers - 1

    # Short hand for the number of cells in the 2d grid
    nc_2d = g.num_cells
    nf_2d = g.num_faces
    nn_2d = g.num_nodes

    # The number of nodes in the 3d grid is given by the number of 2d nodes, and the
    # number of node layers
    nn_3d = nn_2d * num_node_layers
    # The 3d cell count is similar to that for the nodes
    nc_3d = nc_2d * num_cell_layers
    # The number of faces is more intricate: In each layer of cells, there will be as
    # many faces as there is in the 2d grid. In addition, in the direction of extrusion
    # there will be one set of faces per node layer, with each layer containing as many
    # faces as there are cells in the 2d grid
    nf_3d = nf_2d * num_cell_layers + nc_2d * num_node_layers

    ## Nodes - only coorinades are needed
    # The nodes in the 2d grid are copied for all layers, with the z-coordinates changed
    # for each layer. This means that for a vertical pilar, the face-node and cell-node
    # relations can be inferred from that in the original 2d grid, with index increments
    # of size nn_2d
    x_layer = g.nodes[0]
    y_layer = g.nodes[1]

    nodes = np.empty((3, 0))
    # Stack the layers of nodes
    for zloc in z:
        nodes = np.hstack((nodes, np.vstack((x_layer, y_layer, zloc * np.ones(nn_2d)))))

    ## Face-node relations
    # The 3d grid has two types of faces: Those formed by faces in the 2d grid, termed
    # 'vertical' below, and those on the top and bottom of the 3d cells, termed
    # horizontal

    # Face-node relation for the 2d grid. We know there are exactly two nodes in each
    # 2d face.
    fn_2d = g.face_nodes.indices.reshape((2, g.num_faces), order="F")

    # Nodes of the faces for the bottom layer of 3d cells. These are formed by
    # connecting nodes in the bottom layer with those immediately above
    fn_layer = np.vstack((fn_2d[0], fn_2d[1], fn_2d[1] + nn_2d, fn_2d[0] + nn_2d))

    # For the vertical cells, the flux direction indicated in cell_face map will be
    # inherited from the 2d grid (see below). The normal vector, which should be
    # consistent with this value, is effectively computed from the ordering of the
    # face-node relation (and the same is true for several other geometric quantities).
    # This requires that the face-nodes are sorted in a CCW order when seen from the
    # side of a positive cell_face value. To sort this out, we need to flip some of the
    # columns in fn_layer

    # Faces, cells and values of the 2d cell-face map
    [fi, ci, sgn] = sps.find(g.cell_faces)
    # Only consider each face once
    _, idx = np.unique(fi, return_index=True)

    # The node ordering in fn_layer will be CCW seen from cell ci if the cell center of
    # ci is CW relative to the line from the first to the second node of the 2d cell.
    #
    # Example: with p0 = [0, 0, 0], p1 = [1, 0, 0], the 3d face will have further nodes
    #               p2 = [1, 0, 1], p3 = [0, 0, 1].
    # This will be counterclockwise to a 2d cell center of, say, [0.5, -0.5, 0],
    #  (which is CW relative to p0 and p1)
    #
    p0 = g.nodes[:, fn_2d[0, fi[idx]]]
    p1 = g.nodes[:, fn_2d[1, fi[idx]]]
    pc = g.cell_centers[:, ci[idx]]
    ccw_2d = pp.geometry_property_checks.is_ccw_polyline(p0, p1, pc)

    # We should flip those columns in fn_layer where the sign is positive, and the 2d
    # is not ccw (meaning the 3d will be). Similarly, also flip negative signs and 2d
    # ccw.
    flip = np.logical_or(
        np.logical_and(sgn[idx] > 0, np.logical_not(ccw_2d)),
        np.logical_and(sgn[idx] < 0, ccw_2d),
    )

    # Finally, if the extrusion is in the negative direction, the ordering of all
    # face-node relations is the oposite of that indicated above.
    if negative_extrusion:
        flip = np.logical_not(flip)

    fn_layer[:, flip] = fn_layer[np.array([1, 0, 3, 2])][:, flip]

    # The face-node relation for the vertical cells are found by stacking those in the
    # bottom layer, with an appropriate offset. This also implies that the vertical
    # faces of a cell in layer k are the same as the faces of the corresponding 2d cell,
    # with the appropriate adjustments for the number of faces and cells in each layer
    fn_rows_vertical = np.empty((4, 0))
    # Loop over all layers of cells
    for k in range(num_cell_layers):
        fn_rows_vertical = np.hstack((fn_rows_vertical, fn_layer + nn_2d * k))

    # Reshape the node indices into a single array
    fn_rows_vertical = fn_rows_vertical.ravel("F")

    # All vertical faces have exactly four nodes
    nodes_per_face_vertical = 4
    # Aim for a csc-representation of the faces. Column pointers
    fn_cols_vertical = np.arange(
        0, nodes_per_face_vertical * nf_2d * num_cell_layers, nodes_per_face_vertical
    )

    # Next, deal with the horizontal faces. The face-node relation is based on the
    # cell-node relation of the 2d grid.
    # The structure of this constrution is a bit more involved than for the vertical
    # faces, since the 2d cells have an unknown, and generally varying, number of nodes
    cn_2d = g.cell_nodes()

    # Short hand for node indices of each cell.
    cn_ind_2d = cn_2d.indices.copy()

    # Similar to the vertical faces, the face-node relation in 3d should match the
    # sign in the cell-face relation, so that the generated normal vector points out of
    # the cell with cf-value 1.
    # This requires a sorting of the nodes for each cell
    for ci in range(nc_2d):
        # Node indices of this 2d cell
        start = cn_2d.indptr[ci]
        stop = cn_2d.indptr[ci + 1]
        ni = cn_ind_2d[start:stop]

        coord = g.nodes[:2, ni]
        # Sort the points.
        # IMPLEMENTATION NOTE: this probably assumes convexity of the 2d cell.
        sort_ind = pp.utils.sort_points.sort_point_plane(
            np.vstack((coord, np.zeros(coord.shape[1]))),
            g.cell_centers[:, ci].reshape((-1, 1)),
        )
        # Indices that sort the nodes. The sort function contains a rotation, which
        # implies that it is unknown whether the ordering is cw or ccw
        # If the sorted points are ccw, we store them, unless the extrusion is negative
        # in which case the ordering should be cw, and the points are turned.
        if pp.geometry_property_checks.is_ccw_polygon(coord[:, sort_ind]):
            if negative_extrusion:
                cn_ind_2d[start:stop] = cn_ind_2d[start:stop][sort_ind[::-1]]
            else:
                cn_ind_2d[start:stop] = cn_ind_2d[start:stop][sort_ind]
        # Else, the ordering should be negative.
        elif pp.geometry_property_checks.is_ccw_polygon(coord[:, sort_ind[::-1]]):
            if negative_extrusion:
                cn_ind_2d[start:stop] = cn_ind_2d[start:stop][sort_ind]
            else:
                cn_ind_2d[start:stop] = cn_ind_2d[start:stop][sort_ind[::-1]]
        else:
            raise ValueError("this should not happen. Is the cell non-convex??")

    # Compressed column storage for horizontal faces: Store node indices
    fn_rows_horizontal = np.array([], dtype=int)
    # .. and pointers to the start of new faces
    fn_cols_horizontal = np.array(0, dtype=int)
    # Loop over all layers of nodes (one more than number of cells)
    # This means that the horizontal faces of a given cell is given by its index (bottom)
    # and its index + the number of 2d cells, both offset with the total number of
    # vertical faces
    for k in range(num_node_layers):
        # The horizontal cell-node relation for this layer is the bottom one, plus an
        # offset of the number of 2d nodes, per layer
        fn_rows_horizontal = np.hstack((fn_rows_horizontal, cn_ind_2d + nn_2d * k))
        # The index pointers are those of the 2d cell-node relation.
        # Adjustment for the vertical faces is done below
        # Drop the final element of the 2d indptr, which effectively signifies the end
        # of this array (we will add the corresponding element for the full array below)
        fn_cols_horizontal = np.hstack(
            (fn_cols_horizontal, cn_2d.indptr[1:] + cn_ind_2d.size * k)
        )

    # Add the final element which marks the end of the array
    # fn_cols_horizontal = np.hstack((fn_cols_horizontal, fn_rows_horizontal.size))
    # The horizontal faces are appended to the vertical ones. The node indices are the
    # same, but the face indices must be increased by the number of vertical faces
    num_vertical_faces = nf_2d * num_cell_layers
    fn_cols_horizontal += num_vertical_faces * nodes_per_face_vertical

    # Put together the vertical and horizontal data, create the face-node relation
    indptr = np.hstack((fn_cols_vertical, fn_cols_horizontal)).astype(int)
    indices = np.hstack((fn_rows_vertical, fn_rows_horizontal)).astype(int)
    data = np.ones(indices.size, dtype=int)

    # Finally, construct the face-node sparse matrix
    face_nodes = sps.csc_matrix((data, indices, indptr), shape=(nn_3d, nf_3d))

    ### Next the cell-faces.
    # Similar to the face-nodes, the easiest option is first to deal with the vertical
    # faces, which can be inferred directly from faces in the 2d grid, and then the
    # horizontal direction.
    # IMPLEMENTATION NOTE: Since all cells have both horizontal and vertical faces, and
    # these are found in separate operations, the easiest way to assemble the 3d
    # cell-face matrix is to construct information for a coo-matrix (not compressed
    # storage), and then convert later. This has some overhead, but the alternative
    # is to combine and sort the face indices in the horizontal and vertical components
    # so that all faces of any cell is stored together. This is most conveniently
    # left to scipy sparse .tocsc() function

    ## Vertical faces
    # For the vertical faces, the information from the 2d grid can be copied

    cf_rows_2d = g.cell_faces.indices
    cf_cols_2d = g.cell_faces.indptr
    cf_data_2d = g.cell_faces.data

    cf_rows_vertical = np.array([], dtype=int)
    # For the cells, we will store the number of facqes for each cell. This will later
    # be expanded to a full set of cell indices
    cf_vertical_cell_count = np.array([], dtype=int)
    cf_data_vertical = np.array([])

    for k in range(num_cell_layers):
        # The face indices are found from the 2d information, with increaments that
        # reflect how many layers of vertical faces there are below
        cf_rows_vertical = np.hstack((cf_rows_vertical, cf_rows_2d + k * nf_2d))
        # The diff here gives the number of faces per cell
        cf_vertical_cell_count = np.hstack(
            (cf_vertical_cell_count, np.diff(cf_cols_2d))
        )
        # The data is just plus and minus ones, no need to adjust
        cf_data_vertical = np.hstack((cf_data_vertical, cf_data_2d))

    # Expand information of the number of faces per cell into a corresponding full set
    # of cell indices
    cf_cols_vertical = pp.utils.matrix_compression.rldecode(
        np.arange(nc_3d), cf_vertical_cell_count
    )

    ## Horizontal faces
    # There is one set of faces per layer of nodes.
    # The cell_face relation will assign -1 to the upper cells, and +1 to lower cells.
    # This corresponds to normal vectors pointing upwards.
    # The bottom and top layers are special, in that they have only one neighboring
    # cell. All other layers have two (they are internal)

    # Bottom layer
    cf_rows_horizontal = num_vertical_faces + np.arange(nc_2d)
    cf_cols_horizontal = np.arange(nc_2d)
    cf_data_horizontal = -np.ones(nc_2d, dtype=int)

    # Intermediate layers, note
    for k in range(1, num_cell_layers):
        # Face indices are given twice, for the lower and upper neighboring cell
        # The offset of the face index is the number of vertical faces plus the number
        # of horizontal faces in lower layers
        rows_here = (
            num_vertical_faces
            + k * nc_2d
            + np.hstack((np.arange(nc_2d), np.arange(nc_2d)))
        )
        cf_rows_horizontal = np.hstack((cf_rows_horizontal, rows_here))

        # Cell indices, first of the lower layer, then of the upper
        cols_here = np.hstack(
            ((k - 1) * nc_2d + np.arange(nc_2d), k * nc_2d + np.arange(nc_2d))
        )
        cf_cols_horizontal = np.hstack((cf_cols_horizontal, cols_here))
        # Data: +1 for the lower cells, -1 for the upper
        data_here = np.hstack((np.ones(nc_2d), -np.ones(nc_2d)))
        cf_data_horizontal = np.hstack((cf_data_horizontal, data_here))

    # Top layer, with index offset for all other faces
    cf_rows_horizontal = np.hstack(
        (
            cf_rows_horizontal,
            num_vertical_faces + num_cell_layers * nc_2d + np.arange(nc_2d),
        )
    )
    # Similarly, the cell indices of the topbost layer
    cf_cols_horizontal = np.hstack(
        (cf_cols_horizontal, (num_cell_layers - 1) * nc_2d + np.arange(nc_2d))
    )
    # Only +1 in the data (oposite to lowermost layer)
    cf_data_horizontal = np.hstack((cf_data_horizontal, np.ones(nc_2d)))

    # Merge horizontal and vertical layers
    cf_rows = np.hstack((cf_rows_horizontal, cf_rows_vertical))
    cf_cols = np.hstack((cf_cols_horizontal, cf_cols_vertical))
    cf_data = np.hstack((cf_data_horizontal, cf_data_vertical))

    cell_faces = sps.coo_matrix(
        (cf_data, (cf_rows, cf_cols)), shape=(nf_3d, nc_3d)
    ).tocsc()

    tags = _define_tags(g, num_cell_layers)

    name = g.name.copy()
    name.append("Extrude 2d->3d")
    g_info = g.name.copy()
    g_info.append("Extrude 1d->2d")

    g_new = pp.Grid(3, nodes, face_nodes, cell_faces, g_info, external_tags=tags)
    g_new.compute_geometry()

    # Mappings between old and new cells and faces
    cell_map, face_map = _create_mappings(g, g_new, num_cell_layers)

    return g_new, cell_map, face_map


@pp.time_logger(sections=module_sections)
def _extrude_1d(
    g: pp.TensorGrid, z: np.ndarray
) -> Tuple[pp.Grid, np.ndarray, np.ndarray]:
    """Extrude a 1d grid into 2d by prismatic extension in the z-direction.

    The original grid is assumed to be in the xy-plane, that is, any existing non-zero
    z-direction is ignored.

    Both the original and the new grid will have their geometry computed.

    Parameters:
        g (pp.Grid): Original grid to be extruded. Should have dimension 1.
        z (np.ndarray): z-coordinates of the nodes in the extruded grid. Should be
            either non-negative or non-positive, and be sorted in increasing or
            decreasing order, respectively.

    Returns:
        pp.Grid: A grid of dimension 2.
        np.array of np.arrays: Cell mappings, so that element ci gives all indices of
            cells in the extruded grid that comes from cell ci in the original grid.
        np.array of np.arrays: Face mappings, so that element fi gives all indices of
            faces in the extruded grid that comes from face fi in the original grid.

    """
    # Number of cells in the grid
    num_cell_layers = z.size - 1

    # Node coordinates can be extruded in using a tensor product
    x = g.nodes[0]
    y = g.nodes[1]

    x_2d, z_2d = np.meshgrid(x, z)
    y_2d, _ = np.meshgrid(y, z)
    nodes = np.vstack((x_2d.ravel(), y_2d.ravel(), z_2d.ravel()))

    # Bookkeeping
    nn_old = g.num_nodes
    nc_old = g.num_cells
    nf_old = g.num_faces

    nn_new = g.num_nodes * (num_cell_layers + 1)
    nc_new = g.num_cells * num_cell_layers
    nf_new = g.num_faces * num_cell_layers + g.num_cells * (num_cell_layers + 1)

    fn_old = g.face_nodes.indices
    # Vertical faces are made by extruding old face-node relation
    fn_vert = np.empty((2, 0), dtype=int)
    for k in range(num_cell_layers):
        fn_this = k * nn_old + np.vstack((fn_old, nn_old + fn_old))
        fn_vert = np.hstack((fn_vert, fn_this))

    # Horizontal faces are defined from the old cell-node relation
    # Implementation note: This operation is much simpler than in the 2d-3d operation,
    # since much less is assumed on the ordering of face-nodes for 2d grids than 3d
    cn_old = g.cell_nodes().indices.reshape((2, g.num_cells), order="F")
    # Bottom layer
    fn_hor = cn_old
    # All other layers
    for k in range(num_cell_layers):
        fn_hor = np.hstack((fn_hor, cn_old + (k + 1) * nn_old))

    # Finalize the face-node map
    fn_rows = np.hstack((fn_vert, fn_hor))
    fn_cols = np.tile(np.arange(fn_rows.shape[1]), (2, 1))
    fn_data = np.ones(fn_cols.size, dtype=bool)

    fn = sps.coo_matrix(
        (fn_data, (fn_rows.ravel("F"), fn_cols.ravel("F"))), shape=(nn_new, nf_new)
    ).tocsc()

    # Next, cell-faces
    # We know there are exactly four faces for each cell
    cf_rows = np.empty((4, 0), dtype=int)
    cf_old = g.cell_faces.indices.reshape((2, -1), order="F")

    # Create vertical and horizontal faces together
    for k in range(num_cell_layers):
        # Vertical faces are identified by the old cell-face relation
        cf_vert_this = nn_old * k + cf_old

        # Put horizontal faces on top and bottom
        cf_hor_this = np.vstack((np.arange(nc_old), np.arange(nc_old) + nc_old))
        # Add an offset of the number of vertical faces in the grid + previous horizontal
        # faces
        cf_hor_this += nf_old * num_cell_layers + k * nc_old

        cf_rows = np.hstack((cf_rows, np.vstack((cf_vert_this, cf_hor_this))))

    # Finalize Cell-face relation
    cf_rows = cf_rows.ravel("F")
    cf_cols = np.tile(np.arange(nc_new), (4, 1)).ravel("F")
    # Define positive and negative sides. The choices here are somewhat arbitrary.
    tmp = np.ones(nc_new, dtype=int)
    cf_data = np.vstack((-tmp, tmp, -tmp, tmp)).ravel("F")
    cf = sps.coo_matrix((cf_data, (cf_rows, cf_cols)), shape=(nf_new, nc_new)).tocsc()

    tags = _define_tags(g, num_cell_layers)

    # We are ready to define the new grid
    g_info = g.name.copy()
    g_info.append("Extrude 1d->2d")

    g_new = pp.Grid(2, nodes, fn, cf, g_info, external_tags=tags)
    g_new.compute_geometry()

    if hasattr(g, "frac_num"):
        g_new.frac_num = g.frac_num

    # Mappings between old and new cells and faces
    cell_map, face_map = _create_mappings(g, g_new, num_cell_layers)

    return g_new, cell_map, face_map


@pp.time_logger(sections=module_sections)
def _extrude_0d(
    g: pp.PointGrid, z: np.ndarray
) -> Tuple[pp.Grid, np.ndarray, np.ndarray]:
    """Extrude a 0d grid into 1d by prismatic extension in the z-direction.

    The original grid is assumed to be in the xy-plane, that is, any existing non-zero
    z-direction is ignored.

    Both the original and the new grid will have their geometry computed.

    Parameters:
        g (pp.Grid): Original grid to be extruded. Should have dimension 1.
        z (np.ndarray): z-coordinates of the nodes in the extruded grid. Should be
            either non-negative or non-positive, and be sorted in increasing or
            decreasing order, respectively.

    Returns:
        pp.TensorGrid: A grid of dimension 1.
        np.array of np.arrays: Cell mappings, so that element ci gives all indices of
            cells in the extruded grid that comes from cell ci in the original grid.
        np.array of np.arrays: Face mappings, so that element fi gives all indices of
            faces in the extruded grid that comes from face fi in the original grid.

    """
    # Number of nodes
    num_pt = z.size

    # x and y coordinates of the right size
    x = g.cell_centers[0, 0] * np.ones(num_pt)
    y = g.cell_centers[1, 0] * np.ones(num_pt)

    # Initial 1d grid. Coordinates are wrong, but this we will fix later
    # There is no need to do anything special with tags here; the tags of a 0d grid are
    # trivial, and the 1d extrusion can be based on this.
    g_new = pp.TensorGrid(x)

    g_info = g.name.copy()
    g_info.append("Extrude 0d->1d")
    g_new.name = g_info

    # Update coordinates
    g_new.nodes = np.vstack((x, y, z))

    g_new.compute_geometry()

    # The single cell in g has produced all cells in g_new
    cell_map = np.empty(1, dtype=object)
    cell_map[0] = np.arange(g_new.num_cells)
    face_map = np.empty(0)

    return g_new, cell_map, face_map


@pp.time_logger(sections=module_sections)
def _define_tags(g: pp.Grid, num_cell_layers: int) -> Dict[str, np.ndarray]:
    """Define all standard tags (face and nodes) for the extruded grids

    The extrusion functions do not explicitly account for split nodes and faces due to
    intersections with other grids (because of fractures). The standard tag construction
    in the Grid __init__ will therefore not work. Instead, construct the information
    from the original grid.

    For the face tags, the implementation assumes that the vertical faces are indexed
    first, then the horizontal ones.

    Parameters:
        g (pp.Grid): Original grid.
        num_cell_layers (int): Number of cell extrusion layers.

    Returns:
        Dict: The standard tags (see pp.utils.tags) for faces and nodes for the
            extruded grid.

    """
    # Bookkeeping
    nc_old = g.num_cells
    nn_old = g.num_nodes

    # Nodes that are on a fracture tip or a fracture in the original grid will also be
    # that on the extruded grid.
    # The number of node layers is one more than the number of cells.
    tip_node_tag = np.tile(g.tags["tip_nodes"], (num_cell_layers + 1, 1)).ravel()
    fracture_node_tag = np.tile(
        g.tags["fracture_nodes"], (num_cell_layers + 1, 1)
    ).ravel()

    # All nodes in the bottom and top layers are on the domain boundary.
    domain_boundary_node_tag = np.ones(nn_old, dtype=bool)

    # Intermediate layers are as for the original grid
    for _ in range(num_cell_layers - 1):
        domain_boundary_node_tag = np.hstack(
            (domain_boundary_node_tag, g.tags["domain_boundary_nodes"].copy())
        )
    # Top layer is all boundary nodes
    domain_boundary_node_tag = np.hstack((domain_boundary_node_tag, np.ones(nn_old)))

    ## Face tags
    # ASSUMPTION: We know that the vertical faces are defined first. For these, the
    # information can be copied from the original grid.
    fracture_face_tag = np.empty(0, dtype=bool)
    tip_face_tag = np.empty(0, dtype=bool)
    boundary_face_tag = np.empty(0, dtype=bool)
    for _ in range(num_cell_layers):
        fracture_face_tag = np.hstack((fracture_face_tag, g.tags["fracture_faces"]))
        tip_face_tag = np.hstack((tip_face_tag, g.tags["tip_faces"]))
        boundary_face_tag = np.hstack(
            (boundary_face_tag, g.tags["domain_boundary_faces"])
        )

    ## Next the horizontal faces.
    # The horizontal faces are all non-fracture, non-tip
    fracture_face_tag = np.hstack(
        (fracture_face_tag, np.zeros(nc_old * (num_cell_layers + 1), dtype=bool))
    )
    tip_face_tag = np.hstack(
        (tip_face_tag, np.zeros(nc_old * (num_cell_layers + 1), dtype=bool))
    )

    # The bottom and top layer of horizontal faces are on the boundary, the rest is not
    boundary_face_tag = np.hstack(
        (
            boundary_face_tag,
            np.ones(nc_old, dtype=bool),
            # Intermediate layers
            np.zeros(nc_old * (num_cell_layers - 1), dtype=bool),
            # top
            np.ones(nc_old, dtype=bool),
        )
    )

    tags = {
        "fracture_faces": fracture_face_tag,
        "tip_faces": tip_face_tag,
        "domain_boundary_faces": boundary_face_tag,
        "fracture_nodes": fracture_node_tag,
        "tip_nodes": tip_node_tag,
        "domain_boundary_nodes": domain_boundary_node_tag,
    }
    return tags


@pp.time_logger(sections=module_sections)
def _create_mappings(
    g: pp.Grid, g_new: pp.Grid, num_cell_layers: int
) -> Tuple[np.ndarray, np.ndarray]:

    cell_map = np.empty(g.num_cells, dtype=object)
    for c in range(g.num_cells):
        cell_map[c] = np.arange(c, g_new.num_cells, g.num_cells)

    face_map = np.empty(g.num_faces, dtype=object)
    for f in range(g.num_faces):
        face_map[f] = np.arange(f, g.num_faces * num_cell_layers, g.num_faces)

    # Sanity checks on cell and face maps
    for cm in cell_map:
        if cm.size != num_cell_layers:
            raise ValueError("Cell map of wrong size")

    for fm in face_map:
        # The vertical faces should have num_cell_layers items.
        # Horizontal faces are not added to to the face_map.
        if fm.size != num_cell_layers:
            raise ValueError("Face map of wrong size")

    return cell_map, face_map
