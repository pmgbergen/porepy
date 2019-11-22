"""
Module to increase the dimensions of grids by extrusion in the z-direction.

Both individual grids and mixed-dimensional grid_buckets can be extruded. The
dimension of the highest-dimensional grid should be 2 at most.

"""
import numpy as np
import porepy as pp
import scipy.sparse as sps
from typing import Union, Dict
from collections import namedtuple
from porepy.grids import mortar_grid


def extrude_grid_bucket(gb: pp.GridBucket, z: np.ndarray) -> Union[pp.GridBucket, Dict]:
    """ Extrude a GridBucket by extending all fixed-dimensional grids in the z-direction.

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
    gb_new = pp.GridBucket()

    # Data structure for mapping between old and new grids
    g_map = {}

    # Container for grid information
    Mapping = namedtuple("mapping", ["grid", "cell_map", "face_map"])

    # Loop over all grids in the old bucket, extrude the grid, save mapping information
    for g, _ in gb:
        g_new, cell_map, face_map = extrude_grid(g, z)
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

        rows = np.empty(0, dtype=np.int)
        cols = np.empty(0, dtype=np.int)

        # Loop over all the faces, find its extruded face children.
        # Loop over cells in gl would not have been as clean, as each cell is associated
        # with faces on both sides
        # Faces are found from the high-dim grid, cells in the low-dim grid
        for idx in range(faces.size):
            rows = np.hstack((rows, cell_map[cells[idx]]))
            cols = np.hstack((cols, face_map[faces[idx]]))

        data = np.ones(rows.size, dtype=np.bool)
        # Create new face-cell map
        face_cells_new = sps.csc_matrix(
            (data, (rows, cols)), shape=(gl_new.num_cells, gh_new.num_faces)
        )

        # Define the new edge
        e = (gh_new, gl_new)
        # Add to new gb, together with the new face-cell map
        gb_new.add_edge(e, face_cells_new)

        # Create a mortar grid, add to data of new edge
        side_g = {
            mortar_grid.LEFT_SIDE: gl_new.copy(),
            mortar_grid.RIGHT_SIDE: gl_new.copy(),
        }

        mg = pp.MortarGrid(gl_new.dim, side_g, face_cells_new)

        d_new = gb_new.edge_props(e)

        d_new["mortar_grid"] = mg

    return gb_new, g_map


def extrude_grid(g: pp.Grid, z: np.ndarray) -> Union[pp.Grid, np.ndarray, np.ndarray]:
    """ Increase the dimension of a given grid by 1, by extruding the grid in the
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
    if not np.all(z >= 0) or np.all(z <= 0):
        raise ValueError("Extrusion should be in either positive or negative direction")

    if g.dim == 0:
        return _extrude_0d(g, z)
    elif g.dim == 1:
        return _extrude_1d(g, z)
    elif g.dim == 2:
        return _extrude_2d(g, z)
    else:
        raise ValueError("The grid to be extruded should have dimension at most 2")


def _extrude_2d(g: pp.Grid, z: np.ndarray) -> Union[pp.Grid, np.ndarray, np.ndarray]:
    """ Extrude a 2d grid into 3d by prismatic extension.

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
    cn_ind_2d = cn_2d.indices

    # Similar to the vertical faces, the face-node relation in 3d should match the
    # sign in the cell-face relation, so that the generated normal vector points out of
    # the cell with cf-value 1.
    # This requires a sorting of the nodes for each cell
    for ci in range(nc_2d):
        # Node indices of this 2d cell
        start = cn_2d.indptr[ci]
        stop = cn_2d.indptr[ci + 1]
        ni = cn_ind_2d[start:stop]

        # Coordinates
        coord = g.nodes[:2, ni]

        # If the polygon is already ccw, we can keep the ordering, unless the extrusion
        # is in the negative direction
        if pp.geometry_property_checks.is_ccw_polygon(coord):
            if negative_extrusion:
                cn_ind_2d[start:stop] = cn_ind_2d[start:stop][::-1]
            else:
                continue
        # Oposite: If the polygon is in cw, we should switch ordering if the extrusion
        # is positive
        elif pp.geometry_property_checks.is_ccw_polygon(coord[:, ::-1]):
            if not negative_extrusion:
                cn_ind_2d[start:stop] = cn_ind_2d[start:stop][::-1]
            else:
                continue
        # If we get this far, we first need to obtain an ordering (cw or ccw)
        # IMPLEMENTATION NOTE: This part of the code is not very well tested. Errors
        # here will likely give issues with negative subtet volumes in
        # g_new.compute_geometry_3d()
        else:
            # Indices that sort the nodes. The called function contains a rotation, which
            # implies that it is unknown whether the ordering is cw or ccw
            sort_ind = pp.utils.sort_points.sort_point_plane(
                np.vstack((coord, np.zeros(coord.shape[1]))),
                g.cell_centers[:, ci].reshape((-1, 1)),
            )
            # Deal with the two cases as we did above.
            if pp.geometry_property_checks.is_ccw_polygon(coord[:, sort_ind]):
                if negative_extrusion:
                    cn_ind_2d[start:stop] = cn_ind_2d[start:stop][sort_ind[::-1]]
                else:
                    cn_ind_2d[start:stop] = cn_ind_2d[start:stop][sort_ind]
            else:  # Now it should be cw
                if negative_extrusion:
                    cn_ind_2d[start:stop] = cn_ind_2d[start:stop][sort_ind]
                else:
                    cn_ind_2d[start:stop] = cn_ind_2d[start:stop][sort_ind[::-1]]

    # Compressed column storage for horizontal faces: Store node indices
    fn_rows_horizontal = np.array([], dtype=np.int)
    # .. and pointers to the start of new faces
    fn_cols_horizontal = np.array(0, dtype=np.int)
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
    indptr = np.hstack((fn_cols_vertical, fn_cols_horizontal)).astype(np.int)
    indices = np.hstack((fn_rows_vertical, fn_rows_horizontal)).astype(np.int)
    data = np.ones(indices.size, dtype=np.int)

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

    cf_rows_vertical = np.array([], dtype=np.int)
    # For the cells, we will store the number of facqes for each cell. This will later
    # be expanded to a full set of cell indices
    cf_vertical_cell_count = np.array([], dtype=np.int)
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
    cf_data_horizontal = -np.ones(nc_2d, dtype=np.int)

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

    name = g.name.copy()
    name.append("Extrude 2d->3d")
    g_info = g.name.copy()
    g_info.append("Extrude 1d->2d")

    g_new = pp.Grid(3, nodes, face_nodes, cell_faces, g_info)
    g_new.compute_geometry()

    cell_map = np.empty(g.num_cells, dtype=np.object)
    for c in range(g.num_cells):
        cell_map[c] = np.arange(c, g_new.num_cells, g.num_cells)

    face_map = np.empty(g.num_faces, dtype=np.object)
    for f in range(g.num_faces):
        face_map[f] = np.arange(f, g.num_faces * num_cell_layers, g.num_faces)

    return g_new, cell_map, face_map


def _extrude_1d(
    g: pp.TensorGrid, z: np.ndarray
) -> Union[pp.Grid, np.ndarray, np.ndarray]:
    """ Extrude a 1d grid into 2d by prismatic extension in the z-direction.

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
    # Number of nodes

    x = g.nodes[0]
    y = g.nodes[1]

    g_new = pp.TensorGrid(x, z)

    x_2d, z_2d = np.meshgrid(x, z)
    y_2d, _ = np.meshgrid(y, z)

    g_info = g.name.copy()
    g_info.append("Extrude 1d->2d")
    g_new.name = g_info

    g_new.nodes = np.vstack((x_2d.ravel(), y_2d.ravel(), z_2d.ravel()))

    g_new.compute_geometry()

    cell_map = np.empty(g.num_cells, dtype=np.object)
    for c in range(g.num_cells):
        cell_map[c] = np.arange(c, g_new.num_cells, g.num_cells)

    face_map = np.empty(g.num_faces, dtype=np.object)
    for f in range(g.num_faces):
        face_map[f] = np.arange(f, g.num_faces * (z.size - 1), g.num_faces)

    return g_new, cell_map, face_map


def _extrude_0d(
    g: pp.PointGrid, z: np.ndarray
) -> Union[pp.Grid, np.ndarray, np.ndarray]:
    """ Extrude a 0d grid into 1d by prismatic extension in the z-direction.

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
    x = g.nodes[0, 0] * np.ones(num_pt)
    y = g.nodes[1, 0] * np.ones(num_pt)

    # Initial 1d grid. Coordinates are wrong, but this we will fix later
    g_new = pp.TensorGrid(x)

    g_info = g.name.copy()
    g_info.append("Extrude 0d->1d")
    g_new.name = g_info

    # Update coordinates
    g_new.nodes = np.vstack((x, y, z))

    g_new.compute_geometry()

    # The single cell in g has produced all cells in g_new
    cell_map = np.empty(1, dtype=np.object)
    cell_map[0] = np.arange(g_new.num_cells)
    face_map = np.empty(0)

    return g_new, cell_map, face_map
