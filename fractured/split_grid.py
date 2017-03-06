"""
Module for splitting a grid at the fractures.
"""

import numpy as np
from scipy import sparse as sps
from utils.half_space import half_space_int
from utils.graph import Graph
from core.grids.grid import Grid, FaceTag


def split_faces(grids, network):
    """
    Split faces of the grid along each fractures. This function will
    add an extra face to each fracture face. Note that the original
    and new fracture face will share the same nodes. However, the
    cell_faces connectivity is updated such that the fractures are
    be internal boundaries (cells on left side of fractures are not
    connected to cells on right side of fracture and vise versa).
    """

    frac_grids = grids[1]
    g = grids[0][0]
    for i, f in enumerate(frac_grids):
        # Each lower dim grid should only belong to one higher dim face
        assert all(np.diff(f.cell_2_face_pos) == 1)
        # Create convenientmappings
        frac_id = f.cell_2_face

        # duplicate faces along tagged faces.
        duplicate_faces(g, frac_id, f)

        # Set new cell conectivity
        update_cell_connectivity(
            g, frac_id, network.get_normal(i), network.get_center(i))

        #        if i<f.num:
        #            new_faces = np.zeros((f.tag.shape[0],sum(f.tag[i,:])),dtype='bool')
        #            new_faces[i,:] = True
        #            f.tag=np.hstack((f.tag, new_faces))
    return g


def split_nodes(grids, network, offset=0):
    """
    splits the nodes of a grid given a fracture and a colored graph.
    Parameters
    ----------
    g - A grid. All fracture faces should first be duplicated
    f - a Fracture object.
    graph - a Graph object. All the nodes in the graph should be colored
            acording to the fracture regions (Graph.color_nodes())
    offset - float
             Optional, defaults to 0. This gives the offset from the
             fracture to the new nodes. Note that this is only for
             visualization, e.g., g.face_centers is not updated.
    """
    # Create convenient mappings
    int_nodes = []
    for g in grids[1]:
        bdr = g.get_boundary_faces()
        bdr_nodes = np.ravel(
            np.sum(g.face_nodes[:, bdr], axis=1)).astype('bool')
        int_nodes = np.append(int_nodes, g.global_point_ind[~bdr_nodes])
    int_nodes = np.unique(int_nodes)

    row = np.array([], dtype=np.int32)
    col = np.array([], dtype=np.int32)
    node_count = 0
    # Iterate over each internal node and split it according to the graph.
    # For each cell attached to the node, we check wich color the cell has.
    # All cells with the same color is then attached to a new copy of the
    # node.
    h = grids[0][0]
    for i, node in enumerate(int_nodes):
        # Find cells connected to node
        (_, cells, _) = sps.find(h.cell_nodes()[node, :])
        #cells = np.unique(cells)
        (h_frac, unique_faces, unique_nodes) = extract_subgrid(h, cells, sort=True)
        h.child_cell_ind = np.array([-1] * h.num_cells, dtype=np.int)
        h.child_cell_ind[h_frac.parent_cell_ind] = np.arange(h_frac.num_cells)
        graph = Graph(h_frac.cell_connection_map())
        graph.color_nodes()
        colors = graph.color[h.child_cell_ind[cells]]
        colors, ix, _ = np.unique(
            colors, return_inverse=True, return_counts=True)
        new_nodes = np.repeat(h.nodes[:, node, None], colors.size, axis=1)
        for j in range(colors.size):
            # Find faces of each cell that are attached to node
            faces, _, _ = sps.find(h.cell_faces[:, cells[ix == j]])
            faces = np.unique(faces)
            con_to_node = np.ravel(h.face_nodes[node, faces].todense())
            faces = faces[con_to_node]
            col = np.append(col, faces)
            row = np.append(row, [node_count + j] * faces.size)
            # Change position of nodes
            if offset > 0:
                frac_face = np.ravel(
                    np.sum(np.abs(h.cell_faces[faces, :]), axis=1) == 1)
                f, _, sign = sps.find(h.cell_faces[faces[frac_face], :])
                n = h.face_normals[:, faces[frac_face]]
                n = n[:, f] * sign
                n = np.mean(n, axis=1)
                n = n / np.linalg.norm(n)
                new_nodes[:, j] -= n * offset

        h.nodes = np.hstack((h.nodes, new_nodes))
        node_count += colors.size

    # Add new nodes to face-node map
    new_face_nodes = sps.csc_matrix(
        ([True] * row.size, (row, col)), (node_count, h.num_faces))
    h.face_nodes = sps.vstack((h.face_nodes, new_face_nodes), format='csc')
    # Remove old nodes
    h = remove_nodes(h, int_nodes)
    # Update the number of nodes
    h.num_nodes = h.num_nodes + node_count - int_nodes.size
    return True


def split_fractures(grids, network, offset=0):
    """
    Wrapper function to split all fractures. Will split faces and nodes
    to create an internal boundary.

    The tagged faces are split in two along with connected nodes (except
    tips).

    To be added:
    3D fractures

    Parameters
    ----------
    g - A valid grid

    frac_tag - Fracture class
        an object of Fracture class
    tip_nodes_id - ndarray
        Defaults to None. If None, it tries to locate the fracture tips
        based on the number of tagged faces connecting each node. Fracture
        tips are then tagged as nodes only connected to one tagged face.
        If None, tip_nodes_id should be the indices of the tip nodes of
        the fractures. The nodes in the tip_nodes_id will not be split.
    offset - float
        Defaults to 0. The fracture nodes are moved a distance 0.5*offest
        to each side of the fractures. WARNING: this is for visualization
        purposes only. E.g, the face centers are not moved.
    Returns
    -------
    g - A valid grid deformation where with internal boundaries.


    Examples
    >>> import numpy as np
    >>> from core.grids import structured
    >>> from viz import plot_grid
    >>> import matplotlib.pyplot as plt
    >>> import gridding.fractured.split_grid
    >>> # Set up a Cartesian grid
    >>> n = 10
    >>> g = structured.CartGrid([n, n])
    >>> g.compute_geometry()
    >>> # Define fracture
    >>> frac_tag1 = np.logical_and(np.logical_and(g.face_centers[1,:]==n/2,
    >>>            g.face_centers[0,:]>n/4), g.face_centers[0,:]<3*n/4)
    >>> frac_tag2 = np.logical_and(np.logical_and(g.face_centers[0,:]==n/2,
    >>>                            g.face_centers[1,:]>=n/4),
    >>>                    g.face_centers[1,:]<3*n/4)
    >>> f = split_grid.Fracture(h)
    >>> f.add_tag(g,frac_tag1)
    >>> f.add_tag(g,frac_tag2)
    >>> split_grid.split_fractures(g,f,offset=0.25
    >>> plot_grid.plot_grid(g)
    >>> plt.show()
    """
    # Doubplicate all fracture faces
    split_faces(grids, network)
    # Split the nodes along fractures
    split_nodes(grids, network, offset=offset)
    grids[0][0].cell_faces.eliminate_zeros()
    return grids


def duplicate_faces(g, frac_id, frac):
    """
    Duplicate faces along fracture.

    Parameters
    ----------
    g       - The grid for which the faces are dublicated
    frac    - The lower dimensional grid representing the fractures
    frac_id - The indices of the faces that should be duplicated
    """
    frac_nodes = g.face_nodes[:, frac_id]
    g.face_nodes = sps.hstack((g.face_nodes, frac_nodes))
    new_map = g.num_faces + np.arange(frac_id.size)
    frac.cell_2_face = np.insert(new_map, np.arange(
        frac.cell_2_face.size), frac.cell_2_face)
    frac.cell_2_face_pos = np.arange(0, frac.cell_2_face.size + 1, 2)

    # update face info
    g.num_faces += frac_id.size
    g.face_normals = np.hstack(
        (g.face_normals, g.face_normals[:, frac_id]))
    g.face_areas = np.append(g.face_areas, g.face_areas[frac_id])
    g.face_centers = np.hstack(
        (g.face_centers, g.face_centers[:, frac_id]))
    # shoudl fix tag's later
    #g.add_face_tag(frac, FaceTag.FRACTURE | FaceTag.BOUNDARY)
    #g.face_tags = np.append(g.face_tags, g.face_tags[frac])


def update_cell_connectivity(g, frac_id, normal, x0):
    """
    After the faces in a grid is duplicated, we update the cell connectivity list
    Cells on the right side of the fracture does not change, but the cells
    on the left side are attached to the face duplicates. We assume that all
    faces that have been duplicated lie in the same plane. This plane is
    described by a normal and a point, x0. We attach cell on the left side of the
    plane to the duplicate of frac_id. The cells on the right side is attached
    to the face frac_id

    Parameters:
    ----------
    g         - The grid for wich the cell_face mapping is uppdated
    frac_id   - Indices of the faces that have been duplicated
    normal    - Normal of faces that have been duplicated. Note that we assume
                that all faces have the same normal
    x0        - A point in the plane where the faces lie
    """
    # Cells on right side does not change. We first add the new left-faces
    # to the left cells
    cell_frac = g.cell_faces[frac_id, :]
    cell_frac_id = np.argwhere(cell_frac)
    left_cell = half_space_int(normal, x0,
                               g.cell_centers[:, cell_frac_id[:, 1]])
    col = cell_frac_id[left_cell, 1]
    row = cell_frac_id[left_cell, 0]
    data = np.ravel(g.cell_faces[np.ravel(frac_id[row]), col])
    cell_frac_left = sps.csc_matrix((data, (row, col)),
                                    (frac_id.size, g.cell_faces.shape[1]))

    # We remove the right faces of the left cells.
    col = cell_frac_id[~left_cell, 1]
    row = cell_frac_id[~left_cell, 0]
    data = np.ravel(g.cell_faces[np.ravel(frac_id[row]), col])

    cell_frac_right = sps.csc_matrix((data, (row, col)),
                                     (frac_id.size, g.cell_faces.shape[1]))

    g.cell_faces[frac_id, :] = cell_frac_right
    g.cell_faces = sps.vstack((g.cell_faces, cell_frac_left), format='csc')


def remove_nodes(g, rem):
    """
    Remove nodes from grid.
    g - a valid grid definition
    rem - a ndarray of indecies of nodes to be removed
    """
    all_rows = np.arange(g.face_nodes.shape[0])
    rows_to_keep = np.where(np.logical_not(np.in1d(all_rows, rem)))[0]
    g.face_nodes = g.face_nodes[rows_to_keep, :]
    g.nodes = g.nodes[:, rows_to_keep]
    return g


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
        core.grids.Grid: Extracted subgrid. Will share (note, *not* copy)
            geometric fileds with the parent grid. Also has an additional
            field parent_cell_ind giving correspondance between parent and
            child cells.
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

    h.parent_cell_ind = c

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
