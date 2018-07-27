"""
Module for splitting a grid at the fractures.
"""

import numpy as np
from scipy import sparse as sps

from porepy.utils.half_space import half_space_int
from porepy.utils import sparse_mat, tags
from porepy.utils.graph import Graph
from porepy.utils.mcolon import mcolon


def split_fractures(bucket, **kwargs):
    """
    Wrapper function to split all fractures. For each grid in the bucket,
    we locate the corresponding lower-dimensional grids. The faces and
    nodes corresponding to these grids are then split, creating internal
    boundaries.

    Parameters
    ----------
    bucket    - A grid bucket
    **kwargs:
        offset    - FLOAT, defaults to 0. Will perturb the nodes around the
                    faces that are split. NOTE: this is only for visualization.
                    E.g., the face centers are not perturbed.

    Returns
    -------
    bucket    - A valid bucket where the faces are split at
                internal boundaries.


    Examples
    >>> import numpy as np
    >>> from gridding.fractured import meshing, split_grid
    >>> from viz.exporter import export_vtk
    >>>
    >>> f_1 = np.array([[-1, 1, 1, -1 ], [0, 0, 0, 0], [-1, -1, 1, 1]])
    >>> f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1 ], [-.7, -.7, .8, .8]])
    >>> f_set = [f_1, f_2]
    >>> domain = {'xmin': -2, 'xmax': 2,
            'ymin': -2, 'ymax': 2, 'zmin': -2, 'zmax': 2}
    >>> bucket = meshing.create_grid(f_set, domain)
    >>> [g.compute_geometry() for g,_ in bucket]
    >>>
    >>> split_grid.split_fractures(bucket, offset=0.1)
    >>> export_vtk(bucket, "grid")
    """

    offset = kwargs.get("offset", 0)

    # For each vertex in the bucket we find the corresponding lower-
    # dimensional grids.
    for gh, _ in bucket:
        # add new field to grid
        gh.frac_pairs = np.zeros((2, 0), dtype=np.int32)
        if gh.dim < 1:
            # Nothing to do. We can not split 0D grids.
            continue
        # Find connected vertices and corresponding edges.
        neigh = np.array(bucket.node_neighbors(gh))

        # Find the neighbours that are lower dimensional
        is_low_dim_grid = np.where([w.dim < gh.dim for w in neigh])
        edges = [(gh, w) for w in neigh[is_low_dim_grid]]
        if len(edges) == 0:
            # No lower dim grid. Nothing to do.
            continue

        face_cells = [bucket.edge_props(e, "face_cells") for e in edges]
        # We split all the faces that are connected to a lower-dim grid.
        # The new faces will share the same nodes and properties (normals,
        # etc.)
        face_cells = split_faces(gh, face_cells)

        for e, f in zip(edges, face_cells):
            bucket.add_edge_props("face_cells", [e])
            bucket.edge_props(e)["face_cells"] = f

        # We now find which lower-dim nodes correspond to which higher-
        # dim nodes. We split these nodes according to the topology of
        # the connected higher-dim cells. At a X-intersection we split
        # the node into four, while at the fracture boundary it is not split.

        gl = [e[1] for e in edges]
        gl_2_gh_nodes = [bucket.target_2_source_nodes(g, gh) for g in gl]

        split_nodes(gh, gl, gl_2_gh_nodes, offset)

    # Remove zeros from cell_faces

    [g.cell_faces.eliminate_zeros() for g, _ in bucket]
    [g.update_boundary_node_tag() for g, _ in bucket]
    return bucket


def split_faces(gh, face_cells):
    """
    Split faces of the grid along each fracture. This function will
    add an extra face to each fracture face. Note that the original
    and new fracture face will share the same nodes. However, the
    cell_faces connectivity is updated such that the fractures are
    be internal boundaries (cells on left side of fractures are not
    connected to cells on right side of fracture and vise versa).
    The face_cells are updated such that the copy of a face also
    map to the same lower-dim cell.
    """
    gh.frac_pairs = np.zeros((2, 0), dtype=np.int32)
    for i in range(len(face_cells)):
        # We first we duplicate faces along tagged faces. The duplicate
        # faces will share the same nodes as the original faces,
        # however, the new faces are not yet added to the cell_faces map
        # (to save computation).
        face_id = duplicate_faces(gh, face_cells[i])
        face_cells = update_face_cells(face_cells, face_id, i)
        if face_id.size == 0:
            continue

        # We now set the cell_faces map based on which side of the
        # fractures the cells lie. We assume that all fractures are
        # flat surfaces and pick the normal of the first face as
        # a normal for the whole fracture.
        n = np.reshape(gh.face_normals[:, face_id[0]], (3, 1))
        n = n / np.linalg.norm(n)
        x0 = np.reshape(gh.face_centers[:, face_id[0]], (3, 1))
        flag = update_cell_connectivity(gh, face_id, n, x0)

        if flag == 0:
            # if flag== 0 we added left and right faces (if it is -1 no faces
            # was added and we don't have left and right face pairs.
            # we now add the new faces to the frac_pair array.
            left = face_id
            right = np.arange(gh.num_faces - face_id.size, gh.num_faces)
            gh.frac_pairs = np.hstack((gh.frac_pairs, np.vstack((left, right))))

    return face_cells


def split_nodes(gh, gl, gh_2_gl_nodes, offset=0):
    """
    Splits the nodes of a grid given a set of lower-dimensional grids
    and a connection mapping between them.
    Parameters
    ----------
    gh            - Higher-dimension grid.
    gl            - A list of lower dimensional grids
    gh_2_gl_nodes - A list of connection arrays. Each array in the
                    list gives the mapping from the lower-dim nodes
                    to the higher dim nodes. gh_2_gl_nodes[0][0] is
                    the higher-dim index of the first node of the
                    first lower-dim.
    offset - float
             Optional, defaults to 0. This gives the offset from the
             fracture to the new nodes. Note that this is only for
             visualization, e.g., g.face_centers is not updated.
    """
    # We find the higher-dim node indices of all lower-dim nodes
    nodes = np.array([], dtype=int)
    for i in range(len(gl)):
        nodes = np.append(nodes, gh_2_gl_nodes[i])
    nodes = np.unique(nodes)

    # Each of these nodes are duplicated dependig on the cell-
    # topology of the higher-dim around each node. For a X-intersection
    # we get four duplications, for a T-intersection we get three
    # duplications, etc. Each of the duplicates are then attached
    # to the cells on one side of the fractures.
    node_count = duplicate_nodes(gh, nodes, offset)

    # We remove the old nodes.
    # gh = remove_nodes(gh, nodes)

    # Update the number of nodes
    gh.num_nodes = gh.num_nodes + node_count  # - nodes.size

    return True


def duplicate_faces(gh, face_cells):
    """
    Duplicate all faces that are connected to a lower-dim cell

    Parameters
    ----------
    gh         - Higher-dim grid
    face_cells - A list of connection matrices. Each matrix gives
                 the mapping from the cells of a lower-dim grid
                 to the faces of the higher diim grid.
    """
    # We find the indices of the higher-dim faces to be duplicated.
    # Each of these faces are duplicated, and the duplication is
    # attached to the same nodes. We do not attach the faces to
    # any cells as this connection will have to be undone later
    # anyway.
    frac_id = face_cells.nonzero()[1]
    frac_id = np.unique(frac_id)
    rem = tags.all_face_tags(gh.tags)[frac_id]
    gh.tags["fracture_faces"][frac_id[rem]] = True
    gh.tags["tip_faces"][frac_id] = False

    frac_id = frac_id[~rem]
    if frac_id.size == 0:
        return frac_id

    node_start = gh.face_nodes.indptr[frac_id]
    node_end = gh.face_nodes.indptr[frac_id + 1]
    nodes = gh.face_nodes.indices[mcolon(node_start, node_end)]
    added_node_pos = np.cumsum(node_end - node_start) + gh.face_nodes.indptr[-1]
    assert added_node_pos.size == frac_id.size
    assert added_node_pos[-1] - gh.face_nodes.indptr[-1] == nodes.size
    gh.face_nodes.indices = np.hstack((gh.face_nodes.indices, nodes))
    gh.face_nodes.indptr = np.hstack((gh.face_nodes.indptr, added_node_pos))
    gh.face_nodes.data = np.hstack(
        (gh.face_nodes.data, np.ones(nodes.size, dtype=bool))
    )
    gh.face_nodes._shape = (gh.num_nodes, gh.face_nodes.shape[1] + frac_id.size)
    assert gh.face_nodes.indices.size == gh.face_nodes.indptr[-1]

    node_start = gh.face_nodes.indptr[frac_id]
    node_end = gh.face_nodes.indptr[frac_id + 1]

    # frac_nodes = gh.face_nodes[:, frac_id]

    # gh.face_nodes = sps.hstack((gh.face_nodes, frac_nodes))
    # We also copy the attributes of the original faces.
    gh.num_faces += frac_id.size
    gh.face_normals = np.hstack((gh.face_normals, gh.face_normals[:, frac_id]))
    gh.face_areas = np.append(gh.face_areas, gh.face_areas[frac_id])
    gh.face_centers = np.hstack((gh.face_centers, gh.face_centers[:, frac_id]))

    # Not sure if this still does the correct thing. Might have to
    # send in a logical array instead of frac_id.
    gh.tags["fracture_faces"][frac_id] = True
    gh.tags["tip_faces"][frac_id] = False
    update_fields = tags.standard_face_tags()
    update_values = [[]] * len(update_fields)
    for i, key in enumerate(update_fields):
        update_values[i] = gh.tags[key][frac_id]
    tags.append_tags(gh.tags, update_fields, update_values)

    return frac_id


def update_face_cells(face_cells, face_id, i):
    """
    Add duplicate faces to connection map between lower-dim grids
    and higher dim grids. To be run after duplicate_faces.
    """
    # We duplicated the faces associated with lower-dim grid i.
    # The duplications should also be associated with grid i.
    # For the other lower-dim grids we just add zeros to conserve
    # the right matrix dimensions.
    if face_id.size == 0:
        return face_cells

    for j, f_c in enumerate(face_cells):
        assert f_c.getformat() == "csc"
        if j == i:
            f_c_sliced = sparse_mat.slice_mat(f_c, face_id)
            new_indptr = f_c_sliced.indptr + f_c.indptr[-1]
            new_ind = f_c_sliced.indices
            new_data = f_c_sliced.data

            f_c.indptr = np.append(f_c.indptr, new_indptr[1:])
            f_c.indices = np.append(f_c.indices, new_ind)
            f_c.data = np.append(f_c.data, new_data)
            f_c._shape = (f_c._shape[0], f_c._shape[1] + face_id.size)
            # f_c = sps.hstack((f_c, f_c[:, face_id]))
        else:
            new_indptr = f_c.indptr[-1] * np.ones(face_id.size, dtype=f_c.indptr.dtype)
            f_c.indptr = np.append(f_c.indptr, new_indptr)
            f_c._shape = (f_c._shape[0], f_c._shape[1] + face_id.size)
        #            empty = sps.csc_matrix((f_c.shape[0], face_id.size))
        #            f_c = sps.hstack((f_c, empty))
        face_cells[j] = f_c
    return face_cells


def update_cell_connectivity(g, face_id, normal, x0):
    """
    After the faces in a grid is duplicated, we update the cell connectivity list.
    Cells on the right side of the fracture does not change, but the cells
    on the left side are attached to the face duplicates. We assume that all
    faces that have been duplicated lie in the same plane. This plane is
    described by a normal and a point, x0. We attach cell on the left side of the
    plane to the duplicate of face_id. The cells on the right side is attached
    to the face frac_id

    Parameters:
    ----------
    g         - The grid for wich the cell_face mapping is uppdated
    frac_id   - Indices of the faces that have been duplicated
    normal    - Normal of faces that have been duplicated. Note that we assume
                that all faces have the same normal
    x0        - A point in the plane where the faces lie
    """

    # We find the cells attached to the tagged faces.
    g.cell_faces = g.cell_faces.tocsr()
    cell_frac = g.cell_faces[face_id, :]
    cell_face_id = np.argwhere(cell_frac)

    # We devide the cells into the cells on the right side of the fracture
    # and cells on the left side of the fracture.
    left_cell = half_space_int(normal, x0, g.cell_centers[:, cell_face_id[:, 1]])

    if np.all(left_cell) or not np.any(left_cell):
        # Fracture is on boundary of domain. There is nothing to do.
        # Remove the extra faces. We have not yet updated cell_faces,
        # so we should not delete anything from this matrix.
        rem = np.arange(g.cell_faces.shape[0], g.num_faces)
        remove_faces(g, rem, rem_cell_faces=False)
        return -1

    # Assume that fracture is either on boundary (above case) or completely
    # innside domain. Check that each face added two cells:
    assert (
        sum(left_cell) * 2 == left_cell.size
    ), "Fractures must either be" "on boundary or completely innside domain"

    # We create a cell_faces mapping for the new faces. This will be added
    # on the end of the excisting cell_faces mapping. We have here assumed
    # that we do not add any mapping during the duplication of faces.
    col = cell_face_id[left_cell, 1]
    row = cell_face_id[left_cell, 0]
    data = np.ravel(g.cell_faces[np.ravel(face_id[row]), col])
    assert data.size == face_id.size
    cell_frac_left = sps.csr_matrix(
        (data, (row, col)), (face_id.size, g.cell_faces.shape[1])
    )

    # We now update the cell_faces map of the faces on the right side of
    # the fracture. These faces should only be attached to the right cells.
    # We therefore remove their connection to the cells on the left side of
    # the fracture.
    col = cell_face_id[~left_cell, 1]
    row = cell_face_id[~left_cell, 0]
    data = np.ravel(g.cell_faces[np.ravel(face_id[row]), col])
    cell_frac_right = sps.csr_matrix(
        (data, (row, col)), (face_id.size, g.cell_faces.shape[1])
    )

    assert g.cell_faces.getformat() == "csr"

    sparse_mat.merge_matrices(g.cell_faces, cell_frac_right, face_id)
    #   g.cell_faces[face_id, :] = cell_frac_right

    # And then we add the new left-faces to the cell_face map. We do not
    # change the sign of the matrix since we did not flip the normals.
    # This means that the normals of right and left cells point in the same
    # direction, but their cell_faces values have oposite signs.
    sparse_mat.stack_mat(g.cell_faces, cell_frac_left)
    g.cell_faces = g.cell_faces.tocsc()
    # g.cell_faces = sps.vstack((g.cell_faces, cell_frac_left), format='csc')

    return 0


def remove_faces(g, face_id, rem_cell_faces=True):
    """
    Remove faces from grid.

    PARAMETERS:
    -----------
    g              - A grid
    face_id        - Indices of faces to remove
    rem_cell_faces - Defaults to True. If set to false, the g.cell_faces matrix
                     is not changed.
    """
    # update face info
    keep = np.array([True] * g.num_faces)
    keep[face_id] = False
    g.face_nodes = g.face_nodes[:, keep]
    g.num_faces -= face_id.size
    g.face_normals = g.face_normals[:, keep]
    g.face_areas = g.face_areas[keep]
    g.face_centers = g.face_centers[:, keep]
    # Not sure if still works
    update_fields = tags.standard_face_tags()
    for key in update_fields:
        g.tags[key] = g.tags[key][keep]

    if rem_cell_faces:
        g.cell_faces = g.cell_faces[keep, :]


def duplicate_nodes(g, nodes, offset):
    """
    Duplicate nodes on a fracture. The number of duplication will depend on
    the cell topology around the node. If the node is not on a fracture 1
    duplicate will be added. If the node is on a single fracture 2 duplicates
    will be added. If the node is on a T-intersection 3 duplicates will be
    added. If the node is on a X-intersection 4 duplicates will be added.
    Equivalently for other types of intersections.

    Parameters:
    ----------
    g         - The grid for which the nodes are duplicated
    nodes     - The nodes to be duplicated
    offset    - How far from the original node the duplications should be
                placed.
    """
    node_count = 0

    # We wish to convert the sparse csc matrix to a sparse
    # csr matrix to easily add rows. However, the convertion sorts the
    # indices, which will change the node order when we convert back. We
    # therefore find the inverse sorting of the nodes of each face.
    # After we have performed the row operations we will map the nodes
    # back to their original position.

    _, iv = sort_sub_list(g.face_nodes.indices, g.face_nodes.indptr)
    g.face_nodes = g.face_nodes.tocsr()
    # Iterate over each internal node and split it according to the graph.
    # For each cell attached to the node, we check wich color the cell has.
    # All cells with the same color is then attached to a new copy of the
    # node.
    cell_nodes = g.cell_nodes().tocsr()
    for node in nodes:
        # t_node takes into account the added nodes.
        t_node = node + node_count
        # Find cells connected to node

        cells = sparse_mat.slice_indices(cell_nodes, node)
        #        cell_nodes = g.cell_nodes().tocsr()
        #        ind_ptr = cell_nodes.indptr
        #        cells = cell_nodes.indices[
        #            mcolon(ind_ptr[t_node], ind_ptr[t_node + 1])]
        cells = np.unique(cells)
        # Find the color of each cell. A group of cells is given the same color
        # if they are connected by faces. This means that all cells on one side
        # of a fracture will have the same color, but a different color than
        # the cells on the other side of the fracture. Equivalently, the cells
        # at a X-intersection will be given four different colors
        colors = find_cell_color(g, cells)
        # Find which cells share the same color
        colors, ix = np.unique(colors, return_inverse=True)
        # copy coordinate of old node
        new_nodes = np.repeat(g.nodes[:, t_node, None], colors.size, axis=1)
        faces = np.array([], dtype=int)
        face_pos = np.array([g.face_nodes.indptr[t_node]])
        assert g.cell_faces.getformat() == "csc"
        assert g.face_nodes.getformat() == "csr"
        faces_of_node_t = sparse_mat.slice_indices(g.face_nodes, t_node)
        for j in range(colors.size):
            # For each color we wish to add one node. First we find all faces that
            # are connected to the fracture node, and have the correct cell
            # color
            colored_faces = sparse_mat.slice_indices(g.cell_faces, cells[ix == j])
            colored_faces = np.unique(colored_faces)
            is_colored = np.in1d(faces_of_node_t, colored_faces, assume_unique=True)

            faces = np.append(faces, faces_of_node_t[is_colored])
            # These faces are then attached to new node number j.
            face_pos = np.append(face_pos, face_pos[-1] + np.sum(is_colored))
            # If an offset is given, we will change the position of the nodes.
            # We move the nodes a length of offset away from the fracture(s).
            if offset > 0 and colors.size > 1:
                new_nodes[:, j] -= avg_normal(g, faces_of_node_t[is_colored]) * offset
        # The total number of faces should not have changed, only their
        # connection to nodes. We can therefore just update the indices and
        # indptr map.
        g.face_nodes.indices[face_pos[0] : face_pos[-1]] = faces
        node_count += colors.size - 1
        g.face_nodes.indptr = np.insert(g.face_nodes.indptr, t_node + 1, face_pos[1:-1])
        g.face_nodes._shape = (
            g.face_nodes.shape[0] + colors.size - 1,
            g.face_nodes._shape[1],
        )
        # We delete the old node because of the offset. If we do not
        # have an offset we could keep it and add one less node.
        g.nodes = np.delete(g.nodes, t_node, axis=1)
        g.nodes = np.insert(g.nodes, [t_node] * new_nodes.shape[1], new_nodes, axis=1)

    # Transform back to csc format and fix node ordering.
    g.face_nodes = g.face_nodes.tocsc()
    g.face_nodes.indices = g.face_nodes.indices[iv]  # For fast row operation

    return node_count


def sort_sub_list(indices, indptr):
    ix = np.zeros(indices.size, dtype=int)
    for i in range(indptr.size - 1):
        sub_ind = slice(indptr[i], indptr[i + 1])
        loc_ix = np.argsort(indices[sub_ind])
        ix[sub_ind] = loc_ix + indptr[i]
    indices = indices[ix]
    iv = np.zeros(indices.size, dtype=int)
    iv[ix] = np.arange(indices.size)
    return indices, iv


def find_cell_color(g, cells):
    """
    Color the cells depending on the cell connections. Each group of cells
    that are connected (either directly by a shared face or through a series
    of shared faces of many cells) is are given different colors.
           c_1-c_3     c_4
         /
       c_7  |           |
         \
           c_2         c_5
    In this case, cells c_1, c_2, c_3 and c_7 will be given color 0, while
    cells c_4 and c_5 will be given color 1.

    Parameters:
    ----------
    g        - Grid for which the cells belong
    cells    - indecies of cells (=np.array([1,2,3,4,5,7]) for case above)
    """
    c = np.sort(cells)
    # Local cell-face and face-node maps.
    assert g.cell_faces.getformat() == "csc"
    cell_faces = sparse_mat.slice_mat(g.cell_faces, c)
    child_cell_ind = -np.ones(g.num_cells, dtype=np.int)
    child_cell_ind[c] = np.arange(cell_faces.shape[1])

    # Create a copy of the cell-face relation, so that we can modify it at
    # will
    # com Runar: I don't think this is neccessary as slice_mat creates a copy
    #    cell_faces = cf_sub.copy()

    # Direction of normal vector does not matter here, only 0s and 1s
    cell_faces.data = np.abs(cell_faces.data)

    # Find connection between cells via the cell-face map
    c2c = cell_faces.transpose() * cell_faces
    # Only care about absolute values
    c2c.data = np.clip(c2c.data, 0, 1).astype("bool")

    graph = Graph(c2c)
    graph.color_nodes()
    return graph.color[child_cell_ind[cells]]


def avg_normal(g, faces):
    """
    Calculates the average face normal of a set of faces. The average normal
    is only constructed from the boundary faces, that is, a face thatbelongs
    to exactly one cell. If a face is not a boundary face, it will be ignored.
    The faces normals are fliped such that they point out of the cells.

    Parameters:
    ----------
    g         - Grid
    faces     - Face indecies of face normals that should be averaged
    """
    frac_face = np.ravel(np.sum(np.abs(g.cell_faces[faces, :]), axis=1) == 1)
    f, _, sign = sps.find(g.cell_faces[faces[frac_face], :])
    n = g.face_normals[:, faces[frac_face]]
    n = n[:, f] * sign
    n = np.mean(n, axis=1)
    n = n / np.linalg.norm(n)
    return n


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
