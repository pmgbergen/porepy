"""
Module for splitting a grid at the fractures.
"""
from typing import List, Optional

import networkx as nx
import numpy as np
from scipy import sparse as sps

import porepy as pp
from porepy.utils import setmembership, sparse_mat, tags
from porepy.utils.graph import Graph
from porepy.utils.half_space import half_space_int
from porepy.utils.mcolon import mcolon

module_sections = ["gridding"]


@pp.time_logger(sections=module_sections)
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
        gl_2_gh_nodes = []
        for g in gl:
            source = np.atleast_2d(g.global_point_ind).astype(np.int32)
            target = np.atleast_2d(gh.global_point_ind).astype(np.int32)
            _, mapping = setmembership.ismember_rows(source, target)
            gl_2_gh_nodes.append(mapping)

        split_nodes(gh, gl, gl_2_gh_nodes, offset)

    # Remove zeros from cell_faces

    [g.cell_faces.eliminate_zeros() for g, _ in bucket]
    [g.update_boundary_node_tag() for g, _ in bucket]
    return bucket


@pp.time_logger(sections=module_sections)
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
        # We first duplicate faces along tagged faces. The duplicate
        # faces will share the same nodes as the original faces,
        # however, the new faces are not yet added to the cell_faces map
        # (to save computation).
        face_id = duplicate_faces(gh, face_cells[i])
        face_cells = _update_face_cells(face_cells, face_id, i)
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


@pp.time_logger(sections=module_sections)
def split_specific_faces(
    gh: pp.Grid,
    face_cell_list: List[sps.spmatrix],
    faces: np.ndarray,
    cells: np.ndarray,
    gl_ind: np.ndarray,
    non_planar: bool = False,
):
    """
    For a given pair of gh and gl.
    Split only the faces specified by faces (higher-dimensional), corresponding
    to new cells (lower-dimensional). gl_ind identifies gl in
    face_cell_list, i.e.
        face_cell_list[gl_ind] = gb.edge_props((gh, gl), 'face_cells')
    """
    # Idea behind this loop is not clear. Likely change, in case we invoke this
    # function for several g_l is to send fc, not face_cell_list to update_face_cells.
    # However, the implications of this, e.g. with updates of face indices etc.
    # are not clear
    for f_c in face_cell_list:
        # We first we duplicate faces along tagged faces. The duplicate
        # faces will share the same nodes as the original faces,
        # however, the new faces are not yet added to the cell_faces map
        # (to save computation).
        face_id = _duplicate_specific_faces(gh, faces)

        # Update the mapping between higher-dimensional faces and lower-dimensional
        # cells.
        face_cell_list = _update_face_cells(face_cell_list, face_id, gl_ind, cells)
        if face_id.size == 0:
            return face_cell_list

        # We now set the cell_faces map based on which side of the
        # fractures the cells lie.
        if non_planar:
            # In this case, a loop over the elements in face_id should do the job.
            flag_array: List[int] = []
            for fi in face_id:
                n = np.reshape(gh.face_normals[:, fi], (3, 1))
                n = n / np.linalg.norm(n)
                x0 = np.reshape(gh.face_centers[:, fi], (3, 1))
                this_flag: int = update_cell_connectivity(gh, np.array([fi]), n, x0)
                flag_array.append(this_flag)

            if np.allclose(np.asarray(flag_array), 0):
                flag = 0
            elif np.allclose(np.asarray(flag_array), -1):
                flag = -1
            else:
                # Not sure what to do here - probably a partial update of connectivity
                raise ValueError("Split only some faces in non-planar ")
        else:
            # The fracture is considered flat, we can use the same normal vector
            # for all faces. This should make the computations faster
            n = np.reshape(gh.face_normals[:, face_id[0]], (3, 1))
            n = n / np.linalg.norm(n)
            x0 = np.reshape(gh.face_centers[:, face_id[0]], (3, 1))
            flag = update_cell_connectivity(gh, face_id, n, x0)

        if flag == 0:
            # if flag== 0 we added left and right faces (if it is -1 no faces
            # was added and we don't have left and right face pairs).
            # we now add the new faces to the frac_pair array.
            left = face_id
            right = np.arange(gh.num_faces - face_id.size, gh.num_faces)
            gh.frac_pairs = np.hstack((gh.frac_pairs, np.vstack((left, right))))
        return face_cell_list


@pp.time_logger(sections=module_sections)
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

    # Update the number of nodes
    gh.num_nodes = gh.num_nodes + node_count  # - nodes.size

    return True


@pp.time_logger(sections=module_sections)
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
    return _duplicate_specific_faces(gh, frac_id)


@pp.time_logger(sections=module_sections)
def _duplicate_specific_faces(gh: pp.Grid, frac_id: np.ndarray) -> np.ndarray:
    """
    Duplicate faces of gh specified by frac_id.
    """

    # Find which of the faces to split are tagged with a standard face tag,
    # that is, as fracture, tip or domain_boundary
    rem = tags.all_face_tags(gh.tags)[frac_id]

    # Set the faces to be split to fracture faces
    # Q: Why only if the face already had a tag (e.g., why [rem])?
    # Possible answer: We wil not split them (see redefinition of frac_id below),
    # but want them to be tagged as fracture_faces
    gh.tags["fracture_faces"][frac_id[rem]] = True
    # Faces to be split should not be tip
    gh.tags["tip_faces"][frac_id] = False

    # Only consider previously untagged faces for splitting
    frac_id = frac_id[~rem]
    if frac_id.size == 0:
        return frac_id

    # Expand the face-node relation to include duplicated nodes
    # Do this by directly manipulating the CSC-format of the matrix
    # Nodes of the target faces
    node_start = gh.face_nodes.indptr[frac_id]
    node_end = gh.face_nodes.indptr[frac_id + 1]
    nodes = gh.face_nodes.indices[mcolon(node_start, node_end)]

    # Start point for the new columns. They will be appended to the matrix, thus
    # the offset of the previous size of gh.face_nodes
    added_node_pos = np.cumsum(node_end - node_start) + gh.face_nodes.indptr[-1]
    # Sanity checks
    assert added_node_pos.size == frac_id.size
    assert added_node_pos[-1] - gh.face_nodes.indptr[-1] == nodes.size
    # Expand row-data by adding node indices
    gh.face_nodes.indices = np.hstack((gh.face_nodes.indices, nodes))
    # Expand column pointers
    gh.face_nodes.indptr = np.hstack((gh.face_nodes.indptr, added_node_pos))
    # Expand data array
    gh.face_nodes.data = np.hstack(
        (gh.face_nodes.data, np.ones(nodes.size, dtype=bool))
    )
    # Update matrix shape
    gh.face_nodes._shape = (gh.num_nodes, gh.face_nodes.shape[1] + frac_id.size)
    assert gh.face_nodes.indices.size == gh.face_nodes.indptr[-1]

    # We also copy the attributes of the original faces.
    gh.num_faces += frac_id.size
    gh.face_normals = np.hstack((gh.face_normals, gh.face_normals[:, frac_id]))
    gh.face_areas = np.append(gh.face_areas, gh.face_areas[frac_id])
    gh.face_centers = np.hstack((gh.face_centers, gh.face_centers[:, frac_id]))

    # Not sure if this still does the correct thing. Might have to
    # send in a logical array instead of frac_id.
    gh.tags["fracture_faces"][frac_id] = True
    gh.tags["tip_faces"][frac_id] = False
    update_fields = gh.tags.keys()
    update_values: List[List[np.ndarray]] = [[]] * len(update_fields)
    for i, key in enumerate(update_fields):
        # faces related tags are doubled and the value is inherit from the original
        if key.endswith("_faces"):
            update_values[i] = gh.tags[key][frac_id]
    tags.append_tags(gh.tags, update_fields, update_values)

    return frac_id


@pp.time_logger(sections=module_sections)
def _update_face_cells(
    face_cells: List[sps.spmatrix],
    face_id: np.ndarray,
    i: int,
    cell_id: Optional[np.ndarray] = None,
) -> List[sps.spmatrix]:
    """
    Add duplicate faces to connection map between lower-dim grids
    and higher dim grids. To be run after duplicate_faces.
    cell_id refers to new lower-dimensional cells, e.g. after fracture
    propagation. In this case, face_id[i] is the "parent" face of cell_id[i].

    TODO: Consider replacing hstack and vstack by sparse_mat.stack_mat.

    Parameters:
        face_cells: List of face-cell relation between a higher-dimensional grid
            and all its lower-dimensional neighbors.
        face_id: Faces to be duplicated in the face-cell relation.
        i: Index of the lower-dimensional grid to be treated now
        cell_id: List of lower-dimensional cells added in fracture propagation.
            Only used for fracture propagation problem.

    """
    # We duplicated the faces associated with lower-dim grid i.
    # The duplications should also be associated with grid i.
    # For the other lower-dim grids we just add zeros to conserve
    # the right matrix dimensions.

    if face_id.size == 0:
        return face_cells

    # Loop over all lower-dimensional neighbors
    for j, f_c in enumerate(face_cells):
        assert f_c.getformat() == "csc"
        if j == i:
            # We hit the target neighbor
            # Pick out the part of f_c to be used with this neighbor
            f_c_sliced = sparse_mat.slice_mat(f_c, face_id)
            # The new face-cell relations are added to the end of the matrix
            # (since the faces were added to the end of the face arrays in
            # the higher-dimensional grid).
            # Columns (face-indices in the higher dimensional grid) must be
            # added, rows / indices and data are identical for the two
            new_indptr = f_c_sliced.indptr + f_c.indptr[-1]
            new_ind = f_c_sliced.indices
            new_data = f_c_sliced.data

            # Expand face-cell relation
            f_c.indptr = np.append(f_c.indptr, new_indptr[1:])
            f_c.indices = np.append(f_c.indices, new_ind)
            f_c.data = np.append(f_c.data, new_data)
            f_c._shape = (f_c._shape[0], f_c._shape[1] + face_id.size)

            # In cases where cells have been added to the lower-dimensional grid
            # Note that this will not happen for construction of a GridBucket through
            # the standard workflow of post-processing a gmsh grid.
            if cell_id is not None:
                # New cells have been added to gl. We will create a local
                # matrix for the new cells, and append this to the bottom of f_c
                new_rows = sps.csr_matrix((cell_id.size, f_c.shape[1]), dtype=bool)
                # Add connection between old faces and new cells
                # The new cells are (assumed to be) located and the end of the
                # cell array in the lower-dimensional grid
                local_cell_id = cell_id - f_c.shape[0]
                new_rows[local_cell_id, face_id] = True

                # Add connections between new faces and new cells
                new_face_id = np.arange(f_c.shape[1] - face_id.size, f_c.shape[1])
                new_rows[local_cell_id, new_face_id] = True
                # stack them
                f_c = sps.vstack((f_c.tocsr(), new_rows), format="csc")
        else:
            # This is not the target lower-dimensional grid. Add columns to f_c to
            # account for the new cells, but do not add connections.
            new_indptr = f_c.indptr[-1] * np.ones(face_id.size, dtype=f_c.indptr.dtype)
            f_c.indptr = np.append(f_c.indptr, new_indptr)
            f_c._shape = (f_c._shape[0], f_c._shape[1] + face_id.size)

        # Update this part of face_cells
        face_cells[j] = f_c

    # Done
    return face_cells


@pp.time_logger(sections=module_sections)
def update_cell_connectivity(
    g: pp.Grid, face_id: np.ndarray, normal: np.ndarray, x0: np.ndarray
) -> int:
    """
    After the faces in a grid are duplicated, we update the cell connectivity
    list. Cells on the right side of the fracture do not change, but the cells
    on the left side are attached to the face duplicates. We assume that all
    faces that have been duplicated lie in the same plane. This plane is
    described by a normal and a point, x0. We attach cell on the left side of
    the plane to the duplicate of face_id. The cell on the right side is
    attached to the face frac_id.

    Parameters:
    ----------
    g         - The grid for wich the cell_face mapping is uppdated
    frac_id   - Indices of the faces that have been duplicated
    normal    - Normal of faces that have been duplicated. Note that we assume
                that all faces have the same normal
    x0        - A point in the plane where the faces lie

    Returns:
    ----------
    int: Flag that informs on what action has been taken. 0 means g.cell_faces has been
        split. -1 means the fracture was on the boundary, and no action taken.

    Raises:
    ----------
    ValueError: If the fracture is not planar

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
    if sum(left_cell) * 2 != left_cell.size:
        raise ValueError(
            "Fractures must either be" "on boundary or completely innside domain"
        )

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

    # And then we add the new left-faces to the cell_face map. We do not
    # change the sign of the matrix since we did not flip the normals.
    # This means that the normals of right and left cells point in the same
    # direction, but their cell_faces values have oposite signs.
    sparse_mat.stack_mat(g.cell_faces, cell_frac_left)
    g.cell_faces = g.cell_faces.tocsc()

    return 0


@pp.time_logger(sections=module_sections)
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
    for key in tags.standard_face_tags():
        g.tags[key] = g.tags[key][keep]

    if rem_cell_faces:
        g.cell_faces = g.cell_faces[keep, :]


@pp.time_logger(sections=module_sections)
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
    # In the case of a non-zero offset (presumably intended for visualization), use a
    # (somewhat slow) legacy implementation which can handle this.
    if offset != 0:
        return _duplicate_nodes_with_offset(g, nodes, offset)

    # Nodes must be duplicated in the array of node coordinates. Moreover, the face-node
    # relation must be updated so that when a node is split in two or more, all faces on
    # each of the spitting lines / planes are assigned the same version / index of the
    # spit node. The modification of node numbering further means that the face-node relation
    # must be updated also for faces not directly involved in the splitting.
    #
    # The below implementation consists of the following major steps:
    # 1. Isolate clusters of cells surrounding each node to be split, and make connection maps
    #    that include only cells within each cluster.
    # 2. Use the connection map to further subdivide the clusters into parts that lay on
    #    different sides of dividing lines / planes.
    # 3. Modify the face-node relation by splitting nodes. Also update node numbering in
    #    unsplit nodes.
    # 4. Duplicate split nodes in the coordinate array.

    # Bookeeping etc.
    cell_node = g.cell_nodes().tocsr()
    face_node = g.face_nodes.tocsc()
    cell_face = g.cell_faces

    num_nodes_to_duplicate = nodes.size

    ## Step 1
    # Create a list where each item are the cells associated with a node to be expanded.
    cell_clusters = [np.unique(sparse_mat.slice_indices(cell_node, n)) for n in nodes]

    # Number of cells in each cluster.
    sz_cell_clusters = [c.size for c in cell_clusters]
    tot_sz = np.sum([sz_cell_clusters])

    # Create a mapping of cells from linear ordering to the clusters.
    # Separate variable for the rows - these will be used to map back from the cluster
    # cell numbering to the standard numbering
    rows_cell_map = np.hstack(cell_clusters)
    cell_map = sps.coo_matrix(
        (np.ones(tot_sz), (rows_cell_map, np.arange(tot_sz))),
        shape=(g.num_cells, tot_sz),
    ).tocsc()

    # Connection map between cells, limited to the cells included in the clusters.
    # Cells may occur more than once in the map (if several of the cell's nodes are to be
    # split) and there may be connections between cells associated with different nodes.
    cf_loc = cell_face * cell_map
    c2c = cf_loc.T * cf_loc
    # All non-zero data signifies connections; simplify the representation
    c2c.data = np.clip(np.abs(c2c.data), 0, 1)

    # The connection matrix is known to be symmetric, and we only need to handle the upper
    # triangular part
    c2c = sps.triu(c2c)

    # Remove matrix elements outside the blocks to decouple connections between cells
    # associated with different nodes. Do this by identifying elements in the sparse
    # storage format outside the blocks, and set their matrix values to zero.
    # This will leave a block diagonal connection matrix, one block per node.

    # All non-zero elements in c2c.
    row_c2c, col_c2c, dat_c2c = sps.find(c2c)

    # Get sorted (increasing columns) version of the matrix. This allows for iteration through
    # the columns of the matrix.
    sort_ind = np.argsort(col_c2c)
    sorted_rows = row_c2c[sort_ind]
    sorted_cols = col_c2c[sort_ind]
    sorted_data = dat_c2c[sort_ind]

    # Array to keep indices to remove
    remove_ind = np.zeros(sorted_rows.size, dtype=np.bool)

    # Array with the start of the blocks corresponding to each cluster.
    block_start = np.hstack((0, np.cumsum([sz_cell_clusters])))

    # Iteration index for the start of the column group in the matrix fields 'indices' and
    # 'data' (referring to the sparse storage).
    col_group_start: int = 0

    # Loop over all groups of columns (one group per node nodes). Find the matrix elements
    # of this block, take note of elements outside the column indices (these will be
    # couplings to other nodes).
    for bi in range(num_nodes_to_duplicate):
        # Data for this block ends with the first column that belongs to the next block.
        # Note that we only search from the start index of this block, and use this as
        # an offset (saves time).
        col_group_end: int = col_group_start + np.argmax(
            sorted_cols[col_group_start:] == block_start[bi + 1]
        )
        # Special case for the last iteration: the last element in block_start has value
        # one higher than the number of rows, thus the equality above is never met, and
        # argmax returns the first element in the comparison. Correct this to let the
        # slice run to the end of the arrays.
        if bi == num_nodes_to_duplicate - 1:
            col_group_end = sorted_cols.size

        # Indices of elements in these rows.
        block_inds = slice(col_group_start, col_group_end)

        # Rows that are outside this block
        outside = np.logical_or(
            sorted_rows[block_inds] < block_start[bi],
            sorted_rows[block_inds] >= block_start[bi + 1],
        )
        # Mark matrix elements belonging to outside rows for removal
        remove_ind[block_inds][outside] = 1

        # The end of this column group becomes the start of the next one.
        col_group_start = col_group_end

    # Remove all data outside the main blocks.
    sorted_data[remove_ind] = 0

    # Make a new, block-diagonal connection matrix.
    # IMPLEMENTATION NOTE: Going to a csc matrix should be straightforward,
    # since sc already is sorted. It is however not clear networkx will be faster
    # with a non-coo matrix.
    c2c_loc = sps.coo_matrix((sorted_data, (sorted_rows, sorted_cols)), shape=c2c.shape)
    # Drop all zero elements
    c2c_loc.eliminate_zeros()

    ## Step 2
    # Now the connection matrix only contains connection between cells that share a node
    # to be duplicated. These can again be split into subclusters, that have lost their
    # connections due to the previous splitting of faces.
    # Identify these subclusters by the use of networkx
    graph = nx.Graph(c2c_loc)
    subclusters = [sorted(list(c)) for c in nx.connected_components(graph)]

    # For each subcluster, find its associated node (to be split)
    node_of_subcluster = []
    search_start = 0
    for comp in subclusters:
        # Find the first element with index one too much, then subtract one.
        # See the above loop (col_group_end) for further comments.
        # Also note we could have used any element in comp.
        ind = search_start + np.argmax(block_start[search_start:] > comp[0]) - 1
        # Store this node index
        node_of_subcluster.append(ind)
        # Start of next search interval.
        search_start = ind

    node_of_component = np.array(node_of_subcluster)

    ## Step 3
    # Modify the face-node relation by adjusting the node indices (field indices in the
    # sparse storage of the matrix). The duplicated nodes are added right after the
    # original node in the node ordering. Two adjustments are thus needed: First the
    # insertion of extra nodes, second this insertion increases the index of all nodes
    # with higher index.

    # Copy node-indices in the face-node relation. The first copy will preserve the old
    # node ordering. The second will carry the local adjustments due to the
    old_node_ind = face_node.indices.copy()
    new_node_ind = face_node.indices.copy()

    # Loop over all the subclusters of cells. The faces of the cells that have the
    # associated node to be split have the node index increased, depending on how many
    # times the node has been encountered before.

    # Count the number of encounters for a node.
    node_occ = np.zeros(num_nodes_to_duplicate, dtype=int)

    # Loop over combination of nodes and subclusters
    for ni, comp in zip(node_of_component, subclusters):

        # If the increase in node index is zero, there is no need to do anything.
        if node_occ[ni] == 0:
            node_occ[ni] += 1
            continue

        # Map cell indexes from the ordering in the clusters back to global ordering
        loc_cells = rows_cell_map[comp]
        # Faces of these cells
        loc_faces = np.unique(sparse_mat.slice_indices(g.cell_faces, loc_cells))
        # Nodes of the faces, and indices in the sparse storage format where the nodes
        # are located.
        loc_nodes, data_ind = sparse_mat.slice_indices(
            face_node, loc_faces, return_array_ind=True
        )
        # Indices in the sparse storage that should be increased
        incr_ind = data_ind[loc_nodes == nodes[ni]]
        # Increase the node index according to previous encounters.
        new_node_ind[incr_ind] += node_occ[ni]
        # Take note of this iteration
        node_occ[ni] += 1

    # Count the number of repititions in the nodes: The unsplit nodes have 1, the split
    # depends on the number of identified subclusters
    repititions = np.ones(g.num_nodes, dtype=np.int32)
    repititions[nodes] = np.bincount(node_of_component)
    # The number of added nodes
    added = repititions - 1
    num_added = added.sum()

    # Array of cumulative increments due to the splitting of nodes with lower index.
    # Put a zero up front to make the adjustment for the nodes with higher index
    increment = np.cumsum(np.hstack((0, added)))

    # The new node indices are formed by combining the two sources of adjustment.
    # Both split and unsplit nodes are impacted by the increments.
    # The increments must be taken with respect to the old indices
    face_node.indices = (new_node_ind + increment[old_node_ind]).astype(np.int32)
    # Ensure the right format of the sparse storage. Somehow this got messed up somewhere.
    face_node.indptr = face_node.indptr.astype(np.int32)

    # Adjust the shape of face-nodes to account for the added nodes
    face_node._shape = (g.num_nodes + num_added, g.num_faces)

    # From the number of repititions of the node (1 for untouched nodes),
    # get mapping from new to old indices.
    # To see how this works, read the documentation of rldecode, including the examples.
    new_2_old_nodes = pp.utils.matrix_compression.rldecode(
        np.arange(repititions.size), repititions
    )
    g.nodes = g.nodes[:, new_2_old_nodes]
    # The global point ind is shared by all split nodes
    g.global_point_ind = g.global_point_ind[new_2_old_nodes]

    # Also map the tags for nodes that are on fracture tips if this is relevant
    # (that is, if the grid is of the highest dimension)
    keys = ["node_is_fracture_tip", "node_is_tip_of_some_fracture"]
    for key in keys:
        if hasattr(g, "tags") and key in g.tags:
            g.tags[key] = g.tags[key][new_2_old_nodes].astype(bool)

    return num_added


@pp.time_logger(sections=module_sections)
def _duplicate_nodes_with_offset(g: pp.Grid, nodes: np.ndarray, offset: float) -> int:
    """
    Duplicate nodes on a fracture, and perturb the duplicated nodes. This option
    is useful for visualization purposes.

    NOTE: This is a legacy implementation, which should not be invoked directly.
    Instead use duplicate nodes (more efficient, but without the possibility to
    perturb nodes); that method will invoke the present if a perturbation is
    requested.

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
    _, iv = _sort_sub_list(g.face_nodes.indices, g.face_nodes.indptr)

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

        cells = np.unique(sparse_mat.slice_indices(cell_nodes, node))
        # Find the color of each cell. A group of cells is given the same color
        # if they are connected by faces. This means that all cells on one side
        # of a fracture will have the same color, but a different color than
        # the cells on the other side of the fracture. Equivalently, the cells
        # at a X-intersection will be given four different colors
        colors = _find_cell_color(g, cells)
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
            colored_faces = np.unique(
                sparse_mat.slice_indices(g.cell_faces, cells[ix == j])
            )

            is_colored = np.in1d(faces_of_node_t, colored_faces, assume_unique=True)

            faces = np.append(faces, faces_of_node_t[is_colored])

            # These faces are then attached to new node number j.
            face_pos = np.append(face_pos, face_pos[-1] + np.sum(is_colored))

            # If an offset is given, we will change the position of the nodes.
            # We move the nodes a length of offset away from the fracture(s).
            if offset > 0 and colors.size > 1:
                new_nodes[:, j] -= _avg_normal(g, faces_of_node_t[is_colored]) * offset

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

        new_point_ind = np.array([g.global_point_ind[t_node]] * new_nodes.shape[1])
        g.global_point_ind = np.delete(g.global_point_ind, t_node)
        g.global_point_ind = np.insert(
            g.global_point_ind, [t_node] * new_point_ind.shape[0], new_point_ind, axis=0
        )

    # Transform back to csc format and fix node ordering.
    g.face_nodes = g.face_nodes.tocsc()
    g.face_nodes.indices = g.face_nodes.indices[iv]  # For fast row operation

    return node_count


@pp.time_logger(sections=module_sections)
def _sort_sub_list(indices, indptr):
    ix = np.zeros(indices.size, dtype=int)
    for i in range(indptr.size - 1):
        sub_ind = slice(indptr[i], indptr[i + 1])
        loc_ix = np.argsort(indices[sub_ind])
        ix[sub_ind] = loc_ix + indptr[i]
    indices = indices[ix]
    iv = np.zeros(indices.size, dtype=int)
    iv[ix] = np.arange(indices.size)
    return indices, iv


@pp.time_logger(sections=module_sections)
def _find_cell_color(g, cells):
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
    child_cell_ind = -np.ones(g.num_cells, dtype=int)
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


@pp.time_logger(sections=module_sections)
def _avg_normal(g, faces):
    """
    Calculates the average face normal of a set of faces. The average normal
    is only constructed from the boundary faces, that is, a face that belongs
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


@pp.time_logger(sections=module_sections)
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
