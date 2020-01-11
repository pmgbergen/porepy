from porepy.grids import grid_bucket
import copy
import scipy.sparse as sps
import numpy as np


def duplicate_without_dimension(gb, dim):
    """
    Remove all the nodes of dimension dim and add new edges between their
    neighbors by calls to remove_node.

    """
    gb1 = gb.copy()
    for g in gb1.grids_of_dimension(dim):
        remove_node(gb1, g)
    return gb1


# ------------------------------------------------------------------------------#


def remove_node(gb, node):
    """
    Remove the node (and the edges it partakes in) and add new direct
    connections (gb edges) between each of the neighbor pairs. A 0d node
    with n_neighbors gives rise to 1 + 2 + ... + n_neighbors-1 new edges.

    """
    neighbors = gb.node_neighbors(node)
    n_neighbors = len(neighbors)
    for i in range(n_neighbors - 1):
        n1 = neighbors[i]
        for j in range(i + 1, n_neighbors):
            n2 = neighbors[j]
            face_faces = find_shared_face(n1, n2, node, gb)

            gb.add_edge([n1, n2], face_faces)

    # Remove the node and update the ordering of the remaining nodes
    node_number = gb.node_prop(node, "node_number")
    gb.remove_node(node)
    gb.update_node_ordering(node_number)


# ------------------------------------------------------------------------------#


def find_shared_face(n1, n2, node, gb):
    """
    Given two 1d grids meeting at a 0d node (to be removed), find which two
    faces meet at the intersection (one from each grid). Returns the sparse
    matrix face_faces, the 1d-1d equivalent of the face_cells matrix.
    """
    # Sort nodes according to node_number
    n1, n2 = gb.sorted_nodes_of_edge([n1, n2])

    # Identify the faces connecting the neighbors to the grid to be removed
    fc1 = gb.edge_props([n1, node])
    fc2 = gb.edge_props([n2, node])
    _, face_number_1, _ = sps.find(fc1["face_cells"])
    _, face_number_2, _ = sps.find(fc2["face_cells"])

    # The lower dim. node (corresponding to the first dimension, cells,
    # in face_cells) is first in gb.sorted_nodes_of_edge. To be consistent
    # with this, the grid corresponding to the first dimension of face_faces
    # should be the first grid of the node sorting. Connect the two remaining
    # grids through the face_faces matrix, to be placed as a face_cells
    # substitute.
    face_faces = sps.csc_matrix(
        (np.array([True]), (face_number_1, face_number_2)), (n1.num_faces, n2.num_faces)
    )

    return face_faces


# ------------------------------------------------------------------------------#
