import numpy as np

from porepy.utils import comp_geom as cg
from porepy.fracs.fractures import EllipticFracture, Fracture


def _intersection_by_num_node(edges, num):
    """ Find all edges involved in intersections with a certain number of
    intersecting lines.

    Parameters:
        edges: fractures
        num: Target number of nodes in intersections

    Returns:
        nodes: Nodes with the prescribed number of edges meeting.
        edges_of_nodes (n x num): Each row gives edges meeting in a node.

    """
    num_occ = np.bincount(edges[:2].ravel())
    abutments = np.where(num_occ == num)[0]

    num_abut = abutments.size

    edges_of_nodes = np.zeros((num_abut, num), dtype=np.int)
    for i, pi in enumerate(abutments):
        edges_of_nodes[i] = np.where(np.any(edges[:2] == pi, axis=0))[0]
    return nodes, edges_of_nodes


def t_intersections(edges):
    """ Find points involved in T-intersections.

    A t-intersection is defined as a point involved in three fracture segments,
    of which two belong to the same fracture.

    The fractures should have been split (cg.remove_edge_crossings) before
    calling this function.

    Parameters:
        edges (np.array, 3 x n): Fractures. First two rows give indices of
            start and endpoints. Last row gives index of fracture that the
            segment belongs to.

    Returns:
        abutments (np.ndarray, int): indices of points that are
            T-intersections
        primal_frac (np.ndarray, int): Index of edges that are split by a
            t-intersection
        sec_frac (np.ndarray, int): Index of edges that ends in a
            T-intersection
        other_point (np.ndarray, int): For the secondary fractures, the end
            that is not in the t-intersection.

    """
    frac_num = edges[-1]
    abutments, edges_of_abutments = _intersection_by_num_node(edges, 3)

    num_abut = abutments.size
    primal_frac = np.zeros(num_abut, dtype=np.int)
    sec_frac = np.zeros(num_abut, dtype=np.int)
    for i, ei in enumerate(frac_num[edges_of_abutments]):
        fi, count = np.unique(ei, return_counts=True)
        assert fi.size == 2
        if count[0] == 1:
            primal_frac[i] = fi[1]
            sec_frac[i] = fi[0]
        else:
            primal_frac[i] = fi[0]
            sec_frac[i] = fi[1]

    other_point = np.zeros(num_abut, dtype=np.int)
    for i, pi in enumerate(abutments):
        if edges[0, sec_frac[i]] == pi:
            other_point[i] = edges[1, sec_frac[i]]
        else:
            other_point[i] = edges[0, sec_frac[i]]

    return abutments, primal_frac, sec_frac, other_point


def x_intersections(edges):
    """ Obtain nodes and edges involved in an x-intersection

    A x-intersection is defined as a point involved in four fracture segments,
    with two pairs belonging to two fractures each.

    The fractures should have been split (cg.remove_edge_crossings) before
    calling this function.

    Parameters:
        edges (np.array, 3 x n): Fractures. First two rows give indices of
            start and endpoints. Last row gives index of fracture that the
            segment belongs to.
    Returns:
        nodes: Index of nodes that form x intersections
        x_fracs (2xn): Index of fractures crossing in the nodes
        x_edges (4xn): Index of edges crossing in the nodes

    """
    frac_num = edges[-1]
    nodes, x_edges = _intersection_by_num_node(edges, 4)

    # Convert from edges (split fractures) to fractures themselves.
    num_x = nodes.size
    x_fracs = np.zeros((2, num_x))
    for i, ei in enumerate(frac_num[x_fracs]):
        x_fracs[:, i] = np.unique(ei)
    return nodes, x_fracs, x_edges

