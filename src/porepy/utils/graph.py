import numpy as np

import porepy as pp


class Graph:
    """
    A graph class.

    The graph class stores the nodes and edges of the graph in a sparse
    array (equivalently to face_nodes in the Grid class).

    Attributes:
        node_connections (sps.csc-matrix): Should be given at construction.
            node_node connections. Matrix size: num_nodes x num_nodes.
            node_connections[i,j] should be true if there is an edge
            connecting node i and j.
        regions (int) the number of regions. A region is a set of nodes
            that can be reached by traversing the graph. Two nodes are
            int different regions if they can not be reached by traversing
            the graph.
        color (int) the color of each region. Initialized as (NaN). By
            calling color_nodes() all nodes in a region are given the
            same colors and nodes in different regions are given different
            colors.
    """

    def __init__(self, node_connections):
        if node_connections.getformat() != "csr":
            self.node_connections = node_connections.tocsr()
        else:
            self.node_connections = node_connections
        self.regions = 0
        self.color = np.array([np.nan] * node_connections.shape[1])

    def color_nodes(self):
        """
        Color the nodes in each region
        """
        color = 0
        while self.regions <= self.node_connections.shape[1]:
            start = np.ravel(np.argwhere(np.isnan(self.color)))
            if start.size != 0:
                self.bfs(start[0], color)
                color += 1
                self.regions += 1
            else:
                return
        raise RuntimeWarning(
            "number of regions can not be greater than " "number of nodes"
        )

    def bfs(self, start, color):
        """
        Breadth first search
        """
        visited, queue = [], [start]
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.append(node)
                neighbors = pp.matrix_operations.slice_indices(
                    self.node_connections, node
                )
                queue.extend(neighbors)
        self.color[visited] = color
