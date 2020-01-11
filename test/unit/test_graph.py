import unittest
import numpy as np
from scipy import sparse as sps

from porepy.utils import graph

# ------------------------------------------------------------------------------#


class BasicsTest(unittest.TestCase):

    # ------------------------------------------------------------------------------#

    def test_fully_connected_graph(self):
        node_connections = np.array(
            [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]]
        )
        node_connections = sps.csc_matrix(node_connections)
        G = graph.Graph(node_connections)
        G.color_nodes()
        self.assertTrue(np.all(G.color == 0))
        self.assertTrue(G.regions == 1)

    # ------------------------------------------------------------------------------#

    # ------------------------------------------------------------------------------#

    def test_two_region_graph(self):
        node_connections = np.array(
            [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]]
        )
        node_connections = sps.csc_matrix(node_connections)
        G = graph.Graph(node_connections)
        G.color_nodes()
        self.assertTrue(np.all(G.color[:2] == 0))
        self.assertTrue(G.color[3] == 1)
        self.assertTrue(G.regions == 2)

    # ------------------------------------------------------------------------------#

    # ------------------------------------------------------------------------------#

    def test_nodes_connected_to_self(self):
        node_connections = np.array(
            [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 1]]
        )
        node_connections = sps.csc_matrix(node_connections)
        G = graph.Graph(node_connections)
        G.color_nodes()
        self.assertTrue(np.all(G.color[:2] == 0))
        self.assertTrue(G.color[3] == 1)
        self.assertTrue(G.regions == 2)


# ------------------------------------------------------------------------------#
