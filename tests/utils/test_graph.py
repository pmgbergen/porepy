"""Testing the graph implementation, number of regions and region coloring."""
import numpy as np
import pytest
from scipy import sparse as sps

from porepy.utils import graph


def test_fully_connected_graph():
    node_connections = np.array(
        [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]]
    )
    node_connections = sps.csc_matrix(node_connections)
    G = graph.Graph(node_connections)
    G.color_nodes()
    # graph is fully connected -> only one color and one region
    # default first entry of color must be overwritten with 0
    assert np.all(G.color == 0)
    assert G.regions == 1


@pytest.mark.parametrize(
    "node_connections",
    [
        np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]]),
        np.array([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 1]]),
    ],
)
def test_two_region_graph(node_connections):
    G = graph.Graph(sps.csc_matrix(node_connections))
    G.color_nodes()
    # graph has to regions, with different colors
    # first entry in color is the default entry (np.nan) which is overwritten
    # during color_nodes to the same color as the first region
    # (starting point of color coding)
    assert np.all(G.color[:2] == 0)
    assert G.color[3] == 1
    assert G.regions == 2
