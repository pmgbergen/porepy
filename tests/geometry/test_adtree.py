"""
Tests for ADTree usage. We cover the following points:
    * Add elements in the tree, not related to a grid
    * Add elements from a 1d grid, and search for 0d and 1d objects: points and segments
    * Add elements from a 2d grid, and search for 0d, 1d and 2d objects
    * Add elements from a 3d grid, and search for 0d, 1d, 2d and 3d objects
    * Add only some elements from a 1d grid
    In all the cases, we consider also objects that might be shared by more than one
    cell in the corresponding grid.
"""

import numpy as np

import porepy as pp
from porepy import adtree


def test_simple_adtree():
    """Test a simple ADTree with elements added not from a grid.
    We check if the resulting tree is consistent.
    """
    tree = adtree.ADTree(1, 1)

    n0 = adtree.ADTNode(0, 0.4)
    tree.add_node(n0)
    n1 = adtree.ADTNode(1, 0.6)
    tree.add_node(n1)
    n2 = adtree.ADTNode(2, 0.7)
    tree.add_node(n2)
    n3 = adtree.ADTNode(3, 0.8)
    tree.add_node(n3)
    n4 = adtree.ADTNode(4, 0.2)
    tree.add_node(n4)
    n5 = adtree.ADTNode(5, 0.1)
    tree.add_node(n5)

    nodes = [n.key for n in tree.nodes]
    known_nodes = [0, 1, 2, 3, 4, 5]
    assert np.allclose(nodes, known_nodes)
    assert np.allclose(tree.nodes[0].child, [4, 1])
    assert np.allclose(tree.nodes[1].child, [2, 3])
    assert tree.nodes[2].child[0] == -1
    assert tree.nodes[2].child[1] == -1
    assert tree.nodes[3].child[0] == -1
    assert tree.nodes[3].child[1] == -1
    assert tree.nodes[4].child[0] == 5
    assert tree.nodes[4].child[1] == -1
    assert tree.nodes[5].child[0] == -1
    assert tree.nodes[5].child[1] == -1


def test_grid_1d_adtree():
    """Test a ADTree constructed from a 1d Grid.
    We check if the resulting tree is consistent and
    we search for points and segments.
    """

    g = pp.CartGrid(5, 1)
    g.compute_geometry()

    tree = adtree.ADTree(2, 1)
    tree.from_grid(g)

    nodes = [n.key for n in tree.nodes]
    known_nodes = [0, 1, 2, 3, 4]
    assert np.allclose(nodes, known_nodes)
    assert np.allclose(tree.nodes[0].child, [1, 3])
    assert tree.nodes[1].child[0] == -1
    assert tree.nodes[1].child[1] == 2
    assert tree.nodes[2].child[0] == -1
    assert tree.nodes[2].child[1] == -1
    assert tree.nodes[3].child[0] == -1
    assert tree.nodes[3].child[1] == 4
    assert tree.nodes[4].child[0] == -1
    assert tree.nodes[4].child[1] == -1

    # point check
    n = adtree.ADTNode(99, [0.1] * 2)
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [0])

    # point check
    n = adtree.ADTNode(99, [1.1] * 2)
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [])

    # point check
    n = adtree.ADTNode(99, [0.9] * 2)
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [4])

    # point check
    n = adtree.ADTNode(99, [0.2] * 2)
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [0, 1])

    # point check
    n = adtree.ADTNode(99, [0.8] * 2)
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [3, 4])

    # interval check
    n = adtree.ADTNode(99, [0.1, 0.3])
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [0, 1])

    # interval check
    n = adtree.ADTNode(99, [0.3, 0.7])
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [1, 2, 3])

    # interval check
    n = adtree.ADTNode(99, [-1.0, 2.0])
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [0, 1, 2, 3, 4])


def test_grid_2d_adtree():
    """Test a ADTree constructed from a 2d Cartesian Grid.
    We search for points, segments and 2d objects (not a-priori grid cells).
    """

    g = pp.CartGrid([3] * 2, [1] * 2)
    g.compute_geometry()

    tree = adtree.ADTree(4, 2)
    tree.from_grid(g)

    # point check
    n = adtree.ADTNode(99, [0.1] * 4)
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [0])

    # point check
    n = adtree.ADTNode(99, [0.8] * 4)
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [8])

    # point check
    n = adtree.ADTNode(99, [1.1] * 4)
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [])

    # point check
    n = adtree.ADTNode(99, [0.1, 1.0 / 3.0] * 2)
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [0, 3])

    # point check
    n = adtree.ADTNode(99, [1.0 / 3.0] * 4)
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [0, 1, 3, 4])

    # point check
    n = adtree.ADTNode(99, [1.0] * 4)
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [8])

    # segment check
    n = adtree.ADTNode(99, [0.1, 0.1, 0.5, 0.1])
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [0, 1])

    # segment check
    n = adtree.ADTNode(99, [0.8, 0.1, 0.8, 0.8])
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [2, 5, 8])

    # segment check
    n = adtree.ADTNode(99, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.8])
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [0, 1, 3, 4, 6, 7])

    # area check
    n = adtree.ADTNode(99, [0.1, 0.1, 0.2, 0.2])
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [0])

    # area check
    n = adtree.ADTNode(99, [0.8, 0.4, 0.84, 0.43])
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [5])

    # area check
    n = adtree.ADTNode(99, [0.4, 0.2, 0.8, 0.25])
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [1, 2])

    # area check
    n = adtree.ADTNode(99, [0.1, 0.1, 0.8, 0.8])
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [0, 1, 2, 3, 4, 5, 6, 7, 8])

    # area check
    n = adtree.ADTNode(99, [0.5, 0.5, 0.8, 0.8])
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [4, 5, 7, 8])

    # area check
    n = adtree.ADTNode(99, [1.0 / 3.0, 1.0 / 3.0, 0.5, 2.0 / 3.0])
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [0, 1, 3, 4, 6, 7])


def test_grid_3d_adtree():
    """Test a ADTree constructed from a 3d Cartesian Grid.
    We search for points, segments and 2d objects (not a-priori grid faces)
    and 3d objects (not a-priori grid cells).
    """

    g = pp.CartGrid([3] * 3, [1] * 3)
    g.compute_geometry()

    tree = adtree.ADTree(6, 3)
    tree.from_grid(g)

    # point check
    n = adtree.ADTNode(99, [0.1] * 6)
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [0])

    # point check
    n = adtree.ADTNode(99, [0.8] * 6)
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [26])

    # point check
    n = adtree.ADTNode(99, [1.1] * 6)
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [])

    # point check
    n = adtree.ADTNode(99, [0.1, 1 / 3, 0.1] * 2)
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [0, 3])

    # point check
    n = adtree.ADTNode(99, [0.1, 1 / 3, 0.8] * 2)
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [18, 21])

    # point check
    n = adtree.ADTNode(99, [1.0 / 3.0] * 6)
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [0, 1, 3, 4, 9, 10, 12, 13])

    # point check
    n = adtree.ADTNode(99, [1.0] * 6)
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [26])

    # segment check
    n = adtree.ADTNode(99, [0.1, 0.1, 0.1, 0.5, 0.1, 0.1])
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [0, 1])

    # segment check
    n = adtree.ADTNode(99, [0.8, 0.1, 0.5, 0.8, 0.8, 0.5])
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [11, 14, 17])

    # segment check
    n = adtree.ADTNode(99, [0.5, 0.8, 0.1, 0.5, 0.8, 0.8])
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [7, 16, 25])

    # segment check
    n = adtree.ADTNode(99, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.8, 1.0 / 3.0])
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16])

    # area check
    n = adtree.ADTNode(99, [0.1, 0.1, 0.1, 0.2, 0.2, 0.1])
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [0])

    # area check
    n = adtree.ADTNode(99, [0.8, 0.4, 0.5, 0.84, 0.43, 0.5])
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [14])

    # area check
    n = adtree.ADTNode(99, [0.4, 0.2, 0.5, 0.8, 0.2, 0.8])
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [10, 11, 19, 20])

    # area check
    n = adtree.ADTNode(99, [0.1, 0.1, 0.4, 0.8, 0.8, 0.4])
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [9, 10, 11, 12, 13, 14, 15, 16, 17])

    # area check
    n = adtree.ADTNode(99, [0.5, 0.5, 0.1, 0.8, 0.8, 0.1])
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [4, 5, 7, 8])

    # area check
    n = adtree.ADTNode(99, [1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 0.5, 2.0 / 3.0, 2.0 / 3.0])
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25])

    # volume check
    n = adtree.ADTNode(99, [0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [0, 1, 3, 4, 9, 10, 12, 13])

    # volume check
    n = adtree.ADTNode(99, [0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
    n_nodes = tree.search(n)
    assert np.allclose(n_nodes, [0])


def test_grid_1d_adtree_partial():
    """Test a ADTree constructed from a subset of 1d Grid cells.
    We check if the resulting tree is consistent.
    """

    g = pp.CartGrid(10, 1)
    g.compute_geometry()

    tree = adtree.ADTree(2, 1)
    which_cells = 5 + np.arange(5)
    tree.from_grid(g, which_cells)

    nodes = [n.key for n in tree.nodes]
    known_nodes = [5, 6, 7, 8, 9]
    assert np.allclose(nodes, known_nodes)
    assert np.allclose(tree.nodes[0].child, [1, 3])
    assert tree.nodes[1].child[0] == -1
    assert tree.nodes[1].child[1] == 2
    assert tree.nodes[2].child[0] == -1
    assert tree.nodes[2].child[1] == -1
    assert tree.nodes[3].child[0] == -1
    assert tree.nodes[3].child[1] == 4
    assert tree.nodes[4].child[0] == -1
    assert tree.nodes[4].child[1] == -1
