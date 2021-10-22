import unittest
from test import test_utils

import numpy as np

import porepy as pp


class ADTreeTest(unittest.TestCase):
    def test_simple_adtree(self):
        tree = pp.adtree.ADTree(1, 1)

        n0 = pp.adtree.ADTNode(0, 0.4)
        tree.add_node(n0)
        n1 = pp.adtree.ADTNode(1, 0.6)
        tree.add_node(n1)
        n2 = pp.adtree.ADTNode(2, 0.7)
        tree.add_node(n2)
        n3 = pp.adtree.ADTNode(3, 0.8)
        tree.add_node(n3)
        n4 = pp.adtree.ADTNode(4, 0.2)
        tree.add_node(n4)
        n5 = pp.adtree.ADTNode(5, 0.1)
        tree.add_node(n5)

        nodes = [n.key for n in tree.nodes]
        known_nodes = [0, 1, 2, 3, 4, 5]
        self.assertTrue(np.allclose(nodes, known_nodes))
        self.assertTrue(np.allclose(tree.nodes[0].child, [4, 1]))
        self.assertTrue(np.allclose(tree.nodes[1].child, [2, 3]))
        self.assertTrue(tree.nodes[2].child[0] is None)
        self.assertTrue(tree.nodes[2].child[1] is None)
        self.assertTrue(tree.nodes[3].child[0] is None)
        self.assertTrue(tree.nodes[3].child[1] is None)
        self.assertTrue(tree.nodes[4].child[0] == 5)
        self.assertTrue(tree.nodes[4].child[1] is None)
        self.assertTrue(tree.nodes[5].child[0] is None)
        self.assertTrue(tree.nodes[5].child[1] is None)

    def test_grid_1d_adtree(self):
        g = pp.CartGrid(5, 1)
        g.compute_geometry()

        tree = pp.adtree.ADTree(2, 1)
        tree.from_grid(g)

        nodes = [n.key for n in tree.nodes]
        known_nodes = [0, 1, 2, 3, 4]
        self.assertTrue(np.allclose(nodes, known_nodes))
        self.assertTrue(np.allclose(tree.nodes[0].child, [1, 3]))
        self.assertTrue(tree.nodes[1].child[0] is None)
        self.assertTrue(tree.nodes[1].child[1] == 2)
        self.assertTrue(tree.nodes[2].child[0] is None)
        self.assertTrue(tree.nodes[2].child[1] is None)
        self.assertTrue(tree.nodes[3].child[0] is None)
        self.assertTrue(tree.nodes[3].child[1] == 4)
        self.assertTrue(tree.nodes[4].child[0] is None)
        self.assertTrue(tree.nodes[4].child[1] is None)

        # point check
        n = pp.adtree.ADTNode(99, [0.1] * 2)
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [0]))

        # point check
        n = pp.adtree.ADTNode(99, [1.1] * 2)
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, []))

        # point check
        n = pp.adtree.ADTNode(99, [0.9] * 2)
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [4]))

        # point check
        n = pp.adtree.ADTNode(99, [0.2] * 2)
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [0, 1]))

        # point check
        n = pp.adtree.ADTNode(99, [0.8] * 2)
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [3, 4]))

        # interval check
        n = pp.adtree.ADTNode(99, [0.1, 0.3])
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [0, 1]))

        # interval check
        n = pp.adtree.ADTNode(99, [0.3, 0.7])
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [1, 2, 3]))

        # interval check
        n = pp.adtree.ADTNode(99, [-1.0, 2.0])
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [0, 1, 2, 3, 4]))

    def test_grid_2d_adtree(self):
        g = pp.CartGrid([3] * 2, [1] * 2)
        g.compute_geometry()

        tree = pp.adtree.ADTree(4, 2)
        tree.from_grid(g)

        # point check
        n = pp.adtree.ADTNode(99, [0.1] * 4)
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [0]))

        # point check
        n = pp.adtree.ADTNode(99, [0.8] * 4)
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [8]))

        # point check
        n = pp.adtree.ADTNode(99, [1.1] * 4)
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, []))

        # point check
        n = pp.adtree.ADTNode(99, [0.1, 1.0 / 3.0] * 2)
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [0, 3]))

        # point check
        n = pp.adtree.ADTNode(99, [1.0 / 3.0] * 4)
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [0, 1, 3, 4]))

        # point check
        n = pp.adtree.ADTNode(99, [1.0] * 4)
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [8]))

        # segment check
        n = pp.adtree.ADTNode(99, [0.1, 0.1, 0.5, 0.1])
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [0, 1]))

        # segment check
        n = pp.adtree.ADTNode(99, [0.8, 0.1, 0.8, 0.8])
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [2, 5, 8]))

        # segment check
        n = pp.adtree.ADTNode(99, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.8])
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [0, 1, 3, 4, 6, 7]))

        # area check
        n = pp.adtree.ADTNode(99, [0.1, 0.1, 0.2, 0.2])
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [0]))

        # area check
        n = pp.adtree.ADTNode(99, [0.8, 0.4, 0.84, 0.43])
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [5]))

        # area check
        n = pp.adtree.ADTNode(99, [0.4, 0.2, 0.8, 0.25])
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [1, 2]))

        # area check
        n = pp.adtree.ADTNode(99, [0.1, 0.1, 0.8, 0.8])
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [0, 1, 2, 3, 4, 5, 6, 7, 8]))

        # area check
        n = pp.adtree.ADTNode(99, [0.5, 0.5, 0.8, 0.8])
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [4, 5, 7, 8]))

        # area check
        n = pp.adtree.ADTNode(99, [1.0 / 3.0, 1.0 / 3.0, 0.5, 2.0 / 3.0])
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [0, 1, 3, 4, 6, 7]))

    def test_grid_3d_adtree(self):
        g = pp.CartGrid([3] * 3, [1] * 3)
        g.compute_geometry()

        tree = pp.adtree.ADTree(6, 3)
        tree.from_grid(g)

        # point check
        n = pp.adtree.ADTNode(99, [0.1] * 6)
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [0]))

        # point check
        n = pp.adtree.ADTNode(99, [0.8] * 6)
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [26]))

        # point check
        n = pp.adtree.ADTNode(99, [1.1] * 6)
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, []))

        # point check
        n = pp.adtree.ADTNode(99, [0.1, 1 / 3, 0.1] * 2)
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [0, 3]))

        # point check
        n = pp.adtree.ADTNode(99, [0.1, 1 / 3, 0.8] * 2)
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [18, 21]))

        # point check
        n = pp.adtree.ADTNode(99, [1.0 / 3.0] * 6)
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [0, 1, 3, 4, 9, 10, 12, 13]))

        # point check
        n = pp.adtree.ADTNode(99, [1.0] * 6)
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [26]))

        # segment check
        n = pp.adtree.ADTNode(99, [0.1, 0.1, 0.1, 0.5, 0.1, 0.1])
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [0, 1]))

        # segment check
        n = pp.adtree.ADTNode(99, [0.8, 0.1, 0.5, 0.8, 0.8, 0.5])
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [11, 14, 17]))

        # segment check
        n = pp.adtree.ADTNode(99, [0.5, 0.8, 0.1, 0.5, 0.8, 0.8])
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [7, 16, 25]))

        # segment check
        n = pp.adtree.ADTNode(
            99, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.8, 1.0 / 3.0]
        )
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16]))

        # area check
        n = pp.adtree.ADTNode(99, [0.1, 0.1, 0.1, 0.2, 0.2, 0.1])
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [0]))

        # area check
        n = pp.adtree.ADTNode(99, [0.8, 0.4, 0.5, 0.84, 0.43, 0.5])
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [14]))

        # area check
        n = pp.adtree.ADTNode(99, [0.4, 0.2, 0.5, 0.8, 0.2, 0.8])
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [10, 11, 19, 20]))

        # area check
        n = pp.adtree.ADTNode(99, [0.1, 0.1, 0.4, 0.8, 0.8, 0.4])
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [9, 10, 11, 12, 13, 14, 15, 16, 17]))

        # area check
        n = pp.adtree.ADTNode(99, [0.5, 0.5, 0.1, 0.8, 0.8, 0.1])
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [4, 5, 7, 8]))

        # area check
        n = pp.adtree.ADTNode(
            99, [1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 0.5, 2.0 / 3.0, 2.0 / 3.0]
        )
        n_nodes = tree.search(n)
        self.assertTrue(
            np.allclose(n_nodes, [9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25])
        )

        # volume check
        n = pp.adtree.ADTNode(99, [0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [0, 1, 3, 4, 9, 10, 12, 13]))

        # volume check
        n = pp.adtree.ADTNode(99, [0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
        n_nodes = tree.search(n)
        self.assertTrue(np.allclose(n_nodes, [0]))

    def test_grid_1d_adtree_partial(self):
        g = pp.CartGrid(10, 1)
        g.compute_geometry()

        tree = pp.adtree.ADTree(2, 1)
        which_cells = 5 + np.arange(5)
        tree.from_grid(g, which_cells)

        nodes = [n.key for n in tree.nodes]
        known_nodes = [5, 6, 7, 8, 9]
        self.assertTrue(np.allclose(nodes, known_nodes))
        self.assertTrue(np.allclose(tree.nodes[0].child, [1, 3]))
        self.assertTrue(tree.nodes[1].child[0] is None)
        self.assertTrue(tree.nodes[1].child[1] == 2)
        self.assertTrue(tree.nodes[2].child[0] is None)
        self.assertTrue(tree.nodes[2].child[1] is None)
        self.assertTrue(tree.nodes[3].child[0] is None)
        self.assertTrue(tree.nodes[3].child[1] == 4)
        self.assertTrue(tree.nodes[4].child[0] is None)
        self.assertTrue(tree.nodes[4].child[1] is None)


if __name__ == "__main__":
    unittest.main()
