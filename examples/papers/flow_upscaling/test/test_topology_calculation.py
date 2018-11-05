#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various implementation tests of the topology calculation. May be promoted to
a unit test in the future.
"""
import unittest
import numpy as np

from examples.papers.flow_upscaling.fracture_sets import count_node_types_between_families as top_compute
from examples.papers.flow_upscaling import fracture_sets



class TestTopologyComputation(unittest.TestCase):

    def _verify_shape(self, mat, sz):
        for m in mat:
            self.assertTrue( np.all(np.asarray(m.shape) == sz))


    def test_no_intersections_single_family(self):

        e = np.array([[0, 2], [1, 3]])
        top = top_compute(e)
        # All should have a single family
        self._verify_shape(top, 1)

        # All nodes should be i-nodes
        i_nodes = top[0]
        self.assertTrue( np.all(np.isin(i_nodes[0, 0], np.arange(4))))

        # All other elements should be empty
        for i in np.arange(1, 5):
            arr = top[i]
            self.assertTrue( arr[0, 0].size == 0)

    def test_no_intersections_two_families(self):

        e = np.array([[0, 2], [1, 3], [0, 1]])
        top = top_compute(e)
        # All should have a single family
        self._verify_shape(top, 2)

        # All nodes should be i-nodes
        i_nodes = top[0]
        self.assertTrue( i_nodes[0, 0].size == 2)
        self.assertTrue( np.all(np.isin(i_nodes[0, 0], np.array([0, 1]))))
        self.assertTrue( i_nodes[1, 1].size == 2)
        self.assertTrue( np.all(np.isin(i_nodes[1, 1], np.array([2, 3]))))

        # All other elements should be empty
        for i in np.arange(1, 5):
            arr = top[i]
            for r in range(arr.shape[0]):
                for c in range(arr.shape[1]):
                    self.assertTrue( arr[r, c].size == 0)


    def test_x_intersection_single_family(self):

        # Two lines crossing in an X
        e = np.array([[0, 1, 2, 3], [4, 4, 4, 4]])
        top = top_compute(e)
        self._verify_shape(top, 1)

        # The first four nodes should be i-nodes
        i_nodes = top[0]
        self.assertTrue( np.all(np.isin(i_nodes[0, 0], np.arange(4))))

        x_nodes = top[4]
        self.assertTrue( x_nodes[0, 0].size == 1)
        self.assertTrue( x_nodes[0, 0][0] == 4)

        # All other elements should be empty
        for i in [1, 2, 3]:
            arr = top[i]
            self.assertTrue( arr[0, 0].size == 0)

    def test_x_intersection_two_families(self):

        e = np.array([[0, 1, 2, 3], [4, 4, 4, 4], [0, 0, 1, 1]])
        top = top_compute(e)
        # All should have a single family
        self._verify_shape(top, 2)

        # All nodes should be i-nodes
        i_nodes = top[0]
        self.assertTrue( i_nodes[0, 0].size == 2)
        self.assertTrue( np.all(np.isin(i_nodes[0, 0], np.array([0, 1]))))
        self.assertTrue( i_nodes[1, 1].size == 2)
        self.assertTrue( np.all(np.isin(i_nodes[1, 1], np.array([2, 3]))))

        l_nodes = top[1]
        x_nodes = top[4]

        # When the families are not combined, the intersection looks like an L-node
        self.assertTrue( x_nodes[0, 0].size == 0)
        self.assertTrue( x_nodes[1, 1].size == 0)

        self.assertTrue( l_nodes[0, 0].size == 1)
        self.assertTrue( l_nodes[0, 0][0] == 4)
        self.assertTrue( l_nodes[1, 1].size == 1)
        self.assertTrue( l_nodes[1, 1][0] == 4)

        # When combining families, this is an x-connection
        self.assertTrue( l_nodes[1, 0].size == 0)
        self.assertTrue( l_nodes[0, 1].size == 0)

        self.assertTrue( x_nodes[0, 1].size == 1)
        self.assertTrue( x_nodes[0, 1][0] == 4)
        self.assertTrue( x_nodes[1, 0].size == 1)
        self.assertTrue( x_nodes[1, 0][0] == 4)

        # All other elements should be empty
        for i in [2, 3]:
            arr = top[i]
            for r in range(arr.shape[0]):
                for c in range(arr.shape[1]):
                    self.assertTrue( arr[r, c].size == 0)

    def test_y_intersection_single_family(self):

        e = np.array([[0, 1, 2], [3, 3, 3]])
        top = top_compute(e)
        self._verify_shape(top, 1)

        # The first three nodes should be i-nodes
        i_nodes = top[0]
        self.assertTrue( np.all(np.isin(i_nodes[0, 0], np.arange(3))))

        y_nodes_constrained = top[2]
        self.assertTrue( y_nodes_constrained[0, 0].size == 1)
        self.assertTrue( y_nodes_constrained[0, 0][0] == 3)

        y_nodes_full = top[3]
        self.assertTrue( y_nodes_full[0, 0].size == 1)
        self.assertTrue( y_nodes_full[0, 0][0] == 3)

        # All other elements should be empty
        for i in [1, 4]:
            arr = top[i]
            self.assertTrue( arr[0, 0].size == 0)


    def test_y_intersection_two_families(self):

        e = np.array([[0, 1, 2], [3, 3, 3], [0, 0, 1]])
        top = top_compute(e)
        self._verify_shape(top, 2)

        i_nodes = top[0]
        self.assertTrue( i_nodes[0, 0].size == 2)
        self.assertTrue( np.all(np.isin(i_nodes[0, 0], np.array([0, 1]))))
        self.assertTrue( i_nodes[1, 1].size == 2)
        self.assertTrue( np.all(np.isin(i_nodes[1, 1], np.array([2, 3]))))

        l_nodes = top[1]

        # When the families are not combined, the intersection looks like an L-node
        self.assertTrue( l_nodes[0, 0].size == 1)
        self.assertTrue( l_nodes[0, 0][0] == 3)
        self.assertTrue( l_nodes[1, 1].size == 0)

        # When combining families, this is a T-connection
        y_nodes_constrained = top[2]
        y_nodes_full = top[3]
        self.assertTrue( y_nodes_constrained[1, 0].size == 1)
        self.assertTrue( y_nodes_constrained[1, 0] == 3)
        self.assertTrue( y_nodes_full[1, 0].size == 0)

        self.assertTrue( y_nodes_constrained[0, 1].size == 0)
        self.assertTrue( y_nodes_full[0, 1].size == 1)
        self.assertTrue( y_nodes_full[0, 1][0] == 3)

        # All other elements should be empty
        for i in [4]:
            arr = top[i]
            for r in range(arr.shape[0]):
                for c in range(arr.shape[1]):
                    self.assertTrue( arr[r, c].size == 0)

class TestFractureSetIntersectionComputation(unittest.TestCase):

    def define_set(self, p, e):
        domain = {'xmin': p[0].min(), 'xmax': p[0].max(),
                  'ymin': p[1].min(), 'ymax': p[1].max()}
        return fracture_sets.FractureSet(p, e, domain)

    def test_no_intersections_single_family(self):

        e = np.array([[0, 2], [1, 3]])
        p = np.array([[0, 1, 0, 1],
                      [0, 0, 1, 1]])

        fracs = self.define_set(p, e)
        node_types = fracture_sets.analyze_intersections_of_sets(fracs)

        self.assertTrue( np.all(node_types["i_nodes"] == 2))
        self.assertTrue( np.all(node_types["y_nodes"] == 0))
        self.assertTrue( np.all(node_types["x_nodes"] == 0))
        self.assertTrue( np.all(node_types["arrests"] == 0))

    def test_x_intersection_single_family(self):

        e = np.array([[0, 2], [1, 3]])
        p = np.array([[-1, 1, 0, 0],
                      [0, 0, -1, 1]])

        fracs = self.define_set(p, e)
        node_types = fracture_sets.analyze_intersections_of_sets(fracs)

        self.assertTrue( np.all(node_types["i_nodes"] == 2))
        self.assertTrue( np.all(node_types["y_nodes"] == 0))
        self.assertTrue( np.all(node_types["x_nodes"] == 1))
        self.assertTrue( np.all(node_types["arrests"] == 0))


    def test_y_intersection_single_family(self):

        e = np.array([[0, 2], [1, 3]])
        p = np.array([[0, 2, 1, 1],
                      [0, 0, 0, 1]])

        fracs = self.define_set(p, e)
        node_types = fracture_sets.analyze_intersections_of_sets(fracs)

        assert np.all(node_types["i_nodes"] == np.array([2, 1]))
        self.assertTrue( np.all(node_types["y_nodes"] == np.array([0, 1])))
        self.assertTrue( np.all(node_types["x_nodes"] == 0))
        self.assertTrue( np.all(node_types["arrests"] == np.array([1, 0])))


    def test_no_intersections_two_families(self):

        e_1 = np.array([[0], [1]])
        p_1 = np.array([[0, 1],
                      [0, 0]])
        fracs_1 = self.define_set(p_1, e_1)

        e_2 = np.array([[0], [1]])
        p_2 = np.array([[0, 1],
                      [1, 1]])
        fracs_2 = self.define_set(p_2, e_2)

        node_types_1, node_types_2 = fracture_sets.analyze_intersections_of_sets(fracs_1, fracs_2)

        self.assertTrue( np.all(node_types_1["i_nodes"] == 2))
        self.assertTrue( np.all(node_types_1["y_nodes"] == 0))
        self.assertTrue( np.all(node_types_1["x_nodes"] == 0))
        self.assertTrue( np.all(node_types_1["arrests"] == 0))

        self.assertTrue( np.all(node_types_2["i_nodes"] == 2))
        self.assertTrue( np.all(node_types_2["y_nodes"] == 0))
        self.assertTrue( np.all(node_types_2["x_nodes"] == 0))
        self.assertTrue( np.all(node_types_2["arrests"] == 0))


    def test_x_intersection_two_families(self):
        e_1 = np.array([[0], [1]])
        p_1 = np.array([[-1, 1],
                        [0, 0]])
        fracs_1 = self.define_set(p_1, e_1)

        e_2 = np.array([[0], [1]])
        p_2 = np.array([[0, 0],
                       [-1, 1]])
        fracs_2 = self.define_set(p_2, e_2)

        node_types_1, node_types_2 = fracture_sets.analyze_intersections_of_sets(fracs_1, fracs_2)

        self.assertTrue( np.all(node_types_1["i_nodes"] == 2))
        self.assertTrue( np.all(node_types_1["y_nodes"] == 0))
        self.assertTrue( np.all(node_types_1["x_nodes"] == 1))
        self.assertTrue( np.all(node_types_1["arrests"] == 0))

        self.assertTrue( np.all(node_types_2["i_nodes"] == 2))
        self.assertTrue( np.all(node_types_2["y_nodes"] == 0))
        self.assertTrue( np.all(node_types_2["x_nodes"] == 1))
        self.assertTrue( np.all(node_types_2["arrests"] == 0))


    def test_y_intersection_two_families(self):

        e_1 = np.array([[0], [1]])
        p_1 = np.array([[0, 2],
                        [0, 0]])
        fracs_1 = self.define_set(p_1, e_1)

        e_2 = np.array([[0], [1]])
        p_2 = np.array([[1, 1],
                        [0, 1]])
        fracs_2 = self.define_set(p_2, e_2)

        node_types_1, node_types_2 = fracture_sets.analyze_intersections_of_sets(fracs_1, fracs_2)

        self.assertTrue( np.all(node_types_1["i_nodes"] == 2))
        self.assertTrue( np.all(node_types_1["y_nodes"] == 0))
        self.assertTrue( np.all(node_types_1["x_nodes"] == 0))
        self.assertTrue( np.all(node_types_1["arrests"] == 1))

        self.assertTrue( np.all(node_types_2["i_nodes"] == 1))
        self.assertTrue( np.all(node_types_2["y_nodes"] == 1))
        self.assertTrue( np.all(node_types_2["x_nodes"] == 0))
        self.assertTrue( np.all(node_types_2["arrests"] == 0))

    def test_y_intersection_two_families_flip_set_order(self):

        e_1 = np.array([[0], [1]])
        p_1 = np.array([[0, 2],
                        [0, 0]])
        fracs_1 = self.define_set(p_1, e_1)

        e_2 = np.array([[0], [1]])
        p_2 = np.array([[1, 1],
                        [0, 1]])
        fracs_2 = self.define_set(p_2, e_2)

        node_types_2, node_types_1 = fracture_sets.analyze_intersections_of_sets(fracs_2, fracs_1)

        self.assertTrue( np.all(node_types_1["i_nodes"] == 2))
        self.assertTrue( np.all(node_types_1["y_nodes"] == 0))
        self.assertTrue( np.all(node_types_1["x_nodes"] == 0))
        self.assertTrue( np.all(node_types_1["arrests"] == 1))

        self.assertTrue( np.all(node_types_2["i_nodes"] == 1))
        self.assertTrue( np.all(node_types_2["y_nodes"] == 1))
        self.assertTrue( np.all(node_types_2["x_nodes"] == 0))
        self.assertTrue( np.all(node_types_2["arrests"] == 0))


if __name__ == '__main__':
    unittest.main()