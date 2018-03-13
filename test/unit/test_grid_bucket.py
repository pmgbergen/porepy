#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 13:33:32 2018

@author: eke001
"""
import unittest

import porepy as pp


class MockGrid():

    def __init__(self, dim=1):
        self.dim = dim




class TestBucket(unittest.TestCase):

    def simple_bucket(self, num_grids):
        gb = pp.GridBucket()

        [gb.add_nodes(MockGrid()) for i in range(num_grids)]

        return gb

    def test_size(self):
        gb = self.simple_bucket(3)
        assert gb.size() == 3

    def test_add_nodes(self):
        # Simply add grid. Should work.
        gb = pp.GridBucket()
        gb.add_nodes(MockGrid())
        gb.add_nodes(MockGrid())

    def test_add_nodes_same_grid_twice(self):
        # Add the same grid twice. Should raise an exception
        gb = self.simple_bucket(1)
        g = MockGrid()
        gb.add_nodes(g)
        self.assertRaises(ValueError, gb.add_nodes, g)

    def test_add_edge(self):
        gb = pp.GridBucket()
        g1 = MockGrid()
        g2 = MockGrid()
        gb.add_nodes(g1)
        gb.add_nodes(g2)
        gb.add_edge([g1, g2], None)

    def test_node_neighbor_no_dim(self):
        # Test node neighbors, not caring about dimensions
        gb = pp.GridBucket()
        g1 = MockGrid()
        g2 = MockGrid()
        g3 = MockGrid()
        gb.add_nodes(g1)
        gb.add_nodes(g2)
        gb.add_nodes(g3)
        gb.add_edge([g1, g2], None)
        gb.add_edge([g2, g3], None)

        neigh_1 = gb.node_neighbors(g1)
        assert neigh_1.size == 1
        assert neigh_1[0] == g2

        neigh_2 = gb.node_neighbors(g2)
        assert neigh_2.size == 2
        assert neigh_2[0] == g1 or neigh_2[0] == g3
        assert neigh_2[1] == g1 or neigh_2[1] == g3

    def test_node_neighbor_with_dim(self):
        # Test node neighbors, using dim keywords
        gb = pp.GridBucket()
        g1 = MockGrid(1)
        g2 = MockGrid(2)
        g3 = MockGrid(3)
        gb.add_nodes(g1)
        gb.add_nodes(g2)
        gb.add_nodes(g3)
        gb.add_edge([g1, g2], None)
        gb.add_edge([g2, g3], None)

        neigh_1 = gb.node_neighbors(g1, only_higher=True)
        assert neigh_1.size == 1
        assert neigh_1[0] == g2
        neigh_1 = gb.node_neighbors(g1, only_lower=True)
        assert neigh_1.size == 0

        neigh_2 = gb.node_neighbors(g2, only_higher=True)
        assert neigh_2.size == 1
        assert neigh_2[0] == g3
        neigh_2 = gb.node_neighbors(g2, only_lower=True)
        assert neigh_2.size == 1
        assert neigh_2[0] == g1

    # ------------ Tests for add_node_props

    def test_add_single_node_prop(self):
        gb = self.simple_bucket(2)
        gb.add_node_props('a')

        for _, d in gb:
            assert 'a' in d.keys()

    def test_add_multiple_node_props(self):
        gb = self.simple_bucket(2)
        props = ['a', 'b']
        gb.add_node_props(props)

        for _, d in gb:
            for p in props:
                assert p in d.keys()

    def test_add_selective_node_props(self):
        gb = pp.GridBucket()
        g1 = MockGrid()
        g2 = MockGrid()
        gb.add_nodes(g1)
        gb.add_nodes(g2)

        p1 = 'a'
        p2 = 'b'
        pboth = 'c'

        # Add by single grid
        gb.add_node_props(p1, g1)
        # Add by list
        gb.add_node_props(p2, [g2])
        # add by list with two items
        gb.add_node_props(pboth, [g1, g2])

        for g, d in gb:
            assert pboth in d.keys()
            if g == g1:
                assert p1 in d.keys()
                assert not p2 in d.keys()
            else:
                assert p2 in d.keys()
                assert not p1 in d.keys()

    #-------------- Tests for add_edge_props

    def test_add_single_edge_prop(self):
        gb = pp.GridBucket()
        g1 = MockGrid()
        g2 = MockGrid()
        gb.add_nodes(g1)
        gb.add_nodes(g2)
        gb.add_edge([g1, g2], None)

        gb.add_edge_props('a')

        for _, d in gb.edges_props():
            assert 'a' in d.keys()

    def test_add_single_edge_prop_reverse_order(self):
        # Add property when reverting the order of the grid_pair
        gb = pp.GridBucket()
        g1 = MockGrid()
        g2 = MockGrid()
        gb.add_nodes(g1)
        gb.add_nodes(g2)

        gb.add_edge([g1, g2], None)
        # Add property, with reverse order of grid pair
        gb.add_edge_props('a', grid_pairs=[[g2, g1]])

        for _, d in gb.edges_props():
            assert 'a' in d.keys()

    def test_add_multiple_edge_props(self):
        gb = pp.GridBucket()
        g1 = MockGrid()
        g2 = MockGrid()
        gb.add_nodes(g1)
        gb.add_nodes(g2)
        gb.add_edge([g1, g2], None)

        props = ['a', 'b']
        gb.add_edge_props(props)

        for _, d in gb.edges_props():
            for p in props:
                assert p in d.keys()

    def test_add_selective_edge_props(self):
        gb = pp.GridBucket()
        g1 = MockGrid()
        g2 = MockGrid()
        g3 = MockGrid()
        gb.add_nodes(g1)
        gb.add_nodes(g2)
        gb.add_nodes(g3)
        gb.add_edge([g1, g2], None)
        gb.add_edge([g2, g3], None)

        p1 = 'a'
        p2 = 'b'
        pboth = 'c'

        # Add by single grid
        gb.add_edge_props(p1, [[g1, g2]])
        # Add by list
        gb.add_edge_props(p2, [[g2, g3]])
        # add by list with two items
        gb.add_edge_props(pboth, [[g1, g2], [g2, g3]])

        # Try to add test to non-existing edge. Should give error
        self.assertRaises(KeyError, gb.add_edge_props, pboth, [[g1, g3]])

        for g, d in gb.edges_props():
            assert pboth in d.keys()
            if g1 in g and g2 in g:
                assert p1 in d.keys()
                assert not p2 in d.keys()
            else:
                assert p2 in d.keys()
                assert not p1 in d.keys()



if __name__ == '__main__':
    unittest.main()