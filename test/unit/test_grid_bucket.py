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
        g1 = MockGrid(1)
        g2 = MockGrid(2)
        g3 = MockGrid(3)
        gb.add_nodes(g1)
        gb.add_nodes(g2)
        gb.add_nodes(g3)
        gb.add_edge([g1, g2], None)

        # Should not be able to add existing edge
        self.assertRaises(ValueError, gb.add_edge, [g1, g2], None)
        # Should not be able to add couplings two dimensions appart
        self.assertRaises(ValueError, gb.add_edge, [g1, g3], None)


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

        self.assertRaises(ValueError, gb.node_neighbors, g1, True, True)

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

    def test_add_node_prop_node_number(self):
        gb = self.simple_bucket(1)
        self.assertRaises(ValueError, gb.add_node_props, 'node_number')

    #-------------- Tests for add_edge_props

    def test_add_single_edge_prop(self):
        gb = pp.GridBucket()
        g1 = MockGrid()
        g2 = MockGrid()
        gb.add_nodes(g1)
        gb.add_nodes(g2)
        gb.add_edge([g1, g2], None)

        gb.add_edge_props('a')

        for _, d in gb.edges():
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

        for _, d in gb.edges():
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

        for _, d in gb.edges():
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

        for g, d in gb.edges():
            assert pboth in d.keys()
            if g1 in g and g2 in g:
                assert p1 in d.keys()
                assert not p2 in d.keys()
            else:
                assert p2 in d.keys()
                assert not p1 in d.keys()

    # ----------- Tests for getters of node properties ----------

    def test_set_get_node_props_single_grid(self):

        gb = pp.GridBucket()
        g1 = MockGrid()
        gb.add_nodes(g1)
        d = {'a':1, 'b':2, 'c':3}

        keys = d.keys()
        vals = d.values()

        for k, v in zip(keys, vals):
            gb.set_node_prop(g1, k, v)

        # Obtain all keys, check that we have them all
        all_keys = gb.node_props(g1)
        assert all([k in keys for k in all_keys.keys()])
        assert all([k in all_keys.keys() for k in keys])

        # Next obtain values by keyword
        for k, v in zip(keys, vals):
            v2 = gb.node_props(g1, k)
            assert v == v2

    def test_set_get_edge_props(self):

        gb = pp.GridBucket()
        g1 = MockGrid()
        g2 = MockGrid()
        g3 = MockGrid()
        gb.add_nodes(g1)
        gb.add_nodes(g2)
        gb.add_nodes(g3)
        gb.add_edge([g1, g2], None)
        gb.add_edge([g2, g3], None)

        d = {'a':1, 'b':2, 'c':3}
        keys = d.keys()
        vals = d.values()

        pairs = [[g1, g2], [g2, g3]]

        for k, v in zip(keys, vals):
            gb.set_edge_prop(pairs[0], k, v)

        # Obtain all keys, check that we have them all
        all_keys = gb.edge_props(pairs[0])
        assert all([k in all_keys.keys() for k in keys])

        all_keys = gb.edge_props(pairs[0][::-1])
        assert all([k in all_keys.keys() for k in keys])

        # The other edge has no properties, Python should raise KeyError
        self.assertRaises(KeyError, gb.edge_props, gp=pairs[1], key='a')
        # Try a non-existing edge, the method itself should raise KeyError
        self.assertRaises(KeyError, gb.edge_props, gp=[g1, g3], key='a')

    def test_update_nodes(self):
        gb = pp.GridBucket()
        g1 = MockGrid()
        g2 = MockGrid()
        gb.add_nodes(g1)
        gb.add_nodes(g2)
        gb.add_edge([g1, g2], None)

        d = {'a':1, 'b':2}
        keys = d.keys()
        vals = d.values()

        for k, v in zip(keys, vals):
            gb.set_edge_prop([g1, g2], k, v)
            gb.set_node_prop(g1, k, v)

        g3 = MockGrid()

        gb.update_nodes(g1, g3)

        # Check that the new grid and edge inherited data
        for k, v in zip(keys, vals):
            v2 = gb.node_props(g3, k)
            assert v == v2
            v2 = gb.edge_props([g2, g3], k)
            assert v == v2

        # g1 is no longer associated with gb
        self.assertRaises(KeyError, gb.node_props, g1, 'a')



if __name__ == '__main__':
    unittest.main()