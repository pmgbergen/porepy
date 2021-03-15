""" Various tests of GridBucket functionality. Covers getters and setters, topologial
information on the bucket, and pickling and unpickling of buckets.
"""
import unittest
import pickle

import numpy as np
from test import test_utils
import porepy as pp


class MockGrid(pp.Grid):
    def __init__(
        self,
        dim=1,
        diameter=0,
        box=None,
        num_cells=None,
        num_faces=None,
        num_nodes=None,
    ):
        self.dim = dim
        self.diameter = diameter
        self.box = box
        self.num_cells = num_cells
        self.num_faces = num_faces
        self.num_nodes = num_nodes

    def cell_diameters(self):
        return self.diameter

    def bounding_box(self):
        return self.box


class TestBucket(unittest.TestCase):
    def simple_bucket(self, num_grids):
        gb = pp.GridBucket()

        [gb.add_nodes(MockGrid()) for i in range(num_grids)]

        return gb

    def test_size(self):
        gb = self.simple_bucket(3)
        self.assertTrue(gb.size() == 3)

    # ----- Tests of adding nodes and edges ----- #
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

    def test_dimension_ordering_edges(self):
        gb = pp.GridBucket()
        g1 = MockGrid(1)
        g2 = MockGrid(2)
        gb.add_nodes(g1)
        gb.add_nodes(g2)
        gb.add_edge([g1, g2], None)
        gb.add_edge([g1, g1], None)
        for e, _ in gb.edges():
            self.assertTrue(e[0].dim >= e[1].dim)

        gb = pp.GridBucket()
        gb.add_nodes(g1)
        gb.add_nodes(g2)
        gb.add_edge([g2, g1], None)
        gb.add_edge([g1, g1], None)
        for e, _ in gb.edges():
            self.assertTrue(e[0].dim >= e[1].dim)

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
        self.assertTrue(neigh_1.size == 1)
        self.assertTrue(neigh_1[0] == g2)

        neigh_2 = gb.node_neighbors(g2)
        self.assertTrue(neigh_2.size == 2)
        self.assertTrue(neigh_2[0] == g1 or neigh_2[0] == g3)
        self.assertTrue(neigh_2[1] == g1 or neigh_2[1] == g3)

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
        self.assertTrue(neigh_1.size == 1)
        self.assertTrue(neigh_1[0] == g2)
        neigh_1 = gb.node_neighbors(g1, only_lower=True)
        self.assertTrue(neigh_1.size == 0)

        neigh_2 = gb.node_neighbors(g2, only_higher=True)
        self.assertTrue(neigh_2.size == 1)
        self.assertTrue(neigh_2[0] == g3)
        neigh_2 = gb.node_neighbors(g2, only_lower=True)
        self.assertTrue(neigh_2.size == 1)
        self.assertTrue(neigh_2[0] == g1)

        self.assertRaises(ValueError, gb.node_neighbors, g1, True, True)

    # ------ Test of iterators ------*
    def test_node_edge_iterators(self):
        gb = pp.GridBucket()
        g1 = MockGrid(1)
        g2 = MockGrid(2)
        g3 = MockGrid(3)
        gb.add_nodes(g1)
        gb.add_nodes(g2)
        gb.add_nodes(g3)
        gb.add_edge([g1, g2], None)
        gb.add_edge([g2, g3], None)

        # First test traversal by gb.__iter__
        found = {g1: False, g2: False, g3: False}
        for g, _ in gb:
            found[g] = True
        self.assertTrue(all([v for v in list(found.values())]))

        # Next, use the node() function
        found = {g1: False, g2: False, g3: False}
        for g, _ in gb.nodes():
            found[g] = True
        self.assertTrue(all([v for v in list(found.values())]))

        # Finally, check the edges
        found = {(g2, g1): False, (g3, g2): False}
        for e, _ in gb.edges():
            found[e] = True
        self.assertTrue(all([v for v in list(found.values())]))

    def test_contains_node(self):
        gb = self.simple_bucket(1)

        for g, _ in gb.nodes():
            self.assertTrue(g in gb)

        # Define a grid that is not in the gb
        g = MockGrid()
        self.assertTrue(not g in gb)

    def test_contains_edge(self):
        gb = pp.GridBucket()
        g1 = MockGrid(1)
        g2 = MockGrid(2)
        g3 = MockGrid(3)
        gb.add_nodes(g1)
        gb.add_nodes(g2)
        gb.add_nodes(g3)
        gb.add_edge([g1, g2], None)

        # This edge is defined
        self.assertTrue((g1, g2) in gb)
        # this is not
        self.assertFalse((g1, g3) in gb)

        # This is a list, and thus not an edge in the networkx sense
        self.assertFalse([g1, g2] in gb)

    # ------------ Tests for add_node_props

    def test_add_single_node_prop(self):
        gb = self.simple_bucket(2)
        gb.add_node_props("a")

        for _, d in gb:
            self.assertTrue("a" in d.keys())

    def test_add_multiple_node_props(self):
        gb = self.simple_bucket(2)
        props = ["a", "b"]
        gb.add_node_props(props)

        for _, d in gb:
            for p in props:
                self.assertTrue(p in d.keys())

    def test_add_selective_node_props(self):
        gb = pp.GridBucket()
        g1 = MockGrid()
        g2 = MockGrid()
        gb.add_nodes(g1)
        gb.add_nodes(g2)

        p1 = "a"
        p2 = "b"
        pboth = "c"

        # Add by single grid
        gb.add_node_props(p1, g1)
        # Add by list
        gb.add_node_props(p2, [g2])
        # add by list with two items
        gb.add_node_props(pboth, [g1, g2])

        for g, d in gb:
            self.assertTrue(pboth in d.keys())
            if g == g1:
                self.assertTrue(p1 in d.keys())
                self.assertTrue(not p2 in d.keys())
            else:
                self.assertTrue(p2 in d.keys())
                self.assertTrue(not p1 in d.keys())

    def test_add_node_prop_node_number(self):
        gb = self.simple_bucket(1)
        self.assertRaises(ValueError, gb.add_node_props, "node_number")

    def test_overwrite_node_props(self):
        gb = pp.GridBucket()
        g1 = MockGrid()
        gb.add_nodes(g1)
        key = "foo"
        val = 42
        gb.set_node_prop(g1, key, val)

        gb.add_node_props(key, overwrite=False)
        self.assertTrue(gb.node_props(g1, key) == val)

        gb.add_node_props(key)
        self.assertTrue(gb.node_props(g1, key) is None)

    # -------------- Tests for add_edge_props

    def test_add_single_edge_prop(self):
        gb = pp.GridBucket()
        g1 = MockGrid()
        g2 = MockGrid()
        gb.add_nodes(g1)
        gb.add_nodes(g2)
        gb.add_edge([g1, g2], None)

        gb.add_edge_props("a")

        for _, d in gb.edges():
            self.assertTrue("a" in d.keys())

    def test_add_single_edge_prop_reverse_order(self):
        # Add property when reverting the order of the grid_pair
        gb = pp.GridBucket()
        g1 = MockGrid()
        g2 = MockGrid()
        gb.add_nodes(g1)
        gb.add_nodes(g2)

        gb.add_edge([g1, g2], None)
        # Add property, with reverse order of grid pair
        gb.add_edge_props("a", edges=[[g2, g1]])

        for _, d in gb.edges():
            self.assertTrue("a" in d.keys())

    def test_add_multiple_edge_props(self):
        gb = pp.GridBucket()
        g1 = MockGrid()
        g2 = MockGrid()
        gb.add_nodes(g1)
        gb.add_nodes(g2)
        gb.add_edge([g1, g2], None)

        props = ["a", "b"]
        gb.add_edge_props(props)

        for _, d in gb.edges():
            for p in props:
                self.assertTrue(p in d.keys())

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

        p1 = "a"
        p2 = "b"
        pboth = "c"

        # Add by single grid
        gb.add_edge_props(p1, [(g1, g2)])
        # Add by list
        gb.add_edge_props(p2, [(g2, g3)])
        # add by list with two items
        gb.add_edge_props(pboth, [(g1, g2), (g2, g3)])

        # Try to add test to non-existing edge. Should give error
        self.assertRaises(KeyError, gb.add_edge_props, pboth, [[g1, g3]])

        for g, d in gb.edges():
            self.assertTrue(pboth in d.keys())
            if g1 in g and g2 in g:
                self.assertTrue(p1 in d.keys())
                self.assertTrue(not p2 in d.keys())
            else:
                self.assertTrue(p2 in d.keys())
                self.assertTrue(not p1 in d.keys())

    def test_overwrite_edge_props(self):
        gb = pp.GridBucket()
        g1 = MockGrid()
        g2 = MockGrid()
        gb.add_nodes(g1)
        gb.add_nodes(g2)
        e = (g1, g2)
        gb.add_edge(e, None)

        key = "foo"
        val = 42
        gb.set_edge_prop(e, key, val)

        gb.add_edge_props(key, overwrite=False)
        self.assertTrue(gb.edge_props(e, key) == val)

        gb.add_edge_props(key)
        self.assertTrue(gb.edge_props(e, key) is None)

    # ----------- Tests for getters of node properties ----------

    def test_set_get_node_props_single_grid(self):

        gb = pp.GridBucket()
        g1 = MockGrid()
        gb.add_nodes(g1)
        d = {"a": 1, "b": 2, "c": 3}

        keys = d.keys()
        vals = d.values()

        for k, v in zip(keys, vals):
            gb.set_node_prop(g1, k, v)

        # Obtain all keys, check that we have them all
        all_keys = gb.node_props(g1)
        self.assertTrue(all([k in keys for k in all_keys.keys()]))
        self.assertTrue(all([k in all_keys.keys() for k in keys]))

        # Next obtain values by keyword
        for k, v in zip(keys, vals):
            v2 = gb.node_props(g1, k)
            self.assertTrue(v == v2)

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

        d = {"a": 1, "b": 2, "c": 3}
        keys = d.keys()
        vals = d.values()

        pairs = [(g1, g2), (g2, g3)]

        for k, v in zip(keys, vals):
            gb.set_edge_prop(pairs[0], k, v)

        # Obtain all keys, check that we have them all
        all_keys = gb.edge_props(pairs[0])
        self.assertTrue(all([k in all_keys.keys() for k in keys]))

        all_keys = gb.edge_props(pairs[0][::-1])
        self.assertTrue(all([k in all_keys.keys() for k in keys]))

        # The other edge has no properties, Python should raise KeyError
        self.assertRaises(KeyError, gb.edge_props, edge=pairs[1], key="a")
        # Try a non-existing edge, the method itself should raise KeyError
        self.assertRaises(KeyError, gb.edge_props, edge=[g1, g3], key="a")

    def test_update_nodes(self):
        gb = pp.GridBucket()
        g1 = MockGrid()
        g2 = MockGrid()
        gb.add_nodes(g1)
        gb.add_nodes(g2)
        gb.add_edge([g1, g2], None)

        d = {"a": 1, "b": 2}
        keys = d.keys()
        vals = d.values()

        for k, v in zip(keys, vals):
            gb.set_edge_prop([g1, g2], k, v)
            gb.set_node_prop(g1, k, v)

        g3 = MockGrid()

        gb.update_nodes({g1: g3})

        # Check that the new grid and edge inherited data
        for k, v in zip(keys, vals):
            v2 = gb.node_props(g3, k)
            self.assertTrue(v == v2)
            v2 = gb.edge_props([g2, g3], k)
            self.assertTrue(v == v2)

        # g1 is no longer associated with gb
        self.assertRaises(KeyError, gb.node_props, g1, "a")

    def test_diameter(self):
        g1 = MockGrid(1, 2)
        g2 = MockGrid(2, 3)
        gb = pp.GridBucket()
        gb.add_nodes(g1)
        gb.add_nodes(g2)

        self.assertTrue(gb.diameter() == 3)
        self.assertTrue(gb.diameter(lambda g: g.dim == 1) == 2)

    def test_bounding_box(self):
        gb = pp.GridBucket()
        g1 = pp.CartGrid([1, 1, 1])
        g1.nodes = np.random.random((g1.dim, g1.num_nodes))
        g2 = pp.CartGrid([1, 1, 1])
        # Shift g2 with 1
        g2.nodes = 1 + np.random.random((g2.dim, g2.num_nodes))

        gb.add_nodes([g1, g2])

        bmin, bmax = gb.bounding_box()

        # Since g2 is shifted, minimum should be at g1, maximum in g2
        self.assertTrue(np.allclose(bmin, g1.nodes.min(axis=1)))
        self.assertTrue(np.allclose(bmax, g2.nodes.max(axis=1)))

        d = gb.bounding_box(as_dict=True)
        self.assertTrue(d["xmin"] == np.min(g1.nodes[0]))
        self.assertTrue(d["ymin"] == np.min(g1.nodes[1]))
        self.assertTrue(d["zmin"] == np.min(g1.nodes[2]))
        self.assertTrue(d["xmax"] == np.max(g2.nodes[0]))
        self.assertTrue(d["ymax"] == np.max(g2.nodes[1]))
        self.assertTrue(d["zmax"] == np.max(g2.nodes[2]))

    def test_num_cells_faces_nodes(self):

        g1 = MockGrid(dim=1, num_cells=1, num_faces=3, num_nodes=3)
        g2 = MockGrid(dim=2, num_cells=3, num_faces=7, num_nodes=3)
        gb = pp.GridBucket()
        gb.add_nodes([g1, g2])

        self.assertTrue(gb.num_cells() == (g1.num_cells + g2.num_cells))
        self.assertTrue(gb.num_faces() == (g1.num_faces + g2.num_faces))
        self.assertTrue(gb.num_nodes() == (g1.num_nodes + g2.num_nodes))

        l = lambda g: g.dim == 1
        self.assertTrue(gb.num_cells(l) == g1.num_cells)
        self.assertTrue(gb.num_faces(l) == g1.num_faces)
        self.assertTrue(gb.num_nodes(l) == g1.num_nodes)

    def test_num_graph_nodes_edges(self):
        gb = pp.GridBucket()
        g1 = MockGrid(1)
        g2 = MockGrid(2)
        g3 = MockGrid(3)
        gb.add_nodes(g1)
        gb.add_nodes(g2)
        gb.add_nodes(g3)
        gb.add_edge([g1, g2], None)
        gb.add_edge([g2, g3], None)
        self.assertTrue(gb.num_graph_edges() == 2)
        self.assertTrue(gb.num_graph_nodes() == 3)

    def test_str_repr(self):

        g1 = MockGrid(dim=1, num_cells=1, num_faces=3, num_nodes=3)
        g2 = MockGrid(dim=2, num_cells=3, num_faces=7, num_nodes=3)
        gb = pp.GridBucket()
        gb.add_nodes([g1, g2])
        gb.__str__()
        gb.__repr__()

    # ------------ Tests for removers

    def test_remove_single_node_prop(self):
        gb = self.simple_bucket(2)
        props = ["a", "b"]
        gb.add_node_props(props)
        gb.remove_node_props("a")

        for _, d in gb:
            self.assertTrue(not "a" in d.keys())
            self.assertTrue("b" in d.keys())

    def test_remove_multiple_node_props(self):
        gb = self.simple_bucket(2)
        props = ["a", "b"]
        gb.add_node_props(props)
        gb.remove_node_props(props)

        for _, d in gb:
            for p in props:
                self.assertTrue(not p in d.keys())

    def test_remove_selective_node_props(self):
        gb = pp.GridBucket()
        g1 = MockGrid()
        g2 = MockGrid()
        gb.add_nodes(g1)
        gb.add_nodes(g2)

        props = ["a", "b", "c"]
        gb.add_node_props(props)

        gb.remove_node_props("a", g1)
        gb.remove_node_props("b", g2)
        gb.remove_node_props("c", [g1, g2])

        for g, d in gb:
            self.assertTrue(not "c" in d.keys())
            if g == g1:
                self.assertTrue(not "a" in d.keys())
                self.assertTrue("b" in d.keys())
            else:
                self.assertTrue(not "b" in d.keys())
                self.assertTrue("a" in d.keys())

    def test_remove_node_prop_node_number(self):
        gb = self.simple_bucket(1)
        self.assertRaises(ValueError, gb.remove_node_props, "node_number")

    def test_cell_volumes(self):
        gb = pp.GridBucket()
        g1 = pp.CartGrid([1, 1, 1])
        g1.nodes += 0.1 * np.random.random((g1.dim, g1.num_nodes))
        g1.compute_geometry()
        g2 = pp.CartGrid([1, 1, 1])
        g2.nodes += 0.1 * np.random.random((g2.dim, g2.num_nodes))
        g2.compute_geometry()

        gb.add_nodes([g1, g2])

        cond = lambda g: g == g1
        cell_volumes = np.hstack((g1.cell_volumes, g2.cell_volumes))

        self.assertTrue(np.all(cell_volumes == gb.cell_volumes()))
        self.assertTrue(np.all(g1.cell_volumes == gb.cell_volumes(cond)))

    def test_cell_centers(self):
        gb = pp.GridBucket()
        g1 = pp.CartGrid([1, 1, 1])
        g1.nodes += 0.1 * np.random.random((g1.dim, g1.num_nodes))
        g1.compute_geometry()
        g2 = pp.CartGrid([1, 1, 1])
        g2.nodes += 0.1 * np.random.random((g2.dim, g2.num_nodes))
        g2.compute_geometry()

        gb.add_nodes([g1, g2])
        cond = lambda g: g == g1
        cell_centers = np.hstack((g1.cell_centers, g2.cell_centers))

        self.assertTrue(np.all(cell_centers == gb.cell_centers()))
        self.assertTrue(np.all(g1.cell_centers == gb.cell_centers(cond)))

    def test_face_centers(self):
        gb = pp.GridBucket()
        g1 = pp.CartGrid([1, 1, 1])
        g1.nodes += 0.1 * np.random.random((g1.dim, g1.num_nodes))
        g1.compute_geometry()
        g2 = pp.CartGrid([1, 1, 1])
        g2.nodes += 0.1 * np.random.random((g2.dim, g2.num_nodes))
        g2.compute_geometry()

        gb.add_nodes([g1, g2])

        cond = lambda g: g == g1
        face_centers = np.hstack((g1.face_centers, g2.face_centers))

        self.assertTrue(np.all(face_centers == gb.face_centers()))
        self.assertTrue(np.all(g1.face_centers == gb.face_centers(cond)))


def test_pickle_bucket():
    fracs = [np.array([[0, 2], [1, 1]]), np.array([[1, 1], [0, 2]])]
    gb = pp.meshing.cart_grid(fracs, [2, 2])

    fn = "tmp.grid_bucket"
    pickle.dump(gb, open(fn, "wb"))
    gb_read = pickle.load(open(fn, "rb"))

    test_utils.compare_grid_buckets(gb, gb_read)

if __name__ == "__main__":
    unittest.main()
