import numpy as np
import scipy.sparse as sps
import unittest
import warnings

from porepy.grids.grid_bucket import GridBucket
from porepy.fracs import meshing
import porepy as pp


class TestGridBucket(unittest.TestCase):
    def test_cell_global2loc_1_grid(self):
        gb = meshing.cart_grid([], [2, 2])
        gb.cell_global2loc()
        R = sps.eye(4)
        for g, d in gb:
            self.assertTrue(np.sum(d["cell_global2loc"] != R) == 0)

    def test_cell_global2loc_1_frac(self):
        f = np.array([[0, 1], [1, 1]])
        gb = meshing.cart_grid([f], [2, 2])
        gb.cell_global2loc()
        glob = np.arange(5)
        # test grids
        for g, d in gb:
            if g.dim == 2:
                loc = np.array([0, 1, 2, 3])
            elif g.dim == 1:
                loc = np.array([4])
            else:
                self.assertTrue(False)
            R = d["cell_global2loc"]
            self.assertTrue(np.all(R * glob == loc))
        # test mortars
        glob = np.array([0, 1])
        for _, d in gb.edges():
            loc = np.array([0, 1])
            R = d["cell_global2loc"]
            self.assertTrue(np.all(R * glob == loc))

    def test_cell_global2loc_2_fracs(self):
        f1 = np.array([[0, 1], [1, 1]])
        f2 = np.array([[1, 2], [1, 1]])
        f3 = np.array([[1, 1], [0, 1]])
        f4 = np.array([[1, 1], [1, 2]])

        gb = meshing.cart_grid([f1, f2, f3, f4], [2, 2])
        gb.cell_global2loc()
        glob = np.arange(9)
        # test grids
        for g, d in gb:
            if g.dim == 2:
                loc = np.array([0, 1, 2, 3])
            elif g.dim == 1:
                i = d["node_number"]
                loc = np.arange(4 + (i - 1), 4 + i)
            else:
                loc = np.array([8])
            R = d["cell_global2loc"]
            self.assertTrue(np.all(R * glob == loc))

        # test mortars
        glob = np.arange(12)
        start = 0
        end = 0
        for e, d in gb.edges():
            i = d["edge_number"]
            end += d["mortar_grid"].num_cells
            loc = np.arange(start, end)
            start = end
            R = d["cell_global2loc"]
            self.assertTrue(np.all(R * glob == loc))


class MockGrid:
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
        gb.add_edge_props("a", grid_pairs=[[g2, g1]])

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
        gb.add_edge_props(p1, [[g1, g2]])
        # Add by list
        gb.add_edge_props(p2, [[g2, g3]])
        # add by list with two items
        gb.add_edge_props(pboth, [[g1, g2], [g2, g3]])

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

        pairs = [[g1, g2], [g2, g3]]

        for k, v in zip(keys, vals):
            gb.set_edge_prop(pairs[0], k, v)

        # Obtain all keys, check that we have them all
        all_keys = gb.edge_props(pairs[0])
        self.assertTrue(all([k in all_keys.keys() for k in keys]))

        all_keys = gb.edge_props(pairs[0][::-1])
        self.assertTrue(all([k in all_keys.keys() for k in keys]))

        # The other edge has no properties, Python should raise KeyError
        self.assertRaises(KeyError, gb.edge_props, gp=pairs[1], key="a")
        # Try a non-existing edge, the method itself should raise KeyError
        self.assertRaises(KeyError, gb.edge_props, gp=[g1, g3], key="a")

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

        gb.update_nodes(g1, g3)

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


if __name__ == "__main__":
    unittest.main()
