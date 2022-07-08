""" Various tests of MixedDimensionalGrid functionality. Covers getters and setters, topological
information on the bucket, and pickling and unpickling of buckets.
"""
import pickle
import unittest
import os

import numpy as np
import scipy.sparse as sps

import porepy as pp
from tests import test_utils


class MockGrid(pp.CartGrid):
    def __init__(
        self,
        dim=1,
        diameter=0,
        box=None,
        num_cells=0,
        num_faces=0,
        num_nodes=0,
    ):
        # Use at least 2 to enable geometry computation
        nx = [2 for _ in range(dim)]
        super().__init__(np.array(nx))
        self.diameter = diameter
        self.box = box
        self.compute_geometry()
        # self.num_cells = num_cells
        # self.num_faces = num_faces
        # self.num_nodes = num_nodes
        # self.history = ["bar"]
        # self.name = ["foo"]

    def cell_diameters(self):
        return self.diameter

    def bounding_box(self):
        return self.box


class MockMortarGrid:
    def __init__(self, dim, side_grids):
        self.dim = dim
        self.side_grids = side_grids


def mock_mortar_grid(sd_primary: pp.Grid, sd_secondary: pp.Grid) -> pp.MortarGrid:
    """Construct mock mortar grid.

    Ordering may matter, as dimension is determined from sd_secondary.

    Args:
        sd_primary:
            Grid representing primary subdomain.
        sd_secondary:
            Grid representing secondary subdomain.

    Returns:
        MortarGrid for the interface between the two subdomains.

    """
    if sd_primary.dim == sd_secondary.dim:
        shape_0 = sd_primary.num_cells
        ind = np.arange(sd_secondary.num_cells)
    else:
        shape_0 = sd_primary.num_faces
        ind = sd_primary.get_boundary_faces()[: sd_secondary.num_cells]
    array = np.zeros((shape_0, sd_secondary.num_cells))

    array[ind, np.arange(sd_secondary.num_cells)] = 1
    map = sps.csc_matrix(
        array
    )  # (([1], ([0], [0])), shape=(shape_0, sd_secondary.num_cells))
    mg = pp.MortarGrid(sd_secondary.dim, {"left": sd_secondary}, primary_secondary=map)
    return mg
    # return MockMortarGrid(sd_secondary.dim, {"left": sd_primary, "right": sd_secondary})


class TestMixedDimensionalGrid(unittest.TestCase):
    def simple_mdg(self, num_grids):
        mdg = pp.MixedDimensionalGrid()

        [mdg.add_subdomains(MockGrid()) for _ in range(num_grids)]

        return mdg

    # ----- Tests of adding nodes and edges ----- #
    def test_add_subdomains(self):
        # Simply add grid. Should work.
        mdg = pp.MixedDimensionalGrid()
        mdg.add_subdomains(MockGrid())
        mdg.add_subdomains(MockGrid())

    def test_add_subdomains_same_grid_twice(self):
        # Add the same grid twice. Should raise an exception
        mdg = self.simple_mdg(1)
        sd = MockGrid()
        mdg.add_subdomains(sd)
        self.assertRaises(ValueError, mdg.add_subdomains, sd)

    def test_add_interface(self):
        mdg = pp.MixedDimensionalGrid()
        sd_0 = MockGrid(1)
        sd_1 = MockGrid(1)
        sd_3 = MockGrid(3)
        mdg.add_subdomains(sd_0)
        mdg.add_subdomains(sd_1)
        mdg.add_subdomains(sd_3)
        intf_01 = mock_mortar_grid(sd_1, sd_0)
        mdg.add_interface(intf_01, (sd_0, sd_1), None)

        # Should not be able to add existing edge
        self.assertRaises(ValueError, mdg.add_interface, intf_01, (sd_0, sd_1), None)

    def test_dimension_ordering_interfaces(self):
        mdg = pp.MixedDimensionalGrid()
        sd_1 = MockGrid(1)
        sd_2 = MockGrid(2)
        mdg.add_subdomains(sd_1)
        mdg.add_subdomains(sd_2)
        intf_12 = mock_mortar_grid(sd_2, sd_1)
        intf_11 = mock_mortar_grid(sd_2, sd_1)
        mdg.add_interface(intf_12, (sd_1, sd_2), None)
        mdg.add_interface(intf_11, [sd_1, sd_1], None)
        for intf in mdg.interfaces():
            sd_primary, sd_secondary = mdg.interface_to_subdomain_pair(intf)
            self.assertTrue(sd_primary.dim >= sd_secondary.dim)

        # Reverse order of initiation
        mdg = pp.MixedDimensionalGrid()
        mdg.add_subdomains(sd_1)
        mdg.add_subdomains(sd_2)
        intf_21 = mock_mortar_grid(sd_2, sd_1)
        intf_11 = mock_mortar_grid(sd_1, sd_1)
        mdg.add_interface(intf_21, (sd_2, sd_1), None)
        mdg.add_interface(intf_11, [sd_1, sd_1], None)
        for intf in mdg.interfaces():
            sd_primary, sd_secondary = mdg.interface_to_subdomain_pair(intf)
            self.assertTrue(sd_primary.dim >= sd_secondary.dim)

    def test_subdomain_neighbor_no_dim(self):
        # Test node neighbors, not caring about dimensions
        mdg = pp.MixedDimensionalGrid()
        sd_1 = MockGrid()
        sd_2 = MockGrid()
        sd_3 = MockGrid()
        mdg.add_subdomains(sd_1)
        mdg.add_subdomains(sd_2)
        mdg.add_subdomains(sd_3)
        intf_12 = mock_mortar_grid(sd_2, sd_1)
        intf_32 = mock_mortar_grid(sd_3, sd_2)
        mdg.add_interface(intf_12, [sd_1, sd_2], None)
        mdg.add_interface(intf_32, [sd_2, sd_3], None)

        neigh_1 = mdg.neighboring_subdomains(sd_1)
        self.assertTrue(len(neigh_1) == 1)
        self.assertTrue(neigh_1[0] == sd_2)

        neigh_2 = mdg.neighboring_subdomains(sd_2)
        self.assertTrue(len(neigh_2) == 2)
        self.assertTrue(neigh_2[0] == sd_1 or neigh_2[0] == sd_3)
        self.assertTrue(neigh_2[1] == sd_1 or neigh_2[1] == sd_3)

    def test_subdomain_neighbor_with_dim(self):
        # Test node neighbors, using dim keywords
        mdg = pp.MixedDimensionalGrid()
        sd_1 = MockGrid(1)
        sd_2 = MockGrid(2)
        sd_3 = MockGrid(3)
        mdg.add_subdomains(sd_1)
        mdg.add_subdomains(sd_2)
        mdg.add_subdomains(sd_3)
        intf_12 = mock_mortar_grid(sd_2, sd_1)
        intf_32 = mock_mortar_grid(sd_3, sd_2)
        mdg.add_interface(intf_12, [sd_1, sd_2], None)
        mdg.add_interface(intf_32, [sd_2, sd_3], None)

        neigh_1 = mdg.neighboring_subdomains(sd_1, only_higher=True)
        self.assertTrue(len(neigh_1) == 1)
        self.assertTrue(neigh_1[0] == sd_2)
        neigh_1 = mdg.neighboring_subdomains(sd_1, only_lower=True)
        self.assertTrue(len(neigh_1) == 0)

        neigh_2 = mdg.neighboring_subdomains(sd_2, only_higher=True)
        self.assertTrue(len(neigh_2) == 1)
        self.assertTrue(neigh_2[0] == sd_3)
        neigh_2 = mdg.neighboring_subdomains(sd_2, only_lower=True)
        self.assertTrue(len(neigh_2) == 1)
        self.assertTrue(neigh_2[0] == sd_1)

        self.assertRaises(ValueError, mdg.neighboring_subdomains, sd_1, True, True)

    # ------ Test of iterators ------*
    def test_subdomain_and_interface_iterators(self):
        mdg = pp.MixedDimensionalGrid()
        sd_1 = MockGrid(1)
        sd_2 = MockGrid(2)
        sd_3 = MockGrid(3)
        mdg.add_subdomains(sd_1)
        mdg.add_subdomains(sd_2)
        mdg.add_subdomains(sd_3)
        intf_12 = mock_mortar_grid(sd_2, sd_1)
        intf_32 = mock_mortar_grid(sd_3, sd_2)
        mdg.add_interface(intf_12, [sd_1, sd_2], None)
        mdg.add_interface(intf_32, [sd_2, sd_3], None)

        # First test traversal by gb.__iter__
        found = {sd_1: False, sd_2: False, sd_3: False}
        for sd in mdg.subdomains():
            found[sd] = True
        self.assertTrue(all([v for v in list(found.values())]))

        # Next, use the subdomains() function
        found = {sd_1: False, sd_2: False, sd_3: False}
        for sd in mdg.subdomains():
            found[sd] = True
        self.assertTrue(all([v for v in list(found.values())]))

        # Finally, check the interfaces
        found = {intf_12: False, intf_32: False}
        for intf in mdg.interfaces():
            found[intf] = True
        self.assertTrue(all([v for v in list(found.values())]))

    def test_contains_subdomain(self):
        mdg = self.simple_mdg(1)

        for sd in mdg.subdomains():
            self.assertTrue(sd in mdg.subdomains())

        # Define a grid that is not in the mdg
        sd = MockGrid()
        self.assertTrue(not sd in mdg.subdomains())

    def test_contains_interface(self):
        mdg = pp.MixedDimensionalGrid()
        sd_1 = MockGrid(1)
        sd_2 = MockGrid(2)
        sd_3 = MockGrid(3)
        mdg.add_subdomains(sd_1)
        mdg.add_subdomains(sd_2)
        mdg.add_subdomains(sd_3)
        intf_12 = mock_mortar_grid(sd_2, sd_1)
        mdg.add_interface(intf_12, [sd_1, sd_2], None)

        # This edge is defined
        self.assertTrue(intf_12 in mdg.interfaces())
        # this is not
        intf_31 = mock_mortar_grid(sd_3, sd_1)
        self.assertFalse(intf_31 in mdg.interfaces())

    def test_diameter(self):
        sd_1 = MockGrid(1, 2)
        sd_2 = MockGrid(2, 3)
        mdg = pp.MixedDimensionalGrid()
        mdg.add_subdomains(sd_1)
        mdg.add_subdomains(sd_2)

        self.assertTrue(mdg.diameter() == 3)
        self.assertTrue(mdg.diameter(lambda g: g.dim == 1) == 2)

    def test_bounding_box(self):
        mdg = pp.MixedDimensionalGrid()
        sd_1 = pp.CartGrid([1, 1, 1])
        sd_1.nodes = np.random.random((sd_1.dim, sd_1.num_nodes))
        sd_2 = pp.CartGrid([1, 1, 1])
        # Shift g2 with 1
        sd_2.nodes = 1 + np.random.random((sd_2.dim, sd_2.num_nodes))

        mdg.add_subdomains([sd_1, sd_2])

        bmin, bmax = pp.bounding_box.from_md_grid(mdg)

        # Since g2 is shifted, minimum should be at g1, maximum in g2
        self.assertTrue(np.allclose(bmin, sd_1.nodes.min(axis=1)))
        self.assertTrue(np.allclose(bmax, sd_2.nodes.max(axis=1)))

        d = pp.bounding_box.from_md_grid(mdg, as_dict=True)
        self.assertTrue(d["xmin"] == np.min(sd_1.nodes[0]))
        self.assertTrue(d["ymin"] == np.min(sd_1.nodes[1]))
        self.assertTrue(d["zmin"] == np.min(sd_1.nodes[2]))
        self.assertTrue(d["xmax"] == np.max(sd_2.nodes[0]))
        self.assertTrue(d["ymax"] == np.max(sd_2.nodes[1]))
        self.assertTrue(d["zmax"] == np.max(sd_2.nodes[2]))

    def test_num_cells(self):

        sd_1 = MockGrid(dim=1, num_cells=1, num_faces=3, num_nodes=3)
        sd_2 = MockGrid(dim=2, num_cells=3, num_faces=7, num_nodes=3)
        mdg = pp.MixedDimensionalGrid()
        mdg.add_subdomains([sd_1, sd_2])

        self.assertTrue(mdg.num_subdomain_cells() == (sd_1.num_cells + sd_2.num_cells))

        l = lambda g: g.dim == 1
        self.assertTrue(mdg.num_subdomain_cells(l) == sd_1.num_cells)

    def test_num_subdomains_and_interfaces(self):
        mdg = pp.MixedDimensionalGrid()
        sd_1 = MockGrid(1)
        sd_2 = MockGrid(2)
        sd_3 = MockGrid(3)
        mdg.add_subdomains(sd_1)
        mdg.add_subdomains(sd_2)
        mdg.add_subdomains(sd_3)
        intf_12 = mock_mortar_grid(sd_2, sd_1)
        intf_32 = mock_mortar_grid(sd_3, sd_2)
        mdg.add_interface(intf_12, (sd_1, sd_2), None)
        mdg.add_interface(intf_32, (sd_2, sd_3), None)
        self.assertTrue(mdg.num_interfaces() == 2)
        self.assertTrue(mdg.num_subdomains() == 3)

    def test_str_repr(self):

        sd_1 = MockGrid(dim=1, num_cells=1, num_faces=3, num_nodes=3)
        sd_2 = MockGrid(dim=2, num_cells=3, num_faces=7, num_nodes=3)
        mdg = pp.MixedDimensionalGrid()
        mdg.add_subdomains([sd_1, sd_2])
        mdg.__str__()
        mdg.__repr__()

    # ------------ Tests for removers


def test_pickle_md_grid():
    fracs = [np.array([[0, 2], [1, 1]]), np.array([[1, 1], [0, 2]])]
    mdg = pp.meshing.cart_grid(fracs, [2, 2])

    fn = "tmp.md_grid"
    pickle.dump(mdg, open(fn, "wb"))
    mdg_read = pickle.load(open(fn, "rb"))

    test_utils.compare_md_grids(mdg, mdg_read)

    # Delete the temporary file
    os.remove(fn)


if __name__ == "__main__":
    unittest.main()
