""" Various tests of MixedDimensionalGrid functionality. Covers getters and setters,
topological information on the bucket, and pickling and unpickling of buckets.
"""
import os
import pickle

import numpy as np
import pytest
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

    def cell_diameters(self):
        return self.diameter

    def bounding_box(self):
        return self.box


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
    map = sps.csc_matrix(array)
    mg = pp.MortarGrid(sd_secondary.dim, {"left": sd_secondary}, primary_secondary=map)
    return mg


def simple_mdg(num_grids):
    mdg = pp.MixedDimensionalGrid()

    [mdg.add_subdomains(MockGrid()) for _ in range(num_grids)]

    return mdg


# ----- Tests of adding nodes and edges ----- #


def test_add_remove_subdomains():
    # Simply add grid. Should work.
    mdg = pp.MixedDimensionalGrid()
    mdg.add_subdomains(MockGrid())
    mdg.add_subdomains(MockGrid())

    # Check that they are stored.
    subdomains = mdg.subdomains()
    assert len(subdomains) == 2
    assert len(mdg.boundaries()) == 2, "Boundary grids must be created."

    # Check removal.
    for sd in subdomains:
        mdg.remove_subdomain(sd)

    assert len(mdg.subdomains()) == 0
    assert len(mdg.boundaries()) == 0


def test_add_subdomains_same_grid_twice():
    # Add the same grid twice. Should raise an exception
    mdg = simple_mdg(1)
    sd = mdg.subdomains()
    with pytest.raises(ValueError):
        mdg.add_subdomains(sd)


def test_add_same_interface_twice():
    # Add the same interface twice. Should raise an exception
    mdg = pp.MixedDimensionalGrid()
    sd_0 = MockGrid(1)
    sd_1 = MockGrid(1)
    mdg.add_subdomains(sd_0)
    mdg.add_subdomains(sd_1)
    intf_01 = mock_mortar_grid(sd_1, sd_0)
    mdg.add_interface(intf_01, (sd_1, sd_0), None)
    with pytest.raises(ValueError):
        mdg.add_interface(intf_01, (sd_1, sd_0), None)


def test_dimension_ordering_interfaces():
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
        assert sd_primary.dim >= sd_secondary.dim

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
        assert sd_primary.dim >= sd_secondary.dim


def test_subdomain_neighbor_no_dim():
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
    mdg.add_interface(intf_12, (sd_1, sd_2), None)
    mdg.add_interface(intf_32, (sd_2, sd_3), None)

    neigh_1 = mdg.neighboring_subdomains(sd_1)
    assert len(neigh_1) == 1
    assert neigh_1[0] == sd_2

    neigh_2 = mdg.neighboring_subdomains(sd_2)
    assert len(neigh_2) == 2
    assert neigh_2[0] == sd_1 or neigh_2[0] == sd_3
    assert neigh_2[1] == sd_1 or neigh_2[1] == sd_3


def test_subdomain_neighbor_with_dim():
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
    mdg.add_interface(intf_12, (sd_1, sd_2), None)
    mdg.add_interface(intf_32, (sd_2, sd_3), None)

    neigh_1 = mdg.neighboring_subdomains(sd_1, only_higher=True)
    assert len(neigh_1) == 1
    assert neigh_1[0] == sd_2
    neigh_1 = mdg.neighboring_subdomains(sd_1, only_lower=True)
    assert len(neigh_1) == 0

    neigh_2 = mdg.neighboring_subdomains(sd_2, only_higher=True)
    assert len(neigh_2) == 1
    assert neigh_2[0] == sd_3
    neigh_2 = mdg.neighboring_subdomains(sd_2, only_lower=True)
    assert len(neigh_2) == 1
    assert neigh_2[0] == sd_1

    with pytest.raises(ValueError):
        mdg.neighboring_subdomains(sd_1, only_higher=True, only_lower=True)


def mdg_dims_3211():
    """Create a mixed dimensional grid with 4 subdomains and 3 interfaces, to be used
    in sorting tests below."""
    mdg = pp.MixedDimensionalGrid()
    sd_1 = MockGrid(1)
    sd_2 = MockGrid(1)
    sd_3 = MockGrid(2)
    sd_4 = MockGrid(3)
    # Assign subdomains to mdg. Do it in order different from the expected sorting for
    # good measure.
    mdg.add_subdomains(sd_1)
    mdg.add_subdomains(sd_3)
    mdg.add_subdomains(sd_2)
    mdg.add_subdomains(sd_4)
    # Interface dimension is inherited from the secondary subdomain
    intf_21 = mock_mortar_grid(sd_2, sd_1)  # dim=1
    intf_31 = mock_mortar_grid(sd_3, sd_1)  # dim=1
    intf_43 = mock_mortar_grid(sd_4, sd_3)  # dim=2
    mdg.add_interface(intf_21, (sd_2, sd_1), None)
    mdg.add_interface(intf_31, (sd_1, sd_3), None)
    mdg.add_interface(intf_43, (sd_4, sd_3), None)
    return intf_21, intf_31, intf_43, mdg, sd_1, sd_2, sd_3, sd_4


def test_sorting():
    """Test sorting of subdomains.
    The sorting should be based on the dimension of the subdomains and then the
    subdomain id, which corresponds to the order of initiation.
    """
    intf_21, intf_31, intf_43, mdg, sd_1, sd_2, sd_3, sd_4 = mdg_dims_3211()

    sd_list = mdg.subdomains()
    assert sd_list[0] == sd_4
    assert sd_list[1] == sd_3
    assert sd_list[2] == sd_1
    assert sd_list[3] == sd_2

    # Test sorting of interfaces. Same sorting criteria as for subdomains.
    intf_list = mdg.interfaces()
    assert intf_list[0] == intf_43  # First because of dimension
    assert intf_list[1] == intf_21  # Precedes 13 because of interface id
    assert intf_list[2] == intf_31


def test_sort_lists():
    """Test sorting of lists of subdomains and interfaces.
    Includes random lists, non-complete lists, and lists with duplicates.
    """
    intf_21, intf_31, intf_43, mdg, sd_1, sd_2, sd_3, sd_4 = mdg_dims_3211()
    sd_list = mdg.subdomains()
    intf_list = mdg.interfaces()

    sd_list_random = [sd_1, sd_4, sd_2, sd_3]
    sorted_sds = mdg.sort_subdomains(sd_list_random)
    for sd_0, sd_1 in zip(sd_list, sorted_sds):
        assert sd_0 == sd_1
    # Same for interfaces
    intf_list_random = [intf_21, intf_43, intf_31]
    sorted_intfs = mdg.sort_interfaces(intf_list_random)
    for intf_0, intf_1 in zip(intf_list, sorted_intfs):
        assert intf_0 == intf_1

    subdomains_132 = [sd_1, sd_3, sd_2]
    sorted_13 = mdg.sort_subdomains(subdomains_132)
    assert sorted_13[0] == sd_3
    assert sorted_13[1] == sd_1
    assert sorted_13[2] == sd_2

    # Test sorting of lists with duplicates. The duplicates are not removed and should
    # both appear in the sorted list.
    duplicates = [sd_1, sd_2, sd_1, sd_3]
    sorted_duplicates = mdg.sort_subdomains(duplicates)
    assert sorted_duplicates[2] == sd_1
    assert sorted_duplicates[3] == sd_1


def test_interface_to_subdomain_pair():
    """Test the method interface_to_subdomain_pair.
    Check that the returned subdomain pair is sorted according to grid dimension and id.
    """
    intf_21, intf_31, intf_43, mdg, sd_1, sd_2, sd_3, sd_4 = mdg_dims_3211()
    sd_pair_21 = mdg.interface_to_subdomain_pair(intf_21)
    assert sd_pair_21[0] == sd_1  # First because of id
    assert sd_pair_21[1] == sd_2
    sd_pair_31 = mdg.interface_to_subdomain_pair(intf_31)
    assert sd_pair_31[0] == sd_3  # First because of dimension
    assert sd_pair_31[1] == sd_1
    sd_pair_43 = mdg.interface_to_subdomain_pair(intf_43)
    assert sd_pair_43[0] == sd_4  # First because of dimension
    assert sd_pair_43[1] == sd_3


def test_subdomain_to_boundary_grid():
    """Test the method `subdomain_to_boundary_grid`. Also test restoring
    a subdomain from its boundary grid.

    """
    _, _, _, mdg, sd_1d_0, sd_1d_1, sd_2d, sd_3d = mdg_dims_3211()

    # First, check the mapping for the present subdomains and boundaries.
    bg = mdg.subdomain_to_boundary_grid(sd_3d)
    assert bg is not None
    assert bg.dim == 2, "Subdomain is 3D, boundary must be 2D."

    result_sd = bg.parent
    assert result_sd is sd_3d, "Must be the same object."

    # Next, check the grids which are not present in the mdg.
    unwanted_sd = MockGrid(3)
    bg_or_none = mdg.subdomain_to_boundary_grid(unwanted_sd)
    assert bg_or_none is None


# ------ Test of iterators ------*


def test_subdomain_interface_boundary_iterators():
    """Check the iterators `subdomains`, `interfaces` and `boundaries`."""
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

    # First, use the subdomains() function
    found = {sd_1: False, sd_2: False, sd_3: False}
    for sd in mdg.subdomains():
        found[sd] = True
    assert all(found.values())

    # Next, check the interfaces
    found = {intf_12: False, intf_32: False}
    for intf in mdg.interfaces():
        found[intf] = True
    assert all(found.values())

    # Finally, check the boundaries
    parent_grids_found = {sd_1: False, sd_2: False, sd_3: False}
    for bg in mdg.boundaries():
        parent_sd = bg.parent
        parent_grids_found[parent_sd] = True
    assert all(parent_grids_found.values())


def test_iterators_dim_keyword():
    """Check that the iterators `subdomains`, `interfaces` and `boundaries`
    return only the grids of the right dimension when the argument `dim` is
    provided.

    """
    mdg = pp.MixedDimensionalGrid()
    sd_1 = MockGrid(1)
    sd_2 = MockGrid(2)
    sd_3 = MockGrid(3)
    sd_4 = MockGrid(2)
    for sd in [sd_1, sd_2, sd_3, sd_4]:
        mdg.add_subdomains(sd)

    intf_12 = mock_mortar_grid(sd_2, sd_1)
    intf_32 = mock_mortar_grid(sd_3, sd_2)
    mdg.add_interface(intf_12, (sd_1, sd_2), None)
    mdg.add_interface(intf_32, (sd_2, sd_3), None)

    # First, use the subdomains(dim=2) function
    found = {sd_2: False, sd_4: False}
    for sd in mdg.subdomains(dim=2):
        found[sd] = True
    assert all(found.values())

    # Next, use the interfaces(dim=2) function
    found = {intf_32: False}
    for intf in mdg.interfaces(dim=2):
        found[intf] = True
    assert all(found.values())

    # Finally, use the boundaries(dim=1) function
    found_parent_grid = {sd_2: False, sd_4: False}
    for bg in mdg.boundaries(dim=1):
        parent_sd = bg.parent
        found_parent_grid[parent_sd] = True

    assert all(found_parent_grid.values())


def test_data_getters():
    """Check methods `boundary_grid_data`, `interface_data` and `subdomain_data."""
    _, _, _, mdg, _, _, _, _ = mdg_dims_3211()

    # First, check subdomains.
    for sd, data_from_iterator in mdg.subdomains(return_data=True):
        data_from_getter = mdg.subdomain_data(sd)
        assert isinstance(data_from_iterator, dict), "Must be initialized."
        assert data_from_iterator is data_from_getter, "Must be the same object."

    # Next, check interfaces.
    for intf, data_from_iterator in mdg.interfaces(return_data=True):
        data_from_getter = mdg.interface_data(intf)
        assert isinstance(data_from_iterator, dict), "Must be initialized."
        assert data_from_iterator is data_from_getter, "Must be the same object."

    # Finally, check boundaries.
    for bg, data_from_iterator in mdg.boundaries(return_data=True):
        data_from_getter = mdg.boundary_grid_data(bg)
        assert isinstance(data_from_iterator, dict), "Must be initialized."
        assert data_from_iterator is data_from_getter, "Must be the same object."


def test_contains_subdomain():
    mdg = simple_mdg(1)

    for sd in mdg.subdomains():
        assert sd in mdg

    # Define a grid that is not in the mdg
    sd = MockGrid()
    assert sd not in mdg


def test_contains_interface():
    mdg = pp.MixedDimensionalGrid()
    sd_1 = MockGrid(1)
    sd_2 = MockGrid(2)
    sd_3 = MockGrid(3)
    mdg.add_subdomains(sd_1)
    mdg.add_subdomains(sd_2)
    mdg.add_subdomains(sd_3)
    intf_12 = mock_mortar_grid(sd_2, sd_1)
    mdg.add_interface(intf_12, (sd_1, sd_2), None)

    # This edge is defined
    assert intf_12 in mdg
    # this is not
    intf_31 = mock_mortar_grid(sd_3, sd_1)
    assert intf_31 not in mdg


def test_contains_boundary_grid() -> None:
    mdg = simple_mdg(2)
    for bg in mdg.boundaries():
        assert bg in mdg

    # Creating new boundary outside mdg.
    bg = pp.BoundaryGrid(mdg.subdomains()[0])
    assert bg not in mdg


def test_diameter():
    sd_1 = MockGrid(1, 2)
    sd_2 = MockGrid(2, 3)
    mdg = pp.MixedDimensionalGrid()
    mdg.add_subdomains(sd_1)
    mdg.add_subdomains(sd_2)

    assert mdg.diameter() == 3
    assert mdg.diameter(lambda g: g.dim == 1) == 2


def test_bounding_box():
    """Check the method returning the minimum and maximum coordinates of a
    mixed-dimensional grid.
    """
    mdg = pp.MixedDimensionalGrid()
    sd_1 = pp.CartGrid(np.array([1, 1, 1]))
    sd_1.nodes = np.random.random((sd_1.dim, sd_1.num_nodes))
    sd_2 = pp.CartGrid(np.array([1, 1, 1]))
    # Shift g2 with 1
    sd_2.nodes = 1 + np.random.random((sd_2.dim, sd_2.num_nodes))

    mdg.add_subdomains([sd_1, sd_2])
    bmin, bmax = pp.domain.mdg_minmax_coordinates(mdg)

    # Since g2 is shifted, minimum should be at g1, maximum in g2
    assert np.allclose(bmin, sd_1.nodes.min(axis=1))
    assert np.allclose(bmax, sd_2.nodes.max(axis=1))


def test_num_cells():
    sd_1 = MockGrid(dim=1, num_cells=1, num_faces=3, num_nodes=3)
    sd_2 = MockGrid(dim=2, num_cells=3, num_faces=7, num_nodes=3)
    mdg = pp.MixedDimensionalGrid()
    mdg.add_subdomains([sd_1, sd_2])

    assert mdg.num_subdomain_cells() == (sd_1.num_cells + sd_2.num_cells)

    lam = lambda g: g.dim == 1
    assert mdg.num_subdomain_cells(lam) == sd_1.num_cells


def test_num_subdomains_and_interfaces():
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
    assert mdg.num_interfaces() == 2
    assert mdg.num_subdomains() == 3


def test_str_repr():
    sd_1 = MockGrid(dim=1, num_cells=1, num_faces=3, num_nodes=3)
    sd_2 = MockGrid(dim=2, num_cells=3, num_faces=7, num_nodes=3)
    mdg = pp.MixedDimensionalGrid()
    mdg.add_subdomains([sd_1, sd_2])
    mdg.__str__()
    mdg.__repr__()


def test_pickle_md_grid():
    fracs = [np.array([[0, 2], [1, 1]]), np.array([[1, 1], [0, 2]])]
    mdg = pp.meshing.cart_grid(fracs, np.array([2, 2]))

    fn = "tmp.md_grid"
    pickle.dump(mdg, open(fn, "wb"))
    mdg_read = pickle.load(open(fn, "rb"))

    test_utils.compare_md_grids(mdg, mdg_read)

    # Delete the temporary file
    os.remove(fn)
