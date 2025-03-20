"""Testing of

- :func:`porepy.grids.refinement.distort_grid_1d`
- :func:`porepy.grids.refinement.refine_grid_1d`
- :func:`porepy.grids.refinement.refine_triangle_grid`
- :func:`porepy.grids.refinement.remesh_1d`
- :func:`porepy.fracs.meshing.cart_grid`

"""

from __future__ import division

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp
from porepy.applications.test_utils.arrays import compare_arrays
from porepy.fracs import meshing
from porepy.grids import refinement
from porepy.grids.simplex import TriangleGrid
from porepy.grids.structured import TensorGrid


class TestGridPerturbation:
    def test_grid_perturbation_1d_bound_nodes_fixed(self):
        g = TensorGrid(np.array([0, 1, 2]))
        h = refinement.distort_grid_1d(g)
        assert np.allclose(g.nodes[:, [0, 2]], h.nodes[:, [0, 2]])

    def test_grid_perturbation_1d_internal_and_bound_nodes_fixed(self):
        g = TensorGrid(np.arange(4))
        h = refinement.distort_grid_1d(g, fixed_nodes=[0, 1, 3])
        assert np.allclose(g.nodes[:, [0, 1, 3]], h.nodes[:, [0, 1, 3]])

    def test_grid_perturbation_1d_internal_nodes_fixed(self):
        g = TensorGrid(np.arange(4))
        h = refinement.distort_grid_1d(g, fixed_nodes=[1])
        assert np.allclose(g.nodes[:, [0, 1, 3]], h.nodes[:, [0, 1, 3]])


class TestGridRefinement1D:
    """Tests for the 1D grid refinement functions.

    The test are in effect sanity checks on the refined grid, also in cases with
    non-uniform grid spacing, permuted node/cell orderings and split nodes.

    """

    def compare_refined_grids(self, g: pp.Grid, g_ref: pp.Grid, ratio: int) -> None:
        """Helper function to compare the refined grid with the original grid.

        See comments in the below code for which properties are compared.

        Parameters:
            g: The original grid.
            g_ref: The refined grid.
            ratio: The ratio of the number of cells in the refined grid to the original
                grid.

        """

        # The two grids should cover the same domain.
        assert np.sum(g.cell_volumes) == np.sum(g_ref.cell_volumes)

        # If the ratio is given, all cells should be refined by this ratio.
        if ratio is not None:
            assert np.allclose(g_ref.num_cells / g.num_cells, ratio)

        # There should be as many faces as nodes in a 1d grid. Check that the face-node
        # map is 1-1, and that there are as many node coordinates as there are nodes.
        assert g_ref.face_nodes.shape[0] == g_ref.face_nodes.shape[1]
        assert g_ref.face_nodes.shape[0] == g_ref.nodes.shape[1]

        # Also check that the face-node map is 1-1.
        assert np.linalg.det(g_ref.face_nodes.toarray()) == 1

        # These are 1d grids, thus the number of boundary faces should be the same in
        # the two grids. Exploit this to check the refined cell-face mapping, which has
        # a somewhat non-trivial construction.
        num_boundary_g = np.sum(np.sum(g.cell_faces, axis=1) != 0)
        num_boundary_g_ref = np.sum(np.sum(g_ref.cell_faces, axis=1) != 0)
        assert num_boundary_g == num_boundary_g_ref

        # Check that the refined grid has the right node coordinates. It is not
        # meaningful to directly check the order of the nodes, as the API for the
        # refinement function does not give any guarantees of the cell/node ordering,
        # nor on the relation between the topology of the original and refined grid.
        # Still, we know the node coordinates of the old grid, and can calculate the
        # expected node coordinates of the refined grid and verify that the right nodes
        # are in place. Also note that if the cell-node relation of the refined grid is
        # somehow shuffled, so that cells are overlapping, this would lead to too large
        # a volume, and the first assertion would fail (unless the original grid is also
        # broken, in which case the test will quite likely still fail).

        # Shortcut: Assume the grids are algined with the x-axis, so that ordering nodes
        # etc. is easy. This is not a critical assumption, but it makes the test easier.
        # Failure here signifies that a test grid has been created that violates the
        # assumption - if so, either generalize the below test or change the grid.
        assert np.allclose(g.nodes[1:], 0)
        assert np.allclose(g_ref.nodes[1:], 0)

        # Sort the nodes in the old grid and find the grid spacing.
        x_old = np.sort(g.nodes[0])
        dx = np.diff(x_old) / ratio
        # Initialize with the first node.
        x_expected = [x_old[0]]
        # Loop over all intervals, add nodes between the old nodes provided the grid
        # spacing is positive (zero grid size indicates a split node, in which case no
        # nodes should be added [this may not be true for a broken original grid, but we
        # disregard that possibility]).
        for i in range(1, len(x_old)):
            if dx[i - 1] > 0:
                x_expected.extend(
                    [x_old[i - 1] + j * dx[i - 1] for j in range(1, ratio)]
                )
            # Always add the old node at the end of the interval.
            x_expected.append(x_old[i])

        # Sort also the nodes of the refined grid and compare with the expected nodes.
        x_ref = np.sort(g_ref.nodes[0])
        assert np.allclose(x_ref, np.asarray(x_expected))

    @pytest.mark.parametrize("ratio", [2, 3])
    def test_refinement_grid_1D(self, ratio: int):
        """Test the refinement of a 1D grid

        Parameters:
            ratio: The ratio of the number of cells in the refined grid to the original
                grid.

        """
        # First create four grids with increasing complexity, then refine each of them
        # and verify that the refined grid has the expected properties.

        # A simple uniform grid.
        g_0 = pp.TensorGrid(np.array([0, 1, 2, 3, 4]))

        # A grid with non-uniform spacing.
        g_1 = pp.TensorGrid(np.array([0, 1, 2, 2.5, 3, 4]))

        # A grid with a permuted node/face (they can be considered identical in 1d) and
        # cell ordering: The nodes lie along the x-axis, but in the order [1, 0, 2, 3].
        # The cells are ordered (with increasing x-coordinates) [1, 0, 2].
        x = np.array([[1, 0, 2, 3], [0, 0, 0, 0], [0, 0, 0, 0]])
        fn = sps.identity(4, format="csr")
        cf = sps.csc_array(np.array([[-1, 1, 0], [0, -1, 0], [1, 0, -1], [0, 0, -1]]))
        g_2 = pp.Grid(dim=1, nodes=x, face_nodes=fn, cell_faces=cf, name="")

        # A grid where one node (coordinate x=2) has been split, as would happen if two
        # 1d grids representing fractures intersect.
        x = np.array([[0, 1, 2, 2, 3], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        fn = sps.identity(5, format="csr")
        cf = sps.csc_array(
            np.array([[-1, 1, 0, 0, 0], [0, -1, 1, 0, 0], [0, 0, 0, -1, 1]]).T
        )
        g_3 = pp.Grid(dim=1, nodes=x, face_nodes=fn, cell_faces=cf, name="")

        for g in [g_0, g_1, g_2, g_3]:
            g.compute_geometry()
            g_ref = pp.refinement.refine_grid_1d(g, ratio)
            self.compare_refined_grids(g, g_ref, ratio)


class TestGridRefinement2DSimplex:
    @pytest.fixture
    def one_cell_grid(self):
        g = TriangleGrid(
            np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]),
            np.array([[0], [1], [2]]),
            "GridWithOneCell",
        )
        setattr(g, "history", ["OneGrid"])
        return g

    @pytest.fixture
    def two_cell_grid(self):
        g = TriangleGrid(
            np.array([[0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]]),
            np.array([[0, 1], [1, 3], [2, 2]]),
            "GridWithTwoCells",
        )
        setattr(g, "history", ["TwoGrid"])
        return g

    def test_refinement_single_cell(self, one_cell_grid):
        g = one_cell_grid
        h, parent = refinement.refine_triangle_grid(g)
        h.compute_geometry()

        assert h.num_cells == 4
        assert h.num_faces == 9
        assert h.num_nodes == 6
        assert h.cell_volumes.sum() == 0.5
        assert np.allclose(h.cell_volumes, 1 / 8)

        known_nodes = np.array([[0, 0.5, 1, 0.5, 0, 0], [0, 0, 0, 0.5, 1, 0.5]])
        compare_arrays(h.nodes[:2], known_nodes)
        assert np.all(parent == 0)

    def test_refinement_two_cells(self, two_cell_grid):
        g = two_cell_grid
        h, parent = refinement.refine_triangle_grid(g)
        h.compute_geometry()

        assert h.num_cells == 8
        assert h.num_faces == 16
        assert h.num_nodes == 9
        assert h.cell_volumes.sum() == 1
        assert np.allclose(h.cell_volumes, 1 / 8)

        known_nodes = np.array(
            [[0, 0.5, 1, 0.5, 0, 0, 1, 1, 0.5], [0, 0, 0, 0.5, 1, 0.5, 0.5, 1, 1]]
        )
        compare_arrays(h.nodes[:2], known_nodes)
        assert np.sum(parent == 0) == 4
        assert np.sum(parent == 1) == 4
        assert np.allclose(np.bincount(parent, h.cell_volumes), 0.5)


class TestRefinementMortarGrid:
    @pytest.fixture(scope="function")
    def mdg(self):
        """Fixture representing md grid for tests.

        Important:
            Scope is set to ``function``, since the refinement in each step modifies
            the md grid. Otherwise tests are working on a single md-grid modified by the
            previous grid.

        """
        f1 = np.array([[0, 1], [0.5, 0.5]])
        mdg = meshing.cart_grid([f1], [2, 2], **{"physdims": [1, 1]})
        mdg.compute_geometry()
        return mdg

    @pytest.mark.parametrize(
        "ref_num_nodes,high_to_mortar_known,low_to_mortar_known",
        [
            (
                4,
                1.0
                / 3.0
                * np.matrix(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                    ]
                ),
                1.0 / 3.0 * np.matrix([[0, 2], [1, 1], [2, 0], [0, 2], [1, 1], [2, 0]]),
            ),
            (
                None,
                np.matrix(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 2 / 3, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1 / 3, 1 / 3, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 2 / 3, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5],
                    ]
                ),
                np.matrix(
                    [
                        [0, 2 / 3],
                        [1 / 3, 1 / 3],
                        [2 / 3, 0],
                        [0, 0.5],
                        [0, 0.5],
                        [0.5, 0],
                        [0.5, 0],
                    ]
                ),
            ),
        ],
    )
    def test_mortar_grid_1d_refine_mortar_grids(
        self,
        ref_num_nodes,
        high_to_mortar_known,
        low_to_mortar_known,
        request,
    ):
        """Tests the refinement of of side grids of interfaces.

        ``ref_num_nodes`` can either be an integer, for uniform refinement,
        or None, to use the side-grid value for non-uniform refinement.

        Note:
            (VL) probably not a good parametrization of ``ref_num_nodes``.

        """

        mdg: pp.MixedDimensionalGrid = request.getfixturevalue("mdg")

        for intf in mdg.interfaces():
            if ref_num_nodes is None:
                new_side_grids = {
                    s: refinement.remesh_1d(g, num_nodes=s.value + 3)
                    for s, g in intf.side_grids.items()
                }
            else:
                new_side_grids = {
                    s: refinement.remesh_1d(g, num_nodes=ref_num_nodes)
                    for s, g in intf.side_grids.items()
                }

            intf.update_mortar(new_side_grids, 1e-4)

            assert np.allclose(
                high_to_mortar_known, intf.primary_to_mortar_int().todense()
            )
            assert np.allclose(
                low_to_mortar_known, intf.secondary_to_mortar_int().todense()
            )

    @pytest.mark.parametrize(
        "ref_num_nodes,high_to_mortar_known,"
        + "low_to_mortar_known_avg,low_to_mortar_known_int",
        [
            (
                5,
                np.matrix(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    ]
                ),
                np.matrix(
                    [
                        [0, 0, 0.5, 0.5],
                        [0.5, 0.5, 0, 0],
                        [0, 0, 0.5, 0.5],
                        [0.5, 0.5, 0, 0],
                    ]
                ),
                None,
            ),
            (
                4,
                np.matrix(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    ]
                ),
                1.0 / 3.0 * np.matrix([[0, 1, 2], [2, 1, 0], [0, 1, 2], [2, 1, 0]]),
                np.matrix([[0, 0.5, 1], [1, 0.5, 0], [0, 0.5, 1], [1, 0.5, 0]]),
            ),
        ],
    )
    def test_mortar_grid_1d_refine_1d_grid(
        self,
        ref_num_nodes,
        high_to_mortar_known,
        low_to_mortar_known_avg,
        low_to_mortar_known_int,
        request,
    ):
        """Refine the lower-dimensional grid so that it is matching with the
        higher dimensional grid.

        ``low_to_mortar_known_int`` can be None to skip the testing of this,
        if target values are not available.
        """
        mdg: pp.MixedDimensionalGrid = request.getfixturevalue("mdg")

        for intf in mdg.interfaces():
            # refine the 1d-physical grid
            old_g = mdg.interface_to_subdomain_pair(intf)[1]
            new_g = refinement.remesh_1d(old_g, num_nodes=ref_num_nodes)
            new_g.compute_geometry()

            mdg.replace_subdomains_and_interfaces(sd_map={old_g: new_g})
            intf.update_secondary(new_g, 1e-4)

            assert np.allclose(
                high_to_mortar_known, intf.primary_to_mortar_int().todense()
            )
            # The ordering of the cells in the new 1d grid may be flipped on
            # some systems; therefore allow two configurations
            assert np.logical_or(
                np.allclose(
                    low_to_mortar_known_avg, intf.secondary_to_mortar_avg().toarray()
                ),
                np.allclose(
                    low_to_mortar_known_avg,
                    intf.secondary_to_mortar_avg().toarray()[::-1],
                ),
            )

            if low_to_mortar_known_int is not None:
                assert np.logical_or(
                    np.allclose(
                        low_to_mortar_known_int,
                        intf.secondary_to_mortar_int().toarray(),
                    ),
                    np.allclose(
                        low_to_mortar_known_int,
                        intf.secondary_to_mortar_int().toarray()[::-1],
                    ),
                )

    def test_mortar_grid_2d(self):
        f = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]])
        mdg = meshing.cart_grid([f], [2] * 3, **{"physdims": [1] * 3})
        mdg.compute_geometry()

        for intf in mdg.interfaces():
            indices_known = np.array([28, 29, 30, 31, 36, 37, 38, 39])
            assert np.array_equal(intf.primary_to_mortar_int().indices, indices_known)

            indptr_known = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
            assert np.array_equal(intf.primary_to_mortar_int().indptr, indptr_known)

            data_known = np.array([1, 1, 1, 1, 1, 1, 1, 1.0])
            assert np.array_equal(intf.primary_to_mortar_int().data, data_known)

            indices_known = np.array([0, 4, 1, 5, 2, 6, 3, 7])
            assert np.array_equal(intf.secondary_to_mortar_int().indices, indices_known)

            indptr_known = np.array([0, 2, 4, 6, 8])
            assert np.array_equal(intf.secondary_to_mortar_int().indptr, indptr_known)

            data_known = np.array([1, 1, 1, 1, 1, 1, 1, 1.0])
            assert np.array_equal(intf.secondary_to_mortar_int().data, data_known)


class TestGridFactory:
    def test_nested_simple(self):
        # Simple test of the nested generation. Both 2d and 3d domains.
        # Main check: The refinement is indeed by splitting
        mesh_args = {"mesh_size_frac": 1, "mesh_size_bound": 1, "mesh_size_min": 0.1}

        num_ref = 3

        params = {
            "mode": "nested",
            "num_refinements": num_ref,
            "mesh_param": mesh_args,
        }

        network_2d = pp.create_fracture_network(
            domain=pp.domains.unit_cube_domain(dimension=2)
        )
        network_3d = pp.create_fracture_network(
            domain=pp.domains.unit_cube_domain(dimension=3)
        )

        for network in [network_2d, network_3d]:
            factory = pp.refinement.GridSequenceFactory(network, params)

            num_cells_prev = 0

            for counter, mdg in enumerate(factory):
                assert counter < num_ref
                dim = mdg.dim_max()
                if counter > 0:
                    assert mdg.num_subdomain_cells() == num_cells_prev * (2**dim)
                num_cells_prev = mdg.num_subdomain_cells()

    def test_nested_pass_grid_args(self):
        # Check that grid arguments to the fracture network meshing ('grid_param' below)
        # are correctly passed
        mesh_args = {"mesh_size_frac": 1, "mesh_size_bound": 1}

        num_ref = 2

        # Add a single fracture to the network
        fracture = pp.LineFracture(np.array([[0.3, 0.7], [0.5, 0.5]]))
        network = pp.create_fracture_network(
            [fracture], pp.domains.unit_cube_domain(dimension=2)
        )

        params = {
            "mode": "nested",
            "num_refinements": num_ref,
            "mesh_param": mesh_args,
            # The fracture is a constraint
            "grid_param": {"constraints": np.array([0])},
        }

        factory = pp.refinement.GridSequenceFactory(network, params)

        for counter, mdg in enumerate(factory):
            dim = mdg.dim_max()
            for dim in range(dim):
                # Since there are no true fractures in the network (only constraints)
                # there should be no lower-dimensional grids
                assert len(mdg.subdomains(dim=dim)) == 0
