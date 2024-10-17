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

import porepy as pp
from porepy.fracs import meshing
from porepy.grids import refinement
from porepy.grids.simplex import TriangleGrid
from porepy.grids.structured import TensorGrid
from porepy.applications.test_utils.arrays import compare_arrays


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
    @pytest.mark.parametrize(
        "g,target_nodes",
        [
            (TensorGrid(np.array([0, 2, 4])), np.arange(5)),  # uniform
            (TensorGrid(np.array([0, 2, 6])), np.array([0, 1, 2, 4, 6])),  # non-uniform
        ],
    )
    def test_refinement_grid_1D(self, g, target_nodes):
        h = refinement.refine_grid_1d(g, ratio=2)
        assert np.allclose(h.nodes[0], target_nodes)

    def test_refinement_grid_1d_general_orientation(self):
        x = np.array([0, 2, 6]) * np.ones((3, 1))
        g = TensorGrid(x[0])
        g.nodes = x
        h = refinement.refine_grid_1d(g, ratio=2)
        assert np.allclose(
            h.nodes, np.array([[0, 1, 2, 4, 6], [0, 1, 2, 4, 6], [0, 1, 2, 4, 6]])
        )


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
                np.allclose(low_to_mortar_known_avg, intf.secondary_to_mortar_avg().toarray()),
                np.allclose(
                    low_to_mortar_known_avg,
                    intf.secondary_to_mortar_avg().toarray()[::-1],
                ),
            )

            if low_to_mortar_known_int is not None:
                assert np.logical_or(
                    np.allclose(
                        low_to_mortar_known_int, intf.secondary_to_mortar_int().toarray()
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
