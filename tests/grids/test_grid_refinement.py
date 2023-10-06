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
from tests import test_utils


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
        test_utils.compare_arrays(h.nodes[:2], known_nodes)
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
        test_utils.compare_arrays(h.nodes[:2], known_nodes)
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
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            2.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            2.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            2.0,
                            0.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            1.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            2.0,
                        ],
                    ]
                ),
                1.0
                / 3.0
                * np.matrix(
                    [
                        [0.0, 2.0],
                        [1.0, 1.0],
                        [2.0, 0.0],
                        [0.0, 2.0],
                        [1.0, 1.0],
                        [2.0, 0.0],
                    ]
                ),
            ),
            (
                None,
                np.matrix(
                    [
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.66666667,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.33333333,
                            0.33333333,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.66666667,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.5,
                            0.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.5,
                            0.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.5,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.5,
                        ],
                    ]
                ),
                np.matrix(
                    [
                        [0.0, 0.66666667],
                        [0.33333333, 0.33333333],
                        [0.66666667, 0.0],
                        [0.0, 0.5],
                        [0.0, 0.5],
                        [0.5, 0.0],
                        [0.5, 0.0],
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
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                        ],
                    ]
                ),
                np.matrix(
                    [
                        [0.0, 0.0, 0.5, 0.5],
                        [0.5, 0.5, 0.0, 0.0],
                        [0.0, 0.0, 0.5, 0.5],
                        [0.5, 0.5, 0.0, 0.0],
                    ]
                ),
                None,
            ),
            (
                4,
                np.matrix(
                    [
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                        ],
                    ]
                ),
                1.0
                / 3.0
                * np.matrix(
                    [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]
                ),
                np.matrix(
                    [[0.0, 0.5, 1.0], [1.0, 0.5, 0.0], [0.0, 0.5, 1.0], [1.0, 0.5, 0.0]]
                ),
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
                np.allclose(low_to_mortar_known_avg, intf.secondary_to_mortar_avg().A),
                np.allclose(
                    low_to_mortar_known_avg,
                    intf.secondary_to_mortar_avg().A[::-1],
                ),
            )

            if low_to_mortar_known_int is not None:
                assert np.logical_or(
                    np.allclose(
                        low_to_mortar_known_int, intf.secondary_to_mortar_int().A
                    ),
                    np.allclose(
                        low_to_mortar_known_int,
                        intf.secondary_to_mortar_int().A[::-1],
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

            data_known = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            assert np.array_equal(intf.primary_to_mortar_int().data, data_known)

            indices_known = np.array([0, 4, 1, 5, 2, 6, 3, 7])
            assert np.array_equal(intf.secondary_to_mortar_int().indices, indices_known)

            indptr_known = np.array([0, 2, 4, 6, 8])
            assert np.array_equal(intf.secondary_to_mortar_int().indptr, indptr_known)

            data_known = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            assert np.array_equal(intf.secondary_to_mortar_int().data, data_known)
