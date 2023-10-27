""" Tests of methods from porepy.grids.partition.
"""
import pytest

import numpy as np

import porepy as pp
from porepy.applications.test_utils import reference_dense_arrays


@pytest.mark.parametrize(
    "grid_size, coarse_dims",
    (
        [[4, 10], np.array([2, 3])],
        [[4, 4, 4], np.array([2, 2, 2])],
        [[6, 5, 4], np.array([3, 2, 2])],
    ),
)
def test_cartesian_grids(grid_size, coarse_dims):
    """Create a Cartesian grid and partition it. Compare the partitioning to a
    reference value.

    Parameters:
        grid_size: Number of cells in each direction in the fine grid.
        coarse_dims: Number of coarse cells in each direction.

    """
    # Create a Cartesian grid
    g = pp.CartGrid(grid_size)
    # Partition the grid
    p = pp.partition.partition_structured(g, coarse_dims=coarse_dims)

    # Fetch the stored reference value.
    # The key is a string of the grid size, e.g. "4_10" for a 4x10 grid.
    key = "_".join([str(ni) for ni in grid_size])
    p_known = reference_dense_arrays.test_partition["test_cartesian_grids"][key][
        "p_known"
    ]
    # Check that the partitioning is correct
    assert np.allclose(p, p_known)


@pytest.mark.parametrize(
    "nx, target, coarse",
    (
        [np.array([5, 5, 5]), 125, np.array([5, 5, 5])],
        [np.array([5, 5, 5]), 1, np.array([1, 1, 1])],
        [np.array([10, 4]), 4, np.array([2, 2])],
        [np.array([10, 10]), 17, np.array([4, 4])],
        [np.array([10, 10]), 19, np.array([4, 5])],
        # The will ideally require a 5x4 grid, but the limit on two cells in
        # the  y-direction should redirect to a 10x2.
        [np.array([100, 2]), 20, np.array([2, 10])],
        # This will seemingly require 900^1/3 ~ 10 cells in each direction, but
        # the z-dimension has only 1, thus the two other have to do 30 each. y
        # can only do 15, so the loop has to be reset once more to get the
        # final 60.
        [np.array([2000, 15, 1]), 900, np.array([1, 15, 60])],
    ),
)
def test_coarse_dimension_determination(nx, target, coarse):
    """Test functionality to determine coarse dimensions."""
    coarse_dims = pp.partition.determine_coarse_dimensions(target, nx)
    assert np.array_equal(np.sort(coarse_dims), coarse)


class TestExtractSubGrid:
    def compare_grid_geometries(self, g, h, sub_c, sub_f, sub_n):
        assert np.allclose(g.nodes[:, sub_n], h.nodes)
        assert np.allclose(g.face_areas[sub_f], h.cell_volumes)
        assert np.allclose(g.face_centers[:, sub_f], h.cell_centers)

    def test_assertion(self):
        g = pp.CartGrid([1, 1, 1])

        f = np.array([0, 0, 1]) < 0.5

        with pytest.raises(IndexError):
            pp.partition.extract_subgrid(g, f, faces=True)
        with pytest.raises(IndexError):
            pp.partition.extract_subgrid(g, f)

    def test_assertion_extraction(self):
        g = pp.CartGrid([1, 1, 1])

        f = np.array([0, 2])
        with pytest.raises(ValueError):
            pp.partition.extract_subgrid(g, f, faces=True)

    def test_cart_grid_3d(self):
        g = pp.CartGrid([2, 3, 1])
        g.compute_geometry()

        f = g.face_centers[1] < 1e-10
        true_nodes = np.where(g.nodes[1] < 1e-10)[0]

        g.nodes[[0, 2]] = (
            g.nodes[[0, 2]] + 0.2 * np.random.random(g.nodes.shape)[[0, 2]]
        )
        g.nodes[1] = g.nodes[1] + 0.2 * np.random.rand(1)
        g.compute_geometry()

        h, sub_f, sub_n = pp.partition.extract_subgrid(g, f, faces=True)

        assert np.array_equal(true_nodes, sub_n)
        assert np.array_equal(np.where(f)[0], sub_f)
        self.compare_grid_geometries(g, h, None, f, true_nodes)

    def test_cart_2d(self):
        g = pp.CartGrid([2, 3])
        g.nodes = g.nodes + 0.2 * np.random.random(g.nodes.shape)
        g.nodes[2, :] = 0
        g.compute_geometry()

        f = np.array([0, 3, 6])

        h, sub_f, sub_n = pp.partition.extract_subgrid(g, f, faces=True)

        true_nodes = np.array([0, 3, 6, 9])
        true_faces = np.array([0, 3, 6])

        assert np.array_equal(true_nodes, sub_n)
        assert np.array_equal(true_faces, sub_f)

        self.compare_grid_geometries(g, h, f, true_faces, true_nodes)

    def test_cart_1d(self):
        g = pp.TensorGrid(np.array([0, 1, 2]))
        g.nodes = g.nodes + 0.2 * np.random.random(g.nodes.shape)
        g.compute_geometry()

        face_pos = [0, 1, 2]

        for f in face_pos:
            h, sub_f, sub_n = pp.partition.extract_subgrid(g, f, faces=True)

            true_nodes = np.array([f])
            true_faces = np.array([f])

            assert np.array_equal(true_nodes, sub_n)
            assert np.array_equal(true_faces, sub_f)

            self.compare_grid_geometries(g, h, f, true_faces, true_nodes)

    def test_tetrahedron_grid(self):
        g = pp.StructuredTetrahedralGrid([1, 1, 1])
        g.compute_geometry()

        f = g.face_centers[2] < 1e-10
        true_nodes = np.where(g.nodes[2] < 1e-10)[0]

        g.nodes[:2] = g.nodes[:2] + 0.2 * np.random.random(g.nodes.shape)[:2]
        g.nodes[2] = g.nodes[2] + 0.2 * np.random.rand(1)
        g.compute_geometry()

        h, sub_f, sub_n = pp.partition.extract_subgrid(g, f, faces=True)

        assert np.array_equal(true_nodes, sub_n)
        assert np.array_equal(np.where(f)[0], sub_f)

        self.compare_grid_geometries(g, h, None, f, true_nodes)

    def test_cart_2d_with_bend(self):
        """
        grid: |----|----|
              |    |    |
              -----------
              |    |    |
              |    |    |
              -----------
        Extract:
                   -----|
                        |
                        -
                        |
                        |
        """
        g = pp.CartGrid([2, 2])
        g.nodes = g.nodes + 0.2 * np.random.random(g.nodes.shape)
        g.nodes[2, :] = 0
        g.compute_geometry()

        f = np.array([2, 5, 11])

        h, sub_f, sub_n = pp.partition.extract_subgrid(g, f, faces=True)

        true_nodes = np.array([2, 5, 7, 8])
        true_faces = np.array([2, 5, 11])

        assert np.array_equal(true_nodes, sub_n)
        assert np.array_equal(true_faces, sub_f)

        self.compare_grid_geometries(g, h, f, true_faces, true_nodes)


class TestGrids:
    def compare_grid_geometries(self, g, h, sub_c, sub_f, sub_n):
        assert np.array_equal(g.nodes[:, sub_n], h.nodes)
        assert np.array_equal(g.face_areas[sub_f], h.face_areas)
        assert np.array_equal(g.face_normals[:, sub_f], h.face_normals)
        assert np.array_equal(g.face_centers[:, sub_f], h.face_centers)
        assert np.array_equal(g.cell_centers[:, sub_c], h.cell_centers)
        assert np.array_equal(g.cell_volumes[sub_c], h.cell_volumes)

    def test_cart_2d(self):
        g = pp.CartGrid([3, 2])
        g.nodes = g.nodes + 0.2 * np.random.random(g.nodes.shape)
        g.nodes[2, :] = 0
        g.compute_geometry()

        c = np.array([0, 1, 3])

        h, sub_f, sub_n = pp.partition.extract_subgrid(g, c)

        true_nodes = np.array([0, 1, 2, 4, 5, 6, 8, 9])
        true_faces = np.array([0, 1, 2, 4, 5, 8, 9, 11, 12, 14])

        assert np.array_equal(true_nodes, sub_n)
        assert np.array_equal(true_faces, sub_f)

        self.compare_grid_geometries(g, h, c, true_faces, true_nodes)

    def test_cart_3d(self):
        g = pp.CartGrid([4, 3, 3])
        g.nodes = g.nodes + 0.2 * np.random.random((g.dim, g.nodes.shape[1]))
        g.compute_geometry()

        # Pick out cells (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 2), (1, 0, 2)
        c = np.array([0, 1, 4, 12, 24, 25])

        h, sub_f, sub_n = pp.partition.extract_subgrid(g, c)

        # There are 20 nodes in each layer
        nodes_0 = np.array([0, 1, 2, 5, 6, 7, 10, 11])
        nodes_1 = nodes_0 + 20
        nodes_2 = np.array([0, 1, 2, 5, 6, 7]) + 40
        nodes_3 = nodes_2 + 20
        true_nodes = np.hstack((nodes_0, nodes_1, nodes_2, nodes_3))

        faces_x = np.array([0, 1, 2, 5, 6, 15, 16, 30, 31, 32])
        # y-faces start on 45
        faces_y = np.array([45, 46, 49, 50, 53, 61, 65, 77, 78, 81, 82])
        # z-faces start on 93
        faces_z = np.array([93, 94, 97, 105, 106, 109, 117, 118, 129, 130])
        true_faces = np.hstack((faces_x, faces_y, faces_z))

        assert np.array_equal(true_nodes, sub_n)
        assert np.array_equal(true_faces, sub_f)

        self.compare_grid_geometries(g, h, c, true_faces, true_nodes)

    def test_simplex_2d(self):
        g = pp.StructuredTriangleGrid(np.array([3, 2]))
        g.compute_geometry()
        c = np.array([0, 1, 3])
        true_nodes = np.array([0, 1, 4, 5, 6])
        true_faces = np.array([0, 1, 2, 4, 5, 10, 13])

        h, sub_f, sub_n = pp.partition.extract_subgrid(g, c)

        assert np.array_equal(true_nodes, sub_n)
        assert np.array_equal(true_faces, sub_f)

        self.compare_grid_geometries(g, h, c, true_faces, true_nodes)


class TestOverlap:
    def test_overlap_1_layer(self):
        g = pp.CartGrid([5, 5])
        ci = np.array([0, 1, 5, 6])
        ci_overlap = np.array([0, 1, 2, 5, 6, 7, 10, 11, 12])
        assert np.array_equal(pp.partition.overlap(g, ci, 1), ci_overlap)

    def test_overlap_2_layers(self):
        g = pp.CartGrid([5, 5])
        ci = np.array([0, 1, 5, 6])
        ci_overlap = np.array([0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18])
        assert np.array_equal(pp.partition.overlap(g, ci, 2), ci_overlap)


class TestConnectivityChecker:
    def setup(self):
        g = pp.CartGrid([4, 4])
        p = pp.partition.partition_structured(g, coarse_dims=np.array([2, 2]))
        return g, p

    def test_connected(self):
        g, p = self.setup()
        p_sub = np.r_[np.where(p == 0)[0], np.where(p == 1)[0]]
        is_connected, components = pp.partition.grid_is_connected(g, p_sub)
        assert is_connected
        assert np.array_equal(np.sort(components[0]), np.arange(8))

    def test_not_connected(self):
        # Pick out the lower left and upper right squares
        g, p = self.setup()

        p_sub = np.r_[np.where(p == 0)[0], np.where(p == 3)[0]]
        is_connected, components = pp.partition.grid_is_connected(g, p_sub)
        assert not is_connected

        # To test that we pick the right components, we need to map back to
        # global indices.
        assert np.array_equal(np.sort(p_sub[components[0]]), np.array([0, 1, 4, 5]))

        assert np.array_equal(np.sort(p_sub[components[1]]), np.array([10, 11, 14, 15]))

    def test_subgrid_connected(self):
        g, p = self.setup()

        sub_g, _, _ = pp.partition.extract_subgrid(g, np.where(p == 0)[0])
        is_connected, components = pp.partition.grid_is_connected(sub_g)
        assert is_connected
        assert np.array_equal(np.sort(components[0]), np.arange(sub_g.num_cells))

    def test_subgrid_not_connected(self):
        g, p = self.setup()

        p_sub = np.r_[np.where(p == 0)[0], np.where(p == 3)[0]]
        sub_g, _, _ = pp.partition.extract_subgrid(g, p_sub)
        is_connected, components = pp.partition.grid_is_connected(sub_g)
        assert not is_connected
        assert np.array_equal(np.sort(p_sub[components[0]]), np.array([0, 1, 4, 5]))

        assert np.array_equal(np.sort(p_sub[components[1]]), np.array([10, 11, 14, 15]))


class TestCoordinatePartitioner:
    def test_cart_grid_square(self):
        g = pp.CartGrid([4, 4])
        g.compute_geometry()
        num_coarse = 4

        pvec = pp.partition.partition_coordinates(g, num_coarse)

        known_cells = [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]
        for k in known_cells:
            assert np.all(np.abs(pvec[k] - pvec[k[0]]) < 1e-4)

    def test_cart_grid_rectangle(self):
        # Rectangular domain
        g = pp.CartGrid([4, 2])
        g.compute_geometry()
        num_coarse = 2
        pvec = pp.partition.partition_coordinates(g, num_coarse)

        # The grid can be split horizontally or vertically, we can't know how
        possible_cells = [[0, 1, 4, 5], [2, 3, 6, 7]]
        possible_cells_2 = [[0, 1, 2, 3], [4, 5, 6, 7]]
        for k, k2 in zip(possible_cells, possible_cells_2):
            assert np.all(np.abs(pvec[k] - pvec[k[0]]) < 1e-4) or np.all(
                np.abs(pvec[k2] - pvec[k2[0]]) < 1e-4
            )


class TestPartitionGrid:
    def test_identity_partitioning(self):
        # All cells are on the same grid, nothing should happen
        g = pp.CartGrid([3, 4])
        ind = np.zeros(g.num_cells)

        sub_g, face_map_list, node_map_list = pp.partition.partition_grid(g, ind)

        nsg = np.unique(ind).size
        assert len(sub_g) == nsg
        assert len(face_map_list) == nsg
        assert len(node_map_list) == nsg

        sg = sub_g[0]
        assert sg.num_cells == g.num_cells
        assert sg.num_faces == g.num_faces
        assert sg.num_nodes == g.num_nodes

        assert np.allclose(face_map_list[0], np.arange(sg.num_faces))
        assert np.allclose(node_map_list[0], np.arange(sg.num_nodes))

    def test_single_cell_partitioning(self):
        g = pp.CartGrid([3, 3])
        ind = np.arange(g.num_cells)
        sub_g, face_map_list, node_map_list = pp.partition.partition_grid(g, ind)

        nsg = np.unique(ind).size
        assert len(sub_g) == nsg
        assert len(face_map_list) == nsg
        assert len(node_map_list) == nsg

        for ci, (sg, fm, nm) in enumerate(zip(sub_g, face_map_list, node_map_list)):
            assert sg.num_cells == 1
            assert sg.num_faces == 4
            assert sg.num_nodes == 4

            f_known = np.where(g.cell_faces.todense()[:, ci] != 0)[0]
            assert np.all(np.sort(fm) == np.sort(f_known))

            n_known = np.where(g.face_nodes.todense()[:, f_known].sum(axis=1) > 0)[0]
            assert np.all(np.sort(nm) == np.sort(n_known))
