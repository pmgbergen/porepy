"""Tests of methods from porepy.grids.partition."""

import numpy as np
import pytest

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


class TestExtractSubgrid:
    """Test the extract_subgrid function.

    The tests generate a grid of a given size, and extract a subgrid based on a set of
    cell indices. The test criteria is that the extracted grid should have the
    expected geometry, and that the expected faces and nodes are extracted.

    """

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

        # Pick out cells (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 2),
        # (1, 0, 2)
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


class TestExtractSubgridTags:
    """Test that extract_subgrid transfers the standard tags from the parent grid.

    The three standard face tags are mutually exclusive, and together they cover the
    faces that have a single neighbouring cell (apart from periodic faces, which have
    one neighbour but are not on a boundary). A subgrid has more such faces than its
    parent, namely those that appear where the subgrid has been cut out, and these
    should be tagged as domain boundary faces. The tags the parent has assigned should
    be kept.

    """

    def fractured_grid(self, dim: int) -> pp.Grid:
        """Grid with one immersed fracture, as the highest-dimensional subdomain of a
        mixed-dimensional grid."""
        if dim == 2:
            fracture = np.array([[1.0, 3.0], [2.0, 2.0]])
            num_cells = np.array([4, 4])
            physdims = [4, 4]
        else:
            fracture = np.array(
                [[1.0, 3.0, 3.0, 1.0], [1.0, 1.0, 3.0, 3.0], [2.0, 2.0, 2.0, 2.0]]
            )
            num_cells = np.array([4, 4, 4])
            physdims = [4, 4, 4]
        mdg = pp.meshing.cart_grid([fracture], num_cells, physdims=physdims)
        sd = mdg.subdomains(dim=dim)[0]
        sd.compute_geometry()
        return sd

    @pytest.mark.parametrize("dim", [2, 3])
    def test_tags_of_full_extraction(self, dim: int):
        """Extracting all cells should reproduce the tags of the parent.

        This is the case met by the finite volume discretizations, which always work on
        a grid extracted from the one they are asked to discretize.

        """
        sd = self.fractured_grid(dim)
        sub, sub_f, _ = pp.partition.extract_subgrid(sd, np.arange(sd.num_cells))

        assert sd.tags["fracture_faces"].sum() > 0
        for key in pp.utils.tags.standard_face_tags():
            assert np.array_equal(sd.tags[key][sub_f], sub.tags[key])
        for key in pp.utils.tags.standard_node_tags():
            assert sd.tags[key].sum() == sub.tags[key].sum()
        assert np.array_equal(
            np.sort(sub_f[sub.get_all_boundary_faces()]),
            np.sort(sd.get_all_boundary_faces()),
        )

    def test_tags_of_partial_extraction(self):
        """A subgrid keeps the parent's tags, and tags the new boundary faces."""
        sd = self.fractured_grid(2)
        # The four cells around the fracture.
        cells = np.where(
            np.logical_and(
                np.abs(sd.cell_centers[0] - 2) < 1,
                np.abs(sd.cell_centers[1] - 2) < 1,
            )
        )[0]
        sub, sub_f, _ = pp.partition.extract_subgrid(sd, cells)

        # The fracture faces of the parent are still fracture faces.
        assert np.array_equal(
            sd.tags["fracture_faces"][sub_f], sub.tags["fracture_faces"]
        )
        assert sub.tags["fracture_faces"].sum() == sd.tags["fracture_faces"].sum()
        # The tags are mutually exclusive, ...
        assert not np.any(
            np.logical_and(
                sub.tags["fracture_faces"], sub.tags["domain_boundary_faces"]
            )
        )
        # ... and together they cover the faces with a single neighbouring cell.
        single_neighbour = np.where(np.diff(sub.cell_faces.tocsr().indptr) == 1)[0]
        assert np.array_equal(
            np.sort(sub.get_all_boundary_faces()), np.sort(single_neighbour)
        )
        # The faces cut out of the interior of the parent are new domain boundaries.
        new_boundary = np.setdiff1d(
            sub_f[np.where(sub.tags["domain_boundary_faces"])[0]],
            np.where(sd.tags["domain_boundary_faces"])[0],
        )
        assert new_boundary.size > 0

    def test_tags_of_lower_dimensional_grid(self):
        """A fracture grid has tip faces, which should also be transferred."""
        mdg = pp.meshing.cart_grid(
            [np.array([[1.0, 3.0], [2.0, 2.0]])], np.array([4, 4]), physdims=[4, 4]
        )
        sd = mdg.subdomains(dim=1)[0]
        sd.compute_geometry()
        assert sd.tags["tip_faces"].sum() > 0

        sub, sub_f, _ = pp.partition.extract_subgrid(sd, np.arange(sd.num_cells))
        assert np.array_equal(sd.tags["tip_faces"][sub_f], sub.tags["tip_faces"])
        assert not np.any(sub.tags["domain_boundary_faces"])

    def test_tags_of_periodic_grid(self):
        """Periodic faces have one neighbouring cell, but are not boundary faces."""
        sd = pp.CartGrid([4, 4], physdims=[1, 1])
        sd.compute_geometry()
        west = np.where(sd.face_centers[0] < 1e-10)[0]
        east = np.where(sd.face_centers[0] > 1 - 1e-10)[0]
        sd.set_periodic_map(np.vstack((west, east)))

        sub, sub_f, _ = pp.partition.extract_subgrid(sd, np.arange(sd.num_cells))
        assert np.array_equal(
            sd.tags["domain_boundary_faces"][sub_f],
            sub.tags["domain_boundary_faces"],
        )
        assert not np.any(sub.tags["domain_boundary_faces"][np.hstack((west, east))])


class TestComputationOfOverlap:
    """Test that the computation of the overlap of a partition is correct (function
    partition.overlap()).

    From a grid, a partition with overlap is created based on a set of cell indices. The
    overlap is computed, and compared to a reference value.

    """

    @pytest.mark.parametrize(
        "overlap, known_indices",
        [
            (1, np.array([0, 1, 2, 5, 6, 7, 10, 11, 12])),
            (2, np.array([0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18])),
        ],
    )
    def test_overlap(self, overlap, known_indices):
        g = pp.CartGrid([5, 5])
        ci = np.array([0, 1, 5, 6])
        assert np.array_equal(pp.partition.overlap(g, ci, overlap), known_indices)


class TestConnectivityChecker:
    """Test the connectivity checker (function partition.grid_is_connected())."""

    def setup(self):
        # Generate a grid and a partition
        g = pp.CartGrid([4, 4])
        # The structured partitioning scheme will let the assign the lower left 2x2
        # block (cells with indices 0, 1, 4, 5) to block 0, lower right to block 1 etc.
        p = pp.partition.partition_structured(g, coarse_dims=np.array([2, 2]))
        return g, p

    def test_connected(self):
        # Generate a connected partition
        g, p = self.setup()
        # Pick out a subset of the cells, based on the partition
        p_sub = np.r_[np.where(p == 0)[0], np.where(p == 1)[0]]
        # Check if the grid is connected - in this case it should be
        is_connected, components = pp.partition.grid_is_connected(g, p_sub)
        assert is_connected
        # Check that all cells are identified as belonging to the same connected
        # component.
        assert np.array_equal(np.sort(components[0]), np.arange(8))

    def test_not_connected(self):
        # Pick out the lower left and upper right squares
        g, p = self.setup()

        p_sub = np.r_[np.where(p == 0)[0], np.where(p == 3)[0]]
        is_connected, components = pp.partition.grid_is_connected(g, p_sub)
        assert not is_connected
        # There should be two components
        assert len(components) == 2
        # Each coarse cell should be in one component
        assert np.array_equal(np.sort(p_sub[components[0]]), np.array([0, 1, 4, 5]))
        assert np.array_equal(np.sort(p_sub[components[1]]), np.array([10, 11, 14, 15]))

    def test_subgrid_connected(self):
        # A single subgrid is connected
        g, p = self.setup()

        sub_g, _, _ = pp.partition.extract_subgrid(g, np.where(p == 0)[0])
        is_connected, components = pp.partition.grid_is_connected(sub_g)
        assert is_connected
        assert np.array_equal(np.sort(components[0]), np.arange(sub_g.num_cells))

    def test_subgrid_not_connected(self):
        # Generate a subgrid with two disconnected regions.
        g, p = self.setup()

        p_sub = np.r_[np.where(p == 0)[0], np.where(p == 3)[0]]
        sub_g, _, _ = pp.partition.extract_subgrid(g, p_sub)
        is_connected, components = pp.partition.grid_is_connected(sub_g)
        assert not is_connected
        assert np.array_equal(np.sort(p_sub[components[0]]), np.array([0, 1, 4, 5]))
        assert np.array_equal(np.sort(p_sub[components[1]]), np.array([10, 11, 14, 15]))


class TestCoordinatePartitioner:
    """Test partitioning based on coordinates, function
    partition.partition_coordinates().

    """

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
    """Test function to partition a grid, partition.partition_grid().

    From a given grid and partition, generate subgrids and check that the grids are
    correct, as are the face and node maps.
    """

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
        # Partition all cells into single cell grids
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
