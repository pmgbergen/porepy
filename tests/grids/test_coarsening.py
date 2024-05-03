import inspect
import sys
import pytest

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.grids import coarsening as co
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data


class TestPartitioning:
    def reference(self):
        caller_method = inspect.stack()[1][3]
        return pp.test_utils.reference_dense_arrays.test_coarsening[caller_method]

    # Skipped: The test fails after update to scipy 1.13 due to changes in the sparse
    # matrix format. Since the underlying code is marked for deprecation, the test will
    # not be updated.
    @pytest.mark.xfail
    def test_coarse_grid_2d(self):
        g = pp.CartGrid([3, 2])
        g.compute_geometry()
        co.generate_coarse_grid(g, [5, 2, 2, 5, 2, 2])

        assert g.num_cells == 2
        assert g.num_faces == 12
        assert g.num_nodes == 11

        pt = np.tile(np.array([2, 1, 0]), (g.nodes.shape[1], 1)).T
        find = np.isclose(pt, g.nodes).all(axis=0)
        assert find.any() == False

        faces_cell0, _, orient_cell0 = sparse_array_to_row_col_data(g.cell_faces[:, 0])
        assert np.array_equal(faces_cell0, [1, 2, 4, 5, 7, 8, 10, 11])
        assert np.array_equal(orient_cell0, [-1, 1, -1, 1, -1, -1, 1, 1])

        faces_cell1, _, orient_cell1 = sparse_array_to_row_col_data(g.cell_faces[:, 1])
        assert np.array_equal(faces_cell1, [0, 1, 3, 4, 6, 9])
        assert np.array_equal(orient_cell1, [-1, 1, -1, 1, -1, 1])

        known = np.array(
            [
                [0, 4],
                [1, 5],
                [3, 6],
                [4, 7],
                [5, 8],
                [6, 10],
                [1, 0],
                [2, 1],
                [3, 2],
                [8, 7],
                [9, 8],
                [10, 9],
            ]
        )

        for f in np.arange(g.num_faces):
            assert np.array_equal(
                sparse_array_to_row_col_data(g.face_nodes[:, f])[0], known[f, :]
            )

    # Skipped: The test fails after update to scipy 1.13 due to changes in the sparse
    # matrix format. Since the underlying code is marked for deprecation, the test will
    # not be updated.
    @pytest.mark.xfail
    def test_coarse_grid_3d(self):
        g = pp.CartGrid([2, 2, 2])
        g.compute_geometry()
        co.generate_coarse_grid(g, [0, 0, 0, 0, 1, 1, 2, 2])

        assert g.num_cells == 3
        assert g.num_faces == 30
        assert g.num_nodes == 27

        faces_cell0, _, orient_cell0 = sparse_array_to_row_col_data(g.cell_faces[:, 0])
        known = [0, 1, 2, 3, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25]
        assert np.array_equal(faces_cell0, known)
        known = [-1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1]
        assert np.array_equal(orient_cell0, known)

        faces_cell1, _, orient_cell1 = sparse_array_to_row_col_data(g.cell_faces[:, 1])
        known = [4, 5, 12, 13, 14, 15, 22, 23, 26, 27]
        assert np.array_equal(faces_cell1, known)
        known = [-1, 1, -1, -1, 1, 1, -1, -1, 1, 1]
        assert np.array_equal(orient_cell1, known)

        faces_cell2, _, orient_cell2 = sparse_array_to_row_col_data(g.cell_faces[:, 2])
        known = [6, 7, 14, 15, 16, 17, 24, 25, 28, 29]
        assert np.array_equal(faces_cell2, known)
        known = [-1, 1, -1, -1, 1, 1, -1, -1, 1, 1]
        assert np.array_equal(orient_cell2, known)

        reference = self.reference()["face_nodes"]
        for f in np.arange(g.num_faces):
            assert np.array_equal(
                sparse_array_to_row_col_data(g.face_nodes[:, f])[0], reference[f, :]
            )

    def test_coarse_grid_2d_1d(self):
        part = np.array([0, 0, 1, 1, 2, 0, 3, 1])
        f = np.array([[2, 2], [0, 2]])

        mdg = pp.meshing.cart_grid([f], [4, 2])
        mdg.compute_geometry()
        co.generate_coarse_grid(mdg, (None, part))

        # Test
        known = np.array([1, 5, 18, 19])

        for intf in mdg.interfaces():
            faces = sparse_array_to_row_col_data(intf.primary_to_mortar_int())[1]
            assert np.array_equal(faces, known)

    def test_coarse_grid_2d_1d_cross(self):
        # NOTE: Since for python 2.7 and 3.5 the meshes in gridbucket may have
        # non-fixed order, we need to exclude this test.
        if sys.version_info >= (3, 6):
            part = np.zeros(36)
            part[[0, 1, 2, 6, 7]] = 1
            part[[8, 14, 13]] = 2
            part[[12, 18, 19]] = 3
            part[[24, 30, 31, 32]] = 4
            part[[21, 22, 23, 27, 28, 29, 33, 34, 35]] = 5
            part[[9]] = 6
            part[[15, 16, 17]] = 7
            part[[9, 10]] = 8
            part[[20, 26, 25]] = 9
            part[[3, 4, 5, 11]] = 10
            f1 = np.array([[3.0, 3.0], [1.0, 5.0]])
            f2 = np.array([[1.0, 5.0], [3.0, 3.0]])

            mdg = pp.meshing.cart_grid([f1, f2], [6, 6])
            mdg.compute_geometry()

            cell_centers_1 = np.array(
                [
                    [3.00000000e00, 3.00000000e00, 3.00000000e00, 3.00000000e00],
                    [4.50000000e00, 3.50000000e00, 2.50000000e00, 1.50000000e00],
                    [-1.66533454e-16, -5.55111512e-17, 5.55111512e-17, 1.66533454e-16],
                ]
            )
            cell_centers_2 = np.array(
                [
                    [4.50000000e00, 3.50000000e00, 2.50000000e00, 1.50000000e00],
                    [3.00000000e00, 3.00000000e00, 3.00000000e00, 3.00000000e00],
                    [-1.66533454e-16, -5.55111512e-17, 5.55111512e-17, 1.66533454e-16],
                ]
            )

            co.generate_coarse_grid(mdg, (None, part))

            # Test
            for intf in mdg.interfaces():
                faces = sparse_array_to_row_col_data(intf.primary_to_mortar_int())[1]

                sd_primary, sd_secondary = mdg.interface_to_subdomain_pair(intf)

                if sd_secondary.dim == 0 and sd_primary.dim == 1:
                    known = [2, 5]

                elif sd_secondary.dim == 1 and sd_primary.dim == 2:
                    if np.allclose(sd_secondary.cell_centers, cell_centers_1):
                        known = [5, 10, 14, 18, 52, 53, 54, 55]
                    elif np.allclose(sd_secondary.cell_centers, cell_centers_2):
                        known = [37, 38, 39, 40, 56, 57, 58, 59]
                    else:
                        raise ValueError("Grid not found")

                assert np.array_equal(faces, known)

    def test_coarse_grid_3d_2d(self):
        f = np.array([[2.0, 2.0, 2.0, 2.0], [0.0, 2.0, 2.0, 0.0], [0.0, 0.0, 2.0, 2.0]])
        mdg = pp.meshing.cart_grid([f], [4, 2, 2])
        mdg.compute_geometry()

        g = mdg.subdomains(dim=mdg.dim_max())[0]
        part = np.zeros(g.num_cells)
        part[g.cell_centers[0, :] < 2.0] = 1
        co.generate_coarse_grid(mdg, (None, part))
        # Test
        # Be carefull! If the indexing of any grids (including mg) change the hard-coded
        # indexes may be wrong
        known_indices = np.array([1, 3, 0, 2, 5, 7, 4, 6])
        known = np.array([1, 4, 7, 10, 44, 45, 46, 47])

        for intf in mdg.interfaces():
            indices, faces, _ = sparse_array_to_row_col_data(
                intf.primary_to_mortar_int()
            )

            assert np.array_equal(indices, known_indices)
            assert np.array_equal(faces, known)

    def test_coarse_grid_3d_2d_cross(self):
        # NOTE: Since for python 2.7 and 3.5 the meshes in gridbucket may have
        # non-fixed order, we need to exclude this test.
        if sys.version_info >= (3, 6):
            f1 = np.array(
                [[3.0, 3.0, 3.0, 3.0], [1.0, 5.0, 5.0, 1.0], [1.0, 1.0, 5.0, 5.0]]
            )
            f2 = np.array(
                [[1.0, 5.0, 5.0, 1.0], [1.0, 1.0, 5.0, 5.0], [3.0, 3.0, 3.0, 3.0]]
            )
            mdg = pp.meshing.cart_grid([f1, f2], [6, 6, 6])
            mdg.compute_geometry()

            g = mdg.subdomains(dim=mdg.dim_max())[0]
            part = np.zeros(g.num_cells)
            p1, p2 = g.cell_centers[0, :] < 3.0, g.cell_centers[2, :] < 3.0
            part[np.logical_and(p1, p2)] = 1
            part[np.logical_and(p1, np.logical_not(p2))] = 2
            part[np.logical_and(np.logical_not(p1), p2)] = 3
            part[np.logical_and(np.logical_not(p1), np.logical_not(p2))] = 4

            co.generate_coarse_grid(mdg, (None, part))

            cell_centers_1 = self.reference()["cell_centers_1"]
            cell_centers_2 = self.reference()["cell_centers_2"]
            cell_centers_3 = self.reference()["cell_centers_3"]

            # Test
            for intf in mdg.interfaces():
                indices, faces, _ = sparse_array_to_row_col_data(
                    intf.primary_to_mortar_int()
                )
                sd_primary, sd_secondary = mdg.interface_to_subdomain_pair(intf)

                if sd_primary.dim == 2 and sd_secondary.dim == 1:
                    reference_indices = [0, 1, 2, 3, 4, 5, 6, 7]
                    reference_faces = [17, 12, 7, 2, 43, 42, 41, 40]

                if sd_primary.dim == 3 and sd_secondary.dim == 2:
                    if np.allclose(sd_secondary.cell_centers, cell_centers_1):
                        reference_faces = self.reference()["faces_1"]
                        reference_indices = self.reference()["indices_1"]
                    elif np.allclose(sd_secondary.cell_centers, cell_centers_2):
                        reference_faces = self.reference()["faces_2"]
                        reference_indices = self.reference()["indices_2"]
                    elif np.allclose(sd_secondary.cell_centers, cell_centers_3):
                        reference_faces = self.reference()["faces_3"]
                        reference_indices = self.reference()["indices_3"]
                    else:
                        raise ValueError("Grid not found")

                assert np.array_equal(indices, np.array(reference_indices))
                assert np.array_equal(faces, np.array(reference_faces))

    def test_create_partition_2d_cart(self):
        g = pp.CartGrid([5, 5])
        g.compute_geometry()
        part = co.create_partition(co._tpfa_matrix(g), g)
        known = np.array(
            [0, 0, 0, 1, 1, 0, 0, 2, 1, 1, 3, 2, 2, 2, 1, 3, 3, 2, 4, 4, 3, 3, 4, 4, 4]
        )
        part = next(iter(part.values()))[1]
        assert np.array_equal(part, known)

    def test_create_partition_2d_tri(self):
        g = pp.StructuredTriangleGrid([3, 2])
        g.compute_geometry()
        part = co.create_partition(co._tpfa_matrix(g), g)
        known = np.array([1, 1, 1, 0, 0, 1, 0, 2, 2, 0, 2, 2])
        known_map = np.array([4, 3, 7, 5, 11, 8, 1, 2, 10, 6, 12, 9]) - 1
        part = next(iter(part.values()))[1]
        assert np.array_equal(part, known[known_map])

    def test_create_partition_2d_cart_cdepth4(self):
        g = pp.CartGrid([10, 10])
        g.compute_geometry()
        part = co.create_partition(co._tpfa_matrix(g), g, cdepth=4)

        part = next(iter(part.values()))[1]
        assert np.array_equal(part, self.reference()["partition"])

    def test_create_partition_3d_cart(self):
        g = pp.CartGrid([4, 4, 4])
        g.compute_geometry()
        part = co.create_partition(co._tpfa_matrix(g), g)
        part = next(iter(part.values()))[1]
        assert np.array_equal(part, self.reference()["partition"])

    def test_create_partition_2d_1d_test0(self):
        mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
            "cartesian",
            meshing_args={"cell_size": 1 / 2},
            fracture_indices=[1],
        )

        part = co.create_partition(co._tpfa_matrix(mdg), mdg)
        co.generate_coarse_grid(mdg, part)

        # Test
        known_indices = np.array([1, 0, 3, 2])
        known = np.array([6, 7, 10, 11])

        for intf in mdg.interfaces():
            indices, faces, _ = sparse_array_to_row_col_data(
                intf.primary_to_mortar_int()
            )
            assert np.array_equal(faces, known)
            assert np.array_equal(indices, known_indices)

    def test_create_partition_2d_1d_test1(self):
        mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
            "cartesian",
            meshing_args={"cell_size": 1 / 2},
            fracture_indices=[1],
            fracture_endpoints=[np.array([0.5, 0.0])],
        )
        part = co.create_partition(co._tpfa_matrix(mdg), mdg)
        co.generate_coarse_grid(mdg, part)

        # Test
        known_indices = np.array([0, 1])
        known = np.array([6, 9])

        for intf in mdg.interfaces():
            indices, faces, _ = sparse_array_to_row_col_data(
                intf.primary_to_mortar_int()
            )
            assert np.array_equal(faces, known)
            assert np.array_equal(indices, known_indices)

    def test_create_partition_2d_1d_test2(self):
        mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
            "cartesian",
            meshing_args={"cell_size": 1 / 2},
            fracture_indices=[1],
            fracture_endpoints=[np.array([0.0, 0.5])],
        )

        seeds = co.generate_seeds(mdg)
        known_seeds = np.array([0, 2])
        assert np.array_equal(seeds, known_seeds)

        part = co.create_partition(co._tpfa_matrix(mdg), mdg, seeds=seeds)
        co.generate_coarse_grid(mdg, part)

        # Test
        known_indices = np.array([0, 1])
        known = np.array([6, 10])

        for intf in mdg.interfaces():
            indices, faces, _ = sparse_array_to_row_col_data(
                intf.primary_to_mortar_int()
            )
            assert np.array_equal(faces, known)
            assert np.array_equal(indices, known_indices)

    def test_create_partition_2d_1d_test3(self):
        mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
            "cartesian",
            meshing_args={"cell_size": 1 / 2},
            fracture_indices=[1],
            fracture_endpoints=[np.array([0.5, 1.0])],
        )

        part = co.create_partition(co._tpfa_matrix(mdg), mdg)
        co.generate_coarse_grid(mdg, part)

        # Test
        known_indices = np.array([0, 1])
        known = np.array([6, 9])

        for intf in mdg.interfaces():
            indices, faces, _ = sparse_array_to_row_col_data(
                intf.primary_to_mortar_int()
            )
            assert np.array_equal(faces, known)
            assert np.array_equal(indices, known_indices)

    def test_create_partition_2d_1d_test4(self):
        mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
            "cartesian",
            meshing_args={"cell_size": 1 / 2},
            fracture_indices=[1],
            fracture_endpoints=[np.array([0.5, 1.0])],
        )

        seeds = co.generate_seeds(mdg)
        known_seeds = np.array([1, 3])
        assert np.array_equal(seeds, known_seeds)

        part = co.create_partition(co._tpfa_matrix(mdg), mdg, seeds=seeds)
        co.generate_coarse_grid(mdg, part)

        # Test
        known_indices = np.array([0, 1])
        known = np.array([7, 10])

        for intf in mdg.interfaces():
            indices, faces, _ = sparse_array_to_row_col_data(
                intf.primary_to_mortar_int()
            )
            assert np.array_equal(faces, known)
            assert np.array_equal(indices, known_indices)

    def test_create_partition_2d_1d_cross_test5(self):
        # NOTE: Since for python 2.7 and 3.5 the meshes in gridbucket may have
        # non-fixed order, we need to exclude this test.
        if sys.version_info >= (3, 6):
            f1 = np.array([[3.0, 3.0], [1.0, 5.0]])
            f2 = np.array([[1.0, 5.0], [3.0, 3.0]])
            mdg = pp.meshing.cart_grid([f1, f2], [6, 6])
            mdg.compute_geometry()

            part = co.create_partition(co._tpfa_matrix(mdg), mdg, cdepth=3)
            co.generate_coarse_grid(mdg, part)

            cell_centers_1 = np.array(
                [
                    [3.00000000e00, 3.00000000e00, 3.00000000e00, 3.00000000e00],
                    [4.50000000e00, 3.50000000e00, 2.50000000e00, 1.50000000e00],
                    [-1.66533454e-16, -5.55111512e-17, 5.55111512e-17, 1.66533454e-16],
                ]
            )
            cell_centers_2 = np.array(
                [
                    [4.50000000e00, 3.50000000e00, 2.50000000e00, 1.50000000e00],
                    [3.00000000e00, 3.00000000e00, 3.00000000e00, 3.00000000e00],
                    [-1.66533454e-16, -5.55111512e-17, 5.55111512e-17, 1.66533454e-16],
                ]
            )

            # Test
            for intf in mdg.interfaces():
                indices, faces, _ = sparse_array_to_row_col_data(
                    intf.primary_to_mortar_int()
                )
                sd_primary, sd_secondary = mdg.interface_to_subdomain_pair(intf)

                if sd_primary.dim == 1 and sd_secondary.dim == 0:
                    known = [2, 5]
                    known_indices = [0, 1]

                elif sd_primary.dim == 2 and sd_secondary.dim == 1:
                    g = sd_secondary

                    if np.allclose(g.cell_centers, cell_centers_1):
                        known = [4, 9, 12, 16, 44, 45, 46, 47]
                        known_indices = [3, 2, 1, 0, 7, 6, 5, 4]
                    elif np.allclose(g.cell_centers, cell_centers_2):
                        known = [31, 32, 33, 34, 48, 49, 50, 51]
                        known_indices = [3, 2, 1, 0, 7, 6, 5, 4]
                    else:
                        raise ValueError("Grid not found")

                assert np.array_equal(faces, np.array(known))
                assert np.array_equal(indices, np.array(known_indices))

    def test_create_partition_2d_1d_cross_test6(self):
        # NOTE: Since for python 2.7 and 3.5 the meshes in gridbucket may have
        # non-fixed order, we need to exclude this test.
        if sys.version_info >= (3, 6):
            f1 = np.array([[3.0, 3.0], [1.0, 5.0]])
            f2 = np.array([[1.0, 5.0], [3.0, 3.0]])
            mdg = pp.meshing.cart_grid([f1, f2], [6, 6])
            mdg.compute_geometry()

            seeds = co.generate_seeds(mdg)
            known_seeds = np.array([8, 9, 26, 27, 13, 16, 19, 22])
            assert np.array_equal(np.sort(seeds), np.sort(known_seeds))

            part = co.create_partition(co._tpfa_matrix(mdg), mdg, cdepth=3, seeds=seeds)
            co.generate_coarse_grid(mdg, part)

            cell_centers_1 = np.array(
                [
                    [3.00000000e00, 3.00000000e00, 3.00000000e00, 3.00000000e00],
                    [4.50000000e00, 3.50000000e00, 2.50000000e00, 1.50000000e00],
                    [-1.66533454e-16, -5.55111512e-17, 5.55111512e-17, 1.66533454e-16],
                ]
            )
            cell_centers_2 = np.array(
                [
                    [4.50000000e00, 3.50000000e00, 2.50000000e00, 1.50000000e00],
                    [3.00000000e00, 3.00000000e00, 3.00000000e00, 3.00000000e00],
                    [-1.66533454e-16, -5.55111512e-17, 5.55111512e-17, 1.66533454e-16],
                ]
            )

            # Test
            for intf in mdg.interfaces():
                sd_primary, sd_secondary = mdg.interface_to_subdomain_pair(intf)

                indices, faces, _ = sparse_array_to_row_col_data(
                    intf.primary_to_mortar_int()
                )

                if sd_secondary.dim == 0 and sd_primary.dim == 1:
                    known = [2, 5]
                    known_indices = [0, 1]

                if sd_secondary.dim == 1 and sd_primary.dim == 2:
                    if np.allclose(sd_secondary.cell_centers, cell_centers_1):
                        known = [5, 10, 14, 18, 52, 53, 54, 55]
                        known_indices = [3, 2, 1, 0, 7, 6, 5, 4]
                    elif np.allclose(sd_secondary.cell_centers, cell_centers_2):
                        known = [37, 38, 39, 40, 56, 57, 58, 59]
                        known_indices = [3, 2, 1, 0, 7, 6, 5, 4]
                    else:
                        raise ValueError("Grid not found")

                assert np.array_equal(faces, np.array(known))
                assert np.array_equal(indices, np.array(known_indices))
