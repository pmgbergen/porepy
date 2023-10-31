"""
Some of these tests are sensitive to meshing or node ordering. If this turns out to
cause problems, we deactivate the corresponding asserts.
"""

import numpy as np

import porepy as pp


class TestMixedDimensionalGrids:
    """Various tests for the utility methods to generate standard geometries.

    Note that the tests in part are based on hard-coded 'known' information on the
    generated grids, such as the number of cells along a fracture. This type of
    information should be quite fixed, but may change under updates to meshing software
    etc. Thus, it may be permissible to update the tests herein, although this should
    not be done unless it is absolutely clear that external factors caused the change
    in behavior.

    """

    def check_matrix(self, grid_type):
        mdg = self.mdg
        if grid_type.lower() == "simplex":
            assert isinstance(mdg.subdomains(dim=2)[0], pp.TriangleGrid)
        elif grid_type.lower() == "cartesian":
            assert isinstance(mdg.subdomains(dim=2)[0], pp.CartGrid)
        else:
            raise Exception("Unknown grid type specified")

    def check_fractures(self, n_fracs, n_frac_cells):
        assert len(self.mdg.subdomains(dim=1)) == n_fracs
        for i, g in enumerate(self.mdg.subdomains(dim=1)):
            assert n_frac_cells[i] == g.num_cells

    def check_fracture_coordinates(self, cell_centers, face_centers):
        """
        Coords is a list, whereof each element is the coordinates of a fracture.
        This check may be sensitive to gmsh options etc. If so, it should probably
        be deactivated for the test case for which it fails.
        """
        for i, g in enumerate(self.mdg.subdomains(dim=1)):
            assert np.all(np.isclose(g.cell_centers, cell_centers[i]))
            assert np.all(np.isclose(g.face_centers, face_centers[i]))

    def check_intersections(self, n_intersections):
        assert len(self.mdg.subdomains(dim=0)) == n_intersections

    def check_domain(self, x_length, y_length):
        bbox = self.domain.bounding_box
        assert np.isclose(x_length, bbox["xmax"] - bbox["xmin"])
        assert np.isclose(y_length, bbox["ymax"] - bbox["ymin"])

    def test_single_horizontal_2d_custom_values(self):
        """
        Test the single horizontal gb generator for a cartesian grid, 6 by 2 cells and
        a fracture extending from (1/6, 1/2) to (1, 1/2).
        """
        meshing_arguments = {"cell_size_x": 1 / 6, "cell_size_y": 1 / 2}
        (
            self.mdg,
            _,
        ) = pp.mdg_library.square_with_orthogonal_fractures(
            "cartesian",
            meshing_arguments,
            fracture_indices=[1],
            fracture_endpoints=[np.array([1 / 6, 1])],
        )
        self.check_matrix("cartesian")
        self.check_fractures(1, [5])
        cc = np.vstack(
            (
                np.vstack((np.arange(11 / 12, 1 / 12, -1 / 6), 0.5 * np.ones(5))),
                np.zeros(5),
            )
        )
        fc = np.vstack(
            (np.vstack((np.arange(1, 1 / 6, -1 / 6), 0.5 * np.ones(6))), np.zeros(6))
        )
        self.check_fracture_coordinates([cc], [fc])

    def test_two_intersecting_custom_values(self):
        """
        Test the two intersecting fractures gb generator for a cartesian grid, 4 by 4
        cells and the vertical fracture extending from y=1/4 to y=3/4.
        """
        meshing_arguments = {"cell_size_x": 1 / 4, "cell_size_y": 1 / 4}
        fracture_endpoints = [np.array([1 / 4, 3 / 4]), np.array([0, 1])]
        self.mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
            "cartesian",
            meshing_arguments,
            fracture_indices=[0, 1],
            fracture_endpoints=fracture_endpoints,
        )
        self.check_matrix("cartesian")
        self.check_fractures(2, [2, 4])
        # Cell and face centers of first fracture (constant x coordinate)
        cc0 = np.vstack(
            (np.vstack((0.5 * np.ones(2), np.array([5 / 8, 3 / 8]))), np.zeros(2))
        )
        y = np.hstack((np.arange(3 / 4, 0, -1 / 4), 0.5))

        fc0 = np.vstack((np.vstack((0.5 * np.ones(4), y)), np.zeros(4)))
        cc1 = np.vstack(
            (np.vstack((np.arange(7 / 8, 0, -1 / 4), 0.5 * np.ones(4))), np.zeros(4))
        )
        # Add the split middle face at the end
        x = np.hstack((np.arange(1, -1 / 4, -1 / 4), 0.5))
        fc1 = np.vstack((np.vstack((x, 0.5 * np.ones(6))), np.zeros(6)))

        self.check_fracture_coordinates([cc0, cc1], [fc0, fc1])

    def test_benchmark_regular(self):
        """Test the mdg generator for the regular case of the benchmark study.

        Coarse cell size at fractures to obtain stable number of cells. No coarsening is
        applied.
        """

        self.mdg, network = pp.mdg_library.benchmark_regular_2d(
            {"cell_size": 1.0, "cell_size_fracture": 0.5}
        )
        self.domain = network.domain
        self.check_matrix("simplex")
        self.check_fractures(6, [7, 7, 4, 4, 2, 2])
        self.check_intersections(9)
        self.check_domain(1, 1)

    def test_seven_fractures_one_L_intersection(self):
        """Test the mdg generator for the regular case of the benchmark study."""
        (
            self.mdg,
            network,
        ) = pp.mdg_library.seven_fractures_one_L_intersection({"cell_size": 0.2})
        self.domain = network.domain
        self.check_matrix("simplex")
        self.check_fractures(7, [2, 2, 8, 3, 8, 5, 4])
        self.check_intersections(1)
        self.check_domain(2, 1)
