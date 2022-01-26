"""
Tests for the standard grids of the grid_bucket_2d module.

Some of these tests are sensitive to meshing or node ordering. If this turns out to cause problems, we deactivate the corresponding asserts.
"""
import unittest

import numpy as np

import porepy as pp


class TestGridBuckets(unittest.TestCase):
    """Various tests for the utility methods to generate standard geometries.

    Note that the tests in part are based on hard-coded 'known' information on the
    generated grids, such as the number of cells along a fracture. This type of
    information should be quite fixed, but may change under updates to meshing software
    etc. Thus, it may be permissible to update the tests herein, although this should
    not be done unless it is absolutely clear that external factors caused the change
    in behavior.

    """

    def check_matrix(self, grid_type):
        gb = self.gb
        if grid_type.lower() == "simplex":
            self.assertTrue(isinstance(gb.grids_of_dimension(2)[0], pp.TriangleGrid))
        elif grid_type.lower() == "cartesian":
            self.assertTrue(isinstance(gb.grids_of_dimension(2)[0], pp.CartGrid))
        else:
            raise Exception("Unknown grid type specified")

    def check_fractures(self, n_fracs, n_frac_cells):
        self.assertTrue(len(self.gb.grids_of_dimension(1)) == n_fracs)
        for i, g in enumerate(self.gb.grids_of_dimension(1)):
            self.assertEqual(n_frac_cells[i], g.num_cells)

    def check_fracture_coordinates(self, cell_centers, face_centers):
        """
        Coords is a list, whereof each element is the coordinates of a fracture.
        This check may be sensitive to gmsh options etc. If so, it should probably
        be deactivated for the test case for which it fails.
        """
        for i, g in enumerate(self.gb.grids_of_dimension(1)):
            self.assertTrue(np.all(np.isclose(g.cell_centers, cell_centers[i])))
            self.assertTrue(np.all(np.isclose(g.face_centers, face_centers[i])))

    def check_intersections(self, n_intersections):
        self.assertTrue(len(self.gb.grids_of_dimension(0)) == n_intersections)

    def check_domain(self, x_length, y_length):
        d = self.domain
        self.assertAlmostEqual(x_length, d["xmax"] - d["xmin"])
        self.assertAlmostEqual(y_length, d["ymax"] - d["ymin"])

    def test_single_horizontal_default_values(self):
        """
        Test the single horizontal gb generator for the default values: simplex grid,
        mesh_size_fracs of 0.2 and a fracture crossing the domain at y=0.5.
        """

        self.gb, _ = pp.grid_buckets_2d.single_horizontal()
        self.check_matrix("simplex")
        self.check_fractures(1, [5])
        cc = np.vstack(
            (np.vstack((np.arange(0.1, 1.1, 0.2), 0.5 * np.ones(5))), np.zeros(5))
        )
        fc = np.vstack(
            (np.vstack((np.arange(0, 1.2, 0.2), 0.5 * np.ones(6))), np.zeros(6))
        )
        self.check_fracture_coordinates([cc], [fc])

    def test_single_horizontal_custom_values(self):
        """
        Test the single horizontal gb generator for a cartesian grid, 6 by 2 cells and
        a fracture extending from (1/6, 1/2) to (1, 1/2).
        """
        self.gb, _ = pp.grid_buckets_2d.single_horizontal(
            [6, 2], [1 / 6, 1], simplex=False
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

    def test_two_intersecting_default_values(self):
        """
        Test the two intersecting fractures gb generator for the default values: simplex
        grid, mesh_size_fracs of 0.2 and two fractures intersecting at (0.5, 0.5).
        """
        self.gb, _ = pp.grid_buckets_2d.two_intersecting()
        self.check_matrix("simplex")
        self.check_fractures(2, [6, 6])
        cc = np.arange(1 / 12, 1, 1 / 6)
        cc0 = np.vstack((np.vstack((cc, 0.5 * np.ones(6))), np.zeros(6)))
        # Add the split middle face at the end
        fc = np.hstack((np.arange(0, 1.1, 1 / 6), 0.5))
        fc0 = np.vstack((np.vstack((fc, 0.5 * np.ones(8))), np.zeros(8)))
        cc1 = np.vstack((np.vstack((0.5 * np.ones(6), cc)), np.zeros(6)))
        fc1 = np.vstack((np.vstack((0.5 * np.ones(8), fc)), np.zeros(8)))
        self.check_fracture_coordinates([cc0, cc1], [fc0, fc1])

    def test_two_intersecting_custom_values(self):
        """
        Test the two intersecting fractures gb generator for a cartesian grid, 4 by 4
        cells and the vertical fracture extending from y=1/4 to y=3/4.
        """
        self.gb, _ = pp.grid_buckets_2d.two_intersecting(
            [4, 4], y_endpoints=[1 / 4, 3 / 4], simplex=False
        )
        self.check_matrix("cartesian")
        self.check_fractures(2, [4, 2])
        cc0 = np.vstack(
            (np.vstack((np.arange(7 / 8, 0, -1 / 4), 0.5 * np.ones(4))), np.zeros(4))
        )
        # Add the split middle face at the end
        x = np.hstack((np.arange(1, -1 / 4, -1 / 4), 0.5))
        fc0 = np.vstack((np.vstack((x, 0.5 * np.ones(6))), np.zeros(6)))
        cc1 = np.vstack(
            (np.vstack((0.5 * np.ones(2), np.array([5 / 8, 3 / 8]))), np.zeros(2))
        )
        y = np.hstack((np.arange(3 / 4, 0, -1 / 4), 0.5))
        fc1 = np.vstack((np.vstack((0.5 * np.ones(4), y)), np.zeros(4)))
        self.check_fracture_coordinates([cc0, cc1], [fc0, fc1])

    def test_benchmark_regular(self):
        """
        Test the gb generator for the regular case of the benchmark study. No coarsening
        is applied.
        """
        self.gb, self.domain = pp.grid_buckets_2d.benchmark_regular(
            {"mesh_size_frac": 1 / 2}
        )
        self.check_matrix("simplex")
        self.check_fractures(6, [7, 7, 4, 4, 2, 2])
        self.check_intersections(9)
        self.check_domain(1, 1)

    def test_seven_fractures_one_L_intersection(self):
        """
        Test the gb generator for the regular case of the benchmark study. No coarsening
        is applied.
        """
        self.gb, self.domain = pp.grid_buckets_2d.seven_fractures_one_L_intersection(
            {"mesh_size_frac": 1 / 5}
        )
        self.check_matrix("simplex")
        self.check_fractures(7, [2, 2, 8, 3, 8, 5, 4])
        self.check_intersections(1)
        self.check_domain(2, 1)


if __name__ == "__main__":
    unittest.main()
