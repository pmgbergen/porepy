"""
Some of these tests are sensitive to meshing or node ordering. If this turns out to
cause problems, we deactivate the corresponding asserts.
"""

import pytest
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
        """Check that the number of intersections is as expected."""
        assert len(self.mdg.subdomains(dim=0)) == n_intersections

    def check_domain(self, x_length, y_length, z_length=None):
        bbox = self.domain.bounding_box
        assert np.isclose(x_length, bbox["xmax"] - bbox["xmin"])
        assert np.isclose(y_length, bbox["ymax"] - bbox["ymin"])
        if z_length is not None:
            assert np.isclose(z_length, bbox["zmax"] - bbox["zmin"])

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

    @pytest.mark.parametrize("mesh_type", ["simplex", "cartesian"])
    @pytest.mark.parametrize("num_fractures", [0, 1, 2])
    def test_two_intersecting_nonmatching(self, mesh_type, num_fractures):
        """Test meshing of a 2d square domain to generate a non-matching grid.

        This is not a very powerful test, but it does verify that the generated grids
        have  different numbers of cells in the fractures and along the mortars, and
        that the mortar grids have a different number of grids than the matrix faces
        tagged as on a fracture.

        Parameters:
            mesh_type: The type of mesh to generate, either 'simplex' or 'cartesian'.
            num_fractures: The number of fractures to include in the domain.

        """
        meshing_arguments = {"cell_size": 1 / 4}

        # Define the fracture endpoints and indices according to the input.
        fracture_endpoints = []
        fracture_indices = []
        if num_fractures > 0:
            fracture_endpoints.append(np.array([1 / 4, 3 / 4]))
            fracture_indices.append(0)
        if num_fractures > 1:
            fracture_endpoints.append(np.array([0, 1]))
            fracture_indices.append(1)

        # Generate
        self.mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
            mesh_type,
            meshing_arguments,
            fracture_indices=fracture_indices,
            fracture_endpoints=fracture_endpoints,
            non_matching=True,
            **{"interface_refinement_ratio": 2, "fracture_refinement_ratios": 3},
        )
        # Number of faces in the 2d grid that are tagged as fracture faces.
        num_fracture_faces_from_matrix = (
            self.mdg.subdomains(dim=2)[0].tags["fracture_faces"].sum()
        )

        # Loop over the fractures, fetch the projections to the 2d grid, and count the
        # number of non-zero entries in the projection matrix.
        non_zero_projection_primary = 0
        for mg in self.mdg.interfaces(dim=1):
            proj_primary = mg.mortar_to_primary_avg()
            non_zero_projection_primary += proj_primary.nnz
        # A matching grid would have equality here. We have refined the mortar grid,
        # hence there should be more items in the projection matrix than there are
        # fracture faces in the matrix grid.
        assert non_zero_projection_primary > num_fracture_faces_from_matrix

        # For the mortar grid vs fracture grids, we can simply do a cell count
        for mg in self.mdg.interfaces(dim=1):
            _, sd_secondary = self.mdg.neighboring_subdomains(mg)
            assert 2 * sd_secondary.num_cells != mg.num_cells

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


@pytest.mark.parametrize(
    "refinement_level",
    [
        0,
        pytest.param(1, marks=pytest.mark.skipped),
        pytest.param(2, marks=pytest.mark.skipped),
        pytest.param(3, marks=pytest.mark.skipped),
    ],
)
def test_benchmark_3d_case_3(refinement_level):
    """Test the mdg generator for the regular case of the benchmark study.

    By default verify only that the coarsest grid can be generated, to limit the
    computational time.

    """
    # For reference: In January 2024 (in connection with GH PR #1096), EK verified that
    # the benchmark grid could be generated with all supported refinement levels
    # (keyword "refinement_level" in the benchmark function set to {0, 1, 2, 3}). The
    # expectation is that this will continue to be the case, and if not, that the
    # generation will fail for all refinement levels, for the same reason.
    mdg, _ = pp.mdg_library.benchmark_3d_case_3(refinement_level)

    # Weak tests
    assert (mdg.dim_max() == 3) and (mdg.dim_min() == 1)
    assert (len(mdg.subdomains()) == 16) and (len(mdg.interfaces()) == 22)
