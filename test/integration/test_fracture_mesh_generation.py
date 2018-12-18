"""
Various method for creating grids for relatively simple fracture networks.

The module doubles as a test framework (though not unittest), and will report
on any problems if ran as a main method.

"""
import numpy as np
import unittest

import porepy as pp


class TestDFMMeshGeneration(unittest.TestCase):
    def test_no_fracture(self):
        domain = {"xmin": -2, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -2, "zmax": 2}
        network = pp.FractureNetwork3d(domain=domain)
        mesh_args = {"mesh_size_bound": 1, "mesh_size_frac": 1, "mesh_size_min": 0.1}
        network.mesh(mesh_args)

    def test_single_isolated_fracture(self):
        """
        A single fracture completely immersed in a boundary grid.
        """
        f_1 = pp.Fracture(np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))
        domain = {"xmin": -2, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -2, "zmax": 2}
        network = pp.FractureNetwork3d(f_1, domain=domain)
        mesh_args = {"mesh_size_bound": 1, "mesh_size_frac": 1, "mesh_size_min": 0.1}
        network.mesh(mesh_args)

    def test_two_intersecting_fractures(self, **kwargs):
        """
        Two fractures intersecting along a line.

        The example also sets different characteristic mesh lengths on the boundary
        and at the fractures.

        """

        f_1 = pp.Fracture(np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))
        f_2 = pp.Fracture(
            np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-0.7, -0.7, 0.8, 0.8]])
        )
        domain = {"xmin": -2, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -2, "zmax": 2}

        mesh_args = {"mesh_size_frac": 0.5, "mesh_size_bound": 1, "mesh_size_min": 0.2}
        network = pp.FractureNetwork3d([f_1, f_2], domain=domain)

        network.mesh(mesh_args)

    def test_three_intersecting_fractures(self, **kwargs):
        """
        Three fractures intersecting, with intersecting intersections (point)
        """

        f_1 = pp.Fracture(np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))
        f_2 = pp.Fracture(
            np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-0.7, -0.7, 0.8, 0.8]])
        )
        f_3 = pp.Fracture(np.array([[-1, 1, 1, -1], [-1, -1, 1, 1], [0, 0, 0, 0]]))

        # Add some parameters for grid size
        domain = {"xmin": -2, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -2, "zmax": 2}

        mesh_args = {"mesh_size_frac": 0.5, "mesh_size_bound": 1, "mesh_size_min": 0.2}
        network = pp.FractureNetwork3d([f_1, f_2, f_3], domain=domain)

        network.mesh(mesh_args)

    def test_one_fracture_intersected_by_two(self, **kwargs):
        """
        One fracture, intersected by two other (but no point intersections)
        """

        f_1 = pp.Fracture(np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))
        f_2 = pp.Fracture(
            np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-0.7, -0.7, 0.8, 0.8]])
        )
        f_3 = pp.Fracture(f_2.p + np.array([0.5, 0, 0]).reshape((-1, 1)))

        # Add some parameters for grid size
        domain = {"xmin": -2, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -2, "zmax": 2}
        mesh_args = {"mesh_size_frac": 0.4, "mesh_size_bound": 1, "mesh_size_min": 0.2}

        network = pp.FractureNetwork3d([f_1, f_2, f_3], domain=domain)
        network.mesh(mesh_args)

    def test_split_into_octants(self, **kwargs):
        f_1 = pp.Fracture(np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))
        f_2 = pp.Fracture(np.array([[-1, -1, 1, 1], [-1, 1, 1, -1], [0, 0, 0, 0]]))
        f_3 = pp.Fracture(np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 1, 1]]))
        domain = {"xmin": -2, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -2, "zmax": 2}
        mesh_args = {"mesh_size_frac": 0.4, "mesh_size_bound": 1, "mesh_size_min": 0.2}

        network = pp.FractureNetwork3d([f_1, f_2, f_3], domain=domain)
        network.mesh(mesh_args)

    def test_three_fractures_sharing_line_same_segment(self, **kwargs):
        """
        Three fractures that all share an intersection line. This can be considered
        as three intersection lines that coincide.
        """
        f_1 = pp.Fracture(np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))
        f_2 = pp.Fracture(np.array([[-1, 1, 1, -1], [-1, 1, 1, -1], [-1, -1, 1, 1]]))
        f_3 = pp.Fracture(np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 1, 1]]))
        domain = {"xmin": -2, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -2, "zmax": 2}
        mesh_args = {"mesh_size_frac": 0.4, "mesh_size_bound": 1, "mesh_size_min": 0.2}

        network = pp.FractureNetwork3d([f_1, f_2, f_3], domain=domain)
        network.mesh(mesh_args)

    def test_three_fractures_split_segments(self, **kwargs):
        """
        Three fractures that all intersect along the same line, but with the
        intersection between two of them forming an extension of the intersection
        of all three.
        """
        f_1 = pp.Fracture(np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))
        f_2 = pp.Fracture(
            np.array([[-1, 1, 1, -1], [-1, 1, 1, -1], [-0.5, -0.5, 0.5, 0.5]])
        )
        f_3 = pp.Fracture(np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 1, 1]]))
        domain = {"xmin": -2, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -2, "zmax": 2}
        mesh_args = {"mesh_size_frac": 0.4, "mesh_size_bound": 1, "mesh_size_min": 0.2}

        network = pp.FractureNetwork3d([f_1, f_2, f_3], domain=domain)
        network.mesh(mesh_args)

    def test_two_fractures_L_intersection(self, **kwargs):
        """
        Two fractures sharing a segment in an L-intersection.
        """
        f_1 = pp.Fracture(np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]))
        f_2 = pp.Fracture(np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]))
        domain = {"xmin": -2, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -2, "zmax": 2}
        mesh_args = {"mesh_size_frac": 0.4, "mesh_size_bound": 1, "mesh_size_min": 0.2}

        network = pp.FractureNetwork3d([f_1, f_2], domain=domain)
        network.mesh(mesh_args)

    def test_two_fractures_L_intersection_part_of_segment(self, **kwargs):
        """
        Two fractures sharing what is a full segment for one, an part of a segment
        for the other.
        """
        f_1 = pp.Fracture(np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]))
        f_2 = pp.Fracture(np.array([[0, 0, 0, 0], [0.3, 0.7, 0.7, 0.3], [0, 0, 1, 1]]))
        domain = {"xmin": -2, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -2, "zmax": 2}
        mesh_args = {"mesh_size_frac": 0.4, "mesh_size_bound": 1, "mesh_size_min": 0.2}

        network = pp.FractureNetwork3d([f_1, f_2], domain=domain)
        network.mesh(mesh_args)

    def test_two_fractures_L_intersection_one_displaced(self, **kwargs):
        """
        Two fractures sharing what is a part of segments for both.
        """
        f_1 = pp.Fracture(np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]))
        f_2 = pp.Fracture(np.array([[0, 0, 0, 0], [0.5, 1.5, 1.5, 0.5], [0, 0, 1, 1]]))
        domain = {"xmin": -2, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -2, "zmax": 2}
        mesh_args = {"mesh_size_frac": 0.4, "mesh_size_bound": 1, "mesh_size_min": 0.2}

        network = pp.FractureNetwork3d([f_1, f_2], domain=domain)
        network.mesh(mesh_args)

    def test_T_intersection_within_plane(self, **kwargs):
        f_1 = pp.Fracture(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T)
        f_2 = pp.Fracture(np.array([[0.5, 0.5, 1], [0.5, 0.5, 0], [0.5, 0.9, 0.0]]).T)

        domain = {"xmin": -1, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -1, "zmax": 2}
        # This test, when used with certain versions of gmsh (<2.15?) gives
        # a mismatch between 2d cells and 3d faces on fracture surfaces. The bug
        # can be located in the .msh-file. To function as a test, we disband the
        # test of cell-face relations.
        mesh_args = {"mesh_size_frac": 0.4, "mesh_size_bound": 1, "mesh_size_min": 0.2}

        network = pp.FractureNetwork3d([f_1, f_2], domain=domain)
        network.mesh(mesh_args)

    def test_T_intersection_one_outside_plane(self, **kwargs):
        f_1 = pp.Fracture(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T)
        f_2 = pp.Fracture(np.array([[0.5, 0.5, 1], [0.5, 0.5, 0], [0.5, 1.9, 0.0]]).T)

        domain = {"xmin": -1, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -1, "zmax": 2}
        mesh_args = {"mesh_size_frac": 0.4, "mesh_size_bound": 1, "mesh_size_min": 0.2}

        network = pp.FractureNetwork3d([f_1, f_2], domain=domain)
        network.mesh(mesh_args)

    def test_T_intersection_both_outside_plane(self, **kwargs):
        f_1 = pp.Fracture(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T)
        f_2 = pp.Fracture(np.array([[0.5, 0.5, 1], [0.5, -0.5, 0], [0.5, 1.9, 0.0]]).T)

        domain = {"xmin": -1, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -1, "zmax": 2}
        mesh_args = {"mesh_size_frac": 0.4, "mesh_size_bound": 1, "mesh_size_min": 0.2}

        network = pp.FractureNetwork3d([f_1, f_2], domain=domain)
        network.mesh(mesh_args)

    def test_T_intersection_both_on_boundary(self, **kwargs):
        f_1 = pp.Fracture(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T)
        f_2 = pp.Fracture(np.array([[0.0, 0.5, 0], [1, 0.5, 0], [0.5, 0.5, 1.0]]).T)

        domain = {"xmin": -1, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -1, "zmax": 2}
        mesh_args = {"mesh_size_frac": 0.4, "mesh_size_bound": 1, "mesh_size_min": 0.2}

        network = pp.FractureNetwork3d([f_1, f_2], domain=domain)
        network.mesh(mesh_args)

    def test_T_intersection_one_boundary_one_outside(self, **kwargs):
        f_1 = pp.Fracture(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T)
        f_2 = pp.Fracture(np.array([[-0.2, 0.5, 0], [1, 0.5, 0], [0.5, 0.5, 1.0]]).T)

        domain = {"xmin": -1, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -1, "zmax": 2}
        mesh_args = {"mesh_size_frac": 0.4, "mesh_size_bound": 1, "mesh_size_min": 0.2}

        network = pp.FractureNetwork3d([f_1, f_2], domain=domain)
        network.mesh(mesh_args)

    def test_T_intersection_one_boundary_one_inside(self, **kwargs):
        f_1 = pp.Fracture(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T)
        f_2 = pp.Fracture(np.array([[0.2, 0.5, 0], [1, 0.5, 0], [0.5, 0.5, 1.0]]).T)

        domain = {"xmin": -1, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -1, "zmax": 2}
        mesh_args = {"mesh_size_frac": 0.4, "mesh_size_bound": 1, "mesh_size_min": 0.2}

        network = pp.FractureNetwork3d([f_1, f_2], domain=domain)
        network.mesh(mesh_args)

    def test_issue_54(self):
        domain = {"xmin": -1, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}
        f_1 = pp.Fracture(
            np.array([[0.5, 0.5, 0.5, 0.5], [0.4, 0.5, 0.5, 0.4], [0.2, 0.2, 0.8, 0.8]])
        )
        f_2 = pp.Fracture(
            np.array([[0, 0.8, 0.8, 0], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]])
        )
        mesh_args = {"mesh_size_frac": 0.4, "mesh_size_bound": 1, "mesh_size_min": 0.2}

        network = pp.FractureNetwork3d([f_1, f_2], domain=domain)
        network.mesh(mesh_args)

    def test_issue_58_1(self):
        domain = {"xmin": -2, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -2, "zmax": 2}
        mesh_args = {"mesh_size_frac": 0.4, "mesh_size_bound": 1, "mesh_size_min": 0.2}
        f_1 = pp.Fracture(np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]))
        f_2 = pp.Fracture(np.array([[0, 1, 1, 0], [0, 0.5, 0.5, 0], [0, 0, 1, 1]]))

        network = pp.FractureNetwork3d([f_1, f_2], domain=domain)
        network.mesh(mesh_args)

    def test_issue_58_2(self):
        domain = {"xmin": -2, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -2, "zmax": 2}
        mesh_args = {
            "mesh_size_frac": 0.4,
            "mesh_size_bound": 1,
            "mesh_size_min": 0.2,
            "return_expected": True,
        }

        f_1 = pp.Fracture(np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]))
        f_2 = pp.Fracture(np.array([[0, 1, 1, 0], [0.3, 0.5, 0.5, 0.3], [0, 0, 1, 1]]))

        network = pp.FractureNetwork3d([f_1, f_2], domain=domain)
        network.mesh(mesh_args)

    def test_issue_58_3(self):
        domain = {"xmin": -2, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -2, "zmax": 2}
        mesh_args = {
            "mesh_size_frac": 0.5,
            "mesh_size_bound": 1,
            "mesh_size_min": 0.2,
            "return_expected": True,
        }

        f_1 = pp.Fracture(np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]]))
        f_2 = pp.Fracture(np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 1, 1]]))

        network = pp.FractureNetwork3d([f_1, f_2], domain=domain)
        network.mesh(mesh_args)

    def test_geiger_3d_partial(self):
        f_1 = pp.Fracture(
            np.array([[0.5, 0, 0], [0.5, 1, 0], [0.5, 1, 1], [0.5, 0, 1]]).T
        )
        f_2 = pp.Fracture(
            np.array([[0, 0.5, 0], [1, 0.5, 0], [1, 0.5, 1], [0, 0.5, 1]]).T
        )
        f_3 = pp.Fracture(
            np.array([[0, 0, 0.5], [1, 0, 0.5], [1, 1, 0.5], [0, 1, 0.5]]).T
        )
        f_4 = pp.Fracture(
            np.array(
                [[0.5, 0.5, 0.75], [1.0, 0.5, 0.75], [1.0, 1.0, 0.75], [0.5, 1.0, 0.75]]
            ).T
        )
        f_5 = pp.Fracture(
            np.array(
                [[0.75, 0.5, 0.5], [0.75, 1.0, 0.5], [0.75, 1.0, 1.0], [0.75, 0.5, 1.0]]
            ).T
        )
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}
        mesh_args = {
            "mesh_size_frac": 0.4,
            "mesh_size_bound": 1,
            "mesh_size_min": 0.2,
            "return_expected": True,
        }
        network = pp.FractureNetwork3d([f_1, f_2, f_3, f_4, f_5], domain=domain)
        network.mesh(mesh_args)

    def test_issue_90(self):
        f_1 = pp.Fracture(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T)
        f_2 = pp.Fracture(np.array([[0, 0, 1], [0, 0, -1], [1, 1, -1]]).T)

        mesh_args = {"mesh_size_frac": 0.4, "mesh_size_bound": 1, "mesh_size_min": 0.2}

        network = pp.FractureNetwork3d([f_1, f_2])
        network.mesh(mesh_args)


class TestMeshGenerationFractureHitsBoundary(unittest.TestCase):
    def kwarguments_and_domain(self):
        domain = {"xmin": -2, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -2, "zmax": 2}
        mesh_args = {"mesh_size_frac": 1, "mesh_size_bound": 1, "mesh_size_min": 0.2}

        return mesh_args, domain

    def test_one_fracture_touching_one_boundary(self):
        """
        A single fracture completely immersed in a boundary grid.
        """
        f_1 = pp.Fracture(np.array([[-1, 2, 2, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))

        mesh_args, domain = self.kwarguments_and_domain()
        network = pp.FractureNetwork3d([f_1], domain)
        network.mesh(mesh_args)

    def test_one_fracture_touching_two_neighbouring_boundaries(self):
        """
        Two fractures intersecting along a line.

        The example also sets different characteristic mesh lengths on the boundary
        and at the fractures.

        """

        f_1 = pp.Fracture(np.array([[-1, 2, 2, -1], [0, 0, 0, 0], [-1, -1, 2, 2]]))

        mesh_args, domain = self.kwarguments_and_domain()
        network = pp.FractureNetwork3d([f_1], domain)
        network.mesh(mesh_args)

    def test_one_fracture_touching_two_opposing_boundaries(self):
        """
        Two fractures intersecting along a line.

        The example also sets different characteristic mesh lengths on the boundary
        and at the fractures.

        """

        f_1 = pp.Fracture(np.array([[-2, 2, 2, -2], [0, 0, 0, 0], [-1, -1, 1, 1]]))
        mesh_args, domain = self.kwarguments_and_domain()
        network = pp.FractureNetwork3d([f_1], domain)
        network.mesh(mesh_args)

    def test_one_fracture_touching_three_boundaries(self):
        """
        Three fractures intersecting, with intersecting intersections (point)
        """

        f_1 = pp.Fracture(np.array([[-2, 2, 2, -2], [0, 0, 0, 0], [-1, -1, 2, 2]]))
        mesh_args, domain = self.kwarguments_and_domain()
        network = pp.FractureNetwork3d([f_1], domain)
        network.mesh(mesh_args)

    def test_one_fracture_touches_four_boundaries(self):
        """
        One fracture, intersected by two other (but no point intersections)
        """

        f_1 = pp.Fracture(np.array([[-2, 2, 2, -2], [0, 0, 0, 0], [-2, -2, 2, 2]]))

        mesh_args, domain = self.kwarguments_and_domain()
        network = pp.FractureNetwork3d([f_1], domain)
        network.mesh(mesh_args)

    def test_one_fracture_touches_boundary_corners(self):
        """
        One fracture, intersected by two other (but no point intersections)
        """

        f_1 = pp.Fracture(np.array([[-2, 2, 2, -2], [-2, 2, 2, -2], [-2, -2, 2, 2]]))

        mesh_args, domain = self.kwarguments_and_domain()
        network = pp.FractureNetwork3d([f_1], domain)
        network.mesh(mesh_args)

    def test_two_fractures_touching_one_boundary(self):
        f_1 = pp.Fracture(np.array([[-1, 2, 2, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))
        f_2 = pp.Fracture(np.array([[-1, -1, 2, 2], [-1, 1, 1, -1], [0, 0, 0, 0]]))
        mesh_args, domain = self.kwarguments_and_domain()
        network = pp.FractureNetwork3d([f_1, f_2], domain)
        network.mesh(mesh_args)

    def test_one_fracture_touches_boundary_and_corner(self):
        f_1 = pp.Fracture(np.array([[-2, 2, 2, -2], [-2, 1, 1, -2], [-2, -2, 2, 2]]))
        mesh_args, domain = self.kwarguments_and_domain()
        network = pp.FractureNetwork3d([f_1], domain)
        network.mesh(mesh_args)


class TestDFNMeshGeneration(unittest.TestCase):
    def test_conforming_two_fractures(self):
        f_1 = pp.Fracture(np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]).T)
        f_2 = pp.Fracture(np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]]).T)
        network = pp.FractureNetwork3d([f_1, f_2])
        mesh_args = {"mesh_size_frac": 0.4, "mesh_size_bound": 1, "mesh_size_min": 0.2}
        network.mesh(mesh_args, dfn=True)

    def test_conforming_three_fractures(self):
        f_1 = pp.Fracture(np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]).T)
        f_2 = pp.Fracture(np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]]).T)
        f_3 = pp.Fracture(np.array([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]]).T)
        network = pp.FractureNetwork3d([f_1, f_2, f_3])
        mesh_args = {"mesh_size_frac": 0.4, "mesh_size_bound": 1, "mesh_size_min": 0.2}
        network.mesh(mesh_args, dfn=True)


if __name__ == "__main__":
    TestDFMMeshGeneration().test_no_fracture()
    unittest.main()
