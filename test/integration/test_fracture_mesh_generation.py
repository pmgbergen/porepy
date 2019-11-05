"""
Various method for creating grids for relatively simple fracture networks.

The module doubles as a test framework (though not unittest), and will report
on any problems if ran as a main method.

"""
import numpy as np
import unittest
from collections import namedtuple

import porepy as pp


# Named tuple used to identify intersections of fractures by their parent fractures
# and their coordinates
IntersectionInfo = namedtuple("IsectInfo", ["parent_0", "parent_1", "coord"])
IntersectionInfo3Frac = namedtuple(
    "IsectInfo3", ["parent_0", "parent_1", "parent_2", "coord"]
)


class TestDFMMeshGeneration(unittest.TestCase):
    def check_gb(self, gb, domain, fractures=None, isect_line=None, isect_pt=None):

        if fractures is None:
            fractures = []
        if isect_line is None:
            isect_line = []
        if isect_pt is None:
            isect_pt = []

        def compare_bounding_boxes(box_1, box_2):
            for k, v in box_1.items():
                if np.abs(box_2[k] - v) > 1e-10:
                    return False
            return True

        bb = pp.bounding_box.from_points(gb.grids_of_dimension(3)[0].nodes)

        self.assertTrue(compare_bounding_boxes(bb, domain))

        self.assertTrue(len(fractures) == len(gb.grids_of_dimension(2)))

        # Loop over all fractures, find the grid with the corresponding frac_num. Check
        # that their bounding boxes are the same.
        for fi, f in enumerate(fractures):
            for g in gb.grids_of_dimension(2):
                if g.frac_num == fi:
                    self.assertTrue(
                        compare_bounding_boxes(
                            pp.bounding_box.from_points(f.p),
                            pp.bounding_box.from_points(g.nodes),
                        )
                    )

        ii_computed_box = {}

        for isect in isect_line:
            if isinstance(isect, IntersectionInfo):
                p_0, p_1 = sorted([isect.parent_0, isect.parent_1])

                ii_computed_box[(p_0, p_1)] = {
                    "coord": {
                        "xmin": np.inf,
                        "xmax": -np.inf,
                        "ymin": np.inf,
                        "ymax": -np.inf,
                        "zmin": np.inf,
                        "zmax": -np.inf,
                    },
                    "isect": isect,
                }
            elif isinstance(isect, IntersectionInfo3Frac):
                p_0, p_1, p_2 = sorted([isect.parent_0, isect.parent_1, isect.parent_2])
                ii_computed_box[(p_0, p_1, p_2)] = {
                    "coord": {
                        "xmin": np.inf,
                        "xmax": -np.inf,
                        "ymin": np.inf,
                        "ymax": -np.inf,
                        "zmin": np.inf,
                        "zmax": -np.inf,
                    },
                    "isect": isect,
                }

        def update_box(box, update):
            for k in ["xmin", "ymin", "zmin"]:
                box["coord"][k] = min(box["coord"][k], update[k])
            for k in ["xmax", "ymax", "zmax"]:
                box["coord"][k] = max(box["coord"][k], update[k])

        for g in gb.grids_of_dimension(1):
            n = gb.node_neighbors(g, only_higher=True)
            box = pp.bounding_box.from_points(g.nodes)

            if n.size == 2:
                f_0, f_1 = sorted([n[0].frac_num, n[1].frac_num])
                update_box(ii_computed_box[(f_0, f_1)], box)

            elif n.size == 2:
                f_0, f_1, f_2 = sorted([n[0].frac_num, n[1].frac_num, n[2].frac_num])
                update_box(ii_computed_box[(f_0, f_1, f_2)], box)

        for val in ii_computed_box.values():
            coord = val["coord"]
            isect = val["isect"]
            self.assertTrue(
                compare_bounding_boxes(coord, pp.bounding_box.from_points(isect.coord))
            )

    def test_no_fracture(self):
        domain = {"xmin": -2, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -2, "zmax": 2}
        network = pp.FractureNetwork3d(domain=domain)
        mesh_args = {"mesh_size_bound": 1, "mesh_size_frac": 1, "mesh_size_min": 0.1}
        gb = network.mesh(mesh_args)

        self.check_gb(gb, domain)

    def test_single_isolated_fracture(self):
        """
        A single fracture completely immersed in a boundary grid.
        """
        f_1 = pp.Fracture(np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))
        domain = {"xmin": -2, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -2, "zmax": 2}
        network = pp.FractureNetwork3d(f_1, domain=domain)
        mesh_args = {"mesh_size_bound": 1, "mesh_size_frac": 1, "mesh_size_min": 0.1}
        gb = network.mesh(mesh_args)

        self.check_gb(gb, domain, fractures=[f_1])

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

        gb = network.mesh(mesh_args)

        isect_coord = np.array([[0, 0], [0, 0], [-0.7, 0.8]])

        isect = IntersectionInfo(0, 1, isect_coord)

        self.check_gb(gb, domain, fractures=[f_1, f_2], isect_line=[isect])

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

        gb = network.mesh(mesh_args)

        isect_0_coord = np.array([[0, 0], [0, 0], [-0.7, 0.8]])
        isect_1_coord = np.array([[-1, 1], [0, 0], [0, 0]])
        isect_2_coord = np.array([[0, 0], [-1, 1], [0, 0]])

        isect_0 = IntersectionInfo(0, 1, isect_0_coord)
        isect_1 = IntersectionInfo(0, 2, isect_1_coord)
        isect_2 = IntersectionInfo(1, 2, isect_2_coord)

        isect_pt = np.array([0, 0, 0]).reshape((-1, 1))

        self.check_gb(
            gb,
            domain,
            fractures=[f_1, f_2, f_3],
            isect_line=[isect_0, isect_1, isect_2],
            isect_pt=[isect_pt],
        )

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
        gb = network.mesh(mesh_args)

        isect_0_coord = np.array([[0, 0], [0, 0], [-0.7, 0.8]])

        isect_1_coord = np.array([[0.5, 0.5], [0, 0], [-0.7, 0.8]])

        isect_0 = IntersectionInfo(0, 1, isect_0_coord)
        isect_1 = IntersectionInfo(0, 2, isect_1_coord)
        self.check_gb(
            gb, domain, fractures=[f_1, f_2, f_3], isect_line=[isect_0, isect_1]
        )

    def test_split_into_octants(self, **kwargs):
        f_1 = pp.Fracture(np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]]))
        f_2 = pp.Fracture(np.array([[-1, -1, 1, 1], [-1, 1, 1, -1], [0, 0, 0, 0]]))
        f_3 = pp.Fracture(np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1, -1, 1, 1]]))
        domain = {"xmin": -2, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -2, "zmax": 2}
        mesh_args = {"mesh_size_frac": 0.4, "mesh_size_bound": 1, "mesh_size_min": 0.2}

        network = pp.FractureNetwork3d([f_1, f_2, f_3], domain=domain)
        gb = network.mesh(mesh_args)
        isect_0_coord = np.array([[0, 0], [0, 0], [-1, 1]])
        isect_1_coord = np.array([[-1, 1], [0, 0], [0, 0]])
        isect_2_coord = np.array([[0, 0], [-1, 1], [0, 0]])

        isect_0 = IntersectionInfo(0, 2, isect_0_coord)
        isect_1 = IntersectionInfo(0, 1, isect_1_coord)
        isect_2 = IntersectionInfo(1, 2, isect_2_coord)

        isect_pt = np.array([0, 0, 0]).reshape((-1, 1))

        self.check_gb(
            gb,
            domain,
            fractures=[f_1, f_2, f_3],
            isect_line=[isect_0, isect_1, isect_2],
            isect_pt=[isect_pt],
        )

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
        gb = network.mesh(mesh_args)

        isect_coord = np.array([[0, 0], [0, 1], [0, 0]])

        isect_0 = IntersectionInfo(0, 1, isect_coord)

        self.check_gb(gb, domain, fractures=[f_1, f_2], isect_line=[isect_0])

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
        gb = network.mesh(mesh_args)

        isect_coord = np.array([[0, 0], [0.3, 0.7], [0, 0]])

        isect_0 = IntersectionInfo(0, 1, isect_coord)

        self.check_gb(gb, domain, fractures=[f_1, f_2], isect_line=[isect_0])

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
        gb = network.mesh(mesh_args)

        isect_coord = np.array([[0, 0], [0.5, 1], [0, 0]])

        isect_0 = IntersectionInfo(0, 1, isect_coord)
        self.check_gb(gb, domain, fractures=[f_1, f_2], isect_line=[isect_0])

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
        gb = network.mesh(mesh_args)

        isect_coord = np.array([[0.5, 0.5], [0.5, 0.9], [0, 0]])

        isect_0 = IntersectionInfo(0, 1, isect_coord)

        self.check_gb(gb, domain, fractures=[f_1, f_2], isect_line=[isect_0])

    def test_T_intersection_one_outside_plane(self, **kwargs):
        f_1 = pp.Fracture(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T)
        f_2 = pp.Fracture(np.array([[0.5, 0.5, 1], [0.5, 0.5, 0], [0.5, 1.9, 0.0]]).T)

        domain = {"xmin": -1, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -1, "zmax": 2}
        mesh_args = {"mesh_size_frac": 0.4, "mesh_size_bound": 1, "mesh_size_min": 0.2}

        network = pp.FractureNetwork3d([f_1, f_2], domain=domain)
        gb = network.mesh(mesh_args)

        isect_coord = np.array([[0.5, 0.5], [0.5, 1.0], [0, 0]])

        isect_0 = IntersectionInfo(0, 1, isect_coord)

        self.check_gb(gb, domain, fractures=[f_1, f_2], isect_line=[isect_0])

    def test_T_intersection_both_outside_plane(self, **kwargs):
        f_1 = pp.Fracture(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T)
        f_2 = pp.Fracture(np.array([[0.5, 0.5, 1], [0.5, -0.5, 0], [0.5, 1.9, 0.0]]).T)

        domain = {"xmin": -1, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -1, "zmax": 2}
        mesh_args = {"mesh_size_frac": 0.4, "mesh_size_bound": 1, "mesh_size_min": 0.2}

        network = pp.FractureNetwork3d([f_1, f_2], domain=domain)
        gb = network.mesh(mesh_args)

        isect_coord = np.array([[0.5, 0.5], [0.0, 1.0], [0, 0]])

        isect_0 = IntersectionInfo(0, 1, isect_coord)

        self.check_gb(gb, domain, fractures=[f_1, f_2], isect_line=[isect_0])

    def test_T_intersection_both_on_boundary(self, **kwargs):
        f_1 = pp.Fracture(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T)
        f_2 = pp.Fracture(np.array([[0.0, 0.5, 0], [1, 0.5, 0], [0.5, 0.5, 1.0]]).T)

        domain = {"xmin": -1, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -1, "zmax": 2}
        mesh_args = {"mesh_size_frac": 0.4, "mesh_size_bound": 1, "mesh_size_min": 0.2}

        network = pp.FractureNetwork3d([f_1, f_2], domain=domain)
        gb = network.mesh(mesh_args)

        isect_coord = np.array([[0.0, 1.0], [0.5, 0.5], [0, 0]])

        isect_0 = IntersectionInfo(0, 1, isect_coord)

        self.check_gb(gb, domain, fractures=[f_1, f_2], isect_line=[isect_0])

    def test_T_intersection_one_boundary_one_outside(self, **kwargs):
        f_1 = pp.Fracture(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T)
        f_2 = pp.Fracture(np.array([[-0.2, 0.5, 0], [1, 0.5, 0], [0.5, 0.5, 1.0]]).T)

        domain = {"xmin": -1, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -1, "zmax": 2}
        mesh_args = {"mesh_size_frac": 0.4, "mesh_size_bound": 1, "mesh_size_min": 0.2}

        network = pp.FractureNetwork3d([f_1, f_2], domain=domain)
        gb = network.mesh(mesh_args)

        isect_coord = np.array([[0.0, 1.0], [0.5, 0.5], [0, 0]])

        isect_0 = IntersectionInfo(0, 1, isect_coord)

        self.check_gb(gb, domain, fractures=[f_1, f_2], isect_line=[isect_0])

    def test_T_intersection_one_boundary_one_inside(self, **kwargs):
        f_1 = pp.Fracture(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T)
        f_2 = pp.Fracture(np.array([[0.2, 0.5, 0], [1, 0.5, 0], [0.5, 0.5, 1.0]]).T)

        domain = {"xmin": -1, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -1, "zmax": 2}
        mesh_args = {"mesh_size_frac": 0.4, "mesh_size_bound": 1, "mesh_size_min": 0.2}

        network = pp.FractureNetwork3d([f_1, f_2], domain=domain)
        gb = network.mesh(mesh_args)

        isect_coord = np.array([[0.2, 1.0], [0.5, 0.5], [0, 0]])

        isect_0 = IntersectionInfo(0, 1, isect_coord)

        self.check_gb(gb, domain, fractures=[f_1, f_2], isect_line=[isect_0])

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


class TestDFMNonConvexDomain(unittest.TestCase):
    def setUp(self):
        west = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        east = np.array([[1, 1, 1, 1], [0, 1, 1, 0], [0, 0, 1, 1]])
        south = np.array([[0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 1, 1]])
        north = np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 0, 1, 1]])
        bottom = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        top = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [1, 1, 1, 1]])
        self.cart_polyhedron = [west, east, south, north, bottom, top]

        south_w = np.array([[0, 0.5, 0.5, 0], [0, 0, 0, 0], [0, 0.5, 1, 1]])
        south_e = np.array([[0.5, 1, 1, 0.5], [0, 0, 0, 0], [0.5, 0, 1, 1]])
        north_w = np.array([[0, 0.5, 0.5, 0], [1, 1, 1, 1], [0, 0.5, 1, 1]])
        north_e = np.array([[0.5, 1, 1, 0.5], [1, 1, 1, 1], [0.5, 0, 1, 1]])
        bottom_w = np.array([[0, 0.5, 0.5, 0], [0, 0, 1, 1], [0, 0.5, 0.5, 0]])
        bottom_e = np.array([[0.5, 1, 1, 0.5], [0, 0, 1, 1], [0.5, 0.0, 0, 0.5]])
        top_w = np.array([[0, 0.5, 0.5, 0], [0, 0, 1, 1], [1, 1, 1, 1]])
        top_e = np.array([[0.5, 1, 1, 0.5], [0, 0, 1, 1], [1, 1, 1, 1]])
        self.non_convex_polyhedron = [
            west,
            east,
            south_w,
            south_e,
            north_w,
            north_e,
            bottom_w,
            bottom_e,
            top_w,
            top_e,
        ]

    def test_fracture_split_by_domain(self):
        self.setUp()
        f_1 = pp.Fracture(
            np.array([[-1, 2, 2, -1], [0.5, 0.5, 0.5, 0.5], [-1, -1, 0.3, 0.3]])
        )

        network = pp.FractureNetwork3d(f_1, domain=self.non_convex_polyhedron)
        mesh_args = {"mesh_size_bound": 1, "mesh_size_frac": 1, "mesh_size_min": 0.1}
        gb = network.mesh(mesh_args)
        self.assertTrue(len(gb.grids_of_dimension(2)) == 2)

    def test_fracture_cut_not_split_by_domain(self):
        self.setUp()
        f_1 = pp.Fracture(
            np.array([[-1, 2, 2, -1], [0.5, 0.5, 0.5, 0.5], [-1, -1, 0.7, 0.7]])
        )

        network = pp.FractureNetwork3d(f_1, domain=self.non_convex_polyhedron)
        mesh_args = {"mesh_size_bound": 1, "mesh_size_frac": 1, "mesh_size_min": 0.1}
        gb = network.mesh(mesh_args)
        self.assertTrue(len(gb.grids_of_dimension(2)) == 1)


class Test2dDomain(unittest.TestCase):
    def setUp(self):
        self.domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
        self.mesh_args = {
            "mesh_size_bound": 1,
            "mesh_size_frac": 1,
            "mesh_size_min": 0.1,
        }

        self.p1 = np.array([[0.2, 0.8], [0.2, 0.8]])
        self.e1 = np.array([[0], [1]])
        self.p2 = np.array([[0.2, 0.8, 0.2, 0.8], [0.2, 0.8, 0.8, 0.2]])
        self.e2 = np.array([[0, 2], [1, 3]])

    def test_no_fractures(self):
        self.setUp()
        network = pp.FractureNetwork2d(domain=self.domain)
        gb = network.mesh(self.mesh_args)
        self.assertTrue(len(gb.grids_of_dimension(2)) == 1)
        self.assertTrue(len(gb.grids_of_dimension(1)) == 0)
        self.assertTrue(len(gb.grids_of_dimension(0)) == 0)

    def test_one_fracture(self):
        self.setUp()
        network = pp.FractureNetwork2d(self.p1, self.e1, domain=self.domain)
        gb = network.mesh(self.mesh_args)
        self.assertTrue(len(gb.grids_of_dimension(2)) == 1)
        self.assertTrue(len(gb.grids_of_dimension(1)) == 1)
        self.assertTrue(len(gb.grids_of_dimension(0)) == 0)

    def test_two_fractures(self):
        self.setUp()
        network = pp.FractureNetwork2d(self.p2, self.e2, domain=self.domain)
        gb = network.mesh(self.mesh_args)
        self.assertTrue(len(gb.grids_of_dimension(2)) == 1)
        self.assertTrue(len(gb.grids_of_dimension(1)) == 2)
        self.assertTrue(len(gb.grids_of_dimension(0)) == 1)

    def test_one_constraint(self):
        self.setUp()
        network = pp.FractureNetwork2d(self.p1, self.e1, domain=self.domain)
        gb = network.mesh(self.mesh_args, constraints=np.array([0]))
        self.assertTrue(len(gb.grids_of_dimension(2)) == 1)
        self.assertTrue(len(gb.grids_of_dimension(1)) == 0)
        self.assertTrue(len(gb.grids_of_dimension(0)) == 0)

    def test_two_constraints(self):
        self.setUp()
        network = pp.FractureNetwork2d(self.p2, self.e2, domain=self.domain)
        gb = network.mesh(self.mesh_args, constraints=np.arange(2))
        self.assertTrue(len(gb.grids_of_dimension(2)) == 1)
        self.assertTrue(len(gb.grids_of_dimension(1)) == 0)
        self.assertTrue(len(gb.grids_of_dimension(0)) == 0)

    def test_one_fracture_one_constraint(self):
        self.setUp()
        network = pp.FractureNetwork2d(self.p2, self.e2, domain=self.domain)
        gb = network.mesh(self.mesh_args, constraints=np.array(1))
        self.assertTrue(len(gb.grids_of_dimension(2)) == 1)
        self.assertTrue(len(gb.grids_of_dimension(1)) == 1)
        self.assertTrue(len(gb.grids_of_dimension(0)) == 0)


if __name__ == "__main__":
    unittest.main()
