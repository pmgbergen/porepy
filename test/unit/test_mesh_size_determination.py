"""
Tests of mesh size determination for simplex grids.
"""
import unittest

import numpy as np

import porepy as pp


class TestMeshSize(unittest.TestCase):
    def test_one_internal_2d(self):
        """
        Fracture set:
            f_1 = np.array([[.5, 4], [1, 1]])

        """
        pts = np.array([[0.0, 5.0, 5.0, 0.0, 0.5, 4.0], [0.0, 0.0, 5.0, 5.0, 1.0, 1.0]])
        on_boundary = np.array([True, True, True, True, False, False])
        lines = np.array(
            [[0, 3, 1, 2, 4], [1, 0, 2, 3, 5], [1, 1, 1, 1, 3], [0, 3, 1, 2, 4]]
        )

        mesh_sizes_known = np.array(
            [1.11803399, 1.41421356, 2.0, 2.0, 0.5, 1.0, 0.5, 1.0, 1.80277564, 1.0, 1.0]
        )
        pts_split_known = np.array(
            [
                [0.0, 5.0, 5.0, 0.0, 0.5, 4.0, 0.0, 2.5, 5.0, 0.0, 2.25],
                [0.0, 0.0, 5.0, 5.0, 1.0, 1.0, 1.0, 0.0, 2.5, 3.0, 1.0],
            ]
        )

        mesh_size_min = 0.2
        mesh_size_frac = 1
        mesh_size_bound = 2
        mesh_sizes, pts_split, _ = pp.fracs.tools.determine_mesh_size(
            pts,
            on_boundary,
            lines,
            mesh_size_frac=mesh_size_frac,
            mesh_size_min=mesh_size_min,
            mesh_size_bound=mesh_size_bound,
        )

        self.assertTrue(np.all(np.isclose(mesh_sizes, mesh_sizes_known)))
        self.assertTrue(np.all(np.isclose(pts_split, pts_split_known)))

    def test_one_to_boundary_2d(self):
        """
        Fracture set:
            f_1 = np.array([[.5, 5], [1.5, 1.5]])

        """
        pts = np.array([[0.0, 5.0, 5.0, 0.0, 0.5, 5.0], [0.0, 0.0, 5.0, 5.0, 1.5, 1.5]])
        on_boundary = np.array([True, True, True, True, False, False])
        lines = np.array(
            [
                [0, 3, 1, 5, 2, 4],
                [1, 0, 5, 2, 3, 5],
                [1, 1, 1, 1, 1, 3],
                [0, 3, 1, 1, 2, 4],
            ]
        )

        mesh_sizes_known = np.array(
            [1.58113883, 1.5, 2.0, 2.0, 0.5, 1.0, 0.5, 1.5, 0.75, 1.0, 1.0]
        )
        pts_split_known = np.array(
            [
                [0.0, 5.0, 5.0, 0.0, 0.5, 5.0, 0.0, 2.5, 0.0, 0.0, 2.75],
                [0.0, 0.0, 5.0, 5.0, 1.5, 1.5, 1.5, 0.0, 0.75, 3.25, 1.5],
            ]
        )

        mesh_size_min = 0.2
        mesh_size_frac = 1
        mesh_size_bound = 2
        mesh_sizes, pts_split, _ = pp.fracs.tools.determine_mesh_size(
            pts,
            on_boundary,
            lines,
            mesh_size_frac=mesh_size_frac,
            mesh_size_min=mesh_size_min,
            mesh_size_bound=mesh_size_bound,
        )

        self.assertTrue(np.all(np.isclose(mesh_sizes, mesh_sizes_known)))
        self.assertTrue(np.all(np.isclose(pts_split, pts_split_known)))

    def test_one_to_boundary_3d(self):
        """
        One fracture in 3d going all the way to the boundary
        """

        f_1 = np.array([[1, 5, 5, 1], [1, 1, 1, 1], [1, 1, 3, 3]])
        f_set = [pp.Fracture(f_1)]
        domain = {"xmin": 0, "ymin": 0, "zmin": 0, "xmax": 5, "ymax": 5, "zmax": 5}
        mesh_size_min = 0.1
        mesh_size_frac = 0.1
        mesh_size_bound = 2
        on_boundary = np.array(
            [False, False, False, False, True, True, True, True, True, True, True, True]
        )
        network = pp.FractureNetwork3d(f_set)
        network.impose_external_boundary(domain)
        network.find_intersections()
        network.split_intersections()
        network._insert_auxiliary_points(
            mesh_size_frac=mesh_size_frac,
            mesh_size_min=mesh_size_min,
            mesh_size_bound=mesh_size_bound,
        )
        mesh_size = network._determine_mesh_size(point_tags=on_boundary)

        decomp = network.decomposition

        # Many of the points should have mesh size of 2, adjust the exceptions below
        mesh_size_known = np.full(
            decomp["points"].shape[1], mesh_size_bound, dtype=np.float
        )

        # Find the points on the fracture
        fracture_poly = np.where(np.logical_not(network.tags["boundary"]))[0]
        self.assertTrue(fracture_poly.size == 1)
        fracture_points = decomp["polygons"][fracture_poly[0]][0]
        # These should have been assigned fracture mesh size
        mesh_size_known[fracture_points] = mesh_size_frac

        # Two of the domain corners are close enough to the fracture to have their
        # mesh size modified from the default boundary value
        origin = np.zeros((3, 1))
        _, ind = pp.utils.setmembership.ismember_rows(origin, decomp["points"])
        mesh_size_known[ind] = np.sqrt(3)

        corner = np.array([5, 0, 0]).reshape((3, 1))
        _, ind = pp.utils.setmembership.ismember_rows(
            corner, decomp["points"], sort=False
        )
        mesh_size_known[ind] = np.sqrt(2)

        self.assertTrue(np.all(np.isclose(mesh_size, mesh_size_known)))


def make_bucket_2d():
    """
    Helper function to obtain known quantities and the inputs for
    determine_mesh_size in 2d.
    """

    f_1 = np.array([[0.5, 5], [1.5, 1.5]])
    f_set = [f_1]
    mesh_size_min = 0.2
    mesh_size_frac = 1
    mesh_size_bound = 2
    domain = {"xmin": 0, "ymin": 0, "xmax": 5, "ymax": 5}
    bucket = pp.meshing.simplex_grid(
        f_set,
        domain=domain,
        mesh_size_frac=mesh_size_frac,
        mesh_size_min=mesh_size_min,
        mesh_size_bound=mesh_size_bound,
    )
    bucket.compute_geometry()
    return bucket


if __name__ == "__main__":
    unittest.main()
