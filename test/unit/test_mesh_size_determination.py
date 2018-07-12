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
        pts = np.array([[0., 5., 5., 0., 0.5, 4.], [0., 0., 5., 5., 1., 1.]])
        on_boundary = np.array([True, True, True, True, False, False])
        lines = np.array(
            [[0, 3, 1, 2, 4], [1, 0, 2, 3, 5], [1, 1, 1, 1, 3], [0, 3, 1, 2, 4]]
        )

        mesh_sizes_known = np.array(
            [1.11803399, 1.41421356, 2., 2., 0.5, 1., 0.5, 1., 1.80277564, 1., 1.]
        )
        pts_split_known = np.array(
            [
                [0., 5., 5., 0., 0.5, 4., 0., 2.5, 5., 0., 2.25],
                [0., 0., 5., 5., 1., 1., 1., 0., 2.5, 3., 1.],
            ]
        )

        mesh_size_min = .2
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

        assert np.all(np.isclose(mesh_sizes, mesh_sizes_known))
        assert np.all(np.isclose(pts_split, pts_split_known))

    def test_one_to_boundary_2d(self):
        """
        Fracture set:
            f_1 = np.array([[.5, 5], [1.5, 1.5]])

        """
        pts = np.array([[0., 5., 5., 0., 0.5, 5.], [0., 0., 5., 5., 1.5, 1.5]])
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
            [1.58113883, 1.5, 2., 2., 0.5, 1., 0.5, 1.5, 0.75, 1., 1.]
        )
        pts_split_known = np.array(
            [
                [0., 5., 5., 0., 0.5, 5., 0., 2.5, 0., 0., 2.75],
                [0., 0., 5., 5., 1.5, 1.5, 1.5, 0., 0.75, 3.25, 1.5],
            ]
        )

        mesh_size_min = .2
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

        assert np.all(np.isclose(mesh_sizes, mesh_sizes_known))
        assert np.all(np.isclose(pts_split, pts_split_known))

    def test_one_to_boundar_3d(self):
        """
            One fracture in 3d going all the way to the boundary
            """

        f_1 = np.array([[1, 5, 5, 1], [1, 1, 1, 1], [1, 1, 3, 3]])
        f_set = [pp.Fracture(f_1)]
        domain = {"xmin": 0, "ymin": 0, "zmin": 0, "xmax": 5, "ymax": 5, "zmax": 5}
        mesh_size_min = .1
        mesh_size_frac = .1
        mesh_size_bound = 2
        on_boundary = np.array(
            [False, False, False, False, True, True, True, True, True, True, True, True]
        )
        network = pp.FractureNetwork(f_set)
        network.impose_external_boundary(domain)
        network.find_intersections()
        network.split_intersections()
        network.insert_auxiliary_points(
            mesh_size_frac=mesh_size_frac,
            mesh_size_min=mesh_size_min,
            mesh_size_bound=mesh_size_bound,
        )
        mesh_size = network._determine_mesh_size(boundary_point_tags=on_boundary)
        # 0.1 corresponds to fracture corners, 2.0 to domain corners far
        # away from the fracture and the other values to domain corners
        # affected by the
        mesh_size_known = np.array(
            [0.1, 0.1, 0.1, 0.1, 2., 1.73205081, 2., 2., 2., 1.41421356, 2., 2.]
        )
        assert np.all(np.isclose(mesh_size, mesh_size_known))


def make_bucket_2d():
    """
        Helper function to obtain known quantities and the inputs for
        determine_mesh_size in 2d.
        """

    f_1 = np.array([[.5, 5], [1.5, 1.5]])
    f_set = [f_1]
    mesh_size_min = .2
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
