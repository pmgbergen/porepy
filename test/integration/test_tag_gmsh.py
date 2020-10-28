import numpy as np
import unittest

import porepy as pp

"""
In this test we validate the propagation of physical tags from gmsh to porepy.
We consider the case with only boundary, fractures, auxiliary segments, and a mixed of them.

"""


class BasicsTest(unittest.TestCase):
    def test_boundary(self):
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
        network = pp.FractureNetwork2d(domain=domain)
        mesh_args = {"mesh_size_frac": 1}
        gb = network.mesh(mesh_args)

        g = gb.grids_of_dimension(2)[0]

        tag = np.where(g.tags["domain_boundary_0_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 0))

        tag = np.where(g.tags["domain_boundary_1_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 1))

        tag = np.where(g.tags["domain_boundary_2_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 1))

        tag = np.where(g.tags["domain_boundary_3_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 0))

    def test_boundary_refined(self):
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
        network = pp.FractureNetwork2d(domain=domain)
        mesh_args = {"mesh_size_frac": 1}
        gb = network.mesh(mesh_args)

        g = gb.grids_of_dimension(2)[0]

        tag = np.where(g.tags["domain_boundary_0_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 0))

        tag = np.where(g.tags["domain_boundary_1_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 1))

        tag = np.where(g.tags["domain_boundary_2_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 1))

        tag = np.where(g.tags["domain_boundary_3_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 0))

    def test_auxiliary(self):
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
        mesh_args = {"mesh_size_frac": 1}
        subdomain_start = np.array([[0.25], [0.5]])
        subdomain_end = np.array([[0.5], [0.5]])
        p = np.hstack((subdomain_start, subdomain_end))
        e = np.array([[0], [1]])

        network = pp.FractureNetwork2d(p, e, domain=domain)
        mesh_args = {"mesh_size_frac": 1}

        gb = network.mesh(mesh_args, constraints=np.array([0]))
        g = gb.grids_of_dimension(2)[0]

        tag = np.where(g.tags["domain_boundary_1_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 0))

        tag = np.where(g.tags["domain_boundary_2_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 1))

        tag = np.where(g.tags["domain_boundary_3_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 1))

        tag = np.where(g.tags["domain_boundary_4_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 0))

        fc = g.face_centers[:2]

        tag = g.tags["auxiliary_0_faces"]
        dist, _ = pp.distances.points_segments(fc, subdomain_start, subdomain_end)

        self.assertTrue(np.allclose(dist[tag, 0], 0))
        self.assertTrue(
            np.all(np.logical_not(np.allclose(dist[np.logical_not(tag), 0], 0)))
        )

    def test_auxiliary_refined(self):
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
        network = pp.FractureNetwork2d(domain=domain)
        mesh_args = {"mesh_size_frac": 0.33}

        subdomain_start = np.array([[0.25], [0.5]])
        subdomain_end = np.array([[0.5], [0.5]])

        p = np.hstack((subdomain_start, subdomain_end))
        e = np.array([[0], [1]])

        network = pp.FractureNetwork2d(p, e, domain=domain)
        gb = network.mesh(mesh_args, constraints=np.array([0]))
        g = gb.grids_of_dimension(2)[0]

        tag = np.where(g.tags["domain_boundary_1_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 0))

        tag = np.where(g.tags["domain_boundary_2_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 1))

        tag = np.where(g.tags["domain_boundary_3_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 1))

        tag = np.where(g.tags["domain_boundary_4_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 0))

        fc = g.face_centers[:2]

        tag = g.tags["auxiliary_0_faces"]
        dist, _ = pp.distances.points_segments(fc, subdomain_start, subdomain_end)

        self.assertTrue(np.allclose(dist[tag, 0], 0))
        self.assertTrue(
            np.all(np.logical_not(np.allclose(dist[np.logical_not(tag), 0], 0)))
        )

    def test_auxiliary_2(self):
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

        constraint_start_0 = np.array([[0.0], [0.5]])
        constraint_end_0 = np.array([[0.75], [0.5]])
        constraint_start_1 = np.array([[0.5], [0.25]])
        constraint_end_1 = np.array([[0.5], [0.75]])

        p = np.hstack(
            (constraint_start_0, constraint_end_0, constraint_start_1, constraint_end_1)
        )
        e = np.array([[0, 2], [1, 3]])

        network = pp.FractureNetwork2d(p, e, domain=domain)
        mesh_args = {"mesh_size_frac": 1}
        gb = network.mesh(mesh_args, constraints=np.arange(2))
        g = gb.grids_of_dimension(2)[0]

        tag = np.where(g.tags["domain_boundary_2_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 0))

        tag = np.where(g.tags["domain_boundary_3_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 1))

        tag = np.where(g.tags["domain_boundary_4_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 1))

        tag = np.where(g.tags["domain_boundary_5_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 0))

        fc = g.face_centers[:2]

        tag = g.tags["auxiliary_0_faces"]
        dist, _ = pp.distances.points_segments(fc, constraint_start_0, constraint_end_0)

        self.assertTrue(np.allclose(dist[tag, 0], 0))
        self.assertTrue(
            np.all(np.logical_not(np.allclose(dist[np.logical_not(tag), 0], 0)))
        )

        tag = g.tags["auxiliary_1_faces"]
        dist, _ = pp.distances.points_segments(fc, constraint_start_1, constraint_end_1)

        self.assertTrue(np.allclose(dist[tag, 0], 0))
        self.assertTrue(
            np.all(np.logical_not(np.allclose(dist[np.logical_not(tag), 0], 0)))
        )

    def test_auxiliary_2_refined(self):
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
        mesh_args = {"mesh_size_frac": 0.125}

        constraint_start_0 = np.array([[0.0], [0.5]])
        constraint_end_0 = np.array([[0.75], [0.5]])
        constraint_start_1 = np.array([[0.5], [0.25]])
        constraint_end_1 = np.array([[0.5], [0.75]])

        p = np.hstack(
            (constraint_start_0, constraint_end_0, constraint_start_1, constraint_end_1)
        )
        e = np.array([[0, 2], [1, 3]])

        network = pp.FractureNetwork2d(p, e, domain=domain)
        mesh_args = {"mesh_size_frac": 1}
        gb = network.mesh(mesh_args, constraints=np.array([0, 1]))
        g = gb.grids_of_dimension(2)[0]

        tag = np.where(g.tags["domain_boundary_2_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 0))

        tag = np.where(g.tags["domain_boundary_3_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 1))

        tag = np.where(g.tags["domain_boundary_4_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 1))

        tag = np.where(g.tags["domain_boundary_5_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 0))

        fc = g.face_centers[:2]

        tag = g.tags["auxiliary_0_faces"]
        dist, _ = pp.distances.points_segments(fc, constraint_start_0, constraint_end_0)

        self.assertTrue(np.allclose(dist[tag, 0], 0))
        self.assertTrue(
            np.all(np.logical_not(np.allclose(dist[np.logical_not(tag), 0], 0)))
        )

        tag = g.tags["auxiliary_1_faces"]
        dist, _ = pp.distances.points_segments(fc, constraint_start_1, constraint_end_1)

        self.assertTrue(np.allclose(dist[tag, 0], 0))
        self.assertTrue(
            np.all(np.logical_not(np.allclose(dist[np.logical_not(tag), 0], 0)))
        )

    def test_fracture(self):
        frac_start = np.array([[0.25], [0.5]])
        frac_end = np.array([[0.75], [0.5]])

        gb, _ = pp.grid_buckets_2d.single_horizontal(
            {"mesh_size_frac": 1}, x_endpoints=[0.25, 0.75]
        )
        g = gb.grids_of_dimension(2)[0]

        tag = np.where(g.tags["domain_boundary_1_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 0))

        tag = np.where(g.tags["domain_boundary_2_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 1))

        tag = np.where(g.tags["domain_boundary_3_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 1))

        tag = np.where(g.tags["domain_boundary_4_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 0))

        fc = g.face_centers[:2]
        tag = g.tags["fracture_0_faces"]

        dist, _ = pp.distances.points_segments(fc, frac_start, frac_end)

        self.assertTrue(np.allclose(dist[tag, 0], 0))
        self.assertTrue(
            np.all(np.logical_not(np.allclose(dist[np.logical_not(tag), 0], 0)))
        )

    def test_fracture_refined(self):
        frac_start = np.array([[0.25], [0.5]])
        frac_end = np.array([[0.75], [0.5]])

        gb, _ = pp.grid_buckets_2d.single_horizontal(
            {"mesh_size_frac": 0.125}, x_endpoints=[0.25, 0.75]
        )

        g = gb.grids_of_dimension(2)[0]

        tag = np.where(g.tags["domain_boundary_1_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 0))

        tag = np.where(g.tags["domain_boundary_2_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 1))

        tag = np.where(g.tags["domain_boundary_3_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 1))

        tag = np.where(g.tags["domain_boundary_4_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 0))

        fc = g.face_centers[:2]
        tag = g.tags["fracture_0_faces"]

        dist, _ = pp.distances.points_segments(fc, frac_start, frac_end)

        self.assertTrue(np.allclose(dist[tag, 0], 0))
        self.assertTrue(
            np.all(np.logical_not(np.allclose(dist[np.logical_not(tag), 0], 0)))
        )

    def test_fracture_2(self):
        frac_start_0 = np.array([[0.0], [0.5]])
        frac_end_0 = np.array([[0.75], [0.5]])
        frac_start_1 = np.array([[0.5], [0.25]])
        frac_end_1 = np.array([[0.5], [0.75]])

        gb, _ = pp.grid_buckets_2d.two_intersecting(
            {"mesh_size_frac": 1}, x_endpoints=[0.0, 0.75], y_endpoints=[0.25, 0.75]
        )
        g = gb.grids_of_dimension(2)[0]

        tag = np.where(g.tags["domain_boundary_2_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 0))

        tag = np.where(g.tags["domain_boundary_3_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 1))

        tag = np.where(g.tags["domain_boundary_4_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 1))

        tag = np.where(g.tags["domain_boundary_5_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 0))

        fc = g.face_centers[:2]
        tag = g.tags["fracture_0_faces"]

        dist, _ = pp.distances.points_segments(fc, frac_start_0, frac_end_0)

        self.assertTrue(np.allclose(dist[tag, 0], 0))
        self.assertTrue(
            np.all(np.logical_not(np.allclose(dist[np.logical_not(tag), 0], 0)))
        )

        tag = g.tags["fracture_1_faces"]

        dist, _ = pp.distances.points_segments(fc, frac_start_1, frac_end_1)

        self.assertTrue(np.allclose(dist[tag, 0], 0))
        self.assertTrue(
            np.all(np.logical_not(np.allclose(dist[np.logical_not(tag), 0], 0)))
        )

    def test_fracture_auxiliary(self):
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

        frac_start = np.array([[0.5], [0.25]])
        frac_end = np.array([[0.5], [0.75]])

        constraint_start = np.array([[0.25], [0.5]])
        constraint_end = np.array([[0.5], [0.5]])

        p = np.hstack((frac_start, frac_end, constraint_start, constraint_end))
        e = np.array([[0, 2], [1, 3]])
        network = pp.FractureNetwork2d(p, e, domain)

        constraints = np.array([1])

        mesh_args = {"mesh_size_frac": 1}
        gb = network.mesh(mesh_args, constraints=constraints)
        g = gb.grids_of_dimension(2)[0]

        tag = np.where(g.tags["domain_boundary_2_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 0))

        tag = np.where(g.tags["domain_boundary_3_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 1))

        # domain_boundary_2 should be at y=1
        tag = np.where(g.tags["domain_boundary_4_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 1))

        tag = np.where(g.tags["domain_boundary_5_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 0))

        fc = g.face_centers[:2]
        tag = g.tags["fracture_0_faces"]

        dist, _ = pp.distances.points_segments(fc, frac_start, frac_end)

        self.assertTrue(np.allclose(dist[tag, 0], 0))
        self.assertTrue(
            np.all(np.logical_not(np.allclose(dist[np.logical_not(tag), 0], 0)))
        )

        tag = g.tags["auxiliary_1_faces"]
        dist, _ = pp.distances.points_segments(fc, constraint_start, constraint_end)

        self.assertTrue(np.allclose(dist[tag, 0], 0))
        self.assertTrue(
            np.all(np.logical_not(np.allclose(dist[np.logical_not(tag), 0], 0)))
        )

    def test_auxiliary_intersect(self):
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

        constraint_start_0 = np.array([[0.0], [0.5]])
        constraint_end_0 = np.array([[0.75], [0.5]])
        constraint_start_1 = np.array([[0.5], [0.25]])
        constraint_end_1 = np.array([[0.5], [0.75]])

        p = np.hstack(
            (constraint_start_0, constraint_end_0, constraint_start_1, constraint_end_1)
        )
        e = np.array([[0, 2], [1, 3]])

        network = pp.FractureNetwork2d(p, e, domain=domain)
        mesh_args = {"mesh_size_frac": 1}
        gb = network.mesh(mesh_args, constraints=np.array([0, 1]))
        g = gb.grids_of_dimension(2)[0]

        tag = np.where(g.tags["domain_boundary_2_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 0))

        tag = np.where(g.tags["domain_boundary_3_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 1))

        tag = np.where(g.tags["domain_boundary_4_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 1))

        tag = np.where(g.tags["domain_boundary_5_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 0))

        fc = g.face_centers[:2]

        tag = g.tags["auxiliary_0_faces"]
        dist, _ = pp.distances.points_segments(fc, constraint_start_0, constraint_end_0)

        self.assertTrue(np.allclose(dist[tag, 0], 0))
        self.assertTrue(
            np.all(np.logical_not(np.allclose(dist[np.logical_not(tag), 0], 0)))
        )

        tag = g.tags["auxiliary_1_faces"]
        dist, _ = pp.distances.points_segments(fc, constraint_start_1, constraint_end_1)

        self.assertTrue(np.allclose(dist[tag, 0], 0))
        self.assertTrue(
            np.all(np.logical_not(np.allclose(dist[np.logical_not(tag), 0], 0)))
        )


if __name__ == "__main__":
    unittest.main()
