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

        subdomain = {
            "points": np.hstack((subdomain_start, subdomain_end)),
            "edges": np.array([0, 1]).reshape((2, -1), order="F"),
        }
        network = pp.FractureNetwork2d(domain=domain)
        mesh_args = {"mesh_size_frac": 1}

        gb = network.mesh(mesh_args, constraints=subdomain)
        g = gb.grids_of_dimension(2)[0]

        tag = np.where(g.tags["domain_boundary_0_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 0))

        tag = np.where(g.tags["domain_boundary_1_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 1))

        tag = np.where(g.tags["domain_boundary_2_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 1))

        tag = np.where(g.tags["domain_boundary_3_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 0))

        fc = g.face_centers[:2]

        tag = g.tags["auxiliary_4_faces"]
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

        subdomain = {
            "points": np.hstack((subdomain_start, subdomain_end)),
            "edges": np.array([0, 1]).reshape((2, -1), order="F"),
        }

        gb = network.mesh(mesh_args, constraints=subdomain)
        g = gb.grids_of_dimension(2)[0]

        tag = np.where(g.tags["domain_boundary_0_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 0))

        tag = np.where(g.tags["domain_boundary_1_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 1))

        tag = np.where(g.tags["domain_boundary_2_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 1))

        tag = np.where(g.tags["domain_boundary_3_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 0))

        fc = g.face_centers[:2]

        tag = g.tags["auxiliary_4_faces"]
        dist, _ = pp.distances.points_segments(fc, subdomain_start, subdomain_end)

        self.assertTrue(np.allclose(dist[tag, 0], 0))
        self.assertTrue(
            np.all(np.logical_not(np.allclose(dist[np.logical_not(tag), 0], 0)))
        )

    def test_auxiliary_2(self):
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
        network = pp.FractureNetwork2d(domain=domain)
        mesh_args = {"mesh_size_frac": 1}

        subdomain_start_0 = np.array([[0.0], [0.5]])
        subdomain_end_0 = np.array([[0.75], [0.5]])
        subdomain_start_1 = np.array([[0.5], [0.25]])
        subdomain_end_1 = np.array([[0.5], [0.75]])

        subdomain = {
            "points": np.hstack(
                (subdomain_start_0, subdomain_end_0, subdomain_start_1, subdomain_end_1)
            ),
            "edges": np.array([0, 1, 2, 3]).reshape((2, -1), order="F"),
        }

        gb = network.mesh(mesh_args, constraints=subdomain)
        g = gb.grids_of_dimension(2)[0]

        tag = np.where(g.tags["domain_boundary_0_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 0))

        tag = np.where(g.tags["domain_boundary_1_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 1))

        tag = np.where(g.tags["domain_boundary_2_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 1))

        tag = np.where(g.tags["domain_boundary_3_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 0))

        fc = g.face_centers[:2]

        tag = g.tags["auxiliary_4_faces"]
        dist, _ = pp.distances.points_segments(fc, subdomain_start_0, subdomain_end_0)

        self.assertTrue(np.allclose(dist[tag, 0], 0))
        self.assertTrue(
            np.all(np.logical_not(np.allclose(dist[np.logical_not(tag), 0], 0)))
        )

        tag = g.tags["auxiliary_5_faces"]
        dist, _ = pp.distances.points_segments(fc, subdomain_start_1, subdomain_end_1)

        self.assertTrue(np.allclose(dist[tag, 0], 0))
        self.assertTrue(
            np.all(np.logical_not(np.allclose(dist[np.logical_not(tag), 0], 0)))
        )

    def test_auxiliary_2_refined(self):
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
        network = pp.FractureNetwork2d(domain=domain)
        mesh_args = {"mesh_size_frac": 0.125}

        subdomain = {
            "points": np.array([[0, 0.5, 0.75, 0.75], [0.5, 0.5, 0.25, 0.75]]),
            "edges": np.array([0, 1, 2, 3]).reshape((2, -1), order="F"),
        }

        gb = network.mesh(mesh_args, constraints=subdomain)
        g = gb.grids_of_dimension(2)[0]

        tag = np.where(g.tags["domain_boundary_0_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 0))

        tag = np.where(g.tags["domain_boundary_1_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 1))

        tag = np.where(g.tags["domain_boundary_2_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 1))

        tag = np.where(g.tags["domain_boundary_3_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 0))

        known = [14, 17, 112, 118]
        tag = np.where(g.tags["auxiliary_4_faces"])[0]
        self.assertTrue(np.allclose(known, tag))

        known = [22, 29, 128, 134]
        tag = np.where(g.tags["auxiliary_5_faces"])[0]
        self.assertTrue(np.allclose(known, tag))

    def test_fracture(self):
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

        frac_start = np.array([[0.5], [0.25]])
        frac_end = np.array([[0.5], [0.75]])

        p = np.hstack((frac_start, frac_end))

        e = np.array([0, 1]).reshape((2, -1), order="F")
        network = pp.FractureNetwork2d(p, e, domain)
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

        fc = g.face_centers[:2]
        tag = g.tags["fracture_4_faces"]

        dist, _ = pp.distances.points_segments(fc, frac_start, frac_end)

        self.assertTrue(np.allclose(dist[tag, 0], 0))
        self.assertTrue(
            np.all(np.logical_not(np.allclose(dist[np.logical_not(tag), 0], 0)))
        )

    def test_fracture_refined(self):
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
        frac_start = np.array([[0.5], [0.25]])
        frac_end = np.array([[0.5], [0.75]])

        p = np.hstack((frac_start, frac_end))

        e = np.array([0, 1]).reshape((2, -1), order="F")
        network = pp.FractureNetwork2d(p, e, domain)

        mesh_args = {"mesh_size_frac": 0.125}

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

        fc = g.face_centers[:2]
        tag = g.tags["fracture_4_faces"]

        dist, _ = pp.distances.points_segments(fc, frac_start, frac_end)

        self.assertTrue(np.allclose(dist[tag, 0], 0))
        self.assertTrue(
            np.all(np.logical_not(np.allclose(dist[np.logical_not(tag), 0], 0)))
        )

    def test_fracture_2(self):
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

        frac_start_0 = np.array([[0.0], [0.5]])
        frac_end_0 = np.array([[0.75], [0.5]])
        frac_start_1 = np.array([[0.5], [0.25]])
        frac_end_1 = np.array([[0.5], [0.75]])

        p = np.hstack((frac_start_0, frac_end_0, frac_start_1, frac_end_1))
        e = np.array([0, 1, 2, 3]).reshape((2, -1), order="F")
        network = pp.FractureNetwork2d(p, e, domain)

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

        fc = g.face_centers[:2]
        tag = g.tags["fracture_4_faces"]

        dist, _ = pp.distances.points_segments(fc, frac_start_0, frac_end_0)

        self.assertTrue(np.allclose(dist[tag, 0], 0))
        self.assertTrue(
            np.all(np.logical_not(np.allclose(dist[np.logical_not(tag), 0], 0)))
        )

        tag = g.tags["fracture_5_faces"]

        dist, _ = pp.distances.points_segments(fc, frac_start_1, frac_end_1)

        self.assertTrue(np.allclose(dist[tag, 0], 0))
        self.assertTrue(
            np.all(np.logical_not(np.allclose(dist[np.logical_not(tag), 0], 0)))
        )

    def test_fracture_auxiliary(self):
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

        frac_start = np.array([[0.5], [0.25]])
        frac_end = np.array([[0.5], [0.75]])

        p = np.hstack((frac_start, frac_end))
        e = np.array([0, 1]).reshape((2, -1), order="F")
        network = pp.FractureNetwork2d(p, e, domain)

        mesh_args = {"mesh_size_frac": 1}

        subdomain_start = np.array([[0.25], [0.5]])
        subdomain_end = np.array([[0.5], [0.5]])

        subdomain = {
            "points": np.hstack((subdomain_start, subdomain_end)),
            "edges": np.array([0, 1]).reshape((2, -1), order="F"),
        }

        gb = network.mesh(mesh_args, constraints=subdomain)
        g = gb.grids_of_dimension(2)[0]

        tag = np.where(g.tags["domain_boundary_0_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 0))

        tag = np.where(g.tags["domain_boundary_1_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 1))

        # domain_boundary_2 should be at y=1
        tag = np.where(g.tags["domain_boundary_2_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 1))

        tag = np.where(g.tags["domain_boundary_3_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 0))

        fc = g.face_centers[:2]
        tag = g.tags["fracture_4_faces"]

        dist, _ = pp.distances.points_segments(fc, frac_start, frac_end)

        self.assertTrue(np.allclose(dist[tag, 0], 0))
        self.assertTrue(
            np.all(np.logical_not(np.allclose(dist[np.logical_not(tag), 0], 0)))
        )

        tag = g.tags["auxiliary_5_faces"]
        dist, _ = pp.distances.points_segments(fc, subdomain_start, subdomain_end)

        self.assertTrue(np.allclose(dist[tag, 0], 0))
        self.assertTrue(
            np.all(np.logical_not(np.allclose(dist[np.logical_not(tag), 0], 0)))
        )

    def test_auxiliary_intersect(self):
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
        network = pp.FractureNetwork2d(domain=domain)
        mesh_args = {"mesh_size_frac": 1}

        subdomain_start_0 = np.array([[0.0], [0.5]])
        subdomain_end_0 = np.array([[0.75], [0.5]])
        subdomain_start_1 = np.array([[0.5], [0.25]])
        subdomain_end_1 = np.array([[0.5], [0.75]])

        subdomain = {
            "points": np.hstack(
                (subdomain_start_0, subdomain_end_0, subdomain_start_1, subdomain_end_1)
            ),
            "edges": np.array([0, 1, 2, 3]).reshape((2, -1), order="F"),
        }

        gb = network.mesh(mesh_args, constraints=subdomain)
        g = gb.grids_of_dimension(2)[0]

        tag = np.where(g.tags["domain_boundary_0_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 0))

        tag = np.where(g.tags["domain_boundary_1_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 1))

        tag = np.where(g.tags["domain_boundary_2_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 1))

        tag = np.where(g.tags["domain_boundary_3_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 0))

        fc = g.face_centers[:2]

        tag = g.tags["auxiliary_4_faces"]
        dist, _ = pp.distances.points_segments(fc, subdomain_start_0, subdomain_end_0)

        self.assertTrue(np.allclose(dist[tag, 0], 0))
        self.assertTrue(
            np.all(np.logical_not(np.allclose(dist[np.logical_not(tag), 0], 0)))
        )

        tag = g.tags["auxiliary_5_faces"]
        dist, _ = pp.distances.points_segments(fc, subdomain_start_1, subdomain_end_1)

        self.assertTrue(np.allclose(dist[tag, 0], 0))
        self.assertTrue(
            np.all(np.logical_not(np.allclose(dist[np.logical_not(tag), 0], 0)))
        )


if __name__ == "__main__":
    unittest.main()
