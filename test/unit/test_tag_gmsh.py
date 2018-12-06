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
        fracs = []
        mesh_args = {"mesh_size_frac": 1}

        gb = pp.fracs.meshing.simplex_grid(fracs, domain=domain, **mesh_args)
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
        fracs = []
        mesh_args = {"mesh_size_frac": 0.5}

        gb = pp.fracs.meshing.simplex_grid(fracs, domain=domain, **mesh_args)
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
        fracs = []
        mesh_args = {"mesh_size_frac": 1}

        subdomain = {
            "points": np.array([[0, 0.5], [0.5, 0.5]]),
            "edges": np.array([0, 1]).reshape((2, -1), order="F"),
        }

        gb = pp.fracs.meshing.simplex_grid(
            fracs, domain=domain, subdomains=subdomain, **mesh_args
        )
        g = gb.grids_of_dimension(2)[0]

        tag = np.where(g.tags["domain_boundary_0_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 0))

        tag = np.where(g.tags["domain_boundary_1_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 1))

        tag = np.where(g.tags["domain_boundary_2_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 1))

        tag = np.where(g.tags["domain_boundary_3_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 0))

        known = [12]
        tag = np.where(g.tags["auxiliary_4_faces"])[0]
        self.assertTrue(np.allclose(known, tag))

    def test_auxiliary_refined(self):
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
        fracs = []
        mesh_args = {"mesh_size_frac": 0.33}

        subdomain = {
            "points": np.array([[0, 0.5], [0.5, 0.5]]),
            "edges": np.array([0, 1]).reshape((2, -1), order="F"),
        }

        gb = pp.fracs.meshing.simplex_grid(
            fracs, domain=domain, subdomains=subdomain, **mesh_args
        )
        g = gb.grids_of_dimension(2)[0]

        tag = np.where(g.tags["domain_boundary_0_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 0))

        tag = np.where(g.tags["domain_boundary_1_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 1))

        tag = np.where(g.tags["domain_boundary_2_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 1))

        tag = np.where(g.tags["domain_boundary_3_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 0))

        known = [14, 17]
        tag = np.where(g.tags["auxiliary_4_faces"])[0]
        self.assertTrue(np.allclose(known, tag))

    def test_auxiliary_2(self):
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
        fracs = []
        mesh_args = {"mesh_size_frac": 1}

        subdomain = {
            "points": np.array([[0, 0.5, 0.75, 0.75], [0.5, 0.5, 0.25, 0.75]]),
            "edges": np.array([0, 1, 2, 3]).reshape((2, -1), order="F"),
        }

        gb = pp.fracs.meshing.simplex_grid(
            fracs, domain=domain, subdomains=subdomain, **mesh_args
        )
        g = gb.grids_of_dimension(2)[0]

        tag = np.where(g.tags["domain_boundary_0_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 0))

        tag = np.where(g.tags["domain_boundary_1_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 1))

        tag = np.where(g.tags["domain_boundary_2_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 1))

        tag = np.where(g.tags["domain_boundary_3_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 0))

        known = [12, 17]
        tag = np.where(g.tags["auxiliary_4_faces"])[0]
        self.assertTrue(np.allclose(known, tag))

        known = [27, 34]
        tag = np.where(g.tags["auxiliary_5_faces"])[0]
        self.assertTrue(np.allclose(known, tag))

    def test_auxiliary_2_refined(self):
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
        fracs = []
        mesh_args = {"mesh_size_frac": 0.125}

        subdomain = {
            "points": np.array([[0, 0.5, 0.75, 0.75], [0.5, 0.5, 0.25, 0.75]]),
            "edges": np.array([0, 1, 2, 3]).reshape((2, -1), order="F"),
        }

        gb = pp.fracs.meshing.simplex_grid(
            fracs, domain=domain, subdomains=subdomain, **mesh_args
        )
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
        fracs = {
            "points": np.array([[0.5, 0.5], [0.25, 0.75]]),
            "edges": np.array([0, 1]).reshape((2, -1), order="F"),
        }
        mesh_args = {"mesh_size_frac": 1}

        gb = pp.fracs.meshing.simplex_grid(fracs, domain=domain, **mesh_args)
        g = gb.grids_of_dimension(2)[0]

        tag = np.where(g.tags["domain_boundary_0_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 0))

        tag = np.where(g.tags["domain_boundary_1_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 1))

        tag = np.where(g.tags["domain_boundary_2_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 1))

        tag = np.where(g.tags["domain_boundary_3_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 0))

        known = [12, 17, 54, 55]
        tag = np.where(g.tags["fracture_4_faces"])[0]
        self.assertTrue(np.allclose(known, tag))

    def test_fracture_refined(self):
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
        fracs = {
            "points": np.array([[0.5, 0.5], [0.25, 0.75]]),
            "edges": np.array([0, 1]).reshape((2, -1), order="F"),
        }
        mesh_args = {"mesh_size_frac": 0.125}

        gb = pp.fracs.meshing.simplex_grid(fracs, domain=domain, **mesh_args)
        g = gb.grids_of_dimension(2)[0]

        tag = np.where(g.tags["domain_boundary_0_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 0))

        tag = np.where(g.tags["domain_boundary_1_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 1))

        tag = np.where(g.tags["domain_boundary_2_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 1))

        tag = np.where(g.tags["domain_boundary_3_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 0))

        known = [12, 17, 107, 114, 268, 269, 270, 271]
        tag = np.where(g.tags["fracture_4_faces"])[0]
        self.assertTrue(np.allclose(known, tag))

    def test_fracture_2(self):
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
        fracs = {
            "points": np.array([[0.5, 0.5, 0.25, 0.75], [0.25, 0.75, 0.5, 0.5]]),
            "edges": np.array([0, 1, 2, 3]).reshape((2, -1), order="F"),
        }
        mesh_args = {"mesh_size_frac": 1}

        gb = pp.fracs.meshing.simplex_grid(fracs, domain=domain, **mesh_args)
        g = gb.grids_of_dimension(2)[0]
        # Check the ordering of the 1d grids in the grid bucket. This affects the order
        # in which the 2d faces are split, and hence the id of the added faces.
        neigh = np.array(gb.node_neighbors(g))
        right_order = neigh[0].frac_num == 4

        tag = np.where(g.tags["domain_boundary_0_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 0))

        tag = np.where(g.tags["domain_boundary_1_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 1))

        tag = np.where(g.tags["domain_boundary_2_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 1))

        tag = np.where(g.tags["domain_boundary_3_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 0))

        known = [12, 19, 104, 105]
        if not right_order:
            known = [12, 19, 106, 107]
        tag = np.where(g.tags["fracture_4_faces"])[0]
        self.assertTrue(np.allclose(known, tag))

        known = [26, 33, 106, 107]
        if not right_order:
            known = [26, 33, 104, 105]
        tag = np.where(g.tags["fracture_5_faces"])[0]
        self.assertTrue(np.allclose(known, tag))

    def test_fracture_auxiliary(self):
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
        fracs = {
            "points": np.array([[0.5, 0.5], [0.25, 0.75]]),
            "edges": np.array([0, 1]).reshape((2, -1), order="F"),
        }
        mesh_args = {"mesh_size_frac": 1}

        subdomain = {
            "points": np.array([[0.25, 0.5], [0.5, 0.5]]),
            "edges": np.array([0, 1]).reshape((2, -1), order="F"),
        }

        gb = pp.fracs.meshing.simplex_grid(
            fracs, domain=domain, subdomains=subdomain, **mesh_args
        )
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

        known = [12, 18, 79, 80]
        tag = np.where(g.tags["fracture_4_faces"])[0]
        self.assertTrue(np.allclose(known, tag))

        known = [24]
        tag = np.where(g.tags["auxiliary_5_faces"])[0]
        self.assertTrue(np.allclose(known, tag))

    def test_auxiliary_intersect(self):
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
        fracs = []
        mesh_args = {"mesh_size_frac": 1}

        subdomain = {
            "points": np.array([[0, 0.75, 0.5, 0.5], [0.5, 0.5, 0.25, 0.75]]),
            "edges": np.array([0, 1, 2, 3]).reshape((2, -1), order="F"),
        }

        gb = pp.fracs.meshing.simplex_grid(
            fracs, domain=domain, subdomains=subdomain, **mesh_args
        )
        g = gb.grids_of_dimension(2)[0]

        tag = np.where(g.tags["domain_boundary_0_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 0))

        tag = np.where(g.tags["domain_boundary_1_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 1))

        tag = np.where(g.tags["domain_boundary_2_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[1, tag], 1))

        tag = np.where(g.tags["domain_boundary_3_faces"])[0]
        self.assertTrue(np.allclose(g.face_centers[0, tag], 0))

        known = [12, 15, 36]
        tag = np.where(g.tags["auxiliary_4_faces"])[0]
        self.assertTrue(np.allclose(known, tag))

        known = [22, 29]
        tag = np.where(g.tags["auxiliary_5_faces"])[0]
        self.assertTrue(np.allclose(known, tag))


if __name__ == "__main__":
    BasicsTest().test_fracture_auxiliary()
    unittest.main()
