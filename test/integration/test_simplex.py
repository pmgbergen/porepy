import unittest
import numpy as np

import porepy as pp


class TestFaceTags(unittest.TestCase):
    def test_face_tags_from_gmsh(self):
        """ Check that the faces has correct tags for a 2D grid.
        """
        p = np.array([[0, 1, 0.5, 0.5],
                      [0.5, 0.5, 0, 1]])
        e = np.array([[0, 2], [1, 3]])
        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
        network = pp.FractureNetwork2d(p, e, domain=domain)
        mesh_args = {
            'mesh_size_frac': 0.1,
            'mesh_size_bound': 0.1,
        }

        file_name = "mesh_simplex"
        in_file = network.prepare_for_gmsh(
            mesh_args, None, True, None, file_name, False
        )
        out_file = in_file[:-4] + ".msh"

        # Consider the dimension of the problem
        ndim = 2
        pp.grids.gmsh.gmsh_interface.run_gmsh(in_file, out_file, dim=ndim)

        grid_list = pp.fracs.simplex.triangle_grid_from_gmsh(
            out_file,
        )
        pp.meshing._tag_faces(grid_list, False)
        for grid_of_dim in grid_list:
            for g in grid_of_dim:
                g.compute_geometry()
                if g.dim == 2:
                    f_cc0 = g.face_centers[:, g.tags["fracture_0_faces"]]
                    f_cc1 = g.face_centers[:, g.tags["fracture_1_faces"]]

                    # Test position of the two fractures
                    self.assertTrue(np.allclose(f_cc0[1], 0.5))
                    self.assertTrue(np.allclose(f_cc1[0], 0.5))

                if g.dim > 0:
                    db_cc = g.face_centers[:, g.tags["domain_boundary_faces"]]
                    # Test that domain boundary faces are on the boundary
                    self.assertTrue(
                        np.all(
                            (np.abs(db_cc[0]) < 1e-10)
                            | (np.abs(db_cc[1]) < 1e-10)
                            | (np.abs(db_cc[0] - 1) < 1e-10)
                            | (np.abs(db_cc[1] - 1) < 1e-10)
                        )
                    )
                # No tip faces in this test case
                self.assertTrue(np.all(g.tags["tip_faces"] == 0))
