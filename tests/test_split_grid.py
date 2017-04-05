import unittest
import numpy as np

from gridding.fractured import meshing


class TestMeshing(unittest.TestCase):
    def test_x_intersection_3d(self):
        """ Check that no error messages are created in the process of creating a
        split_fracture.
        """

        f_1 = np.array([[-.8, .8, .8, -.8], [0, 0, 0, 0], [-.8, -.8, .8, .8]])
        f_2 = np.array([[0, 0, 0, 0], [-.8, .8, .8, -.8], [-.8, -.8, .8, .8]])

        f_set = [f_1, f_2]
        domain = {'xmin': -1, 'xmax': 1,
                  'ymin': -1, 'ymax': 1, 'zmin': -1, 'zmax': 1}
        # ENDRE DENNE
        path_to_gmsh = '~/gmsh/bin/gmsh'
        bucket = meshing.simplex_grid(f_set, domain, gmsh_path=path_to_gmsh)
        bucket.compute_geometry()
