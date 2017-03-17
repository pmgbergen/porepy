import numpy as np
from scipy import sparse as sps

from gridding.fractured import meshing, split_grid


def test_split_fracture():
    """ Check that no error messages are created in the process of creating a
    split_fracture.
    """

    f_1 = np.array([[-.8, .8, .8, -.8 ], [0, 0, 0, 0], [-.8, -.8, .8, .8]])
    f_2 = np.array([[0, 0, 0, 0], [-.8, .8, .8, -.8 ], [-.8, -.8, .8, .8]])


    f_set = [f_1, f_2]
    domain = {'xmin': -1, 'xmax': 1,
              'ymin': -1, 'ymax': 1, 'zmin': -1, 'zmax': 1}
    # ENDRE DENNE
    path_to_gmsh = '~/gmsh/bin/gmsh'

    bucket = meshing.create_grid(f_set, domain, gmsh_path=path_to_gmsh)

    [g.compute_geometry(is_embedded=True) for g,_ in bucket]

    split_grid.split_fractures(bucket, offset=0)
    [g.compute_geometry(is_embedded=True) for g,_ in bucket]


if __name__ == '__main__':
    test_split_fracture()

