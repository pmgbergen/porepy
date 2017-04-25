import numpy as np
import os

from porepy.grids.gmsh import fractured_domain_2d, mesh_io


def test_gmsh_2d_crossing_fractures():
    """ Check that no error messages are created in the process of creating a
    gmsh geo file, running gmsh, and returning.

    Note that, for now, the path to the gmsh file is hard-coded into
    gridding.gmsh.fractured_domain . Any changes here will lead to errors.
    """
    p = np.array([[-1, 1, 0, 0],
                  [0, 0, -1, 1]])
    lines = np.array([[0, 2],
                      [1, 3]])

    fracs = {'points': p, 'edges': lines}

    box = {'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2, 'lcar': 0.7}

    filename = 'test_gmsh_2d_crossing_fractures'
    fractured_domain_2d.generate_grid(fracs, box, filename)
    geo_file = filename + '.geo'
    os.remove(geo_file)
    msh_file = filename + '.msh'
    point, cells, phys_names, cell_info = mesh_io.read(msh_file)
    os.remove(msh_file)

if __name__ == '__main__':
    test_gmsh_2d_crossing_fractures()
