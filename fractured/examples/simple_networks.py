"""
Various method for creating grids for relatively simple fracture networks.

The module doubles as a test framework (though not unittest), and will report
on any problems if ran as a main method.

"""
import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt
import time
import traceback
import logging
from inspect import isfunction, getmembers

from gridding.fractured import meshing
from viz import plot_grid


def single_isolated_fracture(**kwargs):
    """
    A single fracture completely immersed in a boundary grid.
    """
    f_1 = np.array([[-1, 1, 1, -1 ], [0, 0, 0, 0], [-1, -1, 1, 1]])
    domain = {'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2, 'zmin': -2, 'zmax':
              2}
    grids = meshing.create_grid([f_1], domain, **kwargs)

    if kwargs.get('return_expected', False):
        expected = [1, 1, 0, 0]
        return grids, expected
    else:
        return grids


def two_intersecting_fractures(**kwargs):
    """
    Two fractures intersecting along a line.

    The example also sets different characteristic mesh lengths on the boundary
    and at the fractures.

    """

    f_1 = np.array([[-1, 1, 1, -1 ], [0, 0, 0, 0], [-1, -1, 1, 1]])
    f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1 ], [-.7, -.7, .8, .8]])
    domain = {'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2, 'zmin': -2, 'zmax':
              2}

    mesh_size = {'mode': 'constant', 'value': 0.5, 'bound_value': 1}
    kwargs['mesh_size'] = mesh_size

    grids = meshing.create_grid([f_1, f_2], domain, **kwargs)

    if kwargs.get('return_expected', False):
        return grids, [1, 2, 1, 0]
    else:
        return grids
    


def three_intersecting_fractures(**kwargs):
    """
    Three fractures intersecting, with intersecting intersections (point)
    """

    f_1 = np.array([[-1, 1, 1, -1 ], [0, 0, 0, 0], [-1, -1, 1, 1]])
    f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1 ], [-.7, -.7, .8, .8]])
    f_3 = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1], [0, 0, 0, 0]])

    # Add some parameters for grid size
    domain = {'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2, 'zmin': -2, 'zmax':
              2}
    mesh_size = {'mode': 'constant', 'value': 0.5, 'bound_value': 1}

    kwargs['mesh_size'] = mesh_size
    grids = meshing.create_grid([f_1, f_2, f_3], domain, **kwargs)

    if kwargs.get('return_expected', False):
        return grids, [1, 3, 6, 1]
    else:
        return grids


def one_fracture_intersected_by_two(**kwargs):
    """
    One fracture, intersected by two other (but no point intersections)
    """

    f_1 = np.array([[-1, 1, 1, -1 ], [0, 0, 0, 0], [-1, -1, 1, 1]])
    f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1 ], [-.7, -.7, .8, .8]])
    f_3 = f_2 + np.array([0.5, 0, 0]).reshape((-1, 1))

    # Add some parameters for grid size
    domain = {'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2, 'zmin': -2, 'zmax':
              2}
    mesh_size = {'mode': 'constant', 'value': 0.5, 'bound_value': 1}

    kwargs['mesh_size'] = mesh_size
    grids = meshing.create_grid([f_1, f_2, f_3], domain, **kwargs)

    if kwargs.get('return_expected', False):
        return grids, [1, 3, 2, 0]
    else:
        return grids
    

def split_into_octants(**kwargs):
    f_1 = np.array([[-1, 1, 1, -1 ], [0, 0, 0, 0], [-1, -1, 1, 1]])
    f_2 = np.array([[-1, -1, 1, 1 ], [-1, 1, 1, -1], [0, 0, 0, 0]])
    f_3 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1 ], [-1, -1, 1, 1]])
    domain = {'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2, 'zmin': -2, 'zmax':
              2}
    grids = meshing.create_grid([f_1, f_2, f_3], domain, **kwargs)

    if kwargs.get('return_expected', False):
        return grids, [1, 3, 6, 1]
    else:
        return grids
    return grids


def three_fractures_sharing_line_same_segment(**kwargs):
    """
    Three fractures that all share an intersection line. This can be considered
    as three intersection lines that coincide.
    """
    f_1 = np.array([[-1, 1, 1, -1 ], [0, 0, 0, 0], [-1, -1, 1, 1]])
    f_2 = np.array([[-1, 1, 1, -1 ], [-1, 1, 1, -1], [-1, -1, 1, 1]])
    f_3 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1 ], [-1, -1, 1, 1]])
    domain = {'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2, 'zmin': -2, 'zmax':
              2}
    grids = meshing.create_grid([f_1, f_2, f_3], domain, **kwargs)

    if kwargs.get('return_expected', False):
        return grids, [1, 3, 1, 0]
    else:
        return grids


def three_fractures_split_segments(**kwargs):
    """
    Three fractures that all intersect along the same line, but with the
    intersection between two of them forming an extension of the intersection
    of all three.
    """
    f_1 = np.array([[-1, 1, 1, -1 ], [0, 0, 0, 0], [-1, -1, 1, 1]])
    f_2 = np.array([[-1, 1, 1, -1 ], [-1, 1, 1, -1], [-.5, -.5, .5, .5]])
    f_3 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1 ], [-1, -1, 1, 1]])
    domain = {'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2, 'zmin': -2, 'zmax':
              2}
    grids = meshing.create_grid([f_1, f_2, f_3], domain, **kwargs)
    if kwargs.get('return_expected', False):
        return grids, [1, 3, 3, 2]
    else:
        return grids



def two_fractures_L_intersection(**kwargs):
    """
    Two fractures sharing a segment in an L-intersection.
    """
    f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
    f_2 = np.array([[0, 0, 0, 0], [0, 1, 1, 0 ], [0, 0, 1, 1]])
    domain = {'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2, 'zmin': -2, 'zmax':
              2}
    grids = meshing.create_grid([f_1, f_2], domain, **kwargs)
    if kwargs.get('return_expected', False):
        return grids, [1, 2, 1, 0]
    else:
        return grids


def two_fractures_L_intersection_part_of_segment(**kwargs):
    """
    Two fractures sharing what is a full segment for one, an part of a segment
    for the other.
    """
    f_1 = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
    f_2 = np.array([[0, 0, 0, 0], [0.3, 0.7, 0.7, 0.3 ], [0, 0, 1, 1]])
    domain = {'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2, 'zmin': -2, 'zmax':
              2}
    grids = meshing.create_grid([f_1, f_2], domain, **kwargs)
    if kwargs.get('return_expected', False):
        return grids, [1, 2, 1, 0]
    else:
        return grids


if __name__ == '__main__':


    # If invoked as main, run all tests
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'v:', ['gmsh_path=',
                                                       'verbose=',
                                                        'compute_geometry='])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    gmsh_path = None
    verbose = 0
    compute_geometry=True
    # process options
    for o, a in opts:
        if o == '--gmsh_path':
            gmsh_path = a
        elif o in ('-v', '--verbose'):
            verbose = int(a)
        elif o == '--compute_geometry':
            compute_geometry=True

    return_expected = 1

    success_counter = 0
    failure_counter = 0

    time_tot = time.time()
    functions_list = [o for o in getmembers(sys.modules[__name__]) if isfunction(o[1])]
    for f in functions_list:
        func = f
        if func[0] == 'isfunction':
            continue
        elif func[0] == 'getmembers':
            continue
        if verbose > 0:
            print('Running ' + func[0])

        time_loc = time.time()
        try:
            g, expected = func[1](gmsh_path=gmsh_path,
                                               verbose=verbose,
                                               gmsh_verbose=0,
                                               return_expected=True)
            assert len(g) == 4
            for gl, e in zip(g, expected):
                assert len(gl) == e
            success_counter += 1
        except Exception as exp:
            print('\n')
            print(' ************** FAILURE **********')
            print('Example ' + func[0] + ' failed')
            print(exp)
            logging.error(traceback.format_exc())
            failure_counter += 1

    #################################
    # Summary
    #
    print('\n')
    print(' --- ')
    print('Ran in total ' + str(success_counter + failure_counter) + ' tests,' \
         +' out of which ' + str(failure_counter) + ' failed.')
    print('Total elapsed time is ' + str(time.time() - time_tot) + ' seconds')
    print('\n')


