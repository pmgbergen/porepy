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

    return grids

def split_into_octants(**kwargs):
    f_1 = np.array([[-1, 1, 1, -1 ], [0, 0, 0, 0], [-1, -1, 1, 1]])
    f_2 = np.array([[-1, -1, 1, 1 ], [-1, 1, 1, -1], [0, 0, 0, 0]])
    f_3 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1 ], [-1, -1, 1, 1]])
    domain = {'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2, 'zmin': -2, 'zmax':
              2}
    grids = meshing.create_grid([f_1, f_2, f_3], domain, **kwargs)

    return grids


def three_fractures_sharing_line_same_segment(**kwargs):
    """
    Three fractures that all share an intersection. This can be considered as
    three intersection lines that coincide.
    """
    f_1 = np.array([[-1, 1, 1, -1 ], [0, 0, 0, 0], [-1, -1, 1, 1]])
    f_2 = np.array([[-1, 1, 1, -1 ], [-1, 1, 1, -1], [-1, -1, 1, 1]])
    f_3 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1 ], [-1, -1, 1, 1]])
    domain = {'xmin': -2, 'xmax': 2, 'ymin': -2, 'ymax': 2, 'zmin': -2, 'zmax':
              2}
    grids = meshing.create_grid([f_1, f_2, f_3], domain, **kwargs)

    return grids



if __name__ == '__main__':
    # If invoked as main, run all tests
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'v:', ['gmsh_path=',
                                                       'verbose=',
                                                        'visualize='])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    gmsh_path = None
    verbose = 0
    # process options
    for o, a in opts:
        if o == '--gmsh_path':
            gmsh_path = a
        elif o in ('-v', '--verbose'):
            verbose = int(a)

    success_counter = 0
    failure_counter = 0

    time_tot = time.time()

    #######################
    # Single fracture
    if verbose > 0:
        print('Run single fracture example')
    try:
        time_loc = time.time()
        g = single_isolated_fracture(gmsh_path=gmsh_path, verbose=verbose,
                                     gmsh_verbose=0)
        assert len(g) == 4
        assert len(g[0]) == 1
        assert len(g[1]) == 1
        assert len(g[2]) == 0
        assert len(g[3]) == 0
        if verbose > 0:
            print('Single fracture example completed successfully')
            print('Elapsed time ' + str(time.time() - time_loc))
        success_counter += 1
    except Exception as exp:
        print('\n')
        print(' ***** FAILURE ****')
        print('Gridding of single isolated fracture failed')
        print(exp)
        logging.error(traceback.format_exc())
        failure_counter += 1

    ##########################
    # Two fractures, one intersection
    #
    if verbose > 0:
        print('Run two intersecting fractures example')
    try:
        time_loc = time.time()
        g = two_intersecting_fractures(gmsh_path=gmsh_path, verbose=verbose,
                                     gmsh_verbose=0)
        assert len(g) == 4
        assert len(g[0]) == 1
        assert len(g[1]) == 2
        assert len(g[2]) == 1
        assert len(g[3]) == 0
        if verbose > 0:
            print('Two fractures example completed successfully')
            print('Elapsed time ' + str(time.time() - time_loc))
        success_counter += 1
    except Exception as exp:
        print('\n')
        print(' ***** FAILURE ****')
        print('Gridding of two intersecting fractures failed')
        print(exp)
        logging.error(traceback.format_exc())
        failure_counter += 1

    ##########################
    # Three fractures, three intersection lines
    #
    if verbose > 0:
        print('Run three intersecting fractures example')
    try:
        time_loc = time.time()
        g = three_intersecting_fractures(gmsh_path=gmsh_path, verbose=verbose,
                                     gmsh_verbose=0)
        assert len(g) == 4
        assert len(g[0]) == 1
        assert len(g[1]) == 3
        assert len(g[2]) == 6
        assert len(g[3]) == 1
        if verbose > 0:
            print('Three fractures example completed successfully')
            print('Elapsed time ' + str(time.time() - time_loc))
        success_counter += 1
    except Exception as exp:
        print('\n')
        print(' ***** FAILURE ****')
        print('Gridding of three intersecting fractures failed')
        print(exp)
        logging.error(traceback.format_exc())
        failure_counter += 1

    ##########################
    # Three fractures, two separate intersection lines
    #
    if verbose > 0:
        print('Run one fracture intersected by two')
    try:
        time_loc = time.time()
        g = one_fracture_intersected_by_two(gmsh_path=gmsh_path,
                                            verbose=verbose, gmsh_verbose=0)
        assert len(g) == 4
        assert len(g[0]) == 1
        assert len(g[1]) == 3
        assert len(g[2]) == 2
        assert len(g[3]) == 0
        if verbose > 0:
            print('One by two completed successfully')
            print('Elapsed time ' + str(time.time() - time_loc))
        success_counter += 1
    except Exception as exp:
        print('\n')
        print(' ***** FAILURE ****')
        print('Gridding of one by two failed')
        print(exp)
        logging.error(traceback.format_exc())
        failure_counter += 1

    ##########################
    # Three fractures, splits the domain into octants
    #
    if verbose > 0:
        print('Run one fracture intersected by two')
    try:
        time_loc = time.time()
        g = split_into_octants(gmsh_path=gmsh_path,
                                            verbose=verbose, gmsh_verbose=0)
        assert len(g) == 4
        assert len(g[0]) == 1
        assert len(g[1]) == 3
        assert len(g[2]) == 6
        assert len(g[3]) == 1
        if verbose > 0:
            print('Into octants completed successfully')
            print('Elapsed time ' + str(time.time() - time_loc))
        success_counter += 1
    except Exception as exp:
        print('\n')
        print(' ***** FAILURE ****')
        print('Gridding of into octants failed')
        print(exp)
        logging.error(traceback.format_exc())
        failure_counter += 1

    ###############################
    # Three fractures meeting along a line
    #
    if verbose > 0:
        print('Run three fractures intersecting along a line')
    try:
        time_loc = time.time()
        g = three_fractures_sharing_line_same_segment(gmsh_path=gmsh_path,
                                            verbose=verbose, gmsh_verbose=0)
        assert len(g) == 4
        assert len(g[0]) == 1
        assert len(g[1]) == 3
        assert len(g[2]) == 1
        assert len(g[3]) == 0
        if verbose > 0:
            print('One by two completed successfully')
            print('Elapsed time ' + str(time.time() - time_loc))
        success_counter += 1
    except Exception as exp:
        print('\n')
        print(' ***** FAILURE ****')
        print('Gridding of one by two failed')
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


