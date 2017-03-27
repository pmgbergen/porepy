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


def x_intersection_2d(**kwargs):
    f_1 = np.array([[2, 6], [5, 5]])
    f_2 = np.array([[4, 4], [2, 7]])
    f = [f_1,  f_2]

    gb = meshing.tensor_grid(f, [10, 10], physdims=[10, 10])
    if kwargs.get('return_expected', False):
        expected = [0, 1, 2, 1]
        return gb, expected
    else:
        return gb


def T_intersection_2d(**kwargs):
    f_1 = np.array([[2, 6], [5, 5]])
    f_2 = np.array([[4, 4], [2, 5]])
    f = [f_1,  f_2]

    gb = meshing.tensor_grid(f, [10, 10], physdims=[10, 10])
    if kwargs.get('return_expected', False):
        expected = [0, 1, 2, 1]
        return gb, expected
    else:
        return gb


def L_intersection_2d(**kwargs):
    f_1 = np.array([[2, 6], [5, 5]])
    f_2 = np.array([[6, 6], [2, 5]])
    f = [f_1,  f_2]

    gb = meshing.tensor_grid(f, [10, 10], physdims=[10, 10])
    if kwargs.get('return_expected', False):
        expected = [0, 1, 2, 1]
        return gb, expected
    else:
        return gb


def x_intersection_3d(**kwargs):
    f_1 = np.array([[2, 5, 5, 2], [2, 2, 5, 5], [5, 5, 5, 5]])
    f_2 = np.array([[2, 2, 5, 5], [5, 5, 5, 5], [2, 5, 5, 2]])
    f = [f_1, f_2]
    gb = meshing.tensor_grid(f, np.array([10, 10, 10]))
    if kwargs.get('return_expected', False):
        expected = [1, 2, 1, 0]
        return gb, expected
    else:
        return gb


def several_intersections_3d(**kwargs):
    f_1 = np.array([[2, 5, 5, 2], [2, 2, 5, 5], [5, 5, 5, 5]])
    f_2 = np.array([[2, 2, 5, 5], [5, 5, 5, 5], [2, 5, 5, 2]])
    f_3 = np.array([[4, 4, 4, 4], [1, 1, 8, 8], [1, 8, 8, 1]])
    f_4 = np.array([[3, 3, 6, 6], [3, 3, 3, 3], [3, 7, 7, 3]])
    f = [f_1, f_2, f_3, f_4]
    gb = meshing.tensor_grid(f, np.array([8, 8, 8]))
    if kwargs.get('return_expected', False):
        expected = [1, 4, 9, 2]
        return gb, expected
    else:
        return gb


if __name__ == '__main__':

    # If invoked as main, run all tests
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'v:', ['verbose=',
                                                        'compute_geometry='])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    gmsh_path = None
    verbose = 0
    compute_geometry = True
    # process options
    for o, a in opts:
        if o == '--gmsh_path':
            gmsh_path = a
        elif o in ('-v', '--verbose'):
            verbose = int(a)
        elif o == '--compute_geometry':
            compute_geometry = True

    return_expected = 1
    success_counter = 0
    failure_counter = 0

    time_tot = time.time()
    functions_list = [o for o in getmembers(
        sys.modules[__name__]) if isfunction(o[1])]
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
            gb, expected = func[1](gmsh_path=gmsh_path,
                                   verbose=verbose,
                                   gmsh_verbose=0,
                                   return_expected=True)

            # Check that the bucket has the expected number of grids in each
            # dimension.
            for dim, exp_l in zip(range(3, -1, -1), expected):
                g_loc = gb.grids_of_dimension(dim)
                assert len(g_loc) == exp_l

        except Exception as exp:
            print('\n')
            print(' ************** FAILURE **********')
            print('Example ' + func[0] + ' failed')
            print(exp)
            logging.error(traceback.format_exc())
            failure_counter += 1
            continue

        if compute_geometry:
            try:
                for g, _ in gb:
                    if g.dim == 3:
                        g.compute_geometry()
                    else:
                        g.compute_geometry(is_embedded=True)
            except Exception as exp:
                print('************ FAILURE **********')
                print('Geometry computation failed for grid ' + str(g))
                print(exp)
                logging.error(traceback.format_exc())
                failure_counter += 1
                continue

        # If we have made it this far, this is a success
        success_counter += 1
    #################################
    # Summary
    #
    print('\n')
    print(' --- ')
    print('Ran in total ' + str(success_counter + failure_counter) + ' tests,'
          + ' out of which ' + str(failure_counter) + ' failed.')
    print('Total elapsed time is ' + str(time.time() - time_tot) + ' seconds')
    print('\n')
