"""
Various method for creating grids for relatively simple fracture networks.

The module doubles as a test framework (though not unittest), and will report
on any problems if ran as a main method.

"""
import sys
import getopt
import numpy as np
import scipy.sparse as sps
import time
import traceback
import logging
from inspect import isfunction, getmembers

from porepy.grids import structured, simplex
from porepy.params import tensor

from porepy.viz.plot_grid import plot_grid
from porepy.grids.coarsening import *

# ------------------------------------------------------------------------------#


def coarsening_example0(**kwargs):
    #######################
    # Simple 2d coarsening based on tpfa for Cartesian grids
    # isotropic permeability
    #######################
    Nx = Ny = 7
    g = structured.CartGrid([Nx, Ny], [1, 1])
    g.compute_geometry()

    if kwargs["visualize"]:
        plot_grid(g, info="all", alpha=0)

    part = create_partition(tpfa_matrix(g))
    g = generate_coarse_grid(g, part)
    g.compute_geometry(is_starshaped=True)

    if kwargs["visualize"]:
        plot_grid(g, info="all", alpha=0)

    g = structured.CartGrid([Nx, Ny], [1, 1])
    g.compute_geometry()

    part = create_partition(tpfa_matrix(g), cdepth=3)
    g = generate_coarse_grid(g, part)
    g.compute_geometry(is_starshaped=True)

    if kwargs["visualize"]:
        plot_grid(g, info="all", alpha=0)


# ------------------------------------------------------------------------------#


def coarsening_example1(**kwargs):
    #######################
    # Simple 2d coarsening based on tpfa for simplex grids
    # isotropic permeability
    #######################
    Nx = Ny = 7
    g = simplex.StructuredTriangleGrid([Nx, Ny], [1, 1])
    g.compute_geometry()

    if kwargs["visualize"]:
        plot_grid(g, info="all", alpha=0)

    part = create_partition(tpfa_matrix(g))
    g = generate_coarse_grid(g, part)
    g.compute_geometry(is_starshaped=True)

    if kwargs["visualize"]:
        plot_grid(g, info="all", alpha=0)

    g = simplex.StructuredTriangleGrid([Nx, Ny], [1, 1])
    g.compute_geometry()

    part = create_partition(tpfa_matrix(g), cdepth=3)
    g = generate_coarse_grid(g, part)
    g.compute_geometry(is_starshaped=True)

    if kwargs["visualize"]:
        plot_grid(g, info="all", alpha=0)


# ------------------------------------------------------------------------------#


def coarsening_example2(**kwargs):
    #######################
    # Simple 2d coarsening based on tpfa for Cartesian grids
    # anisotropic permeability
    #######################
    Nx = Ny = 7
    g = structured.CartGrid([Nx, Ny], [1, 1])
    g.compute_geometry()

    if kwargs["visualize"]:
        plot_grid(g, info="all", alpha=0)

    kxx = 3 * np.ones(g.num_cells)
    kyy = np.ones(g.num_cells)
    perm = tensor.SecondOrderTensor(g.dim, kxx=kxx, kyy=kyy)

    part = create_partition(tpfa_matrix(g, perm=perm))
    g = generate_coarse_grid(g, part)
    g.compute_geometry(is_starshaped=True)

    if kwargs["visualize"]:
        plot_grid(g, info="all", alpha=0)

    g = structured.CartGrid([Nx, Ny], [1, 1])
    g.compute_geometry()

    part = create_partition(tpfa_matrix(g, perm=perm), cdepth=3)
    g = generate_coarse_grid(g, part)
    g.compute_geometry(is_starshaped=True)

    if kwargs["visualize"]:
        plot_grid(g, info="all", alpha=0)

    g = structured.CartGrid([Nx, Ny], [1, 1])
    g.compute_geometry()

    part = create_partition(tpfa_matrix(g, perm=perm), cdepth=2, epsilon=1e-2)
    g = generate_coarse_grid(g, part)
    g.compute_geometry(is_starshaped=True)

    if kwargs["visualize"]:
        plot_grid(g, info="all", alpha=0)


#    THERE IS A BUG, NEED TO BE FIXED
#    g = structured.CartGrid( [Nx, Ny], [1,1] )
#    g.compute_geometry()
#
#    part = create_partition(tpfa_matrix(g, perm=perm), cdepth=2, epsilon=1)
#    g = generate_coarse_grid(g, part)
#    g.compute_geometry(is_starshaped=True)
#
#    if kwargs['visualize']: plot_grid(g, info="all", alpha=0)

# ------------------------------------------------------------------------------#


def coarsening_example3(**kwargs):
    #######################
    # Simple 2d coarsening based on tpfa for simplex grids
    # anisotropic permeability
    #######################
    Nx = Ny = 7
    g = simplex.StructuredTriangleGrid([Nx, Ny], [1, 1])
    g.compute_geometry()

    if kwargs["visualize"]:
        plot_grid(g, info="all", alpha=0)

    kxx = 3 * np.ones(g.num_cells)
    kyy = np.ones(g.num_cells)
    perm = tensor.SecondOrderTensor(g.dim, kxx=kxx, kyy=kyy)

    part = create_partition(tpfa_matrix(g, perm=perm))
    g = generate_coarse_grid(g, part)
    g.compute_geometry(is_starshaped=True)

    if kwargs["visualize"]:
        plot_grid(g, info="all", alpha=0)

    g = simplex.StructuredTriangleGrid([Nx, Ny], [1, 1])
    g.compute_geometry()

    part = create_partition(tpfa_matrix(g, perm=perm), cdepth=3)
    g = generate_coarse_grid(g, part)
    g.compute_geometry(is_starshaped=True)

    if kwargs["visualize"]:
        plot_grid(g, info="all", alpha=0)

    g = simplex.StructuredTriangleGrid([Nx, Ny], [1, 1])
    g.compute_geometry()

    part = create_partition(tpfa_matrix(g, perm=perm), cdepth=2, epsilon=1e-2)
    g = generate_coarse_grid(g, part)
    g.compute_geometry(is_starshaped=True)

    if kwargs["visualize"]:
        plot_grid(g, info="all", alpha=0)

    g = simplex.StructuredTriangleGrid([Nx, Ny], [1, 1])
    g.compute_geometry()

    part = create_partition(tpfa_matrix(g, perm=perm), cdepth=2, epsilon=1)
    g = generate_coarse_grid(g, part)
    g.compute_geometry(is_starshaped=True)

    if kwargs["visualize"]:
        plot_grid(g, info="all", alpha=0)


# ------------------------------------------------------------------------------#

# ONLY WHEN THE STAR_SHAPE COMPUTATION IS AVAILABLE ALSO IN 3D
# def coarsening_example_____(**kwargs):
#    #######################
#    # Simple 3d coarsening based on tpfa for Cartesian grids
#    #######################
#    Nx = Ny = Nz = 4
#    g = structured.CartGrid( [Nx, Ny, Nz], [1, 1, 1] )
#    g.compute_geometry()
#
#    if kwargs['visualize']: plot_grid(g, info="all", alpha=0)
#
#    part = create_partition(tpfa_matrix(g))
#    g = generate_coarse_grid(g, part)
#    g.compute_geometry(is_starshaped=True)
#
#    if kwargs['visualize']: plot_grid(g, info="all", alpha=0)
#
#    g = structured.CartGrid( [Nx, Ny, Nz], [1, 1, 1] )
#    g.compute_geometry()
#
#    part = create_partition(tpfa_matrix(g), cdepth=3)
#    g = generate_coarse_grid(g, part)
#    g.compute_geometry(is_starshaped=True)
#
#    if kwargs['visualize']: plot_grid(g, info="all", alpha=0)

# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    # If invoked as main, run all tests
    try:
        opts, args = getopt.getopt(sys.argv[1:], "v:", ["verbose=", "visualize="])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    verbose = True
    visualize = False
    # process options
    for o, a in opts:
        if o in ("-v", "--verbose"):
            verbose = bool(a)
        elif o == "--visualize":
            visualize = bool(a)

    success_counter = 0
    failure_counter = 0

    if not visualize:
        print("It is more fun if the visualization option is active")

    time_tot = time.time()

    functions_list = [o for o in getmembers(sys.modules[__name__]) if isfunction(o[1])]

    for f in functions_list:
        func = f
        if (
            func[0] == "isfunction"
            or func[0] == "getmembers"
            or func[0] == "tpfa_matrix"
            or func[0] == "plot_grid"
            or func[0] == "generate_coarse_grid"
            or func[0] == "create_partition"
        ):
            continue
        if verbose:
            print("Running " + func[0])

        time_loc = time.time()
        try:
            func[1](visualize=visualize)

        except Exception as exp:
            print("\n")
            print(" ************** FAILURE **********")
            print("Example " + func[0] + " failed")
            print(exp)
            logging.error(traceback.format_exc())
            failure_counter += 1
            continue

        # If we have made it this far, this is a success
        success_counter += 1
    #################################
    # Summary
    #
    print("\n")
    print(" --- ")
    print(
        "Ran in total "
        + str(success_counter + failure_counter)
        + " tests,"
        + " out of which "
        + str(failure_counter)
        + " failed."
    )
    print("Total elapsed time is " + str(time.time() - time_tot) + " seconds")
    print("\n")
