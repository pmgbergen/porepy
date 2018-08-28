"""
Various method for creating grids for relatively simple fracture networks.

The module doubles as a test framework (though not unittest), and will report
on any problems if ran as a main method.

"""
# pylint: disable=invalid-name

import sys
import getopt
import time
import traceback
import logging
from inspect import isfunction, getmembers
import numpy as np
import scipy.sparse as sps

from porepy.grids import structured, simplex
from porepy.params import tensor, bc
from porepy.utils import error
from porepy.numerics.vem import dual, hybrid
from porepy.viz.plot_grid import plot_grid
import porepy.utils.comp_geom as cg


# ------------------------------------------------------------------------------#


def darcy_dual_hybridVEM_example0(**kwargs):
    #######################
    # Simple 2d Darcy problem with known exact solution
    #######################
    Nx = Ny = 25
    g = structured.CartGrid([Nx, Ny], [1, 1])
    g.compute_geometry()

    kxx = np.ones(g.num_cells)
    perm = tensor.SecondOrderTensor(g.dim, kxx)

    f = np.ones(g.num_cells)

    b_faces = g.get_all_boundary_faces()
    bnd = bc.BoundaryCondition(g, b_faces, ["dir"] * b_faces.size)
    bnd_val = np.zeros(g.num_faces)

    solver = hybrid.HybridDualVEM()
    data = {"perm": perm, "source": f, "bc": bnd, "bc_val": bnd_val}
    H, rhs = solver.matrix_rhs(g, data)

    l = sps.linalg.spsolve(H, rhs)
    u, p = solver.compute_up(g, l, data)
    P0u = dual.DualVEM().project_u(g, u)

    if kwargs["visualize"]:
        plot_grid(g, p, P0u)

    norms = np.array([error.norm_L2(g, p), error.norm_L2(g, P0u)])
    norms_known = np.array([0.041554943620853595, 0.18738227880674516])
    assert np.allclose(norms, norms_known)


# ------------------------------------------------------------------------------#


def darcy_dual_hybridVEM_example1(**kwargs):
    #######################
    # Simple 2d Darcy problem with known exact solution
    #######################

    Nx = Ny = 25
    g = structured.CartGrid([Nx, Ny], [1, 1])
    g.compute_geometry()

    kxx = np.ones(g.num_cells)
    perm = tensor.SecondOrderTensor(g.dim, kxx)

    def funP_ex(pt):
        return np.sin(2 * np.pi * pt[0]) * np.sin(2 * np.pi * pt[1])

    def funU_ex(pt):
        return [
            -2 * np.pi * np.cos(2 * np.pi * pt[0]) * np.sin(2 * np.pi * pt[1]),
            -2 * np.pi * np.sin(2 * np.pi * pt[0]) * np.cos(2 * np.pi * pt[1]),
            0,
        ]

    def fun(pt):
        return 8 * np.pi ** 2 * funP_ex(pt)

    f = np.array([fun(pt) for pt in g.cell_centers.T])

    b_faces = g.get_all_boundary_faces()
    bnd = bc.BoundaryCondition(g, b_faces, ["dir"] * b_faces.size)
    bnd_val = np.zeros(g.num_faces)
    bnd_val[b_faces] = funP_ex(g.face_centers[:, b_faces])

    solver = hybrid.HybridDualVEM()
    data = {"perm": perm, "source": f, "bc": bnd, "bc_val": bnd_val}
    H, rhs = solver.matrix_rhs(g, data)

    l = sps.linalg.spsolve(H, rhs)
    u, p = solver.compute_up(g, l, data)
    P0u = dual.DualVEM().project_u(g, u)

    if kwargs["visualize"]:
        plot_grid(g, p, P0u)

    p_ex = error.interpolate(g, funP_ex)
    u_ex = error.interpolate(g, funU_ex)

    errors = np.array([error.error_L2(g, p, p_ex), error.error_L2(g, P0u, u_ex)])
    errors_known = np.array([0.0210718223032, 0.00526933885613])
    assert np.allclose(errors, errors_known)


# ------------------------------------------------------------------------------#


def darcy_dual_hybridVEM_example2(**kwargs):
    #######################
    # Simple 2d Darcy problem on a surface with known exact solution
    #######################
    Nx = Ny = 25
    g = simplex.StructuredTriangleGrid([Nx, Ny], [1, 1])
    R = cg.rot(np.pi / 6., [0, 1, 1])
    g.nodes = np.dot(R, g.nodes)
    g.compute_geometry()

    T = cg.tangent_matrix(g.nodes)

    kxx = np.ones(g.num_cells)
    perm = tensor.SecondOrderTensor(g.dim, kxx)

    def funP_ex(pt):
        return np.pi * pt[0] - 6 * pt[1] + np.exp(1) * pt[2] - 4

    def funU_ex(pt):
        return np.dot(T, [-np.pi, 6, -np.exp(1)])

    def fun(pt):
        return 0

    f = np.array([fun(pt) for pt in g.cell_centers.T])

    b_faces = g.get_all_boundary_faces()
    bnd = bc.BoundaryCondition(g, b_faces, ["dir"] * b_faces.size)
    bnd_val = np.zeros(g.num_faces)
    bnd_val[b_faces] = funP_ex(g.face_centers[:, b_faces])

    solver = hybrid.HybridDualVEM()
    data = {"perm": perm, "source": f, "bc": bnd, "bc_val": bnd_val}
    H, rhs = solver.matrix_rhs(g, data)

    l = sps.linalg.spsolve(H, rhs)
    u, p = solver.compute_up(g, l, data)
    P0u = dual.DualVEM().project_u(g, u)

    if kwargs["visualize"]:
        plot_grid(g, p, P0u)

    p_ex = error.interpolate(g, funP_ex)
    u_ex = error.interpolate(g, funU_ex)

    errors = np.array([error.error_L2(g, p, p_ex), error.error_L2(g, P0u, u_ex)])
    errors_known = np.array([0, 0])
    assert np.allclose(errors, errors_known)


# ------------------------------------------------------------------------------#


def darcy_dual_hybridVEM_example3(**kwargs):
    #######################
    # Simple 3d Darcy problem with known exact solution
    #######################
    Nx = Ny = Nz = 7
    g = structured.CartGrid([Nx, Ny, Nz], [1, 1, 1])
    g.compute_geometry()

    kxx = np.ones(g.num_cells)
    perm = tensor.SecondOrderTensor(g.dim, kxx)

    def funP_ex(pt):
        return (
            np.sin(2 * np.pi * pt[0])
            * np.sin(2 * np.pi * pt[1])
            * np.sin(2 * np.pi * pt[2])
        )

    def funU_ex(pt):
        return [
            -2
            * np.pi
            * np.cos(2 * np.pi * pt[0])
            * np.sin(2 * np.pi * pt[1])
            * np.sin(2 * np.pi * pt[2]),
            -2
            * np.pi
            * np.sin(2 * np.pi * pt[0])
            * np.cos(2 * np.pi * pt[1])
            * np.sin(2 * np.pi * pt[2]),
            -2
            * np.pi
            * np.sin(2 * np.pi * pt[0])
            * np.sin(2 * np.pi * pt[1])
            * np.cos(2 * np.pi * pt[2]),
        ]

    def fun(pt):
        return 12 * np.pi ** 2 * funP_ex(pt)

    f = np.array([fun(pt) for pt in g.cell_centers.T])

    b_faces = g.get_all_boundary_faces()
    bnd = bc.BoundaryCondition(g, b_faces, ["dir"] * b_faces.size)
    bnd_val = np.zeros(g.num_faces)
    bnd_val[b_faces] = funP_ex(g.face_centers[:, b_faces])

    solver = hybrid.HybridDualVEM()
    data = {"perm": perm, "source": f, "bc": bnd, "bc_val": bnd_val}
    H, rhs = solver.matrix_rhs(g, data)

    l = sps.linalg.spsolve(H, rhs)
    u, p = solver.compute_up(g, l, data)
    P0u = dual.DualVEM().project_u(g, u)

    if kwargs["visualize"]:
        plot_grid(g, p, P0u)

    p_ex = error.interpolate(g, funP_ex)
    u_ex = error.interpolate(g, funU_ex)

    np.set_printoptions(linewidth=999999)
    np.set_printoptions(precision=16)

    errors = np.array([error.error_L2(g, p, p_ex), error.error_L2(g, P0u, u_ex)])
    errors_known = np.array([0.1010936831876412, 0.0680593765009036])
    assert np.allclose(errors, errors_known)


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

    time_tot = time.time()

    functions_list = [o for o in getmembers(sys.modules[__name__]) if isfunction(o[1])]

    for f in functions_list:
        func = f
        if (
            func[0] == "isfunction"
            or func[0] == "getmembers"
            or func[0] == "spsolve"
            or func[0] == "plot_grid"
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

# ------------------------------------------------------------------------------#
