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
from scipy.sparse.linalg import spsolve


from porepy.grids import structured
from porepy.params import tensor, bc
from porepy.utils import error
from porepy.numerics.vem import dual, dual_coupling
from porepy.viz.plot_grid import plot_grid
from porepy.viz.exporter import export_vtk

from porepy.grids.coarsening import *
from porepy.fracs import meshing

from porepy.numerics.mixed_dim import coupler


# ------------------------------------------------------------------------------#

# def darcy_dualVEM_coupling_example0(**kwargs):
#    #######################
#    # Simple 2d Darcy problem with known exact solution
#    #######################
#    Nx, Ny = 40, 40
#    f = np.array([[1, 1], [0, 1]])
#
#    gb = meshing.cart_grid([f], [Nx, Ny], physdims=[2, 1])
#    gb.assign_node_ordering()
#
#    # Need to remove the boundary flag explicity from the fracture face,
#    # because of the mix formulation
#
#    if kwargs['visualize']: plot_grid(gb, info="all", alpha=0)
#
#
#    # devo mettere l'apertura indicata come a
#
#    gb.add_node_props(['perm', 'source', 'bc', 'bc_val'])
#    for g, d in gb:
#        kxx = np.ones(g.num_cells)
#        d['perm'] = tensor.SecondOrderTensor(g.dim, kxx)
#        d['source'] = np.zeros(g.num_cells)
#
#        b_faces = g.get_all_boundary_faces()
#        b_faces_left = b_faces[g.face_centers[0, b_faces] == 0]
#        b_faces_right = b_faces[g.face_centers[0, b_faces] == 2]
#
#        b_faces_dir = np.hstack((b_faces_left, b_faces_right))
#        idx = np.argsort(b_faces_dir)
#        b_faces_dir_cond = np.hstack((np.zeros(b_faces_left.size),
#                                      np.ones(b_faces_right.size)))
#
#        b_faces_neu = np.setdiff1d(b_faces, b_faces_dir, assume_unique=True)
#        b_faces = np.hstack((b_faces_dir, b_faces_neu))
#        b_faces_label = np.hstack((['dir']*b_faces_dir.size,
#                                   ['neu']*b_faces_neu.size))
#        d['bc'] = bc.BoundaryCondition(g, b_faces, b_faces_label)
#        d['bc_val'] = {'dir': b_faces_dir_cond[idx],
#                       'neu': np.zeros(b_faces_neu.size)}
#
#    gb.add_edge_prop('kn')
#    for e, d in gb.edges_props():
#        g_l = gb.sorted_nodes_of_edge(e)[0]
#        c = np.logical_and(g_l.cell_centers[1,:]>0.25, g_l.cell_centers[1,:]<0.75)
#        d['kn'] = 1e2*(np.ones(g_l.num_cells) + c.astype('float')*(-1+1e-2))
#
#    solver = dual.DualVEM()
#    coupling_conditions = dual_coupling.DualCoupling(solver)
#    solver_coupler = coupler.Coupler(solver, coupling_conditions)
#    A, b = solver_coupler.matrix_rhs(gb)
#
#    solver_coupler.split(gb, "up", sps.linalg.spsolve(A, b))
#
#    gb.add_node_props(["u", 'pressure', "P0u"])
#    for g, d in gb:
#        d["u"] = solver.extractU(g, d["up"])
#        d['pressure'] = solver.extractP(g, d["up"])
#        d["P0u"] = solver.projectU(g, d["u"], d)
#
#    if kwargs['visualize']: plot_grid(gb, 'pressure', "P0u")
#
#    export_vtk(gb, "grid", ['pressure', "P0u"])
#
##    norms = np.array([error.norm_L2(g, p), error.norm_L2(g, P0u)])
##    norms_known = np.array([0.041554943620853595, 0.18738227880674516])
##    assert np.allclose(norms, norms_known)

# ------------------------------------------------------------------------------#

# def darcy_dualVEM_coupling_example1(**kwargs):
#    #######################
#    # Simple 2d Darcy problem with known exact solution
#    #######################
#    np.set_printoptions( linewidth = 999999, threshold=np.nan, precision = 16 )
#
#    Nx, Ny = 10, 10
#    f0 = np.array([[1, 1], [0, 1]])
#    f1 = np.array([[0, 2], [0.5, 0.5]])
#
#    gb = meshing.cart_grid([f0, f1], [Nx, Ny], physdims=[2, 1])
#    gb.remove_nodes(lambda g: g.dim == gb.dim_max())
#    gb.assign_node_ordering()
##
#    if kwargs['visualize']: plot_grid(gb, info="cfo", alpha=0)
#
#    # devo mettere l'apertura indicata come a
#
#    gb.add_node_props(['perm', 'source', 'bc', 'bc_val'])
#    for g, d in gb:
#        kxx = np.ones(g.num_cells)
#        d['perm'] = tensor.SecondOrderTensor(g.dim, kxx)
#        d['source'] = np.zeros(g.num_cells)
#
#        if g.dim != 0:
#            b_faces = g.get_all_boundary_faces()
#            b_faces_dir = np.array([0])
#            b_faces_dir_cond = g.face_centers[0, b_faces_dir]
#
#            b_faces_neu = np.setdiff1d(b_faces, b_faces_dir, assume_unique=True)
#            b_faces = np.hstack((b_faces_dir, b_faces_neu))
#            b_faces_label = np.hstack((['dir']*b_faces_dir.size,
#                                       ['neu']*b_faces_neu.size))
#            d['bc'] = bc.BoundaryCondition(g, b_faces, b_faces_label)
#            d['bc_val'] = {'dir': b_faces_dir_cond,
#                           'neu': np.zeros(b_faces_neu.size)}
#
#    gb.add_edge_prop('kn')
#    for e, d in gb.edges_props():
#        g_l = gb.sorted_nodes_of_edge(e)[0]
#        c = np.logical_and(g_l.cell_centers[1,:]>0.25, g_l.cell_centers[1,:]<0.75)
#        d['kn'] = np.ones(g_l.num_cells)
##        d['kn'] = 1e2*(np.ones(g_l.num_cells) + c.astype('float')*(-1+1e-2))
#
#    solver = dual.DualVEM()
#    coupling_conditions = dual_coupling.DualCoupling(solver)
#    solver_coupler = coupler.Coupler(solver, coupling_conditions)
#    A, b = solver_coupler.matrix_rhs(gb)
#
#    up = sps.linalg.spsolve(A, b)
#    solver_coupler.split(gb, "up", up)
#
#    gb.add_node_props(["u", 'pressure', "P0u"])
#    for g, d in gb:
#        d["u"] = solver.extractU(g, d["up"])
#        d['pressure'] = solver.extractP(g, d["up"])
#        d["P0u"] = solver.projectU(g, d["u"], d)
#
#    if kwargs['visualize']: plot_grid(gb, 'pressure', "P0u")
#
#    export_vtk(gb, "grid", ['pressure', "P0u"])
#
##    norms = np.array([error.norm_L2(g, p), error.norm_L2(g, P0u)])
##    norms_known = np.array([0.041554943620853595, 0.18738227880674516])
##    assert np.allclose(norms, norms_known)

# ------------------------------------------------------------------------------#


def darcy_dualVEM_coupling_example2(**kwargs):
    #######################
    # Simple 2d Darcy problem with known exact solution
    #######################
    np.set_printoptions(linewidth=999999, threshold=np.nan, precision=16)

    f_1 = np.array([[-1, 1, 1, -1], [0, 0, 0, 0], [-1, -1, 1, 1]])
    f_2 = np.array([[0, 0, 0, 0], [-1, 1, 1, -1], [-1.5, -1.5, .8, .8]])
    domain = {"xmin": -2, "xmax": 2, "ymin": -2, "ymax": 2, "zmin": -2, "zmax": 2}

    kwargs = {
        "mesh_size_frac": .25,
        "mesh_size_bound": 10,
        "mesh_size_min": .02,
        "gmsh_path": "~/gmsh/bin/gmsh",
    }

    gb = meshing.simplex_grid([f_1, f_2], domain, **kwargs)
    gb.remove_nodes(lambda g: g.dim == gb.dim_max())

    # It's not really working now the coarsening part with the gb
    #    for g, _ in gb:
    #        if g.dim != 1:
    #            part = create_partition(tpfa_matrix(g))
    #            gb.change_nodes({g: generate_coarse_grid(g, part)})
    #
    #    gb.compute_geometry(is_starshaped=True)

    print([g.num_faces for g, _ in gb])
    gb.assign_node_ordering()

    if kwargs["visualize"]:
        plot_grid(gb, info="f", alpha=0)

    gb.add_node_props(["perm", "source", "bc", "bc_val"])
    for g, d in gb:
        kxx = np.ones(g.num_cells)
        d["perm"] = tensor.SecondOrderTensor(g.dim, kxx)
        d["source"] = np.zeros(g.num_cells)

        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        b_faces_dir = b_faces[
            np.bitwise_or(
                g.face_centers[1, b_faces] == -1, g.face_centers[0, b_faces] == -1
            )
        ]

        b_faces_dir_cond = g.face_centers[0, b_faces_dir]

        b_faces_neu = np.setdiff1d(b_faces, b_faces_dir, assume_unique=True)
        b_faces = np.hstack((b_faces_dir, b_faces_neu))
        b_faces_label = np.hstack(
            (["dir"] * b_faces_dir.size, ["neu"] * b_faces_neu.size)
        )
        d["bc"] = bc.BoundaryCondition(g, b_faces, b_faces_label)
        d["bc_val"] = {"dir": b_faces_dir_cond, "neu": np.zeros(b_faces_neu.size)}

    gb.add_edge_prop("kn")
    for e, d in gb.edges_props():
        g_l = gb.sorted_nodes_of_edge(e)[0]
        c = g_l.cell_centers[2, :] > -0.5
        d["kn"] = np.ones(g_l.num_cells) + c.astype("float") * (-1 + 1e-2)

    solver = dual.DualVEM()
    coupling_conditions = dual_coupling.DualCoupling(solver)
    solver_coupler = coupler.Coupler(solver, coupling_conditions)
    A, b = solver_coupler.matrix_rhs(gb)

    up = sps.linalg.spsolve(A, b)
    solver_coupler.split(gb, "up", up)

    gb.add_node_props(["u", "pressure", "P0u"])
    for g, d in gb:
        d["u"] = solver.extractU(g, d["up"])
        d["pressure"] = solver.extractP(g, d["up"])
        d["P0u"] = solver.projectU(g, d["u"], d)

    if kwargs["visualize"]:
        plot_grid(gb, "pressure", "P0u")

    export_vtk(gb, "grid", ["pressure", "P0u"])


#    norms = np.array([error.norm_L2(g, p), error.norm_L2(g, P0u)])
#    norms_known = np.array([0.041554943620853595, 0.18738227880674516])
#    assert np.allclose(norms, norms_known)

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
            or func[0] == "generate_coarse_grid"
            or func[0] == "tpfa_matrix"
            or func[0] == "create_partition"
            or func[0] == "export_vtk"
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
