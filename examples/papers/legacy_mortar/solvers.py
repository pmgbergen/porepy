import numpy as np
import scipy.sparse as sps

from porepy.viz.exporter import Exporter

import porepy as pp
from porepy.numerics import fv
from porepy.numerics import fem
from porepy.numerics import vem

# ------------------------------------------------------------------------------#


def mortar_dof_size(A, gb, solver_flow):
    dummy = np.zeros(A.shape[0])
    solver_flow.split(gb, "dummy", dummy)
    mortar_dofs = 0
    for e, d in gb.edges_props():
        mortar_dofs += d["mortar_solution"].size
    return mortar_dofs


# ------------------------------------------------------------------------------#


def solve_rt0(gb, folder, return_only_matrix=False):

    # Choose and define the solvers and coupler
    solver_flow = pp.RT0MixedDim("flow")
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    A = A_flow
    if return_only_matrix:
        return A, mortar_dof_size(A, gb, solver_flow)

    up = sps.linalg.spsolve(A, b_flow)
    solver_flow.split(gb, "up", up)

    solver_flow.extract_p(gb, "up", "pressure")

    save = Exporter(gb, "sol", folder=folder)
    save.write_vtk(["pressure"])


# ------------------------------------------------------------------------------#


def solve_tpfa(gb, folder, return_only_matrix=False):

    # Choose and define the solvers and coupler
    solver_flow = pp.TpfaMixedDim("flow")
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    A = A_flow
    if return_only_matrix:
        return A, mortar_dof_size(A, gb, solver_flow)

    p = sps.linalg.spsolve(A, b_flow)
    solver_flow.split(gb, "pressure", p)

    save = Exporter(gb, "sol", folder=folder)
    save.write_vtk(["pressure"])


# ------------------------------------------------------------------------------#


def solve_mpfa(gb, folder, return_only_matrix=False):

    # Choose and define the solvers and coupler
    solver_flow = pp.MpfaMixedDim("flow")
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    A = A_flow
    if return_only_matrix:
        return A, mortar_dof_size(A, gb, solver_flow)

    p = sps.linalg.spsolve(A, b_flow)
    solver_flow.split(gb, "pressure", p)

    save = Exporter(gb, "sol", folder=folder)
    save.write_vtk(["pressure"])


# ------------------------------------------------------------------------------#


def solve_p1(gb, folder, return_only_matrix=False):

    # Choose and define the solvers and coupler
    solver_flow = pp.P1MixedDim("flow")
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    A = A_flow
    if return_only_matrix:
        return A, mortar_dof_size(A, gb, solver_flow)

    p = sps.linalg.spsolve(A, b_flow)
    solver_flow.split(gb, "pressure", p)


#    save = Exporter(gb, "sol", folder=folder, simplicial=True)
#    save.write_vtk(["pressure"])

# ------------------------------------------------------------------------------#


def solve_vem(gb, folder, return_only_matrix=False):

    # Choose and define the solvers and coupler
    solver_flow = pp.DualVEMMixedDim("flow")
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    A = A_flow
    if return_only_matrix:
        return A, mortar_dof_size(A, gb, solver_flow)

    up = sps.linalg.spsolve(A, b_flow)
    solver_flow.split(gb, "up", up)

    solver_flow.extract_p(gb, "up", "pressure")
    solver_flow.extract_u(gb, "up", "discharge")
    solver_flow.project_u(gb, "discharge", "P0u")

    save = Exporter(gb, "sol", folder=folder)
    save.write_vtk(["pressure", "P0u"])


# ------------------------------------------------------------------------------#
