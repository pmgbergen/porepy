import numpy as np
import scipy.sparse as sps

import porepy as pp

#------------------------------------------------------------------------------#

def export(gb, folder):
    save = pp.Exporter(gb, "sol", folder=folder)
    save.write_vtk(["pressure"])

#------------------------------------------------------------------------------#

def solve_rt0(gb, folder):

    # Choose and define the solvers and coupler
    solver_flow = pp.RT0MixedDim('flow')
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    solver_source = pp.DualSourceMixedDim('flow')
    A_source, b_source = solver_source.matrix_rhs(gb)
    A = A_flow + A_source

    up = sps.linalg.spsolve(A, b_flow+b_source)
    solver_flow.split(gb, "up", up)

    solver_flow.extract_p(gb, "up", "pressure")
    export(gb, folder)

#------------------------------------------------------------------------------#

def solve_tpfa(gb, folder):

    # Choose and define the solvers and coupler
    solver_flow = pp.TpfaMixedDim('flow')
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    solver_source = pp.IntegralMixedDim('flow')
    A_source, b_source = solver_source.matrix_rhs(gb)
    A = A_flow + A_source

    p = sps.linalg.spsolve(A, b_flow+b_source)
    solver_flow.split(gb, "pressure", p)

    export(gb, folder)

#------------------------------------------------------------------------------#

def solve_mpfa(gb, folder):

    # Choose and define the solvers and coupler
    solver_flow = pp.MpfaMixedDim('flow')
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    solver_source = pp.IntegralMixedDim('flow')
    A_source, b_source = solver_source.matrix_rhs(gb)
    A = A_flow + A_source

    p = sps.linalg.spsolve(A, b_flow+b_source)
    solver_flow.split(gb, "pressure", p)

    export(gb, folder)

#------------------------------------------------------------------------------#

def solve_p1(gb, folder):

    # Choose and define the solvers and coupler
    solver_flow = pp.P1MixedDim('flow')
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    solver_source = pp.IntegralMixedDim('flow')
    A_source, b_source = solver_source.matrix_rhs(gb)
    A = A_flow + A_source

    p = sps.linalg.spsolve(A, b_flow+b_source)
    solver_flow.split(gb, "pressure", p)

    save = pp.Exporter(gb, "sol", folder=folder, simplicial=True)
    save.write_vtk(["pressure"])

#------------------------------------------------------------------------------#

def solve_vem(gb, folder):

    # Choose and define the solvers and coupler
    solver_flow = pp.DualVEMMixedDim('flow')
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    solver_source = pp.DualSourceMixedDim('flow')
    A_source, b_source = solver_source.matrix_rhs(gb)
    A = A_flow + A_source

    up = sps.linalg.spsolve(A, b_flow+b_source)
    solver_flow.split(gb, "up", up)

    solver_flow.extract_p(gb, "up", "pressure")
    solver_flow.extract_u(gb, "up", "discharge")
    solver_flow.project_u(gb, "discharge", "P0u")

    export(gb, folder)

#------------------------------------------------------------------------------#
