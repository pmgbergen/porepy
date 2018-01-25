import scipy.sparse as sps

from porepy.viz.exporter import Exporter

from porepy.numerics import fv
from porepy.numerics import fem
from porepy.numerics import vem

#------------------------------------------------------------------------------#

def solve_rt0(gb, folder):

    # Choose and define the solvers and coupler
    solver_flow = fem.RT0MixedDim('flow')
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    solver_source = vem.IntegralMixedDim('flow')
    A_source, b_source = solver_source.matrix_rhs(gb)

    up = sps.linalg.spsolve(A_flow+A_source, b_flow+b_source)
    solver_flow.split(gb, "up", up)

    solver_flow.extract_p(gb, "up", "pressure")

    save = Exporter(gb, "sol", folder=folder)
    save.write_vtk(["pressure"])

#------------------------------------------------------------------------------#

def solve_tpfa(gb, folder):

    # Choose and define the solvers and coupler
    solver_flow = fv.TpfaMixedDim('flow')
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    solver_source = fv.IntegralMixedDim('flow')
    A_source, b_source = solver_source.matrix_rhs(gb)

    p = sps.linalg.spsolve(A_flow+A_source, b_flow+b_source)
    solver_flow.split(gb, "pressure", p)

    save = Exporter(gb, "sol", folder=folder)
    save.write_vtk(["pressure"])

#------------------------------------------------------------------------------#

def solve_mpfa(gb, folder):

    # Choose and define the solvers and coupler
    solver_flow = fv.MpfaMixedDim('flow')
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    solver_source = fv.IntegralMixedDim('flow')
    A_source, b_source = solver_source.matrix_rhs(gb)

    p = sps.linalg.spsolve(A_flow+A_source, b_flow+b_source)
    solver_flow.split(gb, "pressure", p)

    save = Exporter(gb, "sol", folder=folder)
    save.write_vtk(["pressure"])

#------------------------------------------------------------------------------#

def solve_p1(gb, folder):

    # Choose and define the solvers and coupler
    solver_flow = fem.P1MixedDim('flow')
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    solver_source = fem.IntegralMixedDim('flow')
    A_source, b_source = solver_source.matrix_rhs(gb)

    p = sps.linalg.spsolve(A_flow+A_source, b_flow+b_source)
    solver_flow.split(gb, "pressure", p)

    save = Exporter(gb, "sol", folder=folder)
    save.write_vtk(["pressure"])

#------------------------------------------------------------------------------#

def solve_vem(gb, folder):

    # Choose and define the solvers and coupler
    solver_flow = vem.DualVEMMixedDim('flow')
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    solver_source = vem.IntegralMixedDim('flow')
    A_source, b_source = solver_source.matrix_rhs(gb)

    up = sps.linalg.spsolve(A_flow+A_source, b_flow+b_source)
    solver_flow.split(gb, "up", up)

    solver_flow.extract_p(gb, "up", "pressure")
    solver_flow.extract_u(gb, "up", "discharge")
    solver_flow.project_u(gb, "discharge", "P0u")

    save = Exporter(gb, "sol", folder=folder)
    save.write_vtk(["pressure", "P0u"])

#------------------------------------------------------------------------------#
