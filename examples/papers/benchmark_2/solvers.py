import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.darcy_and_transport import static_flow_IE_solver as TransportSolver

# ------------------------------------------------------------------------------#


def export(gb, folder):

    gb.add_node_props(["cell_volumes", "cell_centers"])
    for g, d in gb:
        d["cell_volumes"] = g.cell_volumes
        d["cell_centers"] = g.cell_centers

    save = pp.Exporter(gb, "sol", folder=folder)
    save.write_vtk(["pressure", "cell_volumes", "cell_centers"])


# ------------------------------------------------------------------------------#


def solve_rt0(gb, folder):

    # Choose and define the solvers and coupler
    solver_flow = pp.RT0MixedDim("flow")
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    solver_source = pp.DualSourceMixedDim("flow")
    A_source, b_source = solver_source.matrix_rhs(gb)
    A = A_flow + A_source

    up = sps.linalg.spsolve(A, b_flow + b_source)
    solver_flow.split(gb, "up", up)

    solver_flow.extract_p(gb, "up", "pressure")
    export(gb, folder)


# ------------------------------------------------------------------------------#


def solve_tpfa(gb, folder):

    # Choose and define the solvers and coupler
    solver_flow = pp.TpfaMixedDim("flow")
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    solver_source = pp.IntegralMixedDim("flow")
    A_source, b_source = solver_source.matrix_rhs(gb)
    A = A_flow + A_source

    p = sps.linalg.spsolve(A, b_flow + b_source)
    solver_flow.split(gb, "pressure", p)
    pp.fvutils.compute_discharges(gb)

    export(gb, folder)


# ------------------------------------------------------------------------------#


def solve_mpfa(gb, folder):

    # Choose and define the solvers and coupler
    solver_flow = pp.MpfaMixedDim("flow")
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    solver_source = pp.IntegralMixedDim("flow")
    A_source, b_source = solver_source.matrix_rhs(gb)
    A = A_flow + A_source

    p = sps.linalg.spsolve(A, b_flow + b_source)
    solver_flow.split(gb, "pressure", p)
    pp.fvutils.compute_discharges(gb)

    export(gb, folder)


# ------------------------------------------------------------------------------#


def solve_p1(gb, folder):

    # Choose and define the solvers and coupler
    solver_flow = pp.P1MixedDim("flow")
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    solver_source = pp.IntegralMixedDim("flow")
    A_source, b_source = solver_source.matrix_rhs(gb)
    A = A_flow + A_source

    p = sps.linalg.spsolve(A, b_flow + b_source)
    solver_flow.split(gb, "pressure", p)

    save = pp.Exporter(gb, "sol", folder=folder, simplicial=True)
    save.write_vtk(["pressure"])


# ------------------------------------------------------------------------------#


def solve_vem(gb, folder):

    # Choose and define the solvers and coupler
    solver_flow = pp.DualVEMMixedDim("flow")
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    solver_source = pp.DualSourceMixedDim("flow")
    A_source, b_source = solver_source.matrix_rhs(gb)
    A = A_flow + A_source

    up = sps.linalg.spsolve(A, b_flow + b_source)
    solver_flow.split(gb, "up", up)

    solver_flow.extract_p(gb, "up", "pressure")
    solver_flow.extract_u(gb, "up", "discharge")
    solver_flow.project_u(gb, "discharge", "P0u")

    export(gb, folder)


# ------------------------------------------------------------------------------#


def transport(gb, data, solver_name, folder, adv_data_assigner, callback=None):
    physics = "transport"
    for g, d in gb:
        d[physics + "_data"] = adv_data_assigner(g, d, **data)
    advective = AdvectiveProblem(
        gb,
        physics,
        time_step=data["dt"],
        end_time=data["t_max"],
        folder_name=folder,
        file_name="tracer",
        callback=callback,
    )
    advective.solve("tracer")


class AdvectiveProblem(pp.ParabolicModel):
    def space_disc(self):
        return self.source_disc(), self.advective_disc()

    def solver(self):
        "Initiate solver"
        return TransportSolver(self)
