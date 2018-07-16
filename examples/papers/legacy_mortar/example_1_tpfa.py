#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sps

from porepy.viz.exporter import Exporter

from porepy.fracs import mortars
from porepy.grids.grid import FaceTag, Grid
from porepy.numerics.fv import tpfa, source
from porepy.numerics.vem import vem_source
from porepy.numerics.fem import rt0

import example_1_data

# ------------------------------------------------------------------------------#


def solve_tpfa(gb, folder):

    # Choose and define the solvers and coupler
    solver_flow = tpfa.TpfaMixedDim("flow")
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    solver_source = source.IntegralMixedDim("flow")
    A_source, b_source = solver_source.matrix_rhs(gb)

    p = sps.linalg.spsolve(A_flow + A_source, b_flow + b_source)
    solver_flow.split(gb, "pressure", p)

    save = Exporter(gb, "sol", folder=folder)
    save.write_vtk(["pressure"])


# ------------------------------------------------------------------------------#


def solve_rt0(gb, folder):

    # Choose and define the solvers and coupler
    solver_flow = rt0.RT0MixedDim("flow")
    A_flow, b_flow = solver_flow.matrix_rhs(gb)

    solver_source = vem_source.IntegralMixedDim("flow")
    A_source, b_source = solver_source.matrix_rhs(gb)

    up = sps.linalg.spsolve(A_flow + A_source, b_flow + b_source)
    solver_flow.split(gb, "up", up)

    solver_flow.extract_p(gb, "up", "pressure")

    save = Exporter(gb, "sol", folder=folder)
    save.write_vtk(["pressure"])


# ------------------------------------------------------------------------------#


if __name__ == "__main__":

    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    data = {"kf_high": 1e2, "kf_low": 1e-2}
    tol = 1e-8

    # Compute the reference solution with the RT0
    cells_2d = 100000
    data["solver"] = "rt0"
    gb_ref = example_1_data.create_gb(domain, cells_2d, tol=tol)

    # only when solving for the vem case
    internal_flag = FaceTag.FRACTURE
    [g.remove_face_tag_if_tag(FaceTag.BOUNDARY, internal_flag) for g, _ in gb_ref]

    example_1_data.add_data(gb_ref, domain, data, tol)
    folder = "example_1_reference"
    solve_rt0(gb_ref, folder)

    N = 5
    data["solver"] = "tpfa"
    f = open(data["solver"] + "_error.txt", "w")
    for i in np.arange(N):

        cells_2d = 200 * 4 ** i
        alpha_1d = None
        alpha_mortar = 0.5
        gb = example_1_data.create_gb(domain, cells_2d, alpha_1d, alpha_mortar, tol)

        example_1_data.add_data(gb, domain, data, tol)
        folder = "example_1_" + data["solver"] + "_" + str(i)
        solve_tpfa(gb, folder)

        error_0d = 0
        ref_0d = 0
        error_1d = 0
        ref_1d = 0

        for e_ref, d_ref in gb_ref.edges_props():
            for e, d in gb.edges_props():
                if d_ref["edge_number"] == d["edge_number"]:
                    break

            mg_ref = d_ref["mortar_grid"]
            mg = d["mortar_grid"]

            m_ref = d_ref["mortar_solution"]
            m = d["mortar_solution"]
            num_cells = int(mg.num_cells / 2)
            m_switched = np.hstack((m[num_cells:], m[:num_cells]))

            if mg_ref.dim == 0:
                error_0d += np.power(m - m_ref, 2)[0]
                ref_0d += np.power(m_ref, 2)[0]

            if mg_ref.dim == 1:
                Pi_ref = np.empty((mg.num_sides(), mg.num_sides()), dtype=np.object)

                for idx, (side, g_ref) in enumerate(mg_ref.side_grids.items()):
                    g = mg.side_grids[side]
                    Pi_ref[idx, idx] = mortars.split_matrix_1d(g, g_ref, tol)

                Pi_ref = sps.bmat(Pi_ref, format="csc")

                inv_k = 1. / (2. * d_ref["kn"])
                M = sps.diags(inv_k / mg_ref.cell_volumes)
                delta = m_ref - Pi_ref * m
                delta_switched = m_ref - Pi_ref * m_switched

                error_1d_loc = np.dot(delta, M * delta)
                error_1d_loc_switched = np.dot(delta_switched, M * delta_switched)

                error_1d += min(error_1d_loc, error_1d_loc_switched)
                ref_1d += np.dot(m_ref, M * m_ref)

        error_0d = "%1.2e" % np.sqrt(error_0d / ref_0d)
        error_1d = "%1.2e" % np.sqrt(error_1d / ref_1d)

        def cond(g):
            return not (isinstance(g, Grid))

        diam_mg = "%1.2e" % gb.diameter(cond)

        def cond(g):
            return isinstance(g, Grid)

        diam_g = "%1.2e" % gb.diameter(cond)

        f.write(
            str(i)
            + " \t"
            + diam_g
            + " \t"
            + diam_mg
            + " \t"
            + error_0d
            + " \t"
            + error_1d
            + "\n"
        )

    f.close()

# ------------------------------------------------------------------------------#
