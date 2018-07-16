import numpy as np
import scipy.sparse as sps

from porepy.grids.grid import Grid
from porepy.fracs import mortars

import example_1_data
import solvers

# ------------------------------------------------------------------------------#


def reference_solution():

    # Compute the reference solution with the RT0
    cells_2d = 100000
    solver = "rt0"
    gb_ref = example_1_data.create_gb(cells_2d)
    example_1_data.add_data(gb_ref, solver)
    folder = "example_1_reference"
    solvers.solve_rt0(gb_ref, folder)
    return gb_ref


# ------------------------------------------------------------------------------#


def convergence_test(N, gb_ref, solver, solver_fct):

    f = open(solver + "_error.txt", "w")
    for i in np.arange(N):

        cells_2d = 200 * 4 ** i
        alpha_1d = None
        alpha_mortar = 0.75
        gb = example_1_data.create_gb(cells_2d, alpha_1d, alpha_mortar)

        example_1_data.add_data(gb, solver)
        folder = "example_1_" + solver + "_" + str(i)
        solver_fct(gb, folder)

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
                    Pi_ref[idx, idx] = mortars.split_matrix_1d(
                        g, g_ref, example_1_data.tol()
                    )

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
