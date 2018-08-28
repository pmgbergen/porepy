import numpy as np
import scipy.sparse as sps

from porepy.fracs import importer, mortars, meshing
from porepy.numerics import elliptic
from porepy.grids.grid import Grid

import example_2_data
from solvers import *

case = 1
h_list = [0.2, 0.15, 0.1, 0.05]
list_of_solvers = {
    "tpfa": solve_tpfa,
    "p1": solve_p1,
    "mpfa": solve_mpfa,
    "rt0": solve_rt0,
    "vem": solve_vem,
}

# list_of_solvers = {"mpfa": solve_tpfa, "p1": solve_p1}
# list_of_solvers = {'vem': solve_vem, 'rt0': solve_rt0, 'mpfa': solve_mpfa}
# h_list = [0.2, 0.15, 0.1]


def reference_solution(h=0.1):
    # Compute the reference solution with the RT0
    print("Reference solution")
    solver = "rt0"
    gb_ref, domain = example_2_data.create_gb(h)
    example_2_data.add_data(gb_ref, domain, solver, case)
    folder = "example_2_reference"
    solve_rt0(gb_ref, folder)
    return gb_ref


def convergence_test(h_list, list_of_solvers, gb_ref):

    for solver_name, solver_fct in list_of_solvers.items():
        f = open(solver_name + "_error.txt", "w")

        print("Start simulation with " + solver_name)
        for i, h in enumerate(h_list):

            gb, domain = example_2_data.create_gb(h, h_dfn=0.99 * h)
            example_2_data.add_data(gb, domain, solver_name, case)
            folder = "example_2_" + solver_name
            solver_fct(gb, folder)

            error_0d = 0
            ref_0d = 0
            error_1d = 0
            ref_1d = 0
            error_2d = 0
            ref_2d = 0

            for e_ref, d_ref in gb_ref.edges():
                found = False
                for e, d in gb.edges():
                    if d_ref["edge_id"] == d["edge_id"]:
                        found = True
                        break
                assert found

                mg_ref = d_ref["mortar_grid"]
                mg = d["mortar_grid"]

                m_ref = d_ref["mortar_solution"]
                m = d["mortar_solution"]
                num_cells = int(mg.num_cells / 2)
                m_switched = np.hstack((m[num_cells:], m[:num_cells]))

                if mg_ref.dim == 0:
                    error_0d += np.power(m - m_ref, 2)[0]
                    ref_0d += np.power(m_ref, 2)[0]

                elif mg_ref.dim == 1:
                    Pi_ref = np.empty((mg.num_sides(), mg.num_sides()), dtype=np.object)

                    for idx, (side, g_ref) in enumerate(mg_ref.side_grids.items()):
                        g = mg.side_grids[side]
                        Pi_ref[idx, idx] = mortars.split_matrix_1d(
                            g, g_ref, example_2_data.tol()
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
                elif mg_ref.dim == 2:
                    Pi_ref = np.empty((mg.num_sides(), mg.num_sides()), dtype=np.object)

                    for idx, (side, g_ref) in enumerate(mg_ref.side_grids.items()):
                        g = mg.side_grids[side]
                        Pi_ref[idx, idx] = mortars.split_matrix_2d(
                            g, g_ref, example_2_data.tol()
                        )

                    Pi_ref = sps.bmat(Pi_ref, format="csc")

                    inv_k = 1. / (2. * d_ref["kn"])
                    M = sps.diags(inv_k / mg_ref.cell_volumes)
                    delta = m_ref - Pi_ref * m
                    delta_switched = m_ref - Pi_ref * m_switched

                    error_2d_loc = np.dot(delta, M * delta)
                    error_2d_loc_switched = np.dot(delta_switched, M * delta_switched)

                    error_2d += min(error_2d_loc, error_2d_loc_switched)
                    ref_2d += np.dot(m_ref, M * m_ref)

            error_0d = "%1.2e" % np.sqrt(error_0d / ref_0d)
            error_1d = "%1.2e" % np.sqrt(error_1d / ref_1d)
            error_2d = "%1.2e" % np.sqrt(error_2d / ref_2d)

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
                + " \t"
                + error_2d
                + "\n"
            )

        f.close()


if __name__ == "__main__":
    ref_sol = reference_solution(h=0.02)
    convergence_test(h_list, list_of_solvers, ref_sol)
