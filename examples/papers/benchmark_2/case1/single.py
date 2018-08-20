import numpy as np

import single_data as single
import examples.papers.benchmark_2.solvers as solvers


def main(file_geo, folder, solver, solver_name):

    tol = 1e-8
    gb, domain = single.import_grid(file_geo, tol)

    # select the permeability depending on the selected test case
    data = {
        "domain": domain,
        "tol": tol,
        "aperture": 1e-2,
        "km_low": 1e-6,
        "km_high": 1e-5,
        "kf": 1e-1,
        "phi_low": 0.2,
        "phi_high": 0.25,
        "phi_f": 0.4,
        "dt": 1e7,
        "t_max": 1e9,
    }

    single.add_data(gb, data, solver_name)
    solver(gb, folder)
    advective = solvers.transport(gb, data, solver_name, folder, single.AdvectiveDataAssigner)
    np.savetxt(folder+"/outflow.csv", advective._solver.outflow)

if __name__ == "__main__":
    file_geo = "single_lowdim_point_based.geo"

    files_geo = {"geom_1k.geo": 0, "geom_10k.geo": 1, "geom_100k.geo": 2}
    solver_list = [
        solvers.solve_tpfa,
        solvers.solve_vem,
        solvers.solve_rt0,
        solvers.solve_mpfa,
    ]
    solver_names = ["tpfa", "vem", "rt0", "mpfa"]

    files_geo = {"geom_10k.geo": 1}  ####
    for file_geo, idx in files_geo.items():
        for solver, solver_name in zip(solver_list, solver_names):
            folder = solver_name + "_results_" + str(idx)
            main(file_geo, folder, solver, solver_name)
