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
        "kf": 1e-3,
        "phi_low": 0.2,
        "phi_high": 0.25,
        "phi_f": 0.4,
        "dt": 2e6,
        "t_max": 1e8,
    }

    single.add_data(gb, data, solver_name)
    solver(gb, folder)
    solvers.transport(gb, data, solver_name, folder, single.AdvectiveDataAssigner)


if __name__ == "__main__":
    file_geo = "single_lowdim_new.geo"
    solver_list = [solvers.solve_tpfa, solvers.solve_vem]
    solver_names = ["tpfa", "vem"]

    solver_names = ["vem"]
    solver_list = [solvers.solve_vem]
    for solver, solver_name in zip(solver_list, solver_names):
        folder = solver_name + "_results"
        main(file_geo, folder, solver, solver_name)
