import numpy as np

import data as problem_data
import examples.papers.benchmark_2.solvers as solvers


def main(file_geo, folder, solver, solver_name):

    tol = 1e-8
    gb, domain = problem_data.import_grid(file_geo, tol)

    # select the permeability depending on the selected test case
    data = {
        "domain": domain,
        "tol": tol,
        "aperture": 1e-2,
        "km_low": 1e-6,
        "km_high": 1e-5,
        "kf": 1e-1,
        "porosity_low": 0.2,
        "porosity_high": 0.25,
        "porosity_f": 0.4,
        "dt": 1e7,
        "t_max": 1e9,
    }

    problem_data.add_data(gb, data, solver_name)
    solver(gb, folder)

    # to store the results for the current problem
    results = np.empty(5, dtype=np.object)

    # save basic informations
    results[0] = "UiB-" + solver_name.upper()
    results[1] = np.sum([g.num_cells for g in gb.grids_of_dimension(3)])
    results[2] = np.sum([g.num_cells for g in gb.grids_of_dimension(2)])
    results[3] = np.sum([g.num_cells for g in gb.grids_of_dimension(1)])
    results[4] = np.sum([g.num_cells for g in gb.grids_of_dimension(0)])

    file_name = folder + '/info.txt'
    with open(file_name, "w") as f:
        f.write(", ".join(map(str, results)))

    advective = solvers.transport(gb, data, solver_name, folder,
                                  problem_data.AdvectiveDataAssigner)
    np.savetxt(folder+"/outflow.csv", advective._solver.outflow)

if __name__ == "__main__":
    files_geo = {"mesh1k.geo": 0, "mesh10k.geo": 1, "mesh100k.geo": 2}
    solver_list = [
        solvers.solve_tpfa,
        solvers.solve_vem,
        solvers.solve_rt0,
        solvers.solve_mpfa,
    ]
    solver_names = ["tpfa", "vem", "rt0", "mpfa"]

    for file_geo, idx in files_geo.items():
        for solver, solver_name in zip(solver_list, solver_names):
            folder = solver_name + "_results_" + str(idx)
            main(file_geo, folder, solver, solver_name)
