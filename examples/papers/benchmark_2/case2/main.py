import numpy as np
import data as problem_data
import examples.papers.benchmark_2.solvers as solvers


def main(test_case, file_geo, folder, solver, solver_name, N = None):

    tol = 1e-8
    if N is not None:
        gb, domain = problem_data.make_grid_cart(N)
    else:
        gb, domain = problem_data.import_grid(file_geo, tol)

    # select the permeability depending on the selected test case
    if test_case == 0:
        kf = 1e4
        porosity_f = 0.9
    else:
        kf = 1e-4
        porosity_f = 0.01
    data = {
        "domain": domain,
        "tol": tol,
        "aperture": 1e-4,
        "km_low": 1e-1,
        "km": 1,
        "kf": kf,
        "porosity_m": 1e-1,
        "porosity_f": porosity_f,
        "dt": 0.25 / 100,
        "t_max": 0.25,
    }

    problem_data.add_data(gb, data, solver_name)
    solver(gb, folder)
    solvers.transport(gb, data, solver_name, folder, problem_data.AdvectiveDataAssigner)


if __name__ == "__main__":
    files_geo = {
        "mesh500.geo": "0",
        "mesh4k.geo": "1",
        "mesh32k.geo": "2"
    }
    solver_list = [
        solvers.solve_tpfa,
        solvers.solve_vem,
        solvers.solve_rt0,
        solvers.solve_mpfa,
    ]
    solver_names = ["tpfa", "vem", "rt0", "mpfa"]
    test_cases = [0, 1]
    cart_size = {"0": 8, "1": 16, "2": 32}

    for test_case in test_cases:
        for solver, solver_name in zip(solver_list, solver_names):
            for file_geo, mesh_id in files_geo.items():
                folder = solver_name + "_results_" + str(test_case) + "_" + mesh_id
                N = cart_size[mesh_id]
                if solver_name == "rt0":
                    N = None
                main(test_case, file_geo, folder, solver, solver_name, N)
