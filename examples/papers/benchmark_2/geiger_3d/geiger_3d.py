import numpy as np
import geiger_3d_data as geiger
import examples.papers.benchmark_2.solvers as solvers


def main(test_case, file_geo, folder, solver, solver_name, N = None):

    tol = 1e-8
    if N is not None:
        gb, domain = geiger.make_grid_cart(N)
    else:
        gb, domain = geiger.import_grid(file_geo, tol)

    # select the permeability depending on the selected test case
    if test_case == 1:
        kf = 1e4
        phi_f = 0.9
    else:
        kf = 1e-4
        phi_f = 0.01
    data = {
        "domain": domain,
        "tol": tol,
        "aperture": 1e-4,
        "km_low": 1e-1,
        "km": 1,
        "kf": kf,
        "phi_m": 1e-1,
        "phi_f": phi_f,
        "dt": 0.25 / 100,
        "t_max": 0.25,
    }

    geiger.add_data(gb, data, solver_name)
    solver(gb, folder)
    solvers.transport(gb, data, solver_name, folder, geiger.AdvectiveDataAssigner)


if __name__ == "__main__":
    files_geo = {
        "mesh15.geo": "0",
        "mesh1.geo": "1",
        "mesh075.geo": "2",
        "mesh05.geo": "3",
    }
    solver_list = [
        solvers.solve_tpfa,
        solvers.solve_vem,
        solvers.solve_rt0,
        solvers.solve_mpfa,
    ]
    solver_names = ["tpfa", "vem", "rt0", "mpfa"]
    test_cases = [1, 2]

    files_geo = {"mesh15.geo": "0", "mesh05.geo": "3"}  ###
    solver_list = [solvers.solve_tpfa]  ###
    solver_names = ["tpfa"]  ###

    for test_case in test_cases:
        for solver, solver_name in zip(solver_list, solver_names):
            for file_geo, mesh_id in files_geo.items():
                folder = solver_name + "_results_" + str(test_case) + "_" + mesh_id
                if mesh_id == "0" and test_case == 1:
                    continue
                if mesh_id == "0":
                    N = 16
                else:
                    N = 32
                main(test_case, file_geo, folder, solver, solver_name, N)
