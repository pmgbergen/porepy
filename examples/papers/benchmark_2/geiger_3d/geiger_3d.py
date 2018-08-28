import numpy as np
import porepy as pp
import geiger_3d_data as geiger
import examples.papers.benchmark_2.solvers as solvers


def main(test_case, file_geo, folder, solver, solver_name):

    tol = 1e-8
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
        "dt": .1,
        "t_max": 5,
    }

    geiger.add_data(gb, data, solver_name)
    solver(gb, folder)

    solvers.transport(gb, data, solver_name, folder, geiger.AdvectiveDataAssigner)
    return gb, data


if __name__ == "__main__":
    files_geo = ["mesh15.geo", "mesh1.geo", "mesh075.geo", "mesh05.geo"]
    #    solver_list = [solvers.solve_rt0, solvers.solve_tpfa, solvers.solve_mpfa,
    #                    solvers.solve_vem]
    solver_list = [solvers.solve_tpfa, solvers.solve_vem]
    solver_names = ["tpfa", "vem"]
    test_cases = [2]

    for test_case in test_cases:
        for solver, solver_name in zip(solver_list, solver_names):
            for mesh_id, file_geo in enumerate(files_geo):
                folder = (
                    solver_name + "_case_" + str(test_case) + "_mesh_" + str(mesh_id)
                )
                gb, data = main(test_case, file_geo, folder, solver, solver_name)
