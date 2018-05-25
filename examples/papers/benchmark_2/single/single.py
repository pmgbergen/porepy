import numpy as np

import single_data as single
import solvers

def main(file_geo, folder, solver, solver_name):

    tol = 1e-8
    gb, domain = single.import_grid(file_geo, tol)

    # select the permeability depending on the selected test case
    data = {'domain': domain, 'tol': tol, 'aperture': 1e-2, 'mu': 1,
            'km_low': 1e-5, 'km_high': 1e-6, 'kf': 1e-3,
            'phi_low': 0.2, 'phi_high': 0.25, 'phi_f': 0.4}

    single.add_data(gb, data, solver_name)
    solver(gb, folder)

if __name__ == "__main__":
    file_geo = 'single_lowdim_new.geo'
    solvers_list = [solvers.solve_rt0, solvers.solve_tpfa, solvers.solve_mpfa,
                    solvers.solve_vem]
    solvers_name = ['rt0', 'tpfa', 'mpfa', 'vem']

    solvers_list = [solvers.solve_rt0, solvers.solve_vem]
    solvers_name = ['rt0', 'vem']

    for solver, solver_name in zip(solvers_list, solvers_name):
        folder = solver_name+'_results'
        main(file_geo, folder, solver, solver_name)
