import numpy as np

import geiger_3d_data as geiger
import solvers

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
    data = {'domain': domain, 'tol': tol, 'aperture': 1e-4,  'km_low': 1e-1,
            'km': 1, 'kf': kf, 'phi_f': phi_f}

    geiger.add_data(gb, data, solver_name)
    solver(gb, folder)

#    return
#
#    physics = "transport"
#    for g, d in gb:
#        d[physics+'_data'] = AdvectiveDataAssigner(g, d, **data_problem)
#
#    advective = AdvectiveProblem(gb, physics, time_step=0.1, end_time=5)
#    advective.solve("transport")


if __name__ == "__main__":

    files_geo = ['mesh15.geo', 'mesh1.geo', 'mesh075.geo', 'mesh05.geo']
    solvers_list = [solvers.solve_rt0, solvers.solve_tpfa, solvers.solve_mpfa,
                    solvers.solve_vem]
    solvers_name = ['rt0', 'tpfa', 'mpfa', 'vem']
    test_cases = [2]

    for test_case in test_cases:
        for solver, solver_name in zip(solvers_list, solvers_name):
            for mesh_id, file_geo in enumerate(files_geo):
                folder = solver_name+'_results_'+str(test_case)+'_'+str(mesh_id)
                main(test_case, file_geo, folder, solver, solver_name)
