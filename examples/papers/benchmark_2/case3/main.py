import numpy as np
import porepy as pp
import data as problem_data
import examples.papers.benchmark_2.solvers as solvers

def main(grid_file, folder, solver, solver_name, dt):

    tol = 1e-8
    #gb, domain = problem_data.create_grid(grid_file)
    gb, domain = problem_data.create_grid()

    print('Loaded grid with ', gb.num_cells(), ' cells')

    data = {"domain": domain, "t_max": 1}
    data["dt"] = dt


    problem_data.add_data(gb, data, solver_name)

    print('Invoke solver ', solver_name)
    solver(gb, folder)

    file_name = 'concentrations_' + solver_name + '_' +\
                 str(gb.num_cells()) + '_dt_' + str(dt) + '.txt'
    outlet_fluxes(gb, fn=file_name)

    def report_concentrations(problem):
        problem.split()

        print('Max concentration ', problem._solver.p.max())

        mean = np.zeros(8)
        for g, d in problem.grid():
            if g.dim == 2:
                pv = d['param'].porosity * g.cell_volumes
                field = d['solution']

                mean[g.frac_num] = np.sum(pv * field) / np.sum(pv)

        file_name = folder+"/mean_concentration.txt"
        with open(file_name, 'a') as f:
            f.write(", ".join(map(str, mean))+"\n")

    print('Invoke transport solver')

    solvers.transport(gb, data, solver_name, folder,
                      problem_data.AdvectiveDataAssigner,
                      callback=report_concentrations)
    return gb, data

def outlet_fluxes(gb, fn=None):
    for g, d in gb:
        if g.dim == 3:
            break

    flux = d['discharge']
    _, b_out = problem_data.b_pressure(g)
    bound_faces = np.where(g.tags['domain_boundary_faces'])[0]
    xf = g.face_centers[:, bound_faces[b_out]]
    oi = bound_faces[b_out].ravel()
    lower = np.where(xf[2] < 0.5)
    upper = np.where(xf[2] > 0.5)
    n = g.face_normals[1, oi]
    bf = flux[oi] * np.sign(n)

    if fn is not None:
        with open(fn, 'w') as f:
            f.write(str(np.sum(bf[lower[0]]))+ ', ' + str(np.sum(bf[upper[0]])) + '\n')

    print('Num cells in 3D: ' + str(g.num_cells))
    print('Lower boundary outflow: ' + str(np.sum(bf[lower[0]])))
    print('Upper bounadry outflow: ' + str(np.sum(bf[upper[0]])))




if __name__ == "__main__":
    grid_files = ["grid_bucket_31645.grid", "grid_bucket_138430.grid"]
#    solver_list = [solvers.solve_rt0, solvers.solve_tpfa, solvers.solve_mpfa,
#                    solvers.solve_vem]
    solver_list = [solvers.solve_tpfa, solvers.solve_mpfa, solvers.solve_vem,
                   solvers.solve_rt0]
    solver_names = ['tpfa', 'mpfa', 'vem', 'rt0']


    grid_files = ["grid_bucket_31645.grid"]
    solver_names = ['tpfa']
    solver_list = [solvers.solve_tpfa]

    time_step = 0.01*20

    for solver, solver_name in zip(solver_list, solver_names):
        for mesh_id, gf in enumerate(grid_files):
            folder = solver_name + '_mesh_' + str(mesh_id)
            gb, data = main(gf, folder, solver, solver_name, time_step)

