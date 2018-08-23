import logging, sys
import numpy as np
import porepy as pp
import case4_data
try:
    import examples.papers.benchmark_2.solvers as solvers
except:
    pass


root = logging.getLogger()
root.setLevel(logging.INFO)

if not root.hasHandlers():
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    root.addHandler(ch)


def main(folder, solver, solver_name, dt, save_every):

    tol = 1e-8
    gb, domain = case4_data.create_grid(from_file=False)

    print('Loaded grid with ', gb.num_cells(), ' cells')

    data = {"domain": domain, "t_max": 1}
    data["dt"] = dt


    case4_data.add_data(gb, data, solver_name)

    print('Invoke solver ', solver_name)
#    solver(gb, folder)

#    outlet_fluxes(gb, fn='concentrations_' + solver_name + '_' + str(gb.num_cells()) + '_dt_' + str(dt) + '.txt')

    def report_concentrations(problem):
        problem.split()

        print('Max concentration ', problem._solver.p.max())

        for g, d in problem.grid():
            if g.dim == 2:
                pv = d['param'].porosity * g.cell_volumes
                field = d['solution']

                mean = np.sum(pv * field) / np.sum(pv)
                print('Mean concentration in fracture ', g.frac_num, ' ', mean)

    print('Invoke transport solver')
    return gb, data
"""
    solvers.transport(gb, data, solver_name, folder,
                          case4_data.AdvectiveDataAssigner,
                         callback=report_concentrations, save_every=save_every)
"""


def outlet_fluxes(gb, fn=None):
    for g, d in gb:
        if g.dim == 3:
            break

    flux = d['discharge']
    _, b_out = case4_data.b_pressure(g)
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
#    solver_list = [solvers.solve_rt0, solvers.solve_tpfa, solvers.solve_mpfa,
#                    solvers.solve_vem]
    solver_list = [solvers.solve_tpfa, solvers.solve_mpfa, solvers.solve_vem,
                   solvers.solve_rt0]
    solver_names = ['tpfa', 'mpfa', 'vem', 'rt0']
    solver_names = ['tpfa', 'mpfa']
    solver_list = [solvers.solve_tpfa, solvers.solve_mpfa]
#    time_steps = [0.1, 0.05, 0.025, 0.01, 0.005, 0.001]
#    save_every = [1, 2, 4, 10, 20, 100]

    time_steps = [1]
    save_every = [1]

    for solver, solver_name in zip(solver_list, solver_names):
        for dt, save in zip(time_steps, save_every):
            folder = solver_name  + '_dt_' + str(dt)
            gb, data = main(folder, solver, solver_name, dt, save)

