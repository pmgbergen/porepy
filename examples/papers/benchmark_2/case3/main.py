import numpy as np
import porepy as pp
import data as problem_data
import examples.papers.benchmark_2.solvers as solvers

# ------------------------------------------------------------------------------#

def report_concentrations(problem):
    problem.split()
    mean = np.zeros(8)
    for g, d in problem.grid():
        if g.dim == 2:
            pv = d['param'].porosity * g.cell_volumes
            mean[g.frac_num] = np.sum(pv * d['solution']) / np.sum(pv)

    file_name = folder+"/mean_concentration.txt"
    with open(file_name, 'a') as f:
        f.write(", ".join(map(str, mean))+"\n")

# ------------------------------------------------------------------------------#

def outlet_fluxes(gb):
    g = gb.grids_of_dimension(3)[0]
    d = gb.node_props(g)

    flux = d['discharge']
    _, b_out = problem_data.b_pressure(g)
    bound_faces = np.where(g.tags['domain_boundary_faces'])[0]
    xf = g.face_centers[:, bound_faces[b_out]]
    oi = bound_faces[b_out].ravel()
    lower = np.where(xf[2] < 0.5)
    upper = np.where(xf[2] > 0.5)
    n = g.face_normals[1, oi]
    bf = flux[oi] * np.sign(n)

    return np.sum(bf[lower[0]]), np.sum(bf[upper[0]])

# ------------------------------------------------------------------------------#

def mean_inlet_pressure(gb, solver_name):
    g = gb.grids_of_dimension(3)[0]
    d = gb.node_props(g)

    inlet = g.tags["inlet_faces"]
    if solver_name == "tpfa" or solver_name == "mpfa":
        face_pressure = d["bound_pressure_cell"]*d["pressure"] + \
                        d["bound_pressure_face"]*d["param"].get_bc_val("flow")
    else:
        face_pressure = np.abs(g.cell_faces) * d["pressure"]

    weighted_face_pressure = np.multiply(g.face_areas, face_pressure)
    return np.sum(weighted_face_pressure[inlet])/np.sum(g.face_areas[inlet])

# ------------------------------------------------------------------------------#

def summarize_data(mesh_id, solver_names):

    file_out = "UiB_"+mesh_id+".csv"

    with open(file_out, "w") as f_out:
        for solver_name in solver_names:
            file_in = solver_name + "_results_" + mesh_id + "/info.txt"
            with open(file_in, "r") as f_in:
                f_out.write(f_in.readlines()[0]+"\n")

# ------------------------------------------------------------------------------#

def main(grid_file, folder, solver, solver_name, dt):

    tol = 1e-8
    gb, domain = problem_data.create_grid(grid_file)

    data = {"domain": domain, "t_max": 1}
    data["dt"] = dt

    problem_data.add_data(gb, data, solver_name)

    solver(gb, folder)

    outflow_upper, outflow_lower = outlet_fluxes(gb)
    mean = mean_inlet_pressure(gb, solver_name)

    # to store the results for the current problem
    results = np.empty(1+3+3, dtype=np.object)

    # save basic informations
    results[0] = "UiB-" + solver_name.upper()
    results[1] = np.sum([g.num_cells for g in gb.grids_of_dimension(3)])
    results[2] = np.sum([g.num_cells for g in gb.grids_of_dimension(2)])
    results[3] = np.sum([g.num_cells for g in gb.grids_of_dimension(1)])
    results[4] = outflow_upper
    results[5] = outflow_lower
    results[6] = mean

    file_name = folder + '/info.txt'
    with open(file_name, "w") as f:
        f.write(", ".join(map(str, results)))

    solvers.transport(gb, data, solver_name, folder,
                      problem_data.AdvectiveDataAssigner,
                      callback=report_concentrations)

# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    grid_files = ["mesh30k.geo", "mesh140k.geo"]
    solver_list = [solvers.solve_tpfa, solvers.solve_mpfa, solvers.solve_vem,
                   solvers.solve_rt0]
    solver_names = ['tpfa', 'mpfa', 'vem', 'rt0']
    time_step = 0.01

    for solver, solver_name in zip(solver_list, solver_names):
        for mesh_id, gf in enumerate(grid_files):
            folder = solver_name + '_results_' + str(mesh_id)
            main(gf, folder, solver, solver_name, time_step)

    # collect all the data
    for mesh_id, _ in enumerate(grid_files):
        summarize_data(str(mesh_id), solver_names)
