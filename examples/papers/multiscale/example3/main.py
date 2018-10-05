import numpy as np
import os
import scipy.sparse as sps
import porepy as pp

from data import Data

from examples.papers.multiscale.multiscale import Multiscale
from examples.papers.multiscale.domain_decomposition import DomainDecomposition

# ------------------------------------------------------------------------------#

def compute_error(gb):

    err = np.zeros(2)
    for g in gb.grids_of_dimension(1):
        d = gb.node_props(g)

        err[0] += np.linalg.norm(d["pressure_old"]-d["pressure"])**2
        err[1] += np.linalg.norm(d["discharge_old"]-d["discharge"])**2

    err = np.sqrt(err)

    for g, d in gb:
        d["pressure_old"] = d["pressure"]
        d["discharge_old"] = d["discharge"]

    return err

# ------------------------------------------------------------------------------#

def export(gb, x, name, solver_flow):

    solver_flow.split(gb, "up", x)

    gb.add_node_props("pressure")
    solver_flow.extract_p(gb, "up", "pressure")
    solver_flow.extract_u(gb, "up", "discharge")
    solver_flow.project_u(gb, "discharge", "P0u")

    save = pp.Exporter(gb, "rt0", folder=name)
    save.write_vtk(["pressure", "P0u"])

# ------------------------------------------------------------------------------#

def write_out(gb, file_name, data):

    cell_2d = np.sum([g.num_cells for g in gb.grids_of_dimension(2)])
    cell_1d = np.sum([g.num_cells for g in gb.grids_of_dimension(1)])

    with open(file_name, "a") as f:
        f.write(", ".join(map(str, [cell_2d, cell_1d, data])) + "\n")

# ------------------------------------------------------------------------------#

def summarize_data(betas):

    data_dd = np.genfromtxt("dd.txt", delimiter = ",", dtype=np.int)[:, 2]
    data_dd = data_dd.reshape(betas.size, 1)

    data_ms = np.genfromtxt("ms.txt", delimiter = ",", dtype=np.int)[:, 2]
    data_ms = data_ms.reshape(betas.size, 1)

    data = np.zeros((betas.size, 3))
    data[:, 0] = betas
    data[:, 1] = data_dd[:, 0]
    data[:, 2] = data_ms[:, 0]

    name = "results.csv"
    np.savetxt(name, data, delimiter=' & ', fmt='%f', newline=' \\\\\n')

    # remove the // from the end of the file
    with open(name, 'rb+') as f:
        f.seek(-3, os.SEEK_END)
        f.truncate()

# ------------------------------------------------------------------------------#

def main_ms(pb_data):
    # in principle we can re-compute only the matrices related to the
    # fracture network, however to simplify the implementation we re-compute everything

    data = Data(pb_data)
    data.add_to_gb()

    # Choose and define the solvers and coupler
    solver_flow = pp.RT0MixedDim("flow")
    A, b = solver_flow.matrix_rhs(data.gb, return_bmat=True)

    # off-line computation fo the bases
    ms = Multiscale(data.gb)
    ms.extract_blocks_h(A, b)
    info = ms.compute_bases()

    # solve the co-dimensional problem
    x_l = ms.solve_l(A, b)
    solver_flow.split(data.gb, "up", ms.concatenate(None, x_l))
    solver_flow.extract_p(data.gb, "up", "pressure")
    solver_flow.extract_p(data.gb, "up", "pressure_old")
    solver_flow.extract_u(data.gb, "up", "discharge_old")

    i = 0
    err = np.inf
    while np.any(err > pb_data["fix_pt_err"]) and i < pb_data["fix_pt_maxiter"]:

        # update the non-linear term
        solver_flow.project_u(data.gb, "discharge_old", "P0u")
        data.update(solver_flow)

        # we need to recompute the lower dimensional matrices
        # for simplicity we do for everything
        A, b = solver_flow.matrix_rhs(data.gb, return_bmat=True)
        x_l = ms.solve_l(A, b)

        solver_flow.split(data.gb, "up", ms.concatenate(None, x_l))
        solver_flow.extract_p(data.gb, "up", "pressure")
        solver_flow.extract_u(data.gb, "up", "discharge")

        err = compute_error(data.gb)
        i += 1

    # print the summary data
    print("ms")
    print("beta", pb_data["beta"], "gamma", pb_data["gamma"], "iter", i, "err", err, "solve_h", info["solve_h"])

    # post-compute the higher dimensional solution
    x_h = ms.solve_h(x_l)

    # update the number of solution of the higher dimensional problem
    info["solve_h"] += 1

    x = ms.concatenate(x_h, x_l)

    folder = "ms_" + str(pb_data["beta"]) + "_" + str(pb_data["gamma"])
    export(data.gb, x, folder, solver_flow)
    write_out(data.gb, "ms.txt", info["solve_h"])

# ------------------------------------------------------------------------------#

def main_dd(pb_data):
    # in principle we can re-compute only the matrices related to the
    # fracture network, however to simplify the implementation we re-compute everything

    data = Data(pb_data)
    data.add_to_gb()

    # parameters for the dd algorightm
    tol = 1e-6
    maxiter = 1e4
    drop_tol = 0.1*np.amin([1e-3, data.eff_kf_t(), data.eff_kf_n()])
    solve_h = 0

    # Choose and define the solvers and coupler
    solver_flow = pp.RT0MixedDim("flow")
    A, b = solver_flow.matrix_rhs(data.gb, return_bmat=True)

    dd = DomainDecomposition(data.gb)

    # compute the initial guess
    dd.extract_blocks(A, b)
    dd.factorize()

    x, info = dd.solve(tol, maxiter, drop_tol, info=True)
    solve_h += info["solve_h"]
    solver_flow.split(data.gb, "up", x)
    solver_flow.extract_p(data.gb, "up", "pressure")
    solver_flow.extract_p(data.gb, "up", "pressure_old")
    solver_flow.extract_u(data.gb, "up", "discharge_old")

    i = 0
    err = np.inf
    while np.any(err > pb_data["fix_pt_err"]) and i < pb_data["fix_pt_maxiter"]:

        # update the non-linear term
        solver_flow.project_u(data.gb, "discharge_old", "P0u")
        data.update(solver_flow)

        # compute the current solution
        A, b = solver_flow.matrix_rhs(data.gb, return_bmat=True)
        dd.extract_blocks(A, b)
        dd.factorize()
        x, info = dd.solve(tol, maxiter, drop_tol, info=True)
        solve_h += info["solve_h"]

        solver_flow.split(data.gb, "up", x)
        solver_flow.extract_p(data.gb, "up", "pressure")
        solver_flow.extract_u(data.gb, "up", "discharge")

        # error evaluation
        err = compute_error(data.gb)
        i += 1

    # print the summary data
    print("dd")
    print("beta", pb_data["beta"], "gamma", pb_data["gamma"], "iter", i, "err", err, "solve_h", solve_h)

    folder = "dd_" + str(pb_data["beta"]) + "_" + str(pb_data["gamma"])
    export(data.gb, x, folder, solver_flow)
    write_out(data.gb, "dd.txt", solve_h)

# ------------------------------------------------------------------------------#

def main(pb_data):

    data = Data(pb_data)
    data.add_to_gb()

    # Choose and define the solvers and coupler
    solver_flow = pp.RT0MixedDim("flow")

    # Compute the initial guess
    A, b = solver_flow.matrix_rhs(data.gb)
    x = sps.linalg.spsolve(A, b)
    solver_flow.split(data.gb, "up", x)
    solver_flow.extract_p(data.gb, "up", "pressure_old")
    solver_flow.extract_p(data.gb, "up", "pressure")
    solver_flow.extract_u(data.gb, "up", "discharge_old")

    i = 0
    err = np.inf
    while np.any(err > pb_data["fix_pt_err"]) and i < pb_data["fix_pt_maxiter"]:

        solver_flow.project_u(data.gb, "discharge_old", "P0u")
        data.update(solver_flow)

        A, b = solver_flow.matrix_rhs(data.gb)
        x = sps.linalg.spsolve(A, b)

        solver_flow.split(data.gb, "up", x)
        solver_flow.extract_p(data.gb, "up", "pressure")
        solver_flow.extract_u(data.gb, "up", "discharge")

        err = compute_error(data.gb)
        i += 1

    # print the summary data
    print("ref")
    print("beta", pb_data["beta"], "gamma", pb_data["gamma"], "iter", i, "err", err, "solve_h", i)

    folder = "ref_" + str(pb_data["beta"]) + "_" + str(pb_data["gamma"])
    export(data.gb, x, folder, solver_flow)

# ------------------------------------------------------------------------------#

if __name__ == "__main__":

    gammas = np.array([0.5, 5, 7.5])
    for gamma in gammas:
        data = {"kf_n": 1e4,
                "kf_t": 1e4,
                "aperture": 1e-4,
                "beta": 20,
                "gamma": gamma,
                "alpha": 1,
                "mesh_size": 0.045,
                "fix_pt_err": 1e-5,
                "fix_pt_maxiter": 1e3}

        main_ms(data)
        main_dd(data)
        main(data)

    #summarize_data(betas, gammas)
