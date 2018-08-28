import numpy as np
import os
import scipy.sparse as sps
import porepy as pp

from data import Data

from examples.papers.multiscale.multiscale import Multiscale
from examples.papers.multiscale.domain_decomposition import DomainDecomposition

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

def summarize_data(mesh_size, tests):

    for t, n in tests:

        data = np.zeros((mesh_size.size, 4))
        print(data)

        name = "_" + str(t) + "_" + str(n)
        dd = np.genfromtxt("dd"+name+".txt", delimiter = ",", dtype=np.int)
        data[:, 0:3] = np.atleast_2d(dd)
        print()
        ms = np.genfromtxt("ms"+name+".txt", delimiter = ",", dtype=np.int)
        data[:, 3] = np.atleast_2d(ms)[:, 2]

        name = "results"+name+".csv"
        np.savetxt(name, data, delimiter=' & ', fmt='%d', newline=' \\\\\n')

        # remove the // from the end of the file
        with open(name, 'rb+') as f:
            f.seek(-3, os.SEEK_END)
            f.truncate()

# ------------------------------------------------------------------------------#

def main_ms(pb_data, name):

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

    # post-compute the higher dimensional solution
    x_h = ms.solve_h(x_l)

    # update the number of solution of the higher dimensional problem
    info["solve_h"] += 1
    print(info)

    x = ms.concatenate(x_h, x_l)

    folder = "ms_" + name + "_" + str(pb_data["mesh_size"])
    export(data.gb, x, folder, solver_flow)
    write_out(data.gb, "ms_"+name+".txt", info["solve_h"])

# ------------------------------------------------------------------------------#

def main_dd(pb_data, name):

    data = Data(pb_data)
    data.add_to_gb()

    # parameters for the dd algorightm
    tol = 1e-6
    maxiter = 1e4
    drop_tol = 0.1*np.amin([1e-3, data.eff_kf_t(), data.eff_kf_n()])

    # Choose and define the solvers and coupler
    solver_flow = pp.RT0MixedDim("flow")
    A, b = solver_flow.matrix_rhs(data.gb, return_bmat=True)

    dd = DomainDecomposition(data.gb)
    dd.extract_blocks(A, b)

    dd.factorize()
    x, info = dd.solve(tol, maxiter, drop_tol, info=True)
    print(info)

    folder = "dd_" + name + "_" + str(pb_data["mesh_size"])
    export(data.gb, x, folder, solver_flow)
    write_out(data.gb, "dd_"+name+".txt", info["solve_h"])

# ------------------------------------------------------------------------------#

def main(pb_data, name):

    data = Data(pb_data)
    data.add_to_gb()

    # Choose and define the solvers and coupler
    solver_flow = pp.RT0MixedDim("flow")
    A, b = solver_flow.matrix_rhs(data.gb)

    x = sps.linalg.spsolve(A, b)

    folder = "ref_" + name + "_" + str(pb_data["mesh_size"])
    export(data.gb, x, folder, solver_flow)

# ------------------------------------------------------------------------------#

if __name__ == "__main__":

    mesh_sizes = np.array([0.45, 0.045, 0.0275])

    kf = {0: 1e-4, 1: 1e4}
    # it's (kf_t, kf_n)
    tests = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    for t, n in tests:
        name = str(t) + "_" + str(n)
        for mesh_size in mesh_sizes:
            print(name, str(mesh_size))
            data = {"kf_n": kf[n],
                    "kf_t": kf[t],
                    "aperture": 1e-4,
                    "mesh_size": mesh_size}

            main_ms(data, name)
            main_dd(data, name)
            main(data, name)

    summarize_data(mesh_sizes, tests)
