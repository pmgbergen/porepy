import numpy as np
import os
import scipy.sparse as sps
import porepy as pp

#from multiscale import Multiscale
#from domain_decomposition import DomainDecomposition

# ------------------------------------------------------------------------------#


def add_data(gb, domain, kf_t, kf_n):
    """
    Define the permeability, apertures, boundary conditions
    """
    gb.add_node_props(["param", "is_tangential"])
    tol = 1e-8
    a = 1e-4

    for g, d in gb:
        d["is_tangential"] = True
        param = pp.Parameters(g)

        # Permeability
        if g.dim == 2:
            kxx = np.ones(g.num_cells)
            perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=kxx, kzz=1)
        else:
            kxx = kf_t * np.ones(g.num_cells)
            perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=1, kzz=1)
        param.set_tensor("flow", perm)

        # Source term
        param.set_source("flow", np.zeros(g.num_cells))

        # Assign apertures
        aperture = np.power(a, gb.dim_max() - g.dim)
        param.set_aperture(np.ones(g.num_cells) * aperture)

        # Boundaries
        bound_faces = g.get_boundary_faces()
        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            left = bound_face_centers[0, :] < domain["xmin"] + tol
            right = bound_face_centers[0, :] > domain["xmax"] - tol

            labels = np.array(["neu"] * bound_faces.size)
            labels[right] = "dir"

            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces[left]] = -aperture * g.face_areas[bound_faces[left]]
            bc_val[bound_faces[right]] = 1

            param.set_bc("flow", pp.BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val("flow", bc_val)
        else:
            param.set_bc("flow", pp.BoundaryCondition(g, np.empty(0), np.empty(0)))

        d["param"] = param

    # Assign coupling permeability
    gb.add_edge_props("kn")
    for e, d in gb.edges():
        g_l = gb.nodes_of_edge(e)[0]
        mg = d["mortar_grid"]
        check_P = mg.low_to_mortar_avg()

        gamma = np.power(
            check_P * gb.node_props(g_l, "param").get_aperture(),
            1. / (gb.dim_max() - g_l.dim),
        )

        d["kn"] = kf_n / gamma

    return a

# ------------------------------------------------------------------------------#


def update_data(gb, solver_flow, kf_t):

    solver_flow.extract_u(gb, "up", "discharge")
    solver_flow.project_u(gb, "discharge", "P0u")

    beta = 1e5
    for g, d in gb:
        if g.dim == 1:
            # define the non-linear relation with u
            u = beta * np.linalg.norm(d["P0u"], axis=0)

            # to trick the code we need to do the following
            kf = 1./(1./kf_t + u)
            perm = pp.SecondOrderTensor(1, kxx=kf, kyy=1, kzz=1)
            d["param"].set_tensor("flow", perm)

# ------------------------------------------------------------------------------#

def compute_error(gb, x, x_old):

    g_h = self.gb.grids_of_dimension(2)[0]

    norm_x_old = np.linalg.norm(x_old)
    err = np.linalg.norm(x - x_old) / (norm_x_old if norm_x_old else 1)


# ------------------------------------------------------------------------------#

def write_network(file_name):
    network = "FID,START_X,START_Y,END_X,END_Y\n"
    network += "0,0,0.5,1,0.5\n"
    network += "1,0.5,0,0.5,1\n"
    network += "2,0.5,0.75,1,0.75\n"
    network += "3,0.75,0.5,0.75,1\n"
    network += "4,0.5,0.625,0.75,0.625\n"
    network += "5,0.625,0.5,0.625,0.75\n"

    with open(file_name, "w") as text_file:
        text_file.write(network)


# ------------------------------------------------------------------------------#


def make_grid_bucket(mesh_size):
    mesh_kwargs = {}
    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size / 20}

    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

    file_name = "network_geiger.csv"
    write_network(file_name)
    gb = pp.importer.dfm_2d_from_csv(file_name, mesh_kwargs, domain)
    gb.compute_geometry()
    return gb, domain

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

def summarize_data(size):

    data = np.zeros((size, 6))

    values = np.genfromtxt("dd_permeable.txt", delimiter = ",", dtype=np.int)
    # save the number of cells
    data[:, 0:2] = values[:, 0:2]
    # save the other data
    data[:, 2] = values[:, 2]
    data[:, 3] = np.genfromtxt("ms_permeable.txt", delimiter = ",", dtype=np.int)[:, 2]
    data[:, 4] = np.genfromtxt("dd_blocking.txt", delimiter = ",", dtype=np.int)[:, 2]
    data[:, 5] = np.genfromtxt("ms_blocking.txt", delimiter = ",", dtype=np.int)[:, 2]

    name = "results.csv"
    np.savetxt(name, data, delimiter=' & ', fmt='%d', newline=' \\\\\n')

    # remove the // from the end of the file
    with open(name, 'rb+') as f:
        f.seek(-3, os.SEEK_END)
        f.truncate()

# ------------------------------------------------------------------------------#

def main_ms(kf_t, kf_n, name, mesh_size):
    gb, domain = make_grid_bucket(mesh_size)

    # Assign parameters
    add_data(gb, domain, kf_t, kf_n)

    # Choose and define the solvers and coupler
    solver_flow = pp.RT0MixedDim("flow")
    A, b = solver_flow.matrix_rhs(gb, return_bmat=True)

    # off-line computation fo the bases
    ms = Multiscale(gb)
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

    folder = "ms_" + name + "_" + str(mesh_size)
    export(gb, x, folder, solver_flow)
    write_out(gb, "ms_"+name+".txt", info["solve_h"])

# ------------------------------------------------------------------------------#

def main_dd(kf_t, kf_n, name, mesh_size):
    gb, domain = make_grid_bucket(mesh_size)

    # Assign parameters
    aperture = add_data(gb, domain, kf_t, kf_n)

    # Choose and define the solvers and coupler
    solver_flow = pp.RT0MixedDim("flow")
    A, b = solver_flow.matrix_rhs(gb, return_bmat=True)

    dd = DomainDecomposition(gb)
    dd.extract_blocks(A, b)

    tol = 1e-6
    maxiter = 1e4
    drop_tol = np.amin([1e-4, 0.1*aperture*kf_t, 0.1*kf_n/aperture])

    dd.factorize()
    x, info = dd.solve(tol, maxiter, drop_tol, info=True)
    print(info)

    folder = "dd_" + name + "_" + str(mesh_size)
    export(gb, x, folder, solver_flow)
    write_out(gb, "dd_"+name+".txt", info["solve_h"])

# ------------------------------------------------------------------------------#

def main(kf_t, kf_n, name, mesh_size):
    gb, domain = make_grid_bucket(mesh_size)

    # Assign parameters
    add_data(gb, domain, kf_t, kf_n)

    # Choose and define the solvers and coupler
    solver_flow = pp.RT0MixedDim("flow")

    tol = 1e-4
    maxiter = 1e3

    # Compute the initial guess
    A, b = solver_flow.matrix_rhs(gb)
    x_old = sps.linalg.spsolve(A, b)
    solver_flow.split(gb, "up", x_old)

    i = 0
    err = np.inf
    while err > tol and i < maxiter:

        # update the tangetial fracture permeability
        update_data(gb, solver_flow, kf_t)

        A, b = solver_flow.matrix_rhs(gb)
        x = sps.linalg.spsolve(A, b)
        solver_flow.split(gb, "up", x)

        norm_x_old = np.linalg.norm(x_old)
        err = np.linalg.norm(x - x_old) / (norm_x_old if norm_x_old else 1)

        x_old = x
        i += 1


    print(i, err)

    folder = "ref_" + name + "_" + str(mesh_size)
    export(gb, x, folder, solver_flow)

# ------------------------------------------------------------------------------#

if __name__ == "__main__":

    main(1e-4, 1e-4, "", 0.1)

#    mesh_sizes = 0.45*np.array([1, 1e-1, 1e-2])
#
#    kf = {0: 1e-4, 1: 1e4}
#    # it's (kf_t, kf_n)
#    tests = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#
#    mesh_sizes = [0.45, 0.045, 0.01]
#    #tests = np.array([[0, 0], [1, 1]])
#
#    for t, n in tests:
#        name = str(t) + "_" + str(n)
#        for mesh_size in mesh_sizes:
#            print(name, str(mesh_size))
#            main(kf[t], kf[n], name, mesh_size)
#            main_ms(kf[t], kf[n], name, mesh_size)
#            main_dd(kf[t], kf[n], name, mesh_size)
#
#    summarize_data(mesh_sizes.size)
