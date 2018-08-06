import numpy as np
import scipy.sparse as sps
import porepy as pp

from multiscale import Multiscale
from domain_decomposition import DomainDecomposition

# ------------------------------------------------------------------------------#


def add_data(gb, domain, kf):
    """
    Define the permeability, apertures, boundary conditions
    """
    gb.add_node_props(["param"])
    tol = 1e-8
    a = 1e-2

    for g, d in gb:
        param = pp.Parameters(g)

        # Permeability
        kxx = np.ones(g.num_cells) * np.power(kf, g.dim < gb.dim_max())
        if g.dim == 2:
            perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=kxx, kzz=1)
        else:
            perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=1, kzz=1)
        param.set_tensor("flow", perm)

        # Source term
        param.set_source("flow", np.zeros(g.num_cells))

        # Assign apertures
        aperture = np.power(a, gb.dim_max() - g.dim)
        param.set_aperture(np.ones(g.num_cells) * aperture)

        # Boundaries
        bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            left = bound_face_centers[0, :] < domain["xmin"] + tol
            right = bound_face_centers[0, :] > domain["xmax"] - tol

            labels = np.array(["dir"] * bound_faces.size)


#            labels[right] = "dir"
#            labels[left] = "dir" ####
#
            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces] = g.face_centers[1, bound_faces]
#            #bc_val[bound_faces[left]] = -aperture * g.face_areas[bound_faces[left]]
#            bc_val[bound_faces[right]] = 1

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

        gamma = check_P * gb.node_props(g_l, "param").get_aperture()
        d["kn"] = kf * np.ones(mg.num_cells) / gamma

# ------------------------------------------------------------------------------#


def write_network(file_name):
    network = "FID,START_X,START_Y,END_X,END_Y\n"
    network += "0,0.5,0.5,1,0.5"

#    network += "0,0,0.5,1,0.5\n"
#    network += "1,0.5,0,0.5,1\n"
#    network += "2,0.5,0.75,1,0.75\n"
#    network += "3,0.75,0.5,0.75,1\n"
#    network += "4,0.5,0.625,0.75,0.625\n"
#    network += "5,0.625,0.5,0.625,0.75\n"

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

def main_ms(kf, name):

    mesh_size = 0.045
    gb, domain = make_grid_bucket(mesh_size)

    # Assign parameters
    add_data(gb, domain, kf)

    # Choose and define the solvers and coupler
    solver_flow = pp.RT0MixedDim("flow")
    A, b = solver_flow.matrix_rhs(gb, return_bmat=True)

    # off-line computation fo the bases
    ms = Multiscale(gb)
    ms.extract_blocks_h(A, b)
    ms.compute_bases(is_mixed=True)

    # solve the co-dimensional problem
    A_l, b_l = ms.codim_problem(A, b)
    x_l = sps.linalg.spsolve(A_l, b_l)

    # post-compute the higher dimensional solution
    x_h = ms.compute_sol_h(x_l)
    print(x_h)

    x = ms.concatenate(x_h, x_l)
    solver_flow.split(gb, "up", x)

    gb.add_node_props("pressure")
    solver_flow.extract_p(gb, "up", "pressure")

    save = pp.Exporter(gb, "rt0", folder="ms_"+name)
    save.write_vtk("pressure")

# ------------------------------------------------------------------------------#

def main_dd(kf, name):

    mesh_size = 0.045
    #mesh_size = 0.5
    gb, domain = make_grid_bucket(mesh_size)
    #pp.plot_grid(gb, info="all", alpha=0)

    # Assign parameters
    add_data(gb, domain, kf)

    # Choose and define the solvers and coupler
    solver_flow = pp.RT0MixedDim("flow")
    A, b = solver_flow.matrix_rhs(gb, return_bmat=True)

    dd = DomainDecomposition(gb)
    dd.extract_blocks(A, b)

    x_0 = np.zeros(dd.ndof())
    tol = 1e-8
    max_it = 1000
    x = dd.solve(x_0, tol, max_it)

    solver_flow.split(gb, "up", x)

    gb.add_node_props("pressure")
    solver_flow.extract_p(gb, "up", "pressure")

    save = pp.Exporter(gb, "rt0", folder="dd_"+name)
    save.write_vtk("pressure")

# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    kf = 1e0
    #main_ms(kf, "blocking")
    main_dd(kf, "blocking")

    #kf = 1e4
    #main_dd(kf, "permeable")
