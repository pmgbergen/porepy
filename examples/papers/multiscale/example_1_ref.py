import numpy as np
import scipy.sparse as sps
import porepy as pp

# ------------------------------------------------------------------------------#


def add_data(gb, domain, kf):
    """
    Define the permeability, apertures, boundary conditions
    """
    gb.add_node_props(["param"])
    tol = 1e-5
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

            labels = np.array(["neu"] * bound_faces.size)
            labels[np.logical_or(right, left)] = "dir"

            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces[left]] = 0
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

        gamma = check_P * gb.node_props(g_l, "param").get_aperture()
        d["kn"] = kf * np.ones(mg.num_cells) / gamma


# ------------------------------------------------------------------------------#


def write_network(file_name):
    network = "FID,START_X,START_Y,END_X,END_Y\n"
    network += "0,0.5,0,0.5,1\n"
    network += "1,0.25,0.5,1,0.5"
    with open(file_name, "w") as text_file:
        text_file.write(network)


# ------------------------------------------------------------------------------#


def make_grid_bucket(mesh_size, is_coarse=False):
    mesh_kwargs = {}
    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size / 20}

    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

    file_name = "network_geiger.csv"
    write_network(file_name)
    gb = pp.importer.dfm_2d_from_csv(file_name, mesh_kwargs, domain)
    gb.compute_geometry()
    if is_coarse:
        pp.coarsening.coarsen(gb, "by_volume")
    gb.assign_node_ordering()
    return gb, domain


# ------------------------------------------------------------------------------#


def main(kf, is_coarse=False):

    mesh_size = 0.045
    gb, domain = make_grid_bucket(mesh_size)

    # Assign parameters
    add_data(gb, domain, kf)

    # Choose and define the solvers and coupler
    solver_flow = pp.DualVEMMixedDim("flow")
    A, b = solver_flow.matrix_rhs(gb)
    up = sps.linalg.spsolve(A, b)
    solver_flow.split(gb, "up", up)

    gb.add_node_props(["discharge", "pressure", "P0u"])
    solver_flow.extract_u(gb, "up", "discharge")
    solver_flow.extract_p(gb, "up", "pressure")
    solver_flow.project_u(gb, "discharge", "P0u")

    save = pp.Exporter(gb, "vem", folder="vem_ref")
    save.write_vtk(["pressure", "P0u"])


# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    kf = 1e-2
    main(kf)
