import numpy as np
import scipy.sparse as sps
import porepy as pp

# ------------------------------------------------------------------------------#


def add_data(gb, domain, kf):
    """
    Define the permeability, apertures, boundary conditions
    """
    gb.add_node_props(["param", "is_tangential"])
    tol = 1e-5
    a = 1e-4

    for g, d in gb:

        # Permeability
        kxx = np.ones(g.num_cells) * np.power(kf, g.dim < gb.dim_max())
        if g.dim == 2:
            perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=kxx, kzz=1)
        else:
            perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=1, kzz=1)

        # Source term

        # Assign apertures
        a_dim = np.power(a, gb.dim_max() - g.dim)
        aperture = np.ones(g.num_cells) * a_dim

        specified_parameters = {"aperture": aperture, "second_order_tensor": perm}
        # Boundaries
        bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            left = bound_face_centers[0, :] < domain["xmin"] + tol
            right = bound_face_centers[0, :] > domain["xmax"] - tol

            labels = np.array(["neu"] * bound_faces.size)
            labels[right] = "dir"

            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces[left]] = -a_dim * g.face_areas[bound_faces[left]]
            bc_val[bound_faces[right]] = 1

            bound = pp.BoundaryCondition(g, bound_faces, labels)
            specified_parameters.update({"bc": bound, "bc_values": bc_val})
        else:
            bound = pp.BoundaryCondition(g, np.empty(0), np.empty(0))
            specified_parameters.update({"bc": bound})

        d["is_tangential"] = True
        pp.initialize_default_data(g, d, "flow", specified_parameters)

    # Assign coupling permeability
    for e, d in gb.edges():
        g_l = gb.nodes_of_edge(e)[0]
        mg = d["mortar_grid"]
        check_P = mg.slave_to_mortar_avg()
        pa_l = gb.node_props(g_l, pp.PARAMETERS)
        gamma = check_P * pa_l["flow"]["aperture"]
        kn = kf * np.ones(mg.num_cells) / gamma
        d[pp.PARAMETERS] = pp.Parameters(mg, ["flow"], [{"normal_diffusivity": kn}])
        d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}


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


def main(kf, description, is_coarse=False, if_export=False):
    mesh_size = 0.045
    gb, domain = make_grid_bucket(mesh_size, is_coarse)
    # Assign parameters
    add_data(gb, domain, kf)
    key = "flow"

    # Choose and define the solvers and coupler
    discretization_key = key + "_" + pp.DISCRETIZATION
    discr = pp.MVEM(key)

    for _, d in gb:
        d[discretization_key] = discr

    for _, d in gb.edges():
        d[discretization_key] = pp.RobinCoupling(key, discr)

    assembler = pp.EllipticAssembler(key)

    # Discretize
    A, b = assembler.assemble_matrix_rhs(gb)

    # Solve the linear system
    up = sps.linalg.spsolve(A, b)
    assembler.split(gb, "up", up)

    gb.add_node_props(["darcy_flux", "pressure", "P0u"])
    assembler.extract_flux(gb, "up", "darcy_flux")
    assembler.extract_pressure(gb, "up", "pressure")

    # EK: For the mometn, we don't have project_u for the general assembler
    #    assembler.project_u(gb, "darcy_flux", "P0u")

    if if_export:
        save = pp.Exporter(gb, "vem", folder="vem_" + description)
        # save.write_vtk(["pressure", "P0u"])
        save.write_vtk(["pressure"])


# ------------------------------------------------------------------------------#


def test_vem_blocking():
    kf = 1e-4
    if_export = True
    main(kf, "blocking", if_export=if_export)


# ------------------------------------------------------------------------------#


def test_vem_blocking_coarse():
    kf = 1e-4
    if_export = True
    main(kf, "blocking_coarse", is_coarse=True, if_export=if_export)


# ------------------------------------------------------------------------------#


def test_vem_permeable():
    kf = 1e4
    if_export = True
    main(kf, "permeable", if_export=if_export)


# ------------------------------------------------------------------------------#


def test_vem_permeable_coarse():
    kf = 1e4
    if_export = True
    main(kf, "permeable_coarse", is_coarse=True, if_export=if_export)


# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    test_vem_blocking()
    test_vem_blocking_coarse()
    test_vem_permeable()
    test_vem_permeable_coarse()
