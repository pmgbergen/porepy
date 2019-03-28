import pickle

import numpy as np
import porepy as pp


def create_grid(from_file=True, generate_network=False, tol=1e-3):
    """ Obtain domain and grid bucket. Default is to load a pickled bucket;
    alternatively, a .geo file is available.
    """
    if generate_network:
        file_csv = "fracture_network.csv"
        domain = {
            "xmin": -500,
            "xmax": 350,
            "ymin": 100,
            "ymax": 1500,
            "zmin": -100,
            "zmax": 500,
        }

        network = pp.fracture_importer.network_3d_from_csv(
            file_csv, has_domain=False, tol=tol
        )
        network.impose_external_boundary(domain)
        network.find_intersections()
        network.split_intersections()
        network.to_gmsh("dummy.geo")

        pickle.dump(network, open("network_52_fracs", "wb"))

    network = pickle.load(open("network_52_fracs", "rb"))
    domain = network.domain
    if from_file:
        gb = pickle.load(open("gridbucket_case4.grid", "rb"))
    else:
        gb = pp.fracture_importer.dfm_from_gmsh(
            "gmsh_frac_file.msh", 3, network, ensure_matching_face_cell=True
        )
        pickle.dump(gb, open("gridbucket_case4.grid", "wb"))

    return gb, domain


def add_data(gb, data, solver_name):

    matrix_perm = 1e0
    fracture_perm = 1e4
    aperture = 1e-2
    kn = fracture_perm

    data["aperture"] = aperture
    data["matrix_perm"] = matrix_perm
    data["fracture_perm"] = fracture_perm
    data["kn"] = kn
    data["porosity"] = 0.2

    is_fv = solver_name == "tpfa" or solver_name == "mpfa"

    gb.add_node_props(["is_tangential", "frac_num", "Aavatsmark_transmissibilities"])
    for g, d in gb:

        one_vec = np.ones(g.num_cells)
        empty_vec = np.empty(0)

        d["is_tangential"] = True
        d["Aavatsmark_transmissibilities"] = True
        d["aperture"] = aperture * one_vec

        if g.dim == 2:
            d["frac_num"] = g.frac_num * one_vec
        else:
            d["frac_num"] = -1 * one_vec

        # set the permeability
        if g.dim == 3:
            kxx = matrix_perm * one_vec
            if is_fv:
                perm = pp.SecondOrderTensor(3, kxx=kxx)
            else:
                perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=kxx, kzz=kxx)
        elif g.dim == 2:
            kxx = fracture_perm * one_vec
            if is_fv:
                perm = pp.SecondOrderTensor(3, kxx=kxx)
            else:
                perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=kxx, kzz=1)
        else:  # dim == 1
            kxx = fracture_perm * one_vec
            if is_fv:
                perm = pp.SecondOrderTensor(3, kxx=kxx)
            else:
                perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=1, kzz=1)

        # Assign apertures
        aperture_dim = np.power(data["aperture"], 3 - g.dim)

        # Boundaries
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        bc_val = np.zeros(g.num_faces)
        bc_val_t = np.zeros(g.num_faces)
        if b_faces.size != 0:

            b_in, b_out, _, _ = b_pressure(g)
            if b_in is not None and b_out is not None:

                labels = np.array(["neu"] * b_faces.size)
                labels[b_out] = "dir"

                bc_val = np.zeros(g.num_faces)
                bc_val[b_faces[b_in]] = -g.face_areas[b_faces[b_in]]
                bc_val[b_faces[b_out]] = 0

                bc = pp.BoundaryCondition(g, b_faces, labels)
                g.tags["inlet_faces"] = np.zeros(g.num_faces, dtype=np.bool)
                g.tags["inlet_faces"][b_faces[b_in]] = True
            else:
                bc = pp.BoundaryCondition(g, b_faces, "neu")

            # Transport
            bc_t = pp.BoundaryCondition(g, b_faces, "dir")
            bc_val_t[b_faces[b_in]] = 1
        else:
            bc = pp.BoundaryCondition(g, empty_vec, empty_vec)
            bc_t = pp.BoundaryCondition(g, empty_vec, empty_vec)

        specified_parameters_f = {
            "second_order_tensor": perm,
            "aperture": aperture_dim * one_vec,
            "bc": bc,
            "bc_values": bc_val,
        }
        specified_parameters_t = {
            "aperture": aperture_dim * one_vec,
            "bc": bc_t,
            "bc_values": bc_val_t,
            "time_step": data["time_step"],
            "mass_weight": data["porosity"] * one_vec,
            "porosity": data["porosity"] * one_vec,
        }
        pp.initialize_default_data(g, d, "flow", specified_parameters_f)
        pp.initialize_default_data(g, d, "transport", specified_parameters_t)
    # Assign coupling permeability, the aperture is read from the lower dimensional grid
    for _, d in gb.edges():
        mg = d["mortar_grid"]
        normal_perm = 2 * kn * np.ones(mg.num_cells) / aperture
        d[pp.PARAMETERS] = pp.Parameters(
            mg, ["flow", "transport"], [{"normal_diffusivity": normal_perm}, {}]
        )
        d[pp.DISCRETIZATION_MATRICES] = {"flow": {}, "transport": {}}


def b_pressure(g):
    if g.dim < 3:
        return None, None, None, None
    b_faces = np.where(g.tags["domain_boundary_faces"])[0]
    null = np.zeros(b_faces.size, dtype=np.bool)
    if b_faces.size == 0:
        return null, null, null, null
    else:
        xf = g.face_centers[:, b_faces]
        tol = 1e-3
        b_in = np.argwhere(
            np.logical_or(
                np.logical_and.reduce(
                    (xf[0] - tol < -200, xf[1] + tol > 1500, xf[2] + tol > 300)
                ),
                np.logical_and.reduce(
                    (xf[0] - tol < -500, xf[1] + tol > 1200, xf[2] + tol > 300)
                ),
            )
        )

        b_out_2 = np.logical_and.reduce(
            (xf[0] + tol > 350, xf[1] - tol < 400, xf[2] - tol < 100)
        )

        b_out_1 = np.logical_and.reduce(
            (xf[0] - tol < -500, xf[1] - tol < 400, xf[2] - tol < 100)
        )

        b_out = np.argwhere(np.logical_or(b_out_1, b_out_2))
        b_out_1 = np.argwhere(b_out_1)
        b_out_2 = np.argwhere(b_out_2)

        return b_in, b_out, b_out_1, b_out_2


# ------------------------------------------------------------------------------#
