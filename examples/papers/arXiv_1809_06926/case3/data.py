import pickle
import time
import os

import numpy as np
import porepy as pp

# ------------------------------------------------------------------------------#
def create_grid(fn):

    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 2.25, "zmin": 0, "zmax": 1}
    _, ext = os.path.splitext(fn)

    if ext == ".geo":
        f1 = pp.Fracture(
            np.array(
                [[0.05, 0.25, 0.5], [0.95, 0.25, 0.5], [0.95, 2, 0.5], [0.05, 2, 0.5]]
            ).T
        )
        f2 = pp.Fracture(
            np.array(
                [
                    [0.5, 0.05, 0.95],
                    [0.5, 0.05, 0.05],
                    [0.5, 0.3, 0.05],
                    [0.5, 0.3, 0.95],
                ]
            ).T
        )
        f3 = pp.Fracture(
            np.array(
                [[0.05, 1, 0.5], [0.95, 1, 0.5], [0.95, 2.2, 0.85], [0.05, 2.2, 0.85]]
            ).T
        )
        f4 = pp.Fracture(
            np.array(
                [[0.05, 1, 0.48], [0.95, 1, 0.48], [0.95, 2.2, 0.14], [0.05, 2.2, 0.14]]
            ).T
        )
        f5 = pp.Fracture(
            np.array(
                [[0.23, 1.9, 0.3], [0.23, 1.9, 0.7], [0.17, 2.2, 0.7], [0.17, 2.2, 0.3]]
            ).T
        )
        f6 = pp.Fracture(
            np.array(
                [[0.17, 1.9, 0.3], [0.17, 1.9, 0.7], [0.23, 2.2, 0.7], [0.23, 2.2, 0.3]]
            ).T
        )
        f7 = pp.Fracture(
            np.array(
                [[0.77, 1.9, 0.3], [0.77, 1.9, 0.7], [0.77, 2.2, 0.7], [0.77, 2.2, 0.3]]
            ).T
        )
        f8 = pp.Fracture(
            np.array(
                [[0.83, 1.9, 0.3], [0.83, 1.9, 0.7], [0.83, 2.2, 0.7], [0.83, 2.2, 0.3]]
            ).T
        )

        network = pp.FractureNetwork3d([f1, f2, f3, f4, f5, f6, f7, f8])

        network.impose_external_boundary(domain)
        network.find_intersections()
        network.split_intersections()
        network.to_gmsh("tmp.geo")

        tm = time.time()

        gb = pp.fracture_importer.dfm_from_gmsh(fn, 3, network)
        print("Elapsed time " + str(time.time() - tm))
    elif ext == ".grid":
        gb = pickle.load(open(fn, "rb"))
    else:
        raise ValueError("Not supported data format")

    return gb, domain


# ------------------------------------------------------------------------------#


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
        d["Aavatsmark_transmissibilities"] = True

        unity = np.ones(g.num_cells)
        empty = np.empty(0)

        d["is_tangential"] = True

        if g.dim == 2:
            d["frac_num"] = g.frac_num * unity
        else:
            d["frac_num"] = -1 * unity

        # set the permeability
        if g.dim == 3:
            kxx = matrix_perm * unity
            if is_fv:
                perm = pp.SecondOrderTensor(3, kxx=kxx)
            else:
                perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=kxx, kzz=kxx)
        elif g.dim == 2:
            kxx = fracture_perm * unity
            if is_fv:
                perm = pp.SecondOrderTensor(3, kxx=kxx)
            else:
                perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=kxx, kzz=1)
        else:  # dim == 1
            kxx = fracture_perm * unity
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

            b_in, b_out = b_pressure(g)

            labels = np.array(["neu"] * b_faces.size)
            labels[b_out] = "dir"

            bc_val[b_faces[b_in]] = -g.face_areas[b_faces[b_in]]
            bc_val[b_faces[b_out]] = 0

            bc = pp.BoundaryCondition(g, b_faces, labels)

            g.tags["inlet_faces"] = np.zeros(g.num_faces, dtype=np.bool)
            g.tags["inlet_faces"][b_faces[b_in]] = True

            # Transport
            bc_t = pp.BoundaryCondition(g, b_faces, "dir")
            bc_val_t[b_faces[b_in]] = 1
        else:
            bc = pp.BoundaryCondition(g, empty, empty)
            bc_t = pp.BoundaryCondition(g, empty, empty)

        specified_parameters_f = {
            "second_order_tensor": perm,
            "aperture": aperture_dim * unity,
            "bc": bc,
            "bc_values": bc_val,
        }
        specified_parameters_t = {
            "aperture": aperture_dim * unity,
            "bc": bc_t,
            "bc_values": bc_val_t,
            "time_step": data["time_step"],
            "mass_weight": data["porosity"] * unity,
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


# ------------------------------------------------------------------------------#


def b_pressure(g):
    if g.dim != 3:
        raise ValueError("Only 3d domain is allowed here")
    y_max = g.nodes[1].max()
    y_min = g.nodes[1].min()
    b_faces = np.where(g.tags["domain_boundary_faces"])[0]
    null = np.zeros(b_faces.size, dtype=np.bool)
    if b_faces.size == 0:
        return null, null
    else:
        xf = g.face_centers[:, b_faces]
        b_in = np.argwhere(
            np.logical_and(
                np.abs(xf[1] - y_min) < 1e-8,
                np.logical_and(xf[2] > 1.0 / 3.0, xf[2] < 2.0 / 3.0),
            )
        )
        b_out = np.argwhere(
            np.logical_and(
                np.abs(xf[1] - y_max) < 1e-8,
                np.logical_or(xf[2] < 1.0 / 3.0, xf[2] > 2.0 / 3.0),
            )
        )

        return b_in, b_out


# ------------------------------------------------------------------------------#
