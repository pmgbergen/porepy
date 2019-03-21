import numpy as np
import porepy as pp

# ------------------------------------------------------------------------------#


def import_grid(file_geo, tol):

    frac = pp.Fracture(np.array([[0, 10, 10, 0], [0, 0, 10, 10], [8, 2, 2, 8]]) * 10)
    network = pp.FractureNetwork3d([frac], tol=tol)

    domain = {"xmin": 0, "xmax": 100, "ymin": 0, "ymax": 100, "zmin": 0, "zmax": 100}
    network.impose_external_boundary(domain)
    network.find_intersections()
    network.split_intersections()
    network.to_gmsh("dummy.geo")

    gb = pp.fracture_importer.dfm_from_gmsh(file_geo, 3, network)
    gb.compute_geometry()

    return gb, domain


# ------------------------------------------------------------------------------#


def low_zones(g):
    return g.cell_centers[2, :] > 10


# ------------------------------------------------------------------------------#


def add_data(gb, data, solver_name):
    tol = data["tol"]

    is_fv = solver_name == "tpfa" or solver_name == "mpfa"

    gb.add_node_props(
        [
            "is_tangential",
            "problem",
            "frac_num",
            "low_zones",
            "porosity",
            "Aavatsmark_transmissibilities",
        ]
    )
    for g, d in gb:
        d["low_zones"] = low_zones(g)
        d["Aavatsmark_transmissibilities"] = True

        unity = np.ones(g.num_cells)
        empty = np.empty(0)

        if g.dim == 2:
            d["frac_num"] = g.frac_num * unity
        else:
            d["frac_num"] = -1 * unity

        # set the permeability
        if g.dim == 3:
            kxx = data["km_high"] * unity
            kxx[d["low_zones"]] = data["km_low"]
            if is_fv:
                perm = pp.SecondOrderTensor(3, kxx=kxx)
            else:
                perm = pp.SecondOrderTensor(3, kxx=kxx, kyy=kxx, kzz=kxx)
        else:  # g.dim == 2:
            kxx = data["kf"] * unity
            if is_fv:
                perm = pp.SecondOrderTensor(3, kxx=kxx)
            else:
                perm = pp.SecondOrderTensor(2, kxx=kxx, kyy=kxx, kzz=1)

        # Assign apertures
        aperture = np.power(data["aperture"], 3 - g.dim)
        d["aperture"] = aperture * unity

        # Boundaries
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        bc_val = np.zeros(g.num_faces)
        bc_val_t = np.zeros(g.num_faces)
        if b_faces.size != 0:

            b_face_centers = g.face_centers[:, b_faces]

            b_inflow = np.logical_and(
                b_face_centers[0, :] < 0 + tol, b_face_centers[2, :] > 90 - tol
            )

            b_outflow = np.logical_and(
                b_face_centers[1, :] < 0 + tol, b_face_centers[2, :] < 10 + tol
            )

            labels = np.array(["neu"] * b_faces.size)
            labels[b_inflow] = "dir"
            labels[b_outflow] = "dir"
            bc = pp.BoundaryCondition(g, b_faces, labels)

            bc_val[b_faces[b_inflow]] = 4 * pp.METER
            bc_val[b_faces[b_outflow]] = 1 * pp.METER

            # Transport
            labels = np.array(["neu"] * b_faces.size)
            labels[np.logical_or(b_inflow, b_outflow)] = "dir"
            bc_t = pp.BoundaryCondition(g, b_faces, labels)

            bc_val_t[b_faces[b_inflow]] = 0.01
        else:
            bc = pp.BoundaryCondition(g, empty, empty)
            bc_t = pp.BoundaryCondition(g, empty, empty)

        if g.dim == 3:
            d["porosity"] = data["porosity_high"] * unity
            d["porosity"][low_zones(g)] = data["porosity_low"]
        else:
            d["porosity"] = data["porosity_f"] * unity

        specified_parameters_f = {
            "second_order_tensor": perm,
            "aperture": aperture * unity,
            "bc": bc,
            "bc_values": bc_val,
        }
        specified_parameters_t = {
            "aperture": aperture * unity,
            "bc": bc_t,
            "bc_values": bc_val_t,
            "time_step": data["time_step"],
            "mass_weight": d["porosity"],
        }
        pp.initialize_default_data(g, d, "flow", specified_parameters_f)
        pp.initialize_default_data(g, d, "transport", specified_parameters_t)

    # Assign coupling permeability, the aperture is read from the lower dimensional grid
    for _, d in gb.edges():
        mg = d["mortar_grid"]
        kn = 2 * data["kf"] * np.ones(mg.num_cells) / data["aperture"]
        d[pp.PARAMETERS] = pp.Parameters(
            mg, ["flow", "transport"], [{"normal_diffusivity": kn}, {}]
        )
        d[pp.DISCRETIZATION_MATRICES] = {"flow": {}, "transport": {}}


# ------------------------------------------------------------------------------#
