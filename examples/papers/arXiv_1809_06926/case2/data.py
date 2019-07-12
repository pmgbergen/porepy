import csv
import numpy as np
import porepy as pp

# ------------------------------------------------------------------------------#


def import_grid(file_geo, tol):
    # define the mesh size
    file_csv = "fracture_geometry.csv"
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}

    network = pp.fracture_importer.network_3d_from_csv(file_csv, tol=tol)
    network.impose_external_boundary(domain)
    network.find_intersections()
    network.split_intersections()
    network.to_gmsh("dummy.geo")

    gb = pp.fracture_importer.dfm_from_gmsh(file_geo, 3, network)
    gb.compute_geometry()

    return gb, domain


# ------------------------------------------------------------------------------#


def make_grid_cart(N):
    file_csv = "fracture_geometry.csv"

    frac_list = []
    # Extract the data from the csv file
    with open(file_csv, "r") as csv_file:
        spam_reader = csv.reader(csv_file, delimiter=",")

        # Read the domain first
        domain = np.asarray(next(spam_reader), dtype=np.float)
        if domain.size != 6:
            raise ValueError("It has to be 6")
        domain = {
            "xmin": domain[0],
            "xmax": domain[3],
            "ymin": domain[1],
            "ymax": domain[4],
            "zmin": domain[2],
            "zmax": domain[5],
        }

        for row in spam_reader:
            # Read the points
            pts = np.asarray(row, dtype=np.float)
            frac_list.append(pts.reshape((3, -1), order="F"))

    return pp.meshing.cart_grid(frac_list, [N] * 3, physdims=[1] * 3), domain


# ------------------------------------------------------------------------------#


def color(g):
    if g.dim < 3:
        return np.zeros(g.num_cells, dtype=np.int)

    val = np.zeros(g.cell_centers.shape[1], dtype=np.int)
    x = g.cell_centers[0, :]
    y = g.cell_centers[1, :]
    z = g.cell_centers[2, :]

    val[np.logical_and.reduce((x < 0.5, y < 0.5, z < 0.5))] = 0
    val[np.logical_and.reduce((x > 0.5, y < 0.5, z < 0.5))] = 1
    val[np.logical_and.reduce((x < 0.5, y > 0.5, z < 0.5))] = 2
    val[np.logical_and.reduce((x > 0.5, y > 0.5, z < 0.5))] = 3
    val[np.logical_and.reduce((x < 0.5, y < 0.5, z > 0.5))] = 4
    val[np.logical_and.reduce((x > 0.5, y < 0.5, z > 0.5))] = 5
    val[np.logical_and.reduce((x < 0.5, y > 0.5, z > 0.5))] = 6

    val[np.logical_and.reduce((x > 0.75, y > 0.75, z > 0.75))] = 7
    val[np.logical_and.reduce((x > 0.75, y > 0.5, y < 0.75, z > 0.75))] = 8
    val[np.logical_and.reduce((x > 0.5, x < 0.75, y > 0.75, z > 0.75))] = 9
    val[np.logical_and.reduce((x > 0.5, x < 0.75, y > 0.5, y < 0.75, z > 0.75))] = 10
    val[np.logical_and.reduce((x > 0.75, y > 0.75, z > 0.5, z < 0.75))] = 11
    val[np.logical_and.reduce((x > 0.75, y > 0.5, y < 0.75, z > 0.5, z < 0.75))] = 12
    val[np.logical_and.reduce((x > 0.5, x < 0.75, y > 0.75, z > 0.5, z < 0.75))] = 13

    val[
        np.logical_and.reduce(
            (x > 0.5, x < 0.625, y > 0.5, y < 0.625, z > 0.5, z < 0.625)
        )
    ] = 14
    val[
        np.logical_and.reduce(
            (x > 0.625, x < 0.75, y > 0.5, y < 0.625, z > 0.5, z < 0.625)
        )
    ] = 15
    val[
        np.logical_and.reduce(
            (x > 0.5, x < 0.625, y > 0.625, y < 0.75, z > 0.5, z < 0.625)
        )
    ] = 16
    val[
        np.logical_and.reduce(
            (x > 0.625, x < 0.75, y > 0.625, y < 0.75, z > 0.5, z < 0.625)
        )
    ] = 17
    val[
        np.logical_and.reduce(
            (x > 0.5, x < 0.625, y > 0.5, y < 0.625, z > 0.625, z < 0.75)
        )
    ] = 18
    val[
        np.logical_and.reduce(
            (x > 0.625, x < 0.75, y > 0.5, y < 0.625, z > 0.625, z < 0.75)
        )
    ] = 19
    val[
        np.logical_and.reduce(
            (x > 0.5, x < 0.625, y > 0.625, y < 0.75, z > 0.625, z < 0.75)
        )
    ] = 20
    val[
        np.logical_and.reduce(
            (x > 0.625, x < 0.75, y > 0.625, y < 0.75, z > 0.625, z < 0.75)
        )
    ] = 21

    return val


# ------------------------------------------------------------------------------#


def low_zones(g):
    if g.dim < 3:
        return np.zeros(g.num_cells, dtype=np.bool)

    zone_0 = np.logical_and(g.cell_centers[0, :] > 0.5, g.cell_centers[1, :] < 0.5)

    zone_1 = np.logical_and.reduce(
        tuple(
            [
                g.cell_centers[0, :] > 0.75,
                g.cell_centers[1, :] > 0.5,
                g.cell_centers[1, :] < 0.75,
                g.cell_centers[2, :] > 0.5,
            ]
        )
    )

    zone_2 = np.logical_and.reduce(
        tuple(
            [
                g.cell_centers[0, :] > 0.625,
                g.cell_centers[0, :] < 0.75,
                g.cell_centers[1, :] > 0.5,
                g.cell_centers[1, :] < 0.625,
                g.cell_centers[2, :] > 0.5,
                g.cell_centers[2, :] < 0.75,
            ]
        )
    )

    return np.logical_or.reduce(tuple([zone_0, zone_1, zone_2]))


# ------------------------------------------------------------------------------#


def add_data(gb, data, solver_name):
    tol = data["tol"]

    is_fv = solver_name == "tpfa" or solver_name == "mpfa"

    gb.add_node_props(
        [
            "is_tangential",
            "frac_num",
            "porosity",
            "Aavatsmark_transmissibilities",
        ]
    )
    for g, d in gb:
        d["is_tangential"] = True
        d["Aavatsmark_transmissibilities"] = True

        low_zone = low_zones(g)
        d[pp.STATE] = {}
        d[pp.STATE]["low_zones"] = low_zone
        d[pp.STATE]["color"] = color(g)

        unity = np.ones(g.num_cells)
        empty = np.empty(0)

        if g.dim == 2:
            d["frac_num"] = g.frac_num * unity
        else:
            d["frac_num"] = -1 * unity

        # set the permeability
        if g.dim == 3:
            kxx = data["km"] * unity
            kxx[low_zone] = data["km_low"]
            if is_fv:
                perm = pp.SecondOrderTensor(3, kxx=kxx)
            else:
                perm = pp.SecondOrderTensor(3, kxx=kxx, kyy=kxx, kzz=kxx)
        elif g.dim == 2:
            kxx = data["kf"] * unity
            if is_fv:
                perm = pp.SecondOrderTensor(3, kxx=kxx)
            else:
                perm = pp.SecondOrderTensor(2, kxx=kxx, kyy=kxx, kzz=1)
        else:  # dim == 1
            kxx = data["kf"] * unity
            if is_fv:
                perm = pp.SecondOrderTensor(3, kxx=kxx)
            else:
                perm = pp.SecondOrderTensor(1, kxx=kxx, kyy=1, kzz=1)

        # Assign apertures
        aperture = np.power(data["aperture"], 3 - g.dim)

        # Boundaries
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        bc_val = np.zeros(g.num_faces)
        bc_val_t = np.zeros(g.num_faces)

        if b_faces.size != 0:

            b_face_centers = g.face_centers[:, b_faces]
            b_inflow = np.logical_and.reduce(
                tuple(b_face_centers[i, :] < 0.25 - tol for i in range(3))
            )
            b_outflow = np.logical_and.reduce(
                tuple(b_face_centers[i, :] > 0.875 + tol for i in range(3))
            )

            labels = np.array(["neu"] * b_faces.size)
            labels[b_outflow] = "dir"
            bc = pp.BoundaryCondition(g, b_faces, labels)

            f_faces = b_faces[b_inflow]
            bc_val[f_faces] = -aperture * g.face_areas[f_faces]
            bc_val[b_faces[b_outflow]] = 1

            # Transport
            labels = np.array(["neu"] * b_faces.size)
            labels[np.logical_or(b_inflow, b_outflow)] = "dir"
            bc_t = pp.BoundaryCondition(g, b_faces, labels)

            bc_val_t[b_faces[b_inflow]] = 1

        else:
            bc = pp.BoundaryCondition(g, empty, empty)
            bc_t = pp.BoundaryCondition(g, empty, empty)

        if g.dim == 3:
            d["porosity"] = data["porosity_m"] * unity
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
