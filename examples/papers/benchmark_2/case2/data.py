import csv
import numpy as np
import porepy as pp

# ------------------------------------------------------------------------------#


def import_grid(file_geo, tol):

    # define the mesh size
    file_csv = "fracture_geometry.csv"
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}

    _, network, _ = pp.importer.network_3d_from_csv(file_csv, tol=tol)
    network.impose_external_boundary(domain)
    network.find_intersections()
    network.split_intersections()
    network.to_gmsh("dummy.geo")

    gb = pp.importer.dfm_from_gmsh(file_geo, 3, network)
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

    return pp.meshing.cart_grid(frac_list, [N]*3, physdims=[1]*3), domain

# ------------------------------------------------------------------------------#

def color(g):
    if g.dim < 3:
        return np.zeros(g.num_cells, dtype=np.int)

    val = np.zeros(g.cell_centers.shape[1], dtype=np.int)
    x = g.cell_centers[0, :]
    y = g.cell_centers[1, :]
    z = g.cell_centers[2, :]

    val[np.logical_and.reduce((x<.5, y<.5, z<.5))] = 0
    val[np.logical_and.reduce((x>.5, y<.5, z<.5))] = 1
    val[np.logical_and.reduce((x<.5, y>.5, z<.5))] = 2
    val[np.logical_and.reduce((x>.5, y>.5, z<.5))] = 3
    val[np.logical_and.reduce((x<.5, y<.5, z>.5))] = 4
    val[np.logical_and.reduce((x>.5, y<.5, z>.5))] = 5
    val[np.logical_and.reduce((x<.5, y>.5, z>.5))] = 6

    val[np.logical_and.reduce((x>.75, y>.75, z>.75))] = 7
    val[np.logical_and.reduce((x>.75, y>.5, y<.75, z>.75))] = 8
    val[np.logical_and.reduce((x>.5, x<.75, y>.75, z>.75))] = 9
    val[np.logical_and.reduce((x>.5, x<.75, y>.5, y<.75, z>.75))] = 10
    val[np.logical_and.reduce((x>.75, y>.75, z>.5, z<.75))] = 11
    val[np.logical_and.reduce((x>.75, y>.5, y<.75, z>.5, z<.75))] = 12
    val[np.logical_and.reduce((x>.5, x<.75, y>.75, z>.5, z<.75))] = 13

    val[np.logical_and.reduce((x>.5, x<.625, y>.5, y<.625, z>.5, z<.625))] = 14
    val[np.logical_and.reduce((x>.625, x<.75, y>.5, y<.625, z>.5, z<.625))] = 15
    val[np.logical_and.reduce((x>.5, x<.625, y>.625, y<.75, z>.5, z<.625))] = 16
    val[np.logical_and.reduce((x>.625, x<.75, y>.625, y<.75, z>.5, z<.625))] = 17
    val[np.logical_and.reduce((x>.5, x<.625, y>.5, y<.625, z>.625, z<.75))] = 18
    val[np.logical_and.reduce((x>.625, x<.75, y>.5, y<.625, z>.625, z<.75))] = 19
    val[np.logical_and.reduce((x>.5, x<.625, y>.625, y<.75, z>.625, z<.75))] = 20
    val[np.logical_and.reduce((x>.625, x<.75, y>.625, y<.75, z>.625, z<.75))] = 21

    return val

#------------------------------------------------------------------------------#

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

    gb.add_node_props(["is_tangential", "param", "frac_num", "low_zones", "porosity", "color"])
    for g, d in gb:
        param = pp.Parameters(g)
        d["is_tangential"] = True
        d["low_zones"] = low_zones(g)
        d["color"] = color(g)

        unity = np.ones(g.num_cells)
        zeros = np.zeros(g.num_cells)
        empty = np.empty(0)

        if g.dim == 2:
            d["frac_num"] = g.frac_num * unity
        else:
            d["frac_num"] = -1 * unity

        # set the permeability
        if g.dim == 3:
            kxx = data["km"] * unity
            kxx[d["low_zones"]] = data["km_low"]
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
        param.set_tensor("flow", perm)

        # Source term
        param.set_source("flow", zeros)

        # Assign apertures
        aperture = np.power(data["aperture"], 3 - g.dim)
        param.set_aperture(aperture * unity)
        d["aperture"] = aperture * unity

        # Boundaries
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size != 0:

            b_face_centers = g.face_centers[:, b_faces]
            b_flux = np.logical_and.reduce(
                tuple(b_face_centers[i, :] < 0.5 - tol for i in range(3))
            )
            b_pressure = np.logical_and.reduce(
                tuple(b_face_centers[i, :] > 0.75 + tol for i in range(3))
            )

            labels = np.array(["neu"] * b_faces.size)
            labels[b_pressure] = "dir"
            param.set_bc("flow", pp.BoundaryCondition(g, b_faces, labels))

            bc_val = np.zeros(g.num_faces)
            f_faces = b_faces[b_flux]
            bc_val[f_faces] = -aperture * g.face_areas[f_faces]
            bc_val[b_faces[b_pressure]] = 1
            param.set_bc_val("flow", bc_val)
        else:
            param.set_bc("flow", pp.BoundaryCondition(g, empty, empty))

        d["param"] = param

        if g.dim == 3:
            d["porosity"] = data["porosity_m"] * unity
        else:
            d["porosity"] = data["porosity_f"] * unity

    # Assign coupling permeability, the aperture is read from the lower dimensional grid
    gb.add_edge_props("kn")
    for e, d in gb.edges():
        g_l = gb.nodes_of_edge(e)[0]
        mg = d["mortar_grid"]
        check_P = mg.low_to_mortar_avg()

        gamma = check_P * gb.node_props(g_l, "param").get_aperture()
        d["kn"] = data["kf"] * np.ones(mg.num_cells) / gamma


# ------------------------------------------------------------------------------#


class AdvectiveDataAssigner(pp.ParabolicDataAssigner):
    def __init__(self, g, data, physics="transport", **kwargs):
        self.domain = kwargs["domain"]
        self.tol = kwargs["tol"]
        self.max_dim = kwargs.get("max_dim", 3)
        self.porosity_f = kwargs["porosity_f"]
        self.porosity_m = kwargs["porosity_m"]

        # define two pieces of the boundary, useful to impose boundary conditions
        self.inflow = np.empty(0)

        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size > 0:
            b_face_centers = g.face_centers[:, b_faces]

            self.inflow = np.logical_and.reduce(
                tuple(b_face_centers[i, :] < 0.5 - self.tol for i in range(3))
            )
            self.outflow = np.logical_and.reduce(
                tuple(b_face_centers[i, :] > 0.75 + self.tol for i in range(3))
            )

        pp.ParabolicDataAssigner.__init__(self, g, data, physics)

    def porosity(self):
        if self.grid().dim == 3:
            return self.porosity_m * np.ones(self.grid().num_cells)
        else:
            return self.porosity_f * np.ones(self.grid().num_cells)

    def rock_specific_heat(self):
        # hack to remove the rock part
        return 0

    def bc(self):
        b_faces = self.grid().tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size == 0:
            return pp.BoundaryCondition(self.grid(), np.empty(0), np.empty(0))
        else:
            labels = np.array(["neu"] * b_faces.size)
            labels[np.logical_or(self.inflow, self.outflow)] = "dir"
        return pp.BoundaryCondition(self.grid(), b_faces, labels)

    def bc_val(self, _):
        bc_val = np.zeros(self.grid().num_faces)
        b_faces = self.grid().tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size > 0:
            bc_val[b_faces[self.inflow]] = 1
        return bc_val
