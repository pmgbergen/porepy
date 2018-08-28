import numpy as np
import porepy as pp


def import_grid(file_geo, tol):

    # define the mesh size
    file_csv = "geiger_3d.csv"
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}

    _, network, _ = pp.importer.network_3d_from_csv(file_csv, tol=tol)
    network.impose_external_boundary(domain)
    network.find_intersections()
    network.split_intersections()
    network.to_gmsh("tmp.geo")

    gb = pp.importer.dfm_from_gmsh(file_geo, 3, network)
    gb.compute_geometry()

    return gb, domain


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

    gb.add_node_props(["is_tangential", "param", "frac_num", "low_zones", "phi"])
    for g, d in gb:
        param = pp.Parameters(g)
        d["is_tangential"] = True
        d["low_zones"] = low_zones(g)

        ones = np.ones(g.num_cells)
        zeros = np.zeros(g.num_cells)
        empty = np.empty(0)

        if g.dim == 2:
            d["frac_num"] = g.frac_num * ones
        else:
            d["frac_num"] = -1 * ones

        # set the permeability
        if g.dim == 3:
            kxx = data["km"] * ones
            kxx[d["low_zones"]] = data["km_low"]
            if is_fv:
                perm = pp.SecondOrderTensor(3, kxx=kxx)
            else:
                perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=kxx, kzz=kxx)
        elif g.dim == 2:
            kxx = data["kf"] * ones
            if is_fv:
                perm = pp.SecondOrderTensor(3, kxx=kxx)
            else:
                perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=kxx, kzz=1)
        else:  # dim == 1
            kxx = data["kf"] * ones
            if is_fv:
                perm = pp.SecondOrderTensor(3, kxx=kxx)
            else:
                perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=1, kzz=1)
        param.set_tensor("flow", perm)

        # Source term
        param.set_source("flow", zeros)

        # Assign apertures
        aperture = np.power(data["aperture"], 3 - g.dim)
        param.set_aperture(aperture * ones)

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
            d["phi"] = data["phi_m"] * ones
        else:
            d["phi"] = data["phi_f"] * ones

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
        self.phi_f = kwargs["phi_f"]
        self.phi_m = kwargs["phi_m"]

        # define two pieces of the boundary, useful to impose boundary conditions
        self.inflow = np.empty(0)

        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size > 0:
            b_face_centers = g.face_centers[:, b_faces]

            val = 0.5 - self.tol
            self.inflow = np.logical_and.reduce(
                tuple(b_face_centers[i, :] < val for i in range(self.max_dim))
            )

        pp.ParabolicDataAssigner.__init__(self, g, data, physics)

    def bc(self):
        b_faces = self.grid().tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size == 0:
            return pp.BoundaryCondition(self.grid(), np.empty(0), np.empty(0))
        return pp.BoundaryCondition(self.grid(), b_faces, "dir")

    def bc_val(self, _):
        bc_val = np.zeros(self.grid().num_faces)
        b_faces = self.grid().tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size > 0:
            bc_val[b_faces[self.inflow]] = 1
        return bc_val

    def porosity(self):
        dim = self.grid().dim
        ones = np.ones(self.grid().num_cells)
        if dim == 3:
            return self.phi_m * ones
        else:
            return self.phi_f * ones
