import numpy as np
import porepy as pp

# ------------------------------------------------------------------------------#


def import_grid(file_geo, tol):

    frac = pp.Fracture(np.array([[0, 10, 10, 0], [0, 0, 10, 10], [8, 2, 2, 8]]) * 10)
    network = pp.FractureNetwork([frac], tol=tol)

    domain = {"xmin": 0, "xmax": 100, "ymin": 0, "ymax": 100, "zmin": 0, "zmax": 100}
    network.impose_external_boundary(domain)
    network.find_intersections()
    network.split_intersections()
    network.to_gmsh("dummy.geo")

    gb = pp.importer.dfm_from_gmsh(file_geo, 3, network)
    gb.compute_geometry()

    return gb, domain


# ------------------------------------------------------------------------------#


def low_zones(g):
    return g.cell_centers[2, :] < 10


# ------------------------------------------------------------------------------#


def add_data(gb, data, solver_name):
    tol = data["tol"]

    is_fv = solver_name == "tpfa" or solver_name == "mpfa"

    gb.add_node_props(["is_tangential", "problem", "frac_num", "low_zones", "phi"])
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
            kxx = data["km_high"] * ones
            kxx[d["low_zones"]] = data["km_low"]
            if is_fv:
                perm = pp.SecondOrderTensor(3, kxx=kxx)
            else:
                perm = pp.SecondOrderTensor(3, kxx=kxx, kyy=kxx, kzz=kxx)
        else:  # g.dim == 2:
            kxx = data["kf"] * ones
            if is_fv:
                perm = pp.SecondOrderTensor(3, kxx=kxx)
            else:
                perm = pp.SecondOrderTensor(2, kxx=kxx, kyy=kxx, kzz=1)
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

            b_top = np.logical_and(
                b_face_centers[0, :] < 0 + tol, b_face_centers[2, :] > 90 - tol
            )

            b_bottom = np.logical_and(
                b_face_centers[1, :] < 0 + tol, b_face_centers[2, :] < 10 + tol
            )

            labels = np.array(["neu"] * b_faces.size)
            labels[b_top] = "dir"
            labels[b_bottom] = "dir"
            param.set_bc("flow", pp.BoundaryCondition(g, b_faces, labels))

            bc_val = np.zeros(g.num_faces)
            bc_val[b_faces[b_top]] = 4 * pp.METER
            bc_val[b_faces[b_bottom]] = 1 * pp.METER
            param.set_bc_val("flow", bc_val)
        else:
            param.set_bc("flow", pp.BoundaryCondition(g, empty, empty))

        d["param"] = param

        if g.dim == 3:
            d["phi"] = data["phi_high"] * ones
            d["phi"][low_zones(g)] = data["phi_low"]
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

        self.phi_high = kwargs["phi_high"]
        self.phi_low = kwargs["phi_low"]
        self.phi_f = kwargs["phi_f"]

        # define two pieces of the boundary, useful to impose boundary conditions
        self.inflow = np.empty(0)

        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size > 0:
            b_face_centers = g.face_centers[:, b_faces]

            self.inflow = np.logical_and(
                b_face_centers[0, :] < 0 + self.tol,
                b_face_centers[2, :] > 90 - self.tol,
            )

        pp.ParabolicDataAssigner.__init__(self, g, data, physics)

    def porosity(self):
        if self.grid().dim == 3:
            phi = self.phi_high * np.ones(self.grid().num_cells)
            phi[low_zones(self.grid())] = self.phi_low
        else:
            phi = self.phi_f * np.ones(self.grid().num_cells)
        return phi

    def bc(self):
        b_faces = self.grid().tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size == 0:
            return pp.BoundaryCondition(self.grid(), np.empty(0), np.empty(0))
        return pp.BoundaryCondition(self.grid(), b_faces, "dir")

    def bc_val(self, _):
        bc_val = np.zeros(self.grid().num_faces)
        b_faces = self.grid().tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size > 0:
            bc_val[b_faces[self.inflow]] = 0.01
        return bc_val
