import numpy as np
import porepy as pp

# ------------------------------------------------------------------------------#


def import_grid(file_geo, mesh_args, tol):

    p, e = pp.importer.lines_from_csv(file_geo, mesh_args, polyline=True, skip_header=0)
    p, _ = pp.frac_utils.snap_fracture_set_2d(p, e, snap_tol=tol["snap"])

    domain = {'xmin': p[0].min(), 'xmax': p[0].max(),
              'ymin': p[1].min(), 'ymax': p[1].max()}

    frac_dict = {'points': p, 'edges': e}
    gb = pp.meshing.simplex_grid(frac_dict, domain, tol=tol["geo"], **mesh_args)

    return gb, domain

# ------------------------------------------------------------------------------#


def add_data(gb, data):
    tol = data["tol"]

    gb.add_node_props(['is_tangential', 'frac_num', 'porosity', 'aperture'])
    for g, d in gb:
        param = pp.Parameters(g)
        d["is_tangential"] = True

        unity = np.ones(g.num_cells)
        zeros = np.zeros(g.num_cells)
        empty = np.empty(0)

        if g.dim == 1:
            d['frac_num'] = g.frac_num*unity
        else:
            d['frac_num'] = -1*unity

        # set the permeability
        if g.dim == 2:
            kxx = data['km']*unity
            perm = pp.SecondOrderTensor(2, kxx=kxx, kyy=kxx, kzz=1)
        else: #g.dim == 1:
            kxx = data['kf']*unity
            perm = pp.SecondOrderTensor(1, kxx=kxx, kyy=1, kzz=1)
        param.set_tensor("flow", perm)

        # Source term
        param.set_source("flow", zeros)

        # Assign apertures
        aperture = np.power(data['aperture'], 2-g.dim)
        param.set_aperture(aperture*unity)
        d['aperture'] = aperture*unity

        # Boundaries
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size != 0:

            b_face_centers = g.face_centers[:, b_faces]

            b_left = b_face_centers[0, :] < data["domain"]["xmin"] + tol
            b_right = b_face_centers[0, :] > data["domain"]["xmax"] - tol

            labels = np.array(["neu"] * b_faces.size)
            labels[b_left] = "dir"
            labels[b_right] = "dir"
            param.set_bc("flow", pp.BoundaryCondition(g, b_faces, labels))

            bc_val = np.zeros(g.num_faces)
            bc_val[b_faces[b_left]] = 0 * pp.METER
            bc_val[b_faces[b_right]] = 1 * pp.METER
            param.set_bc_val("flow", bc_val)
        else:
            param.set_bc("flow", pp.BoundaryCondition(g, empty, empty))

        d["param"] = param

        if g.dim == 2:
            d['porosity'] = data['porosity'] * unity
        else:
            d['porosity'] = data['porosity_f'] * unity

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
        self.tol = kwargs["tol"]
        self.domain = kwargs["domain"]

        self.porosity_m = kwargs["porosity"]
        self.porosity_f = kwargs["porosity_f"]

        # define two pieces of the boundary, useful to impose boundary conditions
        self.inflow = np.empty(0)

        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size > 0:
            b_face_centers = g.face_centers[:, b_faces]

            self.inflow = b_face_centers[0, :] > self.domain["xmax"] - self.tol
            self.outflow = b_face_centers[0, :] < self.domain["xmin"] + self.tol

        pp.ParabolicDataAssigner.__init__(self, g, data, physics)

    def porosity(self):
        if self.grid().dim == 2:
            porosity = self.porosity_m * np.ones(self.grid().num_cells)
        else:
            porosity = self.porosity_f * np.ones(self.grid().num_cells)
        return porosity

    def rock_specific_heat(self):
        #hack to remove the rock part
        return 0

    def bc(self):
        b_faces = self.grid().tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size == 0:
            return pp.BoundaryCondition(self.grid(), np.empty(0), np.empty(0))
        else:
            labels = np.array(['neu'] * b_faces.size)
            labels[np.logical_or(self.inflow, self.outflow)] = 'dir'
        return pp.BoundaryCondition(self.grid(), b_faces, labels)

    def bc_val(self, _):
        bc_val = np.zeros(self.grid().num_faces)
        b_faces = self.grid().tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size > 0:
            bc_val[b_faces[self.inflow]] = 0.01
        return bc_val

