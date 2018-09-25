import numpy as np
import porepy as pp

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
        d['aperture'] = aperture*unity
        param.set_aperture(d['aperture'])

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
            bc_val[b_faces[b_left]] = 0 * pp.BAR
            bc_val[b_faces[b_right]] = 1 * pp.BAR
            param.set_bc_val("flow", bc_val)
        else:
            param.set_bc("flow", pp.BoundaryCondition(g, empty, empty))

        d["param"] = param

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
    def __init__(self, g, data_grid, data_problem):
        self.data_problem = data_problem
        self.tol = data_problem["tol"]
        self.domain = data_problem["domain"]

        # define two pieces of the boundary, useful to impose boundary conditions
        self.inflow = np.empty(0)

        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size > 0:
            b_face_centers = g.face_centers[:, b_faces]

            self.inflow = b_face_centers[0, :] > self.domain["xmax"] - self.tol
            self.outflow = b_face_centers[0, :] < self.domain["xmin"] + self.tol

        pp.ParabolicDataAssigner.__init__(self, g, data_grid, "transport")

    def porosity(self):
        unity = np.ones(self.grid().num_cells)
        if self.grid().dim == 2:
            return self.data_problem["rock"].POROSITY * unity
        else:
            return self.data_problem["porosity_f"] * unity

    def diffusivity(self):
        unity = np.ones(self.grid().num_cells)
        kxx = self.data_problem["rock"].thermal_conductivity() * unity
        return pp.SecondOrderTensor(self.grid().dim, kxx)

    def rock_specific_heat(self):
        return self.data_problem["rock"].specific_heat_capacity()

    def fluid_specific_heat(self):
        return self.data_problem["fluid"].specific_heat_capacity()

    def rock_density(self):
        return self.data_problem["rock"].DENSITY

    def fluid_density(self):
        return self.data_problem["fluid"].density()

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

    def aperture(self):
        aperture = np.power(self.data_problem['aperture'], 2-self.grid().dim)
        unity = np.ones(self.grid().num_cells)
        return aperture*unity

