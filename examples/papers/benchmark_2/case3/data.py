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
                [
                    [0.05, 0.25, 0.05],
                    [0.95, 0.25, 0.05],
                    [0.95, 2, 0.05],
                    [0.05, 2, 0.05],
                ]
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
                [[0.05, 1, 0.5], [0.95, 1, 0.5], [0.95, 2, 0.85], [0.05, 2, 0.85]]
            ).T
        )
        f4 = pp.Fracture(
            np.array(
                [[0.05, 1, 0.49], [0.95, 1, 0.49], [0.95, 2, 0.14], [0.05, 2, 0.14]]
            ).T
        )
        f5 = pp.Fracture(
            np.array(
                [
                    [0.23, 1.9, 0.195],
                    [0.23, 1.9, 0.795],
                    [0.17, 2.7, 0.795],
                    [0.17, 2.7, 0.195],
                ]
            ).T
        )
        f6 = pp.Fracture(
            np.array(
                [
                    [0.17, 1.9, 0.195],
                    [0.17, 1.9, 0.795],
                    [0.23, 2.7, 0.795],
                    [0.23, 2.7, 0.195],
                ]
            ).T
        )
        f7 = pp.Fracture(
            np.array(
                [
                    [0.77, 1.9, 0.195],
                    [0.77, 1.9, 0.795],
                    [0.77, 2.7, 0.795],
                    [0.77, 2.7, 0.195],
                ]
            ).T
        )
        f8 = pp.Fracture(
            np.array(
                [
                    [0.83, 1.9, 0.195],
                    [0.83, 1.9, 0.795],
                    [0.83, 2.7, 0.795],
                    [0.83, 2.7, 0.195],
                ]
            ).T
        )

        network = pp.FractureNetwork([f1, f2, f3, f4, f5, f6, f7, f8])

        network.impose_external_boundary(domain)
        network.find_intersections()
        network.split_intersections()
        network.to_gmsh("tmp.geo")

        tm = time.time()

        gb = pp.importer.dfm_from_gmsh(fn, 3, network)
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

    is_fv = solver_name == "tpfa" or solver_name == "mpfa"

    gb.add_node_props(["is_tangential", "param", "frac_num"])
    for g, d in gb:

        one_vec = np.ones(g.num_cells)
        zero_vec = np.zeros(g.num_cells)
        empty = np.empty(0)

        param = pp.Parameters(g)
        d["is_tangential"] = True
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
        param.set_tensor("flow", perm)

        # Source term
        param.set_source("flow", zero_vec)

        # Assign apertures
        aperture = np.power(data["aperture"], 3 - g.dim)
        param.set_aperture(aperture)

        # Boundaries
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size != 0:

            b_in, b_out = b_pressure(g)

            labels = np.array(["neu"] * b_faces.size)
            labels[b_out] = "dir"

            bc_val = np.zeros(g.num_faces)
            bc_val[b_faces[b_in]] = -g.face_areas[b_faces[b_in]]
            bc_val[b_faces[b_out]] = 0

            param.set_bc("flow", pp.BoundaryCondition(g, b_faces, labels))
            param.set_bc_val("flow", bc_val)

            g.tags["inlet_faces"] = np.zeros(g.num_faces, dtype=np.bool)
            g.tags["inlet_faces"][b_faces[b_in]] = True

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
        d["kn"] = kn * np.ones(mg.num_cells) / gamma

# ------------------------------------------------------------------------------#

def b_pressure(g):
    assert g.dim == 3
    y_max = g.nodes[1].max()
    b_faces = np.where(g.tags["domain_boundary_faces"])[0]
    null = np.zeros(b_faces.size, dtype=np.bool)
    if b_faces.size == 0:
        return null, null
    else:
        xf = g.face_centers[:, b_faces]
        b_in = np.argwhere(
            np.logical_and(
                np.abs(xf[1]) < 1e-8, np.logical_and(xf[2] > 1./3., xf[2] < 2./3.)
            )
        )
        b_out = np.argwhere(
            np.logical_and(
                np.abs(xf[1] - y_max) < 1e-8,
                np.logical_or(xf[2] < 1./3., xf[2] > 2./3.),
            )
        )

        return b_in, b_out


# ------------------------------------------------------------------------------#


class AdvectiveDataAssigner(pp.ParabolicDataAssigner):
    def __init__(self, g, data, physics="transport", **kwargs):
        self.domain = kwargs["domain"]
        self.max_dim = kwargs.get("max_dim", 3)
        self.phi_f = 0.2

        # define two pieces of the boundary, useful to impose boundary conditions
        self.inflow = np.empty(0)

        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size > 0:
            b_in, b_out = b_pressure(g)

            self.inflow = b_in

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
        return self.phi_f * np.ones(self.grid().num_cells)
