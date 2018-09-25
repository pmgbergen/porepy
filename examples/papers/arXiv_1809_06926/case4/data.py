import pickle

import numpy as np
import porepy as pp


def create_grid(from_file=True):
    """ Obtain domain and grid bucket. Default is to load a pickled bucket;
    alternatively, a .geo file is available.
    """
    network = pickle.load(open('network_52_fracs', 'rb'))
    domain = network.domain
    if from_file:
        gb = pickle.load(open('gridbucket_case4.grid', 'rb'))
    else:
        gb = pp.importer.dfm_from_gmsh("gmsh_frac_file.msh", 3, network,
                                       ensure_matching_face_cell=True)
        pickle.dump(gb, open('gridbucket_case4.grid', 'wb'))

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

    is_fv = solver_name == "tpfa" or solver_name == "mpfa"

    gb.add_node_props(["is_tangential", "param", "frac_num"])
    for g, d in gb:

        one_vec = np.ones(g.num_cells)
        zero_vec = np.zeros(g.num_cells)
        empty_vec = np.empty(0)

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
        param.set_aperture(aperture * one_vec)

        # Boundaries
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size != 0:

            b_in, b_out, _, _ = b_pressure(g)
            if b_in is not None and b_out is not None:

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
                param.set_bc("flow", pp.BoundaryCondition(g, empty_vec, empty_vec))

        else:
            param.set_bc("flow", pp.BoundaryCondition(g, empty_vec, empty_vec))

        d["param"] = param

    # Assign coupling permeability, the aperture is read from the lower dimensional grid
    gb.add_edge_props("kn")
    for e, d in gb.edges():
        g_l = gb.nodes_of_edge(e)[0]
        mg = d["mortar_grid"]
        check_P = mg.low_to_mortar_avg()

        gamma = check_P * gb.node_props(g_l, "param").get_aperture()
        d["kn"] = kn * np.ones(mg.num_cells) / gamma


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
                np.logical_or(np.logical_and.reduce(
                        (xf[0] - tol < -200, xf[1] + tol > 1500, xf[2] + tol > 300)
                        ), np.logical_and.reduce(
                        (xf[0] - tol < -500, xf[1] + tol > 1200, xf[2] + tol > 300)
                        )
            ))

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


class AdvectiveDataAssigner(pp.ParabolicDataAssigner):
    def __init__(self, g, data, physics="transport", **kwargs):
        self.domain = kwargs["domain"]
        self.max_dim = kwargs.get("max_dim", 3)
        self.phi_f = 0.2

        # define two pieces of the boundary, useful to impose boundary conditions
        self.inflow = np.empty(0)

        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size > 0:
            self.inflow = b_pressure(g)[0]

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
