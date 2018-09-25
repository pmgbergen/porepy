import numpy as np
import porepy as pp

import examples.papers.flow_upscaling.import_grid as grid
import examples.papers.flow_upscaling.solvers as solvers

def upscaling(file_geo, data, folder, dfn=True):
    mesh_args = {'mesh_size_frac': 10}
    tol = {"geo": 1e-4, "snap": 1e-3}
    data["tol"] = tol["geo"]

    gb, data["domain"] = grid.from_file(file_geo, mesh_args, tol)

    if dfn:
        gb.remove_node(gb.grids_of_dimension(2)[0])
        gb.assign_node_ordering()

    #compute left to right flow
    left_to_right(gb, data)
    add_data(gb, data)
    solvers.solve_rt0(gb, folder)

    #compute the bottom to top flow
    bottom_to_top(gb, data)
    add_data(gb, data)
    solvers.solve_rt0(gb, folder)

# ------------------------------------------------------------------------------#

def left_to_right(gb, data):
    xmin = data["domain"]["xmin"] + data["tol"]
    xmax = data["domain"]["xmax"] - data["tol"]

    gb.add_node_props(['high', 'low'])
    for g, d in gb:
        # Boundaries
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size != 0:
            b_face_centers = g.face_centers[:, b_faces]
            d['low'] = b_face_centers[0, :] < xmin
            d['high'] = b_face_centers[0, :] > xmax

# ------------------------------------------------------------------------------#

def bottom_to_top(gb, data):
    ymin = data["domain"]["ymin"] + data["tol"]
    ymax = data["domain"]["ymax"] - data["tol"]

    gb.add_node_props(['high', 'low'])
    for g, d in gb:
        # Boundaries
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size != 0:
            b_face_centers = g.face_centers[:, b_faces]
            d['low'] = b_face_centers[1, :] < ymin
            d['high'] = b_face_centers[1, :] > ymax

# ------------------------------------------------------------------------------#

def add_data(gb, data):
    tol = data["tol"]

    gb.add_node_props(['is_tangential', 'frac_num'])
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

            b_low = b_face_centers[0, :] < data["domain"]["xmin"] + tol
            b_high = b_face_centers[0, :] > data["domain"]["xmax"] - tol

            labels = np.array(["neu"] * b_faces.size)
            labels[d["low"]] = "dir"
            labels[d["high"]] = "dir"
            param.set_bc("flow", pp.BoundaryCondition(g, b_faces, labels))

            bc_val = np.zeros(g.num_faces)
            bc_val[b_faces[d["low"]]] = 0 * pp.PASCAL
            bc_val[b_faces[d["high"]]] = 1 * pp.PASCAL
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


