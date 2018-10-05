import numpy as np
import scipy.sparse as sps
import porepy as pp

from examples.papers.flow_upscaling.import_grid import grid
import examples.papers.flow_upscaling.solvers as solvers

# ------------------------------------------------------------------------------#

def upscaling(file_geo, folder, dfn, data, mesh_args, tol):
    gb, data["domain"] = grid(file_geo, mesh_args, tol)

    if dfn:
        gb.remove_node(gb.grids_of_dimension(gb.dim_max())[0])
        gb.assign_node_ordering()

    #compute left to right flow
    add_data(gb, data, left_to_right=True)
    solvers.pressure(gb, folder+"_left_to_right")

    # compute the upscaled permeability
    kxx, kyx = permeability(gb, data, left_to_right=True)

    #compute the bottom to top flow
    add_data(gb, data, bottom_to_top=True)
    solvers.pressure(gb, folder+"_bottom_to_top")

    # compute the upscaled permeability
    kyy, kxy = permeability(gb, data, bottom_to_top=True)

    print(kxx, kyy, 0.5*(kyx + kxy))
    return pp.SecondOrderTensor(2, kxx=kxx, kyy=kyy, kxy=0.5*(kyx + kxy))

# ------------------------------------------------------------------------------#

def permeability(gb, data, left_to_right=False, bottom_to_top=False):
    tol = data["tol"]
    domain = data["domain"]

    coord_min = np.array([domain["xmin"], domain["ymin"]])
    coord_max = np.array([domain["xmax"], domain["ymax"]])
    delta = coord_min - coord_max

    if left_to_right:
        coord, not_coord = 0, 1
    elif bottom_to_top:
        coord, not_coord = 1, 0
    else:
        raise ValueError

    k_parallel = 0
    k_transverse = 0
    for g, d in gb:
        b_faces = g.tags["domain_boundary_faces"]
        if np.any(b_faces):
            faces, cells, sign = sps.find(g.cell_faces)
            index = np.argsort(cells)
            faces, cells, sign = faces[index], cells[index], sign[index]

            u = d["discharge"].copy()
            u[np.logical_not(b_faces)] = 0
            u[faces] *= sign

            # it's ok since we consider only the boundary
            aperture = d["param"].get_aperture()
            measure = 1./(g.face_areas * (np.abs(g.cell_faces) * aperture))

            bc = g.face_centers[coord, :] > coord_max[coord] - tol
            k_parallel += np.abs(np.dot(u[bc], measure[bc]) * delta[coord])

            bc = g.face_centers[not_coord, :] < coord_max[not_coord] + tol
            k_transverse += np.abs(np.dot(u[bc], measure[bc]) * delta[not_coord])

    return k_parallel, k_transverse

# ------------------------------------------------------------------------------#

def add_data(gb, data, left_to_right=False, bottom_to_top=False):
    tol = data["tol"]
    domain = data["domain"]

    coord_min = np.array([domain["xmin"], domain["ymin"]])
    coord_max = np.array([domain["xmax"], domain["ymax"]])
    delta = coord_max - coord_min

    if left_to_right:
        coord = 0
    elif bottom_to_top:
        coord = 1
    else:
        raise ValueError

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
            bc = pp.BoundaryCondition(g, b_faces, ["dir"] * b_faces.size)
            val = g.face_centers[coord, b_faces] - coord_min[coord]

            bc_val = np.zeros(g.num_faces)
            bc_val[b_faces] = (1 - val/delta[coord]) * pp.PASCAL

            param.set_bc_val("flow", bc_val)
        else:
            bc = pp.BoundaryCondition(g, empty, empty)

        param.set_bc("flow", bc)

        d["param"] = param

    # Assign coupling permeability, the aperture is read from the lower dimensional grid
    gb.add_edge_props("kn")
    for e, d in gb.edges():
        g_l = gb.nodes_of_edge(e)[0]
        mg = d["mortar_grid"]
        check_P = mg.low_to_mortar_avg()

        aperture = gb.node_props(g_l, "param").get_aperture()
        gamma = check_P * np.power(aperture, 1/(2.-g.dim))
        d["kn"] = data["kf"] * np.ones(mg.num_cells) / gamma

# ------------------------------------------------------------------------------#
