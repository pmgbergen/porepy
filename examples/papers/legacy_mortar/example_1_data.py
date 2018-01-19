import numpy as np

from porepy.fracs import importer, mortars
from porepy.grids import refinement

from porepy.params.data import Parameters
from porepy.params.bc import BoundaryCondition
from porepy.params import tensor

#------------------------------------------------------------------------------#

def create_gb(domain, cells_2d, cells_1d = None, cells_mortar = None, tol=1e-5):

    mesh_kwargs = {}
    mesh_size = np.sqrt(2/cells_2d)
    mesh_kwargs['mesh_size'] = {'mode': 'constant', 'tol': tol,
                                'value': mesh_size, 'bound_value': mesh_size}

    file_name = 'example_1_split.csv'
    gb = importer.dfm_2d_from_csv(file_name, mesh_kwargs, domain)

    g_map = {}
    if cells_1d is not None:
        for g, d in gb:
            if g.dim == 1:
                num_nodes = g.num_nodes+cells_1d#+1
                g_map[g] = refinement.new_grid_1d(g, num_nodes=num_nodes)
                g_map[g].compute_geometry()

    mg_map = {}
    if cells_mortar is not None:
        for e, d in gb.edges_props():
            mg = d['mortar_grid']
            if mg.dim == 1:
                mg_map[mg] = {}
                for s, g in mg.side_grids.items():
                    num_nodes = g.num_nodes+cells_mortar#+1
                    mg_map[mg][s] = refinement.new_grid_1d(g, num_nodes=num_nodes)

    gb = mortars.replace_grids_in_bucket(gb, g_map, mg_map, tol)
    gb.assign_node_ordering()

    return gb

#------------------------------------------------------------------------------#

def add_data(gb, domain, data = {}, tol=1e-5):

    gb.add_node_props(['param', 'is_tangential'])
    for g, d in gb:
        param = Parameters(g)

        solver = data.get("solver")
        if g.dim == 2:
            kxx = np.ones(g.num_cells) * data.get("km", 1)
            if solver == "vem" or solver == "rt0":
                perm = tensor.SecondOrder(g.dim, kxx=kxx, kyy=kxx, kzz=1)
            elif solver == "tpfa":
                perm = tensor.SecondOrder(3, kxx=kxx)
        elif g.dim == 1:
            if g.cell_centers[1, 0] > 0.25 and g.cell_centers[1, 0] < 0.55:
                kxx = np.ones(g.num_cells) * data.get("kf_low", 1)
            else:
                kxx = np.ones(g.num_cells) * data.get("kf_high", 1)

            if solver == "vem" or solver == "rt0":
                perm = tensor.SecondOrder(g.dim, kxx=kxx, kyy=1, kzz=1)
            elif solver == "tpfa":
                perm = tensor.SecondOrder(3, kxx=kxx)

        param.set_tensor("flow", perm)

        aperture = data.get("aperture", 1e-2)
        aperture = np.power(aperture, gb.dim_max() - g.dim)
        param.set_aperture(aperture*np.ones(g.num_cells))

        param.set_source("flow", np.zeros(g.num_cells))

        bound_faces = g.get_domain_boundary_faces()
        if bound_faces.size == 0:
            bc =  BoundaryCondition(g, np.empty(0), np.empty(0))
            param.set_bc("flow", bc)
        else:
            bound_face_centers = g.face_centers[:, bound_faces]

            top = bound_face_centers[1, :] > domain['ymax'] - tol
            bottom = bound_face_centers[1, :] < domain['ymin'] + tol

            labels = np.array(['neu'] * bound_faces.size)
            labels[np.logical_or(top, bottom)] = 'dir'

            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces[top]] = 1

            param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val("flow", bc_val)

        d['is_tangential'] = True
        d['param'] = param

    gb.add_edge_prop('kn')
    for e, d in gb.edges_props():
        g_l = gb.sorted_nodes_of_edge(e)[0]
        mg = d['mortar_grid']
        check_P = mg.low_to_mortar

        if g_l.dim == 1:
            if g_l.cell_centers[1, 0] > 0.25 and g_l.cell_centers[1, 0] < 0.55:
                kxx = data.get("kf_low", 1)
            else:
                kxx = data.get("kf_high", 1)
        else:
            kxx = data.get("kf_high", 1)

        aperture = check_P * gb.node_prop(g_l, 'param').get_aperture()
        d['kn'] = kxx * np.ones(mg.num_cells) / aperture

#------------------------------------------------------------------------------#
