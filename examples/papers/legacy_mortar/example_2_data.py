import numpy as np
import pickle

from porepy.grids.grid import FaceTag
from porepy.fracs import importer
from porepy.params import tensor

from porepy.params.data import Parameters
from porepy.params.bc import BoundaryCondition
from porepy.params import tensor

#------------------------------------------------------------------------------#

def tol():
    return 1e-8

#------------------------------------------------------------------------------#

def geo_file_name(h, prepared):
    fn = 'geiger_3d_h_' + str(h)
    if prepared:
        fn += '_prepared'
    fn += '.geo'
    return fn


def create_gb(h):

    def str_of_nodes(n):
        return str(n[0]) + '_' + str(n[1]) + '_' + str(n[2])

    file_csv = 'geiger_3d.csv'
    file_geo = geo_file_name(h, True)
    print('Create grid')
    _, _, domain = importer.network_3d_from_csv(file_csv, tol=tol())
    network = pickle.load(open('geiger_3d_network', 'rb'))
    gb = importer.dfm_from_gmsh(file_geo, 3, network=network, tol=tol())
    gb.compute_geometry()
    print(gb)
    gb.add_edge_prop('edge_id')
    for e, d in gb.edges_props():
        g_l, g_h = gb.sorted_nodes_of_edge(e)
        if g_l.dim == 2:
            d['edge_id'] = g_l.frac_num
        elif g_l.dim == 1:
            f1 = g_h.frac_num
            g_neigh = gb.node_neighbors(g_l, only_higher=True)
            assert g_neigh.size == 2
            for g in g_neigh:
                if g.frac_num != g_h.frac_num:
                    f2 = g.frac_num
            min_coord = g_l.nodes.min(axis=1)
            max_coord = g_l.nodes.max(axis=1)
            d['edge_id'] = str(f1) + '_' + str(f2) + '_'\
                + str_of_nodes(min_coord) + '_' +  str_of_nodes(max_coord)
            g_l.edge_id = d['edge_id']

    for e, d in gb.edges_props():
        g_l, g_h = gb.sorted_nodes_of_edge(e)
        if g_h.dim == 1:
            min_coord = g_h.nodes.min(axis=1)
            max_coord = g_h.nodes.max(axis=1)
            coord = g_l.nodes
            d['edge_id'] = str_of_nodes(min_coord) + '_'\
                + str_of_nodes(max_coord) + '_' + str_of_nodes(coord)


    return gb, domain

#------------------------------------------------------------------------------#

def add_data(gb, domain, solver, case):

    if case == 1:
        kf = 1e4
        kf = 1e-8
        kf = 1
    else:
        kf = 1e-4
    data = {'domain': domain, 'aperture': 1e-4, 'km': 1, 'kf': kf}

    if_solver = solver == "vem" or solver == "rt0" or solver == "p1"

    # only when solving for the vem case
    if solver == "vem" or solver == "rt0":
        [g.remove_face_tag_if_tag(FaceTag.BOUNDARY, FaceTag.FRACTURE) \
                                                             for g, _ in gb]

    gb.add_node_props(['param', 'is_tangential'])
    for g, d in gb:
        param = Parameters(g)

        if g.dim == 3:
            kxx = np.ones(g.num_cells) * data["km"]
            perm = tensor.SecondOrder(3, kxx=kxx)
        else:
            kxx = np.ones(g.num_cells) * data["kf"]
            if if_solver:
                if g.dim == 2:
                    perm = tensor.SecondOrder(g.dim, kxx=kxx, kyy=kxx, kzz=1)
                elif g.dim == 1:
                    perm = tensor.SecondOrder(g.dim, kxx=kxx, kyy=1, kzz=1)
                else:  # g.dim == 0
                    perm = tensor.SecondOrder(g.dim, kxx=kxx, kyy=1, kzz=1)
            else:
                perm = tensor.SecondOrder(3, kxx=kxx)

        param.set_tensor("flow", perm)

        aperture = np.power(data["aperture"], gb.dim_max() - g.dim)
        param.set_aperture(aperture*np.ones(g.num_cells))

        param.set_source("flow", np.zeros(g.num_cells))

        bound_faces = g.get_domain_boundary_faces()
        if bound_faces.size == 0:
            bc =  BoundaryCondition(g, np.empty(0), np.empty(0))
            param.set_bc("flow", bc)
        else:
            p_press, b_in, b_out = b_pressure(g)

            labels = np.array(['neu'] * bound_faces.size)
            labels[p_press] = 'dir'

            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces[b_in]] = 1
            bc_val[bound_faces[b_out]] = -1

            param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val("flow", bc_val)

        d['is_tangential'] = True
        d['param'] = param

    gb.add_edge_prop('kn')
    for e, d in gb.edges_props():
        g_l = gb.sorted_nodes_of_edge(e)[0]
        mg = d['mortar_grid']
        check_P = mg.low_to_mortar_avg()

        kxx = data["kf"]
        kxx = 1e-8
        gamma = np.power(check_P * gb.node_prop(g_l, 'param').get_aperture(),
                         1./(gb.dim_max() - g_l.dim))
        d['kn'] = kxx * np.ones(mg.num_cells) / gamma

#------------------------------------------------------------------------------#

def b_pressure(g):

    b_faces = g.get_domain_boundary_faces()
    null = np.zeros(b_faces.size, dtype=np.bool)
    if b_faces.size == 0:
        return null, null, null
    else:
        b_face_centers = g.face_centers[:, b_faces]

        val = 0.4 - tol()
        b_in = np.logical_and.reduce(tuple(b_face_centers[i, :] < val \
                                                             for i in range(3)))

        val = 0.8 + tol()
        b_out = np.logical_and.reduce(tuple(b_face_centers[i, :] > val \
                                                             for i in range(3)))
        return np.logical_or(b_in, b_out), b_in, b_out

#------------------------------------------------------------------------------#
