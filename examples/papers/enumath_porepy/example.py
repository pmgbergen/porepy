import numpy as np
import scipy.sparse as sps
import logging

from porepy.viz import exporter
from porepy.fracs import importer

from porepy.params import tensor
from porepy.params.bc import BoundaryCondition
from porepy.params.data import Parameters

from porepy.grids.grid import FaceTag

from porepy.numerics import elliptic

#------------------------------------------------------------------------------#

def add_data(gb, domain, tol):
    gb.add_node_props(['param', 'if_tangent'])

    apert = 1e-2

    km = 1
    kf = 1e-4

    for g, d in gb:

        param = Parameters(g)
        d['if_tangent'] = True
        if g.dim == gb.dim_max():
            kxx = km
        else:
            if g.frac_num == 6 or g.frac_num == 'FRACTURE_LINE_':
                kxx = 1e6
            else:
                kxx = kf

        perm = tensor.SecondOrder(g.dim, kxx*np.ones(g.num_cells))
        param.set_tensor("flow", perm)

        param.set_source("flow", np.zeros(g.num_cells))

        param.set_aperture(np.power(apert, gb.dim_max() - g.dim))

        bound_faces = g.get_domain_boundary_faces()
        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            top = bound_face_centers[2, :] > domain['zmax'] - tol
            bottom = bound_face_centers[2, :] < domain['zmin'] + tol

            boundary = np.logical_or(top, bottom)

            labels = np.array(['neu'] * bound_faces.size)
            labels[boundary] = ['dir']

            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces[bottom]] = 1

            param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val("flow", bc_val)
        else:
            param.set_bc("flow", BoundaryCondition(
                g, np.empty(0), np.empty(0)))

        d['param'] = param

    # Assign coupling permeability
    gb.add_edge_prop('kn')
    for e, d in gb.edges_props():
        g_l, g_h = gb.sorted_nodes_of_edge(e)
        if g_h.dim == gb.dim_max() and g_l.frac_num == 6:
            kxx = 1e6
        elif g_h.dim < gb.dim_max() and g_h.frac_num == 6:
            kxx = 1e6
        else:
            kxx = kf
        d['kn'] = kxx / gb.node_prop(g_l, 'param').get_aperture()

#------------------------------------------------------------------------------#

logging.basicConfig(filename='example.log',level=logging.DEBUG)

tol = 1e-6

problem_kwargs = {}
problem_kwargs['file_name'] = 'solution'
problem_kwargs['folder_name'] = 'example'

h = 0.3 #0.075
grid_kwargs = {}
grid_kwargs['mesh_size'] = {'mode': 'constant', 'value': h, 'bound_value': h,
                            'tol': tol}

file_dfm = 'dfm.csv'
gb, domain = importer.dfm_from_csv(file_dfm, tol, **grid_kwargs)
gb.compute_geometry()

internal_flag = FaceTag.FRACTURE
[g.remove_face_tag_if_tag(FaceTag.BOUNDARY, internal_flag) for g, _ in gb]

problem = elliptic.DualEllipticModel(gb, problem_kwargs)

# Assign parameters
add_data(gb, domain, tol)

problem.solve()
problem.split()

problem.pressure('pressure')
problem.project_discharge('P0u')
problem.save(['pressure', 'P0u'])

#------------------------------------------------------------------------------#
