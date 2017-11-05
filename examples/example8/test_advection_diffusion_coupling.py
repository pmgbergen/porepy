import numpy as np
import scipy.sparse as sps
import os
import sys

from porepy.viz import exporter
from porepy.fracs import importer

from porepy.params import tensor
from porepy.grids import structured
from porepy.grids.grid import FaceTag

from porepy.numerics.mixed_dim import coupler
from porepy.numerics.vem import dual, dual_coupling
from porepy.numerics.fv.transport import upwind, upwind_coupling
from porepy.numerics.fv import tpfa

from porepy.params.bc import BoundaryCondition
from porepy.params.data import Parameters

from porepy.utils.errors import error

#------------------------------------------------------------------------------#


def add_data_darcy(gb, domain, tol):
    gb.add_node_props(['param'])

    kf = 1e-4
    for g, d in gb:
        param = Parameters(g)

        kxx = np.ones(g.num_cells) * np.power(kf, g.dim < gb.dim_max())
        perm = tensor.SecondOrder(g.dim, kxx)
        param.set_tensor("flow", perm)

        param.set_source("flow", np.zeros(g.num_cells))

        aperture = np.power(1e-2, gb.dim_max() - g.dim)
        param.set_aperture(np.ones(g.num_cells) * aperture)

        bound_faces = g.get_boundary_faces()
        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            top = bound_face_centers[1, :] > domain['ymax'] - tol
            bottom = bound_face_centers[1, :] < domain['ymin'] + tol
            left = bound_face_centers[0, :] < domain['xmin'] + tol
            right = bound_face_centers[0, :] > domain['xmax'] - tol
            boundary = np.logical_or(np.logical_or(np.logical_or(top, bottom),
                                                   left), right)

            labels = np.array(['neu'] * bound_faces.size)
            labels[boundary] = ['dir']

            bc_val = np.zeros(g.num_faces)
            bc_dir = bound_faces[boundary]
            bc_val[bc_dir] = np.sum(
                g.face_centers[:, bc_dir], axis=0) * aperture

            param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val("flow", bc_val)
        else:
            param.set_bc("flow", BoundaryCondition(
                g, np.empty(0), np.empty(0)))

        d['param'] = param

    # Assign coupling permeability
    gb.add_edge_prop('kn')
    for e, d in gb.edges_props():
        gn = gb.sorted_nodes_of_edge(e)
        aperture = np.power(1e-2, gb.dim_max() - gn[0].dim)
        d['kn'] = np.ones(gn[0].num_cells) / aperture * kf

#------------------------------------------------------------------------------#


def add_data_advection_diffusion(gb, domain, tol):

    for g, d in gb:
        param = d['param']

        kxx = 5 * 1e-2 * np.ones(g.num_cells)
        perm = tensor.SecondOrder(g.dim, kxx)
        param.set_tensor("transport", perm)

        # The 0.5 needs to be fixed in a better way
        source = 0.5 * np.ones(g.num_cells) * \
            g.cell_volumes * param.get_aperture()
        param.set_source("transport", source)

        bound_faces = g.get_boundary_faces()
        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            top = bound_face_centers[1, :] > domain['ymax'] - tol
            bottom = bound_face_centers[1, :] < domain['ymin'] + tol
            left = bound_face_centers[0, :] < domain['xmin'] + tol
            right = bound_face_centers[0, :] > domain['xmax'] - tol
            boundary = np.logical_or(np.logical_or(np.logical_or(top, bottom),
                                                   left), right)

            labels = np.array(['neu'] * bound_faces.size)
            labels[boundary] = ['dir']

            bc_val = np.zeros(g.num_faces)
            bc_dir = bound_faces[boundary]

            param.set_bc("transport", BoundaryCondition(
                g, bound_faces, labels))
            param.set_bc_val("transport", bc_val)
        else:
            param.set_bc("transport", BoundaryCondition(
                g, np.empty(0), np.empty(0)))

    # Assign coupling discharge
    gb.add_edge_prop('param')
    for e, d in gb.edges_props():
        g_h = gb.sorted_nodes_of_edge(e)[1]
        discharge = gb.node_prop(g_h, 'param').get_discharge()
        d['param'] = Parameters(g_h)
        d['param'].set_discharge(discharge)

#------------------------------------------------------------------------------#


do_save = False
folder = os.path.dirname(os.path.realpath(__file__)) + "/"
export_folder = folder + 'advection_diffusion_coupling'
tol = 1e-3

mesh_kwargs = {}
mesh_kwargs['mesh_size'] = {'mode': 'constant',
                            'value': 0.045, 'bound_value': 0.045}

domain = {'xmin': -0.2, 'xmax': 1.2, 'ymin': -0.2, 'ymax': 1.2}
print(folder)
gb = importer.from_csv(folder + 'network.csv', mesh_kwargs, domain)
gb.compute_geometry()
gb.assign_node_ordering()

gb.add_node_props(['face_tags'])
for g, d in gb:
    d['face_tags'] = g.face_tags.copy()

internal_flag = FaceTag.FRACTURE
[g.remove_face_tag_if_tag(FaceTag.BOUNDARY, internal_flag) for g, _ in gb]

# Choose and define the solvers and coupler
darcy_discr = dual.DualVEM(physics="flow")

# Assign parameters
add_data_darcy(gb, domain, tol)

darcy_coupling_conditions = dual_coupling.DualCoupling(darcy_discr)
darcy_coupler = coupler.Coupler(darcy_discr, darcy_coupling_conditions)
A, b = darcy_coupler.matrix_rhs(gb)

up = sps.linalg.spsolve(A, b)
darcy_coupler.split(gb, "up", up)

gb.add_node_props(["p", "P0u"])
for g, d in gb:
    discharge = darcy_discr.extract_u(g, d["up"])
    d['param'].set_discharge(discharge)
    d["p"] = darcy_discr.extract_p(g, d["up"])
    d["P0u"] = darcy_discr.project_u(g, discharge, d)

if do_save:
    exporter.export_vtk(gb, 'darcy', ["p", "P0u"], folder=export_folder)

#################################################################

for g, d in gb:
    g.face_tags = d['face_tags']

advection_discr = upwind.Upwind(physics="transport")
diffusion_discr = tpfa.Tpfa(physics="transport")

# Assign parameters
add_data_advection_diffusion(gb, domain, tol)

advection_coupling_conditions = upwind_coupling.UpwindCoupling(advection_discr)
advection_coupler = coupler.Coupler(
    advection_discr, advection_coupling_conditions)
U, rhs_u = advection_coupler.matrix_rhs(gb)

diffusion_coupling_conditions = tpfa.TpfaCoupling(diffusion_discr)
diffusion_coupler = coupler.Coupler(
    diffusion_discr, diffusion_coupling_conditions)
D, rhs_d = diffusion_coupler.matrix_rhs(gb)

theta = sps.linalg.spsolve(D + U, rhs_u + rhs_d)
diffusion_coupler.split(gb, "temperature", theta)

if do_save:
    exporter.export_vtk(gb, 'advection_diffusion', [
        "temperature"], folder=export_folder)
