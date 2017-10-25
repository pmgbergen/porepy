import numpy as np
import scipy.sparse as sps
import os
import sys

from porepy.viz import exporter
from porepy.fracs import importer

from porepy.params import tensor
from porepy.params.bc import BoundaryCondition
from porepy.params.data import Parameters

from porepy.grids.grid import FaceTag
from porepy.grids import coarsening as co

from porepy.numerics.vem import dual
from porepy.numerics.fv.transport import upwind
from porepy.numerics.fv import tpfa, mass_matrix

from porepy.utils.errors import error

#------------------------------------------------------------------------------#

def add_data_darcy(gb, domain, tol):
    gb.add_node_props(['param'])

    apert = 1e-2

    km = 7.5*1e-10 #2.5*1e-11
    kf_t_low = 1e-5*km
    kf_n_low = 1e-5*km

    kf_t_high = 5*1e-5
    kf_n_high = kf_t_high

    for g, d in gb:

        param = Parameters(g)

        if g.dim == gb.dim_max():
            kxx = km
        else:
            kxx = kf_t_high

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
            p = np.abs(domain['zmax'] - domain['zmin'])*1e3*9.81
            bc_val[bound_faces[bottom]] = p

            param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val("flow", bc_val)
        else:
            param.set_bc("flow", BoundaryCondition(
                g, np.empty(0), np.empty(0)))

        d['param'] = param

    # Assign coupling permeability
    gb.add_edge_prop('kn')
    for e, d in gb.edges_props():
        g = gb.sorted_nodes_of_edge(e)[0]
        kf_n = kf_n_high
        d['kn'] = kf_n / gb.node_prop(g, 'param').get_aperture()

#------------------------------------------------------------------------------#

def add_data_advection(gb, domain, tol):

    for g, d in gb:
        param = d['param']

        source = np.zeros(g.num_cells)
        param.set_source("transport", source)

        param.set_porosity(1)
        param.set_discharge(d['u'])

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

            param.set_bc("transport", BoundaryCondition(
                g, bound_faces, labels))
            param.set_bc_val("transport", bc_val)
        else:
            param.set_bc("transport", BoundaryCondition(
                g, np.empty(0), np.empty(0)))
        d['param'] = param

    # Assign coupling discharge
    gb.add_edge_prop('param')
    for e, d in gb.edges_props():
        g_h = gb.sorted_nodes_of_edge(e)[1]
        discharge = gb.node_prop(g_h, 'param').get_discharge()
        d['param'] = Parameters(g_h)
        d['param'].set_discharge(discharge)

#------------------------------------------------------------------------------#

def frac_position(frac_num):

    fracs_num = {'top': np.array([0, 1, 2, 3, 4, 5, 6, 11, 19, 20, 21, 22]),
                 'bottom': np.array([16, 17, 18]),
                 'center': np.array([7, 8, 9, 10, 12, 13, 14, 15])}

    for pos in fracs_num:
        if np.any(pos == frac_num):
            return pos

#------------------------------------------------------------------------------#

sys.path.append('../../example3')
import soultz_grid

folder = os.path.dirname(os.path.realpath(__file__)) + "/"
export_folder = folder + 'heat_coarse_blocking'
tol = 1e-4

T = 40*np.pi*1e7
Nt = 100
deltaT = T/Nt
export_every = 5

mesh_kwargs = {}
mesh_kwargs['mesh_size'] = {'mode': 'constant',
                            'value': 75,
                            'bound_value': 150,
                            'file_name': 'soultz_fracs',
                            'tol': tol}

print("create soultz grid")
gb, domain = soultz_grid.create_grid(**mesh_kwargs)
gb.compute_geometry()
#co.coarsen(gb, 'by_volume')
gb.assign_node_ordering()

print("export grid")
gb.add_node_props(["cell_id", "frac_num"])
for g, d in gb:
    d['cell_id'] = np.arange(g.num_cells)
    if g.dim==2 and hasattr(g, 'frac_num'):
        d['frac_num'] = g.frac_num*np.ones(g.num_cells)
    else:
        d['frac_num'] = -1*np.ones(g.num_cells)

exporter.export_vtk(gb, 'grid', ['cell_id', 'frac_num'], folder=export_folder)

print("solve Darcy problem")
gb.add_node_props(['face_tags'])
for g, d in gb:
    d['face_tags'] = g.face_tags.copy()

internal_flag = FaceTag.FRACTURE
[g.remove_face_tag_if_tag(FaceTag.BOUNDARY, internal_flag) for g, _ in gb]

# Choose and define the solvers and coupler
darcy = dual.DualVEMMixDim("flow")

# Assign parameters
add_data_darcy(gb, domain, tol)

A, b = darcy.matrix_rhs(gb)

up = sps.linalg.spsolve(A, b)
darcy.split(gb, "up", up)

gb.add_node_props(["p", "P0u", "u"])
darcy.extract_u(gb, "up", "u")
darcy.extract_p(gb, "up", "p")
darcy.project_u(gb, "u", "P0u")

# compute the flow rate
total_flow_rate = 0
for g, d in gb:
    bound_faces = g.get_boundary_faces()
    if bound_faces.size != 0:
        bound_face_centers = g.face_centers[:, bound_faces]
        top = bound_face_centers[2, :] > domain['zmax'] - tol
        flow_rate = d['u'][bound_faces[top]]
        total_flow_rate += np.sum(flow_rate)

print("total flow rate", total_flow_rate)
exporter.export_vtk(gb, 'darcy', ["p", "P0u"], folder=export_folder)

#################################################################

for g, d in gb:
    g.face_tags = d['face_tags']

physics = 'transport'
advection = upwind.UpwindMixDim(physics)
mass = mass_matrix.MassMatrixMixDim(physics)
invMass = mass_matrix.InvMassMatrixMixDim(physics)

# Assign parameters
add_data_advection(gb, domain, tol)

gb.add_node_prop('deltaT', prop=deltaT)

U, rhs_u = advection.matrix_rhs(gb)
M, _ = mass.matrix_rhs(gb)
OF = advection.outflow(gb)
M_U = M + U

rhs = rhs_u

# Perform an LU factorization to speedup the solver
IE_solver = sps.linalg.factorized((M_U).tocsc())

theta = np.zeros(rhs.shape[0])

# Loop over the time
time = np.empty(Nt)
file_name = "theta"
i_export = 0
step_to_export = np.empty(0)

production = np.zeros(Nt)

for i in np.arange(Nt):
    print("Time step", i, " of ", Nt)
    # Update the solution
    production[i] = np.sum(OF.dot(theta))/total_flow_rate
    theta = IE_solver(M.dot(theta) + rhs)

    if i%export_every == 0:
        print("Export solution at", i)
        advection.split(gb, "theta", theta)
        exporter.export_vtk(gb, file_name, ["theta"], time_step=i_export,
                            folder=export_folder)
        step_to_export = np.r_[step_to_export, i]
        i_export += 1

#    exporter.export_vtk(gb, "theta", ["theta"], time_step=i, folder=export_folder)

exporter.export_pvd(gb, file_name, step_to_export*deltaT, folder=export_folder)

np.savetxt(export_folder + '/production.txt', (deltaT*np.arange(Nt),
                                               np.abs(production)),
           delimiter=',')

# Consistency check
#assert np.isclose(np.sum(error.norm_L2(g, d['p']) for g, d in gb), 19.8455019189)
