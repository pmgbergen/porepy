import numpy as np
import scipy.sparse as sps
import os
import sys

from porepy.viz import exporter
from porepy.fracs import importer

from porepy.params import tensor
from porepy.params.bc import BoundaryCondition
from porepy.params.data import Parameters

from porepy.grids import coarsening as co

from porepy.numerics.vem import dual
from porepy.numerics.fv.transport import upwind
from porepy.numerics.fv import tpfa, mass_matrix

# ------------------------------------------------------------------------------#


def add_data_darcy(gb, domain, tol):
    gb.add_node_props(["param", "if_tangent"])

    apert = 1e-2

    km = 7.5 * 1e-10  # 2.5*1e-11

    kf = 5 * 1e-5

    for g, d in gb:

        param = Parameters(g)
        d["if_tangent"] = True
        if g.dim == gb.dim_max():
            kxx = km
        else:
            kxx = kf

        perm = tensor.SecondOrderTensor(g.dim, kxx * np.ones(g.num_cells))
        param.set_tensor("flow", perm)

        param.set_source("flow", np.zeros(g.num_cells))

        param.set_aperture(np.power(apert, gb.dim_max() - g.dim))

        bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            top = bound_face_centers[2, :] > domain["zmax"] - tol
            bottom = bound_face_centers[2, :] < domain["zmin"] + tol

            boundary = np.logical_or(top, bottom)

            labels = np.array(["neu"] * bound_faces.size)
            labels[boundary] = ["dir"]

            bc_val = np.zeros(g.num_faces)
            p = np.abs(domain["zmax"] - domain["zmin"]) * 1e3 * 9.81
            bc_val[bound_faces[bottom]] = p

            param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val("flow", bc_val)
        else:
            param.set_bc("flow", BoundaryCondition(g, np.empty(0), np.empty(0)))

        d["param"] = param

    # Assign coupling permeability
    gb.add_edge_prop("kn")
    for e, d in gb.edges_props():
        g = gb.sorted_nodes_of_edge(e)[0]
        d["kn"] = kf / gb.node_prop(g, "param").get_aperture()


# ------------------------------------------------------------------------------#


def add_data_advection(gb, domain, tol):

    for g, d in gb:
        param = d["param"]

        source = np.zeros(g.num_cells)
        param.set_source("transport", source)

        param.set_porosity(1)
        param.set_discharge(d["discharge"])

        bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            top = bound_face_centers[2, :] > domain["zmax"] - tol
            bottom = bound_face_centers[2, :] < domain["zmin"] + tol
            boundary = np.logical_or(top, bottom)
            labels = np.array(["neu"] * bound_faces.size)
            labels[boundary] = ["dir"]

            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces[bottom]] = 1

            param.set_bc("transport", BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val("transport", bc_val)
        else:
            param.set_bc("transport", BoundaryCondition(g, np.empty(0), np.empty(0)))
        d["param"] = param

    # Assign coupling discharge
    gb.add_edge_prop("param")
    for e, d in gb.edges_props():
        g_h = gb.sorted_nodes_of_edge(e)[1]
        discharge = gb.node_prop(g_h, "param").get_discharge()
        d["param"] = Parameters(g_h)
        d["param"].set_discharge(discharge)


# ------------------------------------------------------------------------------#


sys.path.append("../../example3")
import soultz_grid

export_folder = "example_5_3_coarse"
tol = 1e-6

T = 40 * np.pi * 1e7
Nt = 100
deltaT = T / Nt
export_every = 5
if_coarse = True

mesh_kwargs = {
    "mesh_size_frac": 75,
    "mesh_size_bound": 200,
    "mesh_size_min": 10,
    "meshing_algorithm": 4,
    "tol": tol,
}
mesh_kwargs["num_fracs"] = 20
mesh_kwargs["num_points"] = 10
mesh_kwargs["file_name"] = "soultz_fracs"
domain = {
    "xmin": -1200,
    "xmax": 500,
    "ymin": -600,
    "ymax": 600,
    "zmin": 600,
    "zmax": 5500,
}
mesh_kwargs["domain"] = domain

print("create soultz grid")
gb = soultz_grid.create_grid(**mesh_kwargs)
gb.compute_geometry()
if if_coarse:
    co.coarsen(gb, "by_volume")
gb.assign_node_ordering()

print("solve Darcy problem")
for g, d in gb:
    d["cell_id"] = np.arange(g.num_cells)

exporter.export_vtk(gb, "grid", ["cell_id"], folder=export_folder)

# Choose and define the solvers and coupler
darcy = dual.DualVEMMixDim("flow")

# Assign parameters
add_data_darcy(gb, domain, tol)

A, b = darcy.matrix_rhs(gb)

up = sps.linalg.spsolve(A, b)
darcy.split(gb, "up", up)

gb.add_node_props(["pressure", "P0u", "discharge"])
darcy.extract_u(gb, "up", "discharge")
darcy.extract_p(gb, "up", "pressure")
darcy.project_u(gb, "discharge", "P0u")

# compute the flow rate
total_flow_rate = 0
for g, d in gb:
    bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    if bound_faces.size != 0:
        bound_face_centers = g.face_centers[:, bound_faces]
        top = bound_face_centers[2, :] > domain["zmax"] - tol
        flow_rate = d["discharge"][bound_faces[top]]
        total_flow_rate += np.sum(flow_rate)

print("total flow rate", total_flow_rate)
exporter.export_vtk(gb, "darcy", ["pressure", "P0u"], folder=export_folder)

#################################################################

physics = "transport"
advection = upwind.UpwindMixedDim(physics)
mass = mass_matrix.MassMatrixMixedDim(physics)
invMass = mass_matrix.InvMassMatrixMixDim(physics)

# Assign parameters
add_data_advection(gb, domain, tol)

gb.add_node_prop("deltaT", prop=deltaT)

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
    production[i] = np.sum(OF.dot(theta)) / total_flow_rate
    theta = IE_solver(M.dot(theta) + rhs)

    if i % export_every == 0:
        print("Export solution at", i)
        advection.split(gb, "theta", theta)
        exporter.export_vtk(
            gb, file_name, ["theta"], time_step=i_export, folder=export_folder
        )
        step_to_export = np.r_[step_to_export, i]
        i_export += 1

exporter.export_pvd(gb, file_name, step_to_export * deltaT, folder=export_folder)

np.savetxt(
    export_folder + "/production.txt",
    (deltaT * np.arange(Nt), np.abs(production)),
    delimiter=",",
)
