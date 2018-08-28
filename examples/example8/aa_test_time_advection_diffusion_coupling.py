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

from porepy.utils.errors import error

# ------------------------------------------------------------------------------#


def add_data_darcy(gb, domain, tol, a):
    gb.add_node_props(["param"])

    kf = 1e4
    for g, d in gb:
        param = Parameters(g)

        kxx = np.ones(g.num_cells) * np.power(kf, g.dim < gb.dim_max())
        perm = tensor.SecondOrderTensor(g.dim, kxx)
        param.set_tensor("flow", perm)

        param.set_source("flow", np.zeros(g.num_cells))

        aperture = np.power(a, gb.dim_max() - g.dim)
        param.set_aperture(np.ones(g.num_cells) * aperture)

        bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            top = bound_face_centers[1, :] > domain["ymax"] - tol
            bottom = bound_face_centers[1, :] < domain["ymin"] + tol
            left = bound_face_centers[0, :] < domain["xmin"] + tol
            right = bound_face_centers[0, :] > domain["xmax"] - tol
            boundary = np.logical_or(left, right)

            labels = np.array(["neu"] * bound_faces.size)
            labels[boundary] = ["dir"]

            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces[right]] = 1

            param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val("flow", bc_val)
        else:
            param.set_bc("flow", BoundaryCondition(g, np.empty(0), np.empty(0)))

        d["param"] = param

    # Assign coupling permeability
    gb.add_edge_prop("kn")
    for e, d in gb.edges_props():
        gn = gb.nodes_of_edge(e)
        aperture = np.power(a, gb.dim_max() - gn[0].dim)
        d["kn"] = np.ones(gn[0].num_cells) / aperture * kf


# ------------------------------------------------------------------------------#


def add_data_advection_diffusion(gb, domain, tol, a):

    for g, d in gb:
        param = d["param"]

        kxx = 5 * 1e-2 * np.ones(g.num_cells)
        perm = tensor.SecondOrderTensor(g.dim, kxx)
        param.set_tensor("transport", perm)

        # The 0.5 needs to be fixed in a better way
        #        source = 0.5 * np.ones(g.num_cells) * \
        #            g.cell_volumes * param.get_aperture()
        source = np.zeros(g.num_cells)
        param.set_source("transport", source)

        bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            top = bound_face_centers[1, :] > domain["ymax"] - tol
            bottom = bound_face_centers[1, :] < domain["ymin"] + tol
            left = bound_face_centers[0, :] < domain["xmin"] + tol
            right = bound_face_centers[0, :] > domain["xmax"] - tol
            boundary = np.logical_or(
                np.logical_or(np.logical_or(top, bottom), left), right
            )
            labels = np.array(["neu"] * bound_faces.size)
            labels[boundary] = ["dir"]

            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces[right]] = 1

            param.set_bc("transport", BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val("transport", bc_val)
        else:
            param.set_bc("transport", BoundaryCondition(g, np.empty(0), np.empty(0)))

    # Assign coupling discharge
    gb.add_edge_prop("param")
    for e, d in gb.edges_props():
        g_h = gb.nodes_of_edge(e)[1]
        discharge = gb.node_prop(g_h, "param").get_discharge()
        d["param"] = Parameters(g_h)
        d["param"].set_discharge(discharge)


# ------------------------------------------------------------------------------#


folder = os.path.dirname(os.path.realpath(__file__)) + "/"
export_folder = folder + "heat"
tol = 1e-3
a = 1e-2

mesh_kwargs = {}
mesh_kwargs["mesh_size"] = {"mode": "constant", "value": 0.045, "bound_value": 0.025}

domain = {"xmin": -0.2, "xmax": 1.2, "ymin": -0.2, "ymax": 1.2}
gb = importer.from_csv(folder + "network.csv", mesh_kwargs, domain)
gb.compute_geometry()
# co.coarsen(gb, 'by_volume')
gb.assign_node_ordering()

# Choose and define the solvers and coupler
darcy = dual.DualVEMMixDim("flow")

# Assign parameters
add_data_darcy(gb, domain, tol, a)

A, b = darcy.matrix_rhs(gb)

up = sps.linalg.spsolve(A, b)
darcy.split(gb, "up", up)

gb.add_node_props(["pressure", "P0u", "u"])
for g, d in gb:
    d["u"] = darcy.discr.extract_u(g, d["up"])
    d["param"].set_discharge(d["u"])
    d["pressure"] = darcy.discr.extract_p(g, d["up"])
    d["P0u"] = darcy.discr.project_u(g, d["u"], d)

# compute the flow rate
total_flow_rate = 0
for g, d in gb:
    bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    if bound_faces.size != 0:
        bound_face_centers = g.face_centers[:, bound_faces]
        left = bound_face_centers[0, :] < domain["xmin"] + tol
        flow_rate = d["u"][bound_faces[left]]
        total_flow_rate += np.sum(flow_rate)

print("total flow rate", total_flow_rate)
exporter.export_vtk(gb, "darcy", ["pressure", "P0u"], folder=export_folder)

#################################################################

physics = "transport"
advection = upwind.UpwindMixedDim(physics)
diffusion = tpfa.TpfaMixedDim(physics)
mass = mass_matrix.MassMatrixMixedDim(physics)
invMass = mass_matrix.InvMassMatrixMixedDim(physics)

# Assign parameters
add_data_advection_diffusion(gb, domain, tol, a)

T = 1
deltaT = 0.01
gb.add_node_prop("deltaT", prop=deltaT)

U, rhs_u = advection.matrix_rhs(gb)
D, rhs_d = diffusion.matrix_rhs(gb)
M, _ = mass.matrix_rhs(gb)
OF = advection.outflow(gb)

rhs = rhs_u  # + rhs_d

# Perform an LU factorization to speedup the solver
IE_solver = sps.linalg.factorized((M + U).tocsc())

theta = np.zeros(rhs_u.shape[0])

# Loop over the time
Nt = int(T / deltaT)
time = np.empty(Nt)
file_name = "theta"
i_export = 0
export_every = 10
step_to_export = np.empty(0)

production = np.zeros(Nt)

for i in np.arange(Nt):
    print("Time step", i, " of ", Nt)
    # Update the solution
    production[i] = np.sum(OF.dot(theta)) / total_flow_rate
    theta = IE_solver(M.dot(theta) + rhs)

    if i % export_every == 0:
        print("Export solution at", i)
        diffusion.split(gb, "theta", theta)
        exporter.export_vtk(
            gb, file_name, ["theta"], time_step=i_export, folder=export_folder
        )
        step_to_export = np.r_[step_to_export, i]
        i_export += 1

    exporter.export_vtk(gb, "theta", ["theta"], time_step=i, folder=export_folder)

# print(production, np.cumsum(production))
exporter.export_pvd(gb, file_name, step_to_export * deltaT, folder=export_folder)

print(production)
# Consistency check
# assert np.isclose(np.sum(error.norm_L2(g, d['pressure']) for g, d in gb), 19.8455019189)
