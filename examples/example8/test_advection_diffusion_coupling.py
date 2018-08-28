import numpy as np
import scipy.sparse as sps
import os
import sys

from porepy.viz import exporter
from porepy.fracs import importer

from porepy.params import tensor
from porepy.grids import structured

from porepy.numerics.mixed_dim import coupler
from porepy.numerics.vem import vem_dual, vem_source
from porepy.numerics.fv.transport import upwind
from porepy.numerics.fv import tpfa

from porepy.params.bc import BoundaryCondition
from porepy.params.data import Parameters


# ------------------------------------------------------------------------------#


def add_data_darcy(gb, domain, tol):
    gb.add_node_props(["param"])

    kf = 1e-4
    for g, d in gb:
        param = Parameters(g)

        kxx = np.ones(g.num_cells) * np.power(kf, g.dim < gb.dim_max())
        perm = tensor.SecondOrderTensor(g.dim, kxx)
        param.set_tensor("flow", perm)

        param.set_source("flow", np.zeros(g.num_cells))

        aperture = np.power(1e-2, gb.dim_max() - g.dim)
        param.set_aperture(np.ones(g.num_cells) * aperture)

        bound_faces = np.argwhere(g.tags["domain_boundary_faces"]).ravel("F")
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
            bc_dir = bound_faces[boundary]
            bc_val[bc_dir] = np.sum(g.face_centers[:, bc_dir], axis=0) * aperture

            param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val("flow", bc_val)
        else:
            param.set_bc("flow", BoundaryCondition(g, np.empty(0), np.empty(0)))

        d["param"] = param

    # Assign coupling permeability
    gb.add_edge_props("kn")
    for e, d in gb.edges():
        gn = gb.nodes_of_edge(e)
        aperture = np.power(1e-2, gb.dim_max() - gn[0].dim)
        d["kn"] = np.ones(d["mortar_grid"].num_cells) / aperture * kf


# ------------------------------------------------------------------------------#


def add_data_advection_diffusion(gb, domain, tol):

    for g, d in gb:
        param = d["param"]

        kxx = 5 * 1e-2 * np.ones(g.num_cells)
        perm = tensor.SecondOrderTensor(g.dim, kxx)
        param.set_tensor("transport", perm)

        # The 0.5 needs to be fixed in a better way
        source = 0.5 * np.ones(g.num_cells) * g.cell_volumes * param.get_aperture()
        param.set_source("transport", source)

        bound_faces = np.argwhere(g.tags["domain_boundary_faces"]).ravel("F")
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
            bc_dir = bound_faces[boundary]

            param.set_bc("transport", BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val("transport", bc_val)
        else:
            param.set_bc("transport", BoundaryCondition(g, np.empty(0), np.empty(0)))

    # Assign coupling discharge
    gb.add_edge_props("param")
    for e, d in gb.edges():
        g_h = gb.nodes_of_edge(e)[1]
        discharge = gb.node_props(g_h)["discharge"]
        d["param"] = Parameters(g_h)
        d["discharge"] = discharge


# ------------------------------------------------------------------------------#


do_save = False
folder = os.path.dirname(os.path.realpath(__file__)) + "/"
export_folder = folder + "advection_diffusion_coupling"
tol = 1e-3

mesh_kwargs = {"mesh_size_frac": 0.045, "mesh_size_min": 0.01}
domain = {"xmin": -0.2, "xmax": 1.2, "ymin": -0.2, "ymax": 1.2}
gb = importer.dfm_2d_from_csv(folder + "network.csv", mesh_kwargs, domain)
gb.compute_geometry()
gb.assign_node_ordering()

# Assign parameters
add_data_darcy(gb, domain, tol)

# Choose and define the solvers and coupler
darcy = vem_dual.DualVEMMixedDim("flow")
A_flow, b_flow = darcy.matrix_rhs(gb)

solver_source = vem_source.DualSourceMixedDim("flow", coupling=[None])
A_source, b_source = solver_source.matrix_rhs(gb)

up = sps.linalg.spsolve(A_flow + A_source, b_flow + b_source)
darcy.split(gb, "up", up)

gb.add_node_props(["pressure", "P0u"])
for g, d in gb:
    discharge = darcy.discr.extract_u(g, d["up"])
    d["discharge"] = discharge
    d["pressure"] = darcy.discr.extract_p(g, d["up"])
    d["P0u"] = darcy.discr.project_u(g, discharge, d)

if do_save:
    exporter.export_vtk(gb, "darcy", ["pressure", "P0u"], folder=export_folder)

#################################################################

physics = "transport"
advection = upwind.UpwindMixedDim(physics)
diffusion = tpfa.TpfaMixedDim(physics)

# Assign parameters
add_data_advection_diffusion(gb, domain, tol)

U, rhs_u = advection.matrix_rhs(gb)
D, rhs_d = diffusion.matrix_rhs(gb)

theta = sps.linalg.spsolve(D + U, rhs_u + rhs_d)
diffusion.split(gb, "temperature", theta)

if do_save:
    exporter.export_vtk(
        gb, "advection_diffusion", ["temperature"], folder=export_folder
    )
