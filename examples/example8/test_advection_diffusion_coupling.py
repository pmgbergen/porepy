import numpy as np
import scipy.sparse as sps
import os
import porepy as pp


# ------------------------------------------------------------------------------#


def add_data_darcy(gb, domain, tol):
    gb.add_node_props(["param"])

    kf = 1e-4
    for g, d in gb:
        param = pp.Parameters(g)

        kxx = np.ones(g.num_cells) * np.power(kf, g.dim < gb.dim_max())
        perm = pp.SecondOrderTensor(g.dim, kxx)
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

            param.set_bc("flow", pp.BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val("flow", bc_val)
        else:
            param.set_bc("flow", pp.BoundaryCondition(g, np.empty(0), np.empty(0)))

        d["param"] = param

    # Assign coupling permeability
    gb.add_edge_props("kn")
    for e, d in gb.edges():
        g_l = gb.nodes_of_edge(e)[0]
        aperture = np.power(1e-2, gb.dim_max() - g_l.dim) * np.ones(g_l.num_cells)

        mg = d["mortar_grid"]
        check_P = mg.slave_to_mortar_avg()

        gamma = check_P * aperture
        d["kn"] = kf * np.ones(mg.num_cells) / gamma


# ------------------------------------------------------------------------------#


def add_data_advection_diffusion(gb, domain, tol):

    for g, d in gb:
        param = d["param"]

        kxx = 5 * 1e-2 * np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(g.dim, kxx)
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

            param.set_bc("transport", pp.BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val("transport", bc_val)
        else:
            param.set_bc("transport", pp.BoundaryCondition(g, np.empty(0), np.empty(0)))

    # Assign coupling discharge
    gb.add_edge_props("param")
    for e, d in gb.edges():
        g_h = gb.nodes_of_edge(e)[1]
        discharge = gb.node_props(g_h)["discharge"]
        d["param"] = pp.Parameters(g_h)
        d["discharge"] = discharge

        mg = d['mortar_grid']
        d['flux_field'] = mg.master_to_mortar_int * discharge



# ------------------------------------------------------------------------------#


do_save = False
folder = os.path.dirname(os.path.realpath(__file__)) + "/"
export_folder = folder + "advection_diffusion_coupling"
tol = 1e-3

mesh_kwargs = {"mesh_size_frac": 0.045, "mesh_size_min": 0.01}
domain = {"xmin": -0.2, "xmax": 1.2, "ymin": -0.2, "ymax": 1.2}
gb = pp.importer.dfm_2d_from_csv(folder + "network.csv", mesh_kwargs, domain)
gb.compute_geometry()
gb.assign_node_ordering()

# Assign parameters
add_data_darcy(gb, domain, tol)

# Choose and define the solvers and coupler
key = "flow"
discretization_key = key + "_" + pp.keywords.DISCRETIZATION

darcy = pp.DualEllipticModel(gb)
up = darcy.solve()

darcy.split()
darcy.pressure("pressure")
darcy.discharge("discharge")
#darcy.project_discharge("P0u")

if do_save:
    save = pp.Exporter(gb, "darcy", folder=export_folder)
    save.write_vtk(["pressure"])
    #save.write_vtk(["pressure", "P0u"])

#################################################################

physics = "transport"

# Identifier of the advection term
term = 'advection'
adv = 'advection'
diff = 'diffusion'

adv_discr = pp.Upwind(physics)
diff_discr = pp.Tpfa(physics)

adv_coupling = pp.UpwindCoupling(key)
diff_coupling = pp.RobinCoupling(physics, diff_discr)

key = 'temperature'

for _, d in gb:
    d[pp.keywords.PRIMARY_VARIABLES] = {key: {"cells": 1}}
    d[pp.keywords.DISCRETIZATION] = {key: {adv: adv_discr, diff: diff_discr}}

for e, d in gb.edges():
    g1, g2 = gb.nodes_of_edge(e)
    d[pp.keywords.PRIMARY_VARIABLES] = {"lambda_adv": {
            "cells": 1}, "lambda_diff": {"cells": 1}}
    d[pp.keywords.COUPLING_DISCRETIZATION] = {
            adv: {
                g1: (key, adv),
                g2: (key, adv),
                e: ("lambda_adv", adv_coupling)
            },
            diff: {
                g1: (key, diff),
                g2: (key, diff),
                e: ("lambda_diff", diff_coupling)
            }
        }


# Assign parameters
add_data_advection_diffusion(gb, domain, tol)

assembler = pp.Assembler()
A, b, block_dof, full_dof = assembler.assemble_matrix_rhs(gb)


theta = sps.linalg.spsolve(A, b)
assembler.distribute_variable(gb, theta, block_dof, full_dof)

if do_save:
    exporter.export_vtk(
        gb, "advection_diffusion", ["temperature"], folder=export_folder
    )
