""" This test has been disabled, as the DualElliptic class is obsolete.
"""
import numpy as np
import scipy.sparse as sps
import os
import porepy as pp


# ------------------------------------------------------------------------------#


def add_data_darcy(gb, domain, tol):

    kf = 1e-4
    for g, d in gb:
        flow_parameter_dictionary = {}

        kxx = np.ones(g.num_cells) * np.power(kf, g.dim < gb.dim_max())
        perm = pp.SecondOrderTensor(g.dim, kxx)
        flow_parameter_dictionary["second_order_tensor"] = perm

        flow_parameter_dictionary["source"] = np.zeros(g.num_cells)

        aperture = np.power(1e-2, gb.dim_max() - g.dim)
        flow_parameter_dictionary["aperture"] = np.ones(g.num_cells) * aperture

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

            flow_parameter_dictionary["bc"] = pp.BoundaryCondition(
                g, bound_faces, labels
            )
            flow_parameter_dictionary["bc_values"] = bc_val
        else:
            flow_parameter_dictionary["bc"] = pp.BoundaryCondition(
                g, np.empty(0), np.empty(0)
            )
            flow_parameter_dictionary["bc_values"] = np.zeros(g.num_faces)
        param = pp.Parameters(g)
        param.update_dictionaries("flow", flow_parameter_dictionary)
        d[pp.PARAMETERS] = param
        d[pp.DISCRETIZATION_MATRICES] = {}
        d[pp.DISCRETIZATION_MATRICES]["flow"] = {}

    # Assign coupling permeability
    for e, d in gb.edges():
        g_l = gb.nodes_of_edge(e)[0]
        aperture = np.power(1e-2, gb.dim_max() - g_l.dim) * np.ones(g_l.num_cells)

        mg = d["mortar_grid"]
        check_P = mg.slave_to_mortar_avg()

        gamma = check_P * aperture
        kn = 2 * kf * np.ones(mg.num_cells) / gamma
        flow_dictionary = {"normal_diffusivity": kn, "aperture": aperture}
        d[pp.PARAMETERS] = pp.Parameters(
            keywords=["flow"], dictionaries=[flow_dictionary]
        )
        d[pp.DISCRETIZATION_MATRICES] = {}
        d[pp.DISCRETIZATION_MATRICES]["flow"] = {}


# ------------------------------------------------------------------------------#


def add_data_advection_diffusion(gb, domain, tol):
    diffusion_coefficient = 5 * 1e-2
    for g, d in gb:
        param = d[pp.PARAMETERS]
        transport_parameter_dictionary = {}
        param.update_dictionaries("transport", transport_parameter_dictionary)
        param.set_from_other("transport", "flow", ["aperture"])
        d[pp.DISCRETIZATION_MATRICES]["transport"] = {}

        kxx = diffusion_coefficient * np.ones(g.num_cells)
        cond = pp.SecondOrderTensor(g.dim, kxx)
        transport_parameter_dictionary["second_order_tensor"] = cond

        # The 0.5 needs to be fixed in a better way
        #        source = 0.5 * np.ones(g.num_cells) * g.cell_volumes * \
        #            transport_parameter_dictionary["aperture"]
        source = 0.5 * g.cell_volumes * transport_parameter_dictionary["aperture"]
        transport_parameter_dictionary["source"] = source

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

            transport_parameter_dictionary["bc"] = pp.BoundaryCondition(
                g, bound_faces, labels
            )
            transport_parameter_dictionary["bc_values"] = bc_val
        else:
            transport_parameter_dictionary["bc"] = pp.BoundaryCondition(
                g, np.empty(0), np.empty(0)
            )
            transport_parameter_dictionary["bc_values"] = np.zeros(g.num_faces)

    # Assign coupling darcy_flux
    for e, d in gb.edges():
        g_h = gb.nodes_of_edge(e)[1]
        darcy_flux = gb.node_props(g_h)["darcy_flux"]
        d["darcy_flux"] = darcy_flux

        mg = d["mortar_grid"]
        check_P = mg.slave_to_mortar_avg()
        aperture = d[pp.PARAMETERS]["flow"]["aperture"]
        gamma = check_P * aperture
        kn = 2 * diffusion_coefficient * np.ones(mg.num_cells) / gamma
        transport_dictionary = {"normal_diffusivity": kn}
        d["darcy_flux"] = mg.master_to_mortar_int * darcy_flux
        d[pp.PARAMETERS].update_dictionaries("transport", transport_dictionary)
        d[pp.DISCRETIZATION_MATRICES]["transport"] = {}


# ------------------------------------------------------------------------------#


do_save = False
folder = os.path.dirname(os.path.realpath(__file__)) + "/"
export_folder = folder + "advection_diffusion_coupling"
tol = 1e-3

# Define the domain and make a grid bucket
mesh_kwargs = {"mesh_size_frac": 0.045, "mesh_size_min": 0.01}
domain = {"xmin": -0.2, "xmax": 1.2, "ymin": -0.2, "ymax": 1.2}
gb = pp.importer.dfm_2d_from_csv(folder + "network.csv", mesh_kwargs, domain)
gb.compute_geometry()
gb.assign_node_ordering()

# Keyword for parameters and discretization operators
flow_kw = "flow"

# Assign parameters
add_data_darcy(gb, domain, tol)

# Choose and define the solvers and coupler


darcy = pp.DualEllipticModel(gb)
up = darcy.solve()

darcy.split(x_name="pressure")
darcy.pressure("pressure")
darcy.darcy_flux("darcy_flux")

if do_save:
    save = pp.Exporter(gb, "darcy", folder=export_folder)
    save.write_vtk(["pressure"])

#################################################################

# Keyword for parameters and discretization operators
temperature_kw = "transport"

# Identifier of the two terms of the equation
adv = "advection"
diff = "diffusion"

adv_discr = pp.Upwind(temperature_kw)
diff_discr = pp.Tpfa(temperature_kw)

adv_coupling = pp.UpwindCoupling(temperature_kw)
diff_coupling = pp.RobinCoupling(temperature_kw, diff_discr)

key = "temperature"

for _, d in gb:
    d[pp.PRIMARY_VARIABLES] = {key: {"cells": 1}}
    d[pp.DISCRETIZATION] = {key: {adv: adv_discr, diff: diff_discr}}

for e, d in gb.edges():
    g1, g2 = gb.nodes_of_edge(e)
    d[pp.PRIMARY_VARIABLES] = {"lambda_adv": {"cells": 1}, "lambda_diff": {"cells": 1}}
    d[pp.COUPLING_DISCRETIZATION] = {
        adv: {g1: (key, adv), g2: (key, adv), e: ("lambda_adv", adv_coupling)},
        diff: {g1: (key, diff), g2: (key, diff), e: ("lambda_diff", diff_coupling)},
    }


# Assign parameters
add_data_advection_diffusion(gb, domain, tol)

assembler = pp.Assembler()
A, b, block_dof, full_dof = assembler.assemble_matrix_rhs(gb)


theta = sps.linalg.spsolve(A, b)
assembler.distribute_variable(gb, theta, block_dof, full_dof)

if do_save:
    pp.exporter.export_vtk(
        gb, "advection_diffusion", ["temperature"], folder=export_folder
    )
