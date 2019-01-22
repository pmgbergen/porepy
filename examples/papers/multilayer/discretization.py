import logging, time, sys
import scipy.sparse as sps
import numpy as np
import porepy as pp

from examples.papers.multilayer.multilayer_interface_law import RobinCouplingMultiLayer

def setup_custom_logger():
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler('log.txt', mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger

logger = setup_custom_logger()

# ------------------------------------------------------------------------------#

def data_flow(gb, model, data, bc_flag):
    tol = data["tol"]

    model_data = model + "_data"

    for g, d in gb:
        param = {}

        unity = np.ones(g.num_cells)
        zeros = np.zeros(g.num_cells)
        empty = np.empty(0)

        d["is_tangential"] = True
        d["tol"] = tol

        # assign permeability
        if "fault" in g.name:
            data_fault = data["fault"]
            kxx = data_fault["kf_t"] * unity
            perm = pp.SecondOrderTensor(1, kxx=kxx, kyy=1, kzz=1)
            aperture = data_fault["aperture"] * unity

        elif "layer" in g.name:
            data_layer = data["layer"]
            kxx = data_layer["kf_t"] * unity
            perm = pp.SecondOrderTensor(1, kxx=kxx, kyy=1, kzz=1)
            aperture = data_layer["aperture"] * unity

        else:
            kxx = data["k"] * unity
            if g.dim == 2:
                perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=kxx, kzz=1)
            else:
                perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=kxx, kzz=kxx)
            aperture = unity

        param["second_order_tensor"] = perm
        param["aperture"] = aperture

        # source
        param["source"] = zeros

        # Boundaries
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size:
            labels, bc_val = bc_flag(g, data, tol)
            param["bc"] = pp.BoundaryCondition(g, b_faces, labels)
        else:
            bc_val = np.zeros(g.num_faces)
            param["bc"] = pp.BoundaryCondition(g, empty, empty)

        param["bc_values"] = bc_val

        d[pp.PARAMETERS] = pp.Parameters(g, model_data, param)
        d[pp.DISCRETIZATION_MATRICES] = {model_data: {}}

    for e, d in gb.edges():
        g_l = gb.nodes_of_edge(e)[0]

        if "layer" in g_l.name:
            data_interface = data["layer"]
        else:
            data_interface = data["fault"]

        mg = d["mortar_grid"]
        check_P = mg.slave_to_mortar_avg()

        aperture = gb.node_props(g_l, pp.PARAMETERS)[model_data]["aperture"]
        gamma = check_P * aperture
        kn = data_interface["kf_n"] * np.ones(mg.num_cells) / gamma

        param = {"normal_diffusivity": kn}

        d[pp.PARAMETERS] = pp.Parameters(e, model_data, param)
        d[pp.DISCRETIZATION_MATRICES] = {model_data: {}}

    return model_data

# ------------------------------------------------------------------------------#

def flow(gb, param, bc_flag):

    model = "flow"

    model_data = data_flow(gb, model, param, bc_flag)

    # discretization operator name
    flux_id = "flux"

    # master variable name
    variable = "flow_variable"
    mortar = "lambda_" + variable

    # post process variables
    pressure = "pressure"
    flux = "darcy_flux" # it has to be this one

    # save variable name for the advection-diffusion problem
    param["pressure"] = pressure
    param["flux"] = flux
    param["mortar_flux"] = mortar

    discr = pp.RT0(model_data)
    coupling = pp.RobinCoupling(model_data, discr)
    coupling_multilayer = RobinCouplingMultiLayer(model_data, discr)

    # define the dof and discretization for the grids
    for g, d in gb:
        d[pp.PRIMARY_VARIABLES] = {variable: {"cells": 1, "faces": 1}}
        d[pp.DISCRETIZATION] = {variable: {flux_id: discr}}

    # define the interface terms to couple the grids
    for e, d in gb.edges():
        g_slave, g_master = gb.nodes_of_edge(e)
        d[pp.PRIMARY_VARIABLES] = {mortar: {"cells": 1}}
        if g_master.dim == gb.dim_max():
            # classical 2d-1d/3d-2d coupling condition
            d[pp.COUPLING_DISCRETIZATION] = {
                flux: {
                    g_slave:  (variable, flux_id),
                    g_master: (variable, flux_id),
                    e: (mortar, coupling)
                }
            }
        elif g_master.dim < gb.dim_max():
            # the multilayer coupling condition
            d[pp.COUPLING_DISCRETIZATION] = {
                flux: {
                    g_slave:  (variable, flux_id),
                    g_master: (variable, flux_id),
                    e: (mortar, coupling_multilayer)
                }
            }


    # solution of the darcy problem
    assembler = pp.Assembler()

    logger.info("Assemble the flow problem")
    A, b, block_dof, full_dof = assembler.assemble_matrix_rhs(gb)
    logger.info("done")

    logger.info("Solve the linear system")
    x = sps.linalg.spsolve(A, b)
    logger.info("done")

    logger.info("Variable post-process")
    assembler.distribute_variable(gb, x, block_dof, full_dof)
    for g, d in gb:
        d[pressure] = discr.extract_pressure(g, d[variable], d)
        d[flux] = discr.extract_flux(g, d[variable], d)

    # export the P0 flux reconstruction
    P0_flux = "P0_flux"
    param["P0_flux"] = P0_flux
    pp.project_flux(gb, discr, flux, P0_flux, mortar)

    save = pp.Exporter(gb, "solution", folder=param["folder"])
    save.write_vtk([pressure, P0_flux])

    logger.info("done")

# ------------------------------------------------------------------------------#
