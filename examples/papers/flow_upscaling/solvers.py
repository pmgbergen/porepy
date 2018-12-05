import logging, time, sys
import scipy.sparse as sps
import numpy as np
import porepy as pp

import data

# ------------------------------------------------------------------------------#

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

def flow(gb, param):

    model = "flow"

    model_data = data.flow(gb, model, param)

    # discretization operator name
    flux_id = "flux"

    # master variable name
    variable = "flux_pressure"
    mortar = "lambda_" + variable

    # post process variables
    pressure = "pressure"
    flux = "darcy_flux" # it has to be this one
    P0_flux = "P0_flux"

    # save variable name for the advection-diffusion problem
    param["pressure"] = pressure
    param["flux"] = flux
    param["P0_flux"] = P0_flux
    param["mortar_flux"] = mortar

    # discretization operators
    discr = pp.MVEM(model_data)
    coupling = pp.RobinCoupling(model_data, discr)

    # define the dof and discretization for the grids
    for g, d in gb:
        d[pp.PRIMARY_VARIABLES] = {variable: {"cells": 1, "faces": 1}}
        d[pp.DISCRETIZATION] = {variable: {flux_id: discr}}

    # define the interface terms to couple the grids
    for e, d in gb.edges():
        g_slave, g_master = gb.nodes_of_edge(e)
        d[pp.PRIMARY_VARIABLES] = {mortar: {"cells": 1}}
        d[pp.COUPLING_DISCRETIZATION] = {
            variable: {
                g_slave:  (variable, flux_id),
                g_master: (variable, flux_id),
                e: (mortar, coupling)
            }
        }

    # solution of the darcy problem
    assembler = pp.Assembler()

    logger.info("Assemble the flow problem")
    A, b, block_dof, full_dof = assembler.assemble_matrix_rhs(gb)
    logger.info("done")

    logger.info("Solve the linear system")
    up = sps.linalg.spsolve(A, b)
    logger.info("done")

    logger.info("Variable post-process")
    assembler.distribute_variable(gb, up, block_dof, full_dof)
    for g, d in gb:
        d[pressure] = discr.extract_pressure(g, d[variable])
        d[flux] = discr.extract_flux(g, d[variable])

    pp.project_flux(gb, discr, flux, P0_flux, mortar)
    logger.info("done")

    return model_data

# ------------------------------------------------------------------------------#

def advdiff(gb, param, model_flow):

    model = "transport"

    model_data_adv, model_data_diff = data.advdiff(gb, model, model_flow, param)

    # discretization operator names
    adv_id = "advection"
    diff_id = "diffusion"

    # variable names
    variable = "scalar"
    mortar_adv = "lambda_" + variable + "_" + adv_id
    mortar_diff = "lambda_" + variable + "_" + diff_id

    # discretization operatr
    discr_adv = pp.Upwind(model_data_adv)
    discr_diff = pp.Tpfa(model_data_diff)

    coupling_adv = pp.UpwindCoupling(model_data_adv)
    coupling_diff = pp.RobinCoupling(model_data_diff, discr_diff)

    for g, d in gb:
        d[pp.PRIMARY_VARIABLES] = {variable: {"cells": 1}}
        d[pp.DISCRETIZATION] = {variable: {adv_id: discr_adv,
                                           diff_id: discr_diff}}

    for e, d in gb.edges():
        g_slave, g_master = gb.nodes_of_edge(e)
        d[pp.PRIMARY_VARIABLES] = {mortar_adv: {"cells": 1},
                                   mortar_diff: {"cells": 1}}

        d[pp.COUPLING_DISCRETIZATION] = {
                adv_id: {
                    g_slave: (variable, adv_id),
                    g_master: (variable, adv_id),
                    e: (mortar_adv, coupling_adv)
                },
                diff_id: {
                    g_slave: (variable, diff_id),
                    g_master: (variable, diff_id),
                    e: (mortar_diff, coupling_diff)
                }
            }

    # setup the advection-diffusion problem
    assembler = pp.Assembler()
    logger.info("Assemble the advective and diffusive terms of the transport problem")
    A, b, block_dof, full_dof = assembler.assemble_matrix_rhs(gb)
    logger.info("done")

    # mass term
    mass_id = "mass"
    discr_mass = pp.MassMatrix(model_data_adv)

    for g, d in gb:
        d[pp.PRIMARY_VARIABLES] = {variable: {"cells": 1}}
        d[pp.DISCRETIZATION] = {variable: {mass_id: discr_mass}}

    gb.remove_edge_props(pp.COUPLING_DISCRETIZATION)

    for e, d in gb.edges():
        g_slave, g_master = gb.nodes_of_edge(e)
        d[pp.PRIMARY_VARIABLES] = {mortar_adv: {"cells": 1},
                                   mortar_diff: {"cells": 1}}

    logger.info("Assemble the mass term of the transport problem")
    M, _, _, _ = assembler.assemble_matrix_rhs(gb)
    logger.info("done")

    # Perform an LU factorization to speedup the solver
    #IE_solver = sps.linalg.factorized((M + A).tocsc())

    # time loop
    logger.info("Prepare the exporting")
    save = pp.Exporter(gb, "solution", folder=param["folder"])
    logger.info("done")
    variables = [variable, param["pressure"], param["P0_flux"]]

    x = np.ones(A.shape[0]) * param["initial_advdiff"]
    logger.info("Start the time loop with " + str(param["n_steps"]) + " steps")
    for i in np.arange(param["n_steps"]):
        #x = IE_solver(b + M.dot(x))
        logger.info("Solve the linear system for time step " + str(i))
        x = sps.linalg.spsolve(M + A, b + M.dot(x))
        logger.info("done")

        logger.info("Variable post-process")
        assembler.distribute_variable(gb, x, block_dof, full_dof)
        logger.info("done")

        logger.info("Export variable")
        save.write_vtk(variables, time_step=i)
        logger.info("done")

    save.write_pvd(np.arange(param["n_steps"])*param["time_step"])

