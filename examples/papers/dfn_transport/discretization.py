import logging, time, sys
import scipy.sparse as sps
import numpy as np
import porepy as pp

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

def get_discr():
    return { "MVEM": {"scheme": pp.MVEM, "dof": {"cells": 1, "faces": 1}},
             "RT0":  {"scheme": pp.RT0,  "dof": {"cells": 1, "faces": 1}},
             "Tpfa": {"scheme": pp.Tpfa, "dof": {"cells": 1}}}

# ------------------------------------------------------------------------------#

def data_flow(gb, discr, model, data, bc_flag):
    tol = data["tol"]

    model_data = model + "_data"

    for g, d in gb:
        param = {}

        unity = np.ones(g.num_cells)
        zeros = np.zeros(g.num_cells)
        empty = np.empty(0)

        d["frac_num"] = (g.frac_num if g.dim == 2 else -1) * unity
        d["cell_volumes"] = g.cell_volumes
        d["is_tangential"] = True
        d["Aavatsmark_transmissibilities"] = True
        d["tol"] = tol

        # assign permeability
        kxx = data["k"] * unity
        if discr["scheme"] is pp.MVEM or discr["scheme"] is pp.RT0:
            perm = pp.SecondOrderTensor(2, kxx=kxx, kyy=kxx, kzz=1)

        elif discr["scheme"] is pp.Tpfa:
            perm = pp.SecondOrderTensor(3, kxx=kxx, kyy=kxx, kzz=kxx)

        else:
            raise ValueError

        param["second_order_tensor"] = perm

        # assign aperture
        param["aperture"] = unity

        # source
        param["source"] = zeros

        # Boundaries
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        bc_val = np.zeros(g.num_faces)
        if b_faces.size:
            in_flow, out_flow = bc_flag(g, data["domain"], tol)

            labels = np.array(["neu"] * b_faces.size)
            labels[in_flow + out_flow] = "dir"
            param["bc"] = pp.BoundaryCondition(g, b_faces, labels)

            bc_val = np.zeros(g.num_faces)
            bc_val[b_faces[in_flow]] = data.get("bc_flow", 1)
        else:
            param["bc"] = pp.BoundaryCondition(g, empty, empty)

        param["bc_values"] = bc_val

        d[pp.PARAMETERS] = pp.Parameters(g, model_data, param)
        d[pp.DISCRETIZATION_MATRICES] = {model_data: {}}

    for _, d in gb.edges():
        d[pp.DISCRETIZATION_MATRICES] = {model_data: {}}

    return model_data

# ------------------------------------------------------------------------------#

def flow(gb, discr, param, bc_flag):

    model = "flow"

    model_data = data_flow(gb, discr, model, param, bc_flag)

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

    discr_scheme = discr["scheme"](model_data)
    discr_empty = pp.EmptyDiscretization(model_data)

    coupling = pp.FluxPressureContinuity(model_data, discr_scheme, discr_empty)

    # define the dof and discretization for the grids
    for g, d in gb:
        if g.dim == gb.dim_max():
            d[pp.PRIMARY_VARIABLES] = {variable: discr["dof"]}
            d[pp.DISCRETIZATION] = {variable: {flux_id: discr_scheme}}
        else:
            d[pp.PRIMARY_VARIABLES] = {variable: {"cells": 1}}
            d[pp.DISCRETIZATION] = {variable: {flux_id: discr_empty}}

    # define the interface terms to couple the grids
    for e, d in gb.edges():
        g_slave, g_master = gb.nodes_of_edge(e)
        d[pp.PRIMARY_VARIABLES] = {mortar: {"cells": 1}}
        d[pp.COUPLING_DISCRETIZATION] = {
            flux: {
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
    x = sps.linalg.spsolve(A, b)
    logger.info("done")

    logger.info("Variable post-process")
    assembler.distribute_variable(gb, x, block_dof, full_dof)

    # extract the pressure from the solution
    for g, d in gb:
        if g.dim == 2:
            d[pressure] = discr_scheme.extract_pressure(g, d[variable], d)
            d[flux] = discr_scheme.extract_flux(g, d[variable], d)
        else:
            d[pressure] = np.zeros(g.num_cells)
            d[flux] = np.zeros(g.num_faces)

    # export the P0 flux reconstruction only for some scheme
    if discr["scheme"] is pp.MVEM or discr["scheme"] is pp.RT0:
        P0_flux = "P0_flux"
        param["P0_flux"] = P0_flux
        pp.project_flux(gb, discr_scheme, flux, P0_flux, mortar)

    logger.info("done")

    return model_data

# ------------------------------------------------------------------------------#

def data_advdiff(gb, model, model_flow, data, bc_flag):
    tol = data["tol"]

    model_data_adv = model + "_data_adv"
    model_data_diff = model + "_data_diff"

    flux_discharge_name = data["flux"]
    flux_mortar_name = data["mortar_flux"]

    for g, d in gb:
        param_adv = {}
        param_diff = {}

        d["Aavatsmark_transmissibilities"] = True
        unity = np.ones(g.num_cells)
        zeros = np.zeros(g.num_cells)
        empty = np.empty(0)

        # weight for the mass matrix
        param_adv["mass_weight"] = data.get("mass_weight", 1) * unity

        # diffusion term
        kxx = data["diff"] * unity
        param_diff["second_order_tensor"] = pp.SecondOrderTensor(3, kxx)

        # Assign apertures
        param_diff["aperture"] = unity
        param_adv["aperture"] = unity

        # Flux
        param_adv[flux_discharge_name] = data.get("flux_weight", 1) * d[flux_discharge_name]

        # Boundaries
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        bc_val = np.zeros(g.num_faces)
        if b_faces.size:
            in_flow, out_flow = bc_flag(g, data["domain"], tol)

            labels_adv = np.array(["neu"] * b_faces.size)
            labels_adv[in_flow + out_flow] = ["dir"]

            labels_diff = np.array(["neu"] * b_faces.size)
            labels_diff[in_flow] = ["dir"]

            param_adv["bc"] = pp.BoundaryCondition(g, b_faces, labels_adv)
            param_diff["bc"] = pp.BoundaryCondition(g, b_faces, labels_diff)

            bc_val = np.zeros(g.num_faces)
            bc_val[b_faces[in_flow]] = data.get("bc_trans", 1)
        else:
            param_adv["bc"] = pp.BoundaryCondition(g, np.empty(0), np.empty(0))
            param_diff["bc"] = pp.BoundaryCondition(g, np.empty(0), np.empty(0))

        param_adv["bc_values"] = bc_val
        param_diff["bc_values"] = bc_val

        # Assign time step
        param_adv["time_step"] = data["time_step"]

        param = pp.Parameters(g, [model_data_adv, model_data_diff], [param_adv, param_diff])
        d[pp.PARAMETERS] = param
        d[pp.DISCRETIZATION_MATRICES] = {model_data_adv: {}, model_data_diff: {}}

    for e, d in gb.edges():
        param_adv = {}
        param_diff = {}

        param_adv[flux_discharge_name] = data.get("flux_weight", 1) * d[flux_mortar_name]

        param = pp.Parameters(e, [model_data_adv, model_data_diff], [param_adv, param_diff])
        d[pp.PARAMETERS] = param
        d[pp.DISCRETIZATION_MATRICES] = {model_data_adv: {}, model_data_diff: {}}

    return model_data_adv, model_data_diff

# ------------------------------------------------------------------------------#

def advdiff(gb, discr, param, model_flow, bc_flag):

    model = "transport"

    model_data_adv, model_data_diff = data_advdiff(gb, model, model_flow, param, bc_flag)

    # discretization operator names
    adv_id = "advection"
    diff_id = "diffusion"

    # variable names
    variable = "scalar"
    mortar_adv = "lambda_" + variable + "_" + adv_id
    mortar_diff = "lambda_" + variable + "_" + diff_id

    # save variable name for the post-process
    param["scalar"] = variable

    adv = "advection"
    diff = "diffusion"

    discr_adv = pp.Upwind(model_data_adv)
    discr_adv_empty = pp.EmptyDiscretization(model_data_adv)

    discr_diff = pp.Tpfa(model_data_diff)
    discr_diff_empty = pp.EmptyDiscretization(model_data_diff)

    coupling_adv = pp.UpwindCoupling(model_data_adv)
    coupling_diff = pp.FluxPressureContinuity(model_data_diff, discr_diff, discr_diff_empty)

    for g, d in gb:
        d[pp.PRIMARY_VARIABLES] = {variable: {"cells": 1}}
        if g.dim == gb.dim_max():
            d[pp.DISCRETIZATION] = {variable: {adv_id: discr_adv, diff_id: discr_diff}}
        else:
            d[pp.DISCRETIZATION] = {variable: {adv_id: discr_adv_empty, diff_id: discr_diff_empty}}

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
    discr_mass_empty = pp.EmptyDiscretization(model_data_adv)

    for g, d in gb:
        d[pp.PRIMARY_VARIABLES] = {variable: {"cells": 1}}
        if g.dim == gb.dim_max():
            d[pp.DISCRETIZATION] = {variable: {mass_id: discr_mass}}
        else:
            d[pp.DISCRETIZATION] = {variable: {mass_id: discr_mass_empty}}

    gb.remove_edge_props(pp.COUPLING_DISCRETIZATION)

    for e, d in gb.edges():
        g_slave, g_master = gb.nodes_of_edge(e)
        d[pp.PRIMARY_VARIABLES] = {mortar_adv: {"cells": 1},
                                   mortar_diff: {"cells": 1}}

    logger.info("Assemble the mass term of the transport problem")
    M, _, _, _ = assembler.assemble_matrix_rhs(gb)
    logger.info("done")

    # Perform an LU factorization to speedup the solver
    IE_solver = sps.linalg.factorized((M + A).tocsc())

    # time loop
    logger.info("Prepare the exporting")
    save = pp.Exporter(gb, "solution", folder=param["folder"])
    logger.info("done")

    variables = [variable, param["pressure"], "frac_num", "cell_volumes"]
    if discr["scheme"] is pp.MVEM or discr["scheme"] is pp.RT0:
        variables.append(param["P0_flux"])

    # assign the initial condition
    x = np.zeros(A.shape[0])
    assembler.distribute_variable(gb, x, block_dof, full_dof)
    for g, d in gb:
        if g.dim == gb.dim_max():
            d[variable] = param.get("init_trans", 0) * np.ones(g.num_cells)

    x = assembler.merge_variable(gb, variable, block_dof, full_dof)

    outflow = np.zeros(param["n_steps"])

    logger.info("Start the time loop with " + str(param["n_steps"]) + " steps")
    for i in np.arange(param["n_steps"]):
        logger.info("Solve the linear system for time step " + str(i))
        x = IE_solver(b + M.dot(x))
        logger.info("done")

        logger.info("Variable post-process")
        assembler.distribute_variable(gb, x, block_dof, full_dof)
        logger.info("done")

        logger.info("Export variable")
        save.write_vtk(variables, time_step=i)
        logger.info("done")

        logger.info("Compute the production")
        outflow[i] = compute_outflow(gb, param)
        logger.info("done")

    time = np.arange(param["n_steps"])*param["time_step"]
    save.write_pvd(time)

    logger.info("Save outflow on file")
    file_out = param["folder"] + "/outflow.csv"
    data = np.vstack((time, outflow)).T
    np.savetxt(file_out, data, delimiter=',')
    logger.info("done")

# ------------------------------------------------------------------------------#

def compute_outflow(gb, param):
    outflow = 0.0
    for g, d in gb:
        if g.dim < 2:
            continue
        faces, cells, sign = sps.find(g.cell_faces)
        index = np.argsort(cells)
        faces, sign = faces[index], sign[index]

        flux = d[param["flux"]].copy()
        scalar = d[param["scalar"]]

        flux[faces] *= sign
        flux[g.get_internal_faces()] = 0
        flux[flux < 0] = 0
        #outflow += np.dot(flux, np.abs(g.cell_faces).dot(scalar))

        flux[flux != 0] = 1
        outflow += np.dot(flux*g.face_areas, np.abs(g.cell_faces).dot(scalar))

    return outflow

# ------------------------------------------------------------------------------#
