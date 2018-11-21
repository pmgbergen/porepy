import logging, time
import scipy.sparse as sps
import numpy as np
import porepy as pp

import data

# ------------------------------------------------------------------------------#

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------#

def flow(gb, param, physics="flow", keys=("pressure_flux", "pressure", "discharge")):

    param["physics"] = physics
    data.flow(gb, param, param["tol"])

    flux = "flux"
    source = "source"

    flux_discr = pp.MVEM(physics)
    source_discr = pp.DualSource(physics)

    key = keys[0]
    flux_coupling = pp.RobinCoupling(key, flux_discr)

    # define the dof and discretization for the grids
    for g, d in gb:
        d[pp.keywords.PRIMARY_VARIABLES] = {key: {"cells": 1, "faces": 1}}
        d[pp.keywords.DISCRETIZATION] = {key: {flux: flux_discr,
                                               source: source_discr}}

    # define the interface terms to couple the grids
    flux_mortar = "lambda_" + flux
    for e, d in gb.edges():
        g_slave, g_master = gb.nodes_of_edge(e)
        d[pp.keywords.PRIMARY_VARIABLES] = {flux_mortar: {"cells": 1}}
        d[pp.keywords.COUPLING_DISCRETIZATION] = {
            flux: {
                g_slave:  (key, flux),
                g_master: (key, flux),
                e: (flux_mortar, flux_coupling)
            }
        }

    # solution of the darcy problem
    assembler = pp.Assembler()

    A, b, block_dof, full_dof = assembler.assemble_matrix_rhs(gb)
    up = sps.linalg.spsolve(A, b)

    assembler.distribute_variable(gb, up, block_dof, full_dof)
    for g, d in gb:
        flow_discr = d["discretization"][key][flux]
        d[keys[1]] = flow_discr.extract_pressure(g, d[key])
        d[keys[2]] = flow_discr.extract_flux(g, d[key])

    return keys

# ------------------------------------------------------------------------------#

def advdiff(gb, param, physics="transport", key="scalar"):

    param["physics"] = physics
    data.advdiff(gb, param, param["tol"])

    adv = "advection"
    diff = "diffusion"

    adv_discr = pp.Upwind(physics)
    diff_discr = pp.Tpfa(physics)

    adv_coupling = pp.UpwindCoupling(key)
    diff_coupling = pp.RobinCoupling(key, diff_discr)

    for g, d in gb:
        d[pp.keywords.PRIMARY_VARIABLES] = {key: {"cells": 1}}
        d[pp.keywords.DISCRETIZATION] = {key: {adv: adv_discr,
                                               diff: diff_discr}}

    mortar_adv = "lambda_" + adv
    mortar_diff = "lambda_" + diff
    for e, d in gb.edges():
        g_slave, g_master = gb.nodes_of_edge(e)
        d[pp.keywords.PRIMARY_VARIABLES] = {mortar_adv: {"cells": 1},
                                            mortar_diff: {"cells": 1}}
        d[pp.keywords.COUPLING_DISCRETIZATION] = {
                adv: {
                    g_slave: (key, adv),
                    g_master: (key, adv),
                    e: (mortar_adv, adv_coupling)
                },
                diff: {
                    g_slave: (key, diff),
                    g_master: (key, diff),
                    e: (mortar_diff, diff_coupling)
                }
            }

    # setup the advection-diffusion problem
    assembler = pp.Assembler()
    A, b, block_dof, full_dof = assembler.assemble_matrix_rhs(gb)

    # mass term
    mass = "mass"
    mass_discr = pp.MassMatrix(physics)

    for g, d in gb:
        d[pp.keywords.PRIMARY_VARIABLES] = {key: {"cells": 1}}
        d[pp.keywords.DISCRETIZATION] = {key: {mass: mass_discr}}

    gb.remove_edge_props(pp.keywords.COUPLING_DISCRETIZATION)

    for e, d in gb.edges():
        g_slave, g_master = gb.nodes_of_edge(e)
        d[pp.keywords.PRIMARY_VARIABLES] = {mortar_adv: {"cells": 1},
                                            mortar_diff: {"cells": 1}}

    M, _, _, _ = assembler.assemble_matrix_rhs(gb)

    # Perform an LU factorization to speedup the solver
    IE_solver = sps.linalg.factorized((M + A).tocsc())

    # time loop
    save = pp.Exporter(gb, "solution", folder=param["folder"])

    x = np.ones(A.shape[0]) * param["initial_advdiff"]
    for i in np.arange(param["n_steps"]):
        x = IE_solver(b + M.dot(x))

        assembler.distribute_variable(gb, x, block_dof, full_dof)
        save.write_vtk([param["keys_flow"][1], key], time_step=i)

    save.write_pvd(np.arange(param["n_steps"])*param["deltaT"])

