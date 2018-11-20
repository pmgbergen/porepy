import numpy as np
import porepy as pp

def flow(gb, physics="flow", key="pressure_flux"):

    flux = "flux"
    source = "source"

    flux_discr = pp.MVEM(physics)
    source_discr = pp.DualSource(physics)
    trace_discr = pp.PressureTrace(physics)

    flux_coupling = pp.FluxPressureContinuity(key, flux_discr, trace_discr)

    # define the dof and discretization for the grids
    for g, d in gb:
        if g.dim == gb.dim_max():
            d[pp.keywords.PRIMARY_VARIABLES] = {key: {"cells": 1, "faces": 1}}
            d[pp.keywords.DISCRETIZATION] = {key: {flux: flux_discr,
                                                   source: source_discr}}
        else:
            d[pp.keywords.PRIMARY_VARIABLES] = {key: {"cells": 1}}
            d[pp.keywords.DISCRETIZATION] = {key: {flux: trace_discr}}

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

    return (key, flux)

def advdiff(gb, physics="transport", key="scalar"):

    # Identifier of the advection term
    adv = "advection"
    diff = "diffusion"

    adv_discr = pp.Upwind(physics)
    diff_discr = pp.Tpfa(physics)
    trace_discr = pp.PressureTrace(physics)

    adv_coupling = pp.UpwindCoupling(key)
    diff_coupling = pp.FluxPressureContinuity(key, diff_discr, trace_discr)

    for g, d in gb:
        if g.dim == gb.dim_max():
            d[pp.keywords.PRIMARY_VARIABLES] = {key: {"cells": 1}}
            d[pp.keywords.DISCRETIZATION] = {key: {adv: adv_discr,
                                                   diff: diff_discr}}
        else:
            d[pp.keywords.PRIMARY_VARIABLES] = {key: {"cells": 1}}
            d[pp.keywords.DISCRETIZATION] = {key: {adv: trace_discr,
                                                   diff: trace_discr}}

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

    return key
