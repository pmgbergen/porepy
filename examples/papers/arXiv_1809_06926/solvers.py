import logging, time
import scipy.sparse as sps
import scipy.io as sps_io
import numpy as np

import porepy as pp

# ------------------------------------------------------------------------------#

logger = logging.getLogger(__name__)


def run_flow(gb, node_discretization, source_discretization, folder, is_FV):
    kw_t = "transport"  # For storing fluxes

    grid_variable = "pressure"
    mortar_variable = "mortar_flux"

    # Identifier of the discretization operator on each grid
    diffusion_term = "diffusion"
    source_term = "source"
    # Identifier of the discretization operator between grids
    coupling_operator_keyword = "coupling_operator"

    edge_discretization = pp.RobinCoupling(
        "flow", node_discretization, node_discretization
    )

    # Loop over the nodes in the GridBucket, define primary variables and discretization schemes
    for g, d in gb:
        # Assign primary variables on this grid. It has one degree of freedom per cell.
        d[pp.PRIMARY_VARIABLES] = {grid_variable: {"cells": 1, "faces": int(not is_FV)}}
        # Assign discretization operator for the variable.
        d[pp.DISCRETIZATION] = {
            grid_variable: {
                diffusion_term: node_discretization,
                source_term: source_discretization,
            }
        }

    # Loop over the edges in the GridBucket, define primary variables and discretizations
    for e, d in gb.edges():
        g1, g2 = gb.nodes_of_edge(e)
        # The mortar variable has one degree of freedom per cell in the mortar grid
        d[pp.PRIMARY_VARIABLES] = {mortar_variable: {"cells": 1}}

        # The coupling discretization links an edge discretization with variables
        # and discretization operators on each neighboring grid
        d[pp.COUPLING_DISCRETIZATION] = {
            coupling_operator_keyword: {
                g1: (grid_variable, diffusion_term),
                g2: (grid_variable, diffusion_term),
                e: (mortar_variable, edge_discretization),
            }
        }

    logger.info(
        "Assembly and discretization using the discretizer " + str(node_discretization)
    )
    tic = time.time()

    assembler = pp.Assembler(gb, active_variables=[grid_variable, mortar_variable])

    # Assemble the linear system, using the information stored in the GridBucket
    A, b = assembler.assemble_matrix_rhs()
    logger.info("Done. Elapsed time: " + str(time.time() - tic))

    logger.info("Linear solver")
    tic = time.time()
    x = sps.linalg.spsolve(A, b)
    logger.info("Done. Elapsed time " + str(time.time() - tic))
    #    if is_FV:
    assembler.distribute_variable(x)
    if is_FV:
        pp.fvutils.compute_darcy_flux(gb, lam_name=mortar_variable, keyword_store=kw_t)
    else:
        for g, d in gb:
            discr_scheme = d[pp.DISCRETIZATION][grid_variable][diffusion_term]

            d[pp.PARAMETERS][kw_t]["darcy_flux"] = discr_scheme.extract_flux(
                g, d[pp.STATE][grid_variable], d
            )
            # Note the order: we overwrite d["pressure"] so this has to be done after
            # extracting the flux
            d[pp.STATE]["pressure"] = discr_scheme.extract_pressure(g, d[pp.STATE][grid_variable], d)
        for e, d in gb.edges():
            #            g1, g2 = gb.nodes_of_edge(e)
            d[pp.PARAMETERS][kw_t]["darcy_flux"] = d[pp.STATE][mortar_variable].copy()
    export_flow(gb, folder)
    sps_io.mmwrite(folder + "/matrix.mtx", A)
    return A, b, block_dof, full_dof


def export_flow(gb, folder):

    for g, d in gb:
        d[pp.STATE]["cell_volumes"] = g.cell_volumes
        d[pp.STATE]["cell_centers"] = g.cell_centers

    save = pp.Exporter(gb, "sol", folder=folder)

    props = ["pressure", "cell_volumes", "cell_centers"]

    # extra properties, problem specific
    if all(gb.has_nodes_prop(gb.get_grids(), "low_zones")):
        gb.add_node_props("bottom_domain")
        for g, d in gb:
            d["bottom_domain"] = 1 - d["low_zones"]
        props.append("bottom_domain")

    has_key = True
    for _, d in gb:
        if "color" in d[pp.STATE]:
            has_key = has_key and True
        else:
            has_key = has_key and False

    if has_key:
        props.append("color")

    has_key = True
    for _, d in gb:
        if "aperture" in d[pp.STATE]:
            has_key = has_key and True
        else:
            has_key = has_key and False

    if has_key:
        props.append("aperture")


    save.write_vtk(props)


# ------------------------------------------------------------------------------#


def solve_rt0(gb, folder):
    # Choose and define the solvers and coupler
    flow_discretization = pp.RT0("flow")
    source_discretization = pp.DualScalarSource("flow")
    run_flow(gb, flow_discretization, source_discretization, folder, is_FV=False)


# ------------------------------------------------------------------------------#


def solve_tpfa(gb, folder):
    # Choose and define the solvers and coupler
    flow_discretization = pp.Tpfa("flow")
    source_discretization = pp.ScalarSource("flow")
    run_flow(gb, flow_discretization, source_discretization, folder, is_FV=True)


# ------------------------------------------------------------------------------#


def solve_mpfa(gb, folder):
    # Choose and define the solvers and coupler
    flow_discretization = pp.Mpfa("flow")
    source_discretization = pp.ScalarSource("flow")
    run_flow(gb, flow_discretization, source_discretization, folder, is_FV=True)


# ------------------------------------------------------------------------------#


def solve_vem(gb, folder):
    # Choose and define the solvers and coupler
    flow_discretization = pp.MVEM("flow")
    source_discretization = pp.DualScalarSource("flow")
    run_flow(gb, flow_discretization, source_discretization, folder, is_FV=False)


# ------------------------------------------------------------------------------#


def transport(gb, data, solver_name, folder, callback=None, save_every=1):
    grid_variable = "tracer"
    mortar_variable = "mortar_tracer"
    kw = "transport"
    # Identifier of the discretization operator on each grid
    advection_term = "advection"
    source_term = "source"
    mass_term = "mass"

    # Identifier of the discretization operator between grids
    advection_coupling_term = "advection_coupling"

    # Discretization objects
    node_discretization = pp.Upwind(kw)
    source_discretization = pp.ScalarSource(kw)
    mass_discretization = pp.MassMatrix(kw)
    edge_discretization = pp.UpwindCoupling(kw)

    # Loop over the nodes in the GridBucket, define primary variables and discretization schemes
    for g, d in gb:
        # Assign primary variables on this grid. It has one degree of freedom per cell.
        d[pp.PRIMARY_VARIABLES] = {grid_variable: {"cells": 1, "faces": 0}}
        # Assign discretization operator for the variable.
        d[pp.DISCRETIZATION] = {
            grid_variable: {
                advection_term: node_discretization,
                source_term: source_discretization,
                mass_term: mass_discretization,
            }
        }

    # Loop over the edges in the GridBucket, define primary variables and discretizations
    for e, d in gb.edges():
        g1, g2 = gb.nodes_of_edge(e)
        # The mortar variable has one degree of freedom per cell in the mortar grid
        d[pp.PRIMARY_VARIABLES] = {mortar_variable: {"cells": 1}}

        d[pp.COUPLING_DISCRETIZATION] = {
            advection_coupling_term: {
                g1: (grid_variable, advection_term),
                g2: (grid_variable, advection_term),
                e: (mortar_variable, edge_discretization),
            }
        }
        d[pp.DISCRETIZATION_MATRICES] = {kw: {}}

    assembler = pp.Assembler(gb, active_variables=[grid_variable, mortar_variable])

    # Assemble the linear system, using the information stored in the GridBucket. By
    # not adding the matrices, we can arrange them at will to obtain the efficient
    # solver defined below, which LU factorizes the system only once for all time steps.
    A, b, block_dof, full_dof = assembler.assemble_matrix_rhs(add_matrices=False)

    advection_coupling_term += (
        "_" + mortar_variable + "_" + grid_variable + "_" + grid_variable
    )
    mass_term += "_" + grid_variable
    advection_term += "_" + grid_variable
    source_term += "_" + grid_variable

    lhs = A[mass_term] + data["time_step"] * (
        A[advection_term] + A[advection_coupling_term]
    )
    rhs_source_adv = b[source_term] + data["time_step"] * (
        b[advection_term] + b[advection_coupling_term]
    )

    IEsolver = sps.linalg.factorized(lhs.tocsc())
    n_steps = int(np.round(data["t_max"] / data["time_step"]))

    # Initial condition
    tracer = np.zeros(rhs_source_adv.size)
    assembler.distribute_variable(tracer)

    # Exporter
    exporter = pp.Exporter(gb, name="tracer", folder=folder)
    export_fields = ["tracer"]

    # Keep track of the outflow for each time step
    outflow = np.empty(0)

    # Time loop
    for i in range(n_steps):
        # Export existing solution (final export is taken care of below)
        assembler.distribute_variable(tracer)
        if np.isclose(i % save_every, 0):
            exporter.write_vtk(export_fields, time_step=int(i // save_every))
        tracer = IEsolver(A[mass_term] * tracer + rhs_source_adv)

        outflow = compute_flow_rate(gb, grid_variable, outflow)
        if callback is not None:
            callback(gb)

    exporter.write_vtk(export_fields, time_step=(n_steps // save_every))
    time_steps = np.arange(
        0, data["t_max"] + data["time_step"], save_every * data["time_step"]
    )
    exporter.write_pvd(time_steps)
    return tracer, outflow, A, b, block_dof, full_dof


def compute_flow_rate(gb, grid_variable, outflow):
    # this function is only for the first benchmark case
    for g, d in gb:
        if g.dim < 3:
            continue
        faces, cells, sign = sps.find(g.cell_faces)
        index = np.argsort(cells)
        faces, sign = faces[index], sign[index]

        discharge = d[pp.PARAMETERS]["transport"]["darcy_flux"].copy()
        tracer = d[pp.STATE][grid_variable].copy()
        discharge[faces] *= sign
        discharge[g.get_internal_faces()] = 0
        discharge[discharge < 0] = 0
        val = np.dot(discharge, np.abs(g.cell_faces) * tracer)
        outflow = np.r_[outflow, val]
        return outflow
