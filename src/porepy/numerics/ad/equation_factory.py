from typing import Tuple, List, Optional, Dict

from .equation_manager import EquationManager
from .operators import MergedVariable

import porepy as pp
import numpy as np

__all__ = ["flow"]


def _grid_edge_list(
    gb: pp.GridBucket,
) -> Tuple[List[pp.Grid], List[Tuple[pp.Grid, pp.Grid]]]:

    grid_list = [g for g, _ in gb.nodes()]
    edge_list = [e for e, _ in gb.edges()]

    return grid_list, edge_list


def _variables(
    manager: EquationManager,
    grid_variables: Optional[List[str]] = None,
    edge_variables: Optional[List[str]] = None,
    grid_list: Optional[List[pp.Grid]] = None,
    edge_list=None,
) -> Tuple[MergedVariable, MergedVariable]:
    if grid_list is None:
        grid_list = [g for g, _ in manager.gb.nodes()]

    if edge_list is None:
        edge_list = [e for e, _ in manager.gb.edges()]

    if grid_variables is not None:
        grid_vars = [
            pp.ad.MergedVariable([manager._variables[g][var] for g in grid_list])
            for var in grid_variables
        ]
    else:
        grid_vars = []

    if edge_variables is not None:
        edge_vars = [
            pp.ad.MergedVariable([manager._variables[e][var] for e in edge_list])
            for var in edge_variables
        ]
    else:
        edge_vars = []

    return grid_vars, edge_vars


def flow(
    manager,
    projections: Optional = None,
    discr=None,
    keyword="flow",
    pressure_variable="pressure",
    mortar_variable="mortar_flux",
):

    if discr is None:
        discr = pp.Mpfa(keyword)

    if projections is None:
        projections = pp.ad.MortarProjections(manager.gb)

    grid_list, edge_list = _grid_edge_list(manager.gb)

    discr_map = {g: discr for g in grid_list}
    node_discr = pp.ad.Discretization(discr_map, keyword)

    coupling = pp.RobinCoupling(keyword, discr, discr)
    edge_map = {e: coupling for e in edge_list}
    edge_discr = pp.ad.Discretization(edge_map, keyword)

    bc_val = pp.ad.BoundaryCondition(keyword, grid_list)

    div = pp.ad.Divergence(grids=grid_list)

    grid_vars, edge_vars = _variables(manager, [pressure_variable], [mortar_variable])
    p = grid_vars[0]
    lmbda = edge_vars[0]

    flux = (
        node_discr.flux * p
        + node_discr.bound_flux * bc_val
        + node_discr.bound_flux * projections.mortar_to_primary_int * lmbda
    )
    flow_eq = div * flux - projections.mortar_to_secondary_int * lmbda

    interface_flux = edge_discr.mortar_scaling * (
        projections.primary_to_mortar_avg * node_discr.bound_pressure_cell * p
        + projections.primary_to_mortar_avg
        * node_discr.bound_pressure_face
        * projections.mortar_to_primary_int
        * lmbda
        - projections.secondary_to_mortar_avg * p
        + edge_discr.mortar_discr * lmbda
    )

    return flow_eq, interface_flux, flux, projections


def poro_mechanics(
    manager,
    g: pp.Grid,
    data: Dict,
    mechanics_keyword="mechanics",
    flow_keyword: str = "flow",
    displacement_variable: str = "displacement",
    pressure_variable: str = "pressure",
):

    # MISSING PARTS:
    # 1) biot_alpha in div_u
    #
    # Terms having to do with pressure (both accumulation and diffusion)
    # must be added by other equations
    #

    # This is the one who will do the discretization
    # Should be added to data dictionary somehow
    mech_discr = pp.Biot(
        mechanics_keyword,
        flow_keyword,
        vector_variable=displacement_variable,
        scalar_variable=pressure_variable,
    )

    discr = pp.ad.Discretization({g: mech_discr})

    div_scalar = pp.ad.Divergence([g])
    div_vector = pp.ad.Divergence([g], is_scalar=False)

    mpsa = pp.ad.Discretization({g: pp.Mpsa(mechanics_keyword)})
    gradp = pp.ad.Discretization({g: pp.GradP(mechanics_keyword)})
    div_u = pp.ad.Discretization({g: pp.DivU(flow_keyword)}, mat_dict_key="flow")
    stabilization = pp.ad.Discretization({g: pp.BiotStabilization(flow_keyword)})

    mpfa = pp.Mpfa(flow_keyword)
    mass = pp.MassMatrix(flow_keyword)

    mpfa_discr = pp.ad.Discretization({g: mpfa})
    mass_discr = pp.ad.Discretization({g: mass})

    u = pp.ad.MergedVariable([manager._variables[g][displacement_variable]])
    p = pp.ad.MergedVariable([manager._variables[g][pressure_variable]])

    bc_flow = pp.ad.BoundaryCondition(flow_keyword, [g])
    bc_mech = pp.ad.BoundaryCondition(mechanics_keyword, [g])

    stress = mpsa.stress * u + gradp.grad_p * p + mpsa.bound_stress * bc_mech

    momentuum = div_vector * stress

    # TODO: Need biot-alpha here
    volume_term = div_u.div_u * u + stabilization.stabilization * p

    compr = mass_discr.mass
    diffusion = div_scalar * (mpfa_discr.flux * p + mpfa_discr.bound_flux * bc_flow)

    # rhs terms, must be scaled with time step
    div_u_rhs = div_u.div_u
    stab_rhs = stabilization.stabilization

    return momentuum, volume_term, compr, diffusion, stress, div_u_rhs, stab_rhs, p
