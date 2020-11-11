from typing import Tuple, List, Optional

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
    grid_variables: Optional[List[str]],
    edge_variables: Optional[List[str]],
    grid_list: Optional[List[pp.Grid]] = None,
    edge_list=None,
) -> Tuple[MergedVariable, MergedVariable]:
    if grid_list is None:
        grid_list = [g for g, _ in manager.gb.nodes()]

    if edge_list is None:
        edge_list = [e for e, _ in manager.gb.edges()]

    grid_vars = [
        pp.ad.MergedVariable(
            [manager._variables[g][var] for var in grid_variables for g in grid_list]
        )
    ]

    edge_vars = [
        pp.ad.MergedVariable(
            [manager._variables[e][var] for var in edge_variables for e in edge_list]
        )
    ]

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
