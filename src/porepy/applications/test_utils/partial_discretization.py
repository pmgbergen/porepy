"""Utility functions for partial discretization updates."""

import numpy as np

import porepy as pp


def perform_partial_discretization_specified_nodes(
    g: pp.Grid, discr: pp.Mpfa | pp.Mpsa | pp.Biot, specified_data: dict, cell_id: int
):
    """Perform partial discretization update.

    This function is used to perform a partial discretization update, i.e., to
    discretize a single cell. The functionality is only available for finite volume
    discretizations, that is, MPFA, MPSA, and Biot.

    TODO: We may want to allow for update of multiple cells at once. Whether this is
    worth the effort depends on the use case, which currently is unknown.

    Parameters:
        g: The grid.
        discr: The discretization scheme.
        specified_data: The data dictionary with the discretization parameters.
        cell_id: The cell whose node will be used to define the partial update. The
            nodes are identified based on the provided cell_id in this function, i.e.,
            outside of the discretization method.

    Returns:
        data: The data dictionary containing discretization matrices.

    """
    ind = np.zeros(g.num_cells)
    ind[cell_id] = 1
    nodes = np.squeeze(np.where(g.cell_nodes() @ ind > 0))
    specified_data["specified_nodes"] = nodes
    data = pp.initialize_default_data(
        g, {}, discr.keyword, specified_parameters=specified_data
    )

    discr.discretize(g, data)
    return data
