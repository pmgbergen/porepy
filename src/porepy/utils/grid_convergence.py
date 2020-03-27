import os
import logging
from typing import (  # noqa
    Any,
    Coroutine,
    Generator,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import porepy as pp
import numpy as np
import scipy.sparse as sps

logger = logging.getLogger(__name__)


def grid_error(
        gb: pp.GridBucket,
        gb_ref: pp.GridBucket,
        variable: List[str],
        variable_dof: List[int],
) -> dict:
    """ Compute grid errors a grid bucket and refined reference grid bucket

    Assumes that the coarse grid bucket has a node property
    'coarse_fine_cell_mapping' assigned on each grid, which
    maps from coarse to fine cells according to the method
    'coarse_fine_cell_mapping(...)'.

    Parameters
    ----------
    gb, gb_ref : pp.GridBucket
        Coarse and fine grid buckets, respectively
    variable : List[str]
        which variables to compute error over
    variable_dof : List[int]
        Degrees of freedom for each variable in the list 'variable'.

    Returns
    -------
    errors : dict
        Dictionary with top level keys as node_number,
        within which for each variable, the error is
        reported.
    """

    if not isinstance(variable, list):
        variable = [variable]
    if not isinstance(variable_dof, list):
        variable_dof = [variable_dof]
    assert len(variable) == len(variable_dof), "Each variable must have associated " \
                                               "with it a number of degrees of freedom."
    n_variables = len(variable)

    errors = {}

    grids = gb.get_grids()
    grids_ref = gb_ref.get_grids()
    n_grids = len(grids)

    for i in np.arange(n_grids):
        g, g_ref = grids[i], grids_ref[i]
        mapping = gb.node_props(g, "coarse_fine_cell_mapping")

        # Get states
        data = gb.node_props(g)
        data_ref = gb_ref.node_props(g_ref)
        states = data[pp.STATE]
        states_ref = data_ref[pp.STATE]

        node_number = data["node_number"]

        # Initialize errors
        errors[node_number] = {}

        for var_idx in range(0, n_variables):
            var = variable[var_idx]
            var_dof = variable_dof[var_idx]

            # Check if the variable exists on both the grid and reference grid
            state_keys = set(states.keys())
            state_ref_keys = set(states_ref.keys())
            check_keys = state_keys.intersection(state_ref_keys)
            if var not in check_keys:
                logger.info(f"{var} not present on grid number "
                            f"{node_number} of dim {g.dim}.")
                continue

            # Compute errors relative to the reference grid
            # TODO: Should the solution be divided by g.cell_volumes or similar?
            # TODO: If scaling is used, consider that - or use the export-ready variables,
            #   'u_exp', 'p_exp', etc.
            sol = states[var].reshape((var_dof, -1), order='F').T  # (num_cells x var_dof)
            mapped_sol: np.ndarray = mapping.dot(sol)  # (num_cells x variable_dof)
            sol_ref = states_ref[var].reshape((var_dof, -1), order='F').T  # (num_cells x var_dof)

            # axis=0 gives component-wise norm.
            absolute_error = np.linalg.norm(mapped_sol - sol_ref, axis=0)
            norm_ref = np.linalg.norm(sol_ref, axis=0)

            if np.any(norm_ref < 1e-10):
                logger.info(f"Relative error not reportable. "
                            f"Norm of reference solution is {norm_ref}. "
                            f"Reporting absolute error")
                error = absolute_error
                is_relative = False
            else:
                error = absolute_error / norm_ref
                is_relative = True

            errors[node_number][var] = {
                "error":
                    error,
                "is_relative":
                    is_relative,
                }

    return errors



