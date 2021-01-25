import logging
from typing import Dict, List

import numpy as np

import porepy as pp

module_sections = ["utils"]
logger = logging.getLogger(__name__)


@pp.time_logger(sections=module_sections)
def grid_error(
    gb: "pp.GridBucket",
    gb_ref: "pp.GridBucket",
    variable: List[str],
    variable_dof: List[int],
) -> dict:
    """Compute grid errors a grid bucket and refined reference grid bucket

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
    assert len(variable) == len(variable_dof), (
        "Each variable must have associated " "with it a number of degrees of freedom."
    )
    n_variables = len(variable)

    errors: Dict = {}

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
                logger.info(
                    f"{var} not present on grid number "
                    f"{node_number} of dim {g.dim}."
                )
                continue

            # Compute errors relative to the reference grid
            # TODO: Should the solution be divided by g.cell_volumes or similar?
            # TODO: If scaling is used, consider that - or use the export-ready variables,
            #   'u_exp', 'p_exp', etc.
            sol = (
                states[var].reshape((var_dof, -1), order="F").T
            )  # (num_cells x var_dof)
            mapped_sol: np.ndarray = mapping.dot(sol)  # (num_cells x variable_dof)
            sol_ref = (
                states_ref[var].reshape((var_dof, -1), order="F").T
            )  # (num_cells x var_dof)

            # axis=0 gives component-wise norm.
            absolute_error = np.linalg.norm(mapped_sol - sol_ref, axis=0)

            norm_ref = np.linalg.norm(sol_ref)
            if np.any(norm_ref < 1e-10):
                logger.info(
                    f"Relative error not reportable. "
                    f"Norm of reference solution is {norm_ref}. "
                    f"Reporting absolute error"
                )
                error = absolute_error
                is_relative = False
            else:
                error = absolute_error / norm_ref
                is_relative = True

            errors[node_number][var] = {
                "error": error,
                "is_relative": is_relative,
            }

    return errors


@pp.time_logger(sections=module_sections)
def interpolate(g, fun):
    """
    Interpolate a scalar or vector function on the cell centers of the grid.

    Parameters
    ----------
    g : grid
        Grid, or a subclass, with geometry fields computed.
    fun : function
        Scalar or vector function.

    Return
    ------
    out: np.ndarray (dim of fun, g.num_cells)
        Function interpolated in the cell centers.

    Examples
    --------
    @pp.time_logger(sections=module_sections)
    def fun_p(pt): return np.sin(2*np.pi*pt[0])*np.sin(2*np.pi*pt[1])
    @pp.time_logger(sections=module_sections)
    def fun_u(pt): return [\
                      -2*np.pi*np.cos(2*np.pi*pt[0])*np.sin(2*np.pi*pt[1]),
                      -2*np.pi*np.sin(2*np.pi*pt[0])*np.cos(2*np.pi*pt[1])]
    p_ex = interpolate(g, fun_p)
    u_ex = interpolate(g, fun_u)

    """

    return np.array([fun(pt) for pt in g.cell_centers.T]).T


# ------------------------------------------------------------------------------#


@pp.time_logger(sections=module_sections)
def norm_L2(g, val):
    """
    Compute the L2 norm of a scalar or vector field.

    Parameters
    ----------
    g : grid
        Grid, or a subclass, with geometry fields computed.
    val : np.ndarray (dim of val, g.num_cells)
        Scalar or vector field.

    Return
    ------
    out: double
        The L2 norm of the input field.

    Examples
    --------
    @pp.time_logger(sections=module_sections)
    def fun_p(pt): return np.sin(2*np.pi*pt[0])*np.sin(2*np.pi*pt[1])
    p_ex = interpolate(g, fun_p)
    norm_ex = norm_L2(g, p_ex)

    """

    val = np.asarray(val)
    norm_sq = lambda v: np.sum(np.multiply(np.square(v), g.cell_volumes))
    if val.ndim == 1:
        return np.sqrt(norm_sq(val))
    return np.sqrt(np.sum([norm_sq(v) for v in val]))


# ------------------------------------------------------------------------------#


@pp.time_logger(sections=module_sections)
def error_L2(g, val, val_ex, relative=True):
    """
    Compute the L2 error of a scalar or vector field with respect to a reference
    field. It is possible to compute the relative error (default) or the
    absolute error.

    Parameters
    ----------
    g : grid
        Grid, or a subclass, with geometry fields computed.
    val : np.ndarray (dim of val, g.num_cells)
        Scalar or vector field.
    val_ex : np.ndarray (dim of val, g.num_cells)
        Reference scalar or vector field, i.e. the exact solution
    relative: bool (True default)
        Compute the relative error (if True) or the absolute error (if False).

    Return
    ------
    out: double
        The L2 error of the input fields.

    Examples
    --------
    p = ...
    @pp.time_logger(sections=module_sections)
    def fun_p(pt): return np.sin(2*np.pi*pt[0])*np.sin(2*np.pi*pt[1])
    p_ex = interpolate(g, fun_p)
    err_p = err_L2(g, p, p_ex)

    """

    val, val_ex = np.asarray(val), np.asarray(val_ex)
    err = norm_L2(g, np.subtract(val, val_ex))
    den = norm_L2(g, val_ex) if relative else 1
    assert den != 0
    return err / den


# ------------------------------------------------------------------------------#
