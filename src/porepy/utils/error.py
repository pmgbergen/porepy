import numpy as np
import porepy as pp
from typing import List, Optional, Union, Callable, Tuple

# ------------------------------------------------------------------------------#

class ErrorAnalyzer:
    
    def __init__(self, params: Dict, variables: List[str]) -> None:
        self.params = params
        
    def _l2_norm(self, gb: pp.GridBucket,
             variables: Dict[str, str],
             norm_type: Optional[int] = 2,
             target_grids: Optional[Union[pp.Grid, List[pp.Grid]]] = None,
             target_interfaces: Optional[List[Tuple[pp.Grid, pp.Grid]]] = None):
            
        nrm: Callable[[Union[pp.Grid, pp.MortarGrid], np.ndarray], float] = lambda g, val:  g.cell_volumes * np.linalg.norm(nrm, norm_type)
        
        nrm_vec
        
        variable_filter: Callable[[str], bool] = lambda x: x in target_variables

        if target_grids is None:
            grid_filter: Callable[[pp.Grid], bool] = lambda x: True
        else:
            grid_filter: Callable[[pp.Grid], bool] = lambda x: x in target_grids

        if target_interfaces is None:
            interface_filter: Callable[[Tuple[pp.Grid, pp.Grid]], bool] = lambda x: True
        else:
            interface_filter: Callable[[Tuple[pp.Grid, pp.Grid]], bool] = lambda x: x in target_grids
            
        values = {v: 0 for v in variables.keys()}
            
        for g, d in gb:
            if not grid_filter(g):
                continue
            for key, value = d[pp.STATE]:
                if not variable_filter(g):
                    values[(g, var)]
                    
                    
    def _l2_cell(self, val: np.ndarray, g: pp.Grid):
        if val.size == g.num_cells:
            return np.sqrt(val**2 * g.cell_volumes)
        elif val.size == g.num_cells * g.dim:
            v = val.reshape((g.dim, g.num_cells, order='f'))
            return np.sqrt(np.sum(v**2 * g.cell_volumes, axis=0))
        else:
            raise ValueError(f"Mismatch between variable of size {val.size} and grid of dimension {g.dim} with {g.num_cells} cells")            
    
                
                


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
    def fun_p(pt): return np.sin(2*np.pi*pt[0])*np.sin(2*np.pi*pt[1])
    def fun_u(pt): return [\
                      -2*np.pi*np.cos(2*np.pi*pt[0])*np.sin(2*np.pi*pt[1]),
                      -2*np.pi*np.sin(2*np.pi*pt[0])*np.cos(2*np.pi*pt[1])]
    p_ex = interpolate(g, fun_p)
    u_ex = interpolate(g, fun_u)

    """

    return np.array([fun(pt) for pt in g.cell_centers.T]).T


# ------------------------------------------------------------------------------#


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
