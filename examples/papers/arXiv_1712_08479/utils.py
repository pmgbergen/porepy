"""
Utility functions for the run scripts in current folder.
"""
import numpy as np
from porepy.numerics.fv.tpfa import TpfaMixedDim
from porepy.utils.error import error_L2
from porepy.numerics.mixed_dim import condensation as SC
from porepy.params.data import Parameters


has_splitter = TpfaMixedDim()


def compute_errors(gb, p, p_el, p_mrst):
    p_without = p[: p_el.size]
    porepy_errors = gb_error(gb, p_without, p_el)
    mrst_errors = gb_error(gb, p_without, p_mrst)
    porepy_global = global_error(gb, p_without, p_el)
    mrst_global = global_error(gb, p_without, p_mrst)
    return porepy_errors, mrst_errors, porepy_global, mrst_global


def global_error(gb, v1, v2):
    vols = []
    for g, d in gb:
        v = np.multiply(g.cell_volumes, d["param"].get_aperture())
        vols = np.append(vols, v)

    e = np.sqrt(np.sum(np.multiply(np.power(v1 - v2, 2), vols)))
    e = e / np.sqrt(np.sum(vols))
    return e


def gb_error(gb, v1, v2, norm="L2"):
    gb.add_node_props(["error_1", "error_2"])
    has_splitter.split(gb, "error_1", v1)
    has_splitter.split(gb, "error_2", v2)
    e = gb.apply_function_to_nodes(
        lambda g, d: error_L2(g, d["error_1"], d["error_2"], relative=False)
    )
    return e


def perform_condensation(full_problem, reduced_problem, dim):
    """
    Obtain reduced matrix and rhs.
    """
    A = full_problem.lhs
    rhs = full_problem.rhs
    to_be_eliminated = SC.dofs_of_dimension(full_problem.grid(), A, dim)
    a_reduced, rhs_reduced, _, _, _ = SC.eliminate_dofs(A, rhs, to_be_eliminated)

    reduced_problem.lhs = a_reduced
    reduced_problem.rhs = rhs_reduced


def edge_params(gb):
    gb.add_edge_prop("param")
    for e, d in gb.edges_props():
        g_h = gb.sorted_nodes_of_edge(e)[1]
        d["param"] = Parameters(g_h)


def assign_data(gb, data_class, data_key):
    """
    Loop over grids to assign flow problem data.
    Darcy data_key: problem
    Transport data_key: transport_data
    """
    gb.add_node_props([data_key])
    for g, d in gb:
        d[data_key] = data_class(g, d)
