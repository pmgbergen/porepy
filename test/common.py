""" Functionailty that is common for several tests, and therfore stored in a
common module.
"""
import numpy as np

import porepy as pp


def setup_flow_assembler(gb, method, data_key=None, coupler=None):
    """ Setup a standard assembler for the flow problem for a given grid bucket.

    The assembler will be set up with primary variable name 'pressure' on the
    GridBucket nodes, and mortar_flux for the mortar variables.

    Parameters:
        gb: GridBucket.
        method (EllipticDiscretization).
        data_key (str, optional): Keyword used to identify data dictionary for
            node and edge discretization.
        Coupler (EllipticInterfaceLaw): Defaults to RobinCoulping.

    Returns:
        Assembler, ready to discretize and assemble problem.

    """

    if data_key is None:
        data_key = "flow"
    if coupler is None:
        coupler = pp.RobinCoupling(data_key, method)

    if isinstance(method, pp.MVEM) or isinstance(method, pp.RT0):
        mixed_form = True
    else:
        mixed_form = False

    for g, d in gb:
        if mixed_form:
            d[pp.PRIMARY_VARIABLES] = {"pressure": {"cells": 1, "faces": 1}}
        else:
            d[pp.PRIMARY_VARIABLES] = {"pressure": {"cells": 1}}
        d[pp.DISCRETIZATION] = {"pressure": {"diffusive": method}}
    for e, d in gb.edges():
        g1, g2 = gb.nodes_of_edge(e)
        d[pp.PRIMARY_VARIABLES] = {"mortar_flux": {"cells": 1}}
        d[pp.COUPLING_DISCRETIZATION] = {
            "lambda": {
                g1: ("pressure", "diffusive"),
                g2: ("pressure", "diffusive"),
                e: ("mortar_flux", coupler),
            }
        }
        d[pp.DISCRETIZATION_MATRICES] = {"flow": {}}

    assembler = pp.Assembler()
    return assembler


def solve_and_distribute_pressure(gb, assembler):
    """ Given an assembler, assemble and solve the pressure equation, and distribute
    the result.

    Parameters:
        GridBucket: Of problem to be solved
        assembler (Assembler):
    """
    A, b, block_dof, full_dof = assembler.assemble_matrix_rhs(gb)
    p = np.linalg.solve(A.A, b)
    assembler.distribute_variable(gb, p, block_dof, full_dof)
