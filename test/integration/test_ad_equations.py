""" Test the eqution factory which produces what is considered standard equations.

The tests are based on equivalence between the equations defined by the AD framework
and the classical way of defining equations.
"""
import porepy as pp
import numpy as np
import scipy.sparse.linalg as spla


def test_biot():
    """Define a Biot discretization on a 3d Cartesian grid.

    Reference discretization by pp.Biot.discretize()

    The test checks that the discretization matrix and rhs produced by AD and
    reference are identical.
    """
    g = pp.CartGrid([2, 3, 2])
    g.compute_geometry()
    gb = pp.GridBucket()
    gb.compute_geometry()
    gb.add_nodes(g)

    d = gb.node_props(g)

    bc_flow = pp.BoundaryCondition(g, faces=np.array([0]), cond=["dir"])
    bc_val_flow = np.ones(g.num_faces)

    bc_mech = pp.BoundaryConditionVectorial(g, faces=np.array([0]), cond=["dir"])
    bc_val_mech = np.ones(g.num_faces * g.dim)

    mech_param = {"biot_alpha": 1, "bc": bc_mech, "bc_values": bc_val_mech}
    flow_param = {"biot_alpha": 1, "bc": bc_flow, "bc_values": bc_val_flow}
    pp.initialize_default_data(g, d, "mechanics", mech_param)
    pp.initialize_default_data(g, d, "flow", flow_param)

    d[pp.PRIMARY_VARIABLES] = {
        "displacement": {"cells": g.dim},
        "pressure": {"cells": 1},
    }

    biot = pp.Biot()

    state = np.zeros(g.num_cells * (g.dim + 1))
    dt = np.random.rand()

    d[pp.PARAMETERS]["flow"]["time_step"] = dt

    d[pp.STATE] = {}
    d[pp.STATE]["displacement"] = state[: g.num_cells * g.dim]
    d[pp.STATE]["pressure"] = state[g.num_cells * g.dim :]
    # Presumed truth which we will compare with
    biot.discretize(g, d)
    A_biot, b_biot = biot.assemble_matrix_rhs(g, d)

    manager = pp.EquationManager(gb)

    (
        momentuum,
        accumulation,
        compr,
        diffusion,
        _,
        div_u_rhs,
        stab_rhs,
        p,
    ) = pp.ad.equation_factory.poro_mechanics(manager, g, d)

    manager._equations.append(pp.Equation(momentuum, name="Momentuum conservation"))

    flow_momentuum_eq = accumulation + compr * p + pp.ad.Scalar(dt) * diffusion

    u_state, p_state = manager.variable_state(
        [(g, "displacement"), (g, "pressure")], state
    )

    flow_rhs = div_u_rhs * pp.ad.Array(u_state) + stab_rhs * pp.ad.Array(p_state)

    flow_eq = pp.Equation(flow_momentuum_eq + flow_rhs, name="flow eqution")
    manager._equations.append(flow_eq)

    A, b = manager.assemble_matrix_rhs(state)

    dm = A_biot - A
    assert np.max(np.abs(dm.data)) < 1e-10
    # Need a + sign here since the AD residual is computed as a LHS term
    assert np.max(np.abs(b + b_biot)) < 1e-10
