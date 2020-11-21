""" Test the eqution factory which produces what is considered standard equations.

The tests are based on equivalence between the equations defined by the AD framework
and the classical way of defining equations.
"""
import porepy as pp
import numpy as np
import scipy.sparse.linalg as spla


def test_md_flow():

    # Three fractures, will create intersection lines and point
    frac_1 = np.array([[2, 4, 4, 2], [3, 3, 3, 3], [0, 0, 6, 6]])
    frac_2 = np.array([[3, 3, 3, 3], [2, 4, 4, 2], [0, 0, 6, 6]])
    frac_3 = np.array([[0, 6, 6, 0], [2, 2, 4, 4], [3, 3, 3, 3]])

    gb = pp.meshing.cart_grid(fracs=[frac_1, frac_2, frac_3], nx=np.array([6, 6, 6]))
    gb.compute_geometry()

    keyword = "flow"
    discr = pp.Tpfa(keyword)
    coupling_discr = pp.RobinCoupling(keyword, discr, discr)

    for g, d in gb:

        if g.dim == gb.dim_max():
            bc = pp.BoundaryCondition(g, np.array([0, g.num_faces - 1]), ["dir", "dir"])
            bc_values = np.zeros(g.num_faces)
            bc_values[0] = 1
            specified_parameters = {"bc": bc, "bc_values": bc_values}
        else:
            specified_parameters = {}
        pp.initialize_default_data(g, d, "flow", specified_parameters)

        d[pp.PRIMARY_VARIABLES] = {"pressure": {"cells": 1}}
        d[pp.DISCRETIZATION] = {"pressure": {"diff": discr}}
    for e, d in gb.edges():
        pp.initialize_data(d["mortar_grid"], d, "flow", {"normal_diffusivity": 1})

        d[pp.PRIMARY_VARIABLES] = {"mortar_flux": {"cells": 1}}
        d[pp.COUPLING_DISCRETIZATION] = {}
        d[pp.COUPLING_DISCRETIZATION]["coupling"] = {
            e[0]: ("pressure", "diff"),
            e[1]: ("pressure", "diff"),
            e: ("mortar_flux", coupling_discr),
        }

    assembler = pp.Assembler(gb)
    assembler.discretize()

    # Reference discretization
    A_ref, b_ref = assembler.assemble_matrix_rhs()

    manager = pp.EquationManager(gb)

    flow_eq, interface_flux, *rest = pp.ad.equation_factory.flow(
        manager,
    )

    manager._equations = [
        pp.Equation(flow_eq, assembler),
        pp.Equation(interface_flux, assembler),
    ]

    state = np.zeros(gb.num_cells() + gb.num_mortar_cells())
    A, b = manager.assemble_matrix_rhs(state)
    diff = A - A_ref
    if diff.data.size > 0:
        assert np.max(np.abs(diff.data)) < 1e-10
    assert np.max(np.abs(b - b_ref)) < 1e-10


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
    assembler = pp.Assembler(gb)

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

    manager._equations.append(
        pp.Equation(momentuum, dof_manager=assembler, name="Momentuum conservation")
    )

    flow_momentuum_eq = accumulation + compr * p + dt * diffusion

    u_state, p_state = manager.variable_state(
        [(g, "displacement"), (g, "pressure")], state
    )

    flow_rhs = div_u_rhs * pp.ad.Array(u_state) + stab_rhs * pp.ad.Array(p_state)

    flow_eq = pp.Equation(
        flow_momentuum_eq + flow_rhs, dof_manager=assembler, name="flow eqution"
    )
    manager._equations.append(flow_eq)

    A, b = manager.assemble_matrix_rhs(state)

    dm = A_biot - A
    if dm.data.size > 0:
        assert np.max(np.abs(dm.data)) < 1e-10
    assert np.max(np.abs(b - b_biot)) < 1e-10
