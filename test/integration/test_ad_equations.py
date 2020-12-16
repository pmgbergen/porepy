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

    pressure_variable = 'pressure'
    flux_variable = 'mortar_flux'

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

        d[pp.PRIMARY_VARIABLES] = {pressure_variable: {"cells": 1}}
        d[pp.DISCRETIZATION] = {pressure_variable: {"diff": discr}}
    for e, d in gb.edges():
        pp.initialize_data(d["mortar_grid"], d, "flow", {"normal_diffusivity": 1})

        d[pp.PRIMARY_VARIABLES] = {flux_variable: {"cells": 1}}
        d[pp.COUPLING_DISCRETIZATION] = {}
        d[pp.COUPLING_DISCRETIZATION]["coupling"] = {
            e[0]: (pressure_variable, "diff"),
            e[1]: (pressure_variable, "diff"),
            e: (flux_variable, coupling_discr),
        }

    dof_manager = pp.DofManager(gb)
    assembler = pp.Assembler(gb, dof_manager)
    assembler.discretize()

    # Reference discretization
    A_ref, b_ref = assembler.assemble_matrix_rhs()

    manager = pp.ad.EquationManager(gb, dof_manager)

    grid_list = [g for g, _ in gb]
    edge_list = [e for e, _ in gb.edges()]

    discr_map = {g: discr for g in grid_list}
    node_discr = pp.ad.Discretization(discr_map, keyword)

    coupling = pp.RobinCoupling(keyword, discr, discr)
    edge_map = {e: coupling for e in edge_list}
    edge_discr = pp.ad.Discretization(edge_map, keyword)

    bc_val = pp.ad.BoundaryCondition(keyword, grid_list)

    projections = pp.ad.MortarProjections(gb=gb)
    div = pp.ad.Divergence(grids=grid_list)

    p = manager.merge_variables([(g, pressure_variable) for g in grid_list])
    lmbda = manager.merge_variables([(e, flux_variable) for e in edge_list])

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

    flow_eq_ad = pp.ad.Equation(flow_eq, dof_manager)
    interface_eq_ad = pp.ad.Equation(interface_flux, dof_manager)

    manager.equations += [flow_eq_ad, interface_eq_ad]

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

    displacement_variable = "displacement"
    pressure_variable = "pressure"

    mechanics_keyword = "mechanics"
    flow_keyword ="flow"

    d = gb.node_props(g)

    bc_flow = pp.BoundaryCondition(g, faces=np.array([0]), cond=["dir"])
    bc_val_flow = np.ones(g.num_faces)

    bc_mech = pp.BoundaryConditionVectorial(g, faces=np.array([0]), cond=["dir"])
    bc_val_mech = np.ones(g.num_faces * g.dim)

    mech_param = {"biot_alpha": 1, "bc": bc_mech, "bc_values": bc_val_mech, 'inverter': 'python'}
    flow_param = {"biot_alpha": 1, "bc": bc_flow, "bc_values": bc_val_flow, 'inverter': 'python'}
    pp.initialize_default_data(g, d, mechanics_keyword, mech_param)
    pp.initialize_default_data(g, d, flow_keyword, flow_param)

    d[pp.PRIMARY_VARIABLES] = {
        displacement_variable: {"cells": g.dim},
        pressure_variable : {"cells": 1},
    }

    dof_manager = pp.DofManager(gb)

    biot = pp.Biot()

    state = np.zeros(g.num_cells * (g.dim + 1))
    dt = np.random.rand()

    d[pp.PARAMETERS][flow_keyword]["time_step"] = dt

    d[pp.STATE] = {}
    d[pp.STATE][displacement_variable] = state[: g.num_cells * g.dim]
    d[pp.STATE][pressure_variable] = state[g.num_cells * g.dim :]
    # Presumed truth which we will compare with
    biot.discretize(g, d)
    A_biot, b_biot = biot.assemble_matrix_rhs(g, d)

    manager = pp.ad.EquationManager(gb, dof_manager)

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

    u = manager.variables[g][displacement_variable]
    p = manager.variables[g][pressure_variable]

    bc_flow = pp.ad.BoundaryCondition(flow_keyword, [g])
    bc_mech = pp.ad.BoundaryCondition(mechanics_keyword, [g])

    stress = mpsa.stress * u + gradp.grad_p * p + mpsa.bound_stress * bc_mech

    momentuum = div_vector * stress

    # TODO: Need biot-alpha here
    accumulation = div_u.div_u * u + stabilization.stabilization * p

    compr = mass_discr.mass
    diffusion = div_scalar * (mpfa_discr.flux * p + mpfa_discr.bound_flux * bc_flow)

    # rhs terms, must be scaled with time step
    div_u_rhs = div_u.div_u
    stab_rhs = stabilization.stabilization
    manager.equations.append(
        pp.ad.Equation(momentuum, dof_manager=dof_manager, name="Momentuum conservation")
    )

    flow_momentuum_eq = accumulation + compr * p + dt * diffusion

    u_state, p_state = manager.variable_state(
        [(g, displacement_variable), (g, pressure_variable)], state
    )

    flow_rhs = div_u_rhs * pp.ad.Array(u_state) + stab_rhs * pp.ad.Array(p_state)

    flow_eq = pp.ad.Equation(
        flow_momentuum_eq + flow_rhs, dof_manager=dof_manager, name="flow eqution"
    )
    manager.equations.append(flow_eq)

    A, b = manager.assemble_matrix_rhs(state)

    dm = A_biot - A
    if dm.data.size > 0:
        assert np.max(np.abs(dm.data)) < 1e-10
    assert np.max(np.abs(b - b_biot)) < 1e-10


def _single_fracture_2d():
    frac = np.array([[1, 2], [1, 1]])
    gb = pp.meshing.cart_grid(frac, [3, 2])
    pp.contact_conditions.set_projections(gb)
    return gb


class ContactModel(pp.ContactMechanics):

    def __init__(self, param):
        super().__init__(param)

    def _bc_values(self, g):
        return g.face_centers[:g.dim].ravel('f')


def test_contact_mechanics(grid_method):

    model = ContactModel({})
    setattr(model, 'create_grid', grid_method)


