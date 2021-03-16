""" Test the equation factory which produces what is considered standard equations.

The tests are based on equivalence between the equations defined by the AD framework
and the classical way of defining equations.
"""
import pytest
import porepy as pp
import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as sps

from test import test_utils


def test_md_flow():

    # Three fractures, will create intersection lines and point
    frac_1 = np.array([[2, 4, 4, 2], [3, 3, 3, 3], [0, 0, 6, 6]])
    frac_2 = np.array([[3, 3, 3, 3], [2, 4, 4, 2], [0, 0, 6, 6]])
    frac_3 = np.array([[0, 6, 6, 0], [2, 2, 4, 4], [3, 3, 3, 3]])

    gb = pp.meshing.cart_grid(fracs=[frac_1, frac_2, frac_3], nx=np.array([6, 6, 6]))
    gb.compute_geometry()

    pressure_variable = "pressure"
    flux_variable = "mortar_flux"

    keyword = "flow"
    discr = pp.Tpfa(keyword)
    source_discr = pp.ScalarSource(keyword)
    coupling_discr = pp.RobinCoupling(keyword, discr, discr)

    for g, d in gb:

        # Assign data
        if g.dim == gb.dim_max():
            upper_left_ind = np.argmax(np.linalg.norm(g.face_centers, axis=0))
            bc = pp.BoundaryCondition(g, np.array([0, upper_left_ind]), ["dir", "dir"])
            bc_values = np.zeros(g.num_faces)
            bc_values[0] = 1
            sources = np.random.rand(g.num_cells) * g.cell_volumes
            specified_parameters = {
                "bc": bc, 
                "bc_values": bc_values,
                "source": sources
                }
        else:
            sources = np.random.rand(g.num_cells) * g.cell_volumes
            specified_parameters = {"source": sources}
        
        # Initialize data
        pp.initialize_default_data(g, d, keyword, specified_parameters)
        
        # Declare grid primary variable
        d[pp.PRIMARY_VARIABLES] = {pressure_variable: {"cells": 1}}
        
        # Assign discretization
        d[pp.DISCRETIZATION] = {
            pressure_variable: {"diff": discr, "source": source_discr}
            }
        
        # Initialize state
        d[pp.STATE] = {
            pressure_variable: np.zeros(g.num_cells),
            pp.ITERATE: {pressure_variable: np.zeros(g.num_cells)},
        }
    for e, d in gb.edges():
        mg = d["mortar_grid"]
        pp.initialize_data(mg, d, keyword, {"normal_diffusivity": 1})

        d[pp.PRIMARY_VARIABLES] = {flux_variable: {"cells": 1}}
        d[pp.COUPLING_DISCRETIZATION] = {}
        d[pp.COUPLING_DISCRETIZATION]["coupling"] = {
            e[0]: (pressure_variable, "diff"),
            e[1]: (pressure_variable, "diff"),
            e: (flux_variable, coupling_discr),
        }
        d[pp.STATE] = {
            flux_variable: np.zeros(mg.num_cells),
            pp.ITERATE: {flux_variable: np.zeros(mg.num_cells)},
        }

    dof_manager = pp.DofManager(gb)
    assembler = pp.Assembler(gb, dof_manager)
    assembler.discretize()

    # Reference discretization
    A_ref, b_ref = assembler.assemble_matrix_rhs()

    manager = pp.ad.EquationManager(gb, dof_manager)

    grid_list = [g for g, _ in gb]
    edge_list = [e for e, _ in gb.edges()]

    node_discr = pp.ad.MpfaAd(keyword, grid_list)

    edge_discr = pp.ad.RobinCouplingAd(keyword, edge_list)

    bc_val = pp.ad.BoundaryCondition(keyword, grid_list)
    
    source = pp.ad.ParameterArray(param_keyword=keyword, array_keyword='source', grids=grid_list)

    projections = pp.ad.MortarProjections(gb=gb)
    div = pp.ad.Divergence(grids=grid_list)

    p = manager.merge_variables([(g, pressure_variable) for g in grid_list])
    lmbda = manager.merge_variables([(e, flux_variable) for e in edge_list])

    flux = (
        node_discr.flux * p
        + node_discr.bound_flux * bc_val
        + node_discr.bound_flux * projections.mortar_to_primary_int * lmbda
    )
    flow_eq = div * flux - projections.mortar_to_secondary_int * lmbda - source

    interface_flux = edge_discr.mortar_scaling * (
        projections.primary_to_mortar_avg * node_discr.bound_pressure_cell * p
        + projections.primary_to_mortar_avg
        * node_discr.bound_pressure_face
        * projections.mortar_to_primary_int
        * lmbda
        - projections.secondary_to_mortar_avg * p
        + edge_discr.mortar_discr * lmbda
    )

    flow_eq_ad = pp.ad.Expression(flow_eq, dof_manager, "flow on nodes")
    flow_eq_ad.discretize(gb)

    interface_eq_ad = pp.ad.Expression(interface_flux, dof_manager, "flow on interface")

    manager.equations += [flow_eq_ad, interface_eq_ad]

    state = np.zeros(gb.num_cells() + gb.num_mortar_cells())
    A, b = manager.assemble_matrix_rhs(state=state)
    diff = A - A_ref
    if diff.data.size > 0:
        assert np.max(np.abs(diff.data)) < 1e-10
    assert np.max(np.abs(b - b_ref)) < 1e-10

# Below are grid buckte generators to be used for tests of contact-mechanics models


def _single_fracture_2d():
    frac = [np.array([[2, 3], [2, 2]])]
    gb = pp.meshing.cart_grid(frac, [5, 4])
    pp.contact_conditions.set_projections(gb)
    return gb


def _single_fracture_near_global_boundary_2d():
    frac = [np.array([[1, 2], [1, 1]])]
    gb = pp.meshing.cart_grid(frac, [3, 2])
    pp.contact_conditions.set_projections(gb)
    return gb


def _two_fractures_2d():
    frac = [np.array([[2, 3], [1, 1]]), np.array([[2, 3], [3, 3]])]
    gb = pp.meshing.cart_grid(frac, [5, 4])
    pp.contact_conditions.set_projections(gb)
    return gb


def _two_fractures_crossing_2d():
    frac = [np.array([[1, 3], [2, 2]]), np.array([[2, 2], [1, 3]])]
    gb = pp.meshing.cart_grid(frac, [4, 4])
    pp.contact_conditions.set_projections(gb)
    return gb


def _single_fracture_3d():
    frac = [np.array([[2, 3, 3, 2], [2, 2, 3, 3], [2, 2, 2, 2]])]
    gb = pp.meshing.cart_grid(frac, [5, 5, 4])
    pp.contact_conditions.set_projections(gb)
    return gb


def _three_fractures_crossing_3d():
    frac = [
        np.array([[1, 3, 3, 1], [1, 1, 3, 3], [2, 2, 2, 2]]),
        np.array([[1, 1, 3, 3], [2, 2, 2, 2], [1, 3, 3, 1]]),
        np.array([[2, 2, 2, 2], [1, 1, 3, 3], [1, 3, 3, 1]]),
    ]
    gb = pp.meshing.cart_grid(frac, [4, 4, 4])
    pp.contact_conditions.set_projections(gb)
    return gb


class ContactModel(pp.ContactMechanics):
    def __init__(self, param, grid_meth):
        super().__init__(param)
        self._grid_meth = grid_meth

        self.eq_order = ["u", "contact_traction", "mortar_u"]

    def _bc_values(self, g):
        return g.face_centers[: g.dim].ravel("f")

    def create_grid(self):
        np.random.seed(0)
        self.gb = self._grid_meth()


#        pp.contact_conditions.set_projections(self.gb)


class BiotContactModel(pp.ContactMechanicsBiot):
    def __init__(self, param, grid_meth):
        ############# NB: tests should be with and without subtract_fracture_pressure
        ## Time steps should be an input parameters, try a few
        # Make initial states random?
        # Biot alpha should vary
        super().__init__(param)
        self.time_step = 0.5
        self.end_time = 1
        self._grid_meth = grid_meth

        self.eq_order = ["u", "contact_traction", "mortar_u", "p", "mortar_p"]

    def _bc_values_scalar(self, g):
        return g.face_centers[: g.dim].sum(axis=0)

    def _bc_values_mechanics(self, g):
        return g.face_centers[: g.dim].ravel("f")

    def create_grid(self):
        np.random.seed(0)
        self.gb = self._grid_meth()
        xn = self.gb.grids_of_dimension(self.gb.dim_max())[0].nodes
        box = {
            "xmin": xn[0].min(),
            "xmax": xn[0].max(),
            "ymin": xn[1].min(),
            "ymax": xn[0].max(),
        }
        if self.gb.dim_max() == 3:
            box.update({"zmin": xn[2].min(), "zmax": xn[2].max()})
        self.box = box


def _stepwise_newton_with_comparison(model_as, model_ad, prepare=True):

    if prepare:
        # Only do this for time-independent problems
        model_as.prepare_simulation()
        model_ad.prepare_simulation()
    model_as.before_newton_loop()
    model_ad.before_newton_loop()

    is_converged_as = False
    is_converged_ad = False

    prev_sol_as = model_as.get_state_vector()
    init_sol_as = prev_sol_as

    prev_sol_ad = model_ad.get_state_vector()
    init_sol_ad = prev_sol_ad

    tol = 1e-10

    iteration_counter = 0

    dofs = _block_reordering(
        model_as.eq_order, model_as.dof_manager, model_ad._eq_manager
    )

    while iteration_counter <= 20 and (not is_converged_as or not is_converged_ad):
        # Re-discretize the nonlinear term
        model_as.before_newton_iteration()
        model_ad.before_newton_iteration()

        A_as, b_as = model_as.assembler.assemble_matrix_rhs()
        A_ad, b_ad = model_ad._eq_manager.assemble_matrix_rhs()

        A_as = A_as[dofs]
        b_as = b_as[dofs]

        dA = A_as - A_ad
        #       breakpoint()
        if dA.data.size > 0:
            if np.max(np.abs(dA.data)) > tol:
                raise ValueError("Mismatch in Jacobian matrices")

        # No check on equality of right hand sides, they will be different since
        # ad uses a cumulative approach to the updates.

        # Solve linear system
        sol_as = model_as.assemble_and_solve_linear_system(tol)
        sol_ad = model_ad.assemble_and_solve_linear_system(tol)
        #        g = model_ad.gb.grids_of_dimension(2)[0]
        #        print(model_ad.gb.node_props(g, pp.STATE)['p'])
        #        breakpoint()
        model_as.after_newton_iteration(sol_as)
        model_ad.after_newton_iteration(sol_ad)

        #       breakpoint()
        _, is_converged_as, _ = model_as.check_convergence(
            sol_as, prev_sol_as, init_sol_as, {"nl_convergence_tol": tol}
        )
        _, is_converged_ad, is_diverged = model_as.check_convergence(
            sol_ad + prev_sol_ad, prev_sol_ad, init_sol_ad, {"nl_convergence_tol": tol}
        )

        prev_sol_as = sol_as
        #       breakpoint()
        prev_sol_ad += sol_ad
        iteration_counter += 1

        if is_converged_as:
            model_as.after_newton_convergence(sol_as, [], iteration_counter)
        if is_converged_ad:
            model_ad.after_newton_convergence(sol_ad, [], iteration_counter)

    state_as = model_as.get_state_vector()
    state_ad = model_ad.get_state_vector()
    # Solutions should be identical.
    assert np.linalg.norm(state_as - state_ad) < tol


def _block_reordering(eq_names, dof_manager, eqn_manager):

    assert len(eq_names) == len(eqn_manager.equations)

    def compare_grids(g1, g2):
        if g1.dim != g2.dim:
            return False
        if g1.dim == 0:
            return np.allclose(g1.cell_centers, g2.cell_centers)

        n1, n2 = g1.nodes, g2.nodes

        return test_utils.compare_arrays(n1, n2, sort=False)

    keys = []
    for (name, eq) in zip(eq_names, eqn_manager.equations):
        for grid_eq in eq.grid_order:
            for (grid, var) in dof_manager.block_dof:
                if var != name:
                    continue
                # This is the right variable, now sort the grids
                if isinstance(grid, tuple) and isinstance(grid_eq, tuple):
                    g0, g1 = grid
                    ge0, ge1 = grid_eq

                    if compare_grids(g0, ge0) and compare_grids(g1, ge1):
                        if (grid, var) not in keys:
                            keys.append((grid, var))

                elif isinstance(grid, pp.Grid) and isinstance(grid_eq, pp.Grid):
                    if compare_grids(grid, grid_eq):  # and (grid, var) not in keys:
                        keys.append((grid, var))
                else:
                    continue

    assert len(keys) == len(dof_manager.block_dof)

    new_ind = np.hstack([dof_manager.dof_ind(k[0], k[1]) for k in keys])
    return new_ind


def _timestep_stepwise_newton_with_comparison(model_as, model_ad):

    model_as.prepare_simulation()
    model_as.before_newton_loop()

    model_ad.prepare_simulation()
    model_ad.before_newton_loop()

    tol = 1e-10

    model_as.time_index = 0
    model_ad.time_index = 0
    t_end = model_as.end_time

    while model_as.time < t_end:
        for model in (model_as, model_ad):
            model.time += model.time_step
            model.time_index += 1
        _stepwise_newton_with_comparison(model_as, model_ad, prepare=False)

    state_as = model_as.get_state_vector()
    state_ad = model_ad.get_state_vector()
    # Solutions should be identical.
    assert np.linalg.norm(state_as - state_ad) < tol


@pytest.mark.parametrize(
    "grid_method",
    [
        _single_fracture_2d,
        _single_fracture_near_global_boundary_2d,
        _two_fractures_2d,
        _two_fractures_crossing_2d,
        _single_fracture_3d,
        _three_fractures_crossing_3d,
    ],
)
def test_contact_mechanics(grid_method):
    model_as = ContactModel({}, grid_method)

    model_ad = ContactModel({}, grid_method)
    model_ad._use_ad = True

    _stepwise_newton_with_comparison(model_as, model_ad)


@pytest.mark.parametrize(
    "grid_method",
    [
        _single_fracture_2d,
        _single_fracture_near_global_boundary_2d,
        _two_fractures_2d,
        _two_fractures_crossing_2d,
        _single_fracture_3d,
        _three_fractures_crossing_3d,
    ],
)
def test_contact_mechanics_biot(grid_method):
    model_as = BiotContactModel({}, grid_method)

    model_ad = BiotContactModel({}, grid_method)
    model_ad._use_ad = True
    _timestep_stepwise_newton_with_comparison(model_as, model_ad)
