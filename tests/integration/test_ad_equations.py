""" Test the equation factory which produces what is considered standard equations.

The tests are based on equivalence between the equations defined by the AD framework
and the classical way of defining equations.
"""
import pytest
import porepy as pp
import numpy as np



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
            specified_parameters = {"bc": bc, "bc_values": bc_values, "source": sources}
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

    source = pp.ad.ParameterArray(
        param_keyword=keyword, array_keyword="source", grids=grid_list
    )

    projections = pp.ad.MortarProjections(gb=gb, grids=grid_list, edges=edge_list)
    div = pp.ad.Divergence(grids=grid_list)

    p = manager.merge_variables([(g, pressure_variable) for g in grid_list])
    lmbda = manager.merge_variables([(e, flux_variable) for e in edge_list])

    flux = (
        node_discr.flux * p
        + node_discr.bound_flux * bc_val
        + node_discr.bound_flux * projections.mortar_to_primary_int * lmbda
    )
    flow_eq = div * flux - projections.mortar_to_secondary_int * lmbda - source

    interface_flux = (
        edge_discr.mortar_discr
        * projections.primary_to_mortar_avg
        * node_discr.bound_pressure_cell
        * p
        + edge_discr.mortar_discr
        * projections.primary_to_mortar_avg
        * node_discr.bound_pressure_face
        * projections.mortar_to_primary_int
        * lmbda
        - edge_discr.mortar_discr * projections.secondary_to_mortar_avg * p
        + lmbda
    )

    flow_eq.discretize(gb)

    manager.equations.update({"matrix_flow": flow_eq, "mortar_flow": interface_flux})

    state = np.zeros(gb.num_cells() + gb.num_mortar_cells())
    A, b = manager.assemble(state=state)
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

    def _bc_values(self, g):
        return g.face_centers[: g.dim].ravel("f")

    def create_grid(self):
        np.random.seed(0)
        self.gb = self._grid_meth()


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
    """Run two model instances and compare solutions for each time step.


    Parameters
    ----------
    model_as : AbstractModel
        Model class not using ad.
    model_ad : AbstractModel
        Model class using ad.
    prepare : bool, optional
        Whether to run prepare_simulation method. The default is True.

    Returns
    -------
    None.

    """
    if prepare:
        # Only do this for time-independent problems
        model_as.prepare_simulation()
        model_ad.prepare_simulation()
    model_as.before_newton_loop()
    model_ad.before_newton_loop()

    is_converged_as = False
    is_converged_ad = False

    prev_sol_as = model_as.dof_manager.assemble_variable(from_iterate=False)
    init_sol_as = prev_sol_as

    prev_sol_ad = model_ad.dof_manager.assemble_variable(from_iterate=False)
    init_sol_ad = prev_sol_ad

    tol = 1e-8

    iteration_counter = 0


    while iteration_counter <= 20 and (not is_converged_as or not is_converged_ad):
        # Re-discretize the nonlinear term
        model_as.before_newton_iteration()
        model_ad.before_newton_iteration()

        A_as, b_as = model_as.assembler.assemble_matrix_rhs()
        A_ad, b_ad = model_ad._eq_manager.assemble()


        # Solve linear system
        sol_as = model_as.assemble_and_solve_linear_system(tol)
        sol_ad = model_ad.assemble_and_solve_linear_system(tol)
        model_as.after_newton_iteration(sol_as)
        model_ad.after_newton_iteration(sol_ad)

        e_as, is_converged_as, _ = model_as.check_convergence(
            sol_as, prev_sol_as, init_sol_as, {"nl_convergence_tol": tol}
        )
        e_ad, is_converged_ad, is_diverged = model_ad.check_convergence(
            sol_ad, prev_sol_ad, init_sol_ad, {"nl_convergence_tol": tol}
        )

        prev_sol_as = sol_as
        prev_sol_ad += sol_ad
        iteration_counter += 1
        if is_converged_as and is_converged_ad:
            model_as.after_newton_convergence(sol_as, [], iteration_counter)
            model_ad.after_newton_convergence(sol_ad, [], iteration_counter)

    # This is intended as a check that the dof ordering is the same for both models:
    for t0, t1 in zip(model_as.gb, model_ad.gb):
        assert(np.all(np.isclose(t0[0].cell_centers, t1[0].cell_centers)))
    for t0, t1 in zip(model_as.dof_manager.block_dof.keys(), model_ad.dof_manager.block_dof.keys()):
        assert(t0[1]==t1[1])
        assert(model_as.dof_manager.block_dof[t0]==model_ad.dof_manager.block_dof[t1])
        if isinstance(t0[0], tuple):
            mg0 = model_as.gb.edge_props(t0[0])["mortar_grid"]
            mg1 = model_ad.gb.edge_props(t1[0])["mortar_grid"]
            assert(np.all(np.isclose(mg0.cell_centers, mg1.cell_centers)))
        else:
            assert(np.all(np.isclose(t0[0].cell_centers, t1[0].cell_centers)))
    state_as = model_as.dof_manager.assemble_variable(from_iterate=False)
    state_ad = model_ad.dof_manager.assemble_variable(from_iterate=False)
    # Solutions should be identical.
    assert np.linalg.norm(state_as - state_ad) < tol


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

    state_as = model_as.dof_manager.assemble_variable(from_iterate=False)
    state_ad = model_ad.dof_manager.assemble_variable(from_iterate=False)
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
    model_ad.params["use_ad"] = True

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
    model_ad.params["use_ad"] = True

    _timestep_stepwise_newton_with_comparison(model_as, model_ad)
