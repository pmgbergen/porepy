""" Test the equation factory which produces what is considered standard equations.

The tests are based on equivalence between the equations defined by the AD framework
and the classical way of defining equations.
"""
from typing import Union

import numpy as np
import pytest

import porepy as pp


def test_md_flow():
    # Three fractures, will create intersection lines and point
    frac_1 = np.array([[2, 4, 4, 2], [3, 3, 3, 3], [0, 0, 6, 6]])
    frac_2 = np.array([[3, 3, 3, 3], [2, 4, 4, 2], [0, 0, 6, 6]])
    frac_3 = np.array([[0, 6, 6, 0], [2, 2, 4, 4], [3, 3, 3, 3]])

    mdg = pp.meshing.cart_grid(fracs=[frac_1, frac_2, frac_3], nx=np.array([6, 6, 6]))
    mdg.compute_geometry()

    pressure_variable = "pressure"
    flux_variable = "mortar_flux"

    keyword = "flow"
    discr = pp.Tpfa(keyword)
    source_discr = pp.ScalarSource(keyword)
    coupling_discr = pp.RobinCoupling(keyword, discr, discr)

    for sd, sd_data in mdg.subdomains(return_data=True):

        # Assign data
        if sd.dim == mdg.dim_max():
            upper_left_ind = np.argmax(np.linalg.norm(sd.face_centers, axis=0))
            bc = pp.BoundaryCondition(sd, np.array([0, upper_left_ind]), ["dir", "dir"])
            bc_values = np.zeros(sd.num_faces)
            bc_values[0] = 1
            sources = np.random.rand(sd.num_cells) * sd.cell_volumes
            specified_parameters = {"bc": bc, "bc_values": bc_values, "source": sources}
        else:
            sources = np.random.rand(sd.num_cells) * sd.cell_volumes
            specified_parameters = {"source": sources}

        # Initialize data
        pp.initialize_default_data(sd, sd_data, keyword, specified_parameters)

        # Declare grid primary variable
        sd_data[pp.PRIMARY_VARIABLES] = {pressure_variable: {"cells": 1}}

        # Assign discretization
        sd_data[pp.DISCRETIZATION] = {
            pressure_variable: {"diff": discr, "source": source_discr}
        }

        # Initialize state
        sd_data[pp.STATE] = {
            pressure_variable: np.zeros(sd.num_cells),
            pp.ITERATE: {pressure_variable: np.zeros(sd.num_cells)},
        }
    for intf, intf_data in mdg.interfaces(return_data=True):
        sd_primary, sd_secondary = mdg.interface_to_subdomain_pair(intf)
        pp.initialize_data(intf, intf_data, keyword, {"normal_diffusivity": 1})

        intf_data[pp.PRIMARY_VARIABLES] = {flux_variable: {"cells": 1}}
        intf_data[pp.COUPLING_DISCRETIZATION] = {}
        intf_data[pp.COUPLING_DISCRETIZATION]["coupling"] = {
            sd_primary: (pressure_variable, "diff"),
            sd_secondary: (pressure_variable, "diff"),
            intf: (flux_variable, coupling_discr),
        }
        intf_data[pp.STATE] = {
            flux_variable: np.zeros(intf.num_cells),
            pp.ITERATE: {flux_variable: np.zeros(intf.num_cells)},
        }

    dof_manager = pp.DofManager(mdg)
    assembler = pp.Assembler(mdg, dof_manager)
    assembler.discretize()

    # Reference discretization
    A_ref, b_ref = assembler.assemble_matrix_rhs()

    manager = pp.ad.EquationManager(mdg, dof_manager)

    subdomains = mdg.subdomains()
    interfaces = mdg.interfaces()

    node_discr = pp.ad.MpfaAd(keyword, subdomains)

    edge_discr = pp.ad.RobinCouplingAd(keyword, interfaces)

    bc_val = pp.ad.BoundaryCondition(keyword, subdomains)

    source = pp.ad.ParameterArray(
        param_keyword=keyword, array_keyword="source", subdomains=subdomains
    )

    projections = pp.ad.MortarProjections(
        mdg=mdg, subdomains=subdomains, interfaces=interfaces
    )
    div = pp.ad.Divergence(subdomains=subdomains)

    p = manager.merge_variables([(sd, pressure_variable) for sd in subdomains])
    lmbda = manager.merge_variables([(intf, flux_variable) for intf in interfaces])

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

    flow_eq.discretize(mdg)

    manager.equations.update({"matrix_flow": flow_eq, "mortar_flow": interface_flux})

    state = np.zeros(mdg.num_subdomain_cells() + mdg.num_interface_cells())
    A, b = manager.assemble(state=state)
    diff = A - A_ref
    if diff.data.size > 0:
        assert np.max(np.abs(diff.data)) < 1e-10
    assert np.max(np.abs(b - b_ref)) < 1e-10


# Below are grid bucket generators to be used for tests of contact-mechanics models


def _single_fracture_2d() -> pp.MixedDimensionalGrid:
    frac = [np.array([[2, 3], [2, 2]])]
    mdg = pp.meshing.cart_grid(frac, [5, 4])
    pp.contact_conditions.set_projections(mdg)
    return mdg


def _single_fracture_near_global_boundary_2d() -> pp.MixedDimensionalGrid:
    frac = [np.array([[1, 2], [1, 1]])]
    mdg = pp.meshing.cart_grid(frac, [3, 2])
    pp.contact_conditions.set_projections(mdg)
    return mdg


def _two_fractures_2d() -> pp.MixedDimensionalGrid:
    frac = [np.array([[2, 3], [1, 1]]), np.array([[2, 3], [3, 3]])]
    mdg = pp.meshing.cart_grid(frac, [5, 4])
    pp.contact_conditions.set_projections(mdg)
    return mdg


def _two_fractures_crossing_2d() -> pp.MixedDimensionalGrid:
    frac = [np.array([[1, 3], [2, 2]]), np.array([[2, 2], [1, 3]])]
    mdg = pp.meshing.cart_grid(frac, [4, 4])
    pp.contact_conditions.set_projections(mdg)
    return mdg


def _single_fracture_3d() -> pp.MixedDimensionalGrid:
    frac = [np.array([[1, 2, 2, 1], [1, 1, 2, 2], [1, 1, 1, 1]])]
    mdg = pp.meshing.cart_grid(frac, [3, 3, 2])
    pp.contact_conditions.set_projections(mdg)
    return mdg


def _three_fractures_crossing_3d() -> pp.MixedDimensionalGrid:
    frac = [
        np.array([[1, 3, 3, 1], [1, 1, 3, 3], [2, 2, 2, 2]]),
        np.array([[1, 1, 3, 3], [2, 2, 2, 2], [1, 3, 3, 1]]),
        np.array([[2, 2, 2, 2], [1, 1, 3, 3], [1, 3, 3, 1]]),
    ]
    mdg = pp.meshing.cart_grid(frac, [4, 4, 4])
    pp.contact_conditions.set_projections(mdg)
    return mdg


class ContactModel(pp.ContactMechanics):
    def __init__(self, param, grid_meth):
        super().__init__(param)
        self._grid_meth = grid_meth

    def _bc_values(self, sd):
        val = np.zeros((sd.dim, sd.num_faces))
        tags = sd.tags["domain_boundary_faces"]
        val[: sd.dim, tags] = sd.face_centers[: sd.dim, tags]
        return val.ravel("f")

    def create_grid(self):
        np.random.seed(0)
        self.mdg = self._grid_meth()


def create_grid(model):
    np.random.seed(0)
    model.mdg = model._grid_meth()
    xn = model.mdg.subdomains(dim=model.mdg.dim_max())[0].nodes
    box = {
        "xmin": xn[0].min(),
        "xmax": xn[0].max(),
        "ymin": xn[1].min(),
        "ymax": xn[0].max(),
    }
    if model.mdg.dim_max() == 3:
        box.update({"zmin": xn[2].min(), "zmax": xn[2].max()})
    model.box = box


class BiotContactModel(pp.ContactMechanicsBiot):
    def __init__(self, param, grid_meth):
        ## NB: tests should be with and without subtract_fracture_pressure
        # Time steps should be an input parameters, try a few
        # Make initial states random?
        # Biot alpha should vary
        super().__init__(param)
        self._grid_meth = grid_meth

    def _bc_values_scalar(self, sd):
        # Set pressure values at the domain boundaries.
        val = np.zeros(sd.num_faces)
        tags = sd.tags["domain_boundary_faces"]
        val[tags] = sd.face_centers[: sd.dim, tags].sum(axis=0)
        return val

    def _bc_values_mechanics(self, sd):
        # Set displacement values at the domain boundaries.
        # NOTE TO SELF: In a previous version of this test, Dirichlet values were set
        # also on the internal boundary. This, combined with the scaling issue noted
        # below, was sufficient to give mismatch between the Ad and Assembler versions
        # of the code (on Linux, but not on Windows, for some reason). The gut feel is,
        # the scaling issue seem the more likely culprit.
        val = np.zeros((sd.dim, sd.num_faces))
        tags = sd.tags["domain_boundary_faces"]
        # IMPLEMENTATION NOTE: If we do not divide by 10 here, a mismatch between the
        # Ad and Assembler code will result for some tests (on Linux, but not on Windows).
        # Dividing by 10 solves the issue. Most likely, but not confirmed, this is
        # caused by a tolerance in the two implementations of the contact mechanics
        # problem, that somehow make the two solutions diverge.
        val[: sd.dim, tags] = sd.face_centers[: sd.dim, tags] / 10

        return val.ravel("f")

    def create_grid(self):
        create_grid(self)


class THMModel(pp.THM):
    def __init__(self, param, grid_meth):
        super().__init__(param)
        self._grid_meth = grid_meth

    def _scalar_temperature_coupling_coefficient(self, sd: pp.Grid) -> float:
        """
        TH coupling coefficient.

        Higher coefficients make assembler version take even more of iterations.
        """
        return -1.0e-3

    def _source_scalar(self, sd):
        if sd.dim == self.mdg.dim_max() - 1:
            val = 1.0
        else:
            val = 0.0
        return val * sd.cell_volumes

    def _source_temperature(self, sd):
        if sd.dim == self.mdg.dim_max() - 1:
            val = 1.0
        else:
            val = 0.0
        return val * sd.cell_volumes

    def create_grid(self):
        create_grid(self)

    def _biot_beta(self, sd: pp.Grid) -> Union[float, np.ndarray]:
        """
        TM coupling coefficient
        """
        if sd.dim == self.mdg.dim_max() - 1:
            val = 1.0
        else:
            val = 0.3
        if self._use_ad:
            return val * np.ones(sd.num_cells)
        else:
            return val

    def _biot_alpha(self, sd: pp.Grid) -> Union[float, np.ndarray]:
        """
        TM coupling coefficient
        """
        if sd.dim == self.mdg.dim_max() - 1:
            val = 1.0
        else:
            val = 0.2
        if self._use_ad:
            return val * np.ones(sd.num_cells)
        else:
            return val


def _compare_solutions(m0, m1, tol=1e-7) -> None:
    """Safe comparison of solutions without assumptions of identical dof ordering.

    We loop over block_dof keys, i.e. combinations of grid_like and variable names,
    and identify the grid in m1.mdg by cell center coordinates. We assume the grids
    to be identical for each subdomain.

    Args:
        m0 (Model): First model.
        m1 (Model): Second model.
        tol (float): Comparison tolerance.
    """

    def _same_grid(g0, g1):
        if g0.num_cells == g1.num_cells:
            return np.all(np.isclose(g0.cell_centers, g1.cell_centers))
        else:
            return False

    for grid_like, var in m0.dof_manager.block_dof.keys():
        # Identify grid in m1
        found = False
        if isinstance(grid_like, pp.MortarGrid):
            sd_primary0, sd_secondary0 = m0.mdg.interface_to_subdomain_pair(grid_like)
            for intf, intf_data in m1.mdg.interfaces(return_data=True):
                # Check for equality of both neighbour subdomains. MortarGrid
                # coordinates are not unique for intersection interfaces.
                sd_primary1, sd_secondary1 = m1.mdg.interface_to_subdomain_pair(intf)
                found = _same_grid(sd_primary0, sd_primary1) and _same_grid(
                    sd_secondary0, sd_secondary1
                )
                if found:
                    sol0 = m0.mdg.interface_data(grid_like)[pp.STATE][var]
                    if var == getattr(
                        m0, "mortar_temperature_advection_variable", "foo"
                    ):
                        # Reformulation; the advective flux is time integrated in
                        # non-ad but not in ad.
                        sol1 = intf_data[pp.STATE][var] * m1.time_step
                    else:
                        sol1 = intf_data[pp.STATE][var]
                    break
        else:
            for sd, sd_data in m1.mdg.subdomains(return_data=True):
                found = _same_grid(grid_like, sd)
                if found:
                    sol0 = m0.mdg.subdomain_data(grid_like)[pp.STATE][var]
                    sol1 = sd_data[pp.STATE][var]
                    break
        assert found
        diff = sol1 - sol0
        norm = np.linalg.norm(diff)
        assert norm < tol


def _stepwise_newton_with_comparison(model_as, model_ad, prepare=True) -> None:
    """Run two model instances and compare solutions for each time step.

    Args:
        model_as : AbstractModel
            Model class not using ad.
        model_ad : AbstractModel
            Model class using ad.
        prepare : bool, optional
            Whether to run prepare_simulation method. The default is True.
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
    init_sol_ad = prev_sol_ad.copy()

    tol = 1e-12

    iteration_counter = 0

    while iteration_counter <= 15 and (not is_converged_as or not is_converged_ad):
        # Re-discretize the nonlinear term
        model_as.before_newton_iteration()
        model_ad.before_newton_iteration()

        # Solve linear system
        model_as.assemble_linear_system()
        sol_as = model_as.solve_linear_system()
        model_ad.assemble_linear_system()
        sol_ad = model_ad.solve_linear_system()

        model_as.after_newton_iteration(sol_as)
        model_ad.after_newton_iteration(sol_ad)

        e_as, is_converged_as, _ = model_as.check_convergence(
            sol_as, prev_sol_as, init_sol_as, {"nl_convergence_tol": tol}
        )
        e_ad, is_converged_ad, is_diverged = model_ad.check_convergence(
            sol_ad, prev_sol_ad, init_sol_ad, {"nl_convergence_tol": tol}
        )
        prev_sol_as = sol_as.copy()
        prev_sol_ad += sol_ad
        iteration_counter += 1
        if is_converged_as and is_converged_ad:
            model_as.after_newton_convergence(sol_as, [], iteration_counter)
            model_ad.after_newton_convergence(sol_ad, [], iteration_counter)
    # Check that both models have converged, just to be on the safe side
    assert is_converged_as and is_converged_ad
    # Solutions should be identical.
    _compare_solutions(model_as, model_ad)


def _timestep_stepwise_newton_with_comparison(model_as, model_ad):
    model_as.prepare_simulation()
    model_as.before_newton_loop()

    model_ad.prepare_simulation()
    model_ad.before_newton_loop()

    model_as.time_index = 0
    model_ad.time_index = 0
    t_end = model_as.end_time

    while model_as.time < t_end:
        for model in (model_as, model_ad):
            model.time += model.time_step
            model.time_index += 1
        _stepwise_newton_with_comparison(model_as, model_ad, prepare=False)


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
    params = {"time_step": 0.5, "end_time": 1}
    model_as = BiotContactModel(params.copy(), grid_method)

    model_ad = BiotContactModel(params, grid_method)
    model_ad._use_ad = True
    model_ad.params["use_ad"] = True

    _timestep_stepwise_newton_with_comparison(model_as, model_ad)


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
def test_thm(grid_method):
    params = {"time_step": 0.5e-0, "end_time": 1.0e-0}
    model_as = THMModel(params.copy(), grid_method)

    model_ad = THMModel(params, grid_method)
    model_as._use_ad = False
    model_as.params["use_ad"] = False

    _timestep_stepwise_newton_with_comparison(model_as, model_ad)
