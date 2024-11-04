"""Test module to test the functionality of the secondary operator class, with
external evaluation, storage and access of values and derivative values."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

import porepy as pp
from porepy.applications.md_grids.fracture_sets import orthogonal_fractures_2d

VAR1_NAME = "var1"
VAR2_NAME = "var2"
INTFVAR_NAME = "intfvar"


@pytest.fixture
def eqsys() -> pp.ad.EquationSystem:
    """Fixture containing the equation system, variables and MDG for testing the
    secondary operator.

    Unit square with 2 line fractures forming a cross in the middle.

    2 Variables on subdomains, 1 variable on interfaces.

    """

    bounding_box = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    domain = pp.Domain(bounding_box=bounding_box)

    fracs = orthogonal_fractures_2d(0.5, [np.array([0.5, 0.8]), np.array([0.8, 0.5])])

    network_2d = pp.create_fracture_network(fracs, domain)

    mesh_args: dict[str, float] = {"cell_size": 0.5, "cell_size_fracture": 0.5}
    mdg = pp.create_mdg("cartesian", mesh_args, network_2d)
    mdg.compute_geometry()

    sds = mdg.subdomains()
    intfs = mdg.interfaces()

    eqsys_ = pp.ad.EquationSystem(mdg)

    eqsys_.create_variables(VAR1_NAME, {"cells": 1}, subdomains=sds)
    eqsys_.create_variables(VAR2_NAME, {"cells": 1}, subdomains=sds)
    eqsys_.create_variables(INTFVAR_NAME, {"cells": 1}, interfaces=intfs)

    # set zero values at current iterate to kickstart AD
    eqsys_.set_variable_values(np.zeros(eqsys_.num_dofs()), iterate_index=0)

    # setting also variable values at prev time and iter for testing
    eqsys_.set_variable_values(
        np.zeros(eqsys_.num_dofs()), iterate_index=1, time_step_index=0
    )

    return eqsys_


@pytest.fixture
def get_var(eqsys: pp.ad.EquationSystem):
    """Returns callables which return callable representations of variables for a
    variable name."""

    def _get_var(name: str):
        if name in [VAR1_NAME, VAR2_NAME]:

            def _var(subdomains):
                if all(isinstance(d, pp.Grid) for d in subdomains):
                    return eqsys.md_variable(name, subdomains)
                elif all(isinstance(d, pp.BoundaryGrid) for d in subdomains):
                    return pp.ad.TimeDependentDenseArray(name, subdomains)

        elif name in [INTFVAR_NAME]:

            def _var(interfaces):
                return eqsys.md_variable(name, interfaces)

        else:
            raise NotImplementedError(f"get_var fixture not supporting var name {name}")

        return _var

    return _get_var


@pytest.mark.parametrize("on_intf", [False, True])
def test_secondary_operators(
    on_intf: bool,
    eqsys: pp.ad.EquationSystem,
    get_var: Callable[[str], Callable[[pp.GridLikeSequence], pp.ad.Operator]],
):
    """Test all aspects of the secondary operator at current time and iter."""

    mdg = eqsys.mdg
    sds = mdg.subdomains()
    intfs = mdg.interfaces()
    bgs = mdg.boundaries()

    if on_intf:
        nc = mdg.num_interface_cells()
        domains = intfs
        vars = [get_var(INTFVAR_NAME)(domains)]
        diff_vals = [2 * np.ones(nc)]
        expr = pp.ad.SurrogateFactory(
            "interface_expression",
            eqsys.mdg,
            [get_var(INTFVAR_NAME)],
        )
    else:
        nc = mdg.num_subdomain_cells()
        domains = sds
        vars = [get_var(VAR1_NAME)(domains), get_var(VAR2_NAME)(domains)]

        diff_vals = [2 * np.ones(nc), 3 * np.ones(nc)]
        expr = pp.ad.SurrogateFactory(
            "subdomain_expression",
            eqsys.mdg,
            [get_var(VAR1_NAME), get_var(VAR2_NAME)],
        )

    # Testing the created operators
    sop = expr(domains)
    sop_empty = expr([])

    sop_pi = sop.previous_iteration()
    sop_pt = sop.previous_timestep()

    # Test that the right operator was created
    assert isinstance(sop, pp.ad.SurrogateOperator)
    # Calling the secondary expression with no grids, gives a wrapped empty array
    assert isinstance(sop_empty, pp.ad.DenseArray)
    assert sop_empty.value(eqsys).shape == (0,)

    # secondary operator (SOP) starts at current iterate
    assert sop.time_step_index is None
    assert sop.iterate_index == 0
    # assert the prev index was increased for SOP and its children
    assert sop_pi.iterate_index == 0
    assert sop_pi.time_step_index is None
    assert all(v.iterate_index == 0 for v in sop_pi.children)
    assert all(v.time_step_index is None for v in sop_pi.children)
    assert sop_pt.iterate_index is None
    assert sop_pt.time_step_index == 0
    assert all(v.iterate_index is None for v in sop_pt.children)
    assert all(v.time_step_index == 0 for v in sop_pt.children)

    # At this point, no data has been set, check correct return format for no data case
    for g in sds + intfs + bgs:
        with pytest.raises(KeyError):
            expr.fetch_data(sop, g, get_derivatives=True)
        with pytest.raises(KeyError):
            expr.fetch_data(sop, g, get_derivatives=False)

    with pytest.raises(ValueError):
        _ = sop.value_and_jacobian(eqsys)
    with pytest.raises(ValueError):
        _ = sop_pi.value_and_jacobian(eqsys)
    with pytest.raises(ValueError):
        _ = sop_pt.value_and_jacobian(eqsys)

    ## Testing that setting arbitrary data not using the framework will raise errors
    # if data does not fit
    data = mdg.boundary_grid_data(bgs[0])
    if pp.ITERATE_SOLUTIONS not in data:
        data[pp.ITERATE_SOLUTIONS] = {}
    if expr.name not in data[pp.ITERATE_SOLUTIONS]:
        data[pp.ITERATE_SOLUTIONS][expr.name] = {}
    data[pp.ITERATE_SOLUTIONS][expr.name][0] = np.ones(nc - 1)
    with pytest.raises(KeyError):
        _ = expr.boundary_values
    # deleting entry
    del data[pp.ITERATE_SOLUTIONS][expr.name][0]

    # analogous for interfaces and subdomains
    if on_intf:
        data = mdg.interface_data(domains[0])
    else:
        data = mdg.subdomain_data(domains[0])

    if pp.ITERATE_SOLUTIONS not in data:
        data[pp.ITERATE_SOLUTIONS] = {}
    if expr.name not in data[pp.ITERATE_SOLUTIONS]:
        data[pp.ITERATE_SOLUTIONS][expr.name] = {}
    if expr._name_derivatives not in data[pp.ITERATE_SOLUTIONS]:
        data[pp.ITERATE_SOLUTIONS][expr._name_derivatives] = {}
    data[pp.ITERATE_SOLUTIONS][expr.name][0] = np.ones(3 * nc)
    data[pp.ITERATE_SOLUTIONS][expr._name_derivatives][0] = np.ones(3 * nc)
    if on_intf:
        with pytest.raises(KeyError):
            _ = expr.interface_values
        with pytest.raises(KeyError):
            _ = expr.interface_derivatives
    else:
        with pytest.raises(KeyError):
            _ = expr.subdomain_values
        with pytest.raises(KeyError):
            _ = expr.subdomain_derivatives
    # deleting entry
    del data[pp.ITERATE_SOLUTIONS][expr.name][0]
    del data[pp.ITERATE_SOLUTIONS][expr._name_derivatives][0]

    ## Testing the setting of values using the local and global setter for iter values
    for g in domains:
        expr.progress_iterate_values_on_grid(np.zeros(g.num_cells), g)

    # asserting the correct values are fetched by global getter and through parsing
    # and shifting the values iteratively on domains
    if on_intf:
        assert np.all(np.zeros(nc) == expr.interface_values)
        assert np.all(np.zeros(nc) == sop.value(eqsys))
        expr.interface_values = np.ones(nc)
    else:
        assert np.all(np.zeros(nc) == expr.subdomain_values)
        assert np.all(np.zeros(nc) == sop.value(eqsys))
        expr.subdomain_values = np.ones(nc)

    # Check parsing of previous iter and current op
    # Note that prev iter operator has the same values, but no Jacobian (tested later)
    assert np.all(sop.value(eqsys) == np.ones(nc))
    assert np.all(sop_pi.value(eqsys) == np.ones(nc))
    with pytest.raises(ValueError):
        _ = sop.previous_iteration(steps=2).value(eqsys)
    # Still no data at previous time step
    with pytest.raises(ValueError):
        _ = sop_pt.value(eqsys)

    # constructing diffs for current iter, with respective numbers of dependencies
    if on_intf:
        expr.interface_derivatives = np.array(diff_vals)
    else:
        expr.subdomain_derivatives = np.array(diff_vals)

    # everything is set, the derivative values must have a certain structure
    var_vals = [var.value_and_jacobian(eqsys) for var in vars]
    sop_val = sop.value_and_jacobian(eqsys)

    # Assert operator fetches the right values uppon evaluation
    if on_intf:
        assert np.allclose(sop_val.val, expr.interface_values)
    else:
        assert np.allclose(sop_val.val, expr.subdomain_values)

    # The jacobians of the variables are identity blocks
    # The Jacobian of the secondary operator must be the sum of respective identities
    # with new data
    jacs = []
    for v, d in zip(var_vals, diff_vals):
        jac_ = v.jac.copy()
        jac_.data = d
        jacs.append(jac_)

    assert np.all(sop_val.jac.toarray() == sum(jacs).toarray())

    # progress values in time and check that only values are progressed, and that
    # they are correct, i.e. current iter is set as previous time
    expr.progress_values_in_time(domains)
    if on_intf:
        assert np.all(sop_pt.value(eqsys) == expr.interface_values)
    else:
        assert np.all(sop_pt.value(eqsys) == expr.subdomain_values)
    # No support for derivative values of operators at previous time and iterate
    for g in domains:
        with pytest.raises(ValueError):
            expr.fetch_data(sop_pt, g, True)
        with pytest.raises(ValueError):
            expr.fetch_data(sop_pi, g, True)

    # previous iterate and time step have no derivative values, but the parsing should
    # still work because the dependencies at previous iter and time are parsed as
    # numpy arrays: The function of the SOP (AbstractFunction) should not call
    # get_jacobian of the SOP

    sop_pt_val = sop_pt.value_and_jacobian(eqsys)
    if on_intf:
        assert np.all(sop_pt_val.val == expr.interface_values)
    else:
        assert np.all(sop_pt_val.val == expr.subdomain_values)
    assert np.all(sop_pt_val.jac.toarray() == 0.0)

    sop_pi_val = sop_pi.value_and_jacobian(eqsys)
    assert np.all(sop_pi_val.val == np.ones(nc))
    assert np.all(sop_pi_val.jac.toarray() == 0.0)

    ## Test that the user cannot set values of unexpected shape
    if on_intf:
        with pytest.raises(ValueError):
            expr.interface_values = np.ones(nc - 1)
        with pytest.raises(ValueError):
            expr.interface_values = [g] * nc
        with pytest.raises(ValueError):
            expr.interface_derivatives = np.array(diff_vals)[:, : nc - 1]
    else:
        with pytest.raises(ValueError):
            expr.subdomain_values = np.ones(nc - 1)
        with pytest.raises(ValueError):
            expr.subdomain_values = [g] * nc
        with pytest.raises(ValueError):
            expr.subdomain_derivatives = np.array(diff_vals)[:, : nc - 1]

    # NOTE: The setters work if the given array is too large, only the first values are
    # used then. Should we enforce that the array must be of exact size? And not
    # "at least of size"


def test_secondary_operators_on_boundaries(
    eqsys: pp.ad.EquationSystem,
    get_var: Callable[[str], Callable[[pp.GridLikeSequence], pp.ad.Operator]],
):
    """Testing the functionality to create operators on the boundary and to set
    and progress values on time on the boundaries."""

    mdg = eqsys.mdg
    bgs = mdg.boundaries()

    nc = sum([g.num_cells for g in bgs])

    subdomain_expression = pp.ad.SurrogateFactory(
        "subdomain_expression",
        eqsys.mdg,
        [get_var(VAR1_NAME), get_var(VAR2_NAME)],
    )

    sop = subdomain_expression(bgs)

    # we force this behavior to avoid someone implementing duplicate functionality in
    # future
    assert isinstance(sop, pp.ad.TimeDependentDenseArray)
    assert sop.time_step_index is None

    # fetching boundary values should raise an error
    with pytest.raises(KeyError):
        _ = subdomain_expression.boundary_values

    # testing global setter
    subdomain_expression.boundary_values = np.ones(nc)

    # Parsing the operator should give respective values now
    assert np.all(sop.value(eqsys) == np.ones(nc))

    # testing the local setter
    for g in bgs:
        subdomain_expression.update_boundary_values(np.ones(g.num_cells) * 2, g)

    # parsing the operator should give new values
    assert np.all(sop.value(eqsys) == 2 * np.ones(nc))

    # parsing the operator at the previous time step should give the old values
    assert np.all(sop.previous_timestep().value(eqsys) == np.ones(nc))

    # Testing shift in time
    # testing the local setter
    for g in bgs:
        subdomain_expression.update_boundary_values(np.ones(g.num_cells) * 3, g, depth=2)

    assert np.all(sop.value(eqsys) == 3 * np.ones(nc))
    # parsing the operator at the previous time step should give the old values
    assert np.all(sop.previous_timestep().value(eqsys) == 2 * np.ones(nc))
    assert np.all(sop.previous_timestep(steps=2).value(eqsys) == np.ones(nc))
