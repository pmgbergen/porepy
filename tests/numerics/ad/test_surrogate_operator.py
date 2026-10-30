"""Test module to test the functionality of the secondary operator class, with
external evaluation, storage and access of values and derivative values."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp
from porepy.applications.md_grids.fracture_sets import orthogonal_fractures_2d
from porepy.numerics.ad.equation_system import GridEntity

VAR1_NAME = "var1"
VAR2_NAME = "var2"
INTFVAR_NAME = "intfvar"


@pytest.fixture
def equation_system() -> pp.ad.EquationSystem:
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

    subdomains = mdg.subdomains()
    interfaces = mdg.interfaces()

    equation_system = pp.ad.EquationSystem(mdg)

    equation_system.create_variables(
        VAR1_NAME, {GridEntity.cells: 1}, subdomains=subdomains
    )
    equation_system.create_variables(
        VAR2_NAME, {GridEntity.cells: 1}, subdomains=subdomains
    )
    equation_system.create_variables(
        INTFVAR_NAME, {GridEntity.cells: 1}, interfaces=interfaces
    )

    # set zero values at current iterate to kickstart AD
    equation_system.set_variable_values(
        np.zeros(equation_system.num_dofs()), iterate_index=0
    )

    # setting also variable values at prev time and iter for testing
    equation_system.set_variable_values(
        np.zeros(equation_system.num_dofs()), iterate_index=1, time_step_index=0
    )

    return equation_system


@pytest.fixture
def get_var(equation_system: pp.ad.EquationSystem):
    """Returns callables which return callable representations of variables for a
    variable name."""

    def _get_var(name: str):
        if name in [VAR1_NAME, VAR2_NAME]:

            def _var(subdomains):
                if all(isinstance(d, pp.Grid) for d in subdomains):
                    return equation_system.md_variable(name, subdomains)
                elif all(isinstance(d, pp.BoundaryGrid) for d in subdomains):
                    return pp.ad.TimeDependentDenseArray(name, subdomains)

        elif name in [INTFVAR_NAME]:

            def _var(interfaces):
                return equation_system.md_variable(name, interfaces)

        else:
            raise NotImplementedError(f"get_var fixture not supporting var name {name}")

        return _var

    return _get_var


@pytest.mark.parametrize("on_intf", [False, True])
def test_secondary_operators(
    on_intf: bool,
    equation_system: pp.ad.EquationSystem,
    get_var: Callable[[str], Callable[[pp.GridLikeSequence], pp.ad.Operator]],
):
    """Test all aspects of the secondary operator at current time and iter."""

    mdg = equation_system.mdg
    subdomains = mdg.subdomains()
    interfaces = mdg.interfaces()
    boundary_grids = mdg.boundaries()

    if on_intf:
        nc = mdg.num_interface_cells()
        domains = interfaces
        vars = [get_var(INTFVAR_NAME)(domains)]
        diff_vals = [2 * np.ones(nc)]
        expr = pp.ad.SurrogateFactory(
            "interface_expression",
            equation_system.mdg,
            [get_var(INTFVAR_NAME)],
        )
    else:
        nc = mdg.num_subdomain_cells()
        domains = subdomains
        vars = [get_var(VAR1_NAME)(domains), get_var(VAR2_NAME)(domains)]

        diff_vals = [2 * np.ones(nc), 3 * np.ones(nc)]
        expr = pp.ad.SurrogateFactory(
            "subdomain_expression",
            equation_system.mdg,
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
    assert equation_system.evaluate(sop_empty).shape == (0,)

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
    for g in subdomains + interfaces + boundary_grids:
        with pytest.raises(KeyError):
            expr.fetch_data(sop, g, get_derivatives=True)
        with pytest.raises(KeyError):
            expr.fetch_data(sop, g, get_derivatives=False)

    # Evaluate the SOPs using the Ad parsing framework. Since no data is set, fetching
    # the data raises a KeyError, as above. The parser catches this and raises a
    # ValueError instead.
    with pytest.raises(ValueError):
        _ = sop.value_and_jacobian(equation_system)
    with pytest.raises(ValueError):
        _ = sop_pi.value_and_jacobian(equation_system)
    with pytest.raises(ValueError):
        _ = sop_pt.value_and_jacobian(equation_system)

    ## Testing that setting arbitrary data not using the framework will raise errors
    # if data does not fit
    data = mdg.boundary_grid_data(boundary_grids[0])
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
        assert np.all(np.zeros(nc) == equation_system.evaluate(sop))
        expr.interface_values = np.ones(nc)
    else:
        assert np.all(np.zeros(nc) == expr.subdomain_values)
        assert np.all(np.zeros(nc) == equation_system.evaluate(sop))
        expr.subdomain_values = np.ones(nc)

    # Check parsing of previous iter and current op
    # Note that prev iter operator has the same values, but no Jacobian (tested later)
    assert np.all(equation_system.evaluate(sop) == np.ones(nc))
    assert np.all(equation_system.evaluate(sop_pi) == np.ones(nc))
    with pytest.raises(ValueError):
        _ = equation_system.evaluate(sop.previous_iteration(steps=2))
    # Still no data at previous time step
    with pytest.raises(ValueError):
        _ = equation_system.evaluate(sop_pt)

    # constructing diffs for current iter, with respective numbers of dependencies
    if on_intf:
        expr.interface_derivatives = np.array(diff_vals)
    else:
        expr.subdomain_derivatives = np.array(diff_vals)

    # everything is set, the derivative values must have a certain structure
    var_vals = [var.value_and_jacobian(equation_system) for var in vars]
    sop_val = sop.value_and_jacobian(equation_system)

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
        jac_ = v.full_jac.copy()
        jac_.data = d
        jacs.append(jac_)

    assert np.all(sop_val.full_jac.toarray() == sum(jacs).toarray())

    # progress values in time and check that only values are progressed, and that
    # they are correct, i.e. current iter is set as previous time
    expr.progress_values_in_time(domains)
    if on_intf:
        assert np.all(equation_system.evaluate(sop_pt) == expr.interface_values)
    else:
        assert np.all(equation_system.evaluate(sop_pt) == expr.subdomain_values)
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

    sop_pt_val = sop_pt.value_and_jacobian(equation_system)
    if on_intf:
        assert np.all(sop_pt_val.val == expr.interface_values)
    else:
        assert np.all(sop_pt_val.val == expr.subdomain_values)
    assert np.all(sop_pt_val.jac.toarray() == 0.0)

    sop_pi_val = sop_pi.value_and_jacobian(equation_system)
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
    equation_system: pp.ad.EquationSystem,
    get_var: Callable[[str], Callable[[pp.GridLikeSequence], pp.ad.Operator]],
):
    """Testing the functionality to create operators on the boundary and to set
    and progress values on time on the boundaries."""

    mdg = equation_system.mdg
    boundary_grids = mdg.boundaries()

    nc = sum([bg.num_cells for bg in boundary_grids])

    subdomain_expression = pp.ad.SurrogateFactory(
        "subdomain_expression",
        equation_system.mdg,
        [get_var(VAR1_NAME), get_var(VAR2_NAME)],
    )

    sop = subdomain_expression(boundary_grids)

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
    assert np.all(equation_system.evaluate(sop) == np.ones(nc))

    # testing the local setter
    for bg in boundary_grids:
        subdomain_expression.update_boundary_values(np.ones(bg.num_cells) * 2, bg)

    # parsing the operator should give new values
    assert np.all(equation_system.evaluate(sop) == 2 * np.ones(nc))

    # parsing the operator at the previous time step should give the old values
    assert np.all(equation_system.evaluate(sop.previous_timestep()) == np.ones(nc))

    # Testing shift in time
    # testing the local setter
    for bg in boundary_grids:
        subdomain_expression.update_boundary_values(
            np.ones(bg.num_cells) * 3, bg, depth=2
        )

    assert np.all(equation_system.evaluate(sop) == 3 * np.ones(nc))
    # parsing the operator at the previous time step should give the old values
    assert np.all(equation_system.evaluate(sop.previous_timestep()) == 2 * np.ones(nc))
    assert np.all(
        equation_system.evaluate(sop.previous_timestep(steps=2)) == np.ones(nc)
    )


@pytest.mark.parametrize("violation", ["not_variable", "order", "subset", "dofs"])
def test_dependencies_must_conform_to_domains(
    equation_system: pp.ad.EquationSystem, violation: str
):
    """A surrogate factory raises an error when called on domains on which its
    dependencies do not conform to the surrogate operator.

    The values of a surrogate operator are stacked in the order of the domains it is
    called with, and must match the rows of its dependencies. This fails if the
    dependency is not a variable, if the domains are not in the order of the
    sub-variables of the dependency, if the dependency is defined on other domains, and
    if the numbers of DOFs differ.

    """
    subdomains = equation_system.mdg.subdomains()

    def var1(domains):
        return equation_system.md_variable(VAR1_NAME, domains)

    domains = subdomains
    dof_info = None
    dependency: Callable = var1
    if violation == "not_variable":
        dependency = lambda domains: pp.ad.Scalar(2.0) * var1(domains)
    elif violation == "order":
        domains = subdomains[::-1]
    elif violation == "subset":
        # The dependency is defined on all subdomains, the operator on the first only.
        dependency = lambda domains: var1(subdomains)
        domains = subdomains[:1]
    else:
        # Two DOFs per cell in the surrogate operator, one in the dependency.
        dof_info = {GridEntity.cells: 2}

    factory = pp.ad.SurrogateFactory(
        "surrogate", equation_system.mdg, [dependency], dof_info=dof_info
    )
    expected_error = TypeError if violation == "not_variable" else ValueError
    with pytest.raises(expected_error):
        factory(domains)
    # In the order of the md-grid, the factory can be called on a subset of domains.
    pp.ad.SurrogateFactory("surrogate", equation_system.mdg, [var1])(subdomains[1:])


def _surrogate_and_arguments(
    equation_system: pp.ad.EquationSystem,
    dependencies: list[str],
    derivatives: list[float],
) -> tuple[
    pp.ad.SurrogateOperator,
    list[pp.ad.AdArray],
    list[pp.ad.DiagonalAdArray],
    np.ndarray,
]:
    """Set up a surrogate operator, its arguments in both representations, and its
    expected Jacobian.

    Parameters:
        equation_system: The equation system containing the variables.
        dependencies: Names of the variables the surrogate operator depends on. A name
            can be repeated.
        derivatives: Derivative values of the surrogate operator, one per dependency.

    Returns:
        The surrogate operator; Ad arrays in the full representation for the variables
        ``var1`` and ``var2``; diagonal Ad arrays for the same variables; and the
        expected Jacobian of the surrogate operator, as a dense array. The expected
        Jacobian is assembled directly from the derivative values and the degrees of
        freedom of the dependencies, independently of the surrogate operator.

    """
    mdg = equation_system.mdg
    subdomains = mdg.subdomains()
    nc = mdg.num_subdomain_cells()
    num_dofs = equation_system.num_dofs()

    factory = pp.ad.SurrogateFactory(
        "surrogate",
        mdg,
        [lambda d, n=n: equation_system.md_variable(n, d) for n in dependencies],
    )
    sop = factory(subdomains)
    factory.subdomain_values = np.ones(nc)
    # The derivatives are set to a constant value for each variable, but the first entry
    # is zero.
    derivative_values = np.array([d * np.ones(nc) for d in derivatives])
    derivative_values[:, 0] = 0.0
    factory.subdomain_derivatives = derivative_values

    variables = [
        equation_system.md_variable(name, subdomains) for name in [VAR1_NAME, VAR2_NAME]
    ]
    indices = [equation_system.dofs_of([v]) for v in variables]
    values = [np.arange(ind.size, dtype=float) for ind in indices]
    sparse_array = [
        pp.ad.AdArray(
            val,
            sps.csr_matrix(
                (np.ones(ind.size), (np.arange(ind.size), ind)),
                shape=(ind.size, num_dofs),
            ),
        )
        for val, ind in zip(values, indices)
    ]
    diagonal_array_list = pp.ad.initialize_diagonal_ad_arrays(values, indices, num_dofs)

    # Each cell has the derivative with respect to a dependency in the column of the
    # degree of freedom of that dependency in the cell. The derivatives with respect to
    # a repeated dependency add up.
    expected_jacobian = np.zeros((nc, num_dofs))
    for name, derivative in zip(dependencies, derivative_values):
        dofs = equation_system.dofs_of([equation_system.md_variable(name, subdomains)])
        expected_jacobian[np.arange(nc), dofs] += derivative

    return sop, sparse_array, diagonal_array_list, expected_jacobian


def _argument_in_representation(
    representation: str,
    sparse_array: pp.ad.AdArray,
    diagonal_array: pp.ad.DiagonalAdArray,
) -> pp.ad.AdArrayBase:
    """Pick an argument of a surrogate operator in a given representation.

    Parameters:
        representation: ``"full"`` for an Ad array assembled directly in the full
            representation, ``"diagonal"`` for a diagonal Ad array, and ``"to_full"``
            for the diagonal Ad array converted to the full representation. The latter
            differs from ``"full"`` by explicitly stored zeros in its Jacobian.
        sparse_array: The argument in the full representation.
        diagonal_array: The argument in the diagonal representation.

    Returns:
        The argument in the requested representation.

    """
    if representation == "full":
        return sparse_array
    elif representation == "diagonal":
        return diagonal_array
    else:
        return diagonal_array.to_full()


@pytest.mark.parametrize("first", ["full", "diagonal", "to_full"])
@pytest.mark.parametrize("second", ["full", "diagonal", "to_full"])
def test_jacobian_with_arguments_in_different_representations(
    equation_system: pp.ad.EquationSystem, first: str, second: str
):
    """The value and Jacobian of a surrogate operator do not depend on the
    representation of its arguments, which can be full, diagonal or a mix of the two.

    The result is a diagonal Ad array if both arguments are diagonal, and an Ad array
    in the full representation otherwise.

    """
    # Use _surrogate_and_arguments to create a surrogate operator and its arguments in
    # both sparse and diagonal representations. The surrogate operator has a constant
    # value of 1.0, and its derivatives are set to a constant value for each variable,
    # but the first entry is zero (see _surrogate_and_arguments).
    sop, sparse_array, diagonal_array, expected_jacobian = _surrogate_and_arguments(
        equation_system, [VAR1_NAME, VAR2_NAME], [2.0, 3.0]
    )
    num_cells = sparse_array[0].val.size
    args = [
        _argument_in_representation(first, sparse_array[0], diagonal_array[0]),
        _argument_in_representation(second, sparse_array[1], diagonal_array[1]),
    ]

    jac = sop.get_jacobian(*args)
    np.testing.assert_allclose(jac.toarray(), expected_jacobian)

    result = sop.func(*args)
    expected_type = (
        pp.ad.DiagonalAdArray if first == second == "diagonal" else pp.ad.AdArray
    )
    assert isinstance(result, expected_type)
    np.testing.assert_allclose(result.val, np.ones(num_cells))
    np.testing.assert_allclose(result.full_jac.toarray(), expected_jacobian)


def test_diagonal_jacobian_with_variables_not_arguments(
    equation_system: pp.ad.EquationSystem,
):
    """The diagonal Jacobian of a surrogate operator, when the Jacobian structure of its
    arguments also contains variables which are not arguments.

    Setup: The surrogate operator depends on ``var1`` only. The argument ``var1`` is a
    variable, that is, its full Jacobian has a single identity block. However, since
    ``var1`` is initialized jointly with ``var2``, its diagonal Jacobian has two blocks:
    The identity block of ``var1`` and an all-zero block for ``var2``.

    Test: Evaluation of the surrogate operator gives a diagonal Ad array with the same
    two-block structure as its argument. The block of ``var1`` contains the derivatives
    of the surrogate operator, while the block of ``var2`` is zero. The full
    representation equals the expected Jacobian.

    """
    known_derivative_value = 2.0
    # The surrogate operator depends on ``var1`` only, since only this variable is
    # passed as a dependency. Independent of the dependencies, _surrogate_and_arguments
    # initializes the diagonal Ad arrays of ``var1`` and ``var2`` jointly, thus the
    # Jacobian structure of the argument ``var1`` contains a block also for ``var2``.
    sop, sparse_array, diagonal_array, expected_jacobian = _surrogate_and_arguments(
        equation_system, [VAR1_NAME], [known_derivative_value]
    )

    result = sop.func(diagonal_array[0])
    assert isinstance(result, pp.ad.DiagonalAdArray)
    assert result.jac.shape == (2, sparse_array[0].val.size)

    expected_derivatives = known_derivative_value * np.ones(sparse_array[0].val.size)
    expected_derivatives[0] = 0.0
    np.testing.assert_allclose(result.jac[0], expected_derivatives)
    # The derivative with respect to ``var2`` is zero, since the surrogate operator does
    # not depend on ``var2``.
    np.testing.assert_allclose(result.jac[1], 0.0)

    np.testing.assert_allclose(result.full_jac.toarray(), expected_jacobian)
    # The same Jacobian is obtained from ``var1`` in the full representation.
    jac = sop.get_jacobian(sparse_array[0])
    np.testing.assert_allclose(jac.toarray(), expected_jacobian)


@pytest.mark.parametrize("argument", ["scaled", "sum"])
@pytest.mark.parametrize("representation", ["full", "diagonal"])
def test_chain_rule_for_arguments_which_are_not_variables(
    equation_system: pp.ad.EquationSystem, argument: str, representation: str
):
    """The Jacobian of a surrogate operator is assembled by the chain rule, also if its
    argument is not a variable.

    Setup: A surrogate operator depending on a single variable is evaluated with an
    argument whose Jacobian is not an identity block, that is, the argument is not a
    variable.

    Test: The Jacobian is the derivative values of the surrogate operator times the
    Jacobian of the argument, in both representations.

    """
    sop, sparse_array, diagonal_array, _ = _surrogate_and_arguments(
        equation_system, [VAR1_NAME], [2.0]
    )
    var1, var2 = (
        (sparse_array[0], sparse_array[1])
        if representation == "full"
        else (diagonal_array[0], diagonal_array[1])
    )
    arg = 5.0 * var1 if argument == "scaled" else var1 + var2

    # The derivative values as set by _surrogate_and_arguments.
    derivatives = 2.0 * np.ones(arg.val.size)
    derivatives[0] = 0.0
    expected_jacobian = derivatives[:, np.newaxis] * arg.full_jac.toarray()

    np.testing.assert_allclose(sop.get_jacobian(arg).toarray(), expected_jacobian)
    result = sop.func(arg)
    np.testing.assert_allclose(result.val, np.ones(arg.val.size))
    np.testing.assert_allclose(result.full_jac.toarray(), expected_jacobian)


@pytest.mark.parametrize("representation", ["full", "diagonal"])
def test_repeated_argument(equation_system: pp.ad.EquationSystem, representation: str):
    """The derivatives of a surrogate operator with respect to a repeated argument add
    up, in both representations."""
    sop, sparse_array, diagonal_array, expected_jacobian = _surrogate_and_arguments(
        equation_system, [VAR1_NAME, VAR1_NAME], [2.0, 3.0]
    )
    arg = sparse_array[0] if representation == "full" else diagonal_array[0]
    np.testing.assert_allclose(sop.get_jacobian(arg, arg).toarray(), expected_jacobian)
    np.testing.assert_allclose(sop.func(arg, arg).full_jac.toarray(), expected_jacobian)


def test_arguments_which_are_not_ad_arrays(equation_system: pp.ad.EquationSystem):
    """Arguments which are not Ad arrays do not contribute to the Jacobian.

    Without Ad arrays among the arguments, the surrogate operator evaluates to its
    values, and the Jacobian cannot be assembled.

    """
    sop, sparse_array, _, expected_jacobian = _surrogate_and_arguments(
        equation_system, [VAR1_NAME, VAR2_NAME], [2.0, 3.0]
    )
    num_cells = sparse_array[0].val.size
    # Only the derivatives with respect to var2 remain, since var1 is not an Ad array.
    var2_dofs = equation_system.dofs_of([VAR2_NAME])
    expected = np.zeros_like(expected_jacobian)
    expected[:, var2_dofs] = expected_jacobian[:, var2_dofs]

    result = sop.func(sparse_array[0].val, sparse_array[1])
    np.testing.assert_allclose(result.full_jac.toarray(), expected)

    values = sop.func(sparse_array[0].val, sparse_array[1].val)
    assert isinstance(values, np.ndarray)
    np.testing.assert_allclose(values, np.ones(num_cells))
    with pytest.raises(ValueError):
        sop.get_jacobian(sparse_array[0].val, sparse_array[1].val)


def test_number_of_arguments_must_match_dependencies(
    equation_system: pp.ad.EquationSystem,
):
    """A surrogate operator raises an error if the number of arguments differs from the
    number of its dependencies."""
    sop, sparse_array, _, _ = _surrogate_and_arguments(
        equation_system, [VAR1_NAME, VAR2_NAME], [2.0, 3.0]
    )
    with pytest.raises(ValueError):
        sop.func(sparse_array[0])
    with pytest.raises(ValueError):
        sop.get_jacobian(*sparse_array, sparse_array[0])


def test_separately_initialized_diagonal_arguments(
    equation_system: pp.ad.EquationSystem,
):
    """The Jacobian of a surrogate operator is correct also for diagonal arguments that
    were initialized separately, and thus do not share a Jacobian structure. The result
    is then in the full representation."""
    sop, _, diagonal_array, expected_jacobian = _surrogate_and_arguments(
        equation_system, [VAR1_NAME, VAR2_NAME], [2.0, 3.0]
    )
    num_dofs = equation_system.num_dofs()
    args = [
        pp.ad.initialize_diagonal_ad_arrays([arg.val], [arg.row_indices], num_dofs)[0]
        for arg in diagonal_array
    ]
    result = sop.func(*args)
    assert isinstance(result, pp.ad.AdArray)
    np.testing.assert_allclose(result.full_jac.toarray(), expected_jacobian)
