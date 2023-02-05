"""Test creation and composition of Ad operators and their trees.

The main checks performed are:
    test_elementary_operations: Checks of basic arithmetic operations are correctly
        implemented.
    test_copy_operator_tree: Test of functionality under copy.copy and deepcopy.
    test_elementary_wrappers: Test wrapping of scalars, arrays and matrices.
    test_ad_variable_creation: Generate variables, check that copies and new variables
        are returned as expected.
    test_ad_variable_evaluation: Test of the wrappers for variables.
    test_time_differentiation: Covers the pp.ad.dt operator.

"""
import copy

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp

_operations = pp.ad.operators.Operator.Operations

operators = [
    ("+", _operations.add),
    ("-", _operations.sub),
    ("*", _operations.mul),
    ("/", _operations.div),
    ("**", _operations.pow),
]


@pytest.mark.parametrize("operator", operators)
def test_elementary_operations(operator):
    """Test that performing elementary arithmetic operations on operators return operator
    trees with the expected structure.

    The test does not consider evaluation of the numerical values of the operators.
    """
    # Generate two generic operators
    a = pp.ad.Operator()
    b = pp.ad.Operator()

    # Combine the operators with the provided operation.
    c = eval(f"a {operator[0]} b")

    # Check that the combined operator has the expected structure.
    tree = c.tree
    assert tree.op == operator[1]

    assert tree.children[0] == a
    assert tree.children[1] == b


def test_copy_operator_tree():
    """Test that copying of an operator tree works as expected.

    The test makes a simple tree by combining a scalar and a numpy array. The intention
    is to use this as a test of copying trees, while copying of individual operators
    should be done elsewhere.

    """
    # To verify the difference between copy and deepcopy, keep pointers to the data
    # structures to be wrapped
    a_val = 42
    a = pp.ad.Scalar(a_val)

    b_arr = np.arange(3)
    b = pp.ad.Array(b_arr)

    # The combined operator, and two copies
    c = a + b
    c_copy = copy.copy(c)
    c_deepcopy = copy.deepcopy(c)

    # First check that the two copies have behaved as they should.
    # The operators should be the same for all trees.
    assert c.tree.op == c_copy.tree.op
    assert c.tree.op == c_deepcopy.tree.op

    # The deep copy should have made a new list of children.
    # The shallow copy should have the same list.
    assert c.tree.children == c_copy.tree.children
    assert not c.tree.children == c_deepcopy.tree.children

    # Check that the shallow copy also has the same children, while
    # the deep copy has copied these as well.
    for c1, c2 in zip(c.tree.children, c_copy.tree.children):
        assert c1 == c2
    for c1, c2 in zip(c.tree.children, c_deepcopy.tree.children):
        assert not c1 == c2

    # As a second test, also validate that the operators are parsed correctly.
    # This should not be strictly necessary - the above test should be sufficient,
    # but better safe than sorry.
    # Some boilerplate is needed before the expression can be evaluated.
    mdg, _ = pp.grids.standard_grids.md_grids_2d.single_horizontal()
    eq_system = pp.ad.EquationSystem(mdg)
    eq_system.create_variables("foo", {"cells": 1}, mdg.subdomains())
    eq_system.set_variable_values(
        np.zeros(eq_system.num_dofs()), to_iterate=True, to_state=True
    )

    # In their initial state, all operators should have the same values
    assert np.allclose(c.evaluate(eq_system), c_copy.evaluate(eq_system))
    assert np.allclose(c.evaluate(eq_system), c_deepcopy.evaluate(eq_system))

    # Increase the value of the scalar. This should have no effect, since the scalar
    # wrapps an immutable, see comment in pp.ad.Scalar
    a_val += 1
    assert np.allclose(c.evaluate(eq_system), c_copy.evaluate(eq_system))
    assert np.allclose(c.evaluate(eq_system), c_deepcopy.evaluate(eq_system))

    # Increase the value of the Scalar. This will be seen by the copy, but not the
    # deepcopy.
    a._value += 1
    assert np.allclose(c.evaluate(eq_system), c_copy.evaluate(eq_system))
    assert not np.allclose(c.evaluate(eq_system), c_deepcopy.evaluate(eq_system))

    # Next increase the values in the array. This changes the shallow copy, but not the
    # deep one.
    b_arr += 1
    assert np.allclose(c.evaluate(eq_system), c_copy.evaluate(eq_system))
    assert not np.allclose(c.evaluate(eq_system), c_deepcopy.evaluate(eq_system))


## Test of pp.ad.Matrix, pp.ad.Array, pp.ad.Scalar
fields = [
    (pp.ad.Matrix, sps.csr_matrix(np.random.rand(3, 2))),
    (pp.ad.Array, np.random.rand(3)),
    (pp.ad.Scalar, 42),
]


@pytest.mark.parametrize("field", fields)
def test_elementary_wrappers(field):
    """Test the creation and parsing of the Ad wrappers of standard numerical
    objects (scalars, numpy arrays, sparse matrices).

    The test takes a standard object, wraps it in the given Ad class, and verifies
    that parsing returns the expected object.

    Also test the behavior of the classes under copying.

    """
    obj = field[1]
    wrapped_obj = field[0](obj, name="foo")

    # Evaluate the Ad wrapper using parse, which will act directly on the wrapper
    # (as oposed to evaluate, which will invoke the full evaluation machinery of the
    # ad operator tree)

    # We can use None here, since the MixedDimensionalGrid is not used for parsing of
    # these wrappers.
    stored_obj = wrapped_obj.parse(None)

    def compare(one, other):
        if isinstance(one, np.ndarray):
            return np.allclose(one, other)
        elif isinstance(one, sps.spmatrix):  # sparse matrix
            return np.allclose(one.data, other.data)
        else:  # scalar
            return one == other

    assert compare(obj, stored_obj)

    # Create two copies of the object, using respectively copy and deep copy
    wrapped_copy = copy.copy(wrapped_obj)
    wrapped_deep_copy = copy.deepcopy(wrapped_obj)

    # Both the shallow and deep copy should evaluate to the same quantity
    assert compare(obj, wrapped_copy.parse(None))
    assert compare(obj, wrapped_deep_copy.parse(None))

    # Next modify the value of the underlying object.
    # This must be done in slightly different ways, depending on the implementation of
    # the underlying data structures.
    if isinstance(obj, sps.spmatrix):
        obj[0, 0] += 1
    else:
        obj += 1.0

    # The shallow copy of the Ad quantity should also pick up the modification.
    # The exception is the scalar, which is a wrapper of a Python immutable, and thus
    # does not copy by reference.
    if isinstance(wrapped_obj, pp.ad.Scalar):
        assert not compare(obj, wrapped_copy.parse(None))
    else:
        assert compare(obj, wrapped_copy.parse(None))

    # The deep copy should differ independent of which type of Ad quantity this is
    assert not compare(obj, wrapped_deep_copy.parse(None))


def test_time_dependent_array():
    """Test of time-dependent arrays (wrappers around numpy arrays)."""

    # Time-dependent arrays are defined on grids.
    # Some boilerplate is needed to define these.
    mdg, _ = pp.grids.standard_grids.md_grids_2d.single_horizontal()
    for sd, sd_data in mdg.subdomains(return_data=True):
        sd_data[pp.STATE] = {
            "foo": np.zeros(sd.num_cells),
            pp.ITERATE: {"foo": sd.dim * np.ones(sd.num_cells)},
        }
    for intf, intf_data in mdg.interfaces(return_data=True):
        # Create an empty primary variable list
        intf_data[pp.STATE] = {
            "bar": np.arange(intf.num_cells),
            pp.ITERATE: {"bar": np.ones(intf.num_cells)},
        }

    # We make three arrays: One defined on a single subdomain, one on all subdomains of mdg
    # and one on an interface.
    sd_array_top = pp.ad.TimeDependentArray(
        "foo", subdomains=mdg.subdomains(dim=mdg.dim_max())
    )
    sd_array = pp.ad.TimeDependentArray("foo", subdomains=mdg.subdomains())
    intf_array = pp.ad.TimeDependentArray("bar", interfaces=mdg.interfaces())

    # Evaluate each of the Ad objects, verify that they have the expected values.
    sd_array_top_eval = sd_array_top.parse(mdg)
    assert np.allclose(sd_array_top_eval, 2)

    sd_array_eval = sd_array.parse(mdg)
    # Check the values at the different subdomains separately. This assumes that the
    # subdomains are ordered with the higher dimension first.
    assert np.allclose(sd_array_eval[: sd_array_top_eval.size], 2)
    assert np.allclose(sd_array_eval[sd_array_top_eval.size :], 1)
    # The interface.
    intf_val = intf_array.parse(mdg)
    assert np.allclose(intf_val, 1)

    # Evaluate at previous time steps

    # The value is the same on both subdomains, so we can check them together.
    sd_prev_timestep = sd_array.previous_timestep()
    assert np.allclose(sd_prev_timestep.parse(mdg), 0)

    sd_top_prev_timestep = sd_array_top.previous_timestep()
    assert np.allclose(sd_top_prev_timestep.parse(mdg), 0)

    intf_prev_timestep = intf_array.previous_timestep()
    assert np.allclose(intf_prev_timestep.parse(mdg), np.arange(intf_val.size))

    # Create and evaluate a time-dependent array that is a function of neither
    # subdomains nor interfaces.
    empty_array = pp.ad.TimeDependentArray("none", subdomains=[], interfaces=[])
    # In this case evaluation should return an empty array.
    empty_eval = empty_array.parse(mdg)
    assert empty_eval.size == 0
    # Same with the previous timestep
    empty_prev_timestep = empty_array.previous_timestep()
    assert empty_prev_timestep.parse(mdg).size == 0

    with pytest.raises(ValueError):
        # If we try to define an array on both subdomain and interface, we should get an
        # error.
        pp.ad.TimeDependentArray(
            "foobar", subdomains=mdg.subdomains(), interfaces=mdg.interfaces()
        )


def test_ad_variable_creation():
    """Test creation of Ad variables by way of the EquationSystem.
    1) Fetching the same variable twice should get the same variable (same attribute id).
    2) Fetching the same mixed-dimensional variable twice should result in objects with different
       id attributes, but point to the same underlying variable.

    No tests are made of the actual values of the variables, as this is tested in
    test_ad_variable_evaluation() (below).

    """
    mdg, _ = pp.grids.standard_grids.md_grids_2d.single_horizontal()
    eq_system = pp.ad.EquationSystem(mdg)
    eq_system.create_variables("foo", {"cells": 1}, mdg.subdomains())

    var_1 = eq_system.get_variables(["foo"], mdg.subdomains(dim=mdg.dim_max()))[0]
    var_2 = eq_system.get_variables(["foo"], mdg.subdomains(dim=mdg.dim_max()))[0]
    var_3 = eq_system.get_variables(["foo"], mdg.subdomains(dim=mdg.dim_min()))[0]

    # Fetching the same variable twice should give the same variable (idetified by the
    # variable id)
    assert var_1.id == var_2.id
    # A variable with the same name, but on a different grid should have a different id
    assert var_1.id != var_3.id

    # Fetch mixed-dimensional variable representations of the same variables
    mvar_1 = eq_system.md_variable("foo", mdg.subdomains(dim=mdg.dim_max()))
    mvar_2 = eq_system.md_variable("foo", mdg.subdomains(dim=mdg.dim_max()))

    # The two mixed-dimensional variables should have different ids
    assert mvar_2.id != mvar_1.id
    # The id of the merged and atomic variables should be different
    assert mvar_1.id != var_1
    # The underlying variables should have the same id.
    assert mvar_1.sub_vars[0].id == mvar_2.sub_vars[0].id

    ## Test of variable ids under copying

    # First test variables
    var_1_copy = copy.copy(var_1)
    var_1_deepcopy = copy.deepcopy(var_1)

    # The copy model will return variables with the same id.
    assert var_1_copy.id == var_1.id
    assert var_1_deepcopy.id == var_1.id

    # Next merged test variables
    mvar_1_copy = copy.copy(mvar_1)
    mvar_1_deepcopy = copy.deepcopy(mvar_1)

    # The copy model will return variables with the same id.
    assert mvar_1_copy.id == mvar_1.id
    assert mvar_1_deepcopy.id == mvar_1.id

    # Get versions of the variables at previous iteration and time step.
    # This should return variables with different ids

    # First variables
    var_1_prev_iter = var_1.previous_iteration()
    var_1_prev_time = var_1.previous_timestep()
    assert var_1_prev_iter.id != var_1.id
    assert var_1_prev_time.id != var_1.id

    # Then mixed-dimensional variables.
    mvar_1_prev_iter = mvar_1.previous_iteration()
    mvar_1_prev_time = mvar_1.previous_timestep()
    assert mvar_1_prev_iter.id != mvar_1.id
    assert mvar_1_prev_time.id != mvar_1.id


def test_ad_variable_evaluation():
    """Test that the values of Ad variables are as expected under evalutation
    (translation from the abstract Ad framework to forward mode).

    Boththe atomic and mixed-dimensional variables are tested. The tests cover both the
    current values of the variables (pp.ITERATE), and their values at previous
    iterations and time steps.

    See also test_variable_combinations, which specifically tests evaluation of
    variables in a setting of multiple variables, including mixed-dimensional variables.

    """
    # Create a MixedDimensionalGrid with two fractures.
    fracs = [np.array([[0, 2], [1, 1]]), np.array([[1, 1], [0, 2]])]
    mdg = pp.meshing.cart_grid(fracs, np.array([2, 2]))

    state_map = {}
    iterate_map = {}

    state_map_2, iterate_map_2 = {}, {}

    var = "foo"
    var2 = "bar"

    mortar_var = "mv"

    def _compare_ad_objects(a, b):
        # Helper function to compare two Ad objects. The comparison is done by
        # comparing the values of the underlying variables, and the values of the
        # derivatives.
        va, ja = a.val, a.jac
        vb, jb = b.val, b.jac

        assert np.allclose(va, vb)
        assert ja.shape == jb.shape
        d = ja - jb
        if d.data.size > 0:
            assert np.max(np.abs(d.data)) < 1e-10

    eq_system = pp.EquationSystem(mdg)
    # First create a variable on the subdomains. The number of dofs is different for the
    # different subdomains.
    # NOTE: The order of creation is a bit important here: We will iterate of the
    # subdomains (implicitly sorted by dimension) and assign a value to the variables.
    # The order of creation should therefore be consistent with the order of iteration.
    # It should be possible to avoid this by using dof-indices of the subdomains, but EK
    # cannot wrap his head around this at the moment (it is Friday afternoon).
    eq_system.create_variables(
        var, dof_info={"cells": 1}, subdomains=mdg.subdomains(dim=2)
    )
    eq_system.create_variables(
        var, dof_info={"cells": 2}, subdomains=mdg.subdomains(dim=1)
    )
    eq_system.create_variables(
        var, dof_info={"cells": 1}, subdomains=mdg.subdomains(dim=0)
    )
    eq_system.create_variables(
        var2, dof_info={"cells": 1}, subdomains=mdg.subdomains(dim=2)
    )
    # Next create interface variables.
    eq_system.create_variables(
        mortar_var, dof_info={"cells": 2}, interfaces=mdg.interfaces(dim=1)
    )
    eq_system.create_variables(
        mortar_var, dof_info={"cells": 1}, interfaces=mdg.interfaces(dim=0)
    )

    for sd, data in mdg.subdomains(return_data=True):
        if sd.dim == 1:
            num_dofs = 2
        else:
            num_dofs = 1

        data[pp.PRIMARY_VARIABLES] = {var: {"cells": num_dofs}}

        val_state = np.random.rand(sd.num_cells * num_dofs)
        val_iterate = np.random.rand(sd.num_cells * num_dofs)

        data[pp.STATE] = {var: val_state, pp.ITERATE: {var: val_iterate}}
        state_map[sd] = val_state
        iterate_map[sd] = val_iterate

        # Add a second variable to the 2d grid, just for the fun of it
        if sd.dim == 2:
            data[pp.PRIMARY_VARIABLES][var2] = {"cells": 1}
            val_state = np.random.rand(sd.num_cells)
            val_iterate = np.random.rand(sd.num_cells)
            data[pp.STATE][var2] = val_state
            data[pp.STATE][pp.ITERATE][var2] = val_iterate
            state_map_2[sd] = val_state
            iterate_map_2[sd] = val_iterate

    for intf, data in mdg.interfaces(return_data=True):
        if intf.dim == 1:
            num_dofs = 2
        else:
            num_dofs = 1

        data[pp.PRIMARY_VARIABLES] = {mortar_var: {"cells": num_dofs}}

        val_state = np.random.rand(intf.num_cells * num_dofs)
        val_iterate = np.random.rand(intf.num_cells * num_dofs)

        data[pp.STATE] = {mortar_var: val_state, pp.ITERATE: {mortar_var: val_iterate}}
        state_map[intf] = val_state
        iterate_map[intf] = val_iterate

    # Manually assemble state and iterate
    true_state = np.zeros(eq_system.num_dofs())
    true_iterate = np.zeros(eq_system.num_dofs())

    # Also a state array that differs from the stored iterates
    double_iterate = np.zeros(eq_system.num_dofs())

    for v in eq_system.variables:
        g = v.domain
        inds = eq_system.dofs_of([v])
        if v.name == var2:
            true_state[inds] = state_map_2[g]
            true_iterate[inds] = iterate_map_2[g]
            double_iterate[inds] = 2 * iterate_map_2[g]
        else:
            true_state[inds] = state_map[g]
            true_iterate[inds] = iterate_map[g]
            double_iterate[inds] = 2 * iterate_map[g]

    subdomains = [
        mdg.subdomains(dim=2)[0],
        *mdg.subdomains(dim=1),
        mdg.subdomains(dim=0)[0],
    ]

    # Generate mixed-dimensional variables via the EquationManager.
    var_ad = eq_system.md_variable(var, subdomains)

    # Check equivalence between the two approaches to generation.

    # Check that the state is correctly evaluated.
    inds_var = np.hstack(
        [eq_system.dofs_of(eq_system.get_variables([var], [g])) for g in subdomains]
    )
    assert np.allclose(
        true_iterate[inds_var], var_ad.evaluate(eq_system, true_iterate).val
    )

    # Check evaluation when no state is passed to the parser, and information must
    # instead be glued together from the MixedDimensionalGrid
    assert np.allclose(true_iterate[inds_var], var_ad.evaluate(eq_system).val)

    # Evaluate the equation using the double iterate
    assert np.allclose(
        2 * true_iterate[inds_var], var_ad.evaluate(eq_system, double_iterate).val
    )

    # Represent the variable on the previous time step. This should be a numpy array
    prev_var_ad = var_ad.previous_timestep()
    prev_evaluated = prev_var_ad.evaluate(eq_system)
    assert isinstance(prev_evaluated, np.ndarray)
    assert np.allclose(true_state[inds_var], prev_evaluated)

    # Also check that state values given to the ad parser are ignored for previous
    # values
    assert np.allclose(prev_evaluated, prev_var_ad.evaluate(eq_system, double_iterate))

    ## Next, test edge variables. This should be much the same as the grid variables,
    # so the testing is less thorough.
    # Form an edge variable, evaluate this
    interfaces = [intf for intf in mdg.interfaces()]
    variable_interfaces = [
        eq_system.md_variable(mortar_var, [intf]) for intf in interfaces
    ]

    interface_inds = np.hstack(
        [eq_system.dofs_of([var]) for var in variable_interfaces]
    )
    interface_values = np.hstack(
        [var.evaluate(eq_system, true_iterate).val for var in variable_interfaces]
    )
    assert np.allclose(
        true_iterate[interface_inds],
        interface_values,
    )

    # Finally, test a single variable; everything should work then as well
    g = mdg.subdomains(dim=2)[0]
    v1 = eq_system.get_variables([var], [g])[0]
    v2 = eq_system.get_variables([var2], [g])[0]

    ind1 = eq_system.dofs_of(eq_system.get_variables([var], [g]))
    ind2 = eq_system.dofs_of(eq_system.get_variables([var2], [g]))

    assert np.allclose(true_iterate[ind1], v1.evaluate(eq_system, true_iterate).val)
    assert np.allclose(true_iterate[ind2], v2.evaluate(eq_system, true_iterate).val)

    v1_prev = v1.previous_timestep()
    assert np.allclose(true_state[ind1], v1_prev.evaluate(eq_system, true_iterate))


@pytest.mark.parametrize(
    "grids",
    [
        [pp.CartGrid(np.array([4, 1]))],
        [pp.CartGrid(np.array([4, 1])), pp.CartGrid(np.array([2, 2]))],
    ],
)
@pytest.mark.parametrize(
    "variables",
    [["foo"], ["foo", "bar"]],
)
def test_variable_combinations(grids, variables):
    """Test combinations of variables, and mixed-dimensional variables, on different grids.
    The main check is if Jacobian matrices are of the right size.
    """
    # Make MixedDimensionalGrid, populate with necessary information
    mdg = pp.MixedDimensionalGrid()
    mdg.add_subdomains(grids)
    for sd, data in mdg.subdomains(return_data=True):
        data[pp.STATE] = {}
        data[pp.PRIMARY_VARIABLES] = {}
        for var in variables:
            data[pp.PRIMARY_VARIABLES].update({var: {"cells": 1}})
            data[pp.STATE][var] = np.random.rand(sd.num_cells)

    # Ad boilerplate
    eq_system = pp.ad.EquationSystem(mdg)
    for var in variables:
        eq_system.create_variables(var, {"cells": 1}, mdg.subdomains())
        eq_system.set_variable_values(
            np.random.rand(mdg.num_subdomain_cells()),
            [var],
            to_iterate=True,
            to_state=True,
        )
    # Standard Ad variables
    ad_vars = eq_system.get_variables()
    # Merge variables over all grids
    merged_vars = [eq_system.md_variable(var, grids) for var in variables]

    # First check of standard variables. If this fails, something is really wrong
    for sd in grids:
        data = mdg.subdomain_data(sd)
        for var in ad_vars:
            if sd == var.domain:
                expr = var.evaluate(eq_system)
                # Check that the size of the variable is correct
                assert np.allclose(expr.val, data[pp.STATE][var.name])
                # Check that the Jacobian matrix has the right number of columns
                assert expr.jac.shape[1] == eq_system.num_dofs()

    # Next, check that mixed-dimensional variables are handled correctly.
    for var in merged_vars:
        expr = var.evaluate(eq_system)
        vals = []
        for sub_var in var.sub_vars:
            vals.append(mdg.subdomain_data(sub_var.domain)[pp.STATE][sub_var.name])

        assert np.allclose(expr.val, np.hstack([v for v in vals]))
        assert expr.jac.shape[1] == eq_system.num_dofs()

    # Finally, check that the size of the Jacobian matrix is correct when combining
    # variables (this will cover both variables and mixed-dimensional variable with the same name,
    # and with different name).
    for sd in grids:
        for var in ad_vars:
            nc = var.size()
            cols = np.arange(nc)
            data = np.ones(nc)
            for mv in merged_vars:
                nr = mv.size()

                # The variable must be projected to the full set of grid for addition
                # to be meaningful. This requires a bit of work.
                sv_size = np.array([sv.size() for sv in mv.sub_vars])
                mv_grids = [sv._g for sv in mv.sub_vars]
                ind = mv_grids.index(var._g)
                offset = np.hstack((0, np.cumsum(sv_size)))[ind]
                rows = offset + np.arange(nc)
                P = pp.ad.Matrix(sps.coo_matrix((data, (rows, cols)), shape=(nr, nc)))

                eq = eq = mv + P * var
                expr = eq.evaluate(eq_system)
                # Jacobian matrix size is set according to the dof manager,
                assert expr.jac.shape[1] == eq_system.num_dofs()


def test_time_differentiation():
    """Test the dt and time_difference functions in AD.

    For the moment, this is simply a test that backward Euler is correctly implemented.

    All checks are run on the time differentiation. Some checks are also run on the
    time_difference method, however, since the dt-function is a simple extension, full
    testing (which would have required a lot of code duplication) is not done.
    """

    # Create a MixedDimensionalGrid with two subdomains, one interface.
    # The highest-dimensional subdomain has both a variable and a time-dependent array,
    # while the lower-dimensional subdomain has only a variable.
    mdg, _ = pp.grids.standard_grids.md_grids_2d.single_horizontal()
    for sd, sd_data in mdg.subdomains(return_data=True):
        if sd.dim == mdg.dim_max():
            sd_data[pp.STATE] = {
                "foo": -np.ones(sd.num_cells),
                "bar": 2 * np.ones(sd.num_cells),
                pp.ITERATE: {
                    "foo": 3 * np.ones(sd.num_cells),
                    "bar": np.ones(sd.num_cells),
                },
            }
        else:
            sd_data[pp.STATE] = {
                "foo": np.zeros(sd.num_cells),
                pp.ITERATE: {"foo": np.ones(sd.num_cells)},
            }

    for intf, intf_data in mdg.interfaces(return_data=True):
        # Create an empty primary variable list
        intf_data[pp.PRIMARY_VARIABLES] = {}
        # Set a numpy array in state, to be represented as a time-dependent array.
        intf_data[pp.STATE] = {
            "foobar": np.ones(intf.num_cells),
            pp.ITERATE: {"foobar": 2 * np.ones(intf.num_cells)},
        }

    eq_system = pp.ad.EquationSystem(mdg)
    eq_system.create_variables("foo", {"cells": 1}, mdg.subdomains())
    # The time step, represented as a scalar.
    ts = 2
    time_step = pp.ad.Scalar(ts)

    # Differentiate the variable on the highest-dimensional subdomain
    sd = mdg.subdomains(dim=mdg.dim_max())[0]
    var_1 = eq_system.get_variables(["foo"], [sd])[0]
    dt_var_1 = pp.ad.dt(var_1, time_step)
    assert np.allclose(dt_var_1.evaluate(eq_system).val, 2)

    # Also test the time difference function
    diff_var_1 = pp.ad.time_increment(var_1)
    assert np.allclose(diff_var_1.evaluate(eq_system).val, 2 * ts)

    # Differentiate the time dependent array residing on the subdomain
    array = pp.ad.TimeDependentArray(name="bar", subdomains=[sd])
    dt_array = pp.ad.dt(array, time_step)
    assert np.allclose(dt_array.evaluate(eq_system), -0.5)

    # Combine the parameter array and the variable. This is a test that operators that
    # are not leaves are differentiated correctly.
    var_array = var_1 * array
    dt_var_array = pp.ad.dt(var_array, time_step)
    assert np.allclose(dt_var_array.evaluate(eq_system).val, 2.5)
    # Also test the time increment function
    diff_var_array = pp.ad.time_increment(var_array)
    assert np.allclose(diff_var_array.evaluate(eq_system).val, 2.5 * ts)

    # For good measure, add one more level of combination.
    var_array_2 = var_array + var_array
    dt_var_array = pp.ad.dt(var_array_2, time_step)
    assert np.allclose(dt_var_array.evaluate(eq_system).val, 5)

    # Also do a test of the mixed-dimensional variable.
    mvar = eq_system.md_variable("foo", [sd])

    dt_mvar = pp.ad.dt(mvar, time_step)
    assert np.allclose(dt_mvar.evaluate(eq_system).val[: sd.num_cells], 2)
    assert np.allclose(dt_mvar.evaluate(eq_system).val[sd.num_cells :], 0.5)

    # Test the time increment function
    diff_mvar = pp.ad.time_increment(mvar)
    assert np.allclose(diff_mvar.evaluate(eq_system).val[: sd.num_cells], 2 * ts)
    assert np.allclose(diff_mvar.evaluate(eq_system).val[sd.num_cells :], ts)

    # Make a combined operator with the mixed-dimensional variable, test this.
    dt_mvar = pp.ad.dt(mvar * mvar, time_step)
    assert np.allclose(dt_mvar.evaluate(eq_system).val[: sd.num_cells], 4)
    assert np.allclose(dt_mvar.evaluate(eq_system).val[sd.num_cells :], 0.5)

    # Finally create a variable at the previous time step. Its derivative should be
    # zero.
    var_2 = var_1.previous_timestep()
    dt_var_2 = pp.ad.dt(var_2, time_step)
    assert np.allclose(dt_var_2.evaluate(eq_system), 0)
    # Also test the time increment method
    diff_var_2 = pp.ad.time_increment(var_2)
    assert np.allclose(diff_var_2.evaluate(eq_system), 0)
