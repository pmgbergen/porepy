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
    """Test that performing elementary arithmetic operations on operators return
    operator trees with the expected structure.

    The test does not consider evaluation of the numerical values of the operators.
    """
    # Generate two generic operators
    a = pp.ad.Operator()
    b = pp.ad.Operator()

    # Combine the operators with the provided operation.
    c = eval(f"a {operator[0]} b")

    # Check that the combined operator has the expected structure.
    assert c.operation == operator[1]

    assert c.children[0] == a
    assert c.children[1] == b


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
    b = pp.ad.DenseArray(b_arr)

    # The combined operator, and two copies
    c = a + b
    c_copy = copy.copy(c)
    c_deepcopy = copy.deepcopy(c)

    # First check that the two copies have behaved as they should.
    # The operators should be the same for all trees.
    assert c.operation == c_copy.operation
    assert c.operation == c_deepcopy.operation

    # The deep copy should have made a new list of children.
    # The shallow copy should have the same list.
    assert c.children == c_copy.children
    assert not c.children == c_deepcopy.children

    # Check that the shallow copy also has the same children, while
    # the deep copy has copied these as well.
    for c1, c2 in zip(c.children, c_copy.children):
        assert c1 == c2
    for c1, c2 in zip(c.children, c_deepcopy.children):
        assert not c1 == c2

    # As a second test, also validate that the operators are parsed correctly.
    # This should not be strictly necessary - the above test should be sufficient,
    # but better safe than sorry.
    # Some boilerplate is needed before the expression can be evaluated.
    mdg, _ = pp.grids.standard_grids.md_grids_2d.single_horizontal()
    eq_system = pp.ad.EquationSystem(mdg)
    eq_system.create_variables("foo", {"cells": 1}, mdg.subdomains())
    eq_system.set_variable_values(
        np.zeros(eq_system.num_dofs()), iterate_index=0, time_step_index=0
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


## Test of pp.ad.SparseArray, pp.ad.DenseArray, pp.ad.Scalar
fields = [
    (pp.ad.SparseArray, sps.csr_matrix(np.random.rand(3, 2))),
    (pp.ad.DenseArray, np.random.rand(3)),
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


@pytest.mark.parametrize("field", fields)
def test_ad_arrays_unary_minus_parsing(field):
    """Check that __neg__ works as intended for SparseArrays, DenseArrays and Scalars.

    The objects are wrapped in the respective Ad classes, the __neg__ methods are used,
    and it is tested whether parsing returns the expected object.

    """
    obj = field[1]
    wrapped_obj = -field[0](obj, name="foo")  # Using the __neg__ method
    stored_obj = wrapped_obj.parse(None)

    def compare(one, other):
        if isinstance(one, np.ndarray):
            return np.allclose(-one, other)
        elif isinstance(one, sps.spmatrix):
            return np.allclose(-one.data, other.data)
        else:  # Scalar
            return -one == other

    assert compare(obj, stored_obj)


def test_ad_operator_unary_minus_parsing():
    """Check that __neg__ works as intended for a non-trivial pp.ad.Operator object.

    This is done by summing two SparseArray objects to form an Operator object.

    """
    mat1 = sps.csr_matrix(np.random.rand(3))
    mat2 = sps.csr_matrix(np.random.rand(3))
    sp_array1 = pp.ad.SparseArray(mat1)
    sp_array2 = pp.ad.SparseArray(mat2)
    op = sp_array1 + sp_array2
    assert np.allclose(op._parse_operator(-op, None).data, -(mat1 + mat2).data)


def test_time_dependent_array():
    """Test of time-dependent arrays (wrappers around numpy arrays)."""

    # Time-dependent arrays are defined on grids.
    # Some boilerplate is needed to define these.
    mdg, _ = pp.grids.standard_grids.md_grids_2d.single_horizontal()
    for sd, sd_data in mdg.subdomains(return_data=True):
        vals_sol = np.zeros(sd.num_cells)
        pp.set_solution_values(
            name="foo", values=vals_sol, data=sd_data, time_step_index=0
        )

        vals_it = sd.dim * np.ones(sd.num_cells)
        pp.set_solution_values(
            name="foo", values=vals_it, data=sd_data, iterate_index=0
        )

    for intf, intf_data in mdg.interfaces(return_data=True):
        # Create an empty primary variable list
        vals_sol = np.arange(intf.num_cells)
        pp.set_solution_values(
            name="bar", values=vals_sol, data=intf_data, time_step_index=0
        )

        vals_it = np.ones(intf.num_cells)
        pp.set_solution_values(
            name="bar", values=vals_it, data=intf_data, iterate_index=0
        )

    for bg, bg_data in mdg.boundaries(return_data=True):
        vals_sol = np.arange(bg.num_cells)
        pp.set_solution_values(
            name="foobar", values=vals_sol, data=bg_data, time_step_index=0
        )

        vals_it = np.ones(bg.num_cells) * bg.parent.dim
        pp.set_solution_values(
            name="foobar", values=vals_it, data=bg_data, iterate_index=0
        )

    # We make three arrays: One defined on a single subdomain, one on all subdomains of
    # mdg and one on an interface.
    sd_array_top = pp.ad.TimeDependentDenseArray(
        "foo", domains=mdg.subdomains(dim=mdg.dim_max())
    )
    sd_array = pp.ad.TimeDependentDenseArray("foo", domains=mdg.subdomains())
    intf_array = pp.ad.TimeDependentDenseArray("bar", domains=mdg.interfaces())
    bg_array = pp.ad.TimeDependentDenseArray("foobar", domains=mdg.boundaries())

    # Check correct domain types
    assert sd_array.domain_type == sd_array_top.domain_type == "subdomains"
    assert intf_array.domain_type == "interfaces"
    assert bg_array.domain_type == "boundary grids"

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
    # Boundary grids.
    bg_sizes = [bg.num_cells for bg in mdg.boundaries()]
    bg_val = bg_array.parse(mdg)
    assert np.allclose(bg_val[: bg_sizes[0]], 2)
    assert np.allclose(bg_val[bg_sizes[0] :], 1)

    # Evaluate at previous time steps

    # The value is the same on both subdomains, so we can check them together.
    sd_prev_timestep = sd_array.previous_timestep()
    assert np.allclose(sd_prev_timestep.parse(mdg), 0)

    sd_top_prev_timestep = sd_array_top.previous_timestep()
    assert np.allclose(sd_top_prev_timestep.parse(mdg), 0)

    intf_prev_timestep = intf_array.previous_timestep()
    assert np.allclose(intf_prev_timestep.parse(mdg), np.arange(intf_val.size))

    bg_val_prev_timestep = bg_array.previous_timestep().parse(mdg)
    assert np.allclose(bg_val_prev_timestep[: bg_sizes[0]], np.arange(bg_sizes[0]))
    assert np.allclose(bg_val_prev_timestep[bg_sizes[0] :], np.arange(bg_sizes[1]))

    # Create and evaluate a time-dependent array that is a function of neither
    # subdomains nor interfaces.
    empty_array = pp.ad.TimeDependentDenseArray("none", domains=[])
    # In this case evaluation should return an empty array.
    empty_eval = empty_array.parse(mdg)
    assert empty_eval.size == 0
    # Same with the previous timestep
    empty_prev_timestep = empty_array.previous_timestep()
    assert empty_prev_timestep.parse(mdg).size == 0

    with pytest.raises(ValueError):
        # If we try to define an array on both subdomain and interface, we should get an
        # error.
        pp.ad.TimeDependentDenseArray(
            "foofoobar", domains=[*mdg.subdomains(), *mdg.interfaces()]
        )


def test_ad_variable_creation():
    """Test creation of Ad variables by way of the EquationSystem.
    1) Fetching the same variable twice should get the same variable (same attribute id).
    2) Fetching the same mixed-dimensional variable twice should result in objects with
       different id attributes, but point to the same underlying variable.

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

    # We prohibit creating a variable both on previous time step and iter.
    with pytest.raises(ValueError):
        _ = mvar_1_prev_iter.previous_timestep()
    with pytest.raises(ValueError):
        _ = mvar_1_prev_time.previous_iteration()

    # We prohibit creating a variable on more than one iter or time step behind.
    # NOTE: This should be removed when this feature is implemented.
    with pytest.raises(NotImplementedError):
        _ = mvar_1_prev_iter.previous_iteration()
    with pytest.raises(NotImplementedError):
        _ = mvar_1_prev_time.previous_timestep()


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

        pp.set_solution_values(name=var, values=val_state, data=data, time_step_index=0)
        pp.set_solution_values(name=var, values=val_iterate, data=data, iterate_index=0)

        state_map[sd] = val_state
        iterate_map[sd] = val_iterate

        # Add a second variable to the 2d grid, just for the fun of it
        if sd.dim == 2:
            data[pp.PRIMARY_VARIABLES][var2] = {"cells": 1}
            val_state = np.random.rand(sd.num_cells)
            val_iterate = np.random.rand(sd.num_cells)

            pp.set_solution_values(
                name=var2, values=val_state, data=data, time_step_index=0
            )
            pp.set_solution_values(
                name=var2, values=val_iterate, data=data, iterate_index=0
            )

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

        pp.set_solution_values(
            name=mortar_var, values=val_state, data=data, time_step_index=0
        )
        pp.set_solution_values(
            name=mortar_var, values=val_iterate, data=data, iterate_index=0
        )

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
        data[pp.PRIMARY_VARIABLES] = {}
        for var in variables:
            data[pp.PRIMARY_VARIABLES].update({var: {"cells": 1}})

            vals = np.random.rand(sd.num_cells)
            pp.set_solution_values(name=var, values=vals, data=data, time_step_index=0)

    # Ad boilerplate
    eq_system = pp.ad.EquationSystem(mdg)
    for var in variables:
        eq_system.create_variables(var, {"cells": 1}, mdg.subdomains())
        eq_system.set_variable_values(
            np.random.rand(mdg.num_subdomain_cells()),
            [var],
            time_step_index=0,
            iterate_index=0,
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
                values = pp.get_solution_values(
                    name=var.name, data=data, time_step_index=0
                )
                assert np.allclose(expr.val, values)
                # Check that the Jacobian matrix has the right number of columns
                assert expr.jac.shape[1] == eq_system.num_dofs()

    # Next, check that mixed-dimensional variables are handled correctly.
    for var in merged_vars:
        expr = var.evaluate(eq_system)
        vals = []
        for sub_var in var.sub_vars:
            data = mdg.subdomain_data(sub_var.domain)
            values = pp.get_solution_values(
                name=sub_var.name, data=data, time_step_index=0
            )
            vals.append(values)

        assert np.allclose(expr.val, np.hstack([v for v in vals]))
        assert expr.jac.shape[1] == eq_system.num_dofs()

    # Finally, check that the size of the Jacobian matrix is correct when combining
    # variables (this will cover both variables and mixed-dimensional variable with the
    # same name, and with different name).
    for sd in grids:
        for var in ad_vars:
            nc = var.size
            cols = np.arange(nc)
            data = np.ones(nc)
            for mv in merged_vars:
                nr = mv.size

                # The variable must be projected to the full set of grid for addition
                # to be meaningful. This requires a bit of work.
                sv_size = np.array([sv.size for sv in mv.sub_vars])
                mv_grids = [sv._g for sv in mv.sub_vars]
                ind = mv_grids.index(var._g)
                offset = np.hstack((0, np.cumsum(sv_size)))[ind]
                rows = offset + np.arange(nc)
                P = pp.ad.SparseArray(
                    sps.coo_matrix((data, (rows, cols)), shape=(nr, nc))
                )

                eq = eq = mv + P @ var
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
            vals_sol_foo = -np.ones(sd.num_cells)
            vals_sol_bar = 2 * np.ones(sd.num_cells)

            pp.set_solution_values(
                name="foo", values=vals_sol_foo, data=sd_data, time_step_index=0
            )
            pp.set_solution_values(
                name="bar", values=vals_sol_bar, data=sd_data, time_step_index=0
            )

            vals_it_foo = 3 * np.ones(sd.num_cells)
            vals_it_bar = np.ones(sd.num_cells)

            pp.set_solution_values(
                name="foo", values=vals_it_foo, data=sd_data, iterate_index=0
            )
            pp.set_solution_values(
                name="bar", values=vals_it_bar, data=sd_data, iterate_index=0
            )

        else:
            vals_sol_foo = np.zeros(sd.num_cells)
            vals_it_foo = np.ones(sd.num_cells)

            pp.set_solution_values(
                name="foo", values=vals_sol_foo, data=sd_data, time_step_index=0
            )
            pp.set_solution_values(
                name="foo", values=vals_it_foo, data=sd_data, iterate_index=0
            )

    for intf, intf_data in mdg.interfaces(return_data=True):
        # Create an empty primary variable list
        intf_data[pp.PRIMARY_VARIABLES] = {}
        # Set a numpy array in state, to be represented as a time-dependent array.
        vals_sol = np.ones(intf.num_cells)
        vals_it = 2 * np.ones(intf.num_cells)

        pp.set_solution_values(
            name="foobar", values=vals_sol, data=intf_data, time_step_index=0
        )
        pp.set_solution_values(
            name="foobar", values=vals_it, data=intf_data, iterate_index=0
        )

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
    array = pp.ad.TimeDependentDenseArray(name="bar", domains=[sd])
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



def geometry_information(
    mdg: pp.MixedDimensionalGrid, dim: int
) -> tuple[int, int, int]:
    """Geometry information used in multiple test methods.

    Parameters:
        mdg: Mixed-dimensional grid.
        dim: Dimension. Each of the return values is multiplied by dim.

    Returns:
        n_cells (int): Number of subdomain cells.
        n_faces (int): Number of subdomain faces.
        n_mortar_cells (int): Number of interface cells.
    """
    n_cells = sum([sd.num_cells for sd in mdg.subdomains()]) * dim
    n_faces = sum([sd.num_faces for sd in mdg.subdomains()]) * dim
    n_mortar_cells = sum([intf.num_cells for intf in mdg.interfaces()]) * dim
    return n_cells, n_faces, n_mortar_cells


@pytest.fixture
def mdg():
    """Provide a mixed-dimensional grid for the tests."""
    fracs = [np.array([[0, 2], [1, 1]]), np.array([[1, 1], [0, 2]])]
    md_grid = pp.meshing.cart_grid(fracs, np.array([2, 2]))
    return md_grid


@pytest.mark.parametrize("scalar", [True, False])
def test_subdomain_projections(mdg, scalar):
    """Test of subdomain projections. Both face and cell restriction and prolongation.

    Test three specific cases:
        1. Projections generated by passing a md-grid and a list of grids are identical
        2. All projections for all grids (individually) in a simple md-grid.
        3. Combined projections for list of grids.

    """
    proj_dim = 1 if scalar else mdg.dim_max()
    n_cells, n_faces, _ = geometry_information(mdg, proj_dim)

    subdomains = mdg.subdomains()
    proj = pp.ad.SubdomainProjections(subdomains=subdomains, dim=proj_dim)

    cell_start = np.cumsum(
        np.hstack((0, np.array([sd.num_cells for sd in subdomains])))
    )
    face_start = np.cumsum(
        np.hstack((0, np.array([sd.num_faces for sd in subdomains])))
    )

    # Helper method to get indices for sparse matrices
    def _mat_inds(nc, nf, grid_ind, dim, cell_start, face_start):
        cell_inds = np.arange(cell_start[grid_ind], cell_start[grid_ind + 1])
        face_inds = np.arange(face_start[grid_ind], face_start[grid_ind + 1])

        data_cell = np.ones(nc * dim)
        row_cell = np.arange(nc * dim)
        data_face = np.ones(nf * dim)
        row_face = np.arange(nf * dim)
        col_cell = pp.fvutils.expand_indices_nd(cell_inds, dim)
        col_face = pp.fvutils.expand_indices_nd(face_inds, dim)
        return row_cell, col_cell, data_cell, row_face, col_face, data_face

    # Test projection of one fracture at a time for the full set of grids
    for sd in subdomains:
        ind = _list_ind_of_grid(subdomains, sd)

        nc, nf = sd.num_cells, sd.num_faces

        num_rows_cell = nc * proj_dim
        num_rows_face = nf * proj_dim

        row_cell, col_cell, data_cell, row_face, col_face, data_face = _mat_inds(
            nc, nf, ind, proj_dim, cell_start, face_start
        )

        known_cell_proj = sps.coo_matrix(
            (data_cell, (row_cell, col_cell)), shape=(num_rows_cell, n_cells)
        ).tocsr()
        known_face_proj = sps.coo_matrix(
            (data_face, (row_face, col_face)), shape=(num_rows_face, n_faces)
        ).tocsr()

        assert _compare_matrices(proj.cell_restriction([sd]), known_cell_proj)
        assert _compare_matrices(proj.cell_prolongation([sd]), known_cell_proj.T)
        assert _compare_matrices(proj.face_restriction([sd]), known_face_proj)
        assert _compare_matrices(proj.face_prolongation([sd]), known_face_proj.T)

    # Project between the full grid and both 1d grids (to combine two grids)
    g1, g2 = mdg.subdomains(dim=1)
    rc1, cc1, dc1, rf1, cf1, df1 = _mat_inds(
        g1.num_cells,
        g1.num_faces,
        _list_ind_of_grid(subdomains, g1),
        proj_dim,
        cell_start,
        face_start,
    )
    rc2, cc2, dc2, rf2, cf2, df2 = _mat_inds(
        g2.num_cells,
        g2.num_faces,
        _list_ind_of_grid(subdomains, g2),
        proj_dim,
        cell_start,
        face_start,
    )

    # Adjust the indices of the second grid, we will stack the matrices.
    rc2 += rc1.size
    rf2 += rf1.size
    num_rows_cell = (g1.num_cells + g2.num_cells) * proj_dim
    num_rows_face = (g1.num_faces + g2.num_faces) * proj_dim

    known_cell_proj = sps.coo_matrix(
        (np.hstack((dc1, dc2)), (np.hstack((rc1, rc2)), np.hstack((cc1, cc2)))),
        shape=(num_rows_cell, n_cells),
    ).tocsr()
    known_face_proj = sps.coo_matrix(
        (np.hstack((df1, df2)), (np.hstack((rf1, rf2)), np.hstack((cf1, cf2)))),
        shape=(num_rows_face, n_faces),
    ).tocsr()

    assert _compare_matrices(proj.cell_restriction([g1, g2]), known_cell_proj)
    assert _compare_matrices(proj.cell_prolongation([g1, g2]), known_cell_proj.T)
    assert _compare_matrices(proj.face_restriction([g1, g2]), known_face_proj)
    assert _compare_matrices(proj.face_prolongation([g1, g2]), known_face_proj.T)


@pytest.mark.parametrize("scalar", [True, False])
def test_mortar_projections(mdg, scalar):
    # Test of mortar projections between mortar grids and standard subdomain grids.

    proj_dim = 1 if scalar else mdg.dim_max()
    n_cells, n_faces, n_mortar_cells = geometry_information(mdg, proj_dim)

    g0 = mdg.subdomains(dim=2)[0]
    g1, g2 = mdg.subdomains(dim=1)
    g3 = mdg.subdomains(dim=0)[0]

    intf01 = mdg.subdomain_pair_to_interface((g0, g1))
    intf02 = mdg.subdomain_pair_to_interface((g0, g2))

    intf13 = mdg.subdomain_pair_to_interface((g1, g3))
    intf23 = mdg.subdomain_pair_to_interface((g2, g3))

    ########
    # First test projection between all grids and all interfaces
    subdomains = [g0, g1, g2, g3]
    interfaces = [intf01, intf02, intf13, intf23]

    proj = pp.ad.MortarProjections(
        subdomains=subdomains, interfaces=interfaces, mdg=mdg, dim=proj_dim
    )

    cell_start = proj_dim * np.cumsum(
        np.hstack((0, np.array([g.num_cells for g in subdomains])))
    )
    face_start = proj_dim * np.cumsum(
        np.hstack((0, np.array([g.num_faces for g in subdomains])))
    )

    f0 = np.hstack(
        (
            sps.find(intf01.mortar_to_primary_int(nd=proj_dim))[0],
            sps.find(intf02.mortar_to_primary_int(nd=proj_dim))[0],
        )
    )
    f1 = sps.find(intf13.mortar_to_primary_int(nd=proj_dim))[0]
    f2 = sps.find(intf23.mortar_to_primary_int(nd=proj_dim))[0]

    c1 = sps.find(intf01.mortar_to_secondary_int(nd=proj_dim))[0]
    c2 = sps.find(intf02.mortar_to_secondary_int(nd=proj_dim))[0]
    c3 = np.hstack(
        (
            sps.find(intf13.mortar_to_secondary_int(nd=proj_dim))[0],
            sps.find(intf23.mortar_to_secondary_int(nd=proj_dim))[0],
        )
    )

    rows_higher = np.hstack((f0, f1 + face_start[1], f2 + face_start[2]))
    cols_higher = np.arange(n_mortar_cells)
    data = np.ones(n_mortar_cells)

    proj_known_higher = sps.coo_matrix(
        (data, (rows_higher, cols_higher)), shape=(n_faces, n_mortar_cells)
    ).tocsr()

    assert _compare_matrices(proj_known_higher, proj.mortar_to_primary_int)
    assert _compare_matrices(proj_known_higher, proj.mortar_to_primary_avg)
    assert _compare_matrices(proj_known_higher.T, proj.primary_to_mortar_int)
    assert _compare_matrices(proj_known_higher.T, proj.primary_to_mortar_avg)

    rows_lower = np.hstack((c1 + cell_start[1], c2 + cell_start[2], c3 + cell_start[3]))
    cols_lower = np.arange(n_mortar_cells)
    data = np.ones(n_mortar_cells)

    proj_known_lower = sps.coo_matrix(
        (data, (rows_lower, cols_lower)), shape=(n_cells, n_mortar_cells)
    ).tocsr()
    assert _compare_matrices(proj_known_lower, proj.mortar_to_secondary_int)

    # Also test block matrices for the sign of mortar projections.
    # This is a diagonal matrix with first -1, then 1.
    # If this test fails, something is fundamentally wrong.
    vals = np.array([])
    for intf in interfaces:
        sz = int(np.round(intf.num_cells / 2) * proj_dim)
        vals = np.hstack((vals, -np.ones(sz), np.ones(sz)))

    known_sgn_mat = sps.dia_matrix((vals, 0), shape=(n_mortar_cells, n_mortar_cells))
    assert _compare_matrices(known_sgn_mat, proj.sign_of_mortar_sides)


@pytest.mark.parametrize("scalar", [True, False])
def test_boundary_grid_projection(mdg: pp.MixedDimensionalGrid, scalar: bool):
    """Aspects to test:
    1) That we can create a boundary projection operator with the correct size and items.
    2) Specifically that the top-dimensional grid and one of the fracture grids
       contribute to the boundary projection operator, while the third has a projection
       matrix with zero rows.
    3) Projection from a subdomain to a boundary is consistent with its reverse.

    """
    proj_dim = 1 if scalar else mdg.dim_max()
    _, num_faces, _ = geometry_information(mdg, proj_dim)
    num_cells = sum([bg.num_cells for bg in mdg.boundaries()]) * proj_dim

    g_0 = mdg.subdomains(dim=2)[0]
    g_1, g_2 = mdg.subdomains(dim=1)
    # Compute geometry for the mixed-dimensional grid. This is needed for
    # boundary projection operator.
    mdg.compute_geometry()
    projection = pp.ad.BoundaryProjection(mdg, mdg.subdomains(), proj_dim)
    # Obtaining sparse matrices from the AD Operators.
    subdomain_to_boundary = projection.subdomain_to_boundary.parse(mdg)
    boundary_to_subdomain = projection.boundary_to_subdomain.parse(mdg)
    # Check sizes.
    assert subdomain_to_boundary.shape == (num_cells, num_faces)
    assert boundary_to_subdomain.shape == (num_faces, num_cells)

    # Check that the projection matrix for the top-dimensional grid is non-zero.
    # The matrix has eight boundary faces.
    ind0 = 0
    ind1 = g_0.num_faces * proj_dim
    assert np.sum(subdomain_to_boundary[:, ind0:ind1]) == 8 * proj_dim
    # Check that the projection matrix for the first fracture is non-zero. Since the
    # fracture touches the boundary on two sides, we expect two non-zero rows.
    ind0 = ind1
    ind1 += g_1.num_faces * proj_dim
    assert np.sum(subdomain_to_boundary[:, ind0:ind1]) == 2 * proj_dim
    # Check that the projection matrix for the second fracture is non-zero.
    ind0 = ind1
    ind1 += g_2.num_faces * proj_dim
    assert np.sum(subdomain_to_boundary[:, ind0:ind1]) == 2 * proj_dim
    # The projection matrix for the intersection should be zero.
    ind0 = ind1
    assert np.sum(subdomain_to_boundary[:, ind0:]) == 0

    # Make second projection on subset of grids.
    subdomains = [g_0, g_1]
    projection = pp.ad.grid_operators.BoundaryProjection(mdg, subdomains, proj_dim)
    num_faces = proj_dim * (g_0.num_faces + g_1.num_faces)
    num_cells = proj_dim * sum(
        [mdg.subdomain_to_boundary_grid(sd).num_cells for sd in subdomains]
    )
    # Obtaining sparse matrices from the AD Operators.
    subdomain_to_boundary = projection.subdomain_to_boundary.parse(mdg)
    boundary_to_subdomain = projection.boundary_to_subdomain.parse(mdg)
    # Check sizes.
    assert subdomain_to_boundary.shape == (num_cells, num_faces)
    assert boundary_to_subdomain.shape == (num_faces, num_cells)

    # Check that the projection matrix for the top-dimensional grid is non-zero.
    # Same sizes as above.
    ind0 = 0
    ind1 = g_0.num_faces * proj_dim
    assert np.sum(subdomain_to_boundary[:, ind0:ind1]) == 8 * proj_dim
    ind0 = ind1
    ind1 += g_1.num_faces * proj_dim
    assert np.sum(subdomain_to_boundary[:, ind0:ind1]) == 2 * proj_dim

    # Check that subdomain_to_boundary and boundary_to_subdomain are consistent.
    assert np.allclose((subdomain_to_boundary - boundary_to_subdomain.T).data, 0)


# Geometry based operators
def test_trace(mdg: pp.MixedDimensionalGrid):
    """Test Trace operator.

    Parameters:
        mdg: Mixed-dimensional grid.

    This test is not ideal. It follows the implementation of Trace relatively closely,
    but nevertheless provides some coverage, especially if Trace is carelessly changed.
    The test constructs the expected md trace and inv_trace matrices and compares them
    to the ones of Trace. Also checks that an error is raised if a non-scalar trace is
    constructed (not implemented).
    """
    # The operator should work on any subset of mdg.subdomains.
    subdomains = mdg.subdomains(dim=1)

    # Construct expected matrices
    traces, inv_traces = list(), list()
    # No check on this function here.
    # TODO: A separate unit test might be appropriate.
    cell_projections, face_projections = pp.ad.grid_operators._subgrid_projections(
        subdomains, dim=1
    )
    for sd in subdomains:
        local_block = np.abs(sd.cell_faces.tocsr())
        traces.append(local_block * cell_projections[sd].T)
        inv_traces.append(local_block.T * face_projections[sd].T)

    # Compare to operator class
    op = pp.ad.Trace(subdomains)
    _compare_matrices(op.trace, sps.bmat([[m] for m in traces]))
    _compare_matrices(op.inv_trace, sps.bmat([[m] for m in inv_traces]))

    # As of the writing of this test, Trace is not implemented for vector values.
    # If it is ever extended, the test should be extended accordingly (e.g. parametrized with
    # dim=[1, 2]).
    with pytest.raises(NotImplementedError):
        pp.ad.Trace(subdomains, dim=2)


@pytest.mark.parametrize("dim", [1, 4])
def test_divergence(mdg: pp.MixedDimensionalGrid, dim: int):
    """Test Divergence.

    Parameters:
        mdg: Mixed-dimensional grid.
        dim: Dimension of vector field to which Divergence is applied.

    This test is not ideal. It follows the implementation of Divergence relatively
    closely, but nevertheless provides some coverage. Frankly, there is not much more to
    do than comparing against the expected matrices, unless one wants to add more
    integration-type tests e.g. evaluating combinations with other ad entities.

    """
    # The operator should work on any subset of mdg.subdomains.
    subdomains = mdg.subdomains(dim=2) + mdg.subdomains(dim=0)

    # Construct expected matrix
    divergences = list()
    for sd in subdomains:
        # Kron does no harm if dim=1
        local_block = sps.kron(sd.cell_faces.tocsr().T, sps.eye(dim))
        divergences.append(local_block)

    # Compare to operators parsed value
    op = pp.ad.Divergence(subdomains)
    val = op.parse(mdg)
    _compare_matrices(val, sps.block_diag(divergences))


def test_ad_discretization_class():
    # Test of the mother class of all discretizations (pp.ad.Discretization)

    fracs = [np.array([[0, 2], [1, 1]]), np.array([[1, 1], [0, 2]])]
    mdg = pp.meshing.cart_grid(fracs, np.array([2, 2]))

    subdomains = [g for g in mdg.subdomains()]
    sub_list = subdomains[:2]

    # Make two Mock discretizations, with different keywords
    key = "foo"
    sub_key = "bar"
    discr = _MockDiscretization(key)
    sub_discr = _MockDiscretization(sub_key)

    # Ad wrappers
    # This mimics the old init of Discretization, before it was decided to
    # make that class semi-ABC. Still checks the wrap method
    discr_ad = pp.ad.Discretization()
    discr_ad.subdomains = subdomains
    discr_ad._discretization = discr
    pp.ad._ad_utils.wrap_discretization(discr_ad, discr, subdomains)
    sub_discr_ad = pp.ad.Discretization()
    sub_discr_ad.subdomains = sub_list
    sub_discr_ad._discretization = sub_discr
    pp.ad._ad_utils.wrap_discretization(sub_discr_ad, sub_discr, sub_list)

    # values
    known_val = np.random.rand(len(subdomains))
    known_sub_val = np.random.rand(len(sub_list))

    # Assign a value to the discretization matrix, with the right key
    for vi, sd in enumerate(subdomains):
        data = mdg.subdomain_data(sd)
        data[pp.DISCRETIZATION_MATRICES] = {key: {"foobar": known_val[vi]}}

    # Same with submatrix
    for vi, sd in enumerate(sub_list):
        data = mdg.subdomain_data(sd)
        data[pp.DISCRETIZATION_MATRICES].update(
            {sub_key: {"foobar": known_sub_val[vi]}}
        )

    # Compare values under parsing. Note we need to pick out the diagonal, due to the
    # way parsing makes block matrices.
    assert np.allclose(known_val, discr_ad.foobar.parse(mdg).diagonal())
    assert np.allclose(known_sub_val, sub_discr_ad.foobar.parse(mdg).diagonal())


## Below are helpers for tests of the Ad wrappers.


def _compare_matrices(m1, m2):
    if isinstance(m1, pp.ad.SparseArray):
        m1 = m1._mat
    if isinstance(m2, pp.ad.SparseArray):
        m2 = m2._mat
    if m1.shape != m2.shape:
        return False
    d = m1 - m2
    if d.data.size > 0:
        if np.max(np.abs(d.data)) > 1e-10:
            return False
    return True


def _list_ind_of_grid(subdomains, g):
    for i, gl in enumerate(subdomains):
        if g == gl:
            return i

    raise ValueError("grid is not in list")


class _MockDiscretization:
    def __init__(self, key):
        self.foobar_matrix_key = "foobar"
        self.not_matrix_keys = "failed"

        self.keyword = key
