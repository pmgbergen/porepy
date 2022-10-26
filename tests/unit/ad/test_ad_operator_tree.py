"""Test creation and composition of Ad operators and their trees.

The main checks performed are:
    test_elementary_operations: Checks of basic aritmetic operations are correctly
        implemented.
    test_copy_operator_tree: Test of functionality under copy.copy and deepcopy.
    test_elementary_wrappers: Test wrapping of scalars, arrays and matrices.
    test_ad_variable_creation: Generate variables, check that copies and new variables 
        are returned as expected.
    test_ad_variable_wrappers: Test of the wrappers for variables.
    test_time_differentiation: Covers the pp.ad.dt operator.

"""
import copy
import porepy as pp
import pytest
import scipy.sparse as sps
import numpy as np

_operations = pp.ad.operators.Operation

operators = [
    ("+", _operations.add),
    ("-", _operations.sub),
    ("*", _operations.mul),
    ("/", _operations.div),
]


@pytest.mark.parametrize("operator", operators)
def test_elementary_operations(operator):
    """Test that performing elementary aritmetic operations on operators return operator
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

    # Check that the shallow copy also have the same children, while
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
    for sd, sd_data in mdg.subdomains(return_data=True):
        sd_data[pp.PRIMARY_VARIABLES] = {"foo": {"cells": 1}}
        sd_data[pp.STATE] = {"foo": np.zeros(1), pp.ITERATE: {"foo": np.zeros(1)}}
    for intf, intf_data in mdg.interfaces(return_data=True):
        # Create an empty primary variable list
        intf_data[pp.PRIMARY_VARIABLES] = {}
        intf_data[pp.STATE] = {}
    dof_manager = pp.DofManager(mdg)

    # In their initial state, all operators should have the same values
    assert np.allclose(c.evaluate(dof_manager), c_copy.evaluate(dof_manager))
    assert np.allclose(c.evaluate(dof_manager), c_deepcopy.evaluate(dof_manager))
    # Increase the value of the scalar. This should have no effect, since the scalar
    # wrapps an immutable, see comment in pp.ad.Scalar
    a_val += 1
    assert np.allclose(c.evaluate(dof_manager), c_copy.evaluate(dof_manager))
    assert np.allclose(c.evaluate(dof_manager), c_deepcopy.evaluate(dof_manager))

    # Next increase the values in the array. This changes the shallow copy, but not the
    # deep one.
    b_arr += 1
    assert np.allclose(c.evaluate(dof_manager), c_copy.evaluate(dof_manager))
    assert not np.allclose(c.evaluate(dof_manager), c_deepcopy.evaluate(dof_manager))


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

    The test takes a standard object, wrap in in the given Ad class, and verifies
    that parsing returns the expected object.

    Also test the behavior of the classes under copying.

    """
    obj = field[1]
    wrapped_obj = field[0](obj, name="foo")

    # Evaluate the Ad wrapper using parse, which will act directly on the wrapper
    # (as oposed to evaluate, which will invoke the full evaluation machinery of the
    # ad operator tree)

    # We can use None her, since the MixedDimensionalGrid is not used for parsing of
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
    # will return a pointer to the underlying immutable.
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

    # We make three arrays: One defined on a single subdomain, one on a single subdomain
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
    assert np.allclose(sd_prev_timestep.parse(mdg), 0)

    intf_prev_timestep = intf_array.previous_timestep()
    assert np.allclose(intf_prev_timestep.parse(mdg), np.arange(intf_val.size))


def test_ad_variable_creation():
    """Test creation of Ad variables by way of the EquationManager.
    1) Fetching the same variable twice should get the same variable (same attribute id).
    2) Fetching the same merged variable twice should result in objects with different
       id attributes, but point to the same underlying variable.
    """
    mdg, _ = pp.grids.standard_grids.md_grids_2d.single_horizontal()

    for sd, sd_data in mdg.subdomains(return_data=True):
        sd_data[pp.PRIMARY_VARIABLES] = {"foo": {"cells": 1}}
    for intf, intf_data in mdg.interfaces(return_data=True):
        # Create an empty primary variable list
        intf_data[pp.PRIMARY_VARIABLES] = {}

    dof_manager = pp.DofManager(mdg)
    eq_manager = pp.ad.EquationManager(mdg, dof_manager)

    var_1 = eq_manager.variable(mdg.subdomains(dim=mdg.dim_max())[0], "foo")
    var_2 = eq_manager.variable(mdg.subdomains(dim=mdg.dim_max())[0], "foo")
    var_3 = eq_manager.variable(mdg.subdomains(dim=mdg.dim_min())[0], "foo")

    # Fetching the same variable twice should give the same variable (idetified by the
    # variable id)
    assert var_1.id == var_2.id
    # A variable with the same name, but on a different grid should have a different id
    assert var_1.id != var_3.id

    # Fetch merged variable representations of the same variables
    mvar_1 = eq_manager.merge_variables([(mdg.subdomains(dim=mdg.dim_max())[0], "foo")])
    mvar_2 = eq_manager.merge_variables([(mdg.subdomains(dim=mdg.dim_max())[0], "foo")])

    # The two merged variables should have different ids
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

    # Then merged variables.
    mvar_1_prev_iter = mvar_1.previous_iteration()
    mvar_1_prev_time = mvar_1.previous_timestep()
    assert mvar_1_prev_iter.id != mvar_1.id
    assert mvar_1_prev_time.id != mvar_1.id

def test_ad_variable_wrappers():
    # Tests that the wrapping of Ad variables, including previous iterates
    # and time steps, are carried out correctly.
    # See also test_variable_combinations, which specifically tests evaluation of
    # variables in a setting of multiple variables, including merged variables.

    fracs = [np.array([[0, 2], [1, 1]]), np.array([[1, 1], [0, 2]])]
    mdg = pp.meshing.cart_grid(fracs, np.array([2, 2]))

    state_map = {}
    iterate_map = {}

    state_map_2, iterate_map_2 = {}, {}

    var = "foo"
    var2 = "bar"

    mortar_var = "mv"

    def _compare_ad_objects(a, b):
        va, ja = a.val, a.jac
        vb, jb = b.val, b.jac

        assert np.allclose(va, vb)
        assert ja.shape == jb.shape
        d = ja - jb
        if d.data.size > 0:
            assert np.max(np.abs(d.data)) < 1e-10

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

    dof_manager = pp.DofManager(mdg)
    eq_manager = pp.ad.EquationManager(mdg, dof_manager)

    # Manually assemble state and iterate
    true_state = np.zeros(dof_manager.num_dofs())
    true_iterate = np.zeros(dof_manager.num_dofs())

    # Also a state array that differs from the stored iterates
    double_iterate = np.zeros(dof_manager.num_dofs())

    for (g, v) in dof_manager.block_dof:
        inds = dof_manager.grid_and_variable_to_dofs(g, v)
        if v == var2:
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

    # Generate merged variables via the EquationManager.
    var_ad = eq_manager.merge_variables([(g, var) for g in subdomains])

    # Check equivalence between the two approaches to generation.

    # Check that the state is correctly evaluated.
    inds_var = np.hstack(
        [dof_manager.grid_and_variable_to_dofs(g, var) for g in subdomains]
    )
    assert np.allclose(
        true_iterate[inds_var], var_ad.evaluate(dof_manager, true_iterate).val
    )

    # Check evaluation when no state is passed to the parser, and information must
    # instead be glued together from the MixedDimensionalGrid
    assert np.allclose(true_iterate[inds_var], var_ad.evaluate(dof_manager).val)

    # Evaluate the equation using the double iterate
    assert np.allclose(
        2 * true_iterate[inds_var], var_ad.evaluate(dof_manager, double_iterate).val
    )

    # Represent the variable on the previous time step. This should be a numpy array
    prev_var_ad = var_ad.previous_timestep()
    prev_evaluated = prev_var_ad.evaluate(dof_manager)
    assert isinstance(prev_evaluated, np.ndarray)
    assert np.allclose(true_state[inds_var], prev_evaluated)

    # Also check that state values given to the ad parser are ignored for previous
    # values
    assert np.allclose(
        prev_evaluated, prev_var_ad.evaluate(dof_manager, double_iterate)
    )

    ## Next, test edge variables. This should be much the same as the grid variables,
    # so the testing is less thorough.
    # Form an edge variable, evaluate this
    edge_list = [intf for intf in mdg.interfaces()]
    var_edge = eq_manager.merge_variables([(e, mortar_var) for e in edge_list])

    edge_inds = np.hstack(
        [dof_manager.grid_and_variable_to_dofs(e, mortar_var) for e in edge_list]
    )
    assert np.allclose(
        true_iterate[edge_inds], var_edge.evaluate(dof_manager, true_iterate).val
    )

    # Finally, test a single variable; everything should work then as well
    g = mdg.subdomains(dim=2)[0]
    v1 = eq_manager.variable(g, var)
    v2 = eq_manager.variable(g, var2)

    ind1 = dof_manager.grid_and_variable_to_dofs(g, var)
    ind2 = dof_manager.grid_and_variable_to_dofs(g, var2)

    assert np.allclose(true_iterate[ind1], v1.evaluate(dof_manager, true_iterate).val)
    assert np.allclose(true_iterate[ind2], v2.evaluate(dof_manager, true_iterate).val)

    v1_prev = v1.previous_timestep()
    assert np.allclose(true_state[ind1], v1_prev.evaluate(dof_manager, true_iterate))



def test_time_differentiation():
    """Test the dt operator in AD.

    For the moment, this is simply a test that backward Euler is correctly implemented.
    """

    # Create a MixedDimensionalGrid with two subdomains, one interface.
    # The highest-dimensional subdomain has both a variable and a time-dependent array,
    # while the lower-dimensional
    mdg, _ = pp.grids.standard_grids.md_grids_2d.single_horizontal()
    for sd, sd_data in mdg.subdomains(return_data=True):
        sd_data[pp.PRIMARY_VARIABLES] = {"foo": {"cells": 1}}

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

    dof_manager = pp.DofManager(mdg)
    eq_manager = pp.ad.EquationManager(mdg, dof_manager)

    # The time step, represented as a scalar.
    time_step = pp.ad.Scalar(2)

    # Differentiate the variable on the highest-dimensional subdomain
    sd = mdg.subdomains(dim=mdg.dim_max())[0]
    var_1 = eq_manager.variable(sd, "foo")
    dt_var_1 = pp.ad.dt(var_1, time_step)
    assert np.allclose(dt_var_1.evaluate(dof_manager).val, 2)

    # Differentiate the time dependent array residing on the subdomain
    array = pp.ad.TimeDependentArray(name="bar", subdomains=[sd])
    dt_array = pp.ad.dt(array, time_step)
    assert np.allclose(dt_array.evaluate(dof_manager), -0.5)

    # Combine the parameter array and the variable. This is a test that operators that
    # are not leaves are differentiated correctly.
    var_array = var_1 * array
    dt_var_array = pp.ad.dt(var_array, time_step)
    assert np.allclose(dt_var_array.evaluate(dof_manager).val, 2.5)

    # For good measure, add one more level of combination.
    var_array_2 = var_array + var_array
    dt_var_array = pp.ad.dt(var_array_2, time_step)
    assert np.allclose(dt_var_array.evaluate(dof_manager).val, 5)

    # Also do a test of the merged variable.
    mvar = eq_manager.merge_variables([(sd, "foo")])

    dt_mvar = pp.ad.dt(mvar, time_step)
    assert np.allclose(dt_mvar.evaluate(dof_manager).val[: sd.num_cells], 2)
    assert np.allclose(dt_mvar.evaluate(dof_manager).val[sd.num_cells :], 0.5)

    # Make a combined operator with the merged variable, test this.
    dt_mvar = pp.ad.dt(mvar * mvar, time_step)
    assert np.allclose(dt_mvar.evaluate(dof_manager).val[: sd.num_cells], 4)
    assert np.allclose(dt_mvar.evaluate(dof_manager).val[sd.num_cells :], 0.5)

    # Finally create a variable at the previous time step. Its derivative should be
    # zero.
    var_2 = var_1.previous_timestep()
    dt_var_2 = pp.ad.dt(var_2, time_step)
    assert np.allclose(dt_var_2.evaluate(dof_manager), 0)

