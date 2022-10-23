"""Test creation and composition of Ad operators and their trees.

The main checks performed are:
    test_elementary_operations: Checks of basic aritmetic operations are correctly implemented.
    test_copy_operator_tree: Test of functionality under copy.copy and deepcopy.
    test_elementary_wrappers: Test wrapping of scalars, arrays and matrices.
    test_ad_variable_creation: Generate variables, check that copies and new variables are
        returned as expected.

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
    """Test that performing elementary aritmetic operations on operators return operator trees
    with the expected structure.
    
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
        sd_data[pp.PRIMARY_VARIABLES] = {'foo': {'cells': 1}}
        sd_data[pp.STATE] = {'foo': np.zeros(1), pp.ITERATE: {'foo': np.zeros(1)}}
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
    
test_copy_operator_tree()
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
    # This must be done in slightly different ways, depending on the implementation of the
    # underlying data structures.
    if isinstance(obj, sps.spmatrix):
        obj[0,0] += 1
    else:
        obj += 1.0
    
    # The shallow copy of the Ad quantity should also pick up the modification.
    # The exception is the scalar, which is a wrapper of a Python immutable, and thus will return
    # a pointer to the underlying immutable.
    if isinstance(wrapped_obj, pp.ad.Scalar):
        assert not compare(obj, wrapped_copy.parse(None))
    else:
        assert compare(obj, wrapped_copy.parse(None))
    
    # The deep copy should differ independent of which type of Ad quantity this is
    assert not compare(obj, wrapped_deep_copy.parse(None))
    
for field in fields:
    test_elementary_wrappers(field)
    


def test_ad_variable_creation():
    """Test creation of Ad variables by way of the EquationManager.
        1) Fetching the same variable twice should get the same variable (same attribute id).
        2) Fetching the same merged variable twice should result in objects with different id
           attributes, but point to the same underlying variable.
    """
    mdg, _= pp.grids.standard_grids.md_grids_2d.single_horizontal()

    for sd, sd_data in mdg.subdomains(return_data=True):
        sd_data[pp.PRIMARY_VARIABLES] = {'foo': {'cells': 1}}
    for intf, intf_data in mdg.interfaces(return_data=True):
        # Create an empty primary variable list
        intf_data[pp.PRIMARY_VARIABLES] = {}

    dof_manager = pp.DofManager(mdg)
    eq_manager = pp.ad.EquationManager(mdg, dof_manager)

    var_1 = eq_manager.variable(mdg.subdomains(dim=mdg.dim_max())[0], 'foo')
    var_2 = eq_manager.variable(mdg.subdomains(dim=mdg.dim_max())[0], 'foo')
    var_3 = eq_manager.variable(mdg.subdomains(dim=mdg.dim_min())[0], 'foo')

    # Fetching the same variable twice should give the same variable (idetified by the variable id)
    assert var_1.id == var_2.id
    # A variable with the same name, but on a different grid should have a different id
    assert var_1.id != var_3.id
    
    # Fetch merged variable representations of the same variables
    mvar_1 = eq_manager.merge_variables([(mdg.subdomains(dim=mdg.dim_max())[0], 'foo')])
    mvar_2 = eq_manager.merge_variables([(mdg.subdomains(dim=mdg.dim_max())[0], 'foo')])    

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


test_ad_variable_creation()