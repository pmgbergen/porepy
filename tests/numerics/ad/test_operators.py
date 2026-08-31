"""Test collection for Ad representations of several operators.

Checks performed include the following:
    test_elementary_operations: Checks are made to ensure that basic arithmetic
        operations are performed correctly;
    test_copy_operator_tree: Testing of functionality under copy.copy and deepcopy;
    test_elementary_wrappers: Test wrapping of fields (scalars, arrays and matrices);
    test_ad_variable_creation: The generation of variables should ensure that copies and
        newly created variables are returned in the expected manner;
    test_ad_variable_evaluation: Variable wrappers are tested as expected under
    evaluation test_time_differentiation: Covers the pp.ad.dt operator;
    test_ad_discretization_class: test for AD discretizations;
    test_arithmetic_operations_on_ad_objects: Basic Ad operators combined with standard
        arithmetic operations are tested.
    test_hashing: Expectations for the hash function to work correctly with AD objects.
    test_hashing_sparse_array: Edge cases for the hash function of SparseArray.

"""

import copy
from typing import Literal, Union

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp
from porepy.applications.md_grids.model_geometries import (
    SquareDomainOrthogonalFractures,
)
from porepy.models.fluid_mass_balance import SinglePhaseFlow
from porepy.numerics.ad.equation_system import GridEntity
from porepy.numerics.ad.operators import DomainType

AdType = Union[float, np.ndarray, sps.spmatrix, pp.ad.AdArray]


class TestCopyOperatorTree:
    """Test that copying of an operator tree works as expected.

    The test makes a simple tree by combining a scalar and a numpy array. The intention
    is to use this as a test of copying trees, while copying of individual operators
    should be done elsewhere.

    """

    def setup_method(self):
        # To verify the difference between copy and deepcopy, keep pointers to the data
        # structures to be wrapped
        self.a_val = 42
        self.a = pp.ad.Scalar(self.a_val)
        # Use an unclear operator space here since we do not care about the actual
        # domain of the DenseArray (tests of this is carried out elsewhere).
        space = pp.ad.OperatorSpace.unclear()
        b_val = np.arange(3)
        self.b = pp.ad.DenseArray(b_val, source=space, target=space)

        # The combined operator, and two copies
        self.c = self.a + self.b
        self.c_copy = copy.copy(self.c)
        self.c_deepcopy = copy.deepcopy(self.c)
        # Create an EquationSystem defined on a MixedDimensionalGrid (not really used)
        # for parsing the operators.
        mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
            "cartesian",
            {"cell_size": 0.5},
            fracture_indices=[],
        )
        self.equation_system = pp.ad.EquationSystem(mdg)

    def test_operator_properties(self):
        """Unit tests involving no parsing."""

        # First check that the two copies have behaved as they should.
        # The operators should be the same for all trees.
        for item in ["operation", "source", "target"]:
            assert getattr(self.c, item) == getattr(self.c_copy, item)
            assert getattr(self.c, item) == getattr(self.c_deepcopy, item)

        # The operator version of scalars and dense arrays calculates the hash based on
        # the value of the underlying object, hence the comparison operator for
        # pp.ad.Operator should evaluate for True for both the copy and the deepcopy.
        # The id of the underlying object should be the same for the copy, but different
        # for the deepcopy.
        for c1, c2 in zip(self.c.children, self.c_copy.children):
            assert c1 == c2
            assert id(c1) == id(c2)
        for c1, c2 in zip(self.c.children, self.c_deepcopy.children):
            assert c1 == c2
            assert id(c1) != id(c2)

    def test_parsed_evaluation_no_changes(self):
        """Validate that the operators are parsed correctly through an
        EquationSystem."""

        np.testing.assert_allclose(
            self.equation_system.evaluate(self.c),
            self.equation_system.evaluate(self.c_copy),
        )
        np.testing.assert_allclose(
            self.equation_system.evaluate(self.c),
            self.equation_system.evaluate(self.c_deepcopy),
        )

    def test_changing_scalar_outside_operator_has_no_impact(self):
        """Increase the value of the scalar used to construct the operators. This should
        have no effect, since the scalar wrapps an immutable, see comment in
        pp.ad.Scalar.
        """
        self.a_val += 1
        np.testing.assert_allclose(
            self.equation_system.evaluate(self.c),
            self.equation_system.evaluate(self.c_copy),
        )
        assert np.allclose(
            self.equation_system.evaluate(self.c),
            self.equation_system.evaluate(self.c_deepcopy),
        )

    def test_changing_scalar_operator_not_seen_by_deep_copy(self):
        """Increase the value of the scalar used to construct the operators. This should
        not be seen by the deep copy, since it is a separate object.
        """
        self.a._value += 1
        np.testing.assert_allclose(
            self.equation_system.evaluate(self.c),
            self.equation_system.evaluate(self.c_copy),
        )
        np.testing.assert_allclose(
            self.equation_system.evaluate(self.c),
            self.equation_system.evaluate(self.c_deepcopy) + 1,
        )


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
    if field[0] == pp.ad.Scalar:
        wrapped_obj = field[0](obj, name="foo")
    else:
        wrapped_obj = field[0](
            obj,
            name="foo",
            source=pp.ad.OperatorSpace.unclear(),
            target=pp.ad.OperatorSpace.unclear(),
        )

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

    # Next, ensuring that it is impossible to modify the underlying object.
    if not isinstance(wrapped_obj, pp.ad.Scalar):
        with pytest.raises(ValueError):
            if isinstance(obj, sps.spmatrix):
                obj[0, 0] += 1
            else:
                obj += 1

    else:
        # Scalar is a wrapper of a Python immutable and thus does not copy by reference.
        obj += 1
        assert not compare(obj, wrapped_copy.parse(None))


@pytest.mark.parametrize("field", fields)
def test_ad_arrays_unary_minus_parsing(field):
    """Check that __neg__ works as intended for SparseArrays, DenseArrays and Scalars.

    The objects are wrapped in the respective Ad classes, the __neg__ methods are used,
    and it is tested whether parsing returns the expected object.

    """
    obj = field[1]
    if field[0] == pp.ad.Scalar:
        wrapped_obj = -field[0](obj, name="foo")
    else:
        wrapped_obj = -field[0](
            obj,
            name="foo",
            source=pp.ad.OperatorSpace.unclear(),
            target=pp.ad.OperatorSpace.unclear(),
        )
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
    # Use an unclear operator space here since we do not care about the actual domain of
    # the SparseArray (tests of this is carried out elsewhere).
    space = pp.ad.OperatorSpace.unclear()
    sp_array1 = pp.ad.SparseArray(mat1, source=space, target=space)
    sp_array2 = pp.ad.SparseArray(mat2, source=space, target=space)
    equation_system = pp.ad.EquationSystem(pp.MixedDimensionalGrid())
    op = sp_array1 + sp_array2
    assert np.allclose(equation_system.evaluate(-op, None).data, -(mat1 + mat2).data)


def test_time_dependent_array():
    """Test of time-dependent arrays (wrappers around numpy arrays)."""

    # Time-dependent arrays are defined on grids.
    # Some boilerplate is needed to define these.
    mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
        "cartesian",
        {"cell_size": 0.2},
        fracture_indices=[1],
    )
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
    assert (
        sd_array.target.domain_type
        == sd_array_top.target.domain_type
        == DomainType.subdomains
    )
    assert intf_array.target.domain_type == DomainType.interfaces
    assert bg_array.target.domain_type == DomainType.boundary_grids

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
    empty_array = pp.ad.TimeDependentDenseArray(
        "none", domains=[], domain_type=DomainType.subdomains
    )
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

    # Time dependent arrays at two time steps back are possible, but the evaluation
    # should raise a key error because no values are stored
    with pytest.raises(KeyError):
        _ = sd_prev_timestep.previous_timestep().parse(mdg)


def test_ad_variable_creation():
    """Test creation of Ad variables by way of the EquationSystem.
    1) Fetching the same variable twice should get the same variable (same attribute
        id).
    2) Fetching the same mixed-dimensional variable twice should result in objects with
       different id attributes, but point to the same underlying variable.

    No tests are made of the actual values of the variables, as this is tested in
    test_ad_variable_evaluation() (below).

    """
    mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
        "cartesian",
        {"cell_size": 0.2},
        fracture_indices=[1],
    )
    equation_system = pp.ad.EquationSystem(mdg)
    equation_system.create_variables("foo", {GridEntity.cells: 1}, mdg.subdomains())

    var_1 = equation_system.get_variables(["foo"], mdg.subdomains(dim=mdg.dim_max()))[0]
    var_2 = equation_system.get_variables(["foo"], mdg.subdomains(dim=mdg.dim_max()))[0]
    var_3 = equation_system.get_variables(["foo"], mdg.subdomains(dim=mdg.dim_min()))[0]

    # Fetching the same variable twice should give the same variable (idetified by the
    # variable id)
    assert var_1.id == var_2.id
    # A variable with the same name, but on a different grid should have a different id
    assert var_1.id != var_3.id

    # Fetch mixed-dimensional variable representations of the same variables
    mvar_1 = equation_system.md_variable("foo", mdg.subdomains(dim=mdg.dim_max()))
    mvar_2 = equation_system.md_variable("foo", mdg.subdomains(dim=mdg.dim_max()))

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


def test_ad_variable_evaluation():
    """Test that the values of Ad variables are as expected under evalutation
    (translation from the abstract Ad framework to forward mode).

    Both the atomic and mixed-dimensional variables are tested. The tests cover both the
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

    equation_system = pp.EquationSystem(mdg)
    # First create a variable on the subdomains. The number of dofs is different for the
    # different subdomains.
    # NOTE: The order of creation is a bit important here: We will iterate of the
    # subdomains (implicitly sorted by dimension) and assign a value to the variables.
    # The order of creation should therefore be consistent with the order of iteration.
    # It should be possible to avoid this by using dof-indices of the subdomains, but EK
    # cannot wrap his head around this at the moment (it is Friday afternoon).
    equation_system.create_variables(
        var, dof_info={GridEntity.cells: 1}, subdomains=mdg.subdomains(dim=2)
    )
    equation_system.create_variables(
        var, dof_info={GridEntity.cells: 2}, subdomains=mdg.subdomains(dim=1)
    )
    equation_system.create_variables(
        var, dof_info={GridEntity.cells: 1}, subdomains=mdg.subdomains(dim=0)
    )
    equation_system.create_variables(
        var2, dof_info={GridEntity.cells: 1}, subdomains=mdg.subdomains(dim=2)
    )
    # Next create interface variables.
    equation_system.create_variables(
        mortar_var, dof_info={GridEntity.cells: 2}, interfaces=mdg.interfaces(dim=1)
    )
    equation_system.create_variables(
        mortar_var, dof_info={GridEntity.cells: 1}, interfaces=mdg.interfaces(dim=0)
    )

    for sd, data in mdg.subdomains(return_data=True):
        if sd.dim == 1:
            num_dofs = 2
        else:
            num_dofs = 1

        data[pp.PRIMARY_VARIABLES] = {var: {GridEntity.cells: num_dofs}}

        val_state = np.random.rand(sd.num_cells * num_dofs)
        val_iterate = np.random.rand(sd.num_cells * num_dofs)

        pp.set_solution_values(name=var, values=val_state, data=data, time_step_index=0)
        pp.set_solution_values(name=var, values=val_iterate, data=data, iterate_index=0)

        state_map[sd] = val_state
        iterate_map[sd] = val_iterate

        # Add a second variable to the 2d grid, just for the fun of it
        if sd.dim == 2:
            data[pp.PRIMARY_VARIABLES][var2] = {GridEntity.cells: 1}
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

        data[pp.PRIMARY_VARIABLES] = {mortar_var: {GridEntity.cells: num_dofs}}

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
    true_state = np.zeros(equation_system.num_dofs())
    true_iterate = np.zeros(equation_system.num_dofs())

    # Also a state array that differs from the stored iterates
    double_iterate = np.zeros(equation_system.num_dofs())

    for v in equation_system.variables:
        g = v.domains[0]
        inds = equation_system.dofs_of([v])
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

    # Generate mixed-dimensional variables via the EquationSystem.
    var_ad = equation_system.md_variable(var, subdomains)

    # Check equivalence between the two approaches to generation.

    # Check that the state is correctly evaluated.
    inds_var = np.hstack(
        [
            equation_system.dofs_of(equation_system.get_variables([var], [g]))
            for g in subdomains
        ]
    )
    assert np.allclose(
        true_iterate[inds_var], equation_system.evaluate(var_ad, state=true_iterate)
    )

    # Check evaluation when no state is passed to the parser, and information must
    # instead be glued together from the MixedDimensionalGrid
    assert np.allclose(true_iterate[inds_var], equation_system.evaluate(var_ad))

    # Evaluate the equation using the double iterate
    assert np.allclose(
        2 * true_iterate[inds_var],
        equation_system.evaluate(var_ad, state=double_iterate),
    )

    # Represent the variable on the previous time step. This should be a numpy array
    prev_var_ad = var_ad.previous_timestep()
    prev_evaluated = equation_system.evaluate(prev_var_ad)
    assert isinstance(prev_evaluated, np.ndarray)
    assert np.allclose(true_state[inds_var], prev_evaluated)

    # Also check that state values given to the ad parser are ignored for previous
    # values
    assert np.allclose(
        prev_evaluated, equation_system.evaluate(prev_var_ad, state=double_iterate)
    )

    ## Next, test edge variables. This should be much the same as the grid variables,
    # so the testing is less thorough.
    # Form an edge variable, evaluate this
    interfaces = [intf for intf in mdg.interfaces()]
    variable_interfaces = [
        equation_system.md_variable(mortar_var, [intf]) for intf in interfaces
    ]

    interface_inds = np.hstack(
        [equation_system.dofs_of([var]) for var in variable_interfaces]
    )
    interface_values = np.hstack(
        [
            equation_system.evaluate(var, state=true_iterate)
            for var in variable_interfaces
        ]
    )
    assert np.allclose(
        true_iterate[interface_inds],
        interface_values,
    )

    # Finally, test a single variable; everything should work then as well
    g = mdg.subdomains(dim=2)[0]
    v1 = equation_system.get_variables([var], [g])[0]
    v2 = equation_system.get_variables([var2], [g])[0]

    ind1 = equation_system.dofs_of(equation_system.get_variables([var], [g]))
    ind2 = equation_system.dofs_of(equation_system.get_variables([var2], [g]))

    assert np.allclose(
        true_iterate[ind1], equation_system.evaluate(v1, state=true_iterate)
    )
    assert np.allclose(
        true_iterate[ind2], equation_system.evaluate(v2, state=true_iterate)
    )

    v1_prev = v1.previous_timestep()
    assert np.allclose(
        true_state[ind1], equation_system.evaluate(v1_prev, state=true_iterate)
    )


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
    """Test combinations of variables, and mixed-dimensional variables, on different
    grids.

    The main check is if Jacobian matrices are of the right size.
    """
    # Make MixedDimensionalGrid, populate with necessary information
    mdg = pp.MixedDimensionalGrid()
    mdg.add_subdomains(grids)
    for sd, data in mdg.subdomains(return_data=True):
        data[pp.PRIMARY_VARIABLES] = {}
        for var in variables:
            data[pp.PRIMARY_VARIABLES].update({var: {GridEntity.cells: 1}})

            vals = np.random.rand(sd.num_cells)
            pp.set_solution_values(name=var, values=vals, data=data, time_step_index=0)

    # Ad boilerplate
    equation_system = pp.ad.EquationSystem(mdg)
    for var in variables:
        equation_system.create_variables(var, {GridEntity.cells: 1}, mdg.subdomains())
        equation_system.set_variable_values(
            np.random.rand(mdg.num_subdomain_cells()),
            [var],
            time_step_index=0,
            iterate_index=0,
        )
    # Standard Ad variables
    ad_vars = equation_system.get_variables()
    # Merge variables over all grids
    merged_vars = [equation_system.md_variable(var, grids) for var in variables]

    # First check of standard variables. If this fails, something is really wrong
    for sd in grids:
        data = mdg.subdomain_data(sd)
        for var in ad_vars:
            if sd == var.domains[0]:
                expr = var.value_and_jacobian(equation_system)
                # Check that the size of the variable is correct
                values = pp.get_solution_values(
                    name=var.name, data=data, time_step_index=0
                )
                assert np.allclose(expr.val, values)
                # Check that the Jacobian matrix has the right number of columns
                if isinstance(expr, pp.ad.AdArray) and expr._is_diagonal:
                    sz = expr.to_full().jac.shape[1]
                else:
                    sz = expr.jac.shape[1]

                assert sz == equation_system.num_dofs()

    # Next, check that mixed-dimensional variables are handled correctly.
    for var in merged_vars:
        expr = var.value_and_jacobian(equation_system)
        vals = []
        for sub_var in var.sub_vars:
            data = mdg.subdomain_data(sub_var.domains[0])
            values = pp.get_solution_values(
                name=sub_var.name, data=data, time_step_index=0
            )
            vals.append(values)

        assert np.allclose(expr.val, np.hstack([v for v in vals]))
        # Check that the Jacobian matrix size is correct
        if isinstance(expr, pp.ad.AdArray) and expr._is_diagonal:
            sz = expr.to_full().jac.shape[1]
        else:
            sz = expr.jac.shape[1]
        assert sz == equation_system.num_dofs()

    # Finally, check that the size of the Jacobian matrix is correct when combining
    # variables (this will cover both variables and mixed-dimensional variable with the
    # same name, and with different name).
    target = pp.ad.OperatorSpace.from_domains(grids, dof_info={GridEntity.cells: 1})
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
                mv_grids = [sv.domains[0] for sv in mv.sub_vars]
                ind = mv_grids.index(var.domains[0])
                offset = np.hstack((0, np.cumsum(sv_size)))[ind]
                rows = offset + np.arange(nc)
                source = pp.ad.OperatorSpace.from_domains(
                    var.domains, dof_info={GridEntity.cells: 1}
                )
                P = pp.ad.SparseArray(
                    sps.coo_matrix((data, (rows, cols)), shape=(nr, nc)),
                    source=source,
                    target=target,
                )

                eq = mv + P @ var
                expr = eq.value_and_jacobian(equation_system)
                # Jacobian matrix size is set according to the dof manager,
                if isinstance(expr, pp.ad.AdArray) and expr._is_diagonal:
                    sz = expr.to_full().jac.shape[1]
                else:
                    sz = expr.jac.shape[1]
                assert sz == equation_system.num_dofs()


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
    mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
        "cartesian",
        {"cell_size": 0.5},
        fracture_indices=[1],
    )
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

    equation_system = pp.ad.EquationSystem(mdg)
    equation_system.create_variables("foo", {GridEntity.cells: 1}, mdg.subdomains())
    # The time step, represented as a scalar.
    ts = 2
    time_step = pp.ad.Scalar(ts)

    # Differentiate the variable on the highest-dimensional subdomain
    sd = mdg.subdomains(dim=mdg.dim_max())[0]
    var_1 = equation_system.get_variables(["foo"], [sd])[0]
    dt_var_1 = pp.ad.dt(var_1, time_step)
    assert np.allclose(equation_system.evaluate(dt_var_1), 2)

    # Also test the time difference function
    diff_var_1 = pp.ad.time_increment(var_1)
    assert np.allclose(equation_system.evaluate(diff_var_1), 2 * ts)

    # Differentiate the time dependent array residing on the subdomain
    array = pp.ad.TimeDependentDenseArray(name="bar", domains=[sd])
    dt_array = pp.ad.dt(array, time_step)
    assert np.allclose(equation_system.evaluate(dt_array), -0.5)

    # Combine the parameter array and the variable. This is a test that operators that
    # are not leaves are differentiated correctly.
    var_array = var_1 * array
    dt_var_array = pp.ad.dt(var_array, time_step)
    assert np.allclose(equation_system.evaluate(dt_var_array), 2.5)
    # Also test the time increment function
    diff_var_array = pp.ad.time_increment(var_array)
    assert np.allclose(equation_system.evaluate(diff_var_array), 2.5 * ts)

    # For good measure, add one more level of combination.
    var_array_2 = var_array + var_array
    dt_var_array = pp.ad.dt(var_array_2, time_step)
    assert np.allclose(equation_system.evaluate(dt_var_array), 5)

    # Also do a test of the mixed-dimensional variable.
    mvar = equation_system.md_variable("foo", [sd])

    dt_mvar = pp.ad.dt(mvar, time_step)
    assert np.allclose(equation_system.evaluate(dt_mvar)[: sd.num_cells], 2)
    assert np.allclose(equation_system.evaluate(dt_mvar)[sd.num_cells :], 0.5)

    # Test the time increment function
    diff_mvar = pp.ad.time_increment(mvar)
    assert np.allclose(equation_system.evaluate(diff_mvar)[: sd.num_cells], 2 * ts)
    assert np.allclose(equation_system.evaluate(diff_mvar)[sd.num_cells :], ts)

    # Make a combined operator with the mixed-dimensional variable, test this.
    dt_mvar = pp.ad.dt(mvar * mvar, time_step)
    assert np.allclose(equation_system.evaluate(dt_mvar)[: sd.num_cells], 4)
    assert np.allclose(equation_system.evaluate(dt_mvar)[sd.num_cells :], 0.5)


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


def test_ad_discretization_class():
    # Test of the parent class for all AD discretizations (pp.ad.Discretization)

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
    # This mimics the old generic AD discretization wrapper and still checks the
    # wrap_discretization utility directly.
    discr_ad = pp.ad.DiscretizationAd()
    discr_ad.subdomains = subdomains
    discr_ad._discretization = discr
    pp.ad.wrap_discretization(discr_ad, discr, mdg.dim_max(), subdomains)
    sub_discr_ad = pp.ad.DiscretizationAd()
    sub_discr_ad.subdomains = sub_list
    sub_discr_ad._discretization = sub_discr
    pp.ad.wrap_discretization(sub_discr_ad, sub_discr, mdg.dim_max(), sub_list)

    # values
    known_val = np.random.rand(len(subdomains))
    known_sub_val = np.random.rand(len(sub_list))

    # Assign a value to the discretization matrix, with the right key
    for vi, sd in enumerate(subdomains):
        data = mdg.subdomain_data(sd)
        data[pp.DISCRETIZATION_MATRICES] = {
            key: {"foobar": sps.csr_matrix(known_val[vi])}
        }

    # Same with submatrix
    for vi, sd in enumerate(sub_list):
        data = mdg.subdomain_data(sd)
        data[pp.DISCRETIZATION_MATRICES].update(
            {sub_key: {"foobar": sps.csr_matrix(known_sub_val[vi])}}
        )

    # Compare values under parsing. Note we need to pick out the diagonal, due to the
    # way parsing makes block matrices.
    assert np.allclose(known_val, discr_ad.foobar().parse(mdg).diagonal())
    assert np.allclose(known_sub_val, sub_discr_ad.foobar().parse(mdg).diagonal())


class _MockDiscretization:
    def __init__(self, key):
        self.foobar_matrix_key = "foobar"
        self.not_matrix_keys = "failed"

        self.keyword = key

    def get_row_dof_info(self, matrix_key: str = "", nd: int = 1):
        return {GridEntity.cells: 1}

    def get_col_dof_info(self, matrix_key: str = "", nd: int = 1):
        return {GridEntity.faces: 1}


# Arithmetic-combination tests were extracted to
# tests/numerics/ad/test_ad_arithmetic_operations.py.


@pytest.mark.parametrize(
    "generate_ad_list",
    [
        # All variables.
        lambda model: model.equation_system.get_variables(
            [model.pressure_variable, model.interface_darcy_flux_variable],
            model.mdg.subdomains(),
        ),
        # All the possible combinations of MDVariables.
        lambda model: [
            model.equation_system.md_variable(
                model.pressure_variable, model.mdg.subdomains()
            ),
            model.equation_system.md_variable(
                model.pressure_variable, [model.mdg.subdomains()[0]]
            ),
            model.equation_system.md_variable(
                model.pressure_variable, [model.mdg.subdomains()[1]]
            ),
            model.equation_system.md_variable(
                model.interface_darcy_flux_variable, model.mdg.interfaces()
            ),
        ],
        # Some randomly selected operators: Leaves and trees in the Ad operator graph
        # sense.
        lambda model: [
            model.ad_time_step,
            model.permeability(model.mdg.subdomains()),
            model.vector_source_darcy_flux(model.mdg.subdomains()),
            model.interface_darcy_flux_equation(
                model.mdg.interfaces(),
            ),
        ],
    ],
)
def test_hashing(generate_ad_list):
    """Tests the basic functionality regarding hashing.

    The identical AD objects defined on the same subdomains must return the same hash.
    It is assumed that the passed objects are all different, so their hashes must be
    different.

    """

    class Model(SquareDomainOrthogonalFractures, SinglePhaseFlow):
        """Mock-up model."""

    # With the default parameters, the model contains one fracture.
    model = Model({})
    model.prepare_simulation()

    ad_list = generate_ad_list(model)

    # Check that the hash remains the same.
    expected_hashes = []
    for ad in ad_list:
        expected_hashes.append(hash(ad))
    ad_list = generate_ad_list(model)  # Generate new (identical) AD objects.
    for ad, expected_hash in zip(ad_list, expected_hashes):
        assert hash(ad) == expected_hash

    # Check that the hashes are different for all the passed objects.
    for ad1 in ad_list:
        for ad2 in ad_list:
            if ad1 is not ad2:
                assert hash(ad1) != hash(ad2)
            else:
                assert hash(ad1) == hash(ad2)


@pytest.mark.parametrize(
    "two_spmatrices",
    [
        # *_matrix and *_array must have different hashes.
        [sps.coo_matrix(np.eye(2, 2)), sps.coo_array(np.eye(2, 2))],
        # Different sparse formats must have different hashes.
        [sps.csc_matrix(np.eye(2, 2)), sps.dia_matrix(np.eye(2, 2))],
        # Csr matrices with different `data` fields must have different hashes.
        [sps.csr_matrix([1, 0, 1, 0]), sps.csr_matrix([2, 0, 2, 0])],
        # Csr matrices with different `indices` fields must have different hashes.
        [sps.csr_matrix([1, 0, 1, 0]), sps.csr_matrix([0, 1, 0, 1])],
        # Csr matrices with different `indptr` fields must have different hashes.
        [sps.csr_matrix([[1], [0], [1], [0]]), sps.csr_matrix([[1], [1], [0], [0]])],
        # Csr matrices with different `shape` fields must have different hashes.
        [sps.csr_matrix([1, 0]), sps.csr_matrix([1, 0, 0])],
    ],
)
def test_hashing_sparse_array(two_spmatrices):
    """The hash function should account for these edge cases, when two matrices are
    almost identical, but it is crucial to distinguish between them.

    """
    space = pp.ad.OperatorSpace.unclear()
    m1, m2 = [
        pp.ad.SparseArray(mat, source=space, target=space) for mat in two_spmatrices
    ]
    assert hash(m1) != hash(m2)


class MockModelForCacheTesting:
    """A mock model for testing the caching of methods.

    The class has a number of methods with different signatures, all of which are
    decorated with @pp.ad.cached_method. The methods increment a counter each time the
    actual method body is executed (but *not* when the cache decorater reroutes the
    call), and return the current value of the counter.
    """

    def __init__(self):
        self._operator_cache = {}
        self._no_args_counter = 0
        self._one_arg_counter = 0
        self._two_args_counter = 0
        self._list_arg_counter = 0
        self._kwarg_counter = 0
        self._arg_and_kwarg_counter = 0

    @pp.ad.cached_method
    def method_with_no_arguments(self):
        self._no_args_counter += 1
        return self._no_args_counter

    @pp.ad.cached_method
    def method_with_one_arg(self, foo: int):
        self._one_arg_counter += 1
        return self._one_arg_counter

    @pp.ad.cached_method
    def method_with_two_args(self, foo: int, bar: int):
        self._two_args_counter += 1
        return self._two_args_counter

    @pp.ad.cached_method
    def method_with_list_arg(self, foo: list[int]):
        self._list_arg_counter += 1
        return self._list_arg_counter

    @pp.ad.cached_method
    def method_with_kwargs(self, *, foo: int = 1):
        self._kwarg_counter += 1
        return self._kwarg_counter

    @pp.ad.cached_method
    def method_with_arg_and_kwarg(self, foo: int, *, bar: int = 1):
        self._arg_and_kwarg_counter += 1
        return self._arg_and_kwarg_counter


class InheritedMockModel(MockModelForCacheTesting):
    def method_with_no_arguments(self):
        # A method with no caching at all.
        return 0

    def method_with_one_arg(self, foo: int):
        # A method that forwards the call to the parent class, which is cached.
        return super().method_with_one_arg(foo)


def test_operator_method_caching():
    model = MockModelForCacheTesting()

    # The model with no argument should be called once, then have its result cached.
    for _ in range(2):
        assert model.method_with_no_arguments() == 1

    # The model with a single argument should have its counter incremented each time it
    # is called with a different argument. When later called with the same argument, the
    # return should be the same as at the original call.
    for _ in range(2):
        assert model.method_with_one_arg(1) == 1
    assert model.method_with_one_arg(3) == 2
    assert model.method_with_one_arg(1) == 1

    # The model with two arguments should have its counter incremented each time it is
    # called with at least one argument not seen before. When later called with the same
    # arguments, the return should be the same as at the original call.
    for _ in range(2):
        assert model.method_with_two_args(1, 2) == 1
    assert model.method_with_two_args(3, 4) == 2
    assert model.method_with_two_args(1, 3) == 3
    assert model.method_with_two_args(2, 2) == 4
    assert model.method_with_two_args(1, 2) == 1
    # With the current implementation, passing the same arguments, but as keyword
    # arguments should trigger a new evaluation, but then the cache should be used.
    assert model.method_with_two_args(foo=1, bar=2) == 5
    assert model.method_with_two_args(foo=1, bar=2) == 5
    # When the keywords are passed in the opposite order, they will be found in the
    # cache.
    assert model.method_with_two_args(bar=2, foo=1) == 5

    # The model with a list argument should have its counter incremented each time it is
    # called with a different list argument. When later called with the same list
    # argument, the return should be the same as at the original call.
    for _ in range(2):
        assert model.method_with_list_arg([1, 2]) == 1
    assert model.method_with_list_arg([3, 4]) == 2
    assert model.method_with_list_arg([1, 2]) == 1

    # The model with a keyword argument should have its counter incremented each time it
    # is called with a different keyword argument. When later called with the same
    # keyword argument, the return should be the same as at the original call.
    for _ in range(2):
        assert model.method_with_kwargs(foo=1) == 1
    assert model.method_with_kwargs(foo=2) == 2
    assert model.method_with_kwargs(foo=1) == 1

    # Test method with a positional argument and a keyword argument.
    for _ in range(2):
        assert model.method_with_arg_and_kwarg(1, bar=2) == 1
    assert model.method_with_arg_and_kwarg(3, bar=4) == 2
    assert model.method_with_arg_and_kwarg(1, bar=2) == 1

    # Also test that the cache works for inherited classes.
    inherited_model = InheritedMockModel()

    for _ in range(2):
        # A method with no caching should behave the same time every time it is called.
        assert inherited_model.method_with_no_arguments() == 0

    # Test a method that forwards the call to the parent class, which is cached. The
    # behavior should be identical to the test of the parent class.
    for _ in range(2):
        assert inherited_model.method_with_one_arg(1) == 1
    assert inherited_model.method_with_one_arg(3) == 2
    assert inherited_model.method_with_one_arg(1) == 1

    # Test a method that is not overridden in the inherited class. The behavior should
    # be identical to the test of the parent class.
    for _ in range(2):
        assert inherited_model.method_with_list_arg([1, 2]) == 1
    assert inherited_model.method_with_list_arg([3, 4]) == 2
    assert inherited_model.method_with_list_arg([1, 2]) == 1
