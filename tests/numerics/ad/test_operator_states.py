"""The module contains tests of the mixin classes for ad operators that make them
time-dependent, iterative and/or reference operators.

The tests are collected in two classes:
 - TestTimeDependentAndIterative: Tests that the time-dependent and iterative mixins
    behave correctly independently and in combination.
 - TestReferenceOperator: Tests that the reference operator behaves correctly for
    different ad operators.
See the respective classes and their test methods for details.
"""

import copy

import numpy as np
import pytest

import porepy as pp
from porepy.numerics.ad.equation_system import GridEntity


def grid():
    """Provide a 2x2 Cartesian grid for testing. There is no immediate need for a more
    complex grid, nor for a truly mixed-dimensional grid, since the functionality to be
    tested should be agnostic to the grid.
    """
    g = pp.CartGrid([2, 2])
    mdg = pp.MixedDimensionalGrid()
    mdg.add_subdomains(g)
    return mdg


@pytest.mark.parametrize("time_dependent", [True, False])
class TestTimeDependentAndIterative:
    """Test that the mixins that make operators time-dependent and iterative behave
    correctly independently and in combination.

    See the individual test methods for details.

    Regarding coverage, we consider both variables and time dependent dense arrays.
    Other operators should be covered by these cases.

    """

    def setup_method(self):
        """Set up the problem."""
        self.mdg = grid()
        self.equation_system = pp.ad.EquationSystem(self.mdg)

        # NOTE: We use the same name for both the variable and the dense array to avoid
        # more if-else statements when setting and fetching values and expected results.
        self.name = "foo"
        self.var = self.equation_system.create_variables(
            self.name, dof_info={GridEntity.cells: 1}, subdomains=self.mdg.subdomains()
        )
        self.dense_arr = pp.ad.TimeDependentDenseArray(
            name=self.name, domains=self.mdg.subdomains()
        )
        # Recursion depth for the time and iterative operators.
        self.depth = 4
        # Populate the data arrays.
        for i in range(self.depth - 1):
            for sd, data in self.mdg.subdomains(return_data=True):
                for index_key in ["time_step_index", "iterate_index"]:
                    pp.set_solution_values(
                        name=self.name,
                        values=i * np.ones(sd.num_cells),
                        data=data,
                        **{index_key: i},
                    )

    def _get_index_keys(self, time_dependent: bool):
        """Helper method to get the correct index keys and previous methods for time
        dependent and iterative operators.
        """
        if time_dependent:
            index_key = "time_step_index"
            other_index_key = "iterate_index"
            prev_method = "previous_timestep"
            other_prev_method = "previous_iteration"
        else:
            index_key = "iterate_index"
            other_index_key = "time_step_index"
            prev_method = "previous_iteration"
            other_prev_method = "previous_timestep"

        return index_key, other_index_key, prev_method, other_prev_method

    def _get_operator(self, operator_type: str):
        if operator_type == "dense":
            return self.dense_arr
        else:
            return self.var

    def test_time_and_iterative_combined_raises_on_variable(self, time_dependent: bool):
        """Check that, for variables, we cannot call both previous_timestep and
        previous_iteration on the same operator (independent on the order of the calls).

        The test does not apply to dense arrays, which essentially give a void operation
        when calling previous_iterate, hence there is no problem calling previous
        timestep afterwards.

        """
        _, _, prev_method, other_prev_method = self._get_index_keys(time_dependent)
        with pytest.raises(ValueError):
            var_pt = getattr(self.var, prev_method)()
            _ = getattr(var_pt, other_prev_method)()

    @pytest.mark.parametrize("operator_type", ["variable", "dense"])
    def test_time_and_iterative_repeated_calls(
        self, time_dependent: bool, operator_type: str
    ):
        """Verify the behavior of repeated calls to previous_timestep and
        previous_iteration."""
        operator = self._get_operator(operator_type)

        _, _, prev_method, _ = self._get_index_keys(time_dependent)

        op = getattr(operator, prev_method)(steps=0)
        val_current = self.equation_system.evaluate(op)

        for i in range(0, self.depth - 1):
            op_prev = getattr(op, prev_method)(steps=i + 1)
            val_prev = self.equation_system.evaluate(op_prev, derivative=True)
            if operator_type == "dense" and not time_dependent:
                # The dense array is not a function of the current approximation, so
                # the jacobian should be zero.
                np.testing.assert_allclose(val_prev.jac.data, 0)
            else:
                np.testing.assert_allclose(val_prev.val, i)
            if i > 0:
                np.testing.assert_allclose(val_prev.jac.data, 0)

    @pytest.mark.parametrize("operator_type", ["variable", "dense"])
    def test_time_and_iterative_recursive_vs_direct(
        self, time_dependent: bool, operator_type: str
    ):
        """Verify that calling previous_timestep or previous_iteration recursively
        gives the same result as calling it with the correct number of steps directly.
        """
        operator = self._get_operator(operator_type)

        _, _, prev_method, _ = self._get_index_keys(time_dependent)

        # Test creating with explicit stepping and recursive stepping.
        vars_exp = [
            getattr(operator, prev_method)(steps=i) for i in range(0, self.depth)
        ]
        vars_rec = []
        for i in range(0, self.depth):
            var_i = copy.copy(operator)
            for _ in range(i):
                var_i = getattr(var_i, prev_method)()
            vars_rec.append(var_i)

        assert len(vars_exp) == len(vars_rec)
        vals_exp = [self.equation_system.evaluate(v) for v in vars_exp]
        vals_rec = [self.equation_system.evaluate(v) for v in vars_rec]

        for v_e, v_r in zip(vals_exp, vals_rec):
            assert np.allclose(v_e, v_r)


operator_types = ["dense", "variable", "combined"]
state_list = ["current", "previous_timestep", "previous_iteration"]


class TestReferenceOperator:
    """Tests reference-state behavior for AD operators.

    The following terminology is used in parametrization:
      - op_type: Which operator type to test. Can be "dense"
        (pp.ad.TimeDependentDenseArray), "variable" (pp.ad.MixedDimensionalVariable) or
        "combined" (an operator tree formed as a combination of the two previous
        types).
      - state: Which state of the operator to test. Can be "current" (the current
        approximation), "previous_timestep" or "previous_iteration", where the two
        latter will be obtained by calling the corresponding methods on the operator.
      - change_type: How to change the reference values before evaluating the operator
        (only relevant for the test_shifting_reference_values test). Can be "delete"
        (delete the reference value so that the default 0 should be obtained), "set"
        (explicitly set the reference value) or "shift" (shift the reference value from
        the current approximation).

    With this terminology, the following tests are performed:
      - test_reference_on_current_timelag_and_iterates: Test that the reference operator
        evaluates correctly for all operator types and states.
      - test_shifting_reference_values: Test that the methods for changing the reference
        values works correctly for all operator types, states and change types.
      - test_perturbation_from_reference: Test that the perturbation from reference
        operator evaluates correctly for all operator types and states.
      - test_time_difference_of_reference_is_zero: Test that the time difference of a
        reference operator is zero.

    """

    def setup_method(self):
        self.mdg = grid()
        self.equation_system = pp.ad.EquationSystem(self.mdg)

        self.num_cells = sum(sd.num_cells for sd in self.mdg.subdomains())

        # Declear a variable and a dense array to use in the tests. NOTE: Use the *same
        # name* for both fields to avoid more if-else statements when setting and
        # fetching values and expected results.
        name = "foo"
        self.equation_system.create_variables(
            name, dof_info={GridEntity.cells: 1}, subdomains=self.mdg.subdomains()
        )
        self.var = self.equation_system.md_variable(name=name)
        self.dense_arr = pp.ad.TimeDependentDenseArray(
            name=name, domains=self.mdg.subdomains()
        )

        # The default value for the reference operator. This is set equal to the default
        # in the method used to evaluate reference states of operators (in the code
        # proper).
        self.default_reference_value = 0
        # The values to use for the current approximation, previous time step and
        # previous iteration.
        self.iter_value = 1
        self.time_value = 2
        # The value to use for the reference operator.
        self.ref_val = 3
        # Multiplier used in the combined operator.
        self.combined_multiplier = 2.0

        # Set values for the variable/dense array.
        for sd, sd_data in self.mdg.subdomains(return_data=True):
            vec = np.ones(sd.num_cells)
            pp.set_solution_values(
                name=name, values=self.time_value * vec, data=sd_data, time_step_index=0
            )
            pp.set_solution_values(
                name=name, values=self.iter_value * vec, data=sd_data, iterate_index=0
            )
            pp.set_solution_values(
                name=name,
                values=self.iter_value * vec,
                data=sd_data,
                iterate_index=0,
            )
            pp.set_solution_values(
                name=name, values=self.ref_val * vec, data=sd_data, reference=True
            )

    def _operator(self, op_type):
        """For a given operator type, return the corresponding operator."""
        if op_type == "dense":
            return self.dense_arr
        elif op_type == "variable":
            return self.var
        elif op_type == "combined":
            return self.var + pp.ad.Scalar(self.combined_multiplier) * self.dense_arr
        else:
            raise NotImplementedError(f"Operator type {op_type} not implemented.")

    def _operator_to_state(self, op_type, state):
        """Helper method to get the operator to the correct state."""
        op = self._operator(op_type)
        return getattr(op, state)() if state is not "current" else op

    def _expected_reference(self, op_type):
        """Helper method to get the expected reference value for a given operator
        type."""
        # The reference value is ref_val for the dense and variable operators, but not
        # for the combined operator. Send ref_val through the
        # _adjust_combined_expectance method to get the correct expected value also for
        # the combined operator.
        return self._adjust_combined_expectance(self.ref_val, op_type)

    def _adjust_combined_expectance(self, val, op_type):
        """Helper method to adjust the expected value for a combined operator."""
        # Adjust the value for the combined operator, all other operators are unchanged.
        if op_type == "combined":
            # This mirrors the logic in the method _operator, where the combined
            # operator is defined.
            val *= 1 + self.combined_multiplier
        return val

    def _expected(self, op_type, state):
        """Helper method to get the expected value and jacobian for a given operator
        type and state. Assumes that the reference values have not been changed - in
        that case, use the _change_reference_values method to administer the change and
        get the expected value.
        """
        # Set the expected value according to the operator type and state. This mirrors
        # the logic in the setup_method's loop over the subdomains where the values are
        # set.
        if state == "current":
            val = self.iter_value
        elif state == "previous_timestep":
            val = self.time_value
        elif state == "previous_iteration":
            val = self.iter_value
        else:
            raise NotImplementedError(f"State {state} not implemented.")
        # If this is the combined operator, adjust the expected value accordingly.
        val = self._adjust_combined_expectance(val, op_type)

        # The derivative is only non-zero for operators that contain variables and are
        # at the current state.
        if (op_type == "variable" or op_type == "combined") and state == "current":
            # The derivative is non-zero only in this case.
            jac_val = 1.0
        else:
            jac_val = 0.0

        return val, jac_val

    def _change_reference_values_return_expected(self, op_type, change_type):
        """Helper method to administer changes to the reference values and return the
        expected value according to the change type.
        """
        vec = np.ones(self.num_cells)

        if change_type is "delete":
            for _, data in self.mdg.subdomains(return_data=True):
                del data[pp.REFERENCE_SOLUTIONS][self.var.name]
                val = self.default_reference_value
        elif change_type == "set":
            # Explicitly set the reference values.
            self.equation_system.set_variable_values(
                vec * self.time_value, [self.var], reference=True
            )
            val = self.time_value
        elif change_type == "shift":
            # Shift current approximation to reference values.
            for _, data in self.mdg.subdomains(return_data=True):
                pp.shift_solution_values(self.var.name, data, pp.REFERENCE_SOLUTIONS)
            val = self.iter_value
        return self._adjust_combined_expectance(val, op_type)

    @pytest.mark.parametrize("op_type", operator_types)
    @pytest.mark.parametrize("state", state_list)
    def test_reference_on_current_timelag_and_iterates(self, op_type, state):
        """Taking the reference should give the same result for standard, time and
        iterative operators.

        See class docstring for details on terminology.

        """
        op_ref = self._operator_to_state(op_type, state).reference()
        val = self.equation_system.evaluate(op_ref, derivative=True)
        np.testing.assert_allclose(val.val, self._expected_reference(op_type))
        # The Jacobian should always be zero, since the reference operator is not a
        # function of the current approximation.
        np.testing.assert_allclose(val.jac.data, 0.0)

    @pytest.mark.parametrize("op_type", operator_types)
    @pytest.mark.parametrize("state", state_list)
    @pytest.mark.parametrize("change_type", ["delete", "set", "shift"])
    def test_shifting_reference_values(self, op_type, state, change_type):
        """Test that the various ways of changing the reference values works correctly
        for all operator types and states.

        See class docstring for details on terminology.

        """
        op_ref = self._operator_to_state(op_type, state).reference()
        # Here we cannot use the standard _expected method, but instead call the
        # _change_reference_values method, which administers the change and returns the
        # expected value according to the change type.
        expected = self._change_reference_values_return_expected(op_type, change_type)

        val = self.equation_system.evaluate(op_ref, derivative=True)
        np.testing.assert_allclose(val.val, expected)
        np.testing.assert_allclose(val.jac.data, 0.0)

    @pytest.mark.parametrize("op_type", operator_types)
    @pytest.mark.parametrize("state", state_list)
    def test_perturbation_from_reference(self, op_type, state):
        """Test that the perturbation from reference operator evaluates correctly for
        all operator types and states.

        See class docstring for details on terminology.

        """
        op = self._operator_to_state(op_type, state)
        pert = op.perturbation_from_reference()

        pert_val = self.equation_system.evaluate(pert, derivative=True)
        expected_val, expected_jac = self._expected(op_type, state)

        np.testing.assert_allclose(
            pert_val.val, expected_val - self._expected_reference(op_type)
        )
        np.testing.assert_allclose(pert_val.jac.data, expected_jac)

    @pytest.mark.parametrize("op_type", operator_types)
    def test_time_difference_of_reference_is_zero(self, op_type):
        """Test that the time difference of a reference operator is zero.

        See class docstring for details on terminology.

        """
        op = self._operator_to_state(op_type, "current")
        ref_increment = pp.ad.time_increment(op.reference())
        val_ref_increment = self.equation_system.evaluate(ref_increment)
        assert np.allclose(val_ref_increment, 0.0)
