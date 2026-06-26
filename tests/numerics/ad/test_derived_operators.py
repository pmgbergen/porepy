import pytest
import porepy as pp
import numpy as np
import copy


@pytest.fixture(scope="module")
def mdg():
    grid, _ = pp.mdg_library.square_with_orthogonal_fractures(
        "cartesian",
        {"cell_size": 0.2},
        fracture_indices=[1],
    )
    return grid


def grid():
    g = pp.CartGrid([2, 2])
    mdg = pp.MixedDimensionalGrid()
    mdg.add_subdomains(g)
    return mdg


def get_operator(case, mdg):

    equation_system = pp.ad.EquationSystem(mdg)

    name = "foo"
    for sd, sd_data in mdg.subdomains(return_data=True):
        vals_sol = np.ones(sd.num_cells)
        pp.set_solution_values(
            name=name, values=vals_sol, data=sd_data, time_step_index=0
        )
        vals_it = 2 * np.ones(sd.num_cells)
        pp.set_solution_values(name=name, values=vals_it, data=sd_data, iterate_index=0)

    for intf, data in mdg.interfaces(return_data=True):
        vals_sol = np.ones(intf.num_cells)
        pp.set_solution_values(name=name, values=vals_sol, data=data, time_step_index=0)
        vals_it = 2 * np.ones(intf.num_cells)
        pp.set_solution_values(name=name, values=vals_it, data=data, iterate_index=0)

    match case:
        case "dense":
            op = pp.ad.TimeDependentDenseArray(name=name, domains=mdg.subdomains())
        case "sparse":
            op = pp.ad.SparseArray(name=name, domains=mdg.subdomains())
        case "variable":
            equation_system.create_variables(
                name, dof_info={"cells": 1}, subdomains=mdg.subdomains()
            )
            op = equation_system.md_variable(name=name)
        case "combined":
            raise NotImplementedError("Combined operators are not yet supported.")
        case _:
            raise NotImplementedError("Combined operators are not yet supported.")

    return op, equation_system


cases = ["dense", "variable"]


@pytest.mark.parametrize("time_dependent", [True, False])
class TestTimeDependentAndIterative:
    def _get_index_keys(self, time_dependent: bool):
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

    def test_time_and_iterative_combined_raises(self, time_dependent: bool):
        mdg = grid()
        operator, _ = get_operator("variable", mdg)

        index_key, other_index_key, prev_method, other_prev_method = (
            self._get_index_keys(time_dependent)
        )
        # prohibit prev time step variable to also be prev iter
        with pytest.raises(ValueError):
            var_pt = getattr(operator, prev_method)()
            _ = getattr(var_pt, other_prev_method)()

    def test_time_and_iterative_recursive(self, time_dependent: bool):
        mdg = grid()
        operator, equation_system = get_operator("variable", mdg)

        name = operator.name

        index_key, _, prev_method, _ = self._get_index_keys(time_dependent)

        depth = 4

        for i in range(depth - 1):
            for sd, data in mdg.subdomains(return_data=True):
                pp.set_solution_values(
                    name=operator.name,
                    values=i * np.ones(sd.num_cells),
                    data=data,
                    **{index_key: i},
                )

        for sd, data in mdg.subdomains(return_data=True):
            op = getattr(operator, prev_method)(steps=0)
            val_current = equation_system.evaluate(op)
            # np.testing.assert_allclose(val_current, 0)

            for i in range(0, depth - 1):
                op_prev = getattr(op, prev_method)(steps=i + 1)
                val_prev = equation_system.evaluate(op_prev, derivative=True)
                np.testing.assert_allclose(val_prev.val, i)
                if i > 0:
                    np.testing.assert_allclose(val_prev.jac.data, 0)

    def test_time_and_iterative_recursive_vs_direct(self, time_dependent: bool):
        mdg = grid()
        var, equation_system = get_operator("variable", mdg)

        name = var.name

        index_key, _, prev_method, _ = self._get_index_keys(time_dependent)

        depth = 4

        for i in range(depth - 1):
            for sd, data in mdg.subdomains(return_data=True):
                pp.set_solution_values(
                    name=var.name,
                    values=i * np.ones(sd.num_cells),
                    data=data,
                    **{index_key: i},
                )

        # Test creating with explicit stepping and recursive stepping
        vars_exp = [getattr(var, prev_method)(steps=i) for i in range(0, depth)]

        vars_rec = []
        for i in range(0, depth):
            var_i = copy.copy(var)
            for _ in range(i):
                var_i = getattr(var_i, prev_method)()
            vars_rec.append(var_i)

        assert len(vars_exp) == len(vars_rec)
        vals_exp = [equation_system.evaluate(v) for v in vars_exp]
        vals_rec = [equation_system.evaluate(v) for v in vars_rec]

        for v_e, v_r in zip(vals_exp, vals_rec):
            assert np.allclose(v_e, v_r)


operator_types = ["dense", "variable", "combined"]
state_list = ["current", "previous_timestep", "previous_iteration"]


class TestReferenceOperator:
    """Tests reference-state behavior for AD operators.

    The following terminology is used in parametrization:
      - op_type: Which operator type to test. Can be "dense"
        (pp.ad.TimeDependentDenseArray), "variable" (pp.ad.MixedDimensionalVariable) or
        "combined" (an operator three formed as a combination of the two previous
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
            name, dof_info={"cells": 1}, subdomains=self.mdg.subdomains()
        )
        self.var = self.equation_system.md_variable(name=name)
        self.dense_arr = pp.ad.TimeDependentDenseArray(
            name=name, domains=self.mdg.subdomains()
        )

        # The default value for the reference operator.
        self.default_reference_value = 0
        # The values to use for the current approximation, previous time step and previous
        # iteration.
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

    def _expected_reference(self, op_type, state):
        """Helper method to get the expected reference value for a given operator type
        and state.
        """
        return self._adjust_combined_expectence(self.ref_val, op_type)

    def _adjust_combined_expectence(self, val, op_type):
        """Helper method to adjust the expected value for a combined operator."""
        if op_type == "combined":
            val *= 1 + self.combined_multiplier
        return val

    def _expected(self, op_type, state):
        """Helper method to get the expected value and jacobian for a given operator
        type and state. Assumes that the reference values have not been changed - in
        that case, use the _change_reference_values method to administer the change and
        get the expected value.
        """
        if (op_type == "variable" or op_type == "combined") and state == "current":
            # The derivative is non-zero only in this case.
            jac_val = 1.0
        else:
            jac_val = 0.0

        if state == "current":
            val = self.iter_value
        elif state == "previous_timestep":
            val = self.time_value
        elif state == "previous_iteration":
            val = self.iter_value
        else:
            raise NotImplementedError(f"State {state} not implemented.")

        val = self._adjust_combined_expectence(val, op_type)
        return val, jac_val

    def _change_reference_values(self, op_type, change_type):
        """Helper method to administer changes to the reference values and return the
        expected value according to the change type.
        """
        vec = np.ones(self.num_cells)

        if change_type is "delete":
            # TODO: Should we rather erase here, so that the reference will return
            # default?
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
        return self._adjust_combined_expectence(val, op_type)

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

    @pytest.mark.parametrize("op_type", operator_types)
    @pytest.mark.parametrize("state", state_list)
    def test_reference_on_current_timelag_and_iterates(self, op_type, state):
        # Taking the reference should give the same result for standard, time and
        # iterative operators
        op_ref = self._operator_to_state(op_type, state).reference()

        val = self.equation_system.evaluate(op_ref, derivative=True)
        np.testing.assert_allclose(val.val, self._expected_reference(op_type, state))
        # The jacobian should always be zero, since the reference operator is not a
        # function of the current approximation.
        np.testing.assert_allclose(val.jac.data, 0.0)

    @pytest.mark.parametrize("op_type", operator_types)
    @pytest.mark.parametrize("state", state_list)
    @pytest.mark.parametrize("change_type", ["delete", "set", "shift"])
    def test_shifting_reference_values(self, op_type, state, change_type):
        op_ref = self._operator_to_state(op_type, state).reference()
        expected = self._change_reference_values(op_type, change_type)

        val = self.equation_system.evaluate(op_ref, derivative=True)
        np.testing.assert_allclose(val.val, expected)
        np.testing.assert_allclose(val.jac.data, 0.0)

    @pytest.mark.parametrize("op_type", operator_types)
    @pytest.mark.parametrize("state", state_list)
    def test_perturbation_from_reference(self, op_type, state):
        op = self._operator_to_state(op_type, state)
        pert = op.perturbation_from_reference()

        pert_val = self.equation_system.evaluate(pert, derivative=True)
        expected_val, expected_jac = self._expected(op_type, state)

        np.testing.assert_allclose(
            pert_val.val, expected_val - self._expected_reference(op_type, state)
        )
        np.testing.assert_allclose(pert_val.jac.data, expected_jac)

    @pytest.mark.parametrize("op_type", operator_types)
    def test_time_difference_of_reference_is_zero(self, op_type):
        op = self._operator_to_state(op_type, "current")
        ref_increment = pp.ad.time_increment(op.reference())
        val_ref_increment = self.equation_system.evaluate(ref_increment)
        assert np.allclose(val_ref_increment, 0.0)
