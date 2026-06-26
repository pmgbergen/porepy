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


op_type = ["dense", "variable"]


class TestReferenceOperator:
    def setup_method(self):
        self.mdg = grid()
        self.equation_system = pp.ad.EquationSystem(self.mdg)

        self.num_cells = sum(sd.num_cells for sd in self.mdg.subdomains())

        name = "foo"
        self.equation_system.create_variables(
            name, dof_info={"cells": 1}, subdomains=self.mdg.subdomains()
        )
        self.var = self.equation_system.md_variable(name=name)

        self.default_val = 0
        self.iter_value = 1
        self.time_value = 2
        self.ref_val = 3

        self.dense_arr = pp.ad.TimeDependentDenseArray(
            name=name, domains=self.mdg.subdomains()
        )

        name = self.var.name
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

    def _expected(self, op_type, state):
        jac_val = 1.0 if op_type == "variable" and state == "current" else 0.0

        if state == "current":
            return self.iter_value, jac_val
        elif state == "previous_timestep":
            return self.time_value, jac_val
        elif state == "previous_iteration":
            return self.iter_value, jac_val
        else:
            raise NotImplementedError(f"State {state} not implemented.")

    def _change_reference_values(self, state_change):
        vec = np.ones(self.num_cells)

        if state_change is None:
            # TODO: Should we rather erase here, so that the reference will return
            # default?
            for _, data in self.mdg.subdomains(return_data=True):
                del data[pp.REFERENCE_SOLUTIONS][self.var.name]
                return 0
        elif state_change == "set":
            # Explicitly set the reference values.
            self.equation_system.set_variable_values(
                vec * self.time_value, [self.var], reference=True
            )
            return self.time_value
        elif state_change == "shift":
            # Shift current approximation to reference values.
            for _, data in self.mdg.subdomains(return_data=True):
                pp.shift_solution_values(self.var.name, data, pp.REFERENCE_SOLUTIONS)
            return self.iter_value

    def _operator(self, op_type):
        if op_type == "dense":
            return self.dense_arr
        elif op_type == "variable":
            return self.var
        elif op_type == "combined":
            return self.dense_arr + pp.ad.Scalar(2.0) * self.var
        else:
            raise NotImplementedError(f"Operator type {op_type} not implemented.")

    def _operator_to_state(self, op_type, state):
        op = self._operator(op_type)

        return getattr(op, state)() if state is not "current" else op

    @pytest.mark.parametrize("op_type", ["variable", "dense"])
    @pytest.mark.parametrize(
        "state", ["current", "previous_timestep", "previous_iteration"]
    )
    def test_reference_on_time_and_increments(self, op_type, state):
        # Taking the reference should give the same result for standard, time and
        # iterative operators
        op_ref = self._operator_to_state(op_type, state).reference()

        val = self.equation_system.evaluate(op_ref, derivative=True)
        np.testing.assert_allclose(val.val, self.ref_val)
        np.testing.assert_allclose(val.jac.data, 0.0)

    @pytest.mark.parametrize("op_type", ["variable", "dense"])
    @pytest.mark.parametrize(
        "state", ["current", "previous_timestep", "previous_iteration"]
    )
    @pytest.mark.parametrize("state_change", [None, "set", "shift"])
    def test_shifting_reference_values(self, op_type, state, state_change):
        # Shifting reference values should give the same result for standard, time and
        # iterative operators
        op_ref = self._operator_to_state(op_type, state).reference()
        expected = self._change_reference_values(state_change)

        val = self.equation_system.evaluate(op_ref, derivative=True)
        np.testing.assert_allclose(val.val, expected)
        np.testing.assert_allclose(val.jac.data, 0.0)

    @pytest.mark.parametrize("op_type", ["variable", "dense"])
    @pytest.mark.parametrize(
        "state", ["current", "previous_timestep", "previous_iteration"]
    )
    def test_perturbation_from_reference(self, op_type, state):
        # Perturbation from reference should give the same result for standard, time and
        # iterative operators
        op = self._operator_to_state(op_type, state)
        pert = op.perturbation_from_reference()

        pert_val = self.equation_system.evaluate(pert, derivative=True)
        expected_val, expected_jac = self._expected(op_type, state)
        np.testing.assert_allclose(pert_val.val, expected_val - self.ref_val)
        np.testing.assert_allclose(pert_val.jac.data, expected_jac)

    def test_time_difference_of_reference_is_zero(self):
        # The time difference of a reference operator should be zero
        ref_increment = pp.ad.time_increment(self.var.reference())
        val_ref_increment = self.equation_system.evaluate(ref_increment)
        assert np.allclose(val_ref_increment, 0.0)
