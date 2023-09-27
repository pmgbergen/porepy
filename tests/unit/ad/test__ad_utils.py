"""Test suite for ad utils.

The main checks performed are:
    test_get_set_solution_values: Test of functionality related to setting and getting
        values from the data dictionary.

"""

import numpy as np
import pytest

import porepy as pp


def test_get_set_solution_values():
    data = {}

    values_0 = np.linspace(0, 10, 11)
    values_1 = 2 * np.linspace(0, 10, 11)
    values_2 = 3 * np.linspace(0, 10, 11)
    values_3 = 4 * np.linspace(0, 10, 11)

    parameter_name = "parameter_name"

    pp.set_solution_values(
        name=parameter_name, values=values_0, data=data, time_step_index=0
    )
    pp.set_solution_values(
        name=parameter_name, values=values_1, data=data, iterate_index=0
    )
    pp.set_solution_values(
        name=parameter_name, values=values_2, data=data, time_step_index=1
    )
    pp.set_solution_values(
        name=parameter_name, values=values_3, data=data, iterate_index=1
    )

    # Values gathered from data dictionary directly
    values_0_gathered_directly = data[pp.TIME_STEP_SOLUTIONS][parameter_name][0]
    values_1_gathered_directly = data[pp.ITERATE_SOLUTIONS][parameter_name][0]
    values_2_gathered_directly = data[pp.TIME_STEP_SOLUTIONS][parameter_name][1]
    values_3_gathered_directly = data[pp.ITERATE_SOLUTIONS][parameter_name][1]

    # Values gathered with the function
    values_0_test_get = pp.get_solution_values(
        name=parameter_name, data=data, time_step_index=0
    )
    values_1_test_get = pp.get_solution_values(
        name=parameter_name, data=data, iterate_index=0
    )
    values_2_test_get = pp.get_solution_values(
        name=parameter_name, data=data, time_step_index=1
    )
    values_3_test_get = pp.get_solution_values(
        name=parameter_name, data=data, iterate_index=1
    )

    # Checking if values gathered from the data dictionary are as expected.
    # The values gathered directly from the data dirctionary should be equal to both the
    # values set by pp.set_solution_values() and the values gathered by
    # pp.get_solution_values().
    assert (values_0_gathered_directly == values_0).all()
    assert (values_1_gathered_directly == values_1).all()
    assert (values_2_gathered_directly == values_2).all()
    assert (values_3_gathered_directly == values_3).all()

    assert (values_0_test_get == values_0_gathered_directly).all()
    assert (values_1_test_get == values_1_gathered_directly).all()
    assert (values_2_test_get == values_2_gathered_directly).all()
    assert (values_3_test_get == values_3_gathered_directly).all()

    # Check that the expected errors are raised
    with pytest.raises(ValueError):
        # Try to set values without assigning time step index or iterate index.
        pp.set_solution_values(name=parameter_name, values=values_0, data=data)

    with pytest.raises(ValueError):
        # Try to fetch values without assigning time step index or iterate index.
        pp.get_solution_values(name=parameter_name, data=data)
    with pytest.raises(ValueError):
        # Try to fetch values by assigning value to both time step index and iterate
        # index.
        pp.get_solution_values(
            name=parameter_name, data=data, time_step_index=0, iterate_index=0
        )

    with pytest.raises(KeyError):
        # Try to fetch values associated with a parameter name not found in data.
        pp.get_solution_values(name="parameter_name_2", data=data, time_step_index=0)
    with pytest.raises(KeyError):
        # Try to fetch values associated with a time step index that is not assigned.
        pp.get_solution_values(name=parameter_name, data=data, time_step_index=2)
    with pytest.raises(KeyError):
        # Try to fetch values associated with a iterate index that is not assigned.
        pp.get_solution_values(name=parameter_name, data=data, iterate_index=2)
