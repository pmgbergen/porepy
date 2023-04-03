"""The module contains tests for evaluation of Ad operator trees through the
EquationSystem.

TODO: We may repurpose this module to a test of parsing of Ad Operators in general.

Contents:


"""
import numpy as np

import porepy as pp


def setup():
    mdg = pp.MixedDimensionalGrid()
    g_1 = pp.CartGrid([1, 1])
    g_2 = pp.CartGrid([1, 1])
    mdg.add_subdomains([g_1, g_2])

    eq_system = pp.ad.EquationSystem(mdg)

    # Define variables
    var_name = "foo"
    eq_system.create_variables(var_name, subdomains=[g_1, g_2])

    for sd, d in mdg.subdomains(return_data=True):
        d['stored_solutions'] = {var_name: {0: np.ones([sd.num_cells])}}
        d['stored_iterates'] = {var_name: {0: 2 * np.ones([sd.num_cells])}}

    return eq_system


def test_evaluate_variables():
    """Test that the combination of an Ad operator tree and the EquationSystem correctly
    evaluates variables at the current iterate, previous iterate and previous time step.

    The testing is based on parsing of individual variables and of the result of the
    time increment method. The time derivative method is not tested, since this is
    essentially the same as the time increment method (only division by dt is added).
    """
    eq_system = setup()

    # We only need to test a single variable, they should all be the same.
    single_variable = eq_system.variables[0]
    # Make a md wrapper around the variable.
    md_variable = eq_system.md_variable(single_variable.name)

    # Try testing both the individual variable and its mixed-dimensional wrapper.
    # In the below checks, the anticipated value of the variables is based on the
    # values assigned in the setup method.
    for var in [single_variable, md_variable]:

        # First evaluate the variable. This should give the iterate value.
        val = var.evaluate(eq_system)
        assert isinstance(val, pp.ad.AdArray)
        assert np.allclose(val.val, 2)

        # Now create the variable at the previous iterate. This should also give the
        # value in 'stored_iterates', but it should not yield an AdArray.
        var_prev_iter = var.previous_iteration()
        val_prev_iter = var_prev_iter.evaluate(eq_system)
        assert isinstance(val_prev_iter, np.ndarray)
        assert np.allclose(val_prev_iter, 2)

        # Create the variable at the previous time step. This should give the value in
        # 'stored_solutions'.
        var_prev_timestep = var.previous_timestep()
        val_prev_timestep = var_prev_timestep.evaluate(eq_system)
        assert isinstance(val_prev_timestep, np.ndarray)
        assert np.allclose(val_prev_timestep, 1)

        # Use the ad machinery to define the difference between the current and
        # previous time step. This should give an AdArray with the same value as that
        # obtained by subtracting the evaluated variables.
        var_increment = pp.ad.time_increment(var)
        val_increment = var_increment.evaluate(eq_system)
        assert isinstance(val_increment, pp.ad.AdArray)
        assert np.allclose(val_increment.val, val.val - val_prev_timestep)
