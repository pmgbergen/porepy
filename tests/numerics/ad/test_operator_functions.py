"""Test collection testing Ad operator functions.

Testing involves their creation, arithmetic overloads and parsing.

"""
from __future__ import annotations

import pytest

import numpy as np
import porepy as pp


def test_ad_function():
    """Tests involving the Ad wrapper for analytically represented functions.
    
    Note:
        Value and Jacobian of various analytically represented functions are covered
        in test_functions.py.
        
        This test tests the AD wrapper ad.Function and its wrapping functionality using
        the identity function.
    
    """

    func = lambda x: x  # identity

    F = pp.ad.Function(func, 'identity')

    g = pp.CartGrid(np.array([3, 2]))
    mdg = pp.meshing.subdomains_to_mdg([g])
    mdg.compute_geometry()

    sds = mdg.subdomains()
    eqs = pp.ad.EquationSystem(mdg)
    eqs.create_variables('foo', {'cells': 1}, sds)

    var = eqs.md_variable('foo', sds)

    # setting values at current time, previous time and previous iter
    vals = np.ones(mdg.num_subdomain_cells())
    eqs.set_variable_values(vals, [var], iterate_index=0)
    eqs.set_variable_values(vals * 2, [var], iterate_index=1)
    eqs.set_variable_values(vals * 10, [var], time_step_index=0)

    # test that the function without call with operator is inoperable
    for op in ['*', '/', '+', '-', '**']:
        with pytest.raises(TypeError):
            _ = eval(f"F {op} var")

    F_var = F(var)

    val = F_var.value_and_jacobian(eqs)
    # test values at current time step
    assert np.all(val.val == 1.)
    assert np.all(val.jac.A == np.eye(mdg.num_subdomain_cells()))

    # vals at previous iter and zero Jacobian
    F_var_pi = F_var.previous_iteration()
    val = F_var_pi.value_and_jacobian(eqs)
    assert np.all(val.val == 2.)
    assert np.all(val.jac.A == 0.)

    # Analogously for prev time
    F_var_pt = F_var.previous_timestep()
    val = F_var_pt.value_and_jacobian(eqs)
    assert np.all(val.val == 10.)
    assert np.all(val.jac.A == 0.)

    # when evaluating with values only, the result should be a numpy array
    val = F_var.value(eqs)
    val_pi = F_var_pi.value(eqs)
    val_pt = F_var_pt.value(eqs)
    assert isinstance(val, np.ndarray)
    assert isinstance(val_pi, np.ndarray)
    assert isinstance(val_pt, np.ndarray)