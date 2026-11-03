"""Test collection testing Ad operator functions.

Testing involves their creation, arithmetic overloads and parsing.

"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.ad.equation_system import GridEntity
from porepy.numerics.ad.operators import OperatorSpace


def test_ad_function():
    """Tests involving the Ad wrapper for analytically represented functions.

    Note:
        Value and Jacobian of various analytically represented functions are covered
        in test_functions.py.

        This test tests the AD wrapper ad.Function and its wrapping functionality using
        the identity function.

    """

    func = lambda x: x  # identity
    domain = OperatorSpace.scalar()

    F = pp.ad.Function(func, "identity", domain, domain)

    grid = pp.CartGrid(np.array([3, 2]))
    mdg = pp.meshing.subdomains_to_mdg([grid])
    mdg.compute_geometry()

    subdomains = mdg.subdomains()
    equation_system = pp.ad.EquationSystem(mdg)
    equation_system.create_variables("foo", {GridEntity.cells: 1}, subdomains)

    var = equation_system.md_variable("foo", subdomains)

    # setting values at current time, previous time and previous iter
    vals = np.ones(mdg.num_subdomain_cells())
    equation_system.set_variable_values(vals, [var], iterate_index=0)
    equation_system.set_variable_values(vals * 2, [var], iterate_index=1)
    equation_system.set_variable_values(vals * 10, [var], time_step_index=0)

    # test that the function without call with operator is inoperable
    for op in ["*", "/", "+", "-", "**", "@"]:
        with pytest.raises(TypeError):
            _ = eval(f"F {op} var")
        with pytest.raises(TypeError):
            _ = eval(f"var {op} F")

    F_var = F(var)

    val_ad = F_var.value_and_jacobian(equation_system)
    # test values at current time step
    assert np.all(val_ad.val == 1.0)
    assert np.all(val_ad.sparse_jac.toarray() == np.eye(mdg.num_subdomain_cells()))

    # vals at previous iter and zero Jacobian
    # previous iterate has the same values as the original operator, but no Jacobian
    F_var_pi = F_var.previous_iteration()
    val = F_var_pi.value_and_jacobian(equation_system)
    assert np.all(val.val == val_ad.val)
    assert np.all(val.jac.toarray() == 0.0)

    # 1 iterate before has the respective values
    F_var_pii = F_var_pi.previous_iteration()
    val = F_var_pii.value_and_jacobian(equation_system)
    assert np.all(val.val == 2.0)
    assert np.all(val.jac.toarray() == 0.0)

    # Analogously for prev time
    F_var_pt = F_var.previous_timestep()
    val = F_var_pt.value_and_jacobian(equation_system)
    assert np.all(val.val == 10.0)
    assert np.all(val.jac.toarray() == 0.0)

    # when evaluating with values only, the result should be a numpy array
    val = equation_system.evaluate(F_var)
    val_pi = equation_system.evaluate(F_var_pi)
    val_pt = equation_system.evaluate(F_var_pt)
    assert isinstance(val, np.ndarray)
    assert isinstance(val_pi, np.ndarray)
    assert isinstance(val_pt, np.ndarray)


def test_diagonal_jacobian_function_accepts_both_representations():
    """The approximate Jacobian of a DiagonalJacobianFunction sums the scaled Jacobians
    of all Ad arguments, whatever their representation."""

    class Sum(pp.ad.DiagonalJacobianFunction):
        def get_values(self, *args):
            return sum(a.val for a in args)

    domain = OperatorSpace.scalar()
    # Use a DiagonalJacobianFunction to approximate the function f(x, y) = 2x + 3y.
    function = Sum([2.0, 3.0], "sum", source=domain)

    # The arguments are two variables, each with two degrees of freedom, in a system
    # with four degrees of freedom in total. The first variable, x, is in the sparse
    # representation and has degrees of freedom 0 and 1, thus its Jacobian is the
    # identity in the first two columns. The second variable, y, is in the diagonal
    # representation and has degrees of freedom 2 and 3.
    x = pp.ad.SparseAdArray(np.array([1.0, 2.0]), sps.csr_matrix(np.eye(2, 4)))
    y = pp.ad.initialize_diagonal_ad_arrays(
        [np.array([3.0, 4.0])],  # Variable values.
        [np.array([2, 3])],  # Dof indices of the variable in the global system.
        4,  # Total number of degrees of freedom in the system.
    )[0]

    jac = function.get_jacobian(x, y)

    # As per the design of the DiagonalJacobianFunction, the Jacobian should be 2 times
    # the identity in the columns of x, and 3 times the identity in the columns of y.
    expected = np.array([[2.0, 0.0, 3.0, 0.0], [0.0, 2.0, 0.0, 3.0]])
    np.testing.assert_allclose(jac.toarray(), expected)
