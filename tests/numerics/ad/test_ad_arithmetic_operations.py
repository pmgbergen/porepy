"""Tests of arithmetic operations between Ad objects.

Scalars, dense arrays, sparse matrices and Ad arrays, in both the sparse and the
diagonal representation, are combined pairwise with the operators +, -, *, /, ** and
@. Each combination is evaluated both in forward mode, on the numerical values directly,
and wrapped as Ad operators that are parsed by an EquationSystem. The result is
compared with a value computed independently, or, if the combination is not supported,
it is checked that the evaluation fails.

The module is laid out from the test down to the helpers it uses:
    1. The test function and its fixture.
    2. Functions that construct the operands, in wrapped and unwrapped form.
    3. Simple helpers for comparison and type checks.
    4. The computation of expected values. The entry point, _expected_value, comes
       first. It hands each combination of operand types to a function named
       _expected_value_<types> if further dispatch is needed, and otherwise to a
       function named _expected_<types> that computes the value. The case of two Ad
       arrays in wrapped form, which needs special treatment, comes last.

"""

from typing import Literal, Union

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.ad.equation_system import GridEntity

# The types of the operands, and of the values that operations between them evaluate to.
AdType = Union[float, np.ndarray, sps.spmatrix, sps.sparray, pp.ad.AdArrayBase]

# The expected value of an operation is the value its evaluation should give, or False
# if the operation is not supported, that is, if its evaluation should raise an error.
# No supported operation evaluates to a bool, so False is never a valid value.
ExpectedValue = Union[Literal[False], AdType]


@pytest.fixture
def arithmetic_mdg() -> pp.MixedDimensionalGrid:
    """Mixed-dimensional grid with a single subdomain of three cells, on which the
    operands of the tests are defined."""
    g = pp.CartGrid([3, 1])
    mdg = pp.MixedDimensionalGrid()
    mdg.add_subdomains([g])
    return mdg


@pytest.mark.parametrize(
    "var_1", ["scalar", "dense", "sparse_matrix", "sparse_array", "ad", "diag"]
)
@pytest.mark.parametrize(
    "var_2", ["scalar", "dense", "sparse_matrix", "sparse_array", "ad", "diag"]
)
@pytest.mark.parametrize("op", ["+", "-", "*", "/", "**", "@"])
@pytest.mark.parametrize("wrapped", [True, False])
def test_arithmetic_operations_on_ad_objects(
    arithmetic_mdg: pp.MixedDimensionalGrid,
    var_1: str,
    var_2: str,
    op: str,
    wrapped: bool,
) -> None:
    """Test that the fundamental Ad operators can be combined using the standard
    arithmetic operations.

    All combinations of operands and operations are formed, in two different modes:
    Wrapped as Ad operators (subclasses of pp.ad.Operator), or as primitive values
    (float, numpy.ndarray, scipy sparse matrices and arrays, and the two Ad array
    representations SparseAdArray and DiagonalAdArray). In the wrapped form, all
    combinations are tested, since the parsing of the operator tree must handle each of
    them. In the primitive form, which is what is applied in forward-mode algorithmic
    differentiation, only combinations that involve at least one Ad array are tested;
    the remaining combinations are evaluated by numpy and scipy alone.

    For each combination, the result is compared with an expected value, or, if the
    combination is not supported, it is checked that the evaluation raises an error.

    """

    has_ad_operand = var_1 in ["ad", "diag"] or var_2 in ["ad", "diag"]
    if not wrapped and not has_ad_operand:
        # If not wrapped in the abstract layer, these cases should be covered by the
        # tests for the external packages; PorePy just has to rely on e.g., numpy being
        # correctly implemented. For the wrapped case, we need to test that the parsing
        # is okay, thus we do not skip if wrapped is True.
        return

    # In the wrapped form, expressions are evaluated by an EquationSystem, which must
    # know the variables of all Ad operands. The Ad operands are therefore created as
    # separate variables in a common EquationSystem. If none of the operands is an Ad
    # array, an EquationSystem is still formally needed to evaluate the expression, but
    # since it will not actually be used for anything, it is defined on an empty grid.
    equation_system = (
        pp.ad.EquationSystem(arithmetic_mdg)
        if wrapped and has_ad_operand
        else pp.ad.EquationSystem(pp.MixedDimensionalGrid())
    )

    def _var_from_string(v: str, do_wrap: bool, variable_name: str):
        if v == "scalar":
            return _get_scalar(do_wrap)
        if v == "dense":
            return _get_dense_array(do_wrap, arithmetic_mdg)
        if v == "sparse_matrix":
            return _get_sparse_array(do_wrap, use_csr_matrix=True, mdg=arithmetic_mdg)
        if v == "sparse_array":
            return _get_sparse_array(do_wrap, use_csr_matrix=False, mdg=arithmetic_mdg)
        if v == "ad":
            return _get_ad_array(
                do_wrap,
                arithmetic_mdg,
                equation_system=equation_system if do_wrap else None,
                variable_name=variable_name,
            )
        if v == "diag":
            return _get_diag_array(
                do_wrap,
                arithmetic_mdg,
                equation_system=equation_system if do_wrap else None,
                variable_name=variable_name,
            )
        raise ValueError("Unknown variable type")

    # Get the actual variables from the input strings.
    v1 = _var_from_string(var_1, wrapped, "foo_v1")
    v2 = _var_from_string(var_2, wrapped, "foo_v2")

    # Calculate the expected numerical values for this expression from the unwrapped
    # operands. If the operation is not expected to succeed, the expected value is
    # False. When both wrapped operands are Ad arrays, they are formed from different
    # variables, which the expected value must account for, see _expected_wrapped_value.
    v1_as_value = _var_from_string(var_1, False, "foo_v1")
    v2_as_value = _var_from_string(var_2, False, "foo_v2")
    if wrapped and var_1 in ["ad", "diag"] and var_2 in ["ad", "diag"]:
        expected = _expected_wrapped_value(v1_as_value, v2_as_value, op)
    else:
        expected = _expected_value(v1_as_value, v2_as_value, op)

    # Evaluate the expression. This is a bit different for the wrapped and forward mode,
    # but the logic is the same: Try to evaluate. If this breaks, check that the
    # operation is not supported, that is, that the expected value is False.
    if wrapped:
        try:
            # The idea here is to test evaluation on the deepest level, i.e., the method
            # _evaluate_single in the AdParser. This is the method that actually
            # translates an expression into a numerical value. An error here signifies
            # that something is wrong with the parsing itself. Note that testing of the
            # frontend evaluation is done below (calls to equation_system.evaluate()),
            # as well as in the test of equation_system.py and other tests.
            expression = eval(f"v1 {op} v2")
            state = equation_system.get_variable_values(time_step_index=0)
            ad_base = equation_system._ad_parser._initialize_variables(
                [expression],
                state,
                equation_system.variable_indexer,
                derivative=True,
            )
            val = equation_system._ad_parser._evaluate_single(
                expression,
                ad_base,
                equation_system,
                variable_indexer=equation_system.variable_indexer,
            )
        except (TypeError, ValueError, NotImplementedError):
            assert expected is False
            return
    else:
        try:
            val = eval(f"v1 {op} v2")
        except (TypeError, ValueError, NotImplementedError):
            assert expected is False
            return

    # The evaluation succeeded, which it should only do for supported operations.
    assert expected is not False, (
        f"The operation {var_1} {op} {var_2} is not supported, but was evaluated."
    )

    # Compare numerical values between evaluated and expected outcomes.
    _compare(val, expected)

    # Finally, test that evaluation with and without derivatives give the same value.
    # If the expected value is multidimensional, the Jacobian is not implemented.
    try:
        multidimensional = len(expected.shape) > 1
    except AttributeError:
        multidimensional = False

    if wrapped:
        if not multidimensional:
            val_jac = equation_system.evaluate(expression, derivative=True)
            val = equation_system.evaluate(expression)
            np.testing.assert_array_equal(val_jac.val, val)
        else:
            with pytest.raises(
                NotImplementedError,
                match="not implemented because it is multidimensional",
            ):
                equation_system.evaluate(expression, derivative=True)


def _get_scalar(wrapped: bool) -> float | pp.ad.Scalar:
    """Helper to set a scalar. Expected values in the test are hardcoded with respect to
    this value. The scalar is either returned as-is, or wrapped as an Ad scalar."""
    scalar = 2.0
    if wrapped:
        return pp.ad.Scalar(scalar)
    return scalar


def _get_dense_array(wrapped: bool, mdg) -> np.ndarray | pp.ad.DenseArray:
    """Helper to set a dense array (numpy array). Expected values in the test are
    hardcoded with respect to this value. The array is either returned as-is, or wrapped
    as an Ad DenseArray."""
    array = np.array([1, 2, 3]).astype(float)
    space = pp.ad.OperatorSpace.from_domains(mdg.subdomains())
    if wrapped:
        return pp.ad.DenseArray(array, source=space, target=space)
    return array


def _get_sparse_array(
    wrapped: bool, use_csr_matrix: bool, mdg
) -> sps.spmatrix | sps.sparray | pp.ad.SparseArray:
    """Helper to set a sparse array (scipy sparse array). Expected values in the test
    are hardcoded with respect to this value. The array is either returned as-is, or
    wrapped as an Ad SparseArray."""
    inner = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    mat = sps.csr_matrix(inner) if use_csr_matrix else sps.csr_array(inner)
    mat = mat.astype(float)

    space = pp.ad.OperatorSpace.from_domains(mdg.subdomains())
    if wrapped:
        return pp.ad.SparseArray(mat, source=space, target=space)
    return mat


def _get_ad_array(
    wrapped: bool,
    mdg: pp.MixedDimensionalGrid,
    equation_system: pp.ad.EquationSystem | None = None,
    variable_name: str = "foo_ad",
) -> pp.ad.SparseAdArray | pp.ad.Operator:
    """Get a SparseAdArray object which can be used in the tests.

    Parameters:
        wrapped: If True, return an Ad operator that evaluates to the array. If False,
            return the array itself.
        mdg: Grid on which the array is defined.
        equation_system: Only used if wrapped is True, and then required. The equation
            system in which the variable of the operator is created, and which must be
            used to evaluate the operator.
        variable_name: Name of the variable created if wrapped is True. Operands that
            are evaluated in the same equation system must have different names.

    """

    # The construction between the wrapped and unwrapped case differs significantly: For
    # the latter we can simply create a SparseAdArray with any value and Jacobian
    # matrix. The former must be processed through the operator parsing framework, and
    # thus puts stronger conditions on permissible states. The below code defines a
    # variable (variable_val), a matrix (jac), and constructs an expression as jac @
    # variable. This expression is represented in the returned SparseAdArray, either
    # directly or (if wrapped=True) on abstract form.
    #
    # If this is confusing, it may be helpful to recall that a SparseAdArray can
    # represent any state, not only primary variables (e.g., a pp.ad.Variable). The main
    # motivation for using a more complex value is that the Jacobian matrix of primary
    # variables are identity matrices, thus compound expressions give higher chances of
    # uncovering errors.

    # This is the value of the variable
    variable_val = np.ones(3)
    # This is the Jacobian matrix of the returned expression.
    jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    # This is the expression to be used in the tests. The numerical values of val will
    # be np.array([6, 15, 24]), and its Jacobian matrix is jac.
    expression_val = jac @ variable_val

    if wrapped:
        assert equation_system is not None
        equation_system.create_variables(variable_name, subdomains=mdg.subdomains())
        var = equation_system.variables[-1]
        d = mdg.subdomain_data(mdg.subdomains()[0])

        pp.set_solution_values(
            name=variable_name, values=variable_val, data=d, time_step_index=0
        )
        pp.set_solution_values(
            name=variable_name, values=variable_val, data=d, iterate_index=0
        )
        space = pp.ad.OperatorSpace.from_domains(mdg.subdomains())
        mat = pp.ad.SparseArray(jac, source=space, target=space)

        return mat @ var

    return pp.ad.SparseAdArray(expression_val, jac)


def _get_diag_array(
    wrapped: bool,
    mdg: pp.MixedDimensionalGrid,
    equation_system: pp.ad.EquationSystem | None = None,
    variable_name: str = "foo_diag",
) -> pp.ad.DiagonalAdArray | pp.ad.Operator:
    """Get a DiagonalAdArray object which can be used in the tests.

    The array has the same value as the one returned by _get_ad_array, [6, 15, 24], but
    a diagonal Jacobian, with diagonal [6, 7.5, 8]. In the wrapped form, it is
    constructed as the elementwise product of a dense array holding the diagonal and a
    variable with value [1, 2, 3]. The parameters are the same as for _get_ad_array.

    """
    variable_val = np.array([1, 2, 3])
    jac = np.array([6, 7.5, 8])

    if wrapped:
        assert equation_system is not None
        equation_system.create_variables(variable_name, subdomains=mdg.subdomains())
        var = equation_system.variables[-1]
        d = mdg.subdomain_data(mdg.subdomains()[0])
        pp.set_solution_values(
            name=variable_name, values=variable_val, data=d, time_step_index=0
        )
        pp.set_solution_values(
            name=variable_name, values=variable_val, data=d, iterate_index=0
        )

        space = pp.ad.OperatorSpace.from_domains(mdg.subdomains())
        vec = pp.ad.DenseArray(jac, source=space, target=space)

        return vec * var

    return pp.ad.DiagonalAdArray(
        jac * variable_val,
        jac,
        col_indices=[np.arange(variable_val.size)],
        num_derivatives=3,
    )


def _compare(actual: AdType, expected: AdType) -> None:
    """Assert that an evaluated object equals the expected one, both in type and in
    value.

    For Ad arrays, the values and the Jacobians are compared, but not the
    representation: The expected value is always a SparseAdArray, while the evaluated
    one may be a DiagonalAdArray.

    Parameters:
        actual: The value obtained by evaluating an operation.
        expected: The value the evaluation should give. The case of an unsupported
            operation, with expected value False, must be handled by the caller.

    Raises:
        AssertionError: If the types differ, other than in the representation of Ad
            arrays, or the values differ beyond round-off.
        ValueError: If the values are of a type not covered by the test.

    """
    if not (_is_ad(actual) and _is_ad(expected)):
        assert type(actual) is type(expected)
    # Entries that are zero in the expected value may be affected by round-off in the
    # evaluated one, so a small absolute tolerance is needed.
    atol = 1e-12
    if isinstance(actual, (float, np.ndarray)):
        np.testing.assert_allclose(actual, expected, atol=atol)
    elif isinstance(actual, (sps.spmatrix, sps.sparray)):
        np.testing.assert_allclose(actual.toarray(), expected.toarray(), atol=atol)
    elif _is_ad(actual):
        np.testing.assert_allclose(actual.val, expected.val, atol=atol)
        np.testing.assert_allclose(
            _jac_to_dense(actual), _jac_to_dense(expected), atol=atol
        )
    else:
        raise ValueError(f"Unknown type: {type(actual)}")


def _is_sparse(var: AdType) -> bool:
    return isinstance(var, (sps.spmatrix, sps.sparray))


def _is_ad(var: AdType) -> bool:
    return isinstance(var, pp.ad.AdArrayBase)


def _is_nonad(var: AdType) -> bool:
    return isinstance(var, (float, np.ndarray, sps.spmatrix, sps.sparray))


def _jac_to_dense(ad: pp.ad.AdArrayBase) -> np.ndarray:
    return ad.sparse_jac.toarray()


def _create_adarray(
    val: np.ndarray, jac: np.ndarray | sps.spmatrix
) -> pp.ad.SparseAdArray:
    """Create a SparseAdArray from a value and a Jacobian, where the Jacobian is either
    a sparse matrix, or a numpy array holding the diagonal of a diagonal matrix."""
    if isinstance(jac, np.ndarray):
        jac = sps.dia_matrix((jac, [0]), shape=(jac.size, jac.size))
    return pp.ad.SparseAdArray(val, jac)


def _expected_value(
    var_1: AdType, var_2: AdType, op: Literal["+", "-", "*", "/", "**", "@"]
) -> ExpectedValue:
    """Expected value of var_1 op var_2, for operands in unwrapped form.

    The work is handed to a function for each combination of operand types:
        - Neither operand is an Ad array: _expected_no_ad. The unsupported combinations
          are listed explicitly, and all other combinations are evaluated by Python,
          numpy and scipy.
        - Both operands are Ad arrays: _expected_ad_ad, which applies the rules of
          differentiation to the values and Jacobians of the operands. The matrix
          product is not supported.
        - One operand is an Ad array: _expected_value_ad_nonad or
          _expected_value_nonad_ad, depending on which. These dispatch further on the
          type of the other operand, to functions with values computed by hand for the
          operands returned by the _get_* functions, in both Ad array representations.

    Parameters:
        var_1: Left operand.
        var_2: Right operand.
        op: The operation.

    Raises:
        ValueError: If an operand is of a type not covered by the test.

    Returns:
        The value that the evaluation of var_1 op var_2 should give, or False if the
        operation is not supported, that is, if the evaluation should raise an error.

    """
    if _is_ad(var_1) and _is_ad(var_2):
        if op == "@":
            return False
        return _expected_ad_ad(var_1, var_2, op)

    if _is_ad(var_1):
        if not _is_nonad(var_2):
            raise ValueError(f"Unknown classes: {type(var_1)}, {type(var_2)}.")
        return _expected_value_ad_nonad(var_1, var_2, op)

    if _is_ad(var_2):
        if not _is_nonad(var_1):
            raise ValueError(f"Unknown classes: {type(var_1)}, {type(var_2)}.")
        return _expected_value_nonad_ad(var_1, var_2, op)

    if _is_nonad(var_1) and _is_nonad(var_2):
        return _expected_no_ad(var_1, var_2, op)

    raise ValueError(f"Unknown classes: {type(var_1)}, {type(var_2)}.")


def _expected_no_ad(
    var_1: float | np.ndarray | sps.spmatrix | sps.sparray,
    var_2: float | np.ndarray | sps.spmatrix | sps.sparray,
    op: Literal["+", "-", "*", "/", "**", "@"],
) -> ExpectedValue:
    """Expected value of an operation between two operands that are not Ad arrays.

    Combinations that PorePy does not support are returned as False. All other
    combinations are evaluated by Python, numpy and scipy, and the evaluation is
    required to succeed.

    """
    if op == "@" and (isinstance(var_1, float) or isinstance(var_2, float)):
        # The matrix product is not defined for scalars.
        return False
    if _is_sparse(var_1) and op == "**":
        # SciPy has implemented a limited version of matrix powers, but not with a
        # satisfactory flexibility. If we try to evaluate the expression, it may or may
        # not work (see comments in the __pow__ method of Operators), but the operation
        # is anyhow explicitly disallowed.
        return False
    if isinstance(var_1, float) and _is_sparse(var_2) and op != "*":
        return False
    if isinstance(var_1, np.ndarray) and _is_sparse(var_2) and op in ["/", "**"]:
        return False
    if _is_sparse(var_1) and isinstance(var_2, float) and op not in ["*", "/"]:
        return False

    return eval(f"var_1 {op} var_2")


def _expected_value_ad_nonad(
    var_1: pp.ad.AdArrayBase,
    var_2: float | np.ndarray | sps.spmatrix | sps.sparray,
    op: Literal["+", "-", "*", "/", "**", "@"],
) -> Literal[False] | pp.ad.SparseAdArray:
    """Expected value of var_1 op var_2 for an Ad array var_1 and an operand var_2 that
    is not an Ad array."""
    if op == "@":
        # We disallow this operation due to the following reason: The only case in
        # which it is convenient to use the @ operator for the scalars is a generic
        # operator that can accept everything: scalars, ndarrays and sparse matrices.
        # However, we do not know any cases where one operator can be both a scalar or
        # ndarray AND a matrix. This choice will be reconsidered if a reasonable example
        # is found.
        return False
    if isinstance(var_2, float):
        return _expected_ad_scalar(var_1, op)
    if isinstance(var_2, np.ndarray):
        return _expected_ad_dense(var_1, var_2, op)
    # Ad array op sparse matrix is not supported for any operation.
    return False


def _expected_value_nonad_ad(
    var_1: float | np.ndarray | sps.spmatrix | sps.sparray,
    var_2: pp.ad.AdArrayBase,
    op: Literal["+", "-", "*", "/", "**", "@"],
) -> Literal[False] | pp.ad.SparseAdArray:
    """Expected value of var_1 op var_2 for an operand var_1 that is not an Ad array and
    an Ad array var_2."""
    if _is_sparse(var_1):
        return _expected_sparse_ad(var_1, var_2, op)
    if op == "@":
        # See the comment for the case Ad array @ scalar in _expected_value_ad_nonad.
        return False
    if isinstance(var_1, float):
        return _expected_scalar_ad(var_2, op)
    return _expected_dense_ad(var_1, var_2, op)


def _expected_ad_ad(
    var_1: pp.ad.AdArrayBase,
    var_2: pp.ad.AdArrayBase,
    op: Literal["+", "-", "*", "/", "**"],
) -> pp.ad.SparseAdArray:
    """Expected value of var_1 op var_2 for two Ad arrays, with the Jacobian computed
    row by row from the sum, product, quotient and power rules."""
    j1 = _jac_to_dense(var_1)
    j2 = _jac_to_dense(var_2)

    if op == "+":
        val = var_1.val + var_2.val
        jac = j1 + j2
    elif op == "-":
        val = var_1.val - var_2.val
        jac = j1 - j2
    elif op == "*":
        val = var_1.val * var_2.val
        jac = np.vstack(
            (
                j1[0] * var_2.val[0] + var_1.val[0] * j2[0],
                j1[1] * var_2.val[1] + var_1.val[1] * j2[1],
                j1[2] * var_2.val[2] + var_1.val[2] * j2[2],
            )
        )
    elif op == "/":
        val = var_1.val / var_2.val
        jac = np.vstack(
            (
                j1[0] / var_2.val[0] - var_1.val[0] * j2[0] / var_2.val[0] ** 2,
                j1[1] / var_2.val[1] - var_1.val[1] * j2[1] / var_2.val[1] ** 2,
                j1[2] / var_2.val[2] - var_1.val[2] * j2[2] / var_2.val[2] ** 2,
            )
        )
    else:  # op == "**"
        val = var_1.val**var_2.val
        jac = np.vstack(
            (
                var_2.val[0] * var_1.val[0] ** (var_2.val[0] - 1.0) * j1[0]
                + np.log(var_1.val[0]) * (var_1.val[0] ** var_2.val[0]) * j2[0],
                var_2.val[1] * var_1.val[1] ** (var_2.val[1] - 1.0) * j1[1]
                + np.log(var_1.val[1]) * (var_1.val[1] ** var_2.val[1]) * j2[1],
                var_2.val[2] * var_1.val[2] ** (var_2.val[2] - 1.0) * j1[2]
                + np.log(var_1.val[2]) * (var_1.val[2] ** var_2.val[2]) * j2[2],
            )
        )
    return pp.ad.SparseAdArray(val, sps.csr_matrix(jac))


def _expected_ad_scalar(
    var_1: pp.ad.AdArrayBase, op: Literal["+", "-", "*", "/", "**"]
) -> pp.ad.SparseAdArray:
    """Expected value of var_1 op 2.0, for the Ad array returned by _get_ad_array or
    _get_diag_array."""
    # var_1 has value [6, 15, 24]. Its Jacobian has rows [1, 2, 3], [4, 5, 6],
    # [7, 8, 9] if var_1 is a SparseAdArray, and diagonal [6, 7.5, 8] if it is a
    # DiagonalAdArray.
    if op == "+":
        val = np.array([8, 17, 26])
        if isinstance(var_1, pp.ad.DiagonalAdArray):
            jac = np.array([6, 7.5, 8])
        else:
            jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    elif op == "-":
        val = np.array([4, 13, 22])
        if isinstance(var_1, pp.ad.DiagonalAdArray):
            jac = np.array([6, 7.5, 8])
        else:
            jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    elif op == "*":
        val = np.array([12, 30, 48])
        if isinstance(var_1, pp.ad.DiagonalAdArray):
            jac = np.array([12, 15, 16])
        else:
            jac = sps.csr_matrix(np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]]))
    elif op == "/":
        val = np.array([6 / 2, 15 / 2, 24 / 2])
        if isinstance(var_1, pp.ad.DiagonalAdArray):
            jac = np.array([6 / 2, 7.5 / 2, 8 / 2])
        else:
            jac = sps.csr_matrix(
                np.array(
                    [
                        [1 / 2, 2 / 2, 3 / 2],
                        [4 / 2, 5 / 2, 6 / 2],
                        [7 / 2, 8 / 2, 9 / 2],
                    ]
                )
            )
    else:  # op == "**"
        val = np.array([6**2, 15**2, 24**2])
        if isinstance(var_1, pp.ad.DiagonalAdArray):
            jac = 2 * var_1.val * var_1.jac
        else:
            jac = sps.csr_matrix(
                2
                * np.vstack(
                    (
                        var_1.val[0] * var_1.jac[0].toarray(),
                        var_1.val[1] * var_1.jac[1].toarray(),
                        var_1.val[2] * var_1.jac[2].toarray(),
                    )
                ),
            )
    return _create_adarray(val, jac)


def _expected_ad_dense(
    var_1: pp.ad.AdArrayBase, var_2: np.ndarray, op: Literal["+", "-", "*", "/", "**"]
) -> pp.ad.SparseAdArray:
    """Expected value of var_1 op var_2, for the Ad array returned by _get_ad_array or
    _get_diag_array, and the dense array returned by _get_dense_array."""
    # var_1 has value [6, 15, 24]. Its Jacobian has rows [1, 2, 3], [4, 5, 6],
    # [7, 8, 9] if var_1 is a SparseAdArray, and diagonal [6, 7.5, 8] if it is a
    # DiagonalAdArray. The operand var_2 is the dense array [1, 2, 3].
    if op == "+":
        val = np.array([7, 17, 27])
        if isinstance(var_1, pp.ad.DiagonalAdArray):
            jac = np.array([6, 7.5, 8])
        else:
            jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    elif op == "-":
        val = np.array([5, 13, 21])
        if isinstance(var_1, pp.ad.DiagonalAdArray):
            jac = np.array([6, 7.5, 8])
        else:
            jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    elif op == "*":
        val = np.array([6, 30, 72])
        if isinstance(var_1, pp.ad.DiagonalAdArray):
            jac = np.array([6 * 1, 7.5 * 2, 8 * 3])
        else:
            jac = sps.csr_matrix(np.array([[1, 2, 3], [8, 10, 12], [21, 24, 27]]))
    elif op == "/":
        val = np.array([6 / 1, 15 / 2, 24 / 3])
        if isinstance(var_1, pp.ad.DiagonalAdArray):
            jac = np.array([6 / 1, 7.5 / 2, 8 / 3])
        else:
            jac = sps.csr_matrix(
                np.vstack(
                    (
                        var_1.jac[0].toarray() / var_2[0],
                        var_1.jac[1].toarray() / var_2[1],
                        var_1.jac[2].toarray() / var_2[2],
                    )
                )
            )
    else:  # op == "**"
        # The derivative is Array**(var_2 - 1) * var_2 * dArray
        val = np.array([6, 15**2, 24**3])
        if isinstance(var_1, pp.ad.DiagonalAdArray):
            jac = var_2 * (var_1.val ** (var_2 - 1.0)) * var_1.jac
        else:
            jac = sps.csr_matrix(
                np.vstack(
                    (
                        var_2[0]
                        * (var_1.val[0] ** (var_2[0] - 1.0))
                        * var_1.jac[0].toarray(),
                        var_2[1]
                        * (var_1.val[1] ** (var_2[1] - 1.0))
                        * var_1.jac[1].toarray(),
                        var_2[2]
                        * (var_1.val[2] ** (var_2[2] - 1.0))
                        * var_1.jac[2].toarray(),
                    )
                )
            )
    return _create_adarray(val, jac)


def _expected_scalar_ad(
    var_2: pp.ad.AdArrayBase, op: Literal["+", "-", "*", "/", "**"]
) -> pp.ad.SparseAdArray:
    """Expected value of 2.0 op var_2, for the Ad array returned by _get_ad_array or
    _get_diag_array."""
    # var_2 has value [6, 15, 24]. Its Jacobian has rows [1, 2, 3], [4, 5, 6],
    # [7, 8, 9] if var_2 is a SparseAdArray, and diagonal [6, 7.5, 8] if it is a
    # DiagonalAdArray.
    if op == "+":
        val = np.array([8, 17, 26])
        if isinstance(var_2, pp.ad.DiagonalAdArray):
            jac = np.array([6, 7.5, 8])
        else:
            jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    elif op == "-":
        val = np.array([-4, -13, -22])
        if isinstance(var_2, pp.ad.DiagonalAdArray):
            jac = -np.array([6, 7.5, 8])
        else:
            jac = sps.csr_matrix(np.array([[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]]))
    elif op == "*":
        val = np.array([12, 30, 48])
        if isinstance(var_2, pp.ad.DiagonalAdArray):
            jac = np.array([12, 15, 16])
        else:
            jac = sps.csr_matrix(np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]]))
    elif op == "/":
        # The derivative is -2 / Array**2 * dArray
        val = np.array([2 / 6, 2 / 15, 2 / 24])
        if isinstance(var_2, pp.ad.DiagonalAdArray):
            jac = -2 / var_2.val**2 * var_2.jac
        else:
            jac = sps.csr_matrix(
                np.vstack(
                    (
                        -2 / var_2.val[0] ** 2 * var_2.jac[0].toarray(),
                        -2 / var_2.val[1] ** 2 * var_2.jac[1].toarray(),
                        -2 / var_2.val[2] ** 2 * var_2.jac[2].toarray(),
                    )
                ),
            )
    else:  # op == "**"
        # The derivative is 2**Array * log(2) * dArray
        val = np.array([2**6, 2**15, 2**24])
        if isinstance(var_2, pp.ad.DiagonalAdArray):
            jac = np.log(2.0) * (2**var_2.val) * var_2.jac
        else:
            jac = sps.csr_matrix(
                np.vstack(
                    (
                        np.log(2.0) * (2 ** var_2.val[0]) * var_2.jac[0].toarray(),
                        np.log(2.0) * (2 ** var_2.val[1]) * var_2.jac[1].toarray(),
                        np.log(2.0) * (2 ** var_2.val[2]) * var_2.jac[2].toarray(),
                    )
                ),
            )
    return _create_adarray(val, jac)


def _expected_dense_ad(
    var_1: np.ndarray, var_2: pp.ad.AdArrayBase, op: Literal["+", "-", "*", "/", "**"]
) -> pp.ad.SparseAdArray:
    """Expected value of var_1 op var_2, for the dense array returned by
    _get_dense_array, and the Ad array returned by _get_ad_array or _get_diag_array."""
    # var_2 has value [6, 15, 24]. Its Jacobian has rows [1, 2, 3], [4, 5, 6],
    # [7, 8, 9] if var_2 is a SparseAdArray, and diagonal [6, 7.5, 8] if it is a
    # DiagonalAdArray. The operand var_1 is the dense array [1, 2, 3].
    if op == "+":
        val = np.array([7, 17, 27])
        if isinstance(var_2, pp.ad.DiagonalAdArray):
            jac = np.array([6, 7.5, 8])
        else:
            jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    elif op == "-":
        val = np.array([-5, -13, -21])
        if isinstance(var_2, pp.ad.DiagonalAdArray):
            jac = -np.array([6, 7.5, 8])
        else:
            jac = sps.csr_matrix(np.array([[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]]))
    elif op == "*":
        val = np.array([6, 30, 72])
        if isinstance(var_2, pp.ad.DiagonalAdArray):
            jac = np.array([6 * 1, 7.5 * 2, 8 * 3])
        else:
            jac = sps.csr_matrix(np.array([[1, 2, 3], [8, 10, 12], [21, 24, 27]]))
    elif op == "/":
        # The derivative is -var_1 / Array**2 * dArray
        val = np.array([1 / 6, 2 / 15, 3 / 24])
        if isinstance(var_2, pp.ad.DiagonalAdArray):
            jac = -var_1 / var_2.val**2 * var_2.jac
        else:
            jac = sps.csr_matrix(
                np.vstack(
                    (
                        -var_1[0] * var_2.jac[0].toarray() / var_2.val[0] ** 2,
                        -var_1[1] * var_2.jac[1].toarray() / var_2.val[1] ** 2,
                        -var_1[2] * var_2.jac[2].toarray() / var_2.val[2] ** 2,
                    )
                )
            )
    else:  # op == "**"
        # The derivative is var_1**Array * log(var_1) * dArray
        val = np.array([1, 2**15, 3**24])
        if isinstance(var_2, pp.ad.DiagonalAdArray):
            jac = (var_1**var_2.val) * np.log(var_1) * var_2.jac
        else:
            jac = sps.csr_matrix(
                np.vstack(
                    (
                        var_1[0] ** var_2.val[0]
                        * np.log(var_1[0])
                        * var_2.jac[0].toarray(),
                        var_1[1] ** var_2.val[1]
                        * np.log(var_1[1])
                        * var_2.jac[1].toarray(),
                        var_1[2] ** var_2.val[2]
                        * np.log(var_1[2])
                        * var_2.jac[2].toarray(),
                    )
                )
            )
    return _create_adarray(val, jac)


def _expected_sparse_ad(
    var_1: sps.spmatrix | sps.sparray,
    var_2: pp.ad.AdArrayBase,
    op: Literal["+", "-", "*", "/", "**", "@"],
) -> Literal[False] | pp.ad.SparseAdArray:
    """Expected value of var_1 op var_2 for a sparse matrix or array var_1. Only the
    matrix-vector product is supported. The Jacobian of a DiagonalAdArray holds the
    diagonal of the Jacobian matrix, which is formed explicitly for the product."""
    if op != "@":
        return False

    if isinstance(var_1, sps.spmatrix):
        val = var_1 * var_2.val
        if isinstance(var_2, pp.ad.DiagonalAdArray):
            jac = var_1 @ sps.diags(var_2.jac.ravel())
        else:
            jac = var_1 * var_2.jac
    else:
        val = var_1 @ var_2.val
        if isinstance(var_2, pp.ad.DiagonalAdArray):
            jac = var_1 @ sps.diags(var_2.jac.ravel())
        else:
            jac = var_1 @ var_2.jac
    return pp.ad.SparseAdArray(val, jac)


def _expected_wrapped_value(
    var_1: pp.ad.AdArrayBase,
    var_2: pp.ad.AdArrayBase,
    op: Literal["+", "-", "*", "/", "**", "@"],
) -> Literal[False] | pp.ad.SparseAdArray:
    """Expected value of var_1 op var_2 for two Ad arrays in wrapped form.

    In wrapped form, the two Ad operands are formed from two different variables in the
    same EquationSystem, see _get_ad_array and _get_diag_array. The Jacobian of an
    evaluated operand has one column for each degree of freedom in the EquationSystem,
    and thus consists of two blocks of columns, one for each variable. Each operand
    depends only on its own variable, so with J_1 and J_2 the Jacobians of the operands
    with respect to their own variables, the evaluated operands have the Jacobians
    [J_1, 0] and [0, J_2].

    The unwrapped operands var_1 and var_2 have only the Jacobians J_1 and J_2. They
    are therefore extended to [J_1, 0] and [0, J_2] before the expected value is
    computed by _expected_ad_ad.

    """
    if op == "@":
        # The matrix product is not defined between Ad arrays.
        return False
    var_1 = _embed_jacobian_in_block(var_1, block=0, num_blocks=2)
    var_2 = _embed_jacobian_in_block(var_2, block=1, num_blocks=2)
    return _expected_ad_ad(var_1, var_2, op)


def _embed_jacobian_in_block(
    ad_array: pp.ad.AdArrayBase, block: int, num_blocks: int
) -> pp.ad.SparseAdArray:
    """Extend the Jacobian of an Ad array with zero blocks of columns.

    Parameters:
        ad_array: Ad array with Jacobian J, of shape (n, m).
        block: Index of the block of columns that is to hold J.
        num_blocks: Number of blocks of columns in the extended Jacobian.

    Returns:
        An Ad array with the same value as ad_array, and a Jacobian of shape
        (n, m * num_blocks) that is J in the given block and zero elsewhere. For
        instance, block 1 of 2 gives the Jacobian [0, J].

    """
    jac = _jac_to_dense(ad_array)
    num_cols = jac.shape[1]
    extended = np.zeros((jac.shape[0], num_cols * num_blocks))
    extended[:, block * num_cols : (block + 1) * num_cols] = jac
    return pp.ad.SparseAdArray(ad_array.val, sps.csr_matrix(extended))
