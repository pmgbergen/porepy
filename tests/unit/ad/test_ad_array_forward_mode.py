"""Test suite for operations with AdArrays.


NOTE: For combinations of numpy arrays and AdArrays, we do not test the reverse order
operations for addition, subtraction, multiplication, and division, since this will not
work properly with numpy arrays, see https://stackoverflow.com/a/58120561 for more
information. To circumwent this problem, parsing of Ad expressions ensures that numpy
arrays are always left added (and subtracted, multiplied) with AdArrays, but this
should be covered in tests to be written.

"""
from __future__ import annotations
import pytest
import scipy.sparse as sps
import numpy as np

from porepy.numerics.ad.forward_mode import AdArray, initAdArrays
from porepy.numerics.ad import functions as af

from typing import Literal, Union

import porepy as pp

AdType = Union[float, np.ndarray, sps.spmatrix, pp.ad.AdArray]


def _get_scalar(wrapped: bool) -> float | pp.ad.Scalar:
    """Helper to set a scalar. Expected values in the test are hardcoded with respect to
    this value. The scalar is either returned as-is, or wrapped as an Ad scalar."""
    scalar = 2.0
    if wrapped:
        return pp.ad.Scalar(scalar)
    else:
        return scalar


def _get_dense_array(wrapped: bool) -> np.ndarray | pp.ad.DenseArray:
    """Helper to set a dense array (numpy array). Expected values in the test are
    hardcoded with respect to this value. The array is either returned as-is, or wrapped
    as an Ad DenseArray."""
    array = np.array([1, 2, 3]).astype(float)
    if wrapped:
        return pp.ad.DenseArray(array)
    else:
        return array


def _get_sparse_array(wrapped: bool) -> sps.spmatrix | pp.ad.SparseArray:
    """Helper to set a sparse array (scipy sparse array). Expected values in the test
    are hardcoded with respect to this value. The array is either returned as-is, or
    wrapped as an Ad SparseArray."""
    mat = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])).astype(float)
    if wrapped:
        return pp.ad.SparseArray(mat)
    else:
        return mat


def _get_ad_array(
    wrapped: bool,
) -> pp.ad.AdArray | tuple[pp.ad.AdArray, pp.ad.EquationSystem]:
    """Get an AdArray object which can be used in the tests."""

    # The construction between the wrapped and unwrapped case differs significantly: For
    # the latter we can simply create an AdArray with any value and Jacobian matrix.
    # The former must be processed through the operator parsing framework, and thus puts
    # stronger conditions on permissible states. The below code defines a variable
    # (variable_val), a matrix (jac), and constructs an expression as jac @ variable.
    # This expression is represented in the returned AdArray, either directly or (if
    # wrapped=True) on abstract form.
    #
    #  If this is confusing, it may be helpful to recall that an AdArray can represent
    #  any state, not only primary variables (e.g., a pp.ad.Variable). The main
    #  motivation for using a more complex value is that the Jacobian matrix of primary
    #  variables are identity matrices, thus compound expressions give higher chances of
    #  uncovering errors.

    # This is the value of the variable
    variable_val = np.ones(3)
    # This is the Jacobian matrix of the returned expression.
    jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    # This is the expression to be used in the tests. The numerical values of val will
    # be np.array([6, 15, 24]), and its Jacobian matrix is jac.
    expression_val = jac @ variable_val

    if wrapped:
        g = pp.CartGrid([3, 1])
        mdg = pp.MixedDimensionalGrid()
        mdg.add_subdomains([g])

        eq_system = pp.ad.EquationSystem(mdg)
        eq_system.create_variables("foo", subdomains=[g])
        var = eq_system.variables[0]
        d = mdg.subdomain_data(g)
        d["stored_solutions"]["foo"][0] = variable_val
        d["stored_iterates"]["foo"][0] = variable_val
        mat = pp.ad.SparseArray(jac)

        return mat @ var, eq_system

    else:
        ad_arr = pp.ad.AdArray(expression_val, jac)
        return ad_arr


def _expected_value(
    var_1: AdType, var_2: AdType, op: Literal["+", "-", "*", "/", "**", "@"]
) -> bool | float | np.ndarray | sps.spmatrix | pp.ad.AdArray:
    """For a combination of two Ad objects and an operation return either the expected
    value, or False if the operation is not supported.

    The function considers all combinations of types for var_1 and var_2 (as a long list
    of if-else statements that checks isinstance), and returns the expected value of the
    given operation. The calculation of the expected value is done in one of two ways:
        i)  None of the variables are AdArrays. In this case, the operation is evaluated
            using eval (in practice, this means that the evaluation is left to the
            Python, numpy and/or scipy).
        ii) One or both of the variables are AdArrays. In this case, the expected values
            are either hard-coded (this is typically the case where it is easy to do the
            calculation by hand, e.g., for addition), or computed using rules for
            derivation (product rule etc.) by hand, but using matrix-vector products and
            similar to compute the actual values.

    """
    # General comment regarding implementation for cases that do not include the
    # AdArray: We always (except in a few cases which are documented explicitly) use
    # eval to evaluate the expression. To catch cases that are not supported by numpy
    # or/else scipy, the evalutaion is surrounded by a try-except block. The except
    # typically checks that the operation is one that was expected to fail; example:
    # scalar @ scalar is not supported, but scalar + scalar is, so if the latter fails,
    # something is wrong. For a few combinations of operators, the combination will fail
    # in almost all cases, and the assertion is for simplicity put inside the try
    # instead of the except block.

    ### First do all combinations that do not involve AdArrays
    if isinstance(var_1, float) and isinstance(var_2, float):
        try:
            return eval(f"var_1 {op} var_2")
        except TypeError:
            assert op in ["@"]
            return False
    elif isinstance(var_1, float) and isinstance(var_2, np.ndarray):
        try:
            return eval(f"var_1 {op} var_2")
        except ValueError:
            assert op in ["@"]
            return False
    elif isinstance(var_1, float) and isinstance(var_2, sps.spmatrix):
        try:
            # This should fail for all operations expect from multiplication.
            val = eval(f"var_1 {op} var_2")
            assert op == "*"
            return val
        except (ValueError, NotImplementedError, TypeError):
            return False
    elif isinstance(var_1, np.ndarray) and isinstance(var_2, float):
        try:
            return eval(f"var_1 {op} var_2")
        except ValueError:
            assert op in ["@"]
            return False
    elif isinstance(var_1, np.ndarray) and isinstance(var_2, np.ndarray):
        return eval(f"var_1 {op} var_2")
    elif isinstance(var_1, np.ndarray) and isinstance(var_2, sps.spmatrix):
        try:
            return eval(f"var_1 {op} var_2")
        except TypeError:
            assert op in ["/", "**"]
            return False
    elif isinstance(var_1, sps.spmatrix) and isinstance(var_2, float):
        if op == "**":
            # SciPy has implemented a limited version matrix powers to scalars, but not
            # with a satisfactory flexibility. If we try to evaluate the expression, it
            # may or may not work (see comments in the __pow__ method is Operators), but
            # the operation is anyhow explicitly disallowed. Thus, we return False.
            return False

        try:
            # This should fail for all operations expect from multiplication.
            val = eval(f"var_1 {op} var_2")
            assert op in ["*", "/"]
            return val
        except (ValueError, NotImplementedError):
            return False
    elif isinstance(var_1, sps.spmatrix) and isinstance(var_2, np.ndarray):
        if op == "**":
            # SciPy has implemented a limited version matrix powers to numpy arrays, but
            # not with a satisfactory flexibility. If we try to evaluate the expression,
            # it may or may not work (see comments in the __pow__ method is Operators),
            # but the operation is anyhow explicitly disallowed. Thus, we return False.
            return False
        try:
            return eval(f"var_1 {op} var_2")
        except TypeError:
            assert op in ["**"]
            return False

    elif isinstance(var_1, sps.spmatrix) and isinstance(var_2, sps.spmatrix):
        try:
            return eval(f"var_1 {op} var_2")
        except (ValueError, TypeError):
            assert op in ["**"]
            return False

    ### From here on, we have at least one AdArray
    elif isinstance(var_1, pp.ad.AdArray) and isinstance(var_2, float):
        if op == "+":
            # Array + 2.0
            val = np.array([8, 17, 26])
            jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
            return pp.ad.AdArray(val, jac)
        elif op == "-":
            # Array - 2.0
            val = np.array([4, 13, 22])
            jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
            return pp.ad.AdArray(val, jac)
        elif op == "*":
            # Array * 2.0
            val = np.array([12, 30, 48])
            jac = sps.csr_matrix(np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]]))
            return pp.ad.AdArray(val, jac)
        elif op == "/":
            # Array / 2.0
            val = np.array([6 / 2, 15 / 2, 24 / 2])
            jac = sps.csr_matrix(
                np.array(
                    [
                        [1 / 2, 2 / 2, 3 / 2],
                        [4 / 2, 5 / 2, 6 / 2],
                        [7 / 2, 8 / 2, 9 / 2],
                    ]
                )
            )
            return pp.ad.AdArray(val, jac)
        elif op == "**":
            # Array ** 2.0
            val = np.array([6**2, 15**2, 24**2])
            jac = sps.csr_matrix(
                2
                * np.vstack(
                    (
                        var_1.val[0] * var_1.jac[0].A,
                        var_1.val[1] * var_1.jac[1].A,
                        var_1.val[2] * var_1.jac[2].A,
                    )
                ),
            )
            return pp.ad.AdArray(val, jac)
        elif op == "@":
            # Array @ 2.0, which in pratice is Array * 2.0
            val = np.array([12, 30, 48])
            jac = sps.csr_matrix(np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]]))
            return pp.ad.AdArray(val, jac)

    elif isinstance(var_1, float) and isinstance(var_2, pp.ad.AdArray):
        if op == "+":
            # 2.0 + Array
            val = np.array([8, 17, 26])
            jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
            return pp.ad.AdArray(val, jac)
        elif op == "-":
            # 2.0 - Array
            val = np.array([-4, -13, -22])
            jac = sps.csr_matrix(np.array([[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]]))
            return pp.ad.AdArray(val, jac)
        elif op == "*":
            # 2.0 * Array
            val = np.array([12, 30, 48])
            jac = sps.csr_matrix(np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]]))
            return pp.ad.AdArray(val, jac)
        elif op == "/":
            # This is 2 / Array
            # The derivative is -2 / Array**2 * dArray
            val = np.array([2 / 6, 2 / 15, 2 / 24])
            jac = sps.csr_matrix(
                np.vstack(
                    (
                        -2 / var_2.val[0] ** 2 * var_2.jac[0].A,
                        -2 / var_2.val[1] ** 2 * var_2.jac[1].A,
                        -2 / var_2.val[2] ** 2 * var_2.jac[2].A,
                    )
                ),
            )
            return pp.ad.AdArray(val, jac)
        elif op == "**":
            # 2.0 ** Array
            # The derivative is 2**Array * log(2) * dArray
            val = np.array([2**6, 2**15, 2**24])
            jac = sps.csr_matrix(
                np.vstack(
                    (
                        np.log(2.0) * (2 ** var_2.val[0]) * var_2.jac[0].A,
                        np.log(2.0) * (2 ** var_2.val[1]) * var_2.jac[1].A,
                        np.log(2.0) * (2 ** var_2.val[2]) * var_2.jac[2].A,
                    )
                ),
            )
            return pp.ad.AdArray(val, jac)
        elif op == "@":
            # 2.0 @ Array, which in pratice is 2.0 * Array
            val = np.array([12, 30, 48])
            jac = sps.csr_matrix(np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]]))
            return pp.ad.AdArray(val, jac)

    elif isinstance(var_1, pp.ad.AdArray) and isinstance(var_2, np.ndarray):
        # Recall that the numpy array has values np.array([1, 2, 3])
        if op == "+":
            # Array + np.array([1, 2, 3])
            val = np.array([7, 17, 27])
            jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
            return pp.ad.AdArray(val, jac)
        elif op == "-":
            # Array - np.array([1, 2, 3])
            val = np.array([5, 13, 21])
            jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
            return pp.ad.AdArray(val, jac)
        elif op == "*":
            # Array * np.array([1, 2, 3])
            val = np.array([6, 30, 72])
            jac = sps.csr_matrix(np.array([[1, 2, 3], [8, 10, 12], [21, 24, 27]]))
            return pp.ad.AdArray(val, jac)
        elif op == "/":
            # Array / np.array([1, 2, 3])
            val = np.array([6 / 1, 15 / 2, 24 / 3])
            jac = sps.csr_matrix(
                np.vstack(
                    (
                        var_1.jac[0].A / var_2[0],
                        var_1.jac[1].A / var_2[1],
                        var_1.jac[2].A / var_2[2],
                    )
                )
            )
            return pp.ad.AdArray(val, jac)
        elif op == "**":
            # Array ** np.array([1, 2, 3])
            # The derivative is
            #    Array**(np.array([1, 2, 3]) - 1) * np.array([1, 2, 3]) * dArray
            val = np.array([6, 15**2, 24**3])
            jac = sps.csr_matrix(
                np.vstack(
                    (
                        var_2[0] * (var_1.val[0] ** (var_2[0] - 1.0)) * var_1.jac[0].A,
                        var_2[1] * (var_1.val[1] ** (var_2[1] - 1.0)) * var_1.jac[1].A,
                        var_2[2] * (var_1.val[2] ** (var_2[2] - 1.0)) * var_1.jac[2].A,
                    )
                )
            )
            return pp.ad.AdArray(val, jac)
        elif op == "@":
            # The operation is not allowed
            return False
    elif isinstance(var_1, np.ndarray) and isinstance(var_2, pp.ad.AdArray):
        # Recall that the numpy array has values np.array([1, 2, 3])
        if op == "+":
            # Array + np.array([1, 2, 3])
            val = np.array([7, 17, 27])
            jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
            return pp.ad.AdArray(val, jac)
        elif op == "-":
            # np.array([1, 2, 3]) - Array
            val = np.array([-5, -13, -21])
            jac = sps.csr_matrix(np.array([[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]]))
            return pp.ad.AdArray(val, jac)
        elif op == "*":
            # Array * np.array([1, 2, 3])
            val = np.array([6, 30, 72])
            jac = sps.csr_matrix(np.array([[1, 2, 3], [8, 10, 12], [21, 24, 27]]))
            return pp.ad.AdArray(val, jac)
        elif op == "/":
            # np.array([1, 2, 3]) / Array
            val = np.array([1 / 6, 2 / 15, 3 / 24])
            jac = sps.csr_matrix(
                np.vstack(
                    (
                        -var_1[0] * var_2.jac[0].A / var_2.val[0] ** 2,
                        -var_1[1] * var_2.jac[1].A / var_2.val[1] ** 2,
                        -var_1[2] * var_2.jac[2].A / var_2.val[2] ** 2,
                    )
                )
            )
            return pp.ad.AdArray(val, jac)
        elif op == "**":
            # np.array([1, 2, 3]) ** Array
            val = np.array([1, 2**15, 3**24])
            jac = sps.csr_matrix(
                np.vstack(
                    (
                        var_1[0] ** var_2.val[0] * np.log(var_1[0]) * var_2.jac[0].A,
                        var_1[1] ** var_2.val[1] * np.log(var_1[1]) * var_2.jac[1].A,
                        var_1[2] ** var_2.val[2] * np.log(var_1[2]) * var_2.jac[2].A,
                    )
                )
            )
            return pp.ad.AdArray(val, jac)

    elif isinstance(var_1, pp.ad.AdArray) and isinstance(var_2, sps.spmatrix):
        return False
    elif isinstance(var_1, sps.spmatrix) and isinstance(var_2, pp.ad.AdArray):
        # This combination is only allowed for matrix-vector products (op = "@")
        if op == "@":
            val = var_1 * var_2.val
            jac = var_1 * var_2.jac
            return pp.ad.AdArray(val, jac)
        else:
            return False

    elif isinstance(var_1, pp.ad.AdArray) and isinstance(var_2, pp.ad.AdArray):
        # For this case, var_2 was modified manually to be twice var_1, see comments in
        # the main test function. Mirror this here to be consistent.
        var_2 = var_1 + var_1
        if op == "+":
            # This evaluates to 3 * Array (since var_2 = 2 * var_1)
            val = np.array([18, 45, 72])
            jac = sps.csr_matrix(np.array([[3, 6, 9], [12, 15, 18], [21, 24, 27]]))
            return pp.ad.AdArray(val, jac)
        elif op == "-":
            # This evaluates to -Array (since var_2 = 2 * var_1)
            val = np.array([-6, -15, -24])
            jac = sps.csr_matrix(np.array([[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]]))
            return pp.ad.AdArray(val, jac)
        elif op == "*":
            # This evaluates to 2 * Array**2 (since var_2 = 2 * var_1)
            val = np.array([6 * 12, 15 * 30, 24 * 48])
            jac = sps.csr_matrix(
                np.vstack(
                    (
                        var_1.jac[0].A * var_2.val[0] + var_1.val[0] * var_2.jac[0].A,
                        var_1.jac[1].A * var_2.val[1] + var_1.val[1] * var_2.jac[1].A,
                        var_1.jac[2].A * var_2.val[2] + var_1.val[2] * var_2.jac[2].A,
                    )
                )
            )
            return pp.ad.AdArray(val, jac)
        elif op == "/":
            # This evaluates to Array / (2 * Array)
            # The derivative is computed from the product and chain rules
            val = np.array([1 / 2, 1 / 2, 1 / 2])
            jac = sps.csr_matrix(
                np.vstack(  # NBNB
                    (
                        var_1.jac[0].A / var_2.val[0]
                        - var_1.val[0] * var_2.jac[0].A / var_2.val[0] ** 2,
                        var_1.jac[1].A / var_2.val[1]
                        - var_1.val[1] * var_2.jac[1].A / var_2.val[1] ** 2,
                        var_1.jac[2].A / var_2.val[2]
                        - var_1.val[2] * var_2.jac[2].A / var_2.val[2] ** 2,
                    )
                )
            )
            return pp.ad.AdArray(val, jac)
        elif op == "**":
            # This is Array ** (2 * Array)
            # The derivative is
            #    Array**(2 * Array - 1) * (2 * Array) * dArray
            #  + Array**(2 * Array) * log(Array) * dArray
            val = np.array([6**12, 15**30, 24**48])
            jac = sps.csr_matrix(
                np.vstack(  #
                    (
                        var_2.val[0]
                        * var_1.val[0] ** (var_2.val[0] - 1.0)
                        * var_1.jac[0].A
                        + np.log(var_1.val[0])
                        * (var_1.val[0] ** var_2.val[0])
                        * var_2.jac[0].A,
                        var_2.val[1]
                        * var_1.val[1] ** (var_2.val[1] - 1.0)
                        * var_1.jac[1].A
                        + np.log(var_1.val[1])
                        * (var_1.val[1] ** var_2.val[1])
                        * var_2.jac[1].A,
                        var_2.val[2]
                        * var_1.val[2] ** (var_2.val[2] - 1.0)
                        * var_1.jac[2].A
                        + np.log(var_1.val[2])
                        * (var_1.val[2] ** var_2.val[2])
                        * var_2.jac[2].A,
                    )
                )
            )
            return pp.ad.AdArray(val, jac)
        elif op == "@":
            return False


@pytest.mark.parametrize("var_1", ["scalar", "dense", "sparse", "ad"])
@pytest.mark.parametrize("var_2", ["scalar", "dense", "sparse", "ad"])
@pytest.mark.parametrize("op", ["+", "-", "*", "/", "**", "@"])
@pytest.mark.parametrize("wrapped", [True, False])
def test_arithmetic_operations_on_ad_objects(
    var_1: str, var_2: str, op: str, wrapped: bool
) -> None:
    """Test that the fundamental Ad operators can be combined using the standard
    arithmetic operations.

    All combinations of operators and operations are formed, in two different modes:
    Wrapped as Ad operators (subclasses of pp.ad.Operator) or primitive values (float,
    numpy.ndarray, scipy.spmatrix, AdArray). In the wrapped form, all of these
    combinations are actually tested, while in the primitive form (which is what is
    applied when doing forward-mode algorithmic differentiation), only combinations that
    involve at least one AdArray are meaningfully tested, see below if for an
    explanation (there is an exception to this, involving numpy arrays and AdArrays, see
    the second if just below for an explanation).

    """

    if not wrapped and var_1 != "ad" and var_2 != "ad":
        # If not wrapped in the abstract layer, these cases should be covered by the
        # tests for the external packages; PorePy just has to rely on e.g., numpy being
        # correctly implemented. For the wrapped case, we need to test that the parsing
        # is okay, thus we do not skip if wrapped is True.
        return
    if not wrapped and var_1 == "dense" and var_2 == "ad":
        # This is the case where the first operand is a numpy array. This is a
        # problematic setting, since numpy's operators (__add__ etc.) will be invoked.
        # Despite numpy not knowing anything about AdArrays, numpy somehow uses
        # broadcasting to compute and return a value, but the result is not in any sense
        # what is to be expected. In forward mode there is nothing we can do about this
        # (see GH issue #819, tagged as won't fix); the user just has to know that this
        # should not be done. If the arrays are wrapped, we can circumvent the problem
        # in parsing by rewriting the expression so that the AdArray's right  operators
        # (e.g., __radd__) are invoked instead of numpy's left operators. Thus, if
        # wrapped is True, we do not skip the test.
        return

    def _var_from_string(v, do_wrap: bool):
        if v == "scalar":
            return _get_scalar(do_wrap)
        elif v == "dense":
            return _get_dense_array(do_wrap)
        elif v == "sparse":
            return _get_sparse_array(do_wrap)
        elif v == "ad":
            return _get_ad_array(do_wrap)
        else:
            raise ValueError("Unknown variable type")

    # Get the actual variables from the input strings.
    v1 = _var_from_string(var_1, wrapped)
    v2 = _var_from_string(var_2, wrapped)

    # Some gymnastics is needed here: In the wrapped form, expressions need an
    # EquationSystem for evaluation and, if one of the operands is an AdArray, this
    # should be the EquationSystem used to generate this operand (see method
    # _get_ad_array). If this is not the case, the EquationSystem will end up having to
    # evaluate an Ad variable that it does not know about. Therefore, the method
    # _get_ad_array returns the generated EquationSystem together with variable. If none
    # of the operands is an Ad array, we will still formally need an EquationSystem to
    # evaluate the expression, but since this will not actually be used for anything, we
    # can generate a new one and pass it as a formality.
    if wrapped:
        if var_1 == "ad":
            v1, eq_system = v1
        elif var_2 == "ad":
            # The case of both v1 and v2 being Ad variables is dealt with below.
            v2, eq_system = v2
        else:
            mdg = pp.MixedDimensionalGrid()
            eq_system = pp.ad.EquationSystem(mdg)
    if var_1 == "ad" and var_2 == "ad":
        # For the case of two ad variables, they should be associated with the
        # same EquationSystem, or else parsing will fail. We could have set v1 =
        # v2, but this is less likely to catch errors in the parsing. Instead,
        # we reassign v2 = v1 + v1. This also requires some adaptations in the
        # code to get the expected values, see that function.
        v2 = v1 + v1

    # Calculate the expected numerical values for this expression. This inolves
    # hard-coded values for the different operators and their combinations, see the
    # function for more information. If the operation is not expected to succeeed, the
    # function will return False.
    expected = _expected_value(
        _var_from_string(var_1, False), _var_from_string(var_2, False), op
    )

    def _compare(v1, v2):
        # Helper function to compare two evaluated objects.
        assert type(v1) == type(v2)
        if isinstance(v1, float):
            assert np.isclose(v1, v2)
        elif isinstance(v1, np.ndarray):
            assert np.allclose(v1, v2)
        elif isinstance(v1, sps.spmatrix):
            assert np.allclose(v1.toarray(), v2.toarray())
        elif isinstance(v1, pp.ad.AdArray):
            assert np.allclose(v1.val, v2.val)
            assert np.allclose(v1.jac.toarray(), v2.jac.toarray())

    # Evaluate the funtion. This is a bit different for the wrapped and forward mode,
    # but the logic is the same: Try to evaluate. If this breaks, check that this was
    # not a surprize (variable expected is False).
    if wrapped:
        try:
            expression = eval(f"v1 {op} v2")
            val = expression.evaluate(eq_system)
        except (TypeError, ValueError, NotImplementedError):
            assert not expected
            return
    else:
        try:
            val = eval(f"v1 {op} v2")
        except (TypeError, ValueError, NotImplementedError):
            assert not expected
            return

    # Compare numerical values between evaluated and expected outcomes.
    _compare(val, expected)


""" Below follows legacy (though updated) tests for the AdArray class. These tests
    cover initiation of AdArray (joint initiation of multiple dependent
variables). The test also partly cover the arithmetic operations implemented for
AdArrays, e.g., __add__, __sub__, etc., but these are also tested in different tests.
"""


def test_add_two_ad_variables_init():
    a, b = initAdArrays([np.array([1]), np.array([-10])])
    c = a + b
    assert c.val == -9 and np.all(c.jac.A == [1, 1])
    assert a.val == 1 and np.all(a.jac.A == [1, 0])
    assert b.val == -10 and np.all(b.jac.A == [0, 1])


def test_sub_var_init_with_var_init():
    a, b = initAdArrays([np.array([3]), np.array([2])])
    c = b - a
    assert np.allclose(c.val, -1) and np.all(c.jac.A == [-1, 1])
    assert a.val == 3 and np.all(a.jac.A == [1, 0])
    assert b.val == 2 and np.all(b.jac.A == [0, 1])


def test_mul_ad_var_init():
    a, b = initAdArrays([np.array([3]), np.array([2])])
    c = a * b
    assert a.val == 3 and np.all(a.jac.A == [1, 0])
    assert b.val == 2 and np.all(b.jac.A == [0, 1])
    assert c.val == 6 and np.all(c.jac.A == [2, 3])


def test_mul_scal_ad_var_init():
    a, b = initAdArrays([np.array([3]), np.array([2])])
    d = 3.0
    c = d * a
    assert c.val == 9 and np.all(c.jac.A == [3, 0])
    assert a.val == 3 and np.all(a.jac.A == [1, 0])
    assert b.val == 2 and np.all(b.jac.A == [0, 1])


def test_mul_sps_advar_init():
    x = initAdArrays([np.array([1, 2, 3])])[0]
    A = sps.csc_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    f = A @ x
    assert np.all(f.val == [14, 32, 50])
    assert np.all((f.jac == A).A)


def test_advar_init_diff_len():
    a, b = initAdArrays([np.array([1, 2, 3]), np.array([1, 2])])
    A = sps.csc_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    B = sps.csc_matrix(np.array([[1, 2], [4, 5]]))

    f = A @ a
    g = B @ b
    zero_32 = sps.csc_matrix((3, 2))
    zero_23 = sps.csc_matrix((2, 3))

    jac_f = sps.hstack((A, zero_32))
    jac_g = sps.hstack((zero_23, B))
    assert np.all(f.val == [14, 32, 50])
    assert np.all((f.jac == jac_f).A)
    assert np.all(g.val == [5, 14])
    assert np.all((g.jac == jac_g).A)


def test_advar_init_cross_jacobi():
    x, y = initAdArrays([np.array([-1, 4]), np.array([1, 5])])

    z = x * y
    J = np.array([[1, 0, -1, 0], [0, 5, 0, 4]])
    assert np.all(z.val == [-1, 20])
    assert np.all((z.jac == J).A)


def test_exp_scalar_times_ad_var():
    val = np.array([1, 2, 3])
    J = sps.diags(np.array([1, 1, 1]))
    a, _, _ = initAdArrays([val, val, val])
    c = 2.0
    b = af.exp(c * a)

    zero = sps.csc_matrix((3, 3))
    jac = sps.hstack([c * sps.diags(np.exp(c * val)) * J, zero, zero])
    jac_a = sps.hstack([J, zero, zero])
    assert np.allclose(b.val, np.exp(c * val)) and np.allclose(b.jac.A, jac.A)
    assert np.all(a.val == [1, 2, 3]) and np.all(a.jac.A == jac_a.A)


def test_advar_mul_vec():
    x = AdArray(np.array([1, 2, 3]), sps.diags([3, 2, 1]))
    A = np.array([1, 3, 10])
    f = x * A
    sol = np.array([1, 6, 30])
    jac = np.diag([3, 6, 10])

    assert np.all(f.val == sol) and np.all(f.jac == jac)
    assert np.all(x.val == np.array([1, 2, 3])) and np.all(x.jac == np.diag([3, 2, 1]))


def test_advar_m_mul_vec_n():
    x = AdArray(np.array([1, 2, 3]), sps.diags([3, 2, 1]))
    vec = np.array([1, 2])
    R = sps.csc_matrix(np.array([[1, 0, 1], [0, 1, 0]]))
    y = R @ x
    z = y * vec
    Jy = np.array([[3, 0, 1], [0, 2, 0]])
    Jz = np.array([[1, 0, 3], [0, 4, 0]])
    assert np.all(y.val == [4, 2])
    assert np.sum(y.jac.A - Jy) == 0
    assert np.all(z.val == [4, 4])
    assert np.sum(z.jac.A - Jz) == 0


def test_mul_sps_advar():
    J = sps.csc_matrix(np.array([[1, 3, 1], [5, 0, 0], [5, 1, 2]]))
    x = AdArray(np.array([1, 2, 3]), J)
    A = sps.csc_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    f = A @ x

    assert np.all(f.val == [14, 32, 50])
    assert np.all(f.jac == A * J.A)


def test_mul_advar_vectors():
    Ja = sps.csc_matrix(np.array([[1, 3, 1], [5, 0, 0], [5, 1, 2]]))
    Jb = sps.csc_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    a = AdArray(np.array([1, 2, 3]), Ja)
    b = AdArray(np.array([1, 1, 1]), Jb)
    A = sps.csc_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    f = A @ a + b

    assert np.all(f.val == [15, 33, 51])
    assert np.sum(f.jac.A != A * Ja + Jb) == 0
    assert (
        np.sum(Ja != sps.csc_matrix(np.array([[1, 3, 1], [5, 0, 0], [5, 1, 2]]))) == 0
    )
    assert (
        np.sum(Jb != sps.csc_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))) == 0
    )


def test_copy_scalar():
    a = AdArray(np.array([1]), sps.csr_matrix([[0]]))
    b = a.copy()
    assert a.val == b.val
    assert a.jac == b.jac
    a.val = 2
    a.jac = 3
    assert b.val == 1
    assert b.jac == 0


def test_copy_vector():
    a = AdArray(np.ones(3), sps.csr_matrix(np.diag(np.ones((3)))))
    b = a.copy()
    assert np.allclose(a.val, b.val)
    assert np.allclose(a.jac.A, b.jac.A)
    a.val[0] = 3
    a.jac[2] = 4
    assert np.allclose(b.val, np.ones(3))
    assert np.allclose(b.jac.A, sps.csr_matrix(np.diag(np.ones((3)))).A)
