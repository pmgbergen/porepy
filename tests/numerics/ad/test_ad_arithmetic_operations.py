"""Tests of arithmetic operations between Ad objects.

Scalars, dense arrays, sparse matrices and Ad arrays are combined pairwise with the
operators +, -, *, /, ** and @. Each combination is evaluated both in forward mode, on
the numerical values directly, and wrapped as Ad operators that are parsed by an
EquationSystem. The result is compared with a value computed independently, or, if the
combination is not supported, it is checked that the evaluation fails.

"""

from typing import Literal, Union

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.ad.equation_system import GridEntity

AdType = Union[float, np.ndarray, sps.spmatrix, pp.ad.SparseAdArray]


@pytest.fixture
def arithmetic_mdg() -> pp.MixedDimensionalGrid:
    """Mixed-dimensional grid with a single subdomain of three cells, on which the
    operands of the tests are defined."""
    g = pp.CartGrid([3, 1])
    mdg = pp.MixedDimensionalGrid()
    mdg.add_subdomains([g])
    return mdg


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
    space = pp.ad.OperatorSpace.from_domains(
        mdg.subdomains(), dof_info={GridEntity.cells: 1}
    )
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

    space = pp.ad.OperatorSpace.from_domains(
        mdg.subdomains(), dof_info={GridEntity.cells: 1}
    )
    if wrapped:
        return pp.ad.SparseArray(mat, source=space, target=space)
    return mat


def _get_ad_array(
    wrapped: bool, mdg
) -> pp.ad.SparseAdArray | tuple[pp.ad.SparseAdArray, pp.ad.EquationSystem]:
    """Get a SparseAdArray object which can be used in the tests."""

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
        equation_system = pp.ad.EquationSystem(mdg)
        equation_system.create_variables(
            "foo", subdomains=mdg.subdomains(), dof_info={GridEntity.cells: 1}
        )
        var = equation_system.variables[0]
        d = mdg.subdomain_data(mdg.subdomains()[0])

        pp.set_solution_values(
            name="foo", values=variable_val, data=d, time_step_index=0
        )
        pp.set_solution_values(name="foo", values=variable_val, data=d, iterate_index=0)
        space = pp.ad.OperatorSpace.from_domains(
            mdg.subdomains(), dof_info={GridEntity.cells: 1}
        )
        mat = pp.ad.SparseArray(jac, source=space, target=space)

        return mat @ var, equation_system

    return pp.ad.SparseAdArray(expression_val, jac)


def _expected_value(
    var_1: AdType, var_2: AdType, op: Literal["+", "-", "*", "/", "**", "@"]
) -> bool | float | np.ndarray | sps.spmatrix | pp.ad.SparseAdArray:
    """For a combination of two Ad objects and an operation return either the expected
    value, or False if the operation is not supported.

    The function considers all combinations of types for var_1 and var_2 (as a long list
    of if-else statements that checks isinstance), and returns the expected value of the
    given operation. The calculation of the expected value is done in one of two ways:
        i)  None of the variables are SparseAdArrays. In this case, the operation is
            evaluated using eval (in practice, this means that the evaluation is left to
            the Python, numpy and/or scipy).
        ii) One or both of the variables are SparseAdArrays. In this case, the expected
            values are either hard-coded (this is typically the case where it is easy to
            do the calculation by hand, e.g., for addition), or computed using rules for
            derivation (product rule etc.) by hand, but using matrix-vector products and
            similar to compute the actual values.

    """
    # General comment regarding implementation for cases that do not include the
    # SparseAdArray: We always (except in a few cases which are documented explicitly)
    # use eval to evaluate the expression. To catch cases that are not supported by
    # numpy or/else scipy, the evalutation is surrounded by a try-except block. The
    # except typically checks that the operation is one that was expected to fail;
    # example: scalar @ scalar is not supported, but scalar + scalar is, so if the
    # latter fails, something is wrong. For a few combinations of operators, the
    # combination will fail in almost all cases, and the assertion is for simplicity put
    # inside the try instead of the except block.

    ### First do all combinations that do not involve SparseAdArrays
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
    elif isinstance(var_1, float) and isinstance(var_2, (sps.spmatrix, sps.sparray)):
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
    elif isinstance(var_1, np.ndarray) and isinstance(
        var_2, (sps.spmatrix, sps.sparray)
    ):
        try:
            return eval(f"var_1 {op} var_2")
        except TypeError:
            assert op in ["/", "**"]
            return False
    elif isinstance(var_1, (sps.spmatrix, sps.sparray)) and isinstance(var_2, float):
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
    elif isinstance(var_1, (sps.spmatrix, sps.sparray)) and isinstance(
        var_2, np.ndarray
    ):
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

    elif isinstance(var_1, (sps.spmatrix, sps.sparray)) and isinstance(
        var_2, (sps.spmatrix, sps.sparray)
    ):
        try:
            return eval(f"var_1 {op} var_2")
        except (ValueError, TypeError, NotImplementedError):
            assert op in ["**"]
            return False

    ### From here on, we have at least one SparseAdArray
    elif isinstance(var_1, pp.ad.SparseAdArray) and isinstance(var_2, float):
        if op == "+":
            # Array + 2.0
            val = np.array([8, 17, 26])
            jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
            return pp.ad.SparseAdArray(val, jac)
        elif op == "-":
            # Array - 2.0
            val = np.array([4, 13, 22])
            jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
            return pp.ad.SparseAdArray(val, jac)
        elif op == "*":
            # Array * 2.0
            val = np.array([12, 30, 48])
            jac = sps.csr_matrix(np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]]))
            return pp.ad.SparseAdArray(val, jac)
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
            return pp.ad.SparseAdArray(val, jac)
        elif op == "**":
            # Array ** 2.0
            val = np.array([6**2, 15**2, 24**2])
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
            return pp.ad.SparseAdArray(val, jac)
        elif op == "@":
            # We disallow this operation due to the following reason: The only case in
            # which it is convenient to use the @ operator for the scalars is a generic
            # operator that can accept everything: scalars, ndarrays and sparse
            # matrices. However, we do not know any cases where one operator can be both
            # a scalar or ndarray AND a matrix. This choice will be reconsidered if a
            # reasonable example is found.
            return False

    elif isinstance(var_1, float) and isinstance(var_2, pp.ad.SparseAdArray):
        if op == "+":
            # 2.0 + Array
            val = np.array([8, 17, 26])
            jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
            return pp.ad.SparseAdArray(val, jac)
        elif op == "-":
            # 2.0 - Array
            val = np.array([-4, -13, -22])
            jac = sps.csr_matrix(np.array([[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]]))
            return pp.ad.SparseAdArray(val, jac)
        elif op == "*":
            # 2.0 * Array
            val = np.array([12, 30, 48])
            jac = sps.csr_matrix(np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]]))
            return pp.ad.SparseAdArray(val, jac)
        elif op == "/":
            # This is 2 / Array
            # The derivative is -2 / Array**2 * dArray
            val = np.array([2 / 6, 2 / 15, 2 / 24])
            jac = sps.csr_matrix(
                np.vstack(
                    (
                        -2 / var_2.val[0] ** 2 * var_2.jac[0].toarray(),
                        -2 / var_2.val[1] ** 2 * var_2.jac[1].toarray(),
                        -2 / var_2.val[2] ** 2 * var_2.jac[2].toarray(),
                    )
                ),
            )
            return pp.ad.SparseAdArray(val, jac)
        elif op == "**":
            # 2.0 ** Array
            # The derivative is 2**Array * log(2) * dArray
            val = np.array([2**6, 2**15, 2**24])
            jac = sps.csr_matrix(
                np.vstack(
                    (
                        np.log(2.0) * (2 ** var_2.val[0]) * var_2.jac[0].toarray(),
                        np.log(2.0) * (2 ** var_2.val[1]) * var_2.jac[1].toarray(),
                        np.log(2.0) * (2 ** var_2.val[2]) * var_2.jac[2].toarray(),
                    )
                ),
            )
            return pp.ad.SparseAdArray(val, jac)
        elif op == "@":
            # Note: See the comment for the case SparseAdArray @ scalar.
            return False

    elif isinstance(var_1, pp.ad.SparseAdArray) and isinstance(var_2, np.ndarray):
        # Recall that the numpy array has values np.array([1, 2, 3])
        if op == "+":
            # Array + np.array([1, 2, 3])
            val = np.array([7, 17, 27])
            jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
            return pp.ad.SparseAdArray(val, jac)
        elif op == "-":
            # Array - np.array([1, 2, 3])
            val = np.array([5, 13, 21])
            jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
            return pp.ad.SparseAdArray(val, jac)
        elif op == "*":
            # Array * np.array([1, 2, 3])
            val = np.array([6, 30, 72])
            jac = sps.csr_matrix(np.array([[1, 2, 3], [8, 10, 12], [21, 24, 27]]))
            return pp.ad.SparseAdArray(val, jac)
        elif op == "/":
            # Array / np.array([1, 2, 3])
            val = np.array([6 / 1, 15 / 2, 24 / 3])
            jac = sps.csr_matrix(
                np.vstack(
                    (
                        var_1.jac[0].toarray() / var_2[0],
                        var_1.jac[1].toarray() / var_2[1],
                        var_1.jac[2].toarray() / var_2[2],
                    )
                )
            )
            return pp.ad.SparseAdArray(val, jac)
        elif op == "**":
            # Array ** np.array([1, 2, 3])
            # The derivative is
            #    Array**(np.array([1, 2, 3]) - 1) * np.array([1, 2, 3]) * dArray
            val = np.array([6, 15**2, 24**3])
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
            return pp.ad.SparseAdArray(val, jac)
        elif op == "@":
            # The operation is not allowed
            return False
    elif isinstance(var_1, np.ndarray) and isinstance(var_2, pp.ad.SparseAdArray):
        # Recall that the numpy array has values np.array([1, 2, 3])
        if op == "+":
            # Array + np.array([1, 2, 3])
            val = np.array([7, 17, 27])
            jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
            return pp.ad.SparseAdArray(val, jac)
        elif op == "-":
            # np.array([1, 2, 3]) - Array
            val = np.array([-5, -13, -21])
            jac = sps.csr_matrix(np.array([[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]]))
            return pp.ad.SparseAdArray(val, jac)
        elif op == "*":
            # Array * np.array([1, 2, 3])
            val = np.array([6, 30, 72])
            jac = sps.csr_matrix(np.array([[1, 2, 3], [8, 10, 12], [21, 24, 27]]))
            return pp.ad.SparseAdArray(val, jac)
        elif op == "/":
            # np.array([1, 2, 3]) / Array
            val = np.array([1 / 6, 2 / 15, 3 / 24])
            jac = sps.csr_matrix(
                np.vstack(
                    (
                        -var_1[0] * var_2.jac[0].toarray() / var_2.val[0] ** 2,
                        -var_1[1] * var_2.jac[1].toarray() / var_2.val[1] ** 2,
                        -var_1[2] * var_2.jac[2].toarray() / var_2.val[2] ** 2,
                    )
                )
            )
            return pp.ad.SparseAdArray(val, jac)
        elif op == "**":
            # np.array([1, 2, 3]) ** Array
            val = np.array([1, 2**15, 3**24])
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
            return pp.ad.SparseAdArray(val, jac)

    elif isinstance(var_1, pp.ad.SparseAdArray) and isinstance(
        var_2, (sps.spmatrix, sps.sparray)
    ):
        return False
    elif isinstance(var_1, sps.spmatrix) and isinstance(var_2, pp.ad.SparseAdArray):
        # This combination is only allowed for matrix-vector products (op = "@")
        if op == "@":
            val = var_1 * var_2.val
            jac = var_1 * var_2.jac
            return pp.ad.SparseAdArray(val, jac)
        else:
            return False
    elif isinstance(var_1, sps.sparray) and isinstance(var_2, pp.ad.SparseAdArray):
        # This combination is only allowed for matrix-vector products (op = "@")
        if op == "@":
            val = var_1 @ var_2.val
            jac = var_1 @ var_2.jac
            return pp.ad.SparseAdArray(val, jac)
        else:
            return False

    elif isinstance(var_1, pp.ad.SparseAdArray) and isinstance(
        var_2, pp.ad.SparseAdArray
    ):
        # For this case, var_2 was modified manually to be twice var_1, see comments in
        # the main test function. Mirror this here to be consistent.
        var_2 = var_1 + var_1
        if op == "+":
            # This evaluates to 3 * Array (since var_2 = 2 * var_1)
            val = np.array([18, 45, 72])
            jac = sps.csr_matrix(np.array([[3, 6, 9], [12, 15, 18], [21, 24, 27]]))
            return pp.ad.SparseAdArray(val, jac)
        elif op == "-":
            # This evaluates to -Array (since var_2 = 2 * var_1)
            val = np.array([-6, -15, -24])
            jac = sps.csr_matrix(np.array([[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]]))
            return pp.ad.SparseAdArray(val, jac)
        elif op == "*":
            # This evaluates to 2 * Array**2 (since var_2 = 2 * var_1)
            val = np.array([6 * 12, 15 * 30, 24 * 48])
            jac = sps.csr_matrix(
                np.vstack(
                    (
                        var_1.jac[0].toarray() * var_2.val[0]
                        + var_1.val[0] * var_2.jac[0].toarray(),
                        var_1.jac[1].toarray() * var_2.val[1]
                        + var_1.val[1] * var_2.jac[1].toarray(),
                        var_1.jac[2].toarray() * var_2.val[2]
                        + var_1.val[2] * var_2.jac[2].toarray(),
                    )
                )
            )
            return pp.ad.SparseAdArray(val, jac)
        elif op == "/":
            # This evaluates to Array / (2 * Array)
            # The derivative is computed from the product and chain rules
            val = np.array([1 / 2, 1 / 2, 1 / 2])
            jac = sps.csr_matrix(
                np.vstack(  # NBNB
                    (
                        var_1.jac[0].toarray() / var_2.val[0]
                        - var_1.val[0] * var_2.jac[0].toarray() / var_2.val[0] ** 2,
                        var_1.jac[1].toarray() / var_2.val[1]
                        - var_1.val[1] * var_2.jac[1].toarray() / var_2.val[1] ** 2,
                        var_1.jac[2].toarray() / var_2.val[2]
                        - var_1.val[2] * var_2.jac[2].toarray() / var_2.val[2] ** 2,
                    )
                )
            )
            return pp.ad.SparseAdArray(val, jac)
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
                        * var_1.jac[0].toarray()
                        + np.log(var_1.val[0])
                        * (var_1.val[0] ** var_2.val[0])
                        * var_2.jac[0].toarray(),
                        var_2.val[1]
                        * var_1.val[1] ** (var_2.val[1] - 1.0)
                        * var_1.jac[1].toarray()
                        + np.log(var_1.val[1])
                        * (var_1.val[1] ** var_2.val[1])
                        * var_2.jac[1].toarray(),
                        var_2.val[2]
                        * var_1.val[2] ** (var_2.val[2] - 1.0)
                        * var_1.jac[2].toarray()
                        + np.log(var_1.val[2])
                        * (var_1.val[2] ** var_2.val[2])
                        * var_2.jac[2].toarray(),
                    )
                )
            )
            return pp.ad.SparseAdArray(val, jac)
        elif op == "@":
            return False
    else:
        raise ValueError(f"Unknown classes: {type(var_1)}, {type(var_2)}.")


def _compare(actual, expected) -> None:
    """Assert that an evaluated object equals the expected one, both in type and in
    value. For Ad arrays, both the values and the Jacobians are compared."""
    assert type(actual) is type(expected)
    if isinstance(actual, float):
        assert np.isclose(actual, expected)
    elif isinstance(actual, np.ndarray):
        assert np.allclose(actual, expected)
    elif isinstance(actual, (sps.spmatrix, sps.sparray)):
        assert np.allclose(actual.toarray(), expected.toarray())
    elif isinstance(actual, pp.ad.SparseAdArray):
        assert np.allclose(actual.val, expected.val)
        assert np.allclose(actual.jac.toarray(), expected.jac.toarray())
    else:
        raise ValueError(f"Unknown type: {type(actual)}")


@pytest.mark.parametrize(
    "var_1", ["scalar", "dense", "sparse_matrix", "sparse_array", "ad"]
)
@pytest.mark.parametrize(
    "var_2", ["scalar", "dense", "sparse_matrix", "sparse_array", "ad"]
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
    (float, numpy.ndarray, scipy sparse matrices and arrays, SparseAdArray). In the
    wrapped form, all combinations are tested, since the parsing of the operator tree
    must handle each of them. In the primitive form, which is what is applied in
    forward-mode algorithmic differentiation, only combinations that involve at least
    one Ad array are tested; the remaining combinations are evaluated by numpy and scipy
    alone.

    For each combination, the result is compared with an expected value, or, if the
    combination is not supported, it is checked that the evaluation raises an error.

    """

    if not wrapped and var_1 != "ad" and var_2 != "ad":
        # If not wrapped in the abstract layer, these cases should be covered by the
        # tests for the external packages; PorePy just has to rely on e.g., numpy being
        # correctly implemented. For the wrapped case, we need to test that the parsing
        # is okay, thus we do not skip if wrapped is True.
        return

    def _var_from_string(v: str, do_wrap: bool):
        if v == "scalar":
            return _get_scalar(do_wrap)
        if v == "dense":
            return _get_dense_array(do_wrap, arithmetic_mdg)
        if v == "sparse_matrix":
            return _get_sparse_array(do_wrap, use_csr_matrix=True, mdg=arithmetic_mdg)
        if v == "sparse_array":
            return _get_sparse_array(do_wrap, use_csr_matrix=False, mdg=arithmetic_mdg)
        if v == "ad":
            return _get_ad_array(do_wrap, arithmetic_mdg)
        raise ValueError("Unknown variable type")

    # Get the actual variables from the input strings.
    v1 = _var_from_string(var_1, wrapped)
    v2 = _var_from_string(var_2, wrapped)

    # Some gymnastics is needed here: In the wrapped form, expressions need an
    # EquationSystem for evaluation and, if one of the operands is a SparseAdArray, this
    # should be the EquationSystem used to generate this operand (see method
    # _get_ad_array). If this is not the case, the EquationSystem will end up having to
    # evaluate an Ad variable that it does not know about. Therefore, the method
    # _get_ad_array returns the generated EquationSystem together with variable. If none
    # of the operands is an Ad array, we will still formally need an EquationSystem to
    # evaluate the expression, but since this will not actually be used for anything, we
    # can generate a new one and pass it as a formality.
    if wrapped:
        if var_1 == "ad":
            v1, equation_system = v1
        elif var_2 == "ad":
            # The case of both v1 and v2 being Ad variables is dealt with below.
            v2, equation_system = v2
        else:
            equation_system = pp.ad.EquationSystem(pp.MixedDimensionalGrid())
    if var_1 == "ad" and var_2 == "ad":
        # For the case of two ad variables, they should be associated with the
        # same EquationSystem, or else parsing will fail. We could have set v1 =
        # v2, but this is less likely to catch errors in the parsing. Instead,
        # we reassign v2 = v1 + v1. This also requires some adaptations in the
        # code to get the expected values, see that function.
        v2 = v1 + v1

    # Calculate the expected numerical values for this expression. This involves
    # hard-coded values for the different operators and their combinations, see the
    # function for more information. If the operation is not expected to succeed, the
    # function will return False.
    expected = _expected_value(
        _var_from_string(var_1, False), _var_from_string(var_2, False), op
    )

    # Evaluate the expression. This is a bit different for the wrapped and forward mode,
    # but the logic is the same: Try to evaluate. If this breaks, check that this was
    # not a surprise (variable expected is False).
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
            ad_base = pp.ad.initAdArrays([state])[0]
            val = equation_system._ad_parser._evaluate_single(
                expression,
                ad_base,
                equation_system,
                variable_indexer=equation_system.variable_indexer,
            )
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
            assert np.all(val_jac.val == val)
        else:
            with pytest.raises(NotImplementedError):
                equation_system.evaluate(expression, derivative=True)
