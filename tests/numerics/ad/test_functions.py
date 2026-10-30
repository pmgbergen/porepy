"""Test collection of the functions being wrapped in a
:class:`~porepy.numerics.ad.operator_functions.Function`.

For each supported function, the value and Jacobian are compared with a reference.
Functions with a uniform elementwise-derivative shape (exp, log, abs, the
trigonometric/hyperbolic functions, safe_power) are covered by a single table-driven,
parametrized test, run against a plain ndarray, a SparseAdArray, and a
DiagonalAdArray. Functions with more specific behavior (mask_by_threshold, clip) have
dedicated parametrized tests.

Tests that exercise a single function are grouped into a ``Test<FunctionName>`` class;
the table-driven tests above, which each cover many functions, are not.

"""

import functools
import warnings
from typing import Literal

import numpy as np
import pytest
import scipy.sparse as sps

from porepy.numerics.ad import AdArrayBase, DiagonalAdArray, SparseAdArray
from porepy.numerics.ad import functions as af

warnings.simplefilter("ignore", sps.SparseEfficiencyWarning)

AdRepresentation = Literal["full_ad", "diagonal_ad"]


# --- Shared scaffolding for elementwise functions, across representations ---
#
# Each entry maps a function name to the callable, a reference value function, and a
# reference (elementwise) derivative function. Used by test_elementwise_function below
# to check the same function against a plain ndarray, a SparseAdArray, and
# a DiagonalAdArray, without hand-writing a test per representation per function.
#
# `domain` gives a value range avoiding singularities of the function or its derivative
# (e.g. log needs positive values, arcsin/arccos need [-1, 1]).
ELEMENTWISE_FUNCTIONS = {
    "exp": dict(func=af.exp, value=np.exp, derivative=np.exp, domain=(0.5, 3.5)),
    "log": dict(
        func=af.log, value=np.log, derivative=lambda v: 1 / v, domain=(0.5, 3.5)
    ),
    "abs": dict(func=af.abs, value=np.abs, derivative=np.sign, domain=(-3.5, 0.5)),
    "sin": dict(func=af.sin, value=np.sin, derivative=np.cos, domain=(-1.0, 1.0)),
    "cos": dict(
        func=af.cos, value=np.cos, derivative=lambda v: -np.sin(v), domain=(-1.0, 1.0)
    ),
    "tan": dict(
        func=af.tan,
        value=np.tan,
        derivative=lambda v: np.cos(v) ** (-2),
        domain=(-1.0, 1.0),
    ),
    "arcsin": dict(
        func=af.arcsin,
        value=np.arcsin,
        derivative=lambda v: (1 - v**2) ** (-0.5),
        domain=(-0.8, 0.8),
    ),
    "arccos": dict(
        func=af.arccos,
        value=np.arccos,
        derivative=lambda v: -((1 - v**2) ** (-0.5)),
        domain=(-0.8, 0.8),
    ),
    "arctan": dict(
        func=af.arctan,
        value=np.arctan,
        derivative=lambda v: (v**2 + 1) ** (-1),
        domain=(-2.0, 2.0),
    ),
    "sinh": dict(func=af.sinh, value=np.sinh, derivative=np.cosh, domain=(-1.0, 1.0)),
    "cosh": dict(func=af.cosh, value=np.cosh, derivative=np.sinh, domain=(-1.0, 1.0)),
    "tanh": dict(
        func=af.tanh,
        value=np.tanh,
        derivative=lambda v: np.cosh(v) ** (-2),
        domain=(-1.0, 1.0),
    ),
    "arcsinh": dict(
        func=af.arcsinh,
        value=np.arcsinh,
        derivative=lambda v: (v**2 + 1) ** (-0.5),
        domain=(-1.0, 1.0),
    ),
    "arccosh": dict(
        func=af.arccosh,
        value=np.arccosh,
        derivative=lambda v: (v - 1) ** (-0.5) * (v + 1) ** (-0.5),
        domain=(1.2, 3.0),
    ),
    "arctanh": dict(
        func=af.arctanh,
        value=np.arctanh,
        derivative=lambda v: (1 - v**2) ** (-1),
        domain=(-0.8, 0.8),
    ),
    "heaviside_smooth": dict(
        func=af.heaviside_smooth,
        value=lambda v: 0.5 * (1 + 2 * np.pi ** (-1) * np.arctan(v * 1e3)),
        derivative=lambda v: np.pi ** (-1) * (1e-3 * (1e-3**2 + v**2) ** (-1)),
        domain=(-1.0, 1.0),
    ),
    "safe_power": dict(
        func=functools.partial(af.safe_power, -1, 0.0, 1e-10),
        value=lambda v: v ** (-1.0),
        derivative=lambda v: -1.0 * v ** (-2.0),
        domain=(0.5, 3.5),
    ),
}
# characteristic_function and heaviside are locally constant (almost everywhere), so
# their Jacobian is defined to be identically zero, regardless of the Jacobian of the
# argument (unlike the elementwise functions above).
ZERO_DERIVATIVE_FUNCTIONS = {
    "characteristic_function": dict(
        func=functools.partial(af.characteristic_function, 0.5),
        value=lambda v: np.isclose(v, 0, atol=0.5).astype(float),
        domain=(-2.0, 2.0),
    ),
    "heaviside": dict(
        func=functools.partial(af.heaviside, 0.5),
        value=lambda v: np.heaviside(v, 0.5),
        domain=(-2.0, 2.0),
    ),
}


# Jacobians of the Ad arrays passed to the tested functions, with respect to the
# underlying variables. By the chain rule, the Jacobian of an elementwise function of an
# Ad array is the derivative of the function times the Jacobian of the Ad array; using
# Jacobians other than the identity checks that this product is formed. The full
# Jacobian has off-diagonal entries, the diagonal one has only diagonal entries.
ARG_JAC: dict[AdRepresentation, np.ndarray] = {
    "full_ad": np.array([[3.0, 2.0, 1.0], [5.0, 6.0, 1.0], [2.0, 3.0, 5.0]]),
    "diagonal_ad": np.diag([2.0, -1.0, 0.5]),
}
# Functions taking two Ad arrays (e.g. maximum) need an unrelated Jacobian for the
# second argument, since equal Jacobians for both arguments could hide bugs.
SECOND_ARG_JAC: dict[AdRepresentation, np.ndarray] = {
    "full_ad": np.array([[9.0, 9.0, 9.0], [8.0, 8.0, 8.0], [7.0, 7.0, 7.0]]),
    "diagonal_ad": np.diag([4.0, 3.0, -2.0]),
}


def _make_ad(
    val: np.ndarray,
    representation: AdRepresentation,
    jacobians: dict[AdRepresentation, np.ndarray] = ARG_JAC,
) -> AdArrayBase:
    """Helper function to build an Ad array with given values and Jacobian.

    Parameters:
        val: Values of the Ad array.
        representation: Build a SparseAdArray (``"full_ad"``) or a DiagonalAdArray
            (``"diagonal_ad"``).
        jacobians: Jacobian of the Ad array for each representation, as a dense matrix.
            For ``"diagonal_ad"``, only the diagonal is used.

    Returns:
        The Ad array.

    """
    jac = jacobians[representation]
    if representation == "full_ad":
        return SparseAdArray(val, sps.csc_matrix(jac))
    return DiagonalAdArray(
        val,
        np.atleast_2d(np.diag(jac)),
        [np.arange(val.size)],
        val.size,
    )


def _dense_jac(result: AdArrayBase) -> np.ndarray:
    """Return Jacobian of a SparseAdArray or DiagonalAdArray, as a dense array."""
    return result.full_jac.toarray()


@pytest.mark.parametrize("func_name", list(ELEMENTWISE_FUNCTIONS))
@pytest.mark.parametrize("representation", ["ndarray", "full_ad", "diagonal_ad"])
def test_elementwise_function(
    func_name: str, representation: AdRepresentation | Literal["ndarray"]
):
    """Value and Jacobian of an elementwise AD function, for a plain ndarray, a full
    (non-diagonal) SparseAdArray, and a DiagonalAdArray argument."""
    spec = ELEMENTWISE_FUNCTIONS[func_name]
    lo, hi = spec["domain"]
    val = np.linspace(lo, hi, 3)
    expected_val = spec["value"](val)

    if representation == "ndarray":
        result = spec["func"](val)
        np.testing.assert_allclose(result, expected_val)
        return

    var = _make_ad(val, representation)
    expected_jac = np.diag(spec["derivative"](val)) @ ARG_JAC[representation]

    result = spec["func"](var)
    np.testing.assert_allclose(result.val, expected_val)
    np.testing.assert_allclose(_dense_jac(result), expected_jac)
    if representation == "diagonal_ad":
        assert isinstance(result, DiagonalAdArray)
    else:
        assert isinstance(result, SparseAdArray)


@pytest.mark.parametrize("func_name", list(ZERO_DERIVATIVE_FUNCTIONS))
@pytest.mark.parametrize("representation", ["ndarray", "full_ad", "diagonal_ad"])
def test_zero_derivative_function(
    func_name: str, representation: AdRepresentation | Literal["ndarray"]
):
    """Value and (identically zero) Jacobian of characteristic_function/heaviside, for a
    plain ndarray, a SparseAdArray, and a DiagonalAdArray argument."""
    spec = ZERO_DERIVATIVE_FUNCTIONS[func_name]
    lo, hi = spec["domain"]
    val = np.linspace(lo, hi, 3)
    expected_val = spec["value"](val)

    if representation == "ndarray":
        result = spec["func"](val)
        np.testing.assert_allclose(result, expected_val)
        return

    var = _make_ad(val, representation)

    result = spec["func"](var)
    np.testing.assert_allclose(result.val, expected_val)
    np.testing.assert_allclose(_dense_jac(result), np.zeros((3, 3)))
    if representation == "diagonal_ad":
        assert isinstance(result, DiagonalAdArray)


class TestRegularizedHeaviside:
    def test_ndarray(self):
        reg = af.RegularizedHeaviside(af.heaviside_smooth)
        val = np.linspace(-1.0, 1.0, 3)
        np.testing.assert_allclose(reg(val, zerovalue=0.5), np.heaviside(val, 0.5))

    @pytest.mark.parametrize("representation", ["full_ad", "diagonal_ad"])
    def test_ad_representations(self, representation: AdRepresentation):
        """Value and Jacobian of RegularizedHeaviside (using heaviside_smooth as the
        regularization).

        The value is the exact Heaviside function, while the Jacobian is that of the
        regularization, and the representation is that of the argument.
        """
        reg = af.RegularizedHeaviside(af.heaviside_smooth)
        heaviside_smooth_der = ELEMENTWISE_FUNCTIONS["heaviside_smooth"]["derivative"]
        val = np.linspace(-1.0, 1.0, 3)
        expected_val = np.heaviside(val, 0.0)

        var = _make_ad(val, representation)
        expected_jac = np.diag(heaviside_smooth_der(val)) @ ARG_JAC[representation]

        result = reg(var)
        np.testing.assert_allclose(result.val, expected_val)
        np.testing.assert_allclose(_dense_jac(result), expected_jac)
        if representation == "diagonal_ad":
            assert isinstance(result, DiagonalAdArray)
        else:
            assert isinstance(result, SparseAdArray)


class TestMaximum:
    def test_ndarray_only(self):
        a = np.array([1.0, 5.0, 2.0])
        b = np.array([3.0, 2.0, 2.0])
        np.testing.assert_allclose(af.maximum(a, b), np.array([3.0, 5.0, 2.0]))

    @pytest.mark.parametrize("representation", ["full_ad", "diagonal_ad"])
    @pytest.mark.parametrize(
        "constant",
        [np.array([3.0, 2.0, 2.0]), 2.0, np.int64(2)],
        ids=["ndarray", "float", "numpy_int"],
    )
    @pytest.mark.parametrize("ad_first", [True, False])
    def test_ad_and_constant(
        self,
        representation: AdRepresentation,
        constant: np.ndarray | float | np.int64,
        ad_first: bool,
    ):
        """One argument is a SparseAdArray (full or diagonal), the other a constant
        (ndarray or scalar) with an implicit zero Jacobian, in either argument order. A
        scalar is broadcast to the shape of the SparseAdArray. All constants tie with
        the SparseAdArray at index 2, where the first argument wins."""
        val = np.array([1.0, 5.0, 2.0])
        var = _make_ad(val, representation)

        if ad_first:
            result = af.maximum(var, constant)
            pick_ad = val >= constant
        else:
            result = af.maximum(constant, var)
            pick_ad = val > constant

        expected_val = np.maximum(val, constant)
        # Build the Jacobian: Where pick_ad is True, the value 0 is assigned; else, the
        # Jacobian of var_0 (that is, ARG_JAC[representation]) is used.
        expected_jac = np.where(pick_ad.reshape((-1, 1)), ARG_JAC[representation], 0.0)

        np.testing.assert_allclose(result.val, expected_val)
        np.testing.assert_allclose(_dense_jac(result), expected_jac)
        # A mixed call is computed in the full representation.
        assert isinstance(result, SparseAdArray)

    @pytest.mark.parametrize(
        "representation_0,representation_1",
        [
            ("full_ad", "full_ad"),
            ("diagonal_ad", "diagonal_ad"),
            ("diagonal_ad", "full_ad"),
            ("full_ad", "diagonal_ad"),
        ],
    )
    def test_ad_representations(
        self, representation_0: AdRepresentation, representation_1: AdRepresentation
    ):
        """maximum with var_0 and var_1 independently full or diagonal SparseAdArrays,
        including the two mixed combinations. Index 2 is a tie (equal values), where the
        documented convention is that var_0's Jacobian is used."""
        val_0 = np.array([1.0, 5.0, 2.0])
        val_1 = np.array([3.0, 2.0, 2.0])

        var_0 = _make_ad(val_0, representation_0)
        var_1 = _make_ad(val_1, representation_1, SECOND_ARG_JAC)

        result = af.maximum(var_0, var_1)

        expected_val = np.array([3.0, 5.0, 2.0])
        pick_1 = val_1 > val_0
        expected_jac = np.where(
            pick_1[:, None],
            SECOND_ARG_JAC[representation_1],
            ARG_JAC[representation_0],
        )

        np.testing.assert_allclose(result.val, expected_val)
        np.testing.assert_allclose(_dense_jac(result), expected_jac)
        if representation_0 == "diagonal_ad" and representation_1 == "diagonal_ad":
            assert isinstance(result, DiagonalAdArray)
        else:
            assert isinstance(result, SparseAdArray)


class TestL2Norm:
    # Jacobians of the argument, as ARG_JAC at module level, but with four degrees of
    # freedom to match two vectors of dimension two.
    LOC_ARG_JAC: dict[AdRepresentation, np.ndarray] = {
        "full_ad": np.array(
            [
                [2.0, 0.0, 1.0, 0.0],
                [0.0, 3.0, 0.0, 1.0],
                [1.0, 0.0, 2.0, 0.0],
                [0.0, 1.0, 0.0, 3.0],
            ]
        ),
        "diagonal_ad": np.diag([2.0, 3.0, 2.0, 3.0]),
    }

    def test_ndarray(self):
        # Two 2-vectors, ordered [u0, v0, u1, v1] (Fortran/dim-major order): (3, 4) has
        # norm 5; (0, 0) has norm 0.
        var = np.array([3.0, 4.0, 0.0, 0.0])
        np.testing.assert_allclose(af.l2_norm(2, var), np.array([5.0, 0.0]))

    @pytest.mark.parametrize("representation", ["full_ad", "diagonal_ad"])
    def test_dim_one(self, representation: AdRepresentation):
        """For dim=1, l2_norm is the absolute value, and a DiagonalAdArray argument
        gives a DiagonalAdArray."""
        val = np.array([-2.0, 3.0, -0.5])
        var = _make_ad(val, representation)

        result = af.l2_norm(1, var)

        expected_jac = np.diag([-1.0, 1.0, -1.0]) @ ARG_JAC[representation]
        np.testing.assert_allclose(result.val, np.array([2.0, 3.0, 0.5]))
        np.testing.assert_allclose(_dense_jac(result), expected_jac)
        if representation == "diagonal_ad":
            assert isinstance(result, DiagonalAdArray)

    @pytest.mark.parametrize("representation", ["full_ad", "diagonal_ad"])
    def test_dim_two(self, representation: AdRepresentation):
        """L2 norm reduces dim=2 components per output entry, so a DiagonalAdArray input
        must be converted to a SparseAdArray rather than stay diagonal."""
        # Two 2-vectors, ordered [u0, v0, u1, v1]: (3, 4) has norm 5; (0, 0) has norm 0
        # (per the function docstring, a zero vector gets a Jacobian entry of 1, not 0).
        val = np.array([3.0, 4.0, 0.0, 0.0])

        var = _make_ad(val, representation, self.LOC_ARG_JAC)

        result = af.l2_norm(2, var)

        expected_val = np.array([5.0, 0.0])
        d_norm_d_val = np.array([[0.6, 0.8, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
        expected_jac = d_norm_d_val @ self.LOC_ARG_JAC[representation]

        np.testing.assert_allclose(result.val, expected_val)
        np.testing.assert_allclose(_dense_jac(result), expected_jac)
        # dim > 1 always return a SparseAdArray, regardless of the input's
        # representation.
        assert isinstance(result, SparseAdArray)


class TestMaskByThreshold:
    @pytest.mark.parametrize(
        "char_var,var,tol,expected_val",
        [
            pytest.param(
                np.array([0.5, -0.1, 2.0]),
                10.0,
                0.0,
                np.array([10.0, 0.0, 10.0]),
                id="scalar",
            ),
            pytest.param(
                np.array([0.5, -0.1, 2.0]),
                np.array([10, 20, 30]),
                0.0,
                np.array([10, 0, 30]),
                id="ndarray",
            ),
            # Masked entries are zero even if var is NaN there.
            pytest.param(
                np.array([0.5, -0.1, 2.0]),
                np.array([10.0, np.nan, 30.0]),
                0.0,
                np.array([10.0, 0.0, 30.0]),
                id="nd_array_containing_nan",
            ),
        ],
    )
    def test_no_ad(self, char_var, var, tol, expected_val):
        """mask_by_threshold with neither argument an Ad array."""
        result = af.mask_by_threshold(tol, char_var, var)
        np.testing.assert_allclose(result, expected_val)

    @pytest.mark.parametrize(
        "char_val,tol,kept",
        [
            pytest.param(
                np.array([0.05, 0.1, 1.0]), 0.08, [0.0, 1.0, 1.0], id="with_tolerance"
            ),
            pytest.param(
                np.array([0.1, 0.2, 0.3]), 0.5, [0.0, 0.0, 0.0], id="all_masked"
            ),
            pytest.param(
                np.array([0.5, 1.0, 2.0]), 0.0, [1.0, 1.0, 1.0], id="all_kept"
            ),
        ],
    )
    @pytest.mark.parametrize(
        "char_representation,var_representation",
        [
            ("full_ad", "full_ad"),
            ("diagonal_ad", "diagonal_ad"),
            ("diagonal_ad", "full_ad"),
            ("full_ad", "diagonal_ad"),
        ],
    )
    def test_ad(
        self,
        char_val: np.ndarray,
        tol: float,
        kept: list[float],
        char_representation: AdRepresentation,
        var_representation: AdRepresentation,
    ):
        """mask_by_threshold with char_var and var independently full or diagonal
        SparseAdArrays. The Jacobian of the result is that of var in kept rows, and zero
        in masked rows; the Jacobian of char_var does not enter."""
        var_val = np.array([10.0, 20.0, 30.0])
        char_var = _make_ad(char_val, char_representation)
        var = _make_ad(var_val, var_representation, SECOND_ARG_JAC)

        result = af.mask_by_threshold(tol, char_var, var)

        mask = np.diag(kept)
        np.testing.assert_allclose(result.val, mask @ var_val)
        np.testing.assert_allclose(
            _dense_jac(result), mask @ SECOND_ARG_JAC[var_representation]
        )
        if var_representation == "diagonal_ad":
            assert isinstance(result, DiagonalAdArray)
        else:
            assert isinstance(result, SparseAdArray)


class TestClip:
    """Tests for clip, with bounds 0 and 3."""

    @pytest.mark.parametrize(
        "var,expected_val",
        [
            pytest.param(
                np.array([-2.0, 0.0, 1.5, 5.0]),
                np.array([0.0, 0.0, 1.5, 3.0]),
                id="ndarray",
            ),
            pytest.param(2.0, 2.0, id="float_interior"),
            pytest.param(-1.0, 0.0, id="float_below"),
            pytest.param(5.0, 3.0, id="float_above"),
        ],
    )
    def test_no_ad(self, var, expected_val):
        np.testing.assert_allclose(af.clip(var, 0.0, 3.0), expected_val)

    @pytest.mark.parametrize(
        "val,expected_val,kept",
        [
            pytest.param(
                [0.5, 1.0, 2.5], [0.5, 1.0, 2.5], [1.0, 1.0, 1.0], id="interior"
            ),
            pytest.param(
                [0.0, 3.0, 1.0], [0.0, 3.0, 1.0], [0.0, 0.0, 1.0], id="at_bounds"
            ),
            pytest.param(
                [-1.0, 5.0, 1.0], [0.0, 3.0, 1.0], [0.0, 0.0, 1.0], id="outside_bounds"
            ),
        ],
    )
    @pytest.mark.parametrize("representation", ["full_ad", "diagonal_ad"])
    def test_ad(
        self,
        val: list[float],
        expected_val: list[float],
        kept: list[float],
        representation: AdRepresentation,
    ):
        """clip of a full or diagonal SparseAdArray. Rows of the Jacobian are kept for
        values strictly between the bounds, and zeroed for values at or outside the
        bounds."""
        var = _make_ad(np.array(val), representation)

        result = af.clip(var, 0.0, 3.0)

        expected_jac = np.diag(kept) @ ARG_JAC[representation]
        np.testing.assert_allclose(result.val, expected_val)
        np.testing.assert_allclose(_dense_jac(result), expected_jac)
        if representation == "diagonal_ad":
            assert isinstance(result, DiagonalAdArray)

    @pytest.mark.parametrize("representation", ["full_ad", "diagonal_ad"])
    def test_does_not_mutate_input(self, representation: AdRepresentation):
        # Clip values both below and above the bounds, and check that the argument is
        # unchanged.
        truth = ARG_JAC[representation].copy()
        var = _make_ad(np.array([-1.0, 2.0, 5.0]), representation)
        _ = af.clip(var, 0.0, 3.0)
        np.testing.assert_allclose(var.val, np.array([-1.0, 2.0, 5.0]))
        np.testing.assert_allclose(_dense_jac(var), truth)
