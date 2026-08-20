"""Test collection of the functions being wrapped in a
:class:`~porepy.numerics.ad.operator_functions.Function`.

For each supported function, the value and Jacobian are compared with a reference.
Functions with a uniform elementwise-derivative shape (exp, log, abs, the
trigonometric/hyperbolic functions, safe_power) are covered by a single table-driven,
parametrized test, run against a plain ndarray, a full (non-diagonal) AdArray, and a
DiagonalAdArray. Functions with more specific behavior (mask_by_threshold, clip) keep
dedicated parametrized tests.

"""

import functools
import warnings

import numpy as np
import pytest
import scipy.sparse as sps

from porepy.numerics.ad import AdArray
from porepy.numerics.ad import functions as af
from porepy.numerics.ad.forward_mode import DiagonalAdArray

warnings.simplefilter("ignore", sps.SparseEfficiencyWarning)


# --- Shared scaffolding for elementwise functions, across representations ---
#
# Each entry maps a function name to the callable, a reference value function, and a
# reference (elementwise) derivative function. Used by test_elementwise_function below
# to check the same function against a plain ndarray, a full (non-diagonal) AdArray,
# and a DiagonalAdArray, without hand-writing a test per representation per function.
#
# `domain` gives a value range avoiding singularities of the function or its
# derivative (e.g. log needs positive values, arcsin/arccos need [-1, 1]).
ELEMENTWISE_FUNCTIONS = {
    "exp": dict(func=af.exp, value=np.exp, derivative=np.exp, domain=(0.5, 3.5)),
    "log": dict(
        func=af.log, value=np.log, derivative=lambda v: 1 / v, domain=(0.5, 3.5)
    ),
    "abs": dict(func=af.abs, value=np.abs, derivative=np.sign, domain=(-3.5, -0.5)),
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

# A genuinely non-diagonal Jacobian, to make sure the chain rule is applied correctly
# for a full AdArray. A second, unrelated variant is used where a test combines two
# independently-represented AdArrays (e.g. maximum).
_FULL_CHAIN_JAC = np.array([[3.0, 2.0, 1.0], [5.0, 6.0, 1.0], [2.0, 3.0, 5.0]])
_FULL_CHAIN_JAC_2 = np.array([[9.0, 9.0, 9.0], [8.0, 8.0, 8.0], [7.0, 7.0, 7.0]])
# The diagonal counterparts: a per-entry scalar derivative, distinct from the identity
# so that the chain rule is exercised (not just the local derivative).
_DIAGONAL_CHAIN_JAC = np.array([2.0, -1.0, 0.5])
_DIAGONAL_CHAIN_JAC_2 = np.array([4.0, 3.0, -2.0])


def _dense_jac(result: AdArray) -> np.ndarray:
    """Jacobian of an AdArray or DiagonalAdArray, as a dense array."""
    return (result.to_full() if result.is_diagonal else result).jac.toarray()


def _make_ad(val: np.ndarray, representation: str, variant: str = "1") -> AdArray:
    """Build a full or diagonal AdArray for ``val``, using the chain Jacobians above.

    Used where a test needs to construct AdArray/DiagonalAdArray arguments outside of
    the ``test_elementwise_function`` / ``test_zero_derivative_function`` tables, e.g.
    to combine two independently-represented arguments. ``variant="2"`` selects the
    second, unrelated chain Jacobian, for tests combining two AdArrays where using the
    same Jacobian for both would hide bugs (see comment above).
    """
    full_jac = _FULL_CHAIN_JAC if variant == "1" else _FULL_CHAIN_JAC_2
    diagonal_jac = _DIAGONAL_CHAIN_JAC if variant == "1" else _DIAGONAL_CHAIN_JAC_2
    if representation == "full_ad":
        return AdArray(val, sps.csc_matrix(full_jac))
    return DiagonalAdArray(
        val,
        np.atleast_2d(diagonal_jac),
        np.arange(val.size),
        [np.arange(val.size)],
        val.size,
    )


# The dense-Jacobian equivalent of _make_ad(val, representation, variant), independent
# of val (all chain Jacobians above are constant), used to build expected results for
# functions combining two independently-represented AdArray arguments.
_DENSE_CHAIN_JAC = {
    "full_ad": _FULL_CHAIN_JAC,
    "diagonal_ad": np.diag(_DIAGONAL_CHAIN_JAC),
}
_DENSE_CHAIN_JAC_2 = {
    "full_ad": _FULL_CHAIN_JAC_2,
    "diagonal_ad": np.diag(_DIAGONAL_CHAIN_JAC_2),
}


@pytest.mark.parametrize("func_name", list(ELEMENTWISE_FUNCTIONS))
@pytest.mark.parametrize("representation", ["ndarray", "full_ad", "diagonal_ad"])
def test_elementwise_function(func_name: str, representation: str):
    """Value and Jacobian of an elementwise AD function, for a plain ndarray, a full
    (non-diagonal) AdArray, and a DiagonalAdArray argument."""
    spec = ELEMENTWISE_FUNCTIONS[func_name]
    lo, hi = spec["domain"]
    val = np.linspace(lo, hi, 3)
    expected_val = spec["value"](val)

    if representation == "ndarray":
        result = spec["func"](val)
        assert np.allclose(result, expected_val)
        return

    if representation == "full_ad":
        var = AdArray(val, sps.csc_matrix(_FULL_CHAIN_JAC))
        expected_jac = np.diag(spec["derivative"](val)) @ _FULL_CHAIN_JAC
    else:
        var = DiagonalAdArray(
            val,
            np.atleast_2d(_DIAGONAL_CHAIN_JAC),
            np.arange(val.size),
            [np.arange(val.size)],
            val.size,
        )
        expected_jac = np.diag(spec["derivative"](val) * _DIAGONAL_CHAIN_JAC)

    result = spec["func"](var)
    assert np.allclose(result.val, expected_val)
    assert np.allclose(_dense_jac(result), expected_jac)
    if representation == "diagonal_ad":
        assert result.is_diagonal


# --- Shared scaffolding for zero-derivative functions ---
#
# characteristic_function and heaviside are locally constant (almost everywhere), so
# their Jacobian is defined to be identically zero, regardless of the chain rule
# through the argument's own Jacobian (unlike the elementwise functions above).
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


@pytest.mark.parametrize("func_name", list(ZERO_DERIVATIVE_FUNCTIONS))
@pytest.mark.parametrize("representation", ["ndarray", "full_ad", "diagonal_ad"])
def test_zero_derivative_function(func_name: str, representation: str):
    """Value and (identically zero) Jacobian of characteristic_function/heaviside,
    for a plain ndarray, a full AdArray, and a DiagonalAdArray argument."""
    spec = ZERO_DERIVATIVE_FUNCTIONS[func_name]
    lo, hi = spec["domain"]
    val = np.linspace(lo, hi, 3)
    expected_val = spec["value"](val)

    if representation == "ndarray":
        result = spec["func"](val)
        assert np.allclose(result, expected_val)
        return

    if representation == "full_ad":
        var = AdArray(val, sps.csc_matrix(_FULL_CHAIN_JAC))
    else:
        var = DiagonalAdArray(
            val,
            np.atleast_2d(_DIAGONAL_CHAIN_JAC),
            np.arange(val.size),
            [np.arange(val.size)],
            val.size,
        )

    result = spec["func"](var)
    assert np.allclose(result.val, expected_val)
    assert np.allclose(_dense_jac(result), np.zeros((3, 3)))
    if representation == "diagonal_ad":
        assert result.is_diagonal


# Function: RegularizedHeaviside
def test_regularized_heaviside_ndarray():
    reg = af.RegularizedHeaviside(af.heaviside_smooth)
    val = np.linspace(-1.0, 1.0, 3)
    assert np.allclose(reg(val, zerovalue=0.5), np.heaviside(val, 0.5))


@pytest.mark.parametrize("representation", ["full_ad", "diagonal_ad"])
def test_regularized_heaviside(representation: str):
    """Value and Jacobian of RegularizedHeaviside (using heaviside_smooth as the
    regularization), for a full AdArray and a DiagonalAdArray argument.

    The Jacobian is inherited entirely from the regularization function, so this
    doubles as a check that RegularizedHeaviside forwards it (and the diagonal
    representation) rather than rebuilding it.
    """
    reg = af.RegularizedHeaviside(af.heaviside_smooth)
    heaviside_smooth_der = ELEMENTWISE_FUNCTIONS["heaviside_smooth"]["derivative"]
    val = np.linspace(-1.0, 1.0, 3)
    expected_val = np.heaviside(val, 0.0)

    if representation == "full_ad":
        var = AdArray(val, sps.csc_matrix(_FULL_CHAIN_JAC))
        expected_jac = np.diag(heaviside_smooth_der(val)) @ _FULL_CHAIN_JAC
    else:
        var = DiagonalAdArray(
            val,
            np.atleast_2d(_DIAGONAL_CHAIN_JAC),
            np.arange(val.size),
            [np.arange(val.size)],
            val.size,
        )
        expected_jac = np.diag(heaviside_smooth_der(val) * _DIAGONAL_CHAIN_JAC)

    result = reg(var)
    assert np.allclose(result.val, expected_val)
    assert np.allclose(_dense_jac(result), expected_jac)
    if representation == "diagonal_ad":
        assert result.is_diagonal


# Function: maximum
def test_maximum_ndarray_only():
    a = np.array([1.0, 5.0, 2.0])
    b = np.array([3.0, 2.0, 2.0])
    assert np.allclose(af.maximum(a, b), np.array([3.0, 5.0, 2.0]))


@pytest.mark.parametrize("representation", ["full_ad", "diagonal_ad"])
def test_maximum_ad_and_ndarray(representation: str):
    """One argument is an AdArray (full or diagonal), the other a plain ndarray with
    an implicit zero Jacobian."""
    val_0 = np.array([1.0, 5.0, 2.0])
    val_1 = np.array([3.0, 2.0, 2.0])
    var_0 = _make_ad(val_0, representation)

    result = af.maximum(var_0, val_1)

    expected_val = np.array([3.0, 5.0, 2.0])
    pick_1 = val_1 > val_0
    expected_jac = np.where(pick_1[:, None], 0.0, _DENSE_CHAIN_JAC[representation])

    assert np.allclose(result.val, expected_val)
    assert np.allclose(_dense_jac(result), expected_jac)


@pytest.mark.parametrize(
    "representation_0,representation_1",
    [
        ("full_ad", "full_ad"),
        ("diagonal_ad", "diagonal_ad"),
        ("diagonal_ad", "full_ad"),
        ("full_ad", "diagonal_ad"),
    ],
)
def test_maximum_ad_representations(representation_0: str, representation_1: str):
    """maximum with var_0 and var_1 independently full or diagonal AdArrays,
    including the two mixed combinations. Index 2 is a tie (equal values), where the
    documented convention is that var_0's Jacobian is used."""
    val_0 = np.array([1.0, 5.0, 2.0])
    val_1 = np.array([3.0, 2.0, 2.0])

    var_0 = _make_ad(val_0, representation_0, variant="1")
    var_1 = _make_ad(val_1, representation_1, variant="2")

    result = af.maximum(var_0, var_1)

    expected_val = np.array([3.0, 5.0, 2.0])
    pick_1 = val_1 > val_0
    jac_0 = _DENSE_CHAIN_JAC[representation_0]
    jac_1 = _DENSE_CHAIN_JAC_2[representation_1]
    expected_jac = np.where(pick_1[:, None], jac_1, jac_0)

    assert np.allclose(result.val, expected_val)
    assert np.allclose(_dense_jac(result), expected_jac)
    if representation_0 == "diagonal_ad" and representation_1 == "diagonal_ad":
        assert result.is_diagonal
    else:
        assert not result.is_diagonal


# Function: mask_by_threshold
@pytest.mark.parametrize(
    "char_var,var,tol,expected_val,expected_jac",
    [
        # Test case 1: scalar, no AdArray
        pytest.param(
            np.array([0.5, -0.1, 2.0]),
            10.0,
            0.0,
            np.array([10.0, 0.0, 10.0]),
            None,
            id="scalar_no_advar",
        ),
        # Test case 2: ndarray, no AdArray
        pytest.param(
            np.array([0.5, -0.1, 2.0]),
            np.array([10, 20, 30]),
            0.0,
            np.array([10, 0, 30]),
            None,
            id="array_no_advar",
        ),
        # Test case 3: NaN * 0 = 0
        pytest.param(
            np.array([0.5, -0.1, 2.0]),
            np.array([10.0, np.nan, 30.0]),
            0.0,
            np.array([10.0, 0.0, 30.0]),
            None,
            id="nan_times_zero",
        ),
        # Test case 4: AdArray with tolerance
        pytest.param(
            AdArray(np.array([0.05, 0.1, 1.0]), sps.csr_matrix((3, 3))),
            AdArray(np.array([10, 20, 30]), sps.csr_matrix(np.diag([1, 1, 1]))),
            0.08,
            np.array([0, 20, 30]),
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]),
            id="adarray_with_tolerance",
        ),
        # Test case 5: all masked
        pytest.param(
            AdArray(np.array([0.1, 0.2, 0.3]), sps.csr_matrix((3, 3))),
            AdArray(np.array([10, 20, 30]), sps.csr_matrix(np.diag([1, 1, 1]))),
            0.5,
            np.array([0, 0, 0]),
            np.zeros((3, 3)),
            id="all_masked",
        ),
        # Test case 6: all kept
        pytest.param(
            AdArray(np.array([0.5, 1.0, 2.0]), sps.csr_matrix((3, 3))),
            AdArray(np.array([10, 20, 30]), sps.csr_matrix(np.diag([1, 1, 1]))),
            0.0,
            np.array([10, 20, 30]),
            np.diag([1, 1, 1]),
            id="all_kept",
        ),
    ],
)
def test_mask_by_threshold(char_var, var, tol, expected_val, expected_jac):
    """Parametrized test for mask_by_threshold covering multiple cases."""
    result = af.mask_by_threshold(tol, char_var, var)

    # Check values
    assert np.allclose(result.val if hasattr(result, "val") else result, expected_val)

    # Check Jacobian if expected
    if expected_jac is not None:
        assert hasattr(result, "jac"), "Expected AdArray with Jacobian"
        assert np.allclose(result.jac.toarray(), expected_jac)


@pytest.mark.parametrize(
    "char_representation,var_representation",
    [
        ("full_ad", "full_ad"),
        ("diagonal_ad", "diagonal_ad"),
        ("diagonal_ad", "full_ad"),
        ("full_ad", "diagonal_ad"),
    ],
)
def test_mask_by_threshold_representations(
    char_representation: str, var_representation: str
):
    """mask_by_threshold with char_var and var independently full or diagonal
    AdArrays, including the two mixed combinations."""
    tol = 0.15
    char_val = np.array([0.05, 0.2, 0.3])
    var_val = np.array([10.0, 20.0, 30.0])
    char_inds = char_val > tol

    char_var = _make_ad(char_val, char_representation)
    var = _make_ad(var_val, var_representation)

    result = af.mask_by_threshold(tol, char_var, var)

    expected_val = var_val.copy()
    expected_val[~char_inds] = 0.0
    if var_representation == "full_ad":
        expected_jac = np.diag(char_inds.astype(float)) @ _FULL_CHAIN_JAC
    else:
        expected_jac = np.diag(char_inds.astype(float) * _DIAGONAL_CHAIN_JAC)

    assert np.allclose(result.val, expected_val)
    assert np.allclose(_dense_jac(result), expected_jac)
    if var_representation == "diagonal_ad":
        assert result.is_diagonal


# Function: clip


def test_clip_ndarray():
    # Values entirely within bounds, at min, at max, and outside both bounds.
    var = np.array([-2.0, 0.0, 1.5, 5.0])
    result = af.clip(var, 0.0, 3.0)
    assert np.allclose(result, np.array([0.0, 0.0, 1.5, 3.0]))


def test_clip_float():
    assert af.clip(2.0, 0.0, 3.0) == 2.0
    assert af.clip(-1.0, 0.0, 3.0) == 0.0
    assert af.clip(5.0, 0.0, 3.0) == 3.0


def test_clip_adarray_values():
    # Check that values are clipped correctly.
    val = np.array([-1.0, 1.0, 4.0])
    J = sps.eye(3, format="csr")
    a = AdArray(val, J)
    b = af.clip(a, 0.0, 3.0)
    assert np.allclose(b.val, np.array([0.0, 1.0, 3.0]))


def test_clip_adarray_jacobian_interior():
    # For values strictly inside [min_val, max_val], the Jacobian is preserved.
    val = np.array([1.0, 2.0])
    J = sps.csr_matrix(np.array([[3.0, 0.0], [0.0, 5.0]]))
    a = AdArray(val, J)
    b = af.clip(a, 0.0, 3.0)
    assert np.allclose(b.jac.toarray(), J.toarray())


def test_clip_adarray_jacobian_at_bounds():
    # For values exactly at min or max, the Jacobian is zeroed out.
    val = np.array([0.0, 3.0])
    J = sps.eye(2, format="csr")
    a = AdArray(val, J)
    b = af.clip(a, 0.0, 3.0)
    assert np.allclose(b.jac.toarray(), np.zeros((2, 2)))


def test_clip_adarray_jacobian_outside_bounds():
    # For values outside [min_val, max_val], the Jacobian is zeroed out.
    val = np.array([-1.0, 5.0])
    J = sps.eye(2, format="csr")
    a = AdArray(val, J)
    b = af.clip(a, 0.0, 3.0)
    assert np.allclose(b.jac.toarray(), np.zeros((2, 2)))


def test_clip_adarray_mixed():
    # Mix of clipped (below, above) and interior values.
    val = np.array([-1.0, 1.0, 2.0, 5.0])
    J = sps.csr_matrix(
        np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]], dtype=float)
    )
    a = AdArray(val, J)
    b = af.clip(a, 0.0, 3.0)

    expected_val = np.array([0.0, 1.0, 2.0, 3.0])
    # Interior rows (indices 1 and 2) keep their Jacobian; clipped rows (0, 3) are zero.
    expected_jac = np.array(
        [[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 0]], dtype=float
    )
    assert np.allclose(b.val, expected_val)
    assert np.allclose(b.jac.toarray(), expected_jac)


def test_clip_adarray_dense_jacobian():
    # Verify correctness with a non-diagonal (dense) Jacobian.
    val = np.array([0.5, 2.5])
    J = sps.csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    a = AdArray(val, J)
    b = af.clip(a, 0.0, 3.0)
    assert np.allclose(b.val, np.array([0.5, 2.5]))
    assert np.allclose(b.jac.toarray(), J.toarray())


def test_clip_does_not_mutate_input():
    # Ensure the original AdArray is unchanged after clipping.
    val = np.array([-1.0, 2.0, 5.0])
    J = sps.eye(3, format="csr")
    a = AdArray(val.copy(), J.copy())
    # Perform clipping, but ignore the result to check that 'a' is unchanged. The clip
    # affects the values -1 and 5.
    _ = af.clip(a, 0.0, 3.0)
    assert np.allclose(a.val, np.array([-1.0, 2.0, 5.0]))
    assert np.allclose(a.jac.toarray(), sps.eye(3).toarray())


@pytest.mark.parametrize("representation", ["full_ad", "diagonal_ad"])
def test_clip_adarray_representations(representation: str):
    """Mix of clipped (below, above) and interior values, for a full AdArray and a
    DiagonalAdArray argument (mirrors test_clip_adarray_mixed above)."""
    val = np.array([-1.0, 1.0, 5.0])
    var = _make_ad(val, representation)

    result = af.clip(var, 0.0, 3.0)

    expected_val = np.array([0.0, 1.0, 3.0])
    # Interior entry (index 1) keeps its Jacobian; clipped entries (0, 2) are zeroed.
    mask = np.array([0.0, 1.0, 0.0])
    if representation == "full_ad":
        expected_jac = np.diag(mask) @ _FULL_CHAIN_JAC
    else:
        expected_jac = np.diag(mask * _DIAGONAL_CHAIN_JAC)

    assert np.allclose(result.val, expected_val)
    assert np.allclose(_dense_jac(result), expected_jac)
    if representation == "diagonal_ad":
        assert result.is_diagonal
