"""Dedicated tests for arithmetic combinations of AD-capable objects.

This module extracts and restructures the arithmetic-operator tests from
``test_operators.py``.
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.ad.equation_system import GridEntity

AdType = Union[float, np.ndarray, sps.spmatrix, pp.ad.AdArray]


@pytest.fixture
def arithmetic_mdg() -> pp.MixedDimensionalGrid:
    g = pp.CartGrid([3, 1])
    mdg = pp.MixedDimensionalGrid()
    mdg.add_subdomains([g])
    return mdg


def _get_scalar(wrapped: bool) -> float | pp.ad.Scalar:
    scalar = 2.0
    if wrapped:
        return pp.ad.Scalar(scalar)
    return scalar


def _get_dense_array(wrapped: bool, mdg) -> np.ndarray | pp.ad.DenseArray:
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
    wrapped: bool,
    mdg,
    equation_system: pp.ad.EquationSystem | None = None,
    variable_name: str = "foo_ad",
) -> pp.ad.AdArray | tuple[pp.ad.AdArray, pp.ad.EquationSystem]:
    variable_val = np.ones(3)
    jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    expression_val = jac @ variable_val

    if wrapped:
        eq = (
            equation_system
            if equation_system is not None
            else pp.ad.EquationSystem(mdg)
        )
        eq.create_variables(
            variable_name, subdomains=mdg.subdomains(), dof_info={GridEntity.cells: 1}
        )
        var = eq.variables[-1]
        d = mdg.subdomain_data(mdg.subdomains()[0])

        pp.set_solution_values(
            name=variable_name, values=variable_val, data=d, time_step_index=0
        )
        pp.set_solution_values(
            name=variable_name, values=variable_val, data=d, iterate_index=0
        )
        space = pp.ad.OperatorSpace.from_domains(
            mdg.subdomains(), dof_info={GridEntity.cells: 1}
        )
        mat = pp.ad.SparseArray(jac, source=space, target=space)

        return mat @ var, eq

    return pp.ad.AdArray(expression_val, jac)


def _get_diag_array(
    wrapped: bool,
    mdg,
    equation_system: pp.ad.EquationSystem | None = None,
    variable_name: str = "foo_diag",
) -> pp.ad.AdArray | tuple[pp.ad.AdArray, pp.ad.EquationSystem]:
    variable_val = np.array([1, 2, 3])
    jac = np.array([6, 7.5, 8])

    if wrapped:
        eq = (
            equation_system
            if equation_system is not None
            else pp.ad.EquationSystem(mdg)
        )
        eq.create_variables(
            variable_name, subdomains=mdg.subdomains(), dof_info={GridEntity.cells: 1}
        )
        var = eq.variables[-1]
        d = mdg.subdomain_data(mdg.subdomains()[0])
        pp.set_solution_values(
            name=variable_name, values=variable_val, data=d, time_step_index=0
        )
        pp.set_solution_values(
            name=variable_name, values=variable_val, data=d, iterate_index=0
        )

        space = pp.ad.OperatorSpace.from_domains(
            mdg.subdomains(), dof_info={GridEntity.cells: 1}
        )
        vec = pp.ad.DenseArray(jac, source=space, target=space)

        return vec * var, eq

    return pp.ad.DiagonalAdArray(
        jac * variable_val,
        jac,
        row_indices=np.arange(variable_val.size),
        col_indices=[np.arange(variable_val.size)],
        num_derivatives=3,
    )


def _jac_to_dense(ad: pp.ad.AdArray) -> np.ndarray:
    if ad._is_diagonal:
        return ad.to_full().jac.toarray()
    return ad.jac.toarray()


def _expected_ad_ad(
    var_1: pp.ad.AdArray, var_2: pp.ad.AdArray, op: Literal["+", "-", "*", "/", "**"]
) -> pp.ad.AdArray:
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
    else:
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
    return pp.ad.AdArray(val, sps.csr_matrix(jac))


def _expected_value(
    var_1: AdType, var_2: AdType, op: Literal["+", "-", "*", "/", "**", "@"]
) -> bool | float | np.ndarray | sps.spmatrix | pp.ad.AdArray:
    def create_adarray(val, jac):
        if isinstance(jac, np.ndarray):
            jac = sps.dia_matrix((jac, [0]), shape=(jac.size, jac.size))
        return pp.ad.AdArray(val, jac)

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
            return False

        try:
            val = eval(f"var_1 {op} var_2")
            assert op in ["*", "/"]
            return val
        except (ValueError, NotImplementedError):
            return False
    elif isinstance(var_1, (sps.spmatrix, sps.sparray)) and isinstance(
        var_2, np.ndarray
    ):
        if op == "**":
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

    elif isinstance(var_1, pp.ad.AdArray) and isinstance(var_2, float):
        if op == "@":
            return False
        if op == "+":
            val = np.array([8, 17, 26])
            if var_1._is_diagonal:
                jac = np.array([6, 7.5, 8])
            else:
                jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        elif op == "-":
            val = np.array([4, 13, 22])
            if var_1._is_diagonal:
                jac = np.array([6, 7.5, 8])
            else:
                jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        elif op == "*":
            val = np.array([12, 30, 48])
            if var_1._is_diagonal:
                jac = np.array([12, 15, 16])
            else:
                jac = sps.csr_matrix(np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]]))
        elif op == "/":
            val = np.array([6 / 2, 15 / 2, 24 / 2])
            if var_1._is_diagonal:
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
        elif op == "**":
            val = np.array([6**2, 15**2, 24**2])
            if var_1._is_diagonal:
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
        return create_adarray(val, jac)

    elif isinstance(var_1, float) and isinstance(var_2, pp.ad.AdArray):
        if op == "@":
            return False
        if op == "+":
            val = np.array([8, 17, 26])
            if var_2._is_diagonal:
                jac = np.array([6, 7.5, 8])
            else:
                jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        elif op == "-":
            val = np.array([-4, -13, -22])
            if var_2._is_diagonal:
                jac = -np.array([6, 7.5, 8])
            else:
                jac = sps.csr_matrix(
                    np.array([[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]])
                )
        elif op == "*":
            val = np.array([12, 30, 48])
            if var_2._is_diagonal:
                jac = np.array([12, 15, 16])
            else:
                jac = sps.csr_matrix(np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]]))
        elif op == "/":
            val = np.array([2 / 6, 2 / 15, 2 / 24])
            if var_2._is_diagonal:
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
        elif op == "**":
            val = np.array([2**6, 2**15, 2**24])
            if var_2._is_diagonal:
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
        return create_adarray(val, jac)

    elif isinstance(var_1, pp.ad.AdArray) and isinstance(var_2, np.ndarray):
        if op == "@":
            return False

        if op == "+":
            val = np.array([7, 17, 27])
            if var_1._is_diagonal:
                jac = np.array([6, 7.5, 8])
            else:
                jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        elif op == "-":
            val = np.array([5, 13, 21])
            if var_1._is_diagonal:
                jac = np.array([6, 7.5, 8])
            else:
                jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        elif op == "*":
            val = np.array([6, 30, 72])
            if var_1._is_diagonal:
                jac = np.array([6 * 1, 7.5 * 2, 8 * 3])
            else:
                jac = sps.csr_matrix(np.array([[1, 2, 3], [8, 10, 12], [21, 24, 27]]))
        elif op == "/":
            val = np.array([6 / 1, 15 / 2, 24 / 3])
            if var_1._is_diagonal:
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
        elif op == "**":
            val = np.array([6, 15**2, 24**3])
            if var_1._is_diagonal:
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
        return create_adarray(val, jac)

    elif isinstance(var_1, np.ndarray) and isinstance(var_2, pp.ad.AdArray):
        if op == "@":
            return False

        if op == "+":
            val = np.array([7, 17, 27])
            if var_2._is_diagonal:
                jac = np.array([6, 7.5, 8])
            else:
                jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        elif op == "-":
            val = np.array([-5, -13, -21])
            if var_2._is_diagonal:
                jac = -np.array([6, 7.5, 8])
            else:
                jac = sps.csr_matrix(
                    np.array([[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]])
                )
        elif op == "*":
            val = np.array([6, 30, 72])
            if var_2._is_diagonal:
                jac = np.array([6 * 1, 7.5 * 2, 8 * 3])
            else:
                jac = sps.csr_matrix(np.array([[1, 2, 3], [8, 10, 12], [21, 24, 27]]))
        elif op == "/":
            val = np.array([1 / 6, 2 / 15, 3 / 24])
            if var_2._is_diagonal:
                jac = -1 / var_2.val**2 * var_2.jac * np.array([1, 2, 3])
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
        elif op == "**":
            val = np.array([1, 2**15, 3**24])
            if var_2._is_diagonal:
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
        return create_adarray(val, jac)

    elif isinstance(var_1, pp.ad.AdArray) and isinstance(
        var_2, (sps.spmatrix, sps.sparray)
    ):
        return False
    elif isinstance(var_1, sps.spmatrix) and isinstance(var_2, pp.ad.AdArray):
        if op == "@":
            val = var_1 * var_2.val
            if var_2._is_diagonal:
                jac = var_1 @ sps.diags(var_2.jac.ravel())
            else:
                jac = var_1 * var_2.jac
            return pp.ad.AdArray(val, jac)
        return False
    elif isinstance(var_1, sps.sparray) and isinstance(var_2, pp.ad.AdArray):
        if op == "@":
            val = var_1 @ var_2.val
            if var_2._is_diagonal:
                jac = var_1 @ sps.diags(var_2.jac.ravel())
            else:
                jac = var_1 @ var_2.jac
            return pp.ad.AdArray(val, jac)
        return False

    elif isinstance(var_1, pp.ad.DiagonalAdArray) and isinstance(
        var_2, pp.ad.DiagonalAdArray
    ):
        if op == "@":
            return False

        if op == "+":
            val = var_1.val + var_2.val
            jac = var_1.jac + var_2.jac
        elif op == "-":
            val = var_1.val - var_2.val
            jac = var_1.jac - var_2.jac
        elif op == "*":
            val = var_1.val * var_2.val
            jac = var_1.val * var_2.jac + var_2.val * var_1.jac
        elif op == "/":
            val = var_1.val / var_2.val
            jac = var_1.jac / var_2.val - var_1.val * var_2.jac / var_2.val**2
        elif op == "**":
            val = var_1.val**var_2.val
            jac = (
                var_2.val * var_1.val ** (var_2.val - 1.0) * var_1.jac
                + np.log(var_1.val) * (var_1.val**var_2.val) * var_2.jac
            )
        return pp.ad.DiagonalAdArray(
            val,
            jac,
            row_indices=np.arange(3),
            col_indices=[np.arange(3)],
            num_derivatives=3,
        )

    elif isinstance(var_1, pp.ad.AdArray) and isinstance(var_2, pp.ad.AdArray):
        if op == "@":
            return False
        return _expected_ad_ad(var_1, var_2, op)

    raise ValueError(f"Unknown classes: {type(var_1)}, {type(var_2)}.")


def _compare(actual, expected) -> None:
    if not (isinstance(actual, pp.ad.AdArray) and isinstance(expected, pp.ad.AdArray)):
        assert isinstance(actual, expected.__class__)
    if isinstance(actual, float):
        assert np.isclose(actual, expected)
    elif isinstance(actual, np.ndarray):
        assert np.allclose(actual, expected)
    elif isinstance(actual, (sps.spmatrix, sps.sparray)):
        assert np.allclose(actual.toarray(), expected.toarray())
    elif isinstance(actual, pp.ad.AdArray):
        assert np.allclose(actual.val, expected.val)
        jac = _jac_to_dense(actual)
        expected_jac = _jac_to_dense(expected)
        assert np.allclose(jac, expected_jac)
    else:
        raise ValueError(f"Unknown type: {type(actual)}")


def _expand_adarray_derivatives(
    ad_obj: pp.ad.AdArray, slot: int, n_slots: int
) -> pp.ad.AdArray:
    """Embed derivatives of one operand into its own derivative block.

    This mirrors wrapped-mode evaluation when ad/diag operands are created as
    independent variables in the same EquationSystem.
    """

    if n_slots == 1:
        return ad_obj

    jac = _jac_to_dense(ad_obj)
    expanded = np.zeros((jac.shape[0], jac.shape[1] * n_slots), dtype=float)
    start = slot * jac.shape[1]
    expanded[:, start : start + jac.shape[1]] = jac
    return pp.ad.AdArray(ad_obj.val, sps.csr_matrix(expanded))


def _expected_wrapped_value(
    var_1: AdType, var_2: AdType, op: Literal["+", "-", "*", "/", "**", "@"]
) -> bool | float | np.ndarray | sps.spmatrix | pp.ad.AdArray:
    n_ad_operands = int(isinstance(var_1, pp.ad.AdArray)) + int(
        isinstance(var_2, pp.ad.AdArray)
    )
    if isinstance(var_1, pp.ad.AdArray):
        slot_1 = 0
        var_1 = _expand_adarray_derivatives(var_1, slot_1, max(1, n_ad_operands))
    if isinstance(var_2, pp.ad.AdArray):
        slot_2 = 1 if isinstance(var_1, pp.ad.AdArray) and n_ad_operands == 2 else 0
        var_2 = _expand_adarray_derivatives(var_2, slot_2, max(1, n_ad_operands))

    try:
        return eval(f"var_1 {op} var_2")
    except (TypeError, ValueError, NotImplementedError):
        return False


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
    if not wrapped and var_1 != "ad" and var_2 != "ad":
        return

    has_ad_operand = var_1 in ["ad", "diag"] or var_2 in ["ad", "diag"]
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

    v1 = _var_from_string(var_1, wrapped, "foo_v1")
    v2 = _var_from_string(var_2, wrapped, "foo_v2")

    if wrapped and var_1 in ["ad", "diag"]:
        v1, _ = v1
    if wrapped and var_2 in ["ad", "diag"]:
        v2, _ = v2

    v1_as_value = _var_from_string(var_1, False, "foo_v1")
    v2_as_value = _var_from_string(var_2, False, "foo_v2")

    if wrapped and var_1 in ["ad", "diag"] and var_2 in ["ad", "diag"]:
        expected = _expected_wrapped_value(v1_as_value, v2_as_value, op)
    else:
        expected = _expected_value(v1_as_value, v2_as_value, op)

    if wrapped:
        try:
            expression = eval(f"v1 {op} v2")
            state = equation_system.get_variable_values(time_step_index=0)
            ad_base = equation_system._ad_parser._initialize_variables(
                [expression], state, equation_system, derivative=True
            )
            val = equation_system._ad_parser._evaluate_single(
                expression, ad_base, equation_system
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

    _compare(val, expected)

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
