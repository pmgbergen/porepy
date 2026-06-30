"""Parity tests for the sparsa AD backend adapter.

These assert that assembling a PorePy ``EquationSystem`` through the external sparsa
engine (``porepy.numerics.ad.sparsa_backend``) reproduces PorePy's own ``assemble``
output ``(A, b)`` to machine precision. Skipped if ``sparsa`` is not installed.

sparsa itself has no PorePy dependency; all coupling lives in the backend adapter.
"""
import importlib.util

import numpy as np
import scipy.sparse as sps
import pytest

import porepy as pp

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sparsa") is None, reason="sparsa not installed"
)


def _build_system():
    sd = pp.CartGrid([4], np.array([1.0]))
    sd.compute_geometry()
    mdg = pp.MixedDimensionalGrid()
    mdg.add_subdomains([sd])
    eqs = pp.ad.EquationSystem(mdg)
    p = eqs.create_variables("p", dof_info={"cells": 1}, subdomains=[sd])
    z = eqs.create_variables("z", dof_info={"cells": 1}, subdomains=[sd])
    eqs.set_variable_values(np.array([1.0, 1.2, 1.4, 1.6]), [p], iterate_index=0, time_step_index=0)
    eqs.set_variable_values(np.array([0.2, 0.4, 0.5, 0.6]), [z], iterate_index=0, time_step_index=0)
    return sd, mdg, eqs, p, z


def _assert_parity(eqs):
    from porepy.numerics.ad import sparsa_backend

    A_pp, b_pp = eqs.assemble()
    A_s, b_s = sparsa_backend.assemble(eqs)
    assert A_pp.shape == A_s.shape
    np.testing.assert_allclose(A_s.toarray(), A_pp.toarray(), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(b_s, b_pp, atol=1e-12, rtol=1e-12)


def test_parity_nonlinear_coupled_with_matmul():
    sd, mdg, eqs, p, z = _build_system()
    M = pp.ad.SparseArray(sps.csr_matrix(np.array(
        [[2.0, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 2]])))
    q = pp.ad.DenseArray(np.linspace(0.1, 0.4, 4))
    eqs.set_equation(M @ (p ** 2) + p * z - q, [sd], {"cells": 1})
    _assert_parity(eqs)


def test_parity_division():
    sd, mdg, eqs, p, z = _build_system()
    eqs.set_equation(p / z + p * z, [sd], {"cells": 1})
    _assert_parity(eqs)


def test_parity_two_equations_row_ordering():
    sd, mdg, eqs, p, z = _build_system()
    M = pp.ad.SparseArray(sps.csr_matrix(np.diag([1.0, 2.0, 3.0, 4.0])))
    e1 = M @ (p ** 2) - z
    e1.set_name("eq_p")
    e2 = z * z - p * z
    e2.set_name("eq_z")
    eqs.set_equation(e1, [sd], {"cells": 1})
    eqs.set_equation(e2, [sd], {"cells": 1})
    _assert_parity(eqs)


def _stack(ad_list):
    A = ad_list[0].jac if len(ad_list) == 1 else sps.vstack([a.jac for a in ad_list], "csr")
    b = ad_list[0].val if len(ad_list) == 1 else np.concatenate([a.val for a in ad_list])
    return A.tocsr(), b


def test_compiled_path_engages_and_matches_native():
    """The Tier-2 compiled kernel must produce the SAME (A, b) as the Python replay, and the
    per-pattern compile cache must actually populate (i.e. the fast path engaged)."""
    from porepy.numerics.ad import sparsa_backend as sb

    sd, mdg, eqs, p, z = _build_system()
    M = pp.ad.SparseArray(sps.csr_matrix(np.diag([1.0, 2.0, 3.0, 4.0])))
    eqs.set_equation(M @ (p ** 2) + p * z, [sd], {"cells": 1})
    eqlist = list(eqs.equations.values())
    state = eqs.get_variable_values(iterate_index=0)

    old = sb._USE_COMPILED
    try:
        sb._USE_COMPILED = False
        A_n, b_n = _stack(sb.SparsaParser(mdg).evaluate(eqlist, eqs, True, state))
        sb._USE_COMPILED = True
        parser = sb.SparsaParser(mdg)
        A_c, b_c = _stack(parser.evaluate(eqlist, eqs, True, state))
    finally:
        sb._USE_COMPILED = old

    np.testing.assert_array_equal(A_c.toarray(), A_n.toarray())
    np.testing.assert_array_equal(b_c, b_n)
    bundle = next(b for b in parser._bundles.values() if b is not None)
    assert len(bundle.compiled_cache) == 1  # the compiled fast path engaged


def test_compiled_handles_changing_matrix_structure():
    """A constant matrix whose nonzero PATTERN changes between assembles (mimicking an upwind
    matrix flipping its upstream cell) must stay bit-exact: the bridge keys a separate
    compiled kernel per structure and reuses it when a structure recurs."""
    from porepy.numerics.ad import sparsa_backend as sb

    sd, mdg, eqs, p, z = _build_system()
    M = pp.ad.SparseArray(sps.csr_matrix(np.diag([1.0, 2.0, 3.0, 4.0])))
    eqs.set_equation(M @ (p ** 2) + p * z, [sd], {"cells": 1})
    eqlist = list(eqs.equations.values())
    state = eqs.get_variable_values(iterate_index=0)

    patterns = [
        sps.csr_matrix(np.diag([1.0, 2.0, 3.0, 4.0])),
        sps.csr_matrix(np.array(  # a different (cyclic-shift) nonzero pattern
            [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]], dtype=float)),
        sps.csr_matrix(np.diag([1.0, 2.0, 3.0, 4.0])),  # first pattern recurs -> cache hit
    ]

    old = sb._USE_COMPILED
    try:
        sb._USE_COMPILED = True
        parser = sb.SparsaParser(mdg)
        for pat in patterns:
            M._mat = pat
            A_c, b_c = _stack(parser.evaluate(eqlist, eqs, True, state))
            sb._USE_COMPILED = False
            A_n, b_n = _stack(sb.SparsaParser(mdg).evaluate(eqlist, eqs, True, state))
            sb._USE_COMPILED = True
            np.testing.assert_array_equal(A_c.toarray(), A_n.toarray())
            np.testing.assert_array_equal(b_c, b_n)
        bundle = next(b for b in parser._bundles.values() if b is not None)
        assert len(bundle.compiled_cache) == 2  # two distinct patterns; the 3rd was reused
    finally:
        sb._USE_COMPILED = old
