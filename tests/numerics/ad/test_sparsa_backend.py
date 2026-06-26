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
