"""Tests for :meth:`porepy.numerics.ad.operators.Operator.inspect`.

``inspect`` walks the AD operator graph and reports statistics used to spot
optimization opportunities: the number of unique nodes (the graph is a DAG, so shared
subtrees are counted once), the maximum depth, a per-type node breakdown, the set of
variable names encountered, and the number of fully-constant subtrees (constant-folding
candidates).
"""

from __future__ import annotations

import numpy as np
import pytest

import porepy as pp


@pytest.fixture()
def equation_system():
    mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
        "cartesian",
        {"cell_size": 0.5},
        fracture_indices=[1],
    )
    es = pp.ad.EquationSystem(mdg)
    es.create_variables("foo", {"cells": 1}, mdg.subdomains())
    es.set_variable_values(
        np.ones(es.num_dofs()), iterate_index=0, time_step_index=0
    )
    return es


def test_inspect_returns_expected_statistics(equation_system):
    """Basic stats: node count, depth, variable detection, type breakdown."""
    mdg = equation_system.mdg
    x = equation_system.md_variable("foo", mdg.subdomains())
    expr = x * pp.ad.Scalar(2.0) + pp.ad.Scalar(3.0)

    stats = expr.inspect(verbose=False)

    # (x * 2) + 3 -> add, mul, x, Scalar(2), Scalar(3) = 5 unique nodes.
    assert stats["total_nodes"] == 5
    assert stats["max_depth"] == 2
    assert stats["variables"] == {"foo"}
    assert stats["node_types"]["Scalar"] == 2
    # No all-constant operation node here (every operation touches the variable).
    assert stats["constant_subtrees"] == 0


def test_inspect_is_dag_aware(equation_system):
    """A subtree shared by several parents is counted once, not per reference."""
    mdg = equation_system.mdg
    x = equation_system.md_variable("foo", mdg.subdomains())
    shared = x * pp.ad.Scalar(2.0)
    # 'shared' is referenced twice; the DAG walk must not double-count it.
    expr = shared + shared

    stats = expr.inspect(verbose=False)
    # add, mul, x, Scalar(2) = 4 unique nodes (the second 'shared' reference is free).
    assert stats["total_nodes"] == 4
    assert stats["node_types"]["Scalar"] == 1


def test_inspect_counts_constant_subtrees():
    """An operation whose children are all constants is a constant-folding candidate."""
    const_product = pp.ad.Scalar(2.0) * pp.ad.Scalar(3.0)
    expr = const_product + pp.ad.Scalar(1.0)

    stats = expr.inspect(verbose=False)
    # Both the inner 'mul' and the outer 'add' have only constant children.
    assert stats["constant_subtrees"] == 2
    assert stats["variables"] == set()


def test_inspect_verbose_prints_summary(equation_system, capsys):
    """verbose=True prints a human-readable summary to stdout."""
    mdg = equation_system.mdg
    x = equation_system.md_variable("foo", mdg.subdomains())
    (x + pp.ad.Scalar(1.0)).inspect(verbose=True)

    out = capsys.readouterr().out
    assert "AD Graph Inspection" in out
    assert "Total Nodes:" in out
    assert "foo" in out
