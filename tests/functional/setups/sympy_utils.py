"""Utilities for the sympy-based manufactured solutions in this folder."""

from __future__ import annotations

import functools
from typing import Any, Callable

import sympy as sym


@functools.lru_cache(maxsize=None, typed=True)
def lambdify(args: Any, expr: Any, modules: str) -> Callable:
    """Cached version of :func:`sympy.lambdify`.

    The manufactured-solution setups evaluate the same fixed expressions in every time
    step, and some of them in every Newton iteration. Compiling an expression takes a
    few milliseconds, so the compiled function is reused for equal arguments. Sympy
    expressions hash structurally, thus equal expressions share one compiled function
    regardless of which model instance built them.

    With ``typed=True``, arguments such as ``1``, ``1.0`` and ``sym.Integer(1)`` are
    cached separately, since the compiled functions of constant expressions return
    values of different types.

    Parameters:
        args: Symbol or tuple of symbols, as for :func:`sympy.lambdify`. Must be
            hashable, i.e., a tuple rather than a list.
        expr: Sympy expression or number to compile.
        modules: Module passed on to :func:`sympy.lambdify`.

    Returns:
        The compiled function.

    """
    return sym.lambdify(args, expr, modules)
