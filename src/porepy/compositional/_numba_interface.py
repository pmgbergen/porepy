"""Private module to interface :mod:`numba`.

Commonly used functionality of numba is wrapped to avoid errors when disabling numba.

These included usage of un-compiled functions inside compiled functions when disabling
JIT-compilation for debugging.

The main idea is to use Python code instead of numba code in case it is disabled.
It also serves the purpose of not having to import experimental features anywhere else
(like the objects in :mod:`numba.typed`).

Furthermore, it sets some global flags to use caching, parallelization, or fast-math
instructions in the LLVM-compiler. Those should be modified with much care, having
the latest developments in numba in mind.

"""

from __future__ import annotations

import os
from typing import Any, Callable, TypeAlias, TypeVar

# NOTE import numba.typed like this to avoid importing the spurious py.typed file in the
# typed sub-package, which confuses mypy.
import numba
import numba.typed
from numba.core import sigutils

_IS_JIT_DISABLED: bool = False
"""Environment flag checking whether numba JIT is enabled or not.

Used for typing alternatives in case it is not, such that the code remains functional.

"""

if "NUMBA_DISABLE_JIT" in os.environ:
    if os.environ["NUMBA_DISABLE_JIT"].lower() in ["1", "true"]:
        _IS_JIT_DISABLED = True


typeof: Callable[..., TypeAlias]
"""Type inference function depending on whether numba is enabled or not.

If enabled, uses :obj:`numba.typeof`, otherwise the regular Python type.

"""

cfunc: Callable[..., Callable]
"""C-type decorator for Callables, depending on whether numba is enabled or not.

If enabled, uses :obj:`numba.cfunc`, otherwise the identity.

"""

njit: Callable[..., Callable]
"""JIT-compilation decorator without Python fallback for Callables, depending on whether
numba is enabled or not.

If enabled, uses :obj:`numba.njit`, otherwise the identity.

"""


def _no_compile(*args, **kwargs) -> Callable:
    """Dummy compiler for when numba JTI is disabled.

    Does nothing with the decorated object and returns it as is; ignores all other
    arguments.

    """
    if len(args) > 0:
        arg = args[0]
        if callable(arg) and not sigutils.is_signature(arg):
            return arg
    return lambda x: x


if _IS_JIT_DISABLED:
    typeof = lambda x: type(x)
    cfunc = _no_compile
    njit = _no_compile
else:
    typeof = numba.typeof
    cfunc = numba.cfunc
    njit = numba.njit


_NB_KEY_TYPE = TypeVar("_NB_KEY_TYPE")
"""Numba-type for keys of a numba dictionary."""

_NB_VAL_TYPE = TypeVar("_NB_VAL_TYPE")
"""Numba-type for values of a numba dictionary."""


def get_empty_numba_dict(
    key_type: _NB_KEY_TYPE = numba.types.unicode_type,
    val_type: _NB_VAL_TYPE = numba.types.float64,
    default_pair: tuple[Any, Any] = ("__dummy_key__", 0.0),
) -> dict[_NB_KEY_TYPE, _NB_VAL_TYPE]:
    """Returns an empty numba dictionary to be used for passing parameters to
    numba-compiled functions

    Used for type-inference in numba-compiled functions.

    Note:
        Numba does not allow multiple types in keys or strings (as of now).
        If a parameter is actually an integer, it must be explicitly converted to a
        float before setting it in the dictionary. Wherever used, the type must be
        explicitly converted back.

    Parameters:
        key_type: Numba-type of keys. Defaults to unicode type.
        val_type: Numba-type of values. Defaults to float64.
        default_pair: A default key-value pair to be set. For unknown reasons, numba
            sometimes fails to infer the type of an empty dict if some key-value pair
            is not set. This will probably change as Numba evolves.

    Returns:
        An empty numba dictionary.

    """
    if _IS_JIT_DISABLED:
        d = {}
    else:
        # NOTE: typed.Dict is an experimental feature!
        d = numba.typed.Dict.empty(key_type=key_type, value_type=val_type)
    d[default_pair[0]] = default_pair[1]
    return d


NUMBA_CACHE: bool = True
"""Flag to instruct the numba compiler to cache (!and use cached!) functions.

This might cause some confusion in the developing process due to some lack in numba's
caching functionality.
(Does not recognize changes in nested functions and hence does not trigger
re-compilation).

Use with care.

Note:
    Functions which do not use other numba-compiled functions are cached by default.
    This flag is for those who do use other functions.

See Also:
    https://numba.readthedocs.io/en/stable/user/jit.html#cache

"""

NUMBA_FAST_MATH: bool = False
"""Flag to instruct the numba compiler to use its ``fastmath`` functions.

To be used with care, due to loss in precision.

See Also:
    https://numba.readthedocs.io/en/stable/reference/jit-compilation.html#numba.jit

"""

NUMBA_PARALLEL: bool = True
"""Flag to instruct numba to compile functions in parallel mode, where applicable.

By default, the parallel backend will be used.

Flag is introduced for developing processes when involving other packages supporting
parallelism such as numpy and PETSc.

Affected numba functionality includes:

1. `JIT parallelism
   <https://numba.readthedocs.io/en/stable/user/jit.html#parallel>`_
2. `Numpy universal functions
   <https://numba.readthedocs.io/en/stable/user/vectorize.html>`_

"""
