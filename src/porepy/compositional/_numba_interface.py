"""Private module to interface :mod:`numba`.

Commonly used functionality of numba is wrapped to avoid errors when disabling numba.

These included usage of un-compiled functions inside compiled functions when disabling
JIT-compilation for debugging.

It also sets some global flags to use caching, parallelization, or fast-math
instructions in the LLVM-compiler.

"""

from __future__ import annotations

import os
from typing import Callable, TypeAlias

import numba as nb
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
    typeof = nb.typeof
    cfunc = nb.cfunc
    njit = nb.njit


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
