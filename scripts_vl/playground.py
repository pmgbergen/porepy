import numba
import numpy as np
from typing import Callable, Optional

from numba.core import types
from numba.typed import Dict

@numba.njit('float64[:](float64[:],optional(float64))')
def g(x, e = 1e-8):
    a = np.array([False, False, False], dtype=np.bool_)
    s = a.sum()
    if s == 1:
        print("check")
    else:
        print("fail")
    return x + e

print(g(np.ones(3), 2.))
print(g(np.ones(3), 2.))

# @numba.njit('float64[:](float64[:],DictType(unicode_type, float64))')
@numba.njit
def f(x: np.ndarray, p: Optional[dict[str, float]] = Dict.empty(
    key_type=types.unicode_type,
    value_type=types.float64,
)) -> np.ndarray:
    x_ = x.copy()
    if p is not None:
        for k, v in p.items():
            x_ = x_ + float(v)

    return x_

x = np.array([1,2,3], dtype=np.float64)
p = Dict.empty(
    key_type=types.unicode_type,
    value_type=types.float64,
)
p['a'] = 4.
p['b'] = 5.

print(f(x, p))
p['c'] = 6.
p['d'] = 7.
# print(f(x))
print(f(x, p))
print(f(x, {1: 1, 2: 2.}))

class Foo:

    def __init__(self) -> None:
        
        self.compiled: dict[str, Callable] = dict()

        self.compile_bar()

    
    def compile_bar(self):

        @numba.njit("double(double, double)")
        def func1(a, b):
            return a * b

        @numba.njit("double(double, double)")
        def func2(a, b):
            return a / b

        l_c: tuple[Callable, ...] = tuple([func1, func2])
        # l_c: list[Callable]= [func1, func2]

        # l_c = numba.typed.List.empty_list(numba.float64(numba.float64, numba.float64).as_type())
        # l_c.append(func1)
        # l_c.append(func2)

        @numba.njit
        def wrapper(a, b):
            c = 0.
            for func in l_c:
                print(func(a, b))
                c += func(a, b)
            return c

        self.compiled.update({'bar': wrapper})


F = Foo()
print(F.compiled['bar'](1.,2.))
print("here")


### TEST OTHER SOLUTION

# @njit
def ident(x):
    return x


def chain(fs, inner=ident):
    head, tail = fs[-1], fs[:-1]

    # @njit
    def wrap(x):
        return head(inner(x))

    if tail:
        return chain(tail, wrap)
    else:
        return wrap


# @njit
def foo(x):
    return x + 1.2


# @njit
def bar(x):
    return x * 2


# must be used outside of the jit
foobar = chain((foo, bar))
foobarfoo = chain((foo, bar, foo))

# @njit
def test():
    return foobar(3), foobarfoo(3)


print(test())


### WROKING SOLUTION

@njit
def func1(a, b):
    return a * b

@njit
def func2(a, b):
    return a / b


l = (func1, func2)

func_prototype = """
def get_func(i):
"""
for i, func in enumerate(l):
    func_prototype += f"""
    if i == {i}:
        return {func.__name__}
    """
func_prototype += """
    raise ValueError
"""

print(func_prototype)

exec(func_prototype)
pass