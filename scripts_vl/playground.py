import numba
from typing import Callable

class Foo:

    def __init__(self) -> None:
        
        self.compiled: dict[str, Callable] = dict()

        self.compile_bar()

    
    def compile_bar(self):

        @numba.njit
        def func1(a, b):
            return a * b

        @numba.njit
        def func2(a, b):
            return a / b

        l_c: list[Callable] = [func1, func2]

        # l_c = numba.typed.List.empty_list(numba.float64(numba.float64, numba.float64).as_type())
        # l_c.append(func1)
        # l_c.append(func2)

        @numba.njit
        def wrapper(a, b):
            for func in l_c:
                print(func(a, b))

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