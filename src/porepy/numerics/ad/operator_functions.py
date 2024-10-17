"""This module contains callable operators representing functions to be called with
other operators as input arguments.

Operator functions represent a numerical function in the AD framework, with its
arguments represented by other Ad operators.
The actual numerical value is obtained during
:meth:`~porepy.numerics.ad.operators.Operator.value` or
:meth:`~porepy.numerics.ad.operators.Operator.value_and_jacobian`.

Contains also a decorator class for callables, which transforms them automatically in a
specified operator function type.

"""

from __future__ import annotations

import abc
from functools import partial
from typing import Callable, Optional, Sequence, Type

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.ad.forward_mode import AdArray

from .functions import FloatType
from .operators import Operator

__all__ = [
    "AbstractFunction",
    "DiagonalJacobianFunction",
    "Function",
    "InterpolatedFunction",
    "ADmethod",
]


def _raise_no_arithmetics_with_functions_error():
    raise TypeError("Operator functions must be called before applying any operation.")


class AbstractFunction(Operator):
    """Abstract class for all operator functions, i.e. functions called with some other
    AD operators.

    Implements the call with Ad operators, creating an operator with children
    and its operation set to
    :attr:`~porepy.numerics.ad.operators.Operator.Operations.evaluate`.

    Provides abstract methods to implement the computation of value and Jacobian of the
    function independently.

    The abstract function itself has no arithmetic overloads, since its meaning
    is given only by calling it using other operators. Type errors are raised if the
    user attempts to use any overload implemented in the base class.

    Parameters:
        name: Name of this instance as an AD operator.

    """

    def _key(self) -> str:
        raise NotImplementedError("Will be covered later.")

    def __init__(
        self,
        name: Optional[str] = None,
        domains: Optional[pp.GridLikeSequence] = None,
        operation: Optional[Operator.Operations] = None,
        children: Optional[Sequence[Operator]] = None,
        **kwargs,  # Left for inheritance for more complex functions
    ) -> None:
        # NOTE Constructor is overwritten to have a consistent signature
        # But the operation is always overwritten to point to evaluate.
        # Done for reasons of multiple inheritance.
        super().__init__(
            name=name,
            domains=domains,
            operation=pp.ad.Operator.Operations.evaluate,
            children=children,
        )

    def __call__(self, *args: pp.ad.Operator) -> pp.ad.Operator:
        """Renders this function operator callable, fulfilling its notion as 'function'.

        Parameters:
            *args: AD operators representing the arguments of the function represented
                by this instance.

        Returns:
            Operator with assigned operation ``evaluate``.

            It's children are given by this instance, and ``*args``. This is required
            to make the numerical function available during parsing (see :meth:`parse`).

        """
        assert (
            len(args) > 0
        ), "Operator functions must be called with at least 1 argument."

        op = Operator(
            name=f"{self.name}{[a.name for a in args]}",
            # domains=self.domains,
            operation=pp.ad.Operator.Operations.evaluate,
            children=args,
        )
        # Assigning the functional representation by the implementation of this instance
        op.func = self.func  # type: ignore
        return op

    def __repr__(self) -> str:
        """String representation of this operator function.
        Uses currently only the given name."""

        s = f"AD Operator function '{self._name}'"
        return s

    def __neg__(self) -> Operator:
        return _raise_no_arithmetics_with_functions_error()

    def __add__(self, other: Operator) -> Operator:
        return _raise_no_arithmetics_with_functions_error()

    def __radd__(self, other: Operator) -> Operator:
        return _raise_no_arithmetics_with_functions_error()

    def __sub__(self, other: Operator) -> Operator:
        return _raise_no_arithmetics_with_functions_error()

    def __rsub__(self, other: Operator) -> Operator:
        return _raise_no_arithmetics_with_functions_error()

    def __mul__(self, other: Operator) -> Operator:
        return _raise_no_arithmetics_with_functions_error()

    def __rmul__(self, other: Operator) -> Operator:
        return _raise_no_arithmetics_with_functions_error()

    def __truediv__(self, other: Operator) -> Operator:
        return _raise_no_arithmetics_with_functions_error()

    def __rtruediv__(self, other: Operator) -> Operator:
        return _raise_no_arithmetics_with_functions_error()

    def __pow__(self, other: Operator) -> Operator:
        return _raise_no_arithmetics_with_functions_error()

    def __rpow__(self, other: Operator) -> Operator:
        return _raise_no_arithmetics_with_functions_error()

    def __matmul__(self, other: Operator) -> Operator:
        return _raise_no_arithmetics_with_functions_error()

    def __rmatmul__(self, other):
        return _raise_no_arithmetics_with_functions_error()

    def parse(self, mdg: pp.MixedDimensionalGrid):
        """Operator functions return themselves to give the recursion in
        :class:`~porepy.numerics.ad.operators.Operator` access to the underlying
        :meth:`func`."""
        return self

    def func(self, *args: FloatType) -> float | np.ndarray | AdArray:
        """The underlying numerical function which is represented by this operator
        function.

        Called during parsing with the numerical representation of operator arguments
        and returning the numerical value and derivative of this operator instance.

        The numerical function calls in any case :meth:`get_values` with ``*args``.
        If ``*args`` contains an Ad array, it calls also :meth:`get_jacobian`.

        Parameters:
            *args: Numerical representation of the operators with which this instance
                was called. The arguments will be in the same order as the operators
                passed to the call to this instance.

        Returns:
            If ``*args`` contains only numpy arrays, it returns the result of
            :meth:`get_values`.
            If it contains an Ad array, it combines the results of :meth:`get_values`
            and :meth:`get_jacobian` in an Ad array and returns it.

        """

        values = self.get_values(*args)

        if any(isinstance(a, AdArray) for a in args):
            jac = self.get_jacobian(*args)
            if isinstance(values, float):
                assert jac.shape[0] == 1, "Inconsistent Jacobian of scalar function."
                values = np.array([values])
            return AdArray(values, self.get_jacobian(*args))
        else:
            return values

    @abc.abstractmethod
    def get_values(self, *args: float | np.ndarray | AdArray) -> float | np.ndarray:
        """Abstract method for evaluating the callable passed at instantiation.

        The returned numpy array will be set as
        :attr:`~porepy.numerics.ad.forward_mode.AdArray.val` in for cases when any
        child is parsed as an Ad array.
        Otherwise the value returned here will be returned directly as the numerical
        representation of this instance.

        This method is called in :meth:`func`.

        Parameters:
            *args: Numerical representation of the operators with which this instance
                was called. The arguments will be in the same order as the operators
                passed to the call to this instance.

        Returns:
            Function values in numerical format.

        """
        pass

    @abc.abstractmethod
    def get_jacobian(self, *args: float | np.ndarray | AdArray) -> sps.spmatrix:
        """Abstract method for evaluating the Jacobian of the function represented
        by this instance.

        The returned matrix will be set as
        :attr:`~porepy.numerics.ad.forward_mode.AdArray.jac` in for cases when any
        child is parsed as an Ad array.

        This method is called in :meth:`func` if any argument is an Ad array.

        Note:
            The necessary dimensions for the jacobian can be extracted from the
            dimensions of the Jacobians of passed Ad arrays in ``*args``.

        Parameters:
            *args: Numerical representation of the operators with which this instance
                was called. The arguments will be in the same order as the operators
                passed to the call to this instance.

        Returns:
            Function derivatives in numerical format.

        """
        pass


class DiagonalJacobianFunction(AbstractFunction):
    """Partially abstract operator function, which approximates the Jacobian of the
    function using identities and scalar multipliers per dependency.

    Can be used to for functions with approximated derivatives.

    Parameters:
        multipliers: Scalar multipliers for the identity blocks in the Jacobian,
            per function argument. The order in ``multipliers`` is expected to match
            the order of AD operators passed to the call of this instance.

    """

    def __init__(
        self,
        multipliers: float | list[float],
        name: str,
    ):
        super().__init__(name=name)
        # check and format input for further use
        if isinstance(multipliers, list):
            self._multipliers = [float(val) for val in multipliers]
        else:
            self._multipliers = [float(multipliers)]

    def get_jacobian(self, *args: float | np.ndarray | AdArray) -> sps.spmatrix:
        """The approximate Jacobian consists of identity blocks times scalar multiplier
        per every function dependency."""
        jacs = [
            arg.jac * m
            for arg, m in zip(args, self._multipliers)
            if isinstance(arg, AdArray)
        ]
        return sum(jacs).tocsr()


class Function(AbstractFunction):
    """Ad representation of an analytically given function, which can handle both
    numpy arrays and Ad arrays.

    Here the values **and** the Jacobian are obtained exactly by the AD framework.

    The intended use is as a wrapper for callables, which can handle numpy and Ad
    arrays. E.g., exponential or logarithmic functions, which cannot be expressed
    with arithmetic overloads of Ad operators.

    Note:
        This is a special case where the abstract methods for getting values and the
        Jacobian are formally implemented but never used by the AD framework.

        :meth:`func` is overwritten to use the ``func`` passed at instantiation.

    Paramters:
        func: A callable returning a numpy array for numpy array arguments, and an
            Ad array for arguments containing Ad arrays.

    """

    def __init__(self, func: Callable[..., FloatType], name: str) -> None:
        super().__init__(name=name)

        self._func: Callable[..., float | np.ndarray | AdArray] = func
        """Reference to the callable passed at instantiation."""

    def func(self, *args: FloatType) -> float | np.ndarray | AdArray:
        """Overwrites the parent method to call the numerical function passed at
        instantiation."""
        return self._func(*args)

    def get_values(self, *args: float | np.ndarray | AdArray) -> float | np.ndarray:
        result = self._func(*args)
        return result.val if isinstance(result, AdArray) else result

    def get_jacobian(self, *args: float | np.ndarray | AdArray) -> sps.spmatrix:
        assert any(
            isinstance(a, AdArray) for a in args
        ), "No Ad arrays passed as arguments."
        result = self._func(*args)
        assert isinstance(result, AdArray)
        return result.jac


class InterpolatedFunction(AbstractFunction):
    """Represents the passed function as an interpolation of chosen order on a
    Cartesian, uniform grid.

    The image of the function is expected to be of dimension 1, while the domain can be
    multidimensional.

    Note:
        All vector-valued ndarray arguments are assumed to be column vectors.
        Each row-entry represents a value for an argument of ``func`` in
        respective order.

    Important:
        The construction of the Jacobian assumes that the arguments/dependencies of the
        interpolated function are independent variables (their jacobian has only a
        single identity block). The correct behavior of the interpolation in other cases
        is not guaranteed due to how derivative values are stored in the sparse matrix
        of derivatives.

    Parameters:
        min_val: lower bounds for the domain of ``func``.
        max_val: upper bound for the domain.
        npt: number of interpolation points per dimension of the domain.
        order: Order of interpolation. Supports currently only linear order.
        preval (optional): If True, pre-evaluates the values of the function at
            the points of interpolation and stores them.
            If False, evaluates them if necessary.
            Influences the runtime.
            Defaults to False.

    """

    def __init__(
        self,
        func: Callable,
        name: str,
        min_val: np.ndarray,
        max_val: np.ndarray,
        npt: np.ndarray,
        order: int = 1,
        preval: bool = False,
    ):
        super().__init__(name=name)

        ### PUBLIC
        self.order: int = order

        ### PRIVATE
        self._prevaluated: bool = preval
        self._table: pp.InterpolationTable

        if self.order == 1:
            if self._prevaluated:
                self._table = pp.InterpolationTable(min_val, max_val, npt, func)
            else:
                # Find a grid resolution from the provided minimum and maximum values.
                # TODO: This will get an overhaul once we start using the adaptive
                # interpolation tables in actual computations.
                dx = (max_val - min_val) / npt
                self._table = pp.AdaptiveInterpolationTable(
                    dx, base_point=min_val, function=func, dim=1
                )
        else:
            raise NotImplementedError(
                f"Interpolation of order {self.order} not implemented."
            )

    def get_values(self, *args: float | np.ndarray | AdArray) -> np.ndarray:
        # stacking argument values vertically for interpolation
        args_: list[float | np.ndarray] = []
        for a in args:
            if isinstance(a, AdArray):
                args_.append(a.val)
            else:
                args_.append(a)
        X: np.ndarray = np.vstack(args_)
        return self._table.interpolate(X)

    def get_jacobian(self, *args: float | np.ndarray | AdArray) -> sps.spmatrix:
        # get points at which to evaluate the differentiation
        X = np.vstack([x.val if isinstance(x, AdArray) else x for x in args])
        # allocate zero matrix for Jacobian with correct dimensions and in CSR format
        jacs = []

        for axis, arg in enumerate(args):
            if isinstance(arg, AdArray):
                # The trivial Jacobian of one argument gives us the correct position for
                # the entries as ones
                partial_jac = arg.jac
                # replace the ones with actual values
                # Since csr, we can simply replace the data array with the values of the
                # derivative
                partial_jac.data = self._table.gradient(X, axis)[0]
                jacs.append(partial_jac)

        return sum(jacs).tocsr()


### FUNCTION DECORATOR


class ADmethod:
    """(Decorator) Class for numerical functions, to wrap them into operator functions.

    The designated operator function must be able to take a keyword argument ``func``.

    The decorated, numerical function is expected to be able to handle numerical
    arguments including Ad arrays.

    Examples:
        .. code:: python

            import porepy as pp

            # decorating class methods
            class IdealGas:

                @ADmethod(ad_operator=pp.ad.Function,
                        operator_args={'name'='density'})
                def density(self, p: float, T: float) -> float:
                    return p/T

            # decorating function with default operator function (pp.ad.Function)
            @ADmethod
            def dummy_rel_perm(s):
                return s**2

        With above code, the decorated functions can be called with AD operators
        representing the function arguments.

    Note:
        If used as decorator WITHOUT explicit instantiation, the instantiation will be
        done implicitly with default arguments (that's how Python decorators work).

    Parameters:
        func: decorated function object
        ad_function_type: type reference to an AD operator function to be instantiated.
            When instantiated, that instance will effectively replace ``func``.
        operator_kwargs: keyword arguments to be passed when instantiating an operator
            of type ``ad_function_type``.

    """

    def __init__(
        self,
        func: Optional[Callable] = None,
        ad_function_type: Type[AbstractFunction] = Function,
        operator_kwargs: Optional[dict] = None,
    ) -> None:
        if operator_kwargs is None:
            operator_kwargs = {"name": "unnamed_function"}

        assert "name" in operator_kwargs, "Operator functions must be named."
        # reference to decorated function object
        self._func = func
        # mark if decoration without explicit call to constructor
        self._explicit_init = func is None
        # reference to instance, to which the decorated bound method belongs, if any
        # if this remains None, then an unbound method was decorated
        self._bound_to: Optional[object] = None
        # reference to operator type which should wrap the decorated method
        self._ad_func_type = ad_function_type
        # keyword arguments for call to constructor of operator type
        self._op_kwargs = operator_kwargs

    def __call__(self, *args, **kwargs) -> ADmethod | pp.ad.Operator:
        """Wrapper factory.
        The decorated object is wrapped and/or evaluated here.

        Dependent on whether the decorated function is a method belonging to a class,
        or an unbound function, the wrapper will have a different signature.

        If bound to a class instance, the wrapper will include a partial function, where
        the instance of the class was already passed beforehand.

        Note:
            If the decorator was explicitly instantiated during decoration,
            that instance will effectively be replaced by another decorator instance
            created here in the call.
            It is expected that the the call will follow the instantiation immediately
            when used as a decorator, hence properly dereferencing the original
            instance. If used differently or if another reference is saved between
            explicit instantiation and call, this is a potential memory leak.

        """
        # If decorated without explicit init, the function is passed during a call to
        # the decorator as first argument.
        if self._func is None:
            self._func = args[0]

        # If an explicit init was made, mimic a non-explicit init to get an object with
        # the descriptor protocol.
        if self._explicit_init:
            return ADmethod(
                func=self._func,
                ad_function_type=self._ad_func_type,
                operator_kwargs=self._op_kwargs,
            )

        # Without an explicit init, the first decorator itself replaces the decorated
        # function. This results in a call to ADmethod.__call__ instead of
        # a call to the decorated function

        # when calling the decorator, distinguish between bound method call
        # ('args' contains 'self' of the decorated instance) and an unbound function
        # call (whatever 'args' and 'kwargs' contain, we pass it to the wrapper)
        if self._bound_to is None:
            wrapped_function = self.ad_wrapper(*args, **kwargs)
        elif self._bound_to == args[0]:
            wrapped_function = self.ad_wrapper(*args[1:], **kwargs)
        else:
            raise ValueError(
                "Calling bound decorator "
                + str(self)
                + " with unknown instance "
                + str(args[0])
                + "\n This decorator is bound to "
                + str(self._bound_to)
            )

        return wrapped_function

    def __get__(self, binding_instance: object, binding_type: type) -> Callable:
        """Implemenation of descriptor protocol.

        If this ADmethod decorates a class method (and effectively replaces it), it will
        be bound to the class instance, similar to bound methods.

        Every time this instance is syntactically accessed as an attribute of the class
        instance, this getter is called and returns a partially evaluated call to this
        instance instead.
        By calling this instance this way, we can save a reference to the `self`
        argument of the decorated class method and pass it as an argument to the
        decorated method.

        The reason why this is necessary is due to the fact, that class methods
        are always passed in unbound form to the decorator when the code is
        evaluated
        (i.e. they don't have a reference to the `self` argument, contrary to bound
        methods)

        Parameters:
            binding_instance: instance, whose method has been decorated by this class.
            binding_type: type instance of the decorated method's class/owner.

        """
        # Save a reference to the binding instance
        self._bound_to = binding_instance
        # A partial call to the decorator is returned, not the decorator itself.
        # This will trigger the function evaluation.
        return partial(self.__call__, binding_instance)

    def ad_wrapper(self, *args, **kwargs) -> Operator:
        """Actual wrapper function.

        Constructs the necessary AD-Operator class wrapping the decorated callable
        and performs the evaluation/call.

        Parameters:
            *args: arguments for the call to the wrapping AD operator function
            **kwargs: keyword argument for the call to the wrapping Ad operator function

        """
        # Make sure proper assignment of callable was made
        assert self._func is not None

        # extra safety measure to ensure a bound call is done to the right binding
        # instance. We pass only the binding instance referenced in the descr. protocol.
        if self._bound_to is None:
            operator_func = self._func
        else:
            # partial evaluation of a bound function,
            # since the AD operator has no reference to binding instance
            operator_func = partial(self._func, self._bound_to)

        # Resulting AD operator has a special name to mark its origin
        if "name" not in self._op_kwargs.keys():
            name = "ADmethod-"
            name += (
                str(self._ad_func_type.__qualname__)
                + "-decorating-"
                + str(self._func.__qualname__)
            )
            self._op_kwargs.update({"name": name})

        # calling the operator
        wrapping_operator = self._ad_func_type(func=operator_func, **self._op_kwargs)

        return wrapping_operator(*args, **kwargs)
