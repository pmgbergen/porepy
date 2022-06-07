"""
Contains callable operators representing functions to be called with other
operators as input arguments.
Contains also a decorator class for callables, which transforms them automatically in the
specified operator function type.
"""
from __future__ import annotations

import abc
from functools import partial
from types import FunctionType
from typing import Callable, List, Optional, Type, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.ad.forward_mode import Ad_array

from .operators import Operation, Operator, Tree

__all__ = [
    "Function",
    "DiagonalJacobianFunction",
    "ADmethod",
]

### BASE CLASSES ------------------------------------------------------------------------------


class AbstractFunction(Operator):
    """Abstract class for all operator functions.

    Implements the callable-functionality and provides abstract methods
        - :method:`~porepy.numerics.ad.operator_functions.AbstractFunction.get_values`
        - :method:`~porepy.numerics.ad.operator_functions.AbstractFunction.get_jacobian`

    The abstraction intends to provide means for approximating operators, where values are
    e.g. interpolated and the Jacobian is approximated using FD.
    """

    def __init__(
        self,
        func: Callable,
        name: str,
        is_array_func: bool = False,
        is_adarray_func: bool = False,
    ):
        """Constructor. Saves information about the passed callable.
        The passed callable is expected to take numerical information in some form and
        return numerical information in some form.

        Important NOTE: flagging this instance with `is_adarray_func` will lead to a direct
        evaluation of the callable using Ad_array instances, effectively bypassing
        the abstract methods (but still leading to NotImplementedError if not implemented).
        Might not be elegant, but we can put the old `Function` class this way in the same
        category/hierarchy.

        :param func: callable Python object representing the function
        :type func: callable
        :param name: Name of this operator instance
        :type name: str
        :param is_array_func: Indicator if passed callable can be fed with vectors
            (numpy.ndarray)
        :type is_array_func: bool
        :param is_adarray_func: indicator whether callable can be fed with Ad_array instances
            or not
        :type is_adarray_func: bool
        """
        ### PUBLIC
        # Reference to callable passed at instantiation
        self.func: Callable = func
        # indicator whether above callable can be fed with vectors (numpy.ndarray) or not
        self.is_array_func: bool = is_array_func
        # indicator whether above callable can be fed with Ad_array instances
        self.is_adarray_func: bool = is_adarray_func

        ### PRIVATE
        self._name: str = name
        self._operation: Operation = Operation.evaluate
        self._set_tree()

    def __call__(self, *args):
        """
        Call to operator object with 'args' as children.
        Arguments are expected to be of type
        :class:`~porepy.ad.Variable` or :class:`~porepy.ad.MergedVariable`.

        The children are evaluated and respective AD arrays are passed to the abstract
        `get_values` and `get_jacobian` methods.
        """
        children = [self, *args]
        op = Operator(tree=Tree(self._operation, children=children))
        return op

    def __repr__(self) -> str:
        """String representation of this operator function.
        Uses currently only the given name."""

        s = f"AD Operator function with name {self._name}"
        return s

    def __mul__(self, other):
        raise RuntimeError(
            "ad.Operator functions should only be called, not multiplied."
        )

    def __add__(self, other):
        raise RuntimeError("ad.Operator functions should only be called, not added.")

    def __sub__(self, other):
        raise RuntimeError(
            "ad.Operator functions should only be called, not subtracted."
        )

    def __div__(self, other):
        raise RuntimeError("ad.Operator functions should only be called, not divided.")

    def __truediv__(self, other):
        raise RuntimeError("ad.Operator functions should only be called, not divided.")

    def parse(self, gb: pp.GridBucket):
        """Parsing to a numerical value.

        The real work will be done by combining the function with arguments, during
        parsing of an operator tree.

        Parameters:
        :param gb: Mixed-dimensional grid. Not used, but it is needed as input to be
            compatible with parse methods for other operators.
        :type gb: :class:`~porepy.GridBucket`

        :return: the instance itself
        """
        return self

    @abc.abstractmethod
    def get_values(self, *args: Ad_array) -> np.ndarray:
        """
        Abstract method for evaluating the callable passed at instantiation.

        The AD arrays passed as arguments will be in the same order as the operators passed to
        the call to this instance.

        The returned numpy array will be be set as 'val' argument for the AD array representing
        this instance.
        """
        pass

    @abc.abstractmethod
    def get_jacobian(self, *args: Ad_array) -> sps.spmatrix:
        """
        Abstract method for evaluating the Jacobian of the callable passed at instantiation.

        The AD arrays passed as arguments will be in the same order as the operators passed to
        the call to this instance.

        The returned numpy array will be be set as 'jac' argument for the AD array representing
        this instance.

        NOTE: the necessary dimensions for the jacobian can be extracted from the dimensions
        of the jacobian of passed Ad_array instances.
        """
        pass


class AbstractJacobianFunction(AbstractFunction):
    """'Half'-abstract base class, for which a direct evaluation of the passed callable is
    implemented. Only the Jacobian has to be concretized / approximated."""

    def get_values(self, *args: Ad_array) -> np.ndarray:
        """
        Evaluates the passed callable directly using values of Ad_array instances
        passed as arguments.

        :param args: tuple of :class:`~porepy.numerics.ad.forward_mode.Ad_array`
        :type args: tuple
        :return: results from callable packed into an array.
        :rtype: numpy.array
        """
        # get values of argument Ad_arrays.
        vals = (arg.val for arg in args)

        # if the callable is flagged as conform for vector operations, feed vectors
        if self.is_array_func:
            return self.func(*vals)
        else:
            # if not vector-conform, feed element-wise

            # TODO this displays some special behavior when val-arrays have different lengths:
            # it returns None-like things for every iteration more then shortest length
            # These Nones are ignored for some reason by the function call, as well as by the
            # array constructor.
            # If a mortar var and a subdomain var are given as args,
            # then the lengths will be different for example.
            return np.array([self.func(*vals_i) for vals_i in zip(*vals)])


### CONCRETE IMPLEMENTATIONS ------------------------------------------------------------------


class Function(AbstractFunction):
    """Ad representation of an analytically given function,
    where it is expected that feeding the callable with Ad_arrays directly will give
    give the proper result.

    Here the values AND the Jacobian are obtained exactly by standard rules of differentiation.

    The intended use is as a wrapper for operations on pp.ad.Ad_array objects,
    in forms which are not directly or easily expressed by the rest of the Ad
    framework.

    NOTE: this is a special case where `get_values` and `get_jacobian` are formally implemented
    but never used by the AD framework
    (due to the assumed, direct evaluation of result using the callable).
    """

    def __init__(self, func: Callable, name: str, is_vector_func: bool = True):
        super().__init__(func, name, is_vector_func, True)

    def get_values(self, *args: Ad_array) -> np.ndarray:
        result = self.func(*args)
        return result.val

    def get_jacobian(self, *args: Ad_array) -> np.ndarray:
        result = self.func(*args)
        return result.jac


class ConstantFunction(AbstractFunction):
    """"Function representing constant, scalar values with no dependencies, i.e. zero Jacobian.
    """

    def __init__(self, values: np.ndarray, name: str):
        # dummy function, takes whatever and returns only the pre-set values
        def func(*args):
            return values
        super().__init__(func, name)
        self._values = values
    
    def get_values(self, *args: Ad_array) -> np.ndarray:
        """Returns the values passed at instantiation."""
        return self._values

    def get_jacobian(self, *args: Ad_array) -> sps.spmatrix:
        """Returns the trivial derivative of a constant."""
        # NOTE it's not a sparse matrix as imposed by the parent method signature,
        # but the multiplication with a zero always works with any numeric format in
        # numpy, scipy
        return 0.


class DiagonalJacobianFunction(AbstractJacobianFunction):
    """
    Approximates the Jacobian of the function using the a diagonal matrix and a scalar
    multiplier per dependency.
    """

    def __init__(
        self,
        multipliers: Union[List[float], float],
        func: Callable,
        name: str,
        is_vector_func: bool = False,
    ):
        """Constructor.

        The scalar multipliers for the diagonal matrix can be passed for every
        function dependency specifically using a list.
        The order in the list has to mach the order of arguments when evaluating
        the function (i.e. the variables passed to `__call__`).

        :param L: multiplier for identity for L-scheme
        :type L: float / List[float]
        """
        super().__init__(func, name, is_vector_func)
        # check and format input for further use
        if isinstance(multipliers, list):
            self._multipliers = [float(val) for val in multipliers]
        else:
            self._multipliers = [float(multipliers)]

    def get_jacobian(self, *args: Ad_array) -> sps.spmatrix:
        """The approximate jacobian is identity times scalar multiplier for every
        function dependency.

        Where the respective diagonal blocks appears,
        depends on the total DOFs and the order of arguments passed during the
        call to this instance.
        """
        # the Jacobian of a (Merged) Variable is already a properly sized block identity
        if len(args) >= 1:
            jac = args[0].jac * self._multipliers[0]

            # summing identity blocks for each dependency
            if len(args) > 1:
                # TODO think about exception handling in case not enough
                # L-values were provided initially
                for arg, L in zip(args[1:], self._multipliers[1:]):
                    jac += arg.jac * L
        else:
            # TODO assert zero as scalar will cause no type errors with other operators
            # this is the case for a function independent of system variables...
            jac = 0.0

        return jac


### FUNCTION DECORATOR ------------------------------------------------------------------------


class ADmethod:
    """
    Automatic-Differentiation method/function.

    (Decorator) Class for methods representing e.g., physical properties.
    The decorated function is expected to take scalars/vectors and return a scalar/vector.
    See example usage below.

    The return value will be an AD operator of a type passed to the decorator.

    EXAMPLE USAGE:
    .. code-block:: python
        import porepy as pp

        # decorating class methods
        class IdealGas:

            @ADmethod(ad_operator=pp.ad.LJacFunction, operators_args={"L"=[1,1]})
            def density(self, p: float, T: float) -> float:
                return p/T

        # decorating function
        @ADmethod(ad_operator=pp.ad.Function)
        def dummy_rel_perm(s):
            return s**2

    With above code, the density of an instance of 'IdealGas' can be called using
    :class:`~porepy.numerics.ad.operators.MergedVariable` representing
    pressure and temperature.
    Analogously, `dummy_rel_perm` can be called with one representing the saturation.
    """

    def __init__(
        self,
        func: FunctionType = None,
        ad_function_type: Type["AbstractFunction"] = Function,
        operator_kwargs: Optional[dict] = {},
    ) -> None:
        """
        Decorator class constructor.
        Saves information about the requested AD Function type and keyword arguments necessary
        for its instantiation.

        NOTE: If used as decorator WITHOUT explicit instantiation, the instantiation will be
        done implicitly with above default arguments (that's how Python decorators work).

        :param func: decorated function object
        :type func: function
        :param ad_function_type: reference to the requested AD class (type not class instance!)
        :type ad_function_type: Type[:class:`AbstractFunction`]
        :param operator_kwargs: keyword arguments to be passed when instantiating operator
        :type operator_kwargs: dict
        """
        # reference to decorated function object
        self._func = func
        # mark if decoration without explicit call to constructor
        self._explicit_init = func is None
        # reference to instance, to which the decorated bound method belongs, if any
        # if this remains None, then an unbound method was decorated
        self._bound_to: Union[None, object] = None
        # reference to operator type which should wrap the decorated method
        self._ad_func_type = ad_function_type
        # keyword arguments for call to constructor of operator type
        self._op_kwargs = operator_kwargs

    def __call__(self, *args, **kwargs) -> Union["ADmethod", "pp.ad.Operator"]:
        """
        Wrapper factory.
        The decorated object is wrapped and/or evaluated here.

        Dependent on whether the decorated function is a method belonging to a class,
        or an unbound function, the wrapper will have a different signature.

        If bound to a class instance, the wrapper will include a partial function, where the
        instance of the class was already passed beforehand.
        """
        # if decorated without explicit init,
        # the function is passed during a call to the decorator as first argument
        if self._func is None:
            self._func = args[0]

        # if an explicit init was made,
        # mimic a non-explicit init to get an object with descriptor protocol
        if self._explicit_init:
            # TODO VL: check if the ADmethod instance whose __call__ is executed right now
            # gets properly de-referenced and deleted, or if it remains hidden in the memory
            return ADmethod(
                func=self._func,
                ad_function_type=self._ad_func_type,
                operator_kwargs=self._op_kwargs,
            )

        # without an explicit init, the first decorator itself replaces the decorated function
        # This results in a call to ADmethod.__call__ instead of
        # a call to the decorated function

        # when calling the decorator, distinguish between bound method call
        # ('args' contains 'self' of the decorated instance) and an unbound function call
        # (whatever 'args' and 'kwargs' contain, we pass it to the wrapper)
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

    def ad_wrapper(self, *args, **kwargs):
        """
        Actual wrapper function.
        Constructs the necessary AD-Operator class and performs the evaluation.
        """
        # extra safety measure to ensure a a bound call is done to the right, binding instance.
        # We pass only the binding instance referenced in the descriptor protocol.
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

    def __get__(self, binding_instance: object, binding_type: type):
        """
        Descriptor protocol.

        If this ADmethod decorates a class method (and effectively replaces it), it will bound
        to the class instance, similar to bound methods

        Every time this instance is syntactically accessed as an attribute of the
        class instance, this getter is called and returns a partially evaluated call
        to this instance instead.
        By calling this instance this way, we can save a reference to the `self` argument of
        the decorated class method and pass is as an argument to the decorated method.

        The reason why this is necessary is due to the fact, that  class methods
        are always passed in unbound form to the decorator when the code is
        evaluated
        (i.e. they don't have a reference to the `self` argument, contrary to bound methods)

        :param binding_instance: instance for binding this object's call to it.
        :type binding_instance: Any
        :param binding_type: type variable of the binding instance
        :type binding_type: type
        """
        # safe a reference to the binding instance
        self._bound_to = binding_instance
        # a partial call to the decorator is returned, not the decorator itself.
        # This will trigger the function evaluation.
        return partial(self.__call__, binding_instance)
