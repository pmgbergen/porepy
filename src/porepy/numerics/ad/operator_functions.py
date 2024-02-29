"""This module contains callable operators representing functions to be called with other
operators as input arguments.
Contains also a decorator class for callables, which transforms them automatically in the
specified operator function type.

"""

from __future__ import annotations

import abc
from functools import partial
from typing import Callable, Optional, Type

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.ad.forward_mode import AdArray

from .operators import Operator

__all__ = [
    "Function",
    "ConstantFunction",
    "DiagonalJacobianFunction",
    "InterpolatedFunction",
    "ADmethod",
]

### BASE CLASSES ------------------------------------------------------------------------------


class AbstractFunction(Operator):
    """Abstract class for all operator functions, i.e. functions evaluated on some other AD
    operators.

    Implements the callable-functionality and provides abstract methods for obtaining function
    values and the Jacobian.
    The abstraction intends to provide means for approximating operators, where values are
    e.g. interpolated and the Jacobian is approximated using FD.

    Note:
        One can flag the operator as ``ad_compatible``. If flagged, the AD framework passes
        AD arrays directly to the callable ``func`` and will **not** call the abstract methods
        for values and the Jacobian during operator parsing.
        If for some reason one wants to flag the function as AD compatible, but still have the
        abstract methods called, this is as of now **not** supported.

        For now only one child class, porepy.ad.Function, flags itself always as AD compatible.

    Parameters:
        func: callable Python object representing a (numeric) function.
            Expected to take numerical information in some form and return numerical
            information in the same form.
        name: name of this instance as an AD operator
        array_compatible (optional): If true, the callable ``func`` will be called
            using arrays (numpy.typing.ArrayLike). Flagging this true, the user ensures
            that the callable can work with arrays and return respectively
            formatted output. If false, the function will be evaluated element-wise
            (scalar input). Defaults to False.
        ad_compatible (Optional): If true, the callable ``func`` will be called using
            the porepy.ad.AdArray.

            Note that as of now, this will effectively bypass the abstract methods
            for generating values and the Jacobian, assuming both will be provided
            correctly by the return value of ``func``.

            Defaults to False.

    """

    def __init__(
        self,
        func: Callable,
        name: str,
        array_compatible: bool = False,
        ad_compatible: bool = False,
    ):
        ### PUBLIC

        self.func: Callable = func
        """Callable passed at instantiation"""

        self.array_compatible: bool = array_compatible
        """Indicator whether the callable can process arrays."""

        super().__init__(name=name, operation=Operator.Operations.evaluate)

    def __call__(self, *args: pp.ad.Operator) -> pp.ad.Operator:
        """Renders this function operator callable, fulfilling its notion as 'function'.

        Parameters:
            *args: AD operators passed as symbolic arguments for the callable passed at
                instantiation.

        Returns:
            Operator with call-arguments as children in the operator tree.
            The assigned operation is ``evaluate``.

        """
        children = [self, *args]
        op = Operator(children=children, operation=self.operation)
        return op

    def __repr__(self) -> str:
        """String representation of this operator function.
        Uses currently only the given name."""

        s = f"AD Operator function '{self._name}'"
        return s

    def __mul__(self, other):
        raise RuntimeError(
            "AD Operator functions are meant to be called, not multiplied."
        )

    def __add__(self, other):
        raise RuntimeError("AD Operator functions are meant to be called, not added.")

    def __sub__(self, other):
        raise RuntimeError(
            "AD Operator functions are meant to be called, not subtracted."
        )

    def __rsub__(self, other):
        raise RuntimeError(
            "AD Operator functions are meant to be called, not subtracted."
        )

    def __div__(self, other):
        raise RuntimeError("AD Operator functions are meant to be called, not divided.")

    def __truediv__(self, other):
        raise RuntimeError("AD Operator functions are meant to be called, not divided.")

    def parse(self, md: pp.MixedDimensionalGrid):
        """Parsing to a numerical value.

        The real work will be done by combining the function with arguments, during
        parsing of an operator tree.

        Parameters:
            md: Mixed-dimensional grid.

        Returns:
            The instance itself.

        """
        return self

    @abc.abstractmethod
    def get_values(self, *args: AdArray) -> np.ndarray:
        """Abstract method for evaluating the callable passed at instantiation.

        This method will be called during the operator parsing.
        The AD arrays passed as arguments will be in the same order as the operators passed to
        the call to this instance.

        The returned numpy array will be set as 'val' argument for the AD array representing
        this instance.

        Parameters:
            *args: AdArray representation of the operators passed during the call to this
                instance

        Returns:
            Function values in numerical format.

        """
        pass

    @abc.abstractmethod
    def get_jacobian(self, *args: AdArray) -> sps.spmatrix:
        """
        Abstract method for evaluating the Jacobian of the callable passed at instantiation.

        This method will be called during the operator parsing.
        The AD arrays passed as arguments will be in the same order as the operators passed to
        the call to this instance.

        The returned numpy array will be be set as 'jac' argument for the AD array representing
        this instance.

        Note:
            The necessary dimensions for the jacobian can be extracted from the dimensions
            of the Jacobians of passed AdArray instances.

        Parameters:
            *args: AdArray representation of the operators passed during the call to this
                instance

        Returns:
            Numeric representation of the Jacobian of this function.

        """
        pass


class AbstractJacobianFunction(AbstractFunction):
    """Partially abstract base class, providing a call to the callable ``func`` in order to
    obtain numeric function values.

    What remains abstract is the Jacobian.

    """

    def get_values(self, *args: AdArray) -> np.ndarray:
        """
        Returns:
            The direct evaluation of the callable using ``val`` of passed AD arrays.

        """
        # get values of argument AdArrays.
        vals = (arg.val for arg in args)

        # if the callable is flagged as conform for vector operations, feed vectors
        if self.array_compatible:
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
    where it is expected that passing AdArrays directly to ``func`` will
    return the proper result.

    Here the values **and** the Jacobian are obtained exactly by the AD framework.

    The intended use is as a wrapper for operations on pp.ad.AdArray objects,
    in forms which are not directly or easily expressed by the rest of the Ad
    framework.

    Note:
        This is a special case where the abstract methods for getting values and the
        Jacobian are formally implemented but never used by the AD framework. A separate
        operation called ``evaluate`` is implemented instead, which simply feeds the AD
        arrays to ``func``.

    """

    def __init__(self, func: Callable, name: str, array_compatible: bool = True):
        super().__init__(func, name, array_compatible)
        self.ad_compatible = True

    def get_values(self, *args: AdArray) -> np.ndarray:
        result = self.func(*args)
        return result.val

    def get_jacobian(self, *args: AdArray) -> np.ndarray:
        result = self.func(*args)
        return result.jac


class ConstantFunction(AbstractFunction):
    """Function representing constant, scalar values with no dependencies and ergo a
    zero Jacobian.

    It still has to be called though since it fulfills the notion of a 'function'.

    Parameters:
        values: constant values per cell.

    """

    def __init__(self, name: str, values: np.ndarray):
        # dummy function, takes whatever and returns only the pre-set values
        def func(*args):
            return values

        super().__init__(func, name)
        self._values = values

    def get_values(self, *args: AdArray) -> np.ndarray:
        """
        Returns:
            The values passed at instantiation.

        """
        return self._values

    def get_jacobian(self, *args: AdArray) -> sps.spmatrix:
        """
        Note:
            The return value is not a sparse matrix as imposed by the parent method signature,
            but a zero.
            Numerical operations with a zero always works with any numeric formats in
            numpy, scipy and PorePy's AD framework.
            Since the constant function (most likely) gets no arguments passed, we have
            no way of knowing the necessary shape for a zero matrix. Hence scalar.

        Returns: the trivial derivative of a constant.

        """
        return 0.0


class DiagonalJacobianFunction(AbstractJacobianFunction):
    """Approximates the Jacobian of the function using identities and scalar multipliers
    per dependency.

    Parameters:
        multipliers: scalar multipliers for the identity blocks in the Jacobian,
            per dependency of ``func``. The order in ``multipliers`` is expected to match
            the order of AD operators passed to the call of this function.

    """

    def __init__(
        self,
        func: Callable,
        name: str,
        multipliers: float | list[float],
        array_compatible: bool = False,
    ):
        super().__init__(func, name, array_compatible)
        # check and format input for further use
        if isinstance(multipliers, list):
            self._multipliers = [float(val) for val in multipliers]
        else:
            self._multipliers = [float(multipliers)]

    def get_jacobian(self, *args: AdArray) -> sps.spmatrix:
        """The approximate Jacobian consists of identity blocks times scalar multiplier
        per every function dependency.

        """
        # the Jacobian of a (Merged) Variable is already a properly sized block identity
        jac = args[0].jac * self._multipliers[0]

        # summing identity blocks for each dependency
        if len(args) > 1:
            # TODO think about exception handling in case not enough
            # L-values were provided initially
            for arg, L in zip(args[1:], self._multipliers[1:]):
                jac += arg.jac * L

        return jac


class InterpolatedFunction(AbstractFunction):
    """Represents the passed function as an interpolation of chosen order on a cartesian,
    uniform grid.

    The image of the function is expected to be of dimension 1, while the domain can be
    multidimensional.

    Note:
        All vector-valued ndarray arguments are assumed to be column vectors.
        Each row-entry represents a value for an argument of ``func`` in
        respective order.

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
        array_compatible: bool = False,
    ):
        super().__init__(func, name, array_compatible)

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

    def get_values(self, *args: AdArray) -> np.ndarray:
        # stacking argument values vertically for interpolation
        X = np.vstack([x.val for x in args])
        return self._table.interpolate(X)

    def get_jacobian(self, *args: AdArray) -> sps.spmatrix:
        # get points at which to evaluate the differentiation
        X = np.vstack([x.val for x in args])
        # allocate zero matrix for Jacobian with correct dimensions and in CSR format
        jac = sps.csr_matrix(args[0].jac.shape)

        for axis, arg in enumerate(args):
            # The trivial Jacobian of one argument gives us the correct position for the
            # entries as ones
            partial_jac = arg.jac
            # replace the ones with actual values
            # Since csr, we can simply replace the data array with the values of the derivative
            partial_jac.data = self._table.gradient(X, axis)[0]

            # add blocks to complete Jacobian
            jac += partial_jac

        return jac


### FUNCTION DECORATOR ------------------------------------------------------------------------


class ADmethod:
    """(Decorator) Class for methods representing e.g., physical properties.
    The decorated function is expected to take scalars/vectors and return a scalar/vector.

    The return value will be an AD operator of a type passed to the decorator.

    Examples:
        .. code:: python

            import porepy as pp

            # decorating class methods
            class IdealGas:

                @ADmethod(ad_operator=pp.ad.DiagonalJacobianFunction,
                        operators_args={"multipliers"=[1,1]})
                def density(self, p: float, T: float) -> float:
                    return p/T

            # decorating function
            @ADmethod(ad_operator=pp.ad.Function)
            def dummy_rel_perm(s):
                return s**2

        With above code, the density of an instance of ``IdealGas`` can be called using
        :class:`~porepy.numerics.ad.operators.MergedVariable` representing
        pressure and temperature.
        Analogously, ``dummy_rel_perm`` can be called with one representing the saturation.

    Note:
        If used as decorator WITHOUT explicit instantiation, the instantiation will be
        done implicitly with above default arguments (that's how Python decorators work).

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
        operator_kwargs: dict = {},
    ) -> None:
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
            that instance will effectively be replaced by another decorator instance created
            here in the call.
            It is expected that the the call will follow the instantiation immediately when
            used as a decorator, hence properly dereferencing the original instance.
            If used differently or if another reference is saved between explicit instantiation
            and call, this is a potential memory leak.

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

    def __get__(self, binding_instance: object, binding_type: type) -> Callable:
        """
        Descriptor protocol.

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

        # extra safety measure to ensure a bound call is done to the right binding instance.
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
