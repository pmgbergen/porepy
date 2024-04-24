"""This module contains callable operators representing functions to be called with other
operators as input arguments.
Contains also a decorator class for callables, which transforms them automatically in the
specified operator function type.

"""

from __future__ import annotations

import abc
import numbers
from functools import partial
from typing import Callable, Optional, Type, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.ad.forward_mode import AdArray

from .operators import Operator

__all__ = [
    "admethod",
    "Function",
    "ConstantFunction",
    "DiagonalJacobianFunction",
    "InterpolatedFunction",
    "SemiSmoothMin",
]

### BASE CLASSES -----------------------------------------------------------------------


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


### CONCRETE IMPLEMENTATIONS -----------------------------------------------------------


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
        jac = args[0].jac * self._multipliers[0] if isinstance(args[0], AdArray) else 0

        # summing identity blocks for each dependency
        if len(args) > 1:
            # TODO think about exception handling in case not enough
            # L-values were provided initially
            for arg, L in zip(args[1:], self._multipliers[1:]):
                jac += arg.jac * L if isinstance(arg, AdArray) else 0

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


class SemiSmoothMin(AbstractFunction):
    """Function representing the semi-smooth ``min(-,-)``-function for two operators.

    The evaluation of the derivative (and function-values) follows basically an
    active-set-strategy.

    The active set is defined by the those DOFs, where operator 1 has (strictly) larger values.
    On the active set, the values correspond to the values of operator 2.
    On the inactive set, the values correspond to the values of operator 1.

    The Jacobian is chosen from elements of the B-sub-differential.
    On the active set, the derivatives of operator 2 are chosen.
    On the inactive set, the derivatives of operator 1 are chosen.

    In a multi-dimensional setting this corresponds to selecting respective rows of each
    sub-differential and inserting them in the final Jacobian.

    Parameters:
        op1: first AD Operator, representing the first argument to the min-function.
        op1: second AD Operator, representing the second argument to the min-function.

    """

    def __init__(self):
        name = f"semi-smooth MIN operator"
        # dummy function, not supposed to be used

        def func(x, y):
            return x if x < y else y

        super().__init__(func, name, False, False)

        self.ad_compatible = False

    def get_values(self, *args: AdArray | np.ndarray) -> np.ndarray:
        # this will throw an error if more than two arguments were passed
        op1, op2 = args
        if all(isinstance(a, np.ndarray) for a in args):
            # active set choice
            active_set = (op1 - op2) > 0.0
            # default/inactive vals
            vals = op1.copy()
            # replace vals on active set
            vals[active_set] = op2[active_set]
        else:
            # active set choice
            active_set = (op1.val - op2.val) > 0.0
            # default/inactive vals
            vals = op1.val.copy()
            # replace vals on active set
            vals[active_set] = op2.val[active_set]
        return vals

    def get_jacobian(self, *args: AdArray) -> sps.spmatrix:
        # this will throw an error if more than two arguments were passed
        op1, op2 = args
        # active set choice
        active_set = (op1.val - op2.val) > 0.0
        # default/inactive choice (lil format for faster assembly)
        jac = op1.jac.copy()
        # replace (active set) rows with the differential from the other operator
        jac[active_set] = op2.jac[active_set]
        return jac.tocsr()


### FUNCTION DECORATOR -----------------------------------------------------------------

NumericType = Union[pp.number, np.ndarray, AdArray]


def admethod(
    func: Optional[Callable] = None,
    ad_ftype: Type[AbstractFunction] = Function,
    ad_fkwargs: dict = {},
) -> Callable[[Callable], ADMethod] | ADMethod:
    """Decorator to be used for class methods and functions to turn them into
    AD operator functions if they receive AD operators as input arguments.

    Note:
        This function serves as a factory for :class:`ADMethod` instances, which
        will replace the decorated function.

        The process between decorations with ``@admethod`` and ``@admethod(...)`` are
        slightly different. See examples below and documentation of :class:`ADMethod`.

    Examples:

        .. rubric:: Example 1

        Here we demonstrate how to use this function, with and without a call during
        decoration.

        Without a call, the default arguments are used to create a
        :class:`Function` factory using :class:`ADMethod`.

        .. code:: python

            import porepy as pp

            @admethod
            def dummy_rel_perm(saturation):
                return saturation * saturation

        During execution time,
        above example code will call ``admethod`` with the function ``dummy_rel_perm``
        passed as argument ``func``.
        ``dummy_rel_perm`` will then be replaced with an instance of :class:`ADMethod`,
        which creates :class:`Function`
        instances and calls them everytime the argument ``saturation`` is
        an AD operator.
        So calling ``dummy_rel_perm`` with e.g.,
        a :class:`~porepy.numerics.ad.operators.Variable` will return another AD
        operator, equivalent to the syntax.

        >>> pp.ad.Function(dummy_rel_perm, name='')(saturation)

        If ``saturation`` is not an AD operator, but any other numerical format
        accepted by :class:`ADMethod`, then ``dummy_rel_perm`` is callable like usual
        and will immediatly return the value ``saturation * saturation``.

        .. rubric:: Example 2

        For using other operator functions besides :class:`Function`, you can use
        this function with a call during decoration:

        .. code:: python

            @admethod(
                ad_ftype=pp.ad.DiagonalJacobianFunction, ad_fkwargs={'multipliers'=[2]}
            )
            def dummy_rel_perm(saturation):
                return saturation * saturation

        This will give the same result as in above example, only that calls with
        AD operators will not perform a call to :class:`Function`, but to
        :class:`DiagonalJacobianFunction`.
        ``ad_fkwargs`` will be used as arguments in the instantiation of
        :class:`DiagonalJacobianFunction`.

        Warning:
            When decorating with a call to ``admethod``, the arguments
            ``ad_ftype`` and ``ad_fkwargs`` must be given as keyword arguments.

        .. rubric:: Example 3

        Consider a decorated function taking two arguments

        .. code:: python

            @admethod
            def mutiply(x, y):
                return x * y

        As seen above, here the multiplication will be wrapped into an
        :class:`Function`.

        We can compute cheap partial derivatives w.r.t. ``y``, if we let ``y`` be an
        :class:`~porepy.numerics.ad.forward_mode.AdArray` and ``x`` some other
        numeric format.
        ``y`` can either be a simple ``AdArray`` with ``val=...`` and
        ``jac=numpy.array([0, 1])``,
        or a :class:`~porepy.numerics.ad.operators.Variable`,
        which wrapps above construction.

        >>> x = 3
        >>> y = pp.ad.Variable(...)
        >>> dy = multiply(x, y).evaluate(...)

        ``dy`` will in effectively be an expression multiplying the constant value of
        ``x`` with the ``AdArray`` representing ``y``, hence a partial derivative.

        .. rubric:: Example 4

        As of now, this decorator is compatible with Python's
        :obj:`classmethod`, :obj:`staticmethod` and :obj:`~abc.abstractmethod`.

        In combination with other decorators, make sure to use ``admethod`` as the
        outermost/topmost decorator.

        .. code:: python

            import abc
            import porepy as pp

            class Foo(abc.ABC):

                @pp.ad.admethod
                @classmethod  # staticmethod
                @abc.abstractmethod
                def bar():
                    ...

    Note:
        **When to use this.**

        When dealing with complex heuristic laws implemented as functions,
        this decorator enables the user to evaluate the function as usual if
        the input args are not AD operators.

        But if one wants to use the AD framework, it is often very bad for performance
        reasons to implement the heuristic law directly with AD operators, due to
        the possibly large operator tree and expensive recursion in the evaluation.

        This function aims for feeding the heuristic law directly with instances of
        :class:`~porepy.numerics.ad.forward.AdArray`, using the framework provided
        by AD operator functions. It reduces the heuristic law to a single evaluation
        process in the final operator tree.

        Another obvious benefit is, that this approach does not produce any overhead if
        the decorated function is called without AD arguments, but nevertheless
        enabling the user to use both worlds in a flexible way.

    Parameters:
        func: ``default=None``

            Reference to the decorated function.
        ad_ftype: ``default=:class:`Function``

            A type of AD operator function
            (see :class:`AbstractFunction`) which should replace the decorated function
            if it is called with AD Operators as arguments.
        ad_fkwargs: ``default={}``

            A dictionary containing keyword arguments for the
            instantiation of the AD operator function of type ``ad_ftype``.

    Returns:
        An instance of :class:`ADMethod` mimicing the decorated function and capable
        of taking AD operators  and\or other numeric formats as arguments.

    """

    if func is None:
        # Here the first call to admethod was done in the decoration.
        # Meaning we must return a callable which will in the next call take the
        # decorated function as input.
        def admethod_(func_):
            # allow for compatibility with abstractmethod decorator
            decorated = ADMethod(func_, ad_ftype=ad_ftype, ad_fkwargs=ad_fkwargs)
            if hasattr(func_, "__isabstractmethod__"):
                setattr(decorated, "__isabstractmethod__", func_.__isabstractmethod__)
            return decorated

        return admethod_
    else:
        # Here admethod was used in the decorator without call.
        # This means that the first call to admethod contains the reference to the
        # decorated function.

        # allow for compatibility with abstractmethod decorator
        decorated = ADMethod(func, ad_ftype=ad_ftype, ad_fkwargs=ad_fkwargs)
        if hasattr(func, "__isabstractmethod__"):
            setattr(decorated, "__isabstractmethod__", func.__isabstractmethod__)
        return decorated


class ADMethod:
    """(Decorator) Class for methods/functions taking numerical values and
    returning numerical values, e.g. physical properties or heuristic laws.

    The decorated function is expected to take

    - real numbers ,
    - :obj:`~numpy.ndarray` or
    - :class:`~porepy.numerics.ad.forward_mode.AdArray`,

    and return one of above.

    Note:
        In this context, instances of :class:`~porepy.numerics.ad.forward_mode.AdArray`
        are treated as *numerical* values, whereas instances of
        :class:`~porepy.numerics.ad.operator.Operator` are treated as AD operators.

    Important:
        As of now, the framework for AD Operator functions treats only arguments,
        not keyword arguments.

        Therefore, this decorator is restricted to functions which take only positional
        arguments.

    The decorated function will be able to take
    :class:`~porepy.numerics.ad.operator.Operator`-instances as arguments and return
    another operator if done so.

    As soon as a single function argument is recognized to be an AD operator,
    a call to the decorated function will return an operator which needs to be
    evaluated using :meth:`~porepy.numerics.ad.operator.Operator.evaluate`.

    Otherwise the call will instantly pass the arguments to respective function and
    return its respective return values, if successful.

    In the case of AD operators as arguments,
    the return value will be an AD operator of the type passed to the decorator.

    Note:
        For full flexibility in the decoration, the user should resort to using
        the factory :func:`admethod` (see examples therein).

        The argument ``func`` is not optional, due to the creation of instances of this
        class being outsourced to :func:`admethod` for clarity reasons.

        For more information about how references to decorated functions are handled in
        Python, check Python's official docs.

    Parameters:
        func: Decorated function object.
        ad_ftype: Type reference to an AD operator function to be instantiated
            during a call with AD operator arguments.
        ad_fkwargs: Keyword arguments to be passed when instantiating an operator
            of type ``ad_function_type``.

    """

    def __init__(
        self,
        func: Callable,  # : Optional[Callable] = None
        ad_ftype: Type[AbstractFunction] = Function,
        ad_fkwargs: dict = {},
    ) -> None:
        # reference to decorated function object
        self._func: Callable = func
        # mark if decoration without explicit call to constructor
        self._explicit_init = func is None
        # Reference to object, to which the decorated bound method belongs, if any.
        # If this remains None, then an unbound method was decorated
        self._bound_to: Optional[object] = None
        self._bound_to_type: Optional[object] = None
        # reference to operator type which should wrap the decorated method
        self._ad_ftype = ad_ftype
        # keyword arguments for call to constructor of operator type
        self._op_kwargs = ad_fkwargs

    def __call__(self, *args, **kwargs) -> NumericType | Operator:
        """Wrapper factory.
        The decorated object is wrapped and/or evaluated here.

        Dependent on whether the decorated function is a method belonging to a class,
        or an unbound function, the wrapper will have a different signature.

        If bound to a class instance, the wrapper will include a partial function, where
        the instance of the class was already passed beforehand.

        If called with non-AD-operator ``args`` and ``kwargs``, it returns the
        return value of the decorated function in its respective format.

        Parameters:
            *args: Arguments for the call to the decorated function.
            **kwargs: Keyword arguments for the call.

        Returns:
            A call to this class will be treated as a call to the decorated function.

            Meaning, if numerical values are passed, the function is evaluated
            and its return value returned.

            If the function call is performed using AD Operator instances,
            the respective AD Operator function defined by instantiation argument
            ``ad_ftype`` is instantied and a call is performed using the arguments.

        """
        # If an explicit init was made, mimic a non-explicit init to get an object with
        # the descriptor protocol.
        # If decorated without explicit init, the function is passed during a call to
        # the decorator as first argument.
        # Without an explicit init, the first decorator itself replaces the decorated
        # function. This results in a call to ADmethod.__call__ instead of
        # a call to the decorated function
        # NOTE: Below code is used for including a factory in this class
        # but we are currently outsourcing the factory to admethod.

        # if self._func is None:
        #     self._func = args[0]

        # if self._explicit_init:
        #     return ADMethod(
        #         func=self._func,
        #         ad_ftype=self._ad_func_type,
        #         ad_fkwargs=self._op_kwargs,
        #     )

        # Make sure proper assignment of callable was made
        assert self._func is not None

        # when calling the decorator, distinguish between bound method call
        # ('args' contains 'self' or 'cls' of the decorated instance as first argument)
        # and an unbound function call
        # (whatever 'args' and 'kwargs' contain, we pass it to the wrapper)
        if self._bound_to == args[0] or self._bound_to_type == args[0]:
            # extract args which are neither 'self' nor 'cls'
            call_args = args[1:]

            # If bound, we use a partial evaluation of a bound function.
            # If the method was decorated with staticmethod, there is no partial eval.
            if isinstance(self._func, staticmethod):
                decorated_func = self._func.__func__

            # If the method was decorated with classmethod, make a partial eval using
            # type of the binding object.
            elif isinstance(self._func, classmethod):
                assert self._bound_to_type is not None
                decorated_func = partial(self._func.__func__, self._bound_to_type)

            # If the method is a regular instance method, make a partial eval using
            # the binding object itself.
            else:
                assert self._bound_to is not None
                decorated_func = partial(self._func, self._bound_to)

        # If unbound, we take the decorated function as is
        elif self._bound_to is None and self._bound_to_type is None:
            call_args = args
            decorated_func = self._func
        # Else something went wrong. This could happen if someone uses this class
        # in a hacky way
        else:
            raise ValueError(
                "Calling bound ADMethod instance\n\t"
                + str(self)
                + "\nwith unknown owner\n\t"
                + str(args[0])
                + "\nThis decorator is bound to\n\t"
                + str(self._bound_to)
            )

        # TODO: As of now, this decorator will NEVER turn functions without arguments
        # into AD operators, necessitating some additional work for that.

        # If any argument is an AD operator,
        # wrap the decorated method into an AD operator function
        # and perform a call to it.
        if self._check_if_arguments_ad(call_args, kwargs):
            # wrapping non-AD-operator arguments into AD for compatibility
            call_args, call_kwargs = self._wrap_into_ad(call_args, kwargs)

            # If name not explicitely set, give a default name to indicate origin
            if "name" not in self._op_kwargs.keys():
                name = "ADmethod-"
                name += (
                    str(self._ad_ftype.__qualname__)
                    + "-decorating-"
                    + str(self._func.__qualname__)
                )
                self._op_kwargs.update({"name": name})

            # calling the operator function
            ad_function = self._ad_ftype(func=decorated_func, **self._op_kwargs)
            # calling operator function with given arguments and returning
            return ad_function(*call_args, **call_kwargs)

        # If no argument is an AD operator,
        # we evaluate the function directly and return whatever is returned.
        else:
            return decorated_func(*call_args, **kwargs)

    def __get__(self, instance_: object, type_: type) -> Callable:
        """Descriptor protocol.

        If this ADmethod decorates a class method (and effectively replaces it), it will
        be bound to the class instance, similar to bound methods.

        Every time this instance is syntactically accessed as an attribute of the class
        instance, this getter is called and returns a partially evaluated call to this
        instance instead.
        By calling this instance this way, we can save a reference to the `self`
        (or 'cls') argument of the decorated class method and
        pass it as an argument to the decorated method.

        The reason why this is necessary is due to the fact, that class methods
        are always passed in unbound form to the decorator when the code is
        evaluated
        (i.e. they don't have a reference to the `self` argument, contrary to bound
        methods)

        Parameters:
            instance_: Instance, whose method has been decorated by this class and which
                is being accessed through the instance.
            type_: Object (f.e. class type) which has the decorated method in its
                directory

        Returns:
            A partially evaluated call to this instance with the binding instance
            as the first argument.

        """

        # If the decorated function is called using a binding instance of a class
        # ('self' in the decorated function)
        # perform a partial evaluation with 'self' as the first argument
        if instance_ is not None:
            # Save a reference to the binding instance
            # and to type of the instance, in case a classmethod was called
            self._bound_to = instance_
            self._bound_to_type = type_
            return partial(self.__call__, instance_)

        # If the decorated function is a member of a class (static or class method)
        # ('cls' in class methods)
        # perform a partial evaluation with 'cls' as the first argument
        elif isinstance(type_, type):
            # Save a reference to the binding type only
            # This happens if class methods are called without instantiation of
            # respective class
            self._bound_to_type = None
            self._bound_to_type = type_
            return partial(self.__call__, type_)

        # Else raise error? Descriptor protocoll probably never called without
        # instance or owner
        # TODO VL: As far as I know only 'self' and 'cls' are the two *behind-the-scene*
        # arguments for bound methods.
        else:
            self._bound_to = None
            self._bound_to_type = None
            return self.__call__

    def _check_if_arguments_ad(self, arguments: tuple, kwarguments: dict) -> bool:
        """Auxiliary function to check if a call to the AD method was done using
        arguments in AD operator form.

        Parameters:
            arguments: A tuple containing all the arguments in the function call.
            kwarguments: A dictionary containing the keyword-arguments in the function
                call.

        Raises:
            TypeError: If any argument is not a real number, a :obj:`~numpy.ndarray` or
                :class:`~porepy.numerics.ad.forward_mode.AdArray`,
                besides AD Operators.

        Returns:
            True, if any argument or keyword-argument-value is an instance of
            :class:`~porepy.numerics.ad.operator.Operator`.

            False, if they are (real) numbers or numpy arrays.

        """
        faulty_args = []
        faulty_kwargs = dict()
        has_ad_argument = False

        # checking arguments
        # if any is encountered, True is returned
        # TODO VL: The check for faulty arguments can be removed it wrapping is not
        # necessary. See comment about discussion with EK in _wrap_into_ad
        for arg in arguments:
            if isinstance(arg, (numbers.Real, np.ndarray, AdArray)):
                pass  # acceptable arguments
            elif isinstance(arg, Operator):
                has_ad_argument = True  # flag call as AD
            else:
                faulty_args.append(arg)

        for kw, arg in kwarguments.items():
            if isinstance(arg, (numbers.Real, np.ndarray, AdArray)):
                pass  # acceptable arguments
            elif isinstance(arg, Operator):
                has_ad_argument = True  # flag call as AD
            else:
                faulty_kwargs.update({kw: arg})

        if faulty_args or faulty_kwargs:
            msg = "Unsupported argument types:\n"
            for arg in faulty_args:
                msg += f"\t{str(type(arg))}\n"
            for kw, arg in faulty_kwargs.items():
                msg += f"'{kw}': {str(type(arg))}"

            raise TypeError(msg)

        return has_ad_argument

    def _wrap_into_ad(self, call_args: tuple, call_kwargs: dict) -> tuple[tuple, dict]:
        """Auxiliary function wrapping real numbers and numpy arrays into
        :class:`~porepy.numerics.ad.operators.Scalar` and
        :class:`~porepy.numerics.ad.operators.Array` respectively.

        This step is necessary, since as of now the AD framework treats call arguments
        for AD Operator functions as childer (other AD Operators) which must have
        :meth:`~porepy.numerics.ad.operators.Operator.parse` defined.

        Note:
            Instances of :class:`~porepy.numerics.ad.forward_mode.AdArray` and
            :class:`~porepy.numerics.ad.operators.Operator` need no wrapping and are
            not processed.

        Parameters:
            call_args: Arguments for the call to the decorated function
            call_kwargs: Keyword arguments for the call to the decorated function.

        Returns:
            Above (order-preserving) structure,
            where individual arguments and keyword-argument-values are wrapped in
            an AD operator if necessary.

        """
        wrapped_call_args = list()
        wrapped_call_kwargs = dict()

        # TODO VL: This wrapping is NOT necessary if we change Operator.evaluate
        # s.t. it returns AdArray with jac=0 for scalars and numpy arrays for instance.
        # Discuss with EK

        for arg in call_args:
            # wrap real numbers into AD scalars
            if isinstance(arg, numbers.Real):
                wrapped_call_args.append(pp.ad.Scalar(arg))
            # wrap np arrays into AD arrays
            elif isinstance(arg, np.ndarray):
                wrapped_call_args.append(pp.ad.Array(arg))
            # custom AdArray and operator instances are left as is
            elif isinstance(arg, (AdArray, Operator)):
                wrapped_call_args.append(arg)
            # anything else will raise an error, but this should not happen
            # due to the previous call to _check_if_arguments_ad
            else:
                raise TypeError("This should not happen.")

        for kw, arg in call_kwargs.items():
            # wrap real numbers into AD scalars
            if isinstance(arg, numbers.Real):
                wrapped_call_kwargs.update({kw: pp.ad.Scalar(arg)})
            # wrap np arrays into AD arrays
            elif isinstance(arg, np.ndarray):
                wrapped_call_kwargs.update({kw: pp.ad.Array(arg)})
            # custom AdArray and operator instances are left as is
            elif isinstance(arg, (AdArray, Operator)):
                wrapped_call_kwargs.update({kw: arg})
            # anything else will raise an error, but this should not happen
            # due to the previous call to _check_if_arguments_ad
            else:
                raise TypeError("This should not happen.")

        return tuple(wrapped_call_args), wrapped_call_kwargs
