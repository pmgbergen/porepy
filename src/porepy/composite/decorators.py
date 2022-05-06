""" Contains Decorator classes used by the composite submodule.

Decorator classes get instantiated once per decoration at python runtime,
namely at that point when the decorated object gets imported into the runtime,
i.e. the code is interpreted.

Python expects the decorator class to be callable.
A call using the decorated method as an argument is done immediately after instantiation of
the decorator class.
"""

from functools import wraps
from types import FunctionType
from typing import Any, TypeVar

import porepy as pp

from ._composite_utils import COMPUTATIONAL_VARIABLES

__all__ = [
    "ADProperty"
]


class THMSignature:
    """
    Thermal-Hydraulic-Mechanic-Signature/
    Decorator class for methods representing physical attributes.
    The method is expected to take number arguments and to return a number.

    This class specifies the physical character of the method signature,
    i.e. which number argument represent which physical quantity.
    Physical variables are defined in
    :data:`~porepy.composite._composite_utils.COMPUTATIONAL_VARIABLES` (keys).

    EXAMPLE: for a instance bound method calculating the density of carbonized brine:
    .. code-block:: python
        @THMSignature("pressure", "temperature", "component_fraction_in_phase",
                      "component_fraction_in_phase")
        def density(self, p: float, T: float, chi_NaCl: float, chi_CO2: float) -> float:
            return ...

    NOTE: As of now, only (global) pressure and temperature are supported as thermodynamic
    state variables in the composite submodule
    It is intended that this class converts information properly in future.
    This needs some more work.
    """

    def __init__(self, *args) -> None:
        """
        Decorator class constructor.
        Saves information about the signature and checks its validity.
        Throws an error if the physical signature argument is not known to
        :data:`~porepy.composite._composite_utils.COMPUTATIONAL_VARIABLES`.

        :param args: iterable of strings, where the strings represent
        thermal-hydraulic-mechanic physical quantities e.g., pressure, displacement,...
        :type args: Tuple[str]
        """

        valid_args = COMPUTATIONAL_VARIABLES.keys()
        checked_args = list()
        # loop over input to
        for arg in args:
            arg = str(arg)
            if arg in valid_args:
                checked_args.append(arg)
            else:
                raise ValueError("Unknown THM signature argument '%s'" % (arg))

        self._signature = tuple(checked_args)

    def __call__(self, method: FunctionType) -> Any:
        """
        Wrapper factory.
        Instances of decorator classes get called once after their instantiation.
        The decorated function object is wrapped here.

        IMPORTANT: The 'FunctionType' argument 'method' is expected to have a descriptor
        protocol implemented.
        I.e. when called, it is supposed to return a bounded method (class method f.e.).
        Consequently, the first argument of 'method' is supposed to be the instance of a class.

        :param method: function object (preferably a class method) to be wrapped
        :type method: function

        :return: returns the wrapper function object with extended meta-information
        :rtype: function
        """

        # 'wraps' assures that the wrapper has the same magic attributes (meta-information)
        # as the actual method (such as __name__ and __doc__)
        # it also changes the signature, so that the class instance,
        # whose method is being wrapped, appears as a separate, first argument
        @wraps(method)
        def wrapper(instance: Any, *args: Any, **kwds: Any) -> Any:

            output = method(instance, *args, **kwds)

            return output

        ## Adding signature meta-info to tge wrapped method
        wrapper.THM_SIGNATURE = tuple(self._signature)

        ## Adding information to docs from wrapper
        # if original method has no doc string, add generic one
        if wrapper.__doc__ is None:
            wrapper.__doc__ = "Class method '%s'." % (str(method.__name__))

        wrapper.__doc__ += "\nTHM-Signature decorated physical attribute: %s" % (
            str(self._signature)
        )

        return wrapper


class ADProperty:
    """
    Automatic-Differentiation-property.

    Decorator class for methods representing physical properties.
    The method is expected to take number type arguments and to return a number.
    See example usage below.

    It treats the method as a blackbox function and wraps the values on each cell
    (with respect to the physical signature) into an AD expression.
    The Jacobian of the blackbox function is approximated according to the AD type passed at
    instantiation.

    EXAMPLE USAGE:
    .. code-block:: python
        import porepy as pp

        class Substance:

            @AJADAttribute(pp.ad.ApproximateJacobianFunction)
            @THMSignature("pressure", "temperature")
            def density(self, p: float, T: float) -> float:
                return p/T

    With above code, the density of an instance of 'Substance' can be called using
    :class:`~porepy.numerics.ad.operators.MergedVariable` representing
    pressure and temperature.
    The return value will be an AD operator with an approximate Jacobian w.r.t to the arguments
    (see :class:`~porepy.numerics.ad.operators.ApproximateJacobianFunction`)

    If the decorated method was previously decorated with
    :class:`~porepy.composite.decorators.THMSignature`,
    the decorated property does not need input arguments anymore.
    In that case, the MergedVariables and their respective state can be accessed accordingly.

    A keyword argument 'state' can be passed to the method,
    specifying which state is requested:
    'current', 'previous', 'iterated'
    state: str = 'current' is assumed by default
    """

    def __init__(
        self,
        ad_type: TypeVar(
            "ApproximateJacobianFunction-type", pp.ad.ApproximateJacobianFunction
        ),
    ) -> None:
        """
        Decorator class constructor.
        Saves information about the requested AD type.

        :param ad_type: reference to the requested AD class (type not class instance!)
        :type ad_ype: :class:`~porepy.numerics.ad.operators.ApproximateJacobianFunction`
        """

        self._ad_type = ad_type

    def __call__(self, method: FunctionType) -> FunctionType:
        """
        Wrapper factory.
        The decorated object is wrapped here.

        IMPORTANT: The 'FunctionType' argument 'method' is expected to have a
        descriptor protocol implemented.
        I.e. when called, it is supposed to return a bounded method (class method f.e.).
        This means, the first argument of 'method' is supposed to be an
        instance method of a class.

        :param method: function object (preferably a class instance method) to be wrapped
        :type method: function

        :return: returns the wrapper function object with extended meta-information
        :rtype: function
        """

        # 'wraps' assures that the wrapper has the same magic attributes (meta-information) as
        # the actual method (such as __name__ and __doc__)
        # it also changes the signature, so that the class instance, whose method is
        # being wrapped, appears as a separate, first argument
        @wraps(method)
        def wrapper(instance: Any, *args: Any, **kwds: Any) -> Any:

            output = method(instance, *args, **kwds)

            return output

        ## Adding information to docs from wrapper
        # if original method has no doc string, add generic one
        if wrapper.__doc__ is None:
            wrapper.__doc__ = "Class method '%s'." % (str(method.__name__))

        wrapper.__doc__ += "\nDecorated AD-attribute using 'pp.ad.%s'" % (
            str(self._ad_type.__name__)
        )

        return wrapper
