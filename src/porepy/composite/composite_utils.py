"""This module contains utility functions for the composite subpackage.

It also contains some older code kept until ready for removal.

"""
from __future__ import annotations

import abc
import logging
import time
from collections.abc import Mapping
from typing import Any, Callable, Optional, Sequence

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.ad.operator_functions import NumericType

__all__ = [
    "safe_sum",
    "truncexp",
    "trunclog",
    "AdProperty",
    "COMPOSITE_LOGGER",
]


_del_log = "\r" + " " * 120 + "\r"
_logger = logging.getLogger(__name__)
_loghandler = logging.StreamHandler()
_loghandler.terminator = ""
# formatter = logging.Formatter(_del_log + '%(asctime)s : %(message)s')
_logformatter = logging.Formatter("%(del_log)s%(asctime)s : %(message)s")
_loghandler.setFormatter(_logformatter)
_logger.setLevel(logging.WARNING)
_logger.addHandler(_loghandler)


class _CompositeLogger(logging.LoggerAdapter):
    def __init__(self, logger: Any, extra: Mapping[str, object] | None = None) -> None:
        super().__init__(logger, extra)

        self._prog_msg: str = ""
        self._prog_N: int = 0
        self._prog_n: int = 0
        self._prog_log: Callable
        self._prog_start: Optional[float] = None

    def start_progress_log(self, base_msg: str, N: int, verbosity: int = 1) -> None:
        assert self._prog_start is None, "Last progress log not completed."
        # setting logging verbosity
        if verbosity == 1:
            self._prog_log = self.info
        elif verbosity >= 2:
            self._prog_log = self.debug
        else:
            self._prog_log = self.warning

        self._prog_N = N
        self._prog_n = 0
        self._prog_msg = base_msg

        msg = f"{base_msg} 0/{N} (starting) .."
        self._prog_log(msg)
        self._prog_start = time.time()

    def progress(self, msg: Optional[str] = None) -> None:
        assert self._prog_start is not None, "No progress log started"
        self._prog_n += 1
        out = f"{self._prog_msg} ({msg})" if msg is not None else f"{self._prog_msg}"
        if self._prog_n == self._prog_N:
            end = time.time()
            out += (
                f" {self._prog_n}/{self._prog_N} "
                + f"(completed; elapsed time: {end - self._prog_start} (s))\n"
            )
            self._prog_start = None
        else:
            out += f" {self._prog_n}/{self._prog_N} .."
        self._prog_log(out)

    def abort_progress(self) -> None:
        """Aborts the current progressive log"""
        out = f"ABORTED {self._prog_msg}"
        end = time.time()
        out += f"(elapsed time: {end - self._prog_start} (s))\n"
        self._prog_start = None
        self._prog_log(out)


COMPOSITE_LOGGER = _CompositeLogger(_logger, {"del_log": _del_log})


def truncexp(var):
    if isinstance(var, pp.ad.AdArray):
        trunc = var > 700
        val = np.exp(var.val, where=(~trunc))
        val[trunc] = np.exp(700)
        der = var._diagvec_mul_jac(val)
        return pp.ad.AdArray(val, der)
    else:
        trunc = var > 700
        val = np.exp(var, where=(~trunc))
        val[trunc] = np.exp(700)
        return val


def trunclog(var, eps):
    if isinstance(var, pp.ad.AdArray):
        trunc_val = np.maximum(var.val, eps)
        val = np.log(trunc_val)
        der = var._diagvec_mul_jac(1 / trunc_val)
        return pp.ad.AdArray(val, der)
    else:
        return np.log(np.maximum(var, eps))


def safe_sum(x: Sequence[Any]) -> Any:
    """Auxiliary method to safely sum the elements, without creating
    a first addition with 0 (important for AD operators to avoid overhead)."""
    if len(x) >= 1:
        sum_ = x[0]
        # need to deep copy to avoid change by reference
        if isinstance(sum_, (np.ndarray, pp.ad.AdArray)):
            sum_ = sum_.copy()
        for i in range(1, len(x)):
            sum_ = sum_ + x[i]
        return sum_
    else:
        return 0


class CompositionalSingleton(abc.ABCMeta):
    """Meta class for name- and AD-system-based singletons.

    This ensures that only a single object per AD System is instantiated with that name
    (and returned in successive instantiations).

    If name is not given as a keyword argument,
    the class name is used and the whole class becomes a singleton.

    The intended use is for classes which represent for example variables with specific
    names.
    This approach ensures a conflict-free usage of the central storage of values in the
    AD system.

    Note:
        As of now, the implications of having to use ``abc.ABCMeta`` are not clear.
        Python demands that custom meta-classes must be derived from meta classes used
        in other base classes.

        For now we demand that objects in the compositional framework are this type of
        singleton to avoid nonphysical conflicts like 2 times the same phase or
        component.
        This allows for multiple instantiations of components for phases or
        pseudo-components in various compounds,
        without having to worry about dependencies by reference and
        uniqueness of variables in a given model or AD system.

    Parameters:
        ad_system: A reference to respective AD system.
        name: ``default=None``

            Given name for an object. By default, the class name will be used.

    """

    # contains per AD system the singleton, using the given name as a unique identifier
    __ad_singletons: dict[pp.ad.EquationSystem, dict[str, object]] = dict()

    def __call__(cls, ad_system: pp.ad.EquationSystem, *args, **kwargs) -> object:
        # search for name, use class name if not given
        name = kwargs.get("name", str(cls.__name__))

        if ad_system in CompositionalSingleton.__ad_singletons:
            if name in CompositionalSingleton.__ad_singletons[ad_system]:
                # If there already is an object with this name instantiated
                # using this ad system, return it
                return CompositionalSingleton.__ad_singletons[ad_system][name]
        # prepare storage
        else:
            CompositionalSingleton.__ad_singletons.update({ad_system: dict()})

        # create a new object and store it
        new_instance = super(CompositionalSingleton, cls).__call__(
            ad_system, *args, **kwargs
        )
        CompositionalSingleton.__ad_singletons[ad_system].update({name: new_instance})
        # return new instance
        return new_instance


class AdProperty(pp.ad.Operator):
    """A leaf-operator returning an assigned value whenever it is parsed."""

    def __init__(self, name: str = "") -> None:
        super().__init__(name=name)

        self.value: NumericType = 0.0
        """The numerical value of this operator."""

    def parse(self, mdg: pp.MixedDimensionalGrid) -> NumericType:
        """Returns the value assigned to this operator."""
        return self.value


class PropertyFunction(pp.ad.AbstractFunction):
    """A function whose values and derivatives must be filled by the user.

    Values and derivatives are not assigned at instantiation.

    This function is **not** meant to be used inside nested functions, but to depend
    on genuine :class:`~porepy.numerics.ad.operators.Variable` and
    :class:`~porepy.numerics.ad.operators.MixedDimensionalVariable` instances.
    I.e., the AD representation has an identity block in its derivatives.

    The assumption of identity blocks in derivatives is used to fill in the derivative
    values in the right place of resulting AD array.

    The Jacobian of the first argument is used as reference to determine the shape.

    Parameters:
        name: A name assigned to this function.

    """

    def __init__(self, name: str) -> None:
        self.value: np.ndarray
        """Value of the filler function."""

        self.derivatives: Optional[Sequence[np.ndarray]] = None
        """Values of derivatives per dependency. Defaults to None."""

        def func(*args: Sequence[pp.ad.AdArray]) -> pp.ad.AdArray:
            """Inner function filling provided values and and derivatives."""

            if self.derivatives is None or len(args) == 0:
                return self.value
            else:
                num_args = len(args)
                assert num_args == len(
                    self.derivatives
                ), "Not enough derivatives provided."

                idx = args[0].jac.nonzero()
                shape = args[0].jac.shape()
                assert (
                    idx[0].shape == self.derivatives[0].shape
                ), "Mismatch in derivative values for argument 1."
                jac = sps.coo_matrix((self.derivatives[0], idx), shape=shape)

                for i in range(1, num_args):
                    idx = args[i].jac.nonzero()
                    assert (
                        args[i].jac.shape == shape
                    ), "Mismatch in shapes of Jacobians of arguments."
                    assert (
                        idx[0].shape == self.derivatives[i].shape
                    ), f"Mismatch in shape of derivatives for argument {i + 1}."
                    jac += sps.coo_matrix((self.derivatives[i], idx), shape=shape)

                return pp.ad.AdArray(self.value, jac.tocsr())

        super().__init__(func, name, False, True)

        self.ad_compatible = True

    def __call__(
        self, *args: Sequence[pp.ad.Variable | pp.ad.MixedDimensionalVariable]
    ) -> pp.ad.Operator:
        """Performs an input validation when assembling the operator function:
        All arguments must be instances of
        :class:`~porepy.numerics.ad.operators.Variable`.
        """
        for i, arg in enumerate(args):
            if not isinstance(arg, pp.ad.Variable):  # covers md-vars by inheritance
                raise TypeError(f"Argument {i + 1} not a variable.")
        return super().__call__(*args)
