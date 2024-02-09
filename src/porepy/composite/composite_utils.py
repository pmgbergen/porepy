"""This module contains utility functions for the composite subpackage.

It also contains some older code kept until ready for removal.

"""
from __future__ import annotations

import abc
import logging
import time
from collections.abc import Mapping
from typing import Any, Callable, Optional, Sequence, cast

import numpy as np
import scipy.sparse as sps

import porepy as pp

__all__ = [
    "safe_sum",
    "truncexp",
    "trunclog",
    "COMPOSITE_LOGGER",
    "DomainProperty",
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


class DomainProperty:
    """A representation of some dependent quantity on a set of subdomains and their
    boundary grids.

    This is meant for terms where the evaluation is done elsewhere and then stored.

    The property can depend on other operators. It is treated as a function, where the
    other operators are evaluated as children, and their values are passed to a
    place-holder function, which returns the values stored here.

    The place holder functions can be modified upon inheritance.

    Parameters:
        name: Assigned name of the property.
        subdomains: A sequence of subdomains on which the property is defined.
        boundaries: A sequence of boundary grids corresponding to ``subdomains``.
        *dependencies: Callables/ constructors for independent variables on which the
            expression depends. Their order is should be reflected in the assigned
            :meth:`derivatives`.

            They should be defined on ``subdomains`` and have boundary values on
            respective boundary grids.

    """

    def __init__(
        self,
        name: str,
        subdomains: Sequence[pp.Grid],
        boundaries: Sequence[pp.BoundaryGrid],
        *dependencies: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator],
    ) -> None:
        assert len(set(subdomains)) == len(subdomains), "Must pass unique subdomains."
        assert len(set(boundaries)) == len(boundaries), "Must pass unique subdomains."
        assert len(subdomains) > 0, "A property must be defined on a subdomain."
        assert len(boundaries) > 0, "A property must be defined on boundaries."

        self._subdomains: Sequence[pp.Grid] = subdomains
        self._boundaries: Sequence[pp.BoundaryGrid] = boundaries
        self._name: str = name
        self._dependencies = dependencies

        self._nc_subdomains: int = sum([grid.num_cells for grid in self._subdomains])
        """Number of subdomain cells."""
        self._nc_boundaries: int = sum([grid.num_cells for grid in boundaries])
        self._nd: int = len(dependencies)
        """See :meth:`nd`"""
        self._value = np.zeros(self._nc_subdomains)
        """See :meth:`value`."""
        self._derivatives = np.array(
            [np.zeros(self._nc_subdomains) for _ in range(self._nd)]
        )
        """See :meth:`derivatives`."""
        self._boundary_value = np.zeros(self._nc_boundaries)
        """See :meth:`boundary_value`"""

    def __call__(self, domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        """Call this operator on a subset of subdomains to get a property restricted
        to the passed subdomains.

        The resulting operator is a function-evaluations fetching the values and
        derivatives from the "mother object" and simply restricting them onto the target
        subdomains.

        If ``subdomains`` contains all subdomains in the md-grid passed at
        instantiation, this method returns this instance itself.

        Parameters:
            domains: A subset of domains passed at instantiation.

        Raises:
            AssertionError: If unknown, or no, or mixed types of domains are passed.

        """
        # sanity check
        assert len(domains) > 0, "Cannot access property without defining domain."
        assert all(
            [isinstance(d, domains[0]) for d in domains]
        ), "Cannot access property with mixed domain types."

        if isinstance(domains[0], pp.BoundaryGrid):
            assert all(
                [d in self._boundaries for d in domains]
            ), "Property accessed on unknown boundary."

            children = [child(list(self._boundaries)) for child in self._dependencies]
            op = pp.ad.Function(
                func=self.boundary_function(), name=f"bc_values_{self._name}"
            )(*children)

            if len(domains) == len(self._boundaries):  # return without restriction
                return op
            else:  # restrict to (by logic) smaller subdomain
                raise NotImplementedError(
                    "Restriction of properties on subset of boundaries not implemented."
                )

        elif isinstance(domains[0], pp.Grid):
            assert all(
                [d in self._subdomains for d in domains]
            ), "Property accessed on unknown subdomains."

            children = [child(list(self._subdomains)) for child in self._dependencies]
            op = pp.ad.Function(func=self.subdomain_function(), name=self._name)(
                *children
            )

            if len(domains) == len(self._subdomains):  # return without restriction
                return op
            else:  # restrict to (by logic) smaller subdomain
                restriction = pp.ad.SubdomainProjections(
                    list(self._subdomains)
                ).cell_restriction(list(domains))
                op = restriction @ op
                op.set_name(f"domain_restricted_{self._name}")
                return op
        else:
            raise ValueError(
                f"Properties are not defined on grids of type {type(domains[0])}"
            )

    def subdomain_function(
        self,
    ) -> Callable[[*pp.ad.AdArray], pp.ad.AdArray | np.ndarray]:
        """Returns a function which represents the property on the subdomains.

        It is consistent with the derivatives with the children passed at instantiation.

        For values and derivatives, the assigned :meth:`value` and :meth:`derivatives`
        are used.

        """

        def func(*args: pp.ad.AdArray) -> pp.ad.AdArray | np.ndarray:
            """Inner function filling provided values and and derivatives."""
            n = len(args)
            if n == 0 and self.nd == 0:  # case with no dependency
                self.value
            elif n == self.nd:  # case with dependency
                idx = cast(tuple[np.ndarray, np.ndarray], args[0].jac.nonzero())
                shape = args[0].jac.shape()
                # number of derivatives per dependency must be equal to the total number
                # of values
                assert (
                    idx[0].shape == self.value.shape
                ), "Mismatch in shape of derivatives for arg 1."
                assert (
                    idx[0].shape == self.derivatives[0].shape
                ), "Mismatch in shape of provided derivatives for arg 1."
                jac = sps.coo_matrix((self.derivatives[0], idx), shape=shape)

                for i in range(1, len(args)):
                    idx = cast(tuple[np.ndarray, np.ndarray], args[i].jac.nonzero())
                    # checks consistency with number of values
                    assert (
                        idx[0].shape == self.value.shape
                    ), f"Mismatch in shape of derivatives for arg {i + 1}"
                    assert (
                        idx[0].shape == self.derivatives[i].shape
                    ), f"Mismatch in shape of provided derivatives for arg {i + 1}."
                    jac += sps.coo_matrix((self.derivatives[i], idx), shape=shape)

                return pp.ad.AdArray(self.value, jac.tocsr())
            else:
                raise ValueError(
                    f"Subdomain function of property {self._name} requires {self._nd}"
                    + f" argument, {n} given."
                )

        return func

    def boundary_function(self) -> Callable[[*pp.ad.AdArray | np.ndarray], np.ndarray]:
        """Returns a function which represents the property on the boundaries.

        As of now, properties on boundaries pass return only the boundary value, no
        derivatives.

        To stored values are accessed using :meth:`boundary_value`

        """

        def func(*args: pp.ad.AdArray) -> np.ndarray:
            """Inner function returning the stored boundary values."""
            return self.boundary_value

        return func

    @property
    def nd(self) -> int:
        """Number of first order dependencies of this operator.

        Given by the number of unique variables in the operator tree.

        This determines the number of required :meth:`derivatives`.

        """
        return self._nd

    @property
    def value(self) -> np.ndarray:
        """The value of this property given by an array with values in each subdomain
        cell.

        Parameters:
            val: ``shape=(num_cells,)``

                A new value to be set.

        Raises:
            ValueError: If the size of the value mismatches what is expected.

        """
        return self._value

    @value.setter
    def value(self, val) -> None:
        self._value[:] = val  # let numpy handle the broadcasting errors

    @property
    def derivatives(self) -> Sequence[np.ndarray]:
        """The derivatives of this property, w.r.t. to its first-order dependencies.

        This is a sequence of length :meth:`nd` where each element is an array with
        ``shape=(num_cells,)``.

        Important:
            The order of derivatives should reflect the order of dependencies
            passed at instantiation.

        Parameters:
            val: The new derivatives values.

        Raises:
            ValueError: If an insufficient number of derivatives is passed.

        """
        self._derivatives

    @derivatives.setter
    def derivatives(self, val: Sequence[np.ndarray]) -> None:
        if len(val) != self._nd:
            raise ValueError(f"{len(val)} derivatives provided, {self._nd} required.")
        for i in range(self._nd):
            self._derivatives[i, :] = val[i]  # stored as np array to ensure proper size

    @property
    def boundary_value(self) -> np.ndarray:
        """The value of this property on the boundary, given by an array with values in
        each boundary grid cell.

        Parameters:
            val: ``shape=(num_boundary_cells,)``

                A new value to be set.

        Raises:
            ValueError: If the size of the value mismatches what is expected.

        """
        return self._value

    @boundary_value.setter
    def boundary_value(self, val) -> None:
        self._boundary_value[:] = val  # let numpy handle the broadcasting errors
