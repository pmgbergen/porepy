"""This module contains utility functions for the composite subpackage.

It also contains some older code kept until ready for removal.

"""
from __future__ import annotations

import abc
import logging
import time
from collections.abc import Mapping
from typing import Any, Callable, Optional, Sequence, Tuple, cast

import numpy as np
import scipy.sparse as sps

import porepy as pp

__all__ = [
    "safe_sum",
    "truncexp",
    "trunclog",
    "COMPOSITE_LOGGER",
    "SecondaryExpression",
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


class SecondaryExpression:
    """A **cell-wise** representation of some dependent quantity on a set of domains
    boundary grids.

    This is meant for terms where the evaluation is done elsewhere and then stored.

    **On subdomains:**

    The operator can depend on other operators. It is treated as a function, where the
    other operators are evaluated as children, and their values are passed to a
    place-holder function, which returns the values stored here (see operator functions
    in PorePy's Ad).
    The place holder functions can be modified upon inheritance.

    **On boundary grids:**

    The class creates a
    :class:`~porepy.numerics.ad.opeators.TimeDependentDenseArray` using its given
    name and the boundar grids passed to the call.
    Boundary values must hence be updated like any other term in the model framework.
    They are not stored in the class, but in the data dictionaries using the
    expression's name as key

    When calling this expression on a (sub-) set of domains on which it is defined,
    it creates AD compatible representations of this expression, using the operator
    function framework on subdomains and time-dependent dense arrays on boundaries.

    Note:
        Future work might add support for interfaces.
        The restriction from the total set of domains to a subset when calling this
        expression can be optimized.

    Parameters:
        name: Assigned name of the expression.
        domains: A sequence of subdomains or interfaces on which the expression is
            defined.
        boundaries: ``default=None``

            A sequence of boundary grids on which the secondary expression is defined.
        *dependencies: Callables/ constructors for independent variables on which the
            expression depends. Their order is reflected in the assigned
            :meth:`derivatives`.

            They should be defined on ``domains`` and have boundary values on
            respective boundary grids stored in the boundary grid data dictionaries.

    """

    def __init__(
        self,
        name: str,
        domains: Sequence[pp.Grid] | Sequence[pp.MortarGrid],
        boundaries: Optional[Sequence[pp.BoundaryGrid]] = None,
        *dependencies: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator],
    ) -> None:
        # some checks on proper usage
        assert len(domains) > 0, "A property must be defined on at least one domain."
        assert all(
            [isinstance(d, type(domains[0])) for d in domains]
        ), "Secondary expression cannot be defined on mixed types of domains."

        self._domains: Sequence[pp.Grid] | Sequence[pp.MortarGrid] = domains

        self._boundaries: Optional[Sequence[pp.BoundaryGrid]]
        if boundaries is not None:
            assert all(
                [isinstance(d, pp.BoundaryGrid) for d in boundaries]
            ), "Expexting only boundary grids for argument `boundaries`."
            self._boundaries = boundaries
        else:
            self._boundaries = None

        self._name: str = name
        """Name passed at instantiation. Used to name resulting operators."""
        self._dependencies = dependencies
        """Sequence of callable first order dependencies. Called when constructing
        operators on domains."""

        self._ndep: int = len(dependencies)
        """See :meth:`nd`"""
        self._values_on_domains: dict[pp.Grid | pp.MortarGrid, np.ndarray] = dict(
            [(d, np.zeros(d.num_cells)) for d in self._domains]
        )
        """Values stored per single domain on which the expression is defined."""
        self._derivatives_on_domains: dict[
            pp.Grid | pp.MortarGrid, Sequence[np.ndarray]
        ] = dict(
            [
                (d, np.array([np.zeros(d.num_cells) for _ in range(self._ndep)]))
                for d in self._domains
            ]
        )
        """Values of derivatives stored per single domain on which the expression is
        defined."""

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
        assert len(domains) > 0, "Cannot call expression without defining domain."
        assert all(
            [isinstance(d, domains[0]) for d in domains]
        ), "Cannot call expresion with mixed domain types."

        if isinstance(domains[0], pp.BoundaryGrid):
            assert all(
                [d in self._boundaries for d in domains]
            ), "Property accessed on unknown boundary."

            return pp.ad.TimeDependentDenseArray(self._name, domains)

        elif isinstance(domains[0], pp.Grid):
            assert all(
                [d in self._domains for d in domains]
            ), "Property accessed on unknown subdomains."

            children = [child(domains) for child in self._dependencies]
            op = pp.ad.Function(func=self.subdomain_function(domains), name=self._name)(
                *children
            )

            if len(domains) < len(self._domains):  # restrition (by logic)
                restriction = pp.ad.SubdomainProjections(
                    list(self._domains)
                ).cell_restriction(list(domains))
                op = restriction @ op
                op.set_name(f"domain_restricted_{self._name}")
            return op
        else:
            raise ValueError(
                f"Properties are not defined on grids of type {type(domains[0])}"
            )

    def subdomain_function(
        self, domains: Sequence[pp.Grid] | Sequence[pp.MortarGrid]
    ) -> Callable[[Tuple[pp.ad.AdArray, ...]], pp.ad.AdArray | np.ndarray]:
        """Returns a function which represents the expression on the ``domains``.

        The function takes numerical values of the dependencies passed at instantiation
        and returns the stored values on the requested ``domains``.

        This includes the derivatives in Ad form.

        Parameters:
            domains: A sequence of subdomains or interfaces, on which the numerical
                function should be defined.

        Raises:
            ValueError: If ``domains`` contains grid not passed at instantiation.

        """
        if not all([d in self._domains for d in domains]):
            raise ValueError("Function requested on unknown domains.")

        def func(*args: pp.ad.AdArray) -> pp.ad.AdArray | np.ndarray:
            """Inner function filling provided values and and derivatives."""
            n = len(args)
            value = np.hstack([self._values_on_domains[d] for d in domains])
            if n == 0 and self.ndep == 0:  # case with no dependency
                return value
            elif n == self.ndep:  # case with dependency
                idx = cast(tuple[np.ndarray, np.ndarray], args[0].jac.nonzero())
                shape = args[0].jac.shape()

                # derivative values for first dependency
                d_0 = np.hstack([self._derivatives_on_domains[d][0] for d in domains])
                # number of derivatives per dependency must be equal to the total number
                # of values
                assert (
                    idx[0].shape == value.shape
                ), "Mismatch in shape of derivatives for arg 1."
                assert (
                    idx[0].shape == d_0.shape
                ), "Mismatch in shape of provided derivatives for arg 1."
                jac = sps.coo_matrix((d_0, idx), shape=shape)

                for i in range(1, len(args)):
                    idx = cast(tuple[np.ndarray, np.ndarray], args[i].jac.nonzero())
                    # derivatives w.r.t. to i-th dependency
                    d_i = np.hstack(
                        [self._derivatives_on_domains[d][i] for d in domains]
                    )
                    # checks consistency with number of values
                    assert (
                        idx[0].shape == value.shape
                    ), f"Mismatch in shape of derivatives for arg {i + 1}"
                    assert (
                        idx[0].shape == d_i.shape
                    ), f"Mismatch in shape of provided derivatives for arg {i + 1}."
                    jac += sps.coo_matrix((d_i, idx), shape=shape)

                return pp.ad.AdArray(value, jac.tocsr())
            else:
                raise ValueError(
                    f"Subdomain function of expression {self._name} requires"
                    + f" {self.ndep} arguments, {n} given."
                )

        return func

    @property
    def ndep(self) -> int:
        """Number of first order dependencies of this operator.

        Given by the number of unique variables in the operator tree.

        This determines the number of required :meth:`derivatives`.

        """
        return self._ndep

    @property
    def value(self) -> np.ndarray:
        """The global value of this expression given by an array with values in each
        subdomain cell.

        The setter is for convenience to set the values on all domains of definition.

        Parameters:
            val: ``shape=(num_cells,)``

                A new value to be set.

        Raises:
            ValueError: If the size of the value mismatches what is expected.

        """
        return np.hstack([v for _, v in self._values_on_domains.items()])

    @value.setter
    def value(self, val: np.ndarray) -> None:
        idx = 0
        for d in self._domains:
            nc_d = d.num_cells
            self.set_value_on_domain(d, val[idx : idx + nc_d])
            idx += nc_d

    def set_value_on_domain(
        self, domain: pp.Grid | pp.MortarGrid, value: np.ndarray
    ) -> None:
        """Set the value of the secondary expression on an individual domain.

        Parameters:
            domain: One of the grids passed at instantiation.
            value: Cell-wise value of the expression on ``domain``.

        Raises:
            ValueError: If ``domain`` not among the domains of defintion
            ValueError: If ``value.shape != (domain.num_cells)``.

        """
        if domain not in self._domains:
            raise ValueError(f"Unknown domain {domain}")

        shape = (domain.num_cells,)
        if value.shape != shape:
            raise ValueError(f"Values must be of shape {shape}, got {value.shape}.")

        self._values_on_domains[domain] = value

    @property
    def derivatives(self) -> Sequence[np.ndarray]:
        """The global derivatives of this expression, w.r.t. to its first-order
        dependencies on all domains of definition.

        This is a sequence of length :meth:`ndep` where each element is an array with
        ``shape=(num_cells,)``, with ``num_cells`` being the total number of cells.

        The setter is for convenience to set the derivative values on all domains of
        definition.

        Important:
            The order of derivatives should reflect the order of dependencies
            passed at instantiation.

        Parameters:
            val: The new derivatives values.

        Raises:
            ValueError: If an insufficient number of derivatives is passed.

        """
        return np.array(
            [
                np.hstack([v[i] for _, v in self._derivatives_on_domains.items()])
                for i in range(self.ndep)
            ]
        )

    @derivatives.setter
    def derivatives(self, val: Sequence[np.ndarray]) -> None:
        if len(val) != self.ndep:
            raise ValueError(f"{len(val)} derivatives provided, {self.ndep} required.")

        for d in self._domains:
            idx = 0
            nc_d = d.num_cells
            d_vals = []
            for i in range(self.ndep):
                d_vals.append(val[i][idx : idx + nc_d])
            idx += nc_d
            self.set_derivatives_on_domain(d, np.array(d_vals))

    def set_derivatives_on_domain(
        self, domain: pp.Grid | pp.MortarGrid, value: Sequence[np.ndarray]
    ) -> None:
        """Set the value the derivatives w.r.t. the first-order dependencies on an
        individual domain.

        Analogous to :meth:`set_value_on_domain`.

        Parameters:
            domain: One of the grids passed at instantiation.
            value: ``shape=(ndep, domain.num_cells)``

                The derivatives values, row-wise for dependencies, column-wise for cells
                in ``domain``.

        Raises:
            ValueError: If ``domain`` not among the domains of defintion
            ValueError: If ``value.shape != (ndep, domain.num_cells)``.

        """
        if domain not in self._domains:
            raise ValueError(f"Unknown domain {domain}")

        # convert to array for simple shape comparison
        # this raises an error if different amoung of values per dependency is given
        shape = (self.ndep, domain.num_cells)
        value_ = np.array(value) if not isinstance(value, np.ndarray) else value
        if value_.shape != shape:
            raise ValueError(f"Values must be of shape {shape}, got {value_.shape}.")

        self._derivatives_on_domains[domain] = value_
