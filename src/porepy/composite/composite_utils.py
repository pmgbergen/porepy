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
    "SecondaryOperator",
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


class SecondaryOperator(pp.ad.Operator):
    """Operator representing a :class:`SecondaryExpression` in AD operator form on
    some domains.

    Not meant to be instantiated directly, only by calling the mother class
    :class:`SecondaryExpression`.

    This operator fetches upon evaluation the values stored under its name, as well as
    the derivatives.

    It supports the notion of previous time step and iterate values, which are
    mutually exclusive. I.e., it cannot represent a previous time step and iterate at
    the same time.
    But previous time steps and iterate can go further backwords in their respective
    direction.

    Note:
        As of now, the mixed-dimensional variables do not support a greater timestepping
        and iterate depth of 1.

    Parameters:
        name: Name of the called :class:`SecondaryExpression`.
        domains: Arguments to its call.
        timestepping_depth: Assigned by the called expression
        timestep_index: Assigned as 0 by the expression, increased by
            :meth:`previous_timestep`.
        iterate_depth: Assigned by the called expression
        iterate_index: Assigned as 0 by the expression, increased by
            :meth:`previous_iteration`.
        *children: The first-order dependencies of the called
            :class:`SecondaryExpression` in AD form (defined on the same ``domains``).

    """

    def __init__(
        self,
        name: str,
        domains: Sequence[pp.Grid] | Sequence[pp.MortarGrid],
        timestepping_depth: int,
        timestep_index: int,
        iterate_depth: int,
        iterate_index: int,
        *children: pp.ad.MixedDimensionalVariable,
    ) -> None:
        super().__init__(name, domains, pp.ad.Operator.Operations.evaluate, children)

        assert (
            timestep_index <= timestepping_depth
        ), "Assigned time step index must be smaller or equal the depth."
        assert (
            iterate_index <= iterate_depth
        ), "Assigned iterate index must be smaller or equal the depth."

        self._time_depth: int = timestepping_depth
        """Depth of stored values in time, with 0 denoting the current time, positive
        numbers going backwards in time."""
        self._iterate_depth: int = iterate_depth
        """Depth of stored iterate values, with 0 denoting the current iteration,
        positive numbers denoting previous iterations."""

        self._time_index: int = timestep_index
        """Time index assigned to this instance. Capped by :attr:`_time_depth`.
        """
        self._iterate_index: int = iterate_index
        """Time index assigned to this instance. Capped by :attr:`_time_depth`.
        """

        self.original_operator: SecondaryOperator
        """A reference to the original operator (zero-th time and iterate index).

        Only instances with either :meth:`prev_time` or :meth:`prev_iter` being True
        have this attribute.

        """

        self.func: Callable[[Tuple[pp.ad.AdArray, ...]], pp.ad.AdArray | np.ndarray]
        """The function accessing the data stored for this secondary expression.

        Assigned by :class:`Secondary Expression` upon creation of this operator."""

        self.ad_compatible: bool = True
        """To trigger the same parsing as for the regular AD function."""

    def __repr__(self) -> str:
        return (
            f"Secondary operator with name {self.name}"
            + f" at time index {self.timestep_index}"
            + f" and iterate index {self.iterate_index}\n"
            + f"Defined on {len(self._domains)} {self._domain_type}.\n"
            + f"Dependent on {len(self.children)} independent operators.\n"
        )

    @property
    def timestep_index(self) -> int:
        """Returns the time step index this instance represents."""
        return self._time_index

    @property
    def prev_time(self) -> bool:
        """True, if the operator represents a previous time-step."""
        return True if self._time_index > 0 else False

    @property
    def iterate_index(self) -> int:
        """Returns the time step index this instance represents."""
        return self._time_index

    @property
    def prev_iter(self) -> bool:
        """True, if the operator represents a previous iterate."""
        return True if self._iterate_index > 0 else False

    @property
    def is_original_operator(self) -> bool:
        """Returns True if this is the operator at the zero-th time step and iterate
        index."""
        # TODO this check should be performed using the index values.
        if hasattr(self, "original_operator"):
            return False
        else:
            return True

    def previous_timestep(self) -> SecondaryOperator:
        """Return a representation of this operator on the previous time step.

        Raises:
            ValueError: If the variable is a representation of the previous iteration,
                previously set by :meth:`~previous_iteration`.
            ValueError: If the timestepping depth was reached.

        Returns:
            A representation of this operator at one timestep backwards.

        """
        if self.prev_iter:
            raise ValueError(
                "Cannot create a variable both on the previous time step and "
                "previous iteration."
            )
        if self.timestep_index == self._time_depth:
            raise ValueError(
                f"Cannot go further back than {self._time_depth} steps in time."
            )

        # there is such a traversion implemented in the base class, calling the
        # variables at the previous time step.
        prev_time_children = tuple(pp.ad.Operator.previous_timestep(self).children)

        op = SecondaryOperator(
            self.name,
            self.domains,
            timestepping_depth=self._time_depth,
            timestep_index=self.timestep_index + 1,  # increase time step index
            iterate_depth=self._iterate_depth,
            iterate_index=self.iterate_index,
            *prev_time_children,
        )

        # keeping track to the very first one
        if self.is_original_operator:
            op.original_operator = self
        else:
            op.original_operator = self.original_operator

        return op

    def previous_iteration(self) -> SecondaryOperator:
        """Return a representation of this operator at the previous iteration.

        Raises:
            ValueError: If the variable is a representation of the previous time step,
                previously set by :meth:`~previous_timestep`.
            ValueError: If the iterate depth was reached.

        Returns:
            A representation of this operator at one iteration before.

        """
        if self.prev_time:
            raise ValueError(
                "Cannot create a variable both on the previous time step and "
                "previous iteration."
            )
        if self.iterate_index == self._iterate_depth:
            raise ValueError(
                f"Cannot go further back than {self._iterate_depth} iterations."
            )

        def _traverse_tree(op: pp.ad.Operator) -> pp.ad.Operator:
            """Helper function which traverses an operator tree by recursion and
            gets the children on previous iterations.

            Analogous to what is found in the base operator class. Room for code
            recycling. TODO."""

            children = op.children

            if len(children) == 0:
                if isinstance(op, (pp.ad.Variable, pp.ad.MixedDimensionalVariable)):
                    return op.previous_iteration()
                else:
                    return op
            else:

                new_children: list[pp.ad.Operator] = list()
                for ci, child in enumerate(children):
                    # Recursive call to fix the subtree.
                    new_children.append(_traverse_tree(child))

                # Use the same lists of domains as in the old operator.
                domains = op.domains

                # Create new operator from the tree.
                new_op = pp.ad.Operator(
                    name=op.name,
                    domains=domains,
                    operation=op.operation,
                    children=new_children,
                )
                return new_op

        prev_iter_children = tuple(_traverse_tree(self).children)

        op = SecondaryOperator(
            self.name,
            self.domains,
            timestepping_depth=self._time_depth,
            timestep_index=self.timestep_index,
            iterate_depth=self._iterate_depth,
            iterate_index=self.iterate_index + 1,  # increase iterate index
            *prev_iter_children,
        )

        # keeping track to the very first one
        if hasattr(self, "original_operator"):
            op.original_operator = self.original_operator
        # if the original operator itself was accessed
        else:
            op.original_operator = self

        return op


class SecondaryExpression:
    """A representation of some dependent quantity in the PorePy modelling framework.

    This is meant for terms where the evaluation is done elsewhere and then stored.

    The expression can be called using grids to obtain an AD-compatible representation.

    **On the boundary:**

    The class creates a
    :class:`~porepy.numerics.ad.opeators.TimeDependentDenseArray` using its given
    name and the boundar grids passed to the call.
    Boundary values can hence be updated like any other term in the model framework.
    But they can also be updated using :meth:`boundary_values` for convenience.

    The secondary expression has no derivatives and no iterate values on the boundary.

    **On subdomains:**

    If the secondary expression has no ``dependencies``, it creates a time-dependent
    dense array as well (analogous to the boundary case).

    Otherwise it creates a :class:`SecondaryOperator`, which represents the data managed
    by this class on respective subdomains.
    The operator can depend on instances of
    :class:`~porepy.numerics.ad.operators.MixedDimensionalVariable`.

    The secondary operator is essentially like a
    :class:`~porepy.numerics.ad.operator_functions.Function`. Upon evaluation it fetches
    the stored values and the derivative values.
    The derivative values are inserted into the Jacobians of the first-order
    dependencies (identity blocks).

    The secondary operator suppoerts the notion of previous timesteps and iterate
    values. Updates of respective values are handled by this mother class.

    **On interfaces:**

    Analogous to the case on subdomains.

    Note:
        As of now:

        1. The ``dependencies`` are assumed to be of first order. I.e., independent
           variables. Nested dependencies are not supported.
        2. It also supports only ``dependencies`` which have cell-wise a single DOF.

        This might change if a use case arises.

    Note:
        This class prepares the storage in the data dictionaries.
        For expressions without dependencies, it stores data as usual.
        For Expressions with dependencies, it stores 2-tuples (value-diff pairs).

        The dictionaries are populated with None to support the user in keeping track
        which values have been stored.

    Note:
        Iterate and timestep values can be progressed on individual grids using
        various ``progress_*`` methods.

        The properties defined by the class are for convenience to do it on all grids.

        Though, for subdomains and interfaces the properties progress the iterate.
        Time step values have to be progressed explicitely.

        For boundaries, the properties progress the time step value, since there are no
        iterate values on boundaries.

    Parameters:
        name: Assigned name of the expression. Used to name operators and to store
            values in the data dictionaries
        mdg: The mixed-dimensional grid on which the expression is defined.
        *dependencies: Callables/ constructors for independent variables on which the
            expression depends. The order passed here is reflected in the order of
            stored derivative values.

            When calling the secondary expression on some grids, it is expected that the
            dependencies are defined there.
        timestepping_depth: ``default=1``

            Depth of storage of values backwards in time. Default to 1 (implicit Euler).
        iterate_depth: ``default=1``

            Depth of storage of iterate values.

    """

    def __init__(
        self,
        name: str,
        mdg: pp.MixedDimensionalGrid,
        *dependencies: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator],
        timestepping_depth: int = 1,
        iterate_depth: int = 1,
    ) -> None:
        self._dependencies: Tuple[
            Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator], ...
        ] = dependencies
        """Sequence of callable first order dependencies. Called when constructing
        operators on domains."""

        self._ndep: int = len(dependencies)
        """See :meth:`nd`"""

        self._time_depth: int = timestepping_depth
        """Depth of stored values in time, with 0 denoting the current time, positive
        numbers going backwards in time."""
        self._iterate_depth: int = iterate_depth
        """Depth of stored iterate values, with 0 denoting the current iteration,
        positive numbers denoting previous iterations."""

        self.mdg: pp.MixedDimensionalGrid = mdg
        """Mixed-dimensional domain passed at instantiation."""
        self.name: str = name
        """Name passed at instantiation. Used to name resulting operators and to store
        values in data dictionaries."""

        self._set_up_dictionaries()

    def __call__(self, domains: pp.GridLikeSequence) -> pp.ad.Operator:
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
            op = pp.ad.TimeDependentDenseArray(self.name, domains)
        elif isinstance(domains[0], (pp.Grid, pp.MortarGrid)):
            if self.num_dependencies == 0:
                op = pp.ad.TimeDependentDenseArray(self.name, domains)
            else:
                children = [child(domains) for child in self._dependencies]

                # always start with the current time and iterate index
                op = SecondaryOperator(
                    self.name,
                    domains,
                    timestepping_depth=self._time_depth,
                    timestep_index=0,
                    iterate_depth=self._iterate_depth,
                    iterate_index=0,
                    *tuple(children),
                )

                # assign the function which extracts the data
                op.func = self._assign_function(op)

        else:
            raise ValueError(f"Unknown grid type {type(domains[0])}")

        return op

    def _set_up_dictionaries(self) -> None:
        """Helper method to populate the data dictionaries in the md-grid and prepare
        the data storage for this expression, when creating an instance.

        Parameters:
            grid_type: A string containing ``'s', 'i', 'b'`` or any combination,
                indicating if the Expression is defined on Subdomains, Interfaces or
                Boundaries.

                Determins which data dictionaries are populated.

        """
        # On subdomains and interfaces, both time stepping and iterate values are
        # available

        # If the number of dependencies is not zero, this pepares the storage of a
        # 2-tuple (value-diff pair), instead of an empty array.
        for _, data in self.mdg.subdomains(True):
            if pp.TIME_STEP_SOLUTIONS not in data:
                data[pp.TIME_STEP_SOLUTIONS] = {}
            if self.name not in data[pp.TIME_STEP_SOLUTIONS]:
                data[pp.TIME_STEP_SOLUTIONS][self.name] = {}

            # prepare time step entries
            # arrays if no dependencies, 2 tuple of value-derivative pairs if dependent
            if self.num_dependencies == 0:
                for t in range(self._time_depth + 1):
                    data[pp.TIME_STEP_SOLUTIONS][self.name][t] = None
            else:
                for t in range(self._time_depth + 1):
                    data[pp.TIME_STEP_SOLUTIONS][self.name][t] = (None, None)

            # Do the same for iterate solutions
            if pp.ITERATE_SOLUTIONS not in data:
                data[pp.ITERATE_SOLUTIONS] = {}
            if self.name not in data[pp.ITERATE_SOLUTIONS]:
                data[pp.ITERATE_SOLUTIONS][self.name] = {}

            # prepare time step entries
            # arrays if no dependencies, 2 tuple of value-derivative pairs if dependent
            if self.num_dependencies == 0:
                for i in range(self._iterate_depth + 1):
                    data[pp.ITERATE_SOLUTIONS][self.name][i] = None
            else:
                for i in range(self._iterate_depth + 1):
                    data[pp.ITERATE_SOLUTIONS][self.name][i] = (None, None)

        for _, data in self.mdg.interfaces(True):
            if pp.TIME_STEP_SOLUTIONS not in data:
                data[pp.TIME_STEP_SOLUTIONS] = {}
            if self.name not in data[pp.TIME_STEP_SOLUTIONS]:
                data[pp.TIME_STEP_SOLUTIONS][self.name] = {}

            # prepare time step entries
            # arrays if no dependencies, 2 tuple of value-derivative pairs if dependent
            if self.num_dependencies == 0:
                for t in range(self._time_depth + 1):
                    data[pp.TIME_STEP_SOLUTIONS][self.name][t] = None
            else:
                for t in range(self._time_depth + 1):
                    data[pp.TIME_STEP_SOLUTIONS][self.name][t] = (None, None)

            # Do the same for iterate solutions
            if pp.ITERATE_SOLUTIONS not in data:
                data[pp.ITERATE_SOLUTIONS] = {}
            if self.name not in data[pp.ITERATE_SOLUTIONS]:
                data[pp.ITERATE_SOLUTIONS][self.name] = {}

            # prepare time step entries
            # arrays if no dependencies, 2 tuple of value-derivative pairs if dependent
            if self.num_dependencies == 0:
                for i in range(self._iterate_depth + 1):
                    data[pp.ITERATE_SOLUTIONS][self.name][i] = None
            else:
                for i in range(self._iterate_depth + 1):
                    data[pp.ITERATE_SOLUTIONS][self.name][i] = (None, None)

        # on boundaries we have only values in time, and no derivatives
        for _, data in self.mdg.boundaries(True):
            if pp.TIME_STEP_SOLUTIONS not in data:
                data[pp.TIME_STEP_SOLUTIONS] = {}
            if self.name not in data[pp.TIME_STEP_SOLUTIONS]:
                data[pp.TIME_STEP_SOLUTIONS][self.name] = {}

            for t in range(self._time_depth + 1):
                data[pp.TIME_STEP_SOLUTIONS][self.name][t] = None

    def _get_op_data(
        self, op: SecondaryOperator, g: pp.Grid | pp.MortarGrid
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Helper function to extract the data stored for this expression on a grid.

        Which time step index or iterate index is accessed is defined by the
        secondary operator ``op``.

        """
        if isinstance(g, pp.MortarGrid):
            d = self.mdg.interface_data(g)
        elif isinstance(g, pp.Grid):
            d = self.mdg.subdomain_data(g)
        else:  # should not happen
            raise NotImplementedError("Unclear storage access")
        # if op is at previous time, get those values
        if op.prev_time:
            return d[pp.TIME_STEP_SOLUTIONS][self.name][op.timestep_index]
        # otherwise get the iterate values
        # iterate_index is always assigned (0 if not prev_iter)
        else:
            return d[pp.ITERATE_SOLUTIONS][self.name][op.iterate_index]

    def _assign_function(
        self, op: SecondaryOperator
    ) -> Callable[[Tuple[pp.ad.AdArray, ...]], pp.ad.AdArray | np.ndarray]:
        """Creates the function which provides an evaluation of the secondary operator
        ``op`` created by a call to this instance with ``op.domains``.

        The function takes the evaluated Ad arrays of the first-order dependencies
        (if any) and returns the values stored and managed by this instance.

        Parameters:
            op: Secondary operator created on the ``domains``.

        Returns:
            The callable to be assigned to :attr:`SecondaryOperator.func`

        """

        nc = sum([g.num_cells for g in op.domains])

        # In this case the function is easy and returns only the stored arrays
        if self.num_dependencies == 0:

            def func(*args) -> np.ndarray:
                assert len(args) == 0, f"Evaluation of {self.name} expects 0 args."

                # Extracting and stacking vals
                # We expect only arrays stored as data
                vals: list[np.ndarray] = []
                for g in op.domains:
                    # cast to the type we know will come
                    val = cast(np.ndarray, self._get_op_data(op, g))
                    vals.append(val)
                value = np.hstack(vals)

                # some sanity check to ensure the data is appropriatly shaped
                assert value.shape == (
                    nc,
                ), f"Operator {op} requires {nc} values stored in domains {op.domains}"
                return value

        # if it has dependencies, the data stored are 2-tuples with value-diff pairs
        # and the function returns an AdArray
        else:

            def func(*args: pp.ad.AdArray) -> pp.ad.AdArray:
                assert (
                    len(args) == self.num_dependencies
                ), f"Evaluation of {self.name} expects {self.num_dependencies} args."

                vals: list[np.ndarray] = []
                diffs: list[np.ndarray] = []
                for g in op.domains:
                    # cast to the type we know will come
                    val = cast(tuple[np.ndarray, np.ndarray], self._get_op_data(op, g))
                    vals.append(val[0])
                    diffs.append(val[1])

                # values per domain per cell
                value = np.hstack(vals)
                # derivatives, row-wise dependencies, column-wise per domain per cell
                derivatives = np.hstack(diffs)

                # sanity checks
                assert value.shape == (
                    nc,
                ), f"Operator {op} requires {nc} values stored in domains {op.domains}"
                assert derivatives.shape == (self.num_dependencies, nc), (
                    f"Operator {op} requires {nc} derivative values per dependency"
                    + f" stored in domains {op.domains}."
                )

                # The Jacobian is assembled additively, starting with the first
                # dependency. We assume the same shape for all args and use the Jacobian
                # to check if enough values where stored for the expression
                idx = cast(tuple[np.ndarray, np.ndarray], args[0].jac.nonzero())
                shape = args[0].jac.shape()

                # number of provided values must be consistent with the DOFs of the
                # dependencies (and the size of their identity blocks)
                assert (
                    idx[0].shape == value.shape
                ), "Mismatch in shape of derivatives for arg 1."
                assert (
                    idx[0].shape == derivatives[0].shape
                ), "Mismatch in shape of provided derivatives for arg 1."
                jac = sps.coo_matrix((derivatives[0], idx), shape=shape)

                # Do the same with the other dependencies and add respective blocks
                for i in range(1, len(args)):
                    idx = cast(tuple[np.ndarray, np.ndarray], args[i].jac.nonzero())
                    assert (
                        idx[0].shape == value.shape
                    ), f"Mismatch in shape of derivatives for arg {i + 1}"
                    assert (
                        idx[0].shape == derivatives[i].shape
                    ), f"Mismatch in shape of provided derivatives for arg {i + 1}."
                    jac += sps.coo_matrix((derivatives[i], idx), shape=shape)

                return pp.ad.AdArray(value, jac.tocsr())

        return func

    def _data_of(self, grid: pp.GridLike) -> dict:
        """Convenience function to get the data dictionary of any grid."""
        if isinstance(grid, pp.Grid):
            data = self.mdg.subdomain_data(grid)
        elif isinstance(grid, pp.MortarGrid):
            data = self.mdg.interface_data(grid)
        elif isinstance(grid, pp.BoundaryGrid):
            data = self.mdg.boundary_grid_data(grid)
        else:
            raise ValueError(f"Unknown grid type {type(grid)}")
        return data

    def progress_value_in_time_on_boundaries(self, new_values: np.ndarray) -> None:
        """Convenience function to progress in time on all boundary grids.

        ``new_values`` must be of shape ``(num_boundary_cells,)``."""
        idx = 0
        shape = (sum([grid.num_cells for grid in self.mdg.boundaries()]),)
        assert new_values.shape == shape, (
            f"Need array of shape {shape}," + f" but {new_values.shape} given."
        )

        for grid in self.mdg.boundaries():
            nc_d = grid.num_cells
            self.progress_value_in_time_on_grid(new_values[idx : idx + nc_d], grid)
            idx += nc_d

    def progress_value_in_time_on_subdomains(self, new_values: np.ndarray) -> None:
        """Convenience function to progress in time on all subdomains.

        ``new_values`` must be of shape ``(num_subdomain_cells,)``."""
        idx = 0
        shape = (self.mdg.num_subdomain_cells(),)
        assert new_values.shape == shape, (
            f"Need array of shape {shape}," + f" but {new_values.shape} given."
        )

        for grid in self.mdg.subdomains():
            nc_d = grid.num_cells
            self.progress_value_in_time_on_grid(new_values[idx : idx + nc_d], grid)
            idx += nc_d

    def progress_derivatives_in_time_on_subdomains(
        self, new_values: np.ndarray
    ) -> None:
        """Convenience function to progress derivatives in time on all subdomains.

        ``new_values`` must be of shape ``(num_dependencies, num_subdomain_cells)``."""
        idx = 0
        shape = (self.num_dependencies, self.mdg.num_subdomain_cells())
        # assert correct size
        assert new_values.shape == shape, (
            f"Need array of shape {shape}," + f" but {new_values.shape} given."
        )

        for grid in self.mdg.subdomains():
            idx = 0
            nc_d = grid.num_cells
            # expecting row-wise diff per dependency, and column wise the values per
            # grid
            self.progress_derivatives_in_time_on_grid(
                new_values[:, idx : idx + nc_d], grid
            )
            idx += nc_d

    def progress_value_in_time_on_interfaces(self, new_values: np.ndarray) -> None:
        """Convenience function to progress in time on all interfaces.

        ``new_values`` must be of shape ``(num_interfaces_cells,)``."""
        idx = 0
        shape = (self.mdg.num_interface_cells(),)
        assert new_values.shape == shape, (
            f"Need array of shape {shape}," + f" but {new_values.shape} given."
        )

        for grid in self.mdg.interfaces():
            nc_d = grid.num_cells
            self.progress_value_in_time_on_grid(new_values[idx : idx + nc_d], grid)
            idx += nc_d

    def progress_derivatives_in_time_on_interfaces(
        self, new_values: np.ndarray
    ) -> None:
        """Convenience function to progress derivatives in time on all interfaces.

        ``new_values`` must be of shape ``(num_dependencies, num_interface_cells)``."""
        idx = 0
        shape = (self.num_dependencies, self.mdg.num_interface_cells())
        # assert correct size
        assert new_values.shape == shape, (
            f"Need array of shape {shape}," + f" but {new_values.shape} given."
        )

        for grid in self.mdg.interfaces():
            idx = 0
            nc_d = grid.num_cells
            # expecting row-wise diff per dependency, and column wise the values per
            # grid
            self.progress_derivatives_in_time_on_grid(
                new_values[:, idx : idx + nc_d], grid
            )
            idx += nc_d

    def progress_value_in_time_on_grid(
        self, new_values: np.ndarray, grid: pp.GridLike
    ) -> None:
        """Shifts timestepping values backwards in times and sets ``new_value`` as the
        most recent one on the given ``grid``.

        Note:
            This function can be used to set numerical information of any shape.
            No checks are performed.

        """
        data = self._data_of(grid)

        # if grid is a boundary or the expression has no dependencies,
        # only values are stored, not a tuple of value and derivatives
        if isinstance(grid, pp.BoundaryGrid) or self.num_dependencies == 0:
            # shift
            for t in range(0, self._time_depth):
                # use .get in case no data was stored previously
                data[pp.TIME_STEP_SOLUTIONS][self.name][t + 1] = data[
                    pp.TIME_STEP_SOLUTIONS
                ][self.name].get(t, np.zeros(grid.num_cells))
            # set new value (first element of 2-tuple)
            data[pp.TIME_STEP_SOLUTIONS][self.name][0] = new_values
        # otherwise the secondary expression has value-diff tuple stored.
        else:
            # shift
            for t in range(0, self._time_depth):
                # use .get in case no data was stored previously
                data[pp.TIME_STEP_SOLUTIONS][self.name][t + 1][0] = data[
                    pp.TIME_STEP_SOLUTIONS
                ][self.name].get(t, (np.zeros(grid.num_cells), None))[0]
            # set new value (first element of 2-tuple)
            data[pp.TIME_STEP_SOLUTIONS][self.name][0][0] = new_values

    def progress_derivatives_in_time_on_grid(
        self, new_values: np.ndarray, grid: pp.GridLike
    ) -> None:
        """Shifts values of derivatives backwards in times and sets ``new_value`` as the
        most recent one on the given ``grid``.

        Note:
            This function can be used to set numerical information of any shape.
            No checks are performed.

        Raises:
            TypeError: If ``grid`` is a boundary grid (no derivatives on boundaries
                w.r.t. dependencies).
            ValueError: If :attr:`num_dependencies` is zero, i.e. no derivatives.
        """
        data = self._data_of(grid)

        if isinstance(grid, pp.BoundaryGrid):
            raise TypeError(f"Derivatives are not defined on boundary.")
        elif self.num_dependencies == 0:
            raise ValueError(
                f"Derivatives are not defined for expressions without dependencies."
            )

        # shift
        for t in range(0, self._time_depth):
            # use .get in case no data was stored previously
            data[pp.TIME_STEP_SOLUTIONS][self.name][t + 1][1] = data[
                pp.TIME_STEP_SOLUTIONS
            ][self.name].get(
                t, (None, np.zeros((self.num_dependencies, grid.num_cells)))
            )[
                1
            ]

        # set new value (second element of 2-tuple)
        data[pp.TIME_STEP_SOLUTIONS][self.name][0][1] = new_values

    def progress_iterate_values_on_domain(
        self,
        new_values: np.ndarray,
        grid: pp.Grid | pp.MortarGrid,
    ) -> None:
        """Shifts the iterate values backwards and sets ``new_value`` as the most
        recent one on the given ``grid``.

        Raises:
            TypeError: If the user attempts to set iterate values on the boundary.

        """
        if isinstance(grid, pp.BoundaryGrid):
            raise TypeError(f"Cannot set iterate values on a boundary grid.")

        data = self._data_of(grid)

        # if no dependencies, the values are stored
        if self.num_dependencies == 0:
            # shift
            for t in range(0, self._iterate_depth):
                # use .get in case no data was stored previously
                data[pp.ITERATE_SOLUTIONS][self.name][t + 1] = data[
                    pp.ITERATE_SOLUTIONS
                ][self.name].get(t, np.zeros(grid.num_cells))
            # set new value (first element of 2-tuple)
            data[pp.ITERATE_SOLUTIONS][self.name][0] = new_values
        # otherwise the value-derivative tuple is stored
        else:
            # shift
            for t in range(0, self._iterate_depth):
                # use .get in case no data was stored previously
                data[pp.ITERATE_SOLUTIONS][self.name][t + 1][0] = data[
                    pp.ITERATE_SOLUTIONS
                ][self.name].get(t, (np.zeros(grid.num_cells), None))[0]
            # set new value (first element of 2-tuple)
            data[pp.ITERATE_SOLUTIONS][self.name][0][0] = new_values

    def progress_iterate_derivatives_on_domain(
        self,
        new_values: np.ndarray,
        grid: pp.Grid | pp.MortarGrid,
    ) -> None:
        """Shifts the iterate values of derivatives backwards and sets ``new_value`` as
        the most recent one on the given ``grid``.

        Raises:
            TypeError: If the user attempts to set iterate values on the boundary.
            ValueError: If the expression has no dependencies, i.e., no derivatives.

        """
        if isinstance(grid, pp.BoundaryGrid):
            raise TypeError(f"Cannot set iterate derivative values on a boundary grid.")
        if self.num_dependencies == 0:
            raise ValueError(
                f"Derivatives are not defined for expressions without dependencies."
            )

        data = self._data_of(grid)
        # shift
        for t in range(0, self._iterate_depth):
            # use .get in case no data was stored previously
            data[pp.ITERATE_SOLUTIONS][self.name][t + 1][1] = data[
                pp.ITERATE_SOLUTIONS
            ][self.name].get(
                t, (None, np.zeros((self.num_dependencies, grid.num_cells)))
            )[
                1
            ]
        # set new value (first element of 2-tuple)
        data[pp.ITERATE_SOLUTIONS][self.name][0][1] = new_values

    @property
    def num_dependencies(self) -> int:
        """Number of first order dependencies of this operator.

        Given by the number of unique variables in the operator tree.

        This determines the number of required :meth:`derivatives`.

        """
        return self._ndep

    @property
    def boundary_values(self) -> np.ndarray:
        """Property to access and store the values on all boundaries, for convenience.

        The getter fetches the most recent time setp.

        The setter shifts time step values backwards and set the given value as the most
        recent one.

        Important:
            This is a different convention than for e.g. :attr:`subdomain_values`, where
            the iterate is accessed.

        Parameters:
            val: ``shape=(num_subdomain_cells,)``

                A new value to be set.

        """
        vals = []
        for _, data in self.mdg.boundaries(return_data=True):
            # no derivatives here, only values
            vals.append(data[pp.TIME_STEP_SOLUTIONS][self.name][0])
        if len(vals) > 0:
            return np.hstack((vals))
        else:
            return np.zeros(0, dtype=float)

    @boundary_values.setter
    def boundary_values(self, val: np.ndarray) -> None:
        self.progress_value_in_time_on_boundaries(val)

    @property
    def subdomain_values(self) -> np.ndarray:
        """Property to access and store the values on all subdomain, for convenience.

        The getter fetches the most recent iterate.

        The setter shifts iterate values backwards and set the given value as the most
        recent iterate.

        Parameters:
            val: ``shape=(num_subdomain_cells,)``

                A new value to be set.

        Raises:
            ValueError: If the size of the value mismatches what is expected.

        """
        vals = []
        if self.num_dependencies == 0:
            for _, data in self.mdg.subdomains(return_data=True):
                vals.append(data[pp.ITERATE_SOLUTIONS][self.name][0])
        else:
            for _, data in self.mdg.subdomains(return_data=True):
                vals.append(data[pp.ITERATE_SOLUTIONS][self.name][0][0])
        if len(vals) > 0:
            return np.hstack((vals))
        else:
            return np.zeros(0, dtype=float)

    @subdomain_values.setter
    def subdomain_values(self, val: np.ndarray) -> None:
        idx = 0
        shape = (self.mdg.num_subdomain_cells(),)
        assert val.shape == shape, (
            f"Need array of shape {shape}," + f" but {val.shape} given."
        )
        for g in self.mdg.subdomains():
            nc_d = g.num_cells
            self.progress_iterate_values_on_domain(val[idx : idx + nc_d], g)
            idx += nc_d

    @property
    def subdomain_derivatives(self) -> np.ndarray:
        """The global derivatives of this expression, w.r.t. to its first-order
        dependencies on all domains of definition.

        The derivatives values are stored row-wise per dependency, column wise per
        subdomain.

        The setter is for convenience to set the derivative values on grids.

        Important:
            The order of derivatives should reflect the order of dependencies
            passed at instantiation.

        Parameters:
            val: The new derivatives values.

        Raises:
            ValueError: If the expression has no dependencies, i.e. no derivatives.

        """
        if self.num_dependencies == 0:
            raise ValueError(
                f"Derivatives are not defined for expressions without dependencies."
            )
        vals = []
        for _, data in self.mdg.subdomains(return_data=True):
            vals.append(data[pp.ITERATE_SOLUTIONS][self.name][0][1])
        if len(vals) > 0:
            return np.hstack((vals))
        else:
            # TODO number of dependencies or a completely empty array?
            return np.zeros(self.num_dependencies, dtype=float)

    @subdomain_derivatives.setter
    def subdomain_derivatives(self, val: np.ndarray) -> None:
        shape = (self.num_dependencies, self.mdg.num_subdomain_cells())
        assert val.shape == shape, (
            f"Need array of shape {shape}," + f" but {val.shape} given."
        )
        for grid in self.mdg.subdomains():
            idx = 0
            nc_d = grid.num_cells
            self.progress_iterate_derivatives_on_domain(val[:, idx : idx + nc_d], grid)
            idx += nc_d

    @property
    def interface_values(self) -> np.ndarray:
        """Property to access and store the values on all interfaces, for convenience.

        The getter fetches the most recent iterate.

        The setter shifts iterate values backwards and set the given value as the most
        recent iterate.

        Parameters:
            val: ``shape=(num_interface_cells,)``

                A new value to be set.

        Raises:
            ValueError: If the size of the value mismatches what is expected.

        """
        vals = []
        if self.num_dependencies == 0:
            for _, data in self.mdg.interfaces(return_data=True):
                vals.append(data[pp.ITERATE_SOLUTIONS][self.name][0])
        else:
            for _, data in self.mdg.interfaces(return_data=True):
                vals.append(data[pp.ITERATE_SOLUTIONS][self.name][0][0])
        if len(vals) > 0:
            return np.hstack((vals))
        else:
            return np.zeros(0, dtype=float)

    @interface_values.setter
    def interface_values(self, val: np.ndarray) -> None:
        idx = 0
        shape = (self.mdg.num_interface_cells(),)
        assert val.shape == shape, (
            f"Need array of shape {shape}," + f" but {val.shape} given."
        )
        for g in self.mdg.interfaces():
            nc_d = g.num_cells
            self.progress_iterate_values_on_domain(val[idx : idx + nc_d], g)
            idx += nc_d

    @property
    def interface_derivatives(self) -> np.ndarray:
        """The global derivatives of this expression, w.r.t. to its first-order
        dependencies on all interfaces.

        The derivatives values are stored row-wise per dependency, column wise per
        interface.

        The setter is for convenience to set the derivative values on all grids.

        Important:
            The order of derivatives should reflect the order of dependencies
            passed at instantiation.

        Parameters:
            val: The new derivatives values.

        Raises:
            ValueError: If the expression has no dependencies, i.e. no derivatives.

        """
        if self.num_dependencies == 0:
            raise ValueError(
                f"Derivatives are not defined for expressions without dependencies."
            )
        vals = []
        for _, data in self.mdg.interfaces(return_data=True):
            vals.append(data[pp.ITERATE_SOLUTIONS][self.name][0][1])
        if len(vals) > 0:
            return np.hstack((vals))
        else:
            # TODO number of dependencies or a completely empty array?
            return np.zeros(self.num_dependencies, dtype=float)

    @interface_derivatives.setter
    def interface_derivatives(self, val: np.ndarray) -> None:
        shape = (self.num_dependencies, self.mdg.num_interface_cells())
        assert val.shape == shape, (
            f"Need array of shape {shape}," + f" but {val.shape} given."
        )
        for grid in self.mdg.interfaces():
            idx = 0
            nc_d = grid.num_cells
            self.progress_iterate_derivatives_on_domain(val[:, idx : idx + nc_d], grid)
            idx += nc_d
