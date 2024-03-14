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

    Not meant to be instantiated directly, only by calling the factory class
    :class:`SecondaryExpression`.

    This operator represents a function, which fetches the values stored for the
    secondary expression on respective domains.

    It supports the notion of previous time step and iterate values, which are
    mutually exclusive. I.e., it cannot represent a previous time step and iterate at
    the same time.
    But previous time steps and iterate can go further backwords in their respective
    direction.

    Parameters:
        name: Name of the called :class:`SecondaryExpression`.
        domains: Arguments to its call.
        time_step_depth: Assigned by the called expression
        iterate_depth: Assigned by the called expression
        time_step_index: Assigned as -1 by the expression, increased by
            :meth:`previous_timestep`.

            Operators representing the current time step or some iterate step, must have
            -1 assigned.

        iterate_index: Assigned as 0 by the expression, increased by
            :meth:`previous_iteration`.
        *children: The first-order dependencies of the called
            :class:`SecondaryExpression` in AD form (defined on the same ``domains``).

    """

    def __init__(
        self,
        name: str,
        domains: Sequence[pp.Grid] | Sequence[pp.MortarGrid],
        time_step_depth: int,
        iterate_depth: int,
        time_step_index: int,
        iterate_index: int,
        *children: pp.ad.MixedDimensionalVariable,
    ) -> None:
        assert (
            -1 <= time_step_index < time_step_depth
        ), f"Assigned time step index must be in interval [-1, {time_step_depth - 1})."
        assert (
            0 <= iterate_index < iterate_depth
        ), f"Assigned iterate index must be in interval [0, {iterate_depth - 1}]."

        super().__init__(name, domains, pp.ad.Operator.Operations.evaluate, children)

        self._time_depth: int = time_step_depth
        """Depth of stored values in time, with 0 denoting the first previous time,
        and higher numbers going backwards in time.

        Operators which present the next time step, or some iterate, should have -1.
        """
        self._iterate_depth: int = iterate_depth
        """Depth of stored iterate values, with 0 denoting the current iteration,
        positive numbers denoting previous iterations."""

        self._time_index: int = time_step_index
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

        self.func: Callable[
            [Tuple[pp.ad.AdArray | np.ndarray, ...]], pp.ad.AdArray | np.ndarray
        ]
        """The function accessing the data stored for this secondary expression.

        Assigned by :class:`SecondaryExpression` upon creation of this operator."""

        self.ad_compatible: bool = True
        """To trigger the same parsing as for the regular AD function."""

    def __repr__(self) -> str:
        return (
            f"Secondary operator with name {self.name}"
            + f" at time index {self.time_step_index}"
            + f" and iterate index {self.iterate_index}\n"
            + f"Defined on {len(self._domains)} {self._domain_type}.\n"
            + f"Dependent on {len(self.children)} independent operators.\n"
        )

    @property
    def time_step_index(self) -> int:
        """Returns the time step index this instance represents.

        - -1 represents the current time step
        - 0 represents the first previous time step
        - 1 represents the next time step further back in time
        - ...

        """
        return self._time_index

    @property
    def prev_time(self) -> bool:
        """True, if the operator represents a previous time-step."""
        return True if self._time_index > -1 else False

    @property
    def iterate_index(self) -> int:
        """Returns the time step index this instance represents.

        - 0 represents the current iterate
        - 1 represents the first previous iterate
        - 2 represents the iterate before that
        - ...

        """
        return self._iterate_index

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
        """
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
        if self._time_index == self._time_depth - 1:
            raise ValueError(
                f"Cannot go further back than {self._time_depth} steps in time."
            )

        # there is such a traversion implemented in the base class, calling the
        # variables at the previous time step.
        prev_time_children = tuple(pp.ad.Operator.previous_timestep(self).children)

        op = SecondaryOperator(
            self.name,
            self.domains,
            time_step_depth=self._time_depth,
            iterate_depth=self._iterate_depth,
            time_step_index=self.time_step_index + 1,  # increase time step index
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
        """
        Raises:
            ValueError: If the variable is a representation of the previous time step,
                previously set by :meth:`~previous_timestep`.
            ValueError: If the iterate depth was reached.

        Returns:
            A representation of this operator at the current time step, at one iteration
            before.

        """
        if self.prev_time:
            raise ValueError(
                "Cannot create a variable both on the previous time step and "
                "previous iteration."
            )
        if self.iterate_index == self._iterate_depth - 1:
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
            time_step_depth=self._time_depth,
            iterate_depth=self._iterate_depth,
            time_step_index=self.time_step_index,
            iterate_index=self.iterate_index + 1,  # increase iterate index
            *prev_iter_children,
        )

        # keeping track to the very first one
        if self.is_original_operator:
            op.original_operator = self
        else:
            op.original_operator = self.original_operator

        return op


class SecondaryExpression:
    """A representation of some dependent quantity in the PorePy modelling framework.

    This is a factory class, callable using some domains in the md-setting to create
    AD operators representing this expression on respective domains.
    It is meant for terms where the evaluation is done elsewhere and then stored using
    the functionality of this instnace.

    **On the boundary:**

    The class creates a
    :class:`~porepy.numerics.ad.opeators.TimeDependentDenseArray` using its given
    name and the boundar grids passed to the call.
    Boundary values can hence be updated like any other term in the model framework.
    But they can also be updated using :meth:`boundary_values` for convenience.

    The secondary expression has no derivatives and no iterate values on the boundary.

    **On subdomains:**

    The expression creates a :class:`SecondaryOperator`, which represents the data
    managed by this class on respective subdomains.
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

        1. The ``dependencies`` are assumed to be of first order. I.e., independent
           variables. Nested dependencies are not supported.
        2. It also supports only ``dependencies`` which have cell-wise a single DOF.
        3. Data is stored in a grid's data dictionary. On subdomains and interfaces,
           a list of length 2 is created, representing a value-derivative pair
           (1D array and 2D array with row-wise derivatives w.r.t. dependencies).
           To not bother with this, use the functionality of this class to set values
           and progress them iteratively/ in time.

    Important:
        This class (and :class:`SecondaryOperator`) support the notion of iterate
        and time step values according to the convention in PorePy.

        Current time step (to be solved for) is stored using iterate indices,
        starting with 0 and increasing for previous iterates.

        Previous time steps are stored using time step indices, starting with 0 for the
        most recent previous time (e.g., impl. Euler requires this) and increasing
        with further steps back in time.

        Iterate and timestep values can be progressed on individual grids using
        various ``progress_*`` methods.

        The properties defined by the class are for convenience to do it on all grids.

        Though, for subdomains and interfaces the properties progress the iterate value.
        Time step values have to be progressed explicitely.

        For boundaries, this class does not implement a progress in the iterative sense.
        If boundary values are set using the property, it stores them as current
        iterate values, and progress is made in time, by copying the previous value
        stored at the (single) iterate index to the previous time step index.

        This is for consistency with the remaining framework.

        **As of now, the porepy framework does not fully support index depths greater
        than 1, but this class does.**

    Parameters:
        name: Assigned name of the expression. Used to name operators and to store
            values in the data dictionaries
        mdg: The mixed-dimensional grid on which the expression is defined.
        *dependencies: Callables/ constructors for independent variables on which the
            expression depends. The order passed here is reflected in the order of
            stored derivative values.

            When calling the secondary expression on some grids, it is expected that the
            dependencies are defined there.
        time_step_depth: ``default=0``

            Depth of storage of values backwards in time.
            Default to 0 (no time-dependent storage for static problems).

        iterate_depth: ``default=1``

            Depth of storage of iterate values. By default only 1 iterate value is
            stored. The iterate values represent the current time step.
        prev_time_has_diffs: ``default=False``

            If True, operators representing this expression with
            :meth:`SecondaryOperator.prev_time` ``== True`` return an Ad array when
            evaluated. Else only the values at previous time are returned.
        prev_iter_has_diffs: ``default=False``

            If True, operators representing this expression with
            :meth:`SecondaryOperator.prev_iter` ``== True`` return an Ad array when
            evaluated. Else only the values at previous iterate are returned.

    Raises:
        ValueError: If ``time_step_depth`` is smaller than 0.
        ValueError: If ``iterate_depth`` is smaller than 1.
        ValueError: If there are no ``*dependencies``. The user should use other
            solutions in ``porepy.ad`` for this case.

    """

    def __init__(
        self,
        name: str,
        mdg: pp.MixedDimensionalGrid,
        *dependencies: Callable[[pp.GridLikeSequence], pp.ad.Operator],
        time_step_depth: int = 0,
        iterate_depth: int = 1,
        prev_time_has_diffs: bool = False,
        prev_iter_has_diffs: bool = False,
    ) -> None:
        if len(dependencies) == 0:
            raise ValueError("Secondary expressions must have dependencies.")

        if time_step_depth < 0:
            raise ValueError("Time step depth must be at least 0.")
        if iterate_depth < 1:
            raise ValueError("Iterate depth must be at least 1.")

        self._prev_t_has_d: bool = bool(prev_time_has_diffs)
        """Indicator if representations at previous time steps have derivatives w.r.t.
        their dependencies."""

        self._prev_i_has_d: bool = bool(prev_iter_has_diffs)
        """Indicator if representations at previous iterates have derivatives w.r.t.
        their dependencies."""

        self._dependencies: Tuple[
            Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator], ...
        ] = dependencies
        """Sequence of callable first order dependencies. Called when constructing
        operators on domains."""

        self._domains: list[pp.Grid | pp.MortarGrid] = list()
        """Keeping track of all domains on which the secondary expression was accessed.
        """
        self._boundaries: list[pp.BoundaryGrid] = list()
        """Keeping track of all boundary grids on which the secondary expressions was
        accessed."""

        self._time_depth: int = time_step_depth
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

    def __call__(self, domains: pp.GridLikeSequence) -> pp.ad.Operator:
        """Call this operator on a subset of grid in the md-grid to get a property
        restricted to the passed domains.

        The resulting operator is a function-evaluations fetching the values and
        derivatives from the "mother object" and simply restricting them onto the target
        subdomains.

        If ``subdomains`` contains all subdomains in the md-grid passed at
        instantiation, this method returns this instance itself.

        Parameters:
            domains: A subset of either grids, mortar grids or boundary grids in the
                md-grid passed at instantiation.

        Raises:
            AssertionError: If ``domains`` is empty or contains mixed-type grids
            TypeError: If ``domains`` is not composed of grids, mortar grids or boundary
                grids.

        """
        # prepare data storage when called on domains
        self._set_up_dictionaries(domains)

        if isinstance(domains[0], pp.BoundaryGrid):
            # keep track of domains of definition
            for d in domains:
                if d not in self._boundaries:
                    self._boundaries.append(d)
            op = pp.ad.TimeDependentDenseArray(self.name, domains)
        elif isinstance(domains[0], (pp.Grid, pp.MortarGrid)):
            children = [child(domains) for child in self._dependencies]

            # Check if first-order dependency
            assert all(isinstance(child, pp.ad.Variable) for child in children), (
                "Secondary expressions must depend on independent variables, not"
                + f" {[type(c) for c in children]}."
            )

            # keep track of domains of definition
            for d in domains:
                if d not in self._domains:
                    self._domains.append(d)

            # always start with operator at current time step, current iterate
            op = SecondaryOperator(
                self.name,
                domains,
                time_step_depth=self._time_depth,
                iterate_depth=self._iterate_depth,
                time_step_index=-1,
                iterate_index=0,
                *tuple(children),
            )

            # assign the function which extracts the data
            op.func = self._assign_function(op)

        else:
            raise ValueError(f"Unknown grid type {type(domains[0])}")

        return op

    def _set_up_dictionaries(self, domains: pp.GridLikeSequence) -> None:
        """Helper method to populate the data dictionaries in the md-grid and prepare
        the data storage for this expression, when calling it.

        Includes also a validation.

        Prepares the ``pp.ITERATE_SOLUTIONS`` dictionary, and ``pp.TIME_STEP_SOLUTIONS``
        if the time depth is not zero.

        """
        assert len(domains) > 0, "Cannot call expression without defining domain."
        assert all(
            [isinstance(d, type(domains[0])) for d in domains]
        ), "Cannot call expresion with mixed domain types."

        # access to data dicts depending on grid type
        data_getter: Callable[[pp.GridLike], dict]
        if isinstance(domains[0], pp.Grid):
            data_getter = self.mdg.subdomain_data
        elif isinstance(domains[0], pp.MortarGrid):
            data_getter = self.mdg.interface_data
        elif isinstance(domains[0], pp.BoundaryGrid):
            data_getter = self.mdg.boundary_grid_data
        else:
            raise TypeError(f"Unknown grid type {type(domains[0])}.")

        for grid in domains:
            data = data_getter(grid)

            # Every expression has at least one iterate value
            if pp.ITERATE_SOLUTIONS not in data:
                data[pp.ITERATE_SOLUTIONS] = {}
            if self.name not in data[pp.ITERATE_SOLUTIONS]:
                data[pp.ITERATE_SOLUTIONS][self.name] = {}

            # If an expression has a time-step depth, prepare dicts analogously in
            # TIME_STEP_SOLUTIONS
            if self._time_depth > 0:
                if pp.TIME_STEP_SOLUTIONS not in data:
                    data[pp.TIME_STEP_SOLUTIONS] = {}
                if self.name not in data[pp.TIME_STEP_SOLUTIONS]:
                    data[pp.TIME_STEP_SOLUTIONS][self.name] = {}

    def _get_op_data(
        self, op: SecondaryOperator, g: pp.Grid | pp.MortarGrid
    ) -> list[np.ndarray | None]:
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
            return d[pp.TIME_STEP_SOLUTIONS][self.name][op.time_step_index]
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
        and returns the values stored and managed by this instance.

        Parameters:
            op: Secondary operator created on the ``domains``.

        Returns:
            The callable to be assigned to :attr:`SecondaryOperator.func`

        """

        # This many values are expected to be stored for a secondary expression
        nc = sum([g.num_cells for g in op.domains])

        def func(*args: pp.ad.AdArray | np.ndarray) -> pp.ad.AdArray | np.ndarray:
            assert (
                len(args) == self.num_dependencies
            ), f"Evaluation of {self.name} expects {self.num_dependencies} args."

            vals: list[np.ndarray] = []
            diffs: list[np.ndarray] = []
            for g in op.domains:
                # The function expects 2 arrays, one for values, one for derivatives
                vd = self._get_op_data(op, g)

                if None in vd:
                    raise ValueError(
                        "No values/derivatives stored for secondary expression"
                        + f" {self.name} at time or iterate index"
                        + f" {(op.time_step_index, op.iterate_index)} on grid {g}."
                    )
                else:
                    assert isinstance(vd[0], np.ndarray) and isinstance(
                        vd[1], np.ndarray
                    ), (
                        "Expecting [array, array] stored for secondary expression"
                        + f" {self.name}, not {[type(vd[0]), type(vd[1])]}."
                    )

                vals.append(vd[0])
                diffs.append(vd[1])

            # values per domain per cell
            value = np.hstack(vals)

            # if no derivatives requested for prev time or iter, return value
            if op.prev_time and not self._prev_t_has_d:
                return value
            if op.prev_iter and not self._prev_i_has_d:
                return value

            # derivatives, row-wise dependencies, column-wise per domain per cell
            derivatives = np.hstack(diffs)

            # sanity checks
            assert value.shape == (nc,), (
                f"Secondary expression {self.name} requires {nc} values stored in"
                + f" domains {op.domains}"
            )
            assert derivatives.shape == (self.num_dependencies, nc), (
                f"Secondary expression {self.name} requires {nc} derivative values"
                + f" per dependency stored in domains {op.domains}."
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

    @property
    def num_dependencies(self) -> int:
        """Number of first order dependencies of this operator, passed at instantiation.

        This determines the number of required :meth:`derivatives`."""
        return len(self._dependencies)

    # Convenience properties to access and progress iterative values on grids and mortar

    @property
    def boundary_values(self) -> np.ndarray:
        """Property to access and store the value at the current time step, current
        iterate on all boundaries on which it was accessed.

        The getter fetches the values in the order imposed by the md-grid.

        The setter shifts iterate values backwards and set the given value as the most
        recent iterate.

        Note:
            This is a convenience functionality for :meth:`update_boundary_values`,
            which operates on all boundary grids on which the property was accessed.

            Hence, when setting boundary values this way, the user shifts their values
            in the **time sense**, copyig the current iterate value to the first,
            previous time step value.

        Parameters:
            val: ``shape=(num_boundary_cells,)``

                A new value to be set.

        Raises:
            ValueError: If the size of the value mismatches what is expected.

        """
        vals = []
        for grid, data in self.mdg.boundaries(return_data=True):
            if grid not in self._boundaries:
                continue
            vals.append(data[pp.ITERATE_SOLUTIONS][self.name][0][0])
        if len(vals) > 0:
            return np.hstack(vals)
        else:
            return np.zeros(0, dtype=float)

    @boundary_values.setter
    def boundary_values(self, val: np.ndarray) -> None:
        shape = (sum([g.num_cells for g in self._boundaries]),)
        assert val.shape == shape, (
            f"Need array of shape {shape}," + f" but {val.shape} given."
        )

        idx = 0
        for grid in self.mdg.boundaries():
            if grid not in self._boundaries:
                continue
            nc_d = grid.num_cells
            self.update_boundary_value(val[idx : idx + nc_d], grid)
            idx += nc_d

    @property
    def subdomain_values(self) -> np.ndarray:
        """Property to access and store the value at the current time step, current
        iterate on all subdomains on which it was accessed.

        The getter fetches the values in the order imposed by the md-grid.

        The setter shifts iterate values backwards and set the given value as the most
        recent iterate.

        Parameters:
            val: ``shape=(num_subdomain_cells,)``

                A new value to be set.

        Raises:
            ValueError: If the size of the value mismatches what is expected.

        """
        vals = []
        for grid, data in self.mdg.subdomains(return_data=True):
            if grid not in self._domains:
                continue
            vals.append(data[pp.ITERATE_SOLUTIONS][self.name][0][0])
        if len(vals) > 0:
            return np.hstack(vals)
        else:
            return np.zeros(0, dtype=float)

    @subdomain_values.setter
    def subdomain_values(self, val: np.ndarray) -> None:
        shape = (sum([g.num_cells for g in self._domains if isinstance(g, pp.Grid)]),)
        assert val.shape == shape, (
            f"Need array of shape {shape}," + f" but {val.shape} given."
        )

        idx = 0
        for grid in self.mdg.subdomains():
            if grid not in self._domains:
                continue
            nc_d = grid.num_cells
            self.progress_iterate_values_on_grid(val[idx : idx + nc_d], grid)
            idx += nc_d

    @property
    def subdomain_derivatives(self) -> np.ndarray:
        """Property to access and store the derivatives at the current time step,
        current iterate on all subdomains on which it was accessed.

        The derivatives values are stored row-wise per dependency, column wise per
        subdomain.

        The setter is for convenience to set the derivative values on all subdomains.

        Important:
            The order of derivatives should reflect the order of dependencies
            passed at instantiation.

        Parameters:
            val: ``shape=(num_dependencies, num_subdomain_cells)``

                A new value to be set.

        Raises:
            ValueError: If the size of the value mismatches what is expected.

        """
        vals = []
        for grid, data in self.mdg.subdomains(return_data=True):
            if grid not in self._domains:
                continue
            vals.append(data[pp.ITERATE_SOLUTIONS][self.name][0][1])
        if len(vals) > 0:
            return np.hstack(vals)
        else:
            return np.zeros(self.num_dependencies, dtype=float)

    @subdomain_derivatives.setter
    def subdomain_derivatives(self, val: np.ndarray) -> None:
        shape = (
            self.num_dependencies,
            sum([g.num_cells for g in self._domains if isinstance(g, pp.Grid)]),
        )
        assert val.shape == shape, (
            f"Need array of shape {shape}," + f" but {val.shape} given."
        )

        idx = 0
        for grid in self.mdg.subdomains():
            if grid not in self._domains:
                continue
            nc_d = grid.num_cells
            self.progress_iterate_derivatives_on_grid(val[:, idx : idx + nc_d], grid)
            idx += nc_d

    @property
    def interface_values(self) -> np.ndarray:
        """Property to access and store the value at the current time step, current
        iterate on all interfaces on which it was accessed.

        The getter fetches the values in the order imposed by the md-grid.

        The setter shifts iterate values backwards and set the given value as the most
        recent iterate.

        Parameters:
            val: ``shape=(num_interface_cells,)``

                A new value to be set.

        Raises:
            ValueError: If the size of the value mismatches what is expected.

        """
        vals = []
        for grid, data in self.mdg.interfaces(return_data=True):
            if grid not in self._domains:
                continue
            vals.append(data[pp.ITERATE_SOLUTIONS][self.name][0][0])
        if len(vals) > 0:
            return np.hstack(vals)
        else:
            return np.zeros(0, dtype=float)

    @interface_values.setter
    def interface_values(self, val: np.ndarray) -> None:
        shape = (
            sum([g.num_cells for g in self._domains if isinstance(g, pp.MortarGrid)]),
        )
        assert val.shape == shape, (
            f"Need array of shape {shape}," + f" but {val.shape} given."
        )

        idx = 0
        for grid in self.mdg.interfaces():
            if grid not in self._domains:
                continue
            nc_d = grid.num_cells
            self.progress_iterate_values_on_grid(val[idx : idx + nc_d], grid)
            idx += nc_d

    @property
    def interface_derivatives(self) -> np.ndarray:
        """Property to access and store the derivatives at the current time step,
        current iterate on all interfaces on which it was accessed.

        The derivatives values are stored row-wise per dependency, column wise per
        mortar grid.

        The setter is for convenience to set the derivative values on all interfaces.

        Important:
            The order of derivatives should reflect the order of dependencies
            passed at instantiation.

        Parameters:
            val: ``shape=(num_dependencies, num_interface_cells)``

                A new value to be set.

        Raises:
            ValueError: If the size of the value mismatches what is expected.

        """
        vals = []
        for grid, data in self.mdg.interfaces(return_data=True):
            if grid not in self._domains:
                continue
            vals.append(data[pp.ITERATE_SOLUTIONS][self.name][0][1])
        if len(vals) > 0:
            return np.hstack(vals)
        else:
            return np.zeros(self.num_dependencies, dtype=float)

    @interface_derivatives.setter
    def interface_derivatives(self, val: np.ndarray) -> None:
        shape = (
            self.num_dependencies,
            sum([g.num_cells for g in self._domains if isinstance(g, pp.MortarGrid)]),
        )
        assert val.shape == shape, (
            f"Need array of shape {shape}," + f" but {val.shape} given."
        )

        idx = 0
        for grid in self.mdg.interfaces():
            if grid not in self._domains:
                continue
            nc_d = grid.num_cells
            self.progress_iterate_derivatives_on_grid(val[:, idx : idx + nc_d], grid)
            idx += nc_d

    # Methods operating on single grids

    def update_boundary_value(
        self, value: np.ndarray, boundary_grid: pp.BoundaryGrid
    ) -> None:
        """Function to update the value of the secondary expression on the boundary.

        The update process of boundary values is different from the process on
        subdomains and interfaces.

        1. Boundary values have only a single iterate value (current time step)
        2. Boundary values can have multiple previous time steps, if time step depth is
           not zero (can be discussed why).
        3. Boundary cannot be updated collectively, but must be done per grid.

        Parameters:
            value: ``shape=(bg.num_cells,)``

                A new value to be set for the boundary.
            boundary_grid: A boundary grid in the mixed-dimensional domain.

        Raises:
            ValueError: If ``value`` is not of the expected shape.

        """
        shape = (boundary_grid.num_cells,)
        assert value.shape == shape, (
            f"Need array of shape {shape}," + f" but {value.shape} given."
        )
        data = self._data_of(boundary_grid)

        # take the current single iterate value, and store it in time if time-depth
        # given
        if self._time_depth > 0:
            for t in range(self._time_depth - 1, 0, -1):
                val = data[pp.TIME_STEP_SOLUTIONS][self.name].get(
                    t - 1, np.zeros(boundary_grid.num_cells)
                )
                data[pp.TIME_STEP_SOLUTIONS][self.name][t] = val
            data[pp.TIME_STEP_SOLUTIONS][self.name][0] = data[pp.ITERATE_SOLUTIONS][
                self.name
            ].get(0, np.zeros(boundary_grid.num_cells))
        # set the given value as the new single iterate value (current time)
        data[pp.ITERATE_SOLUTIONS][self.name][0] = value

    def _shift_values(
        self, grid: pp.Grid | pp.MortarGrid, loc: str, new_values: np.ndarray
    ) -> None:
        """Helper function to shift the values in the storage.

        Parameters:
            data: A grid data dictionary, assuming it is prepared
            loc: Either ``pp.ITERATE_SOLUTIONS`` or ``pp.TIME_STEP_SOLUTIONS``
            new_values: New 1D array to be stored.

        """
        if loc == pp.ITERATE_SOLUTIONS:
            range_ = range(self._iterate_depth - 1, 0, -1)
        elif loc == pp.TIME_STEP_SOLUTIONS:
            # abort shift if no time step depth
            if self._time_depth == 0:
                return
            range_ = range(self._time_depth - 1, 0, -1)
        else:
            raise ValueError(f"Unsupported location {loc}")

        data = self._data_of(grid)

        for i in range_:
            vd = data[loc][self.name].get(i - 1, [np.zeros(grid.num_cells), None])
            # if derivative data present, don't overwrite
            if i in data[loc][self.name]:
                vd[1] = data[loc][self.name][i][1]
            data[loc][self.name][i] = vd

        if 0 in data[loc][self.name]:
            data[loc][self.name][0][0] = new_values
        else:
            data[loc][self.name][0] = [new_values, None]

    def _shift_derivatives(
        self, grid: pp.Grid | pp.MortarGrid, loc: str, new_values: np.ndarray
    ) -> None:
        """Helper function to shift the derivative values in the storage.

        Parameters:
            grid: A grid, assuming data dictionaries are prepard.
            loc: Either ``pp.ITERATE_SOLUTIONS`` or ``pp.TIME_STEP_SOLUTIONS``
            new_values: New 2D array to be stored.

        """
        if loc == pp.ITERATE_SOLUTIONS:
            range_ = range(self._iterate_depth - 1, 0, -1)
        elif loc == pp.TIME_STEP_SOLUTIONS:
            # abort shift if no time step depth
            if self._time_depth == 0:
                return
            range_ = range(self._time_depth - 1, 0, -1)
        else:
            raise ValueError(f"Unsupported location {loc}")

        data = self._data_of(grid)

        for i in range_:
            vd = data[loc][self.name].get(i - 1, [None, np.zeros(grid.num_cells)])
            # if value data present, don't overwrite
            if i in data[loc][self.name]:
                vd[0] = data[loc][self.name][i][0]
            data[loc][self.name][i] = vd

        if 0 in data[loc][self.name]:
            data[loc][self.name][0][1] = new_values
        else:
            data[loc][self.name][0] = [None, new_values]

    def progress_values_in_time(
        self, domains: Optional[Sequence[pp.Grid | pp.MortarGrid]] = None
    ) -> None:
        """Shifts timestepping values backwards in times and sets the most recent
        iterate value as the most recent, previous time step value.

        Parameters:
            domains ``default=None``

                If given, performs the progress only on given domains. Otherwise it is
                performed on all domains on which the secondary expression was
                accessed so far.

        Raises:
            ValueError: If ``time_step_depth`` at instantiation was set to zero.
            AssertionError: If progress is requested on domains on which the expression
                was never called.
            KeyError: If nothing is stored as the current iterate

        """
        if self._time_depth < 1:
            raise ValueError(
                f"Cannot progress secondary expression {self.name} in time with"
                + f" time step depth set to zero."
            )
        if domains is None:
            domains = self._domains
        else:
            assert all(d in self._domains for d in domains), (
                f"Progressing secondary expression {self.name} on domains on which it"
                + " was not accessed."
            )

        for grid in domains:
            data = self._data_of(grid)
            current_vals = data[pp.ITERATE_SOLUTIONS][self.name][0][0]
            self._shift_values(grid, pp.TIME_STEP_SOLUTIONS, current_vals)

    def progress_derivatives_in_time(
        self, domains: Optional[Sequence[pp.Grid | pp.MortarGrid]] = None
    ) -> None:
        """Analogous to :meth:`progress_values_in_time`, but for derivatives."""
        if self._time_depth < 1:
            raise ValueError(
                f"Cannot progress secondary expression {self.name} in time with"
                + f" time step depth set to zero."
            )
        if domains is None:
            domains = self._domains
        else:
            assert all(d in self._domains for d in domains), (
                f"Progressing secondary expression {self.name} on domains on which it"
                + " was not accessed."
            )

        for grid in domains:
            data = self._data_of(grid)
            current_vals = data[pp.ITERATE_SOLUTIONS][self.name][0][1]
            self._shift_derivatives(grid, pp.TIME_STEP_SOLUTIONS, current_vals)

    def progress_iterate_values_on_grid(
        self,
        new_values: np.ndarray,
        grid: pp.Grid | pp.MortarGrid,
    ) -> None:
        """Shifts the iterate values backwards and sets ``new_value`` as the most
        recent one on the given ``grid``.

        Raises:
            AssertionError: If ``grid`` not among the subdomains or interfaces on which
                the expression was accessed.
            AssertionError: If ``new_vales`` is not of shape ``(grid.num_cells,)``.

        """
        assert (
            grid in self._domains
        ), f"Secondary expression {self.name} not defined on grid {grid}."
        shape = (grid.num_cells,)
        assert new_values.shape == shape, (
            f"Need array of shape {shape}," + f" but {new_values.shape} given."
        )
        self._shift_values(grid, pp.ITERATE_SOLUTIONS, new_values)

    def progress_iterate_derivatives_on_grid(
        self,
        new_values: np.ndarray,
        grid: pp.Grid | pp.MortarGrid,
    ) -> None:
        """Shifts the iterate values of derivatives backwards and sets ``new_value`` as
        the most recent one on the given ``grid``.

        Raises:
            AssertionError: If ``grid`` not among the subdomains or interfaces on which
                the expression was accessed.
            AssertionError: If ``new_vales`` is not of shape
                ``(num_dependencies, grid.num_cells)``.

        """
        assert (
            grid in self._domains
        ), f"Secondary expression {self.name} not defined on grid {grid}."
        shape = (grid.num_cells,)
        assert new_values.shape == shape, (
            f"Need array of shape {shape}," + f" but {new_values.shape} given."
        )
        self._shift_derivatives(grid, pp.ITERATE_SOLUTIONS, new_values)
