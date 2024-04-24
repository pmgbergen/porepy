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
    "CompositeModellingError",
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


class CompositeModellingError(Exception):
    """Custom exception class to alert the user when using the composite framework
    logically inconsistent."""


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
        children: The first-order dependencies of the called
            :class:`SecondaryExpression` in AD form (defined on the same ``domains``).
        time_step_index: ``default=-1``

            Assigned as -1 by the expression, increased by
            :meth:`previous_timestep`.

            Operators representing the current time step or some iterate step, must have
            -1 assigned.

        iterate_index: ``default=0``

            Assigned as 0 by the expression, increased by :meth:`previous_iteration`.

    """

    def __init__(
        self,
        name: str,
        domains: Sequence[pp.Grid] | Sequence[pp.MortarGrid],
        children: Sequence[pp.ad.Variable],
        time_step_index: int = -1,
        iterate_index: int = 0,
    ) -> None:
        assert -1 <= time_step_index, "Assigned time step index must be >= -1."
        assert 0 <= iterate_index, "Assigned iterate index must be >= 0."

        if (iterate_index > 0 and time_step_index != -1) or (
            time_step_index > -1 and iterate_index != 0
        ):
            raise ValueError(
                "Cannot represent previous time step and iterate at the same time."
            )

        super().__init__(name, domains, pp.ad.Operator.Operations.secondary, children)

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

        self.fetch_data: Callable[
            [SecondaryOperator, pp.GridLike], Sequence[np.ndarray]
        ]
        """A function returning the stored data for a secondary operator on a grid,
        which is time step and iterate index dependent.

        This function is assigned by the factory class :class:`SecondaryExpression`
        which has a reference to the md-grid and must not be touched by the user.

        """

    def __str__(self) -> str:
        if self.prev_time:
            msg = f"Secondary operator {self.name} at time step {self.time_step_index}"
        else:
            msg = f"Secondary operator {self.name} at iterate {self.iterate_index}"

        return (
            msg
            + f"\nDefined on {len(self._domains)} {self._domain_type}.\n"
            + f"Dependent on {len(self.children)} independent expressions.\n"
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

        # This works because children are always first order (variables)
        prev_time_children = [child.previous_timestep() for child in self.children]

        op = SecondaryOperator(
            self.name,
            self.domains,
            prev_time_children,
            self.time_step_index + 1,  # increase time step index
            self.iterate_index,
        )
        op.fetch_data = self.fetch_data

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

        prev_iter_children = [child.previous_iteration() for child in self.children]

        op = SecondaryOperator(
            self.name,
            self.domains,
            prev_iter_children,
            self.time_step_index,
            self.iterate_index + 1,  # increase iterate index
        )
        op.fetch_data = self.fetch_data

        # keeping track to the very first one
        if self.is_original_operator:
            op.original_operator = self
        else:
            op.original_operator = self.original_operator

        return op

    def func(self, *args: pp.ad.AdArray | np.ndarray) -> pp.ad.AdArray | np.ndarray:
        """Function which returns the data stored for this secondary operator at its
        time step or iterate index.

        Uses :attr:`fetch_data` assigned by its factory class
        :class:`SecondaryExpression`.

        If ``*args`` is composed of Ad arrays, this function returns also an Ad array,
        which contains the stored derivative values per dependency.

        If on the other hand ``*args`` is composed of numpy arrays (like in the case
        of an evaluation on the previous time step or iterate), this function
        will return only values as a numpy array.

        Note:
            This function contains several validation steps to ensure the correct usage.

            It expects arguments equal to the amount of dependencies.
            It expects values and derivatives of proper shape stored under respective
            time or iterate index.

            It performes all those validations at every evaluation, and raises
            errors respectively.

        """

        nd = len(self.children)  # number of expected dependencies
        nc = sum([g.num_cells for g in self.domains])  # number of expected values
        with_derivatives: bool  # flag indicating if stored derivatives are requested

        assert len(args) == nd, f"Evaluation of {self.name} expects {nd} args."

        # case when evaluated at current time step, current iterate
        if all(isinstance(a, pp.ad.AdArray) for a in args):
            with_derivatives = True
        # other cases, some previous time or iterate
        else:
            assert all(
                isinstance(a, np.ndarray) for a in args
            ), "Functions of sec. operators expect either Ad or numpy arrays as args."
            with_derivatives = False

        vals: list[np.ndarray] = []
        diffs: list[np.ndarray] = []
        for g in self.domains:
            # The function expects objects, one for values, one for derivatives
            # It also validates if data is stored or not
            vd = self.fetch_data(self, g)
            vals.append(vd[0])
            diffs.append(vd[1])

        # Check if values present and properly shaped
        if np.any([v is None for v in vals]):
            idx = [i for i, v in enumerate(vals) if v is None]
            raise ValueError(
                f"No values stored for secondary operator {self}"
                + f" on grids {self.domains[idx]}."
            )
        value = np.hstack(vals)
        assert value.shape == (nc,), (
            f"Secondary expression {self.name} requires {nc} values stored in"
            + f" domains {self.domains}"
        )

        # if no derivatives requested, return value
        if not with_derivatives:
            return value

        # proceeding with derivatives, by filling the identity blocks of the first-order
        # dependencies. Check if derivatives are stored
        if np.any([d is None for d in diffs]):
            idx = [i for i, d in enumerate(diffs) if d is None]
            raise ValueError(
                f"No derivative values stored for secondary operator {self}"
                + f" on grids {self.domains[idx]}."
            )
        # derivatives, row-wise dependencies, column-wise per domain per cell
        derivatives = np.hstack(diffs)
        assert derivatives.shape == (nd, nc), (
            f"Secondary expression {self.name} requires {nc} derivative values"
            + f" per dependency stored in domains {self.domains}."
        )

        # list of jacs per dependency, assuming porepy.ad makes consistent shapes
        jacs: list[sps.coo_matrix] = []

        for i, arg in enumerate(args):
            # by logic, arg has to be an AD array
            arg = cast(pp.ad.AdArray, arg)
            # by logic, this is a 1D array
            d_i = cast(np.ndarray, derivatives[i])

            idx = cast(tuple[np.ndarray, np.ndarray], arg.jac.nonzero())
            shape = arg.jac.shape
            # sanity checks that the amount of derivatives (per cell) match
            assert (
                idx[0].shape == value.shape
            ), "Mismatch in shape of derivatives for arg 1."
            assert (
                idx[0].shape == d_i.shape
            ), "Mismatch in shape of provided derivatives for arg 1."
            jacs.append(sps.coo_matrix((d_i, idx), shape=shape))

        return pp.ad.AdArray(value, cast(sps.coo_matrix, sum(jacs)).tocsr())


class SecondaryExpression:
    """A representation of some dependent quantity in the PorePy modelling framework.

    This is a factory class, callable using some domains in the md-setting to create
    AD operators representing this expression on respective domains.
    It is meant for terms where the evaluation is done elsewhere and then stored using
    the functionality of this instance.

    **On the boundary:**

    The class creates a
    :class:`~porepy.numerics.ad.opeators.TimeDependentDenseArray` using its given
    name and the boundar grids passed to the call.
    Boundary values can hence be updated like any other term in the model framework.
    But they can also be updated using :meth:`boundary_values` for convenience.

    The secondary expression has no derivatives and no iterate values on the boundary.

    **On subdomains and interfaces:**

    The expression creates a :class:`SecondaryOperator`, which represents the data
    managed by this class on subdomains and interfaces.
    The operator can depend on instances of
    :class:`~porepy.numerics.ad.operators.MixedDimensionalVariable`.

    The secondary operator is essentially like a
    :class:`~porepy.numerics.ad.operator_functions.Function`. Upon evaluation it fetches
    the stored values and the derivative values.
    The derivative values are inserted into the Jacobians of the first-order
    dependencies (identity blocks).

    The secondary operator supports the notion of previous timesteps and iterate
    values. Updates of respective values are handled by this factory class.

    **Call with empty domain list:**

    This functionality is implemented for completeness reasons, such that the secondary
    expression can also be called on an empty list of domains. In this case the property
    returns an empty array wrapped as an Ad array.

    This functionality is implemented since general PorePy models implement equations
    in the md-setting. It makes this class compatible for models without fractures or
    interfaces.

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

        The properties defined by the class are for convenience to do it on all grids of
        a certain type (subdomain, interface or boundary).

        For subdomains and interfaces, the properties progress the iterate value.
        Time step values have to be progressed explicitely using
        :meth:`progress_values_in_time` and :meth:`progress_derivatives_in_time`.

        For boundaries, this class does not implement a progress in the iterative sense.
        If boundary values are set using the property :meth:`boundary_values`, it stores
        them as current iterate values (index 0), and progress is made in time by
        copying the previous value stored at the single iterate value to the previous
        time step index.

        This is for consistency with the remaining framework.

        **As of now, the porepy framework does not fully support index depths greater
        than 1, but this class does.**

    Parameters:
        name: Assigned name of the expression. Used to name operators and to store
            values in the data dictionaries
        mdg: The mixed-dimensional grid on which the expression is defined.
        dependencies: A sequence of callables/ constructors for independent variables on
            which the expression depends. The order passed here is reflected in the
            order of stored derivative values.

            When calling the secondary expression on some grids, it is expected that the
            dependencies are defined there.
        time_step_depth: ``default=0``

            Depth of storage of values backwards in time.
            Default to 0 (no time-dependent storage for static problems).

        iterate_depth: ``default=1``

            Depth of storage of iterate values. By default only 1 iterate value is
            stored. The iterate values represent the current time step.

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
        dependencies: Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]],
        time_step_depth: int = 0,
        iterate_depth: int = 1,
    ) -> None:
        if len(dependencies) == 0:
            raise ValueError("Secondary expressions must have dependencies.")

        if time_step_depth < 0:
            raise ValueError("Time step depth must be at least 0.")
        if iterate_depth < 1:
            raise ValueError("Iterate depth must be at least 1.")

        self._dependencies: Tuple[
            Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator], ...
        ] = dependencies
        """Sequence of callable first order dependencies. Called when constructing
        operators on domains."""

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

        self._set_up_dictionaries()

    def __call__(self, domains: pp.GridLikeSequence) -> pp.ad.Operator:
        """Call this operator on a subset of grid in the md-grid to get a property
        restricted to the passed domains.

        1. If called on an empty list, a wrapped, empty array is returned.
        2. If called on boundary grids, a time-dependent dense array is returned.
        3. If called on subdomains on interfaces, a seoncdary operator is returned.

        Parameters:
            domains: A subset of either grids, mortar grids or boundary grids in the
                md-grid passed at instantiation.

        Raises:
            ValueError: If ``domains`` is not composed of either grids, mortar grids or
                boundary grids.

        """

        # This is for completeness reasons, when calling equations on empty list
        if len(domains) == 0:
            return pp.wrap_as_dense_ad_array(np.zeros((0,)), name=self.name)
        # On the boundary, this is a Time-Dependent dense array
        elif all(isinstance(g, pp.BoundaryGrid) for g in domains):
            op = pp.ad.TimeDependentDenseArray(self.name, domains)
        # On subdomains or interfaces, create the secondary operators
        elif all(isinstance(g, pp.Grid) for g in domains) or all(
            isinstance(g, pp.MortarGrid) for g in domains
        ):
            children = [child(domains) for child in self._dependencies]

            # Check if first-order dependency
            assert all(isinstance(child, pp.ad.Variable) for child in children), (
                "Secondary expressions must depend on independent variables, not"
                + f" {[type(c) for c in children]}."
            )

            # always start with operator at current time step, current iterate
            op = SecondaryOperator(self.name, domains, children)

            # assign the function which extracts the data
            op.fetch_data = self.fetch_data
        else:
            raise ValueError(
                f"Unsupported domain configuration {[type(g) for g in domains]}"
            )

        return op

    def fetch_data(
        self, op: SecondaryOperator, grid: pp.Grid | pp.MortarGrid
    ) -> Sequence[np.ndarray | None]:
        """Function fetching the data stored for this secondary expression, represented
        in Ad form by ``op``.

        ``op`` has a time step or iterate index, which specifies which data should be
        fetched.

        This function is assigned to :attr:`SecondaryOperator.fetch_data`
        (a work-around such that the Ad operator has no reference to the md-grid like
        the other operators).

        Note:
            (Implementation) This method does not validate if data is stored or not, in
            order to allow for more flexibility in the evaluation of ``op``.

        Parameters:
            op: This expression in operator form, created by calling it on subdomains or
                interfaces.
            g: A grid or mortar grid, on which ``op`` is defined.

        Returns:
            The data stored in the grid dictionary for this expression.
            It is a Sequence of length 2, representing a value and derivative pair.

            It may contain a None, if data is not present.

        """
        # if op is at previous time, get those values
        if op.prev_time:
            loc = pp.TIME_STEP_SOLUTIONS
            index = op.time_step_index
        else:
            loc = pp.ITERATE_SOLUTIONS
            index = op.iterate_index

        data = self._data_of(grid)
        return data[loc][self.name].get(index, [None, None])

    def _fetch_values(self, grid: pp.GridLike, loc: str, index: int) -> np.ndarray:
        """Helper function to fetch the value of the secondary expression,
        at a given location and index.

        Performs also validations if data is stored.

        Parameters:
            grid: A grid in the md-grid.
            loc: Either ``pp.ITERATE_SOLUTIONS`` or ``pp.TIME_STEP_SOLUTIONS``
            index: Iterate or time step index.

        Raises:
            KeyError: If no values were stored for this instance at given location and
                index.

        """
        data = self._data_of(grid)
        values = None

        if index in data[loc][self.name]:
            values = data[loc][self.name][index][0]

        if values is None:
            raise KeyError(
                f"No values stored for secondary expression {self.name} at location"
                + f" {loc} and index {index} on grid {grid}."
            )
        else:
            return cast(np.ndarray, values)

    def _fetch_derivative_values(
        self, grid: pp.GridLike, loc: str, index: int
    ) -> np.ndarray:
        """Helper function to fetch the derivative values of the secondary expression,
        at a given location and index.

        Performs also validations if data is stored.

        Parameters:
            grid: A grid in the md-grid.
            loc: Either ``pp.ITERATE_SOLUTIONS`` or ``pp.TIME_STEP_SOLUTIONS``
            index: Iterate or time step index.

        Raises:
            KeyError: If no derivative values were stored for this instance at given
                location and index.

        """
        data = self._data_of(grid)
        values = None

        if index in data[loc][self.name]:
            values = data[loc][self.name][index][1]

        if values is None:
            raise KeyError(
                f"No derivative values stored for secondary expression {self.name} at"
                + f" location {loc} and index {index} on grid {grid}."
            )
        else:
            return cast(np.ndarray, values)

    def _set_up_dictionaries(self) -> None:
        """Helper method to populate the data dictionaries in the md-grid and prepare
        the data storage for this expression, when calling it.

        Prepares the ``pp.ITERATE_SOLUTIONS`` dictionary, and ``pp.TIME_STEP_SOLUTIONS``
        if the time depth is not zero.

        """
        # subdomains
        for _, data in self.mdg.subdomains(return_data=True):
            # Every expression has at least one iterate value (current time step)
            if pp.ITERATE_SOLUTIONS not in data:
                data[pp.ITERATE_SOLUTIONS] = {}
            if self.name not in data[pp.ITERATE_SOLUTIONS]:
                data[pp.ITERATE_SOLUTIONS][self.name] = {}

            # If an expression has a time-step depth, prepare dicts analogously in
            # TIME_STEP_SOLUTIONS (previous time steps)
            if self._time_depth > 0:
                if pp.TIME_STEP_SOLUTIONS not in data:
                    data[pp.TIME_STEP_SOLUTIONS] = {}
                if self.name not in data[pp.TIME_STEP_SOLUTIONS]:
                    data[pp.TIME_STEP_SOLUTIONS][self.name] = {}
        # interfaces
        for _, data in self.mdg.interfaces(return_data=True):
            if pp.ITERATE_SOLUTIONS not in data:
                data[pp.ITERATE_SOLUTIONS] = {}
            if self.name not in data[pp.ITERATE_SOLUTIONS]:
                data[pp.ITERATE_SOLUTIONS][self.name] = {}

            if self._time_depth > 0:
                if pp.TIME_STEP_SOLUTIONS not in data:
                    data[pp.TIME_STEP_SOLUTIONS] = {}
                if self.name not in data[pp.TIME_STEP_SOLUTIONS]:
                    data[pp.TIME_STEP_SOLUTIONS][self.name] = {}
        # boundaries
        for _, data in self.mdg.boundaries(return_data=True):
            if pp.ITERATE_SOLUTIONS not in data:
                data[pp.ITERATE_SOLUTIONS] = {}
            if self.name not in data[pp.ITERATE_SOLUTIONS]:
                data[pp.ITERATE_SOLUTIONS][self.name] = {}

            if self._time_depth > 0:
                if pp.TIME_STEP_SOLUTIONS not in data:
                    data[pp.TIME_STEP_SOLUTIONS] = {}
                if self.name not in data[pp.TIME_STEP_SOLUTIONS]:
                    data[pp.TIME_STEP_SOLUTIONS][self.name] = {}

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

    # Convenience properties/methods to access and progress values collectively

    @property
    def boundary_values(self) -> np.ndarray:
        """Property to access and store the value at the current time step, current
        iterate on **all** boundaries in the md grid.

        The getter fetches the values in the order imposed by the md-grid.

        The setter shifts iterate values backwards and set the given value as the most
        recent iterate.

        Note:
            This is a convenience functionality for :meth:`update_boundary_values`,
            which operates on all boundary grids.

            Hence, when setting boundary values this way, the user shifts their values
            in the **time sense**, copyig the current iterate value to the first,
            previous time step value.

        Parameters:
            val: ``shape=(num_boundary_cells,)``

                A new value to be set.

        Raises:
            ValueError: If the size of the value mismatches what is expected.
            KeyError: If no data was stored on a grid, but accessed.

        """
        vals = []
        for grid in self.mdg.boundaries():
            vals.append(self._fetch_values(grid, pp.ITERATE_SOLUTIONS, 0))
        if len(vals) > 0:
            return np.hstack(vals)
        else:
            return np.zeros(0, dtype=float)

    @boundary_values.setter
    def boundary_values(self, val: np.ndarray) -> None:
        shape = (sum([g.num_cells for g in self.mdg.boundaries()]),)
        assert val.shape == shape, (
            f"Need array of shape {shape}," + f" but {val.shape} given."
        )

        idx = 0
        for grid in self.mdg.boundaries():
            nc_d = grid.num_cells
            self.update_boundary_value(val[idx : idx + nc_d], grid)
            idx += nc_d

    @property
    def subdomain_values(self) -> np.ndarray:
        """Property to access and store the value at the current time step, current
        iterate on **all** subdomains in the md grid for convenience.

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
        for grid in self.mdg.subdomains():
            vals.append(self._fetch_values(grid, pp.ITERATE_SOLUTIONS, 0))
        if len(vals) > 0:
            return np.hstack(vals)
        else:
            return np.zeros(0, dtype=float)

    @subdomain_values.setter
    def subdomain_values(self, val: np.ndarray) -> None:
        shape = (self.mdg.num_subdomain_cells(),)
        assert val.shape == shape, (
            f"Need array of shape {shape}," + f" but {val.shape} given."
        )

        idx = 0
        for grid in self.mdg.subdomains():
            nc_d = grid.num_cells
            self.progress_iterate_values_on_grid(val[idx : idx + nc_d], grid)
            idx += nc_d

    @property
    def subdomain_derivatives(self) -> np.ndarray:
        """Property to access and store the derivatives at the current time step,
        current iterate on **all** subdomains in the md grid for convenience.

        The derivatives values are stored row-wise per dependency, column wise per
        subdomain.

        The getter fetches the values in the order imposed by the md-grid.

        The setter shifts iterate values backwards and set the given value as the most
        recent iterate.

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
        for grid in self.mdg.subdomains():
            vals.append(self._fetch_derivative_values(grid, pp.ITERATE_SOLUTIONS, 0))
        if len(vals) > 0:
            return np.hstack(vals)
        else:
            return np.zeros(self.num_dependencies, dtype=float)

    @subdomain_derivatives.setter
    def subdomain_derivatives(self, val: np.ndarray) -> None:
        shape = (self.num_dependencies, self.mdg.num_subdomain_cells())
        assert val.shape == shape, (
            f"Need array of shape {shape}," + f" but {val.shape} given."
        )

        idx = 0
        for grid in self.mdg.subdomains():
            nc_d = grid.num_cells
            self.progress_iterate_derivatives_on_grid(val[:, idx : idx + nc_d], grid)
            idx += nc_d

    @property
    def interface_values(self) -> np.ndarray:
        """Analogous to :meth:`subdomain_values`, but for all interfaces in the md-grid.

        Parameters:
            val: ``shape=(num_interface_cells,)``

                A new value to be set.

        """
        vals = []
        for grid in self.mdg.interfaces():
            vals.append(self._fetch_values(grid, pp.ITERATE_SOLUTIONS, 0))
        if len(vals) > 0:
            return np.hstack(vals)
        else:
            return np.zeros(0, dtype=float)

    @interface_values.setter
    def interface_values(self, val: np.ndarray) -> None:
        shape = (self.mdg.num_interface_cells(),)
        assert val.shape == shape, (
            f"Need array of shape {shape}," + f" but {val.shape} given."
        )

        idx = 0
        for grid in self.mdg.interfaces():
            nc_d = grid.num_cells
            self.progress_iterate_values_on_grid(val[idx : idx + nc_d], grid)
            idx += nc_d

    @property
    def interface_derivatives(self) -> np.ndarray:
        """Analogous to :meth:`subdomain_derivatives`, but for all interfaces in the
        md-grid.

        Parameters:
            val: ``shape=(num_dependencies, num_interface_cells)``

                A new value to be set.

        """
        vals = []
        for grid in self.mdg.interfaces():
            vals.append(self._fetch_derivative_values(grid, pp.ITERATE_SOLUTIONS, 0))
        if len(vals) > 0:
            return np.hstack(vals)
        else:
            return np.zeros(self.num_dependencies, dtype=float)

    @interface_derivatives.setter
    def interface_derivatives(self, val: np.ndarray) -> None:
        shape = (self.num_dependencies, self.mdg.num_interface_cells())
        assert val.shape == shape, (
            f"Need array of shape {shape}," + f" but {val.shape} given."
        )

        idx = 0
        for grid in self.mdg.interfaces():
            nc_d = grid.num_cells
            self.progress_iterate_derivatives_on_grid(val[:, idx : idx + nc_d], grid)
            idx += nc_d

    def progress_values_in_time(
        self, domains: Sequence[pp.Grid | pp.MortarGrid]
    ) -> None:
        """Shifts timestepping values backwards in times and sets the most recent
        iterate value as the most recent, previous time step value.

        Parameters:
            domains: Performs the progress on given list.

        Raises:
            ValueError: If ``time_step_depth`` at instantiation was set to zero.
            KeyError: If nothing is stored as the current time step (iterate index 0).

        """
        if self._time_depth < 1:
            raise ValueError(
                f"Cannot progress secondary expression {self.name} in time with"
                + f" time step depth set to zero."
            )
        for grid in domains:
            current_vals = self._fetch_values(grid, pp.ITERATE_SOLUTIONS, 0)
            self._shift_values(grid, pp.TIME_STEP_SOLUTIONS, current_vals)

    def progress_derivatives_in_time(
        self, domains: Sequence[pp.Grid | pp.MortarGrid]
    ) -> None:
        """Analogous to :meth:`progress_values_in_time`, but for derivatives."""
        if self._time_depth < 1:
            raise ValueError(
                f"Cannot progress secondary expression {self.name} in time with"
                + f" time step depth set to zero."
            )
        for grid in domains:
            current_vals = self._fetch_derivative_values(grid, pp.ITERATE_SOLUTIONS, 0)
            self._shift_derivatives(grid, pp.TIME_STEP_SOLUTIONS, current_vals)

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

        # If boundary values stored in time, shift them and store the current time step
        # (iterate idx 0) as the most recent previous time step
        if self._time_depth > 0:
            for t in range(self._time_depth - 1, 0, -1):
                val = data[pp.TIME_STEP_SOLUTIONS][self.name].get(t - 1, None)
                if val is not None:
                    data[pp.TIME_STEP_SOLUTIONS][self.name][t] = val
            new_prev_val = data[pp.ITERATE_SOLUTIONS][self.name].get(0, None)
            if new_prev_val is not None:
                data[pp.TIME_STEP_SOLUTIONS][self.name][0] = new_prev_val
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
            vd = data[loc][self.name].get(i - 1, None)
            if vd is not None:  # shift only data if available
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
            vd = data[loc][self.name].get(i - 1, None)
            if vd is not None:  # shift only data if available
                # if value data present, don't overwrite
                if i in data[loc][self.name]:
                    vd[0] = data[loc][self.name][i][0]
                data[loc][self.name][i] = vd

        if 0 in data[loc][self.name]:
            data[loc][self.name][0][1] = new_values
        else:
            data[loc][self.name][0] = [None, new_values]

    def progress_iterate_values_on_grid(
        self,
        new_values: np.ndarray,
        grid: pp.Grid | pp.MortarGrid,
    ) -> None:
        """Shifts the iterate values backwards and sets ``new_value`` as the most
        recent one on the given ``grid``.

        Raises:
            AssertionError: If ``new_vales`` is not of shape ``(grid.num_cells,)``.

        """
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
            AssertionError: If ``new_vales`` is not of shape
                ``(num_dependencies, grid.num_cells)``.

        """
        shape = (self.num_dependencies, grid.num_cells)
        assert new_values.shape == shape, (
            f"Need array of shape {shape}," + f" but {new_values.shape} given."
        )
        self._shift_derivatives(grid, pp.ITERATE_SOLUTIONS, new_values)
