"""This module contains AD functionality for represening complex terms for which an
evaluation directly with (forward) AD would be too expensive, or the values are provided
by some external computations.

The classes contained here are essentially wrappers for values provided by the user.

The base class for representing some term which depends on independent variables is
:class:`SecondaryExpression`. The user can call it using a sequence of grids to get a
representation as an AD operator on respective domains.

Note:
    Due to how AD parsing is implemented, the :class:`SecondaryExpression` requires a
    reference to the mixed-dimensional grid to store and later access the data.

:class:`SecondaryExpression` is a factory and management class for
:class:`SecondaryOperator`. The latter is the (actual) AD-compatible representation,
on some specified grids, at a specified time and iterate index.

Note:
    A connection between operator and factory class is established via
    :attr:`SecondaryOperator.fetch_data`. Since operators are in general *not aware* of
    the mixed-dimensional grid, a connection to the factory is required to fetch the
    data for the correct grids, at the correct time and iterate index.
    Therefore, the class :class:`SecondaryOperator` should not be instantiated directly,
    but only calling the factory class.

    Note that a :class:`SecondaryExpression` can be represented on boundaries,
    subdomains and interfaces. Each call to it creates a new instance of
    :class:`SecondaryOperator` which during parsing needs to fetch the right data for
    itself.

Example:

    Let's consider the set-up of a model, which needs to represent the fluid density as
    a :class:`SecondaryExpression` assuming it's computation is provided from some
    third-party package.

    The class follows the mixin approach, i.e. the variables pressure and temperature
    are provided by some other class as callables.

    .. code-block:: python

        class MyFluid:

            mdg: porepy.MixedDimensionalGrid

            pressure: Callable[
                [porepy.SubdomainsOrBoundaries], porepy.MixedDimensionalVariable
            ]

            temperature: Callable[
                [porepy.SubdomainsOrBoundaries], porepy.MixedDimensionalVariable
            ]

            def __init__(self):

                self.fluid_density: Callable[
                    [porepy.GridlikeSequence], porepy.Operator
                ] = SecondaryExpression(
                    name = 'fluid_density',
                    mdg = self.mdg,
                    dependencies = [self.pressure, self.temperature],
                    time_dependent = True,
                )

    The model will now have a callable ``fluid_density``, which returns an AD operator
    if called with some grids in the mixed-dimensional grid.

    The fluid density depends on pressure and temperature. If respective operators are
    parsed, they have two (cell-wise) derivatives with respect to the dependencies and
    the result is an AD array.

    Since the fluid density appears in the accumulation term of any balance equation,
    it is ``time_dependent`` and the user can update boundary values or progress
    values (and derivative values) in time.
    To avoid unintentional, possibly memory-heavy storage of data, the user must
    declare the expression explicitly as time-dependent. If the expression is not
    time-dependent, the update and progress methods will do nothing.
    (Except for the boundary values, since the current boundary values are stored at
    the iterate index 0.)

    Arbitrary iterate depth is supported by default.

    To get a representation of the accumulation term in all subdomains, the user has now
    the option to make the following call somewhere else in the model mixins:

    >>> rho = self.fluid_density(self.mdg.subdomains())
    >>> rho_prev = rho.previous_timestep()

    Since the fluid density may also appear as a non-linear weight in some advective
    flux, boundary values are also required

    >>> rho_bc = self.fluid_density(self.mdg.boundaries())

    Values can be updated in time in the solution strategy using

    >>> new_boundary_values: numpy.array = ...
    >>> self.fluid_density.boundary_values = new_boundary_values

    Note, that the update is performed on the instance of
    :class:`SecondaryExpression`, not on its products :class:`SecondaryOperator`.

    Iterate values at the current time step (to be solved for) can be updated in a
    similar fashion:

    >>> new_vals: numpy.array = ...
    >>> new_derivatives: numpy.array = ...
    >>> self.fluid_density.subdomain_values = new_vals
    >>> self.fluid_density.subdomain_derivatives = new_derivatives

    In the next parsing of the fluid density at the current time and iterate, an
    AD array will be returned with values equaling ``new_vals`` and a sparse CSR matrix
    with diagonal blocks containing the two derivative values for pressure and
    temperature in their respective columns.

    Note that the user must inform the :class:`SecondaryExpression` about it's number
    of dofs (similar to when creating variables in the equation system).
    This information is used to support the user and raise exceptions if various
    ``new_values`` are not of expected shape.

"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, cast

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.ad.forward_mode import AdArray
from porepy.numerics.ad.operators import Operator

from .functions import FloatType
from .operators import IterativeOperator, TimeDependentOperator

__all__ = [
    "SecondaryOperator",
    "SecondaryExpression",
]


class SecondaryOperator(
    TimeDependentOperator,
    IterativeOperator,
):
    """Operator representing a :class:`SecondaryExpression` in AD operator form on
    specified subdomains or interfaces, at a time or iterate index.

    Not meant to be instantiated directly, only by calling the factory class
    :class:`SecondaryExpression`.

    The secondary operator is essentially like a
    :class:`~porepy.numerics.ad.operator_functions.Function`. Upon evaluation it fetches
    the stored values and the derivative values.
    The derivative values are inserted into the Jacobians of the first-order
    dependencies (identity blocks).

    Parameters:
        name: Name of the called :class:`SecondaryExpression`.
        domains: Arguments to its call.
        children: The first-order dependencies of the called
            :class:`SecondaryExpression` in AD form (defined on the same ``domains``).

    """

    def __init__(
        self,
        name: str,
        domains: Sequence[pp.Grid] | Sequence[pp.MortarGrid],
        children: Sequence[pp.ad.Variable],
    ) -> None:

        super().__init__(
            name=name,
            domains=domains,
            operation=pp.ad.Operator.Operations.evaluate,
            children=children,
        )

        self.fetch_data: Callable[[SecondaryOperator, pp.GridLike, bool], np.ndarray]
        """A function returning the stored data for a secondary operator on a grid,
        which is time step and iterate index dependent.

        This function is assigned by the factory class :class:`SecondaryExpression`
        which has a reference to the md-grid and must not be touched by the user.

        """

    def __repr__(self) -> str:
        """String representation giving information on name, time and iterate index, as
        well as domains and dependencies."""

        msg = (
            f"Secondary operator {self.name}.\n"
            + f"\nDefined on {len(self._domains)} {self._domain_type}.\n"
            + f"Dependent on {len(self.children)} independent expressions.\n"
        )

        if self.is_previous_time:
            msg += f"Evaluated at the previous time step {self.time_step_index}.\n"
        elif self.is_previous_iterate:
            msg += f"Evaluated at the previous iterate {self.iterate_index}.\n"

        return msg

    def __call__(self, *args: Operator) -> Operator:
        """By calling this operator with new children, the user can change the
        structure of the Jacobian.

        The dependencies/children are replaced and hence the non-trivial entries in
        the Jacobian in the md-setting.

        Note:
            Call this operator only if you know what you are doing.

            It will reset the time step and iterate indices to current time and iterate.

        """
        op = SecondaryOperator(
            name=self.name,
            domains=self.domains,  # type:ignore[arg-type]
            children=args,  # type:ignore[arg-type]
        )
        op.fetch_data = self.fetch_data
        return op

    def previous_timestep(self, steps: int = 1) -> SecondaryOperator:
        """Secondary operators have children which also need to be obtained at
        the previous time step."""

        op = super().previous_timestep(steps=steps)
        op.children = [child.previous_timestep(steps=steps) for child in self.children]
        return op

    def previous_iteration(self, steps: int = 1) -> SecondaryOperator:
        """Secondary operators have children which also need to be obtained at
        the previous iteration."""
        op = super().previous_iteration(steps=steps)
        op.children = [child.previous_iteration(steps=steps) for child in self.children]
        return op

    def func(self, *args: FloatType) -> float | np.ndarray | AdArray:
        """See :meth:`~porepy.numerics.ad.operator_functions.AbstractFunction.func`.

        Note:
            This class cannot inherit from
            :class:`~porepy.numerics.ad.operator_functions.AbstractFunction` because
            that parent would block arithmetic overloads.
            But it uses the operation ``evaluate`` and the ``AbstractFunction.func``
            with ``self`` as an explicit argument.

        """
        # mypy complains that self is not an instance of AbstractFunction because of the
        # hack
        return pp.ad.AbstractFunction.func(self, *args)  # type:ignore[arg-type]

    def get_values(self, *args: float | np.ndarray | AdArray) -> np.ndarray:
        """Fetches the values stored for this secondary operator at its time or iterate
        index."""
        return np.hstack([self.fetch_data(self, g, False) for g in self.domains])

    def get_jacobian(self, *args: float | np.ndarray | AdArray) -> sps.spmatrix:
        """Fetches the derivative values stored for this secondary operator at its time
        or iterate index.

        Uses the structure of the Jacobians of arguments to insert the values,
        assuming the Jacobians contain only identity blocks.

        """

        derivatives = np.hstack([self.fetch_data(self, g, True) for g in self.domains])

        # list of jacs per dependency, assuming porepy.ad makes consistent shapes
        jacs: list[sps.csr_matrix] = []

        for arg, d_i in zip(args, derivatives):
            if isinstance(arg, AdArray):
                # At this point we assume that the derivative has only an identity block
                # Shape checks for derivatives are done in the factory class
                jacs.append(
                    sps.csr_matrix((d_i, arg.jac.nonzero()), shape=arg.jac.shape)
                )

        return sum(jacs).tocsr()


class SecondaryExpression:
    """A representation of some dependent quantity in the PorePy modelling and AD
    framework, meant for terms where the evaluation is done elsewhere and then stored.

    This is a factory class, callable using some domains in the md-setting to create
    AD operators representing this expression on respective domains.

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
    The operator has instances of
    :class:`~porepy.numerics.ad.operators.MixedDimensionalVariable` as children
    (dependencies).

    The secondary operator supports the notion of previous timesteps and iterate
    values. Updates of respective values are handled by this factory class.

    **Call with empty domain list:**

    This functionality is implemented for completeness reasons, such that the secondary
    expression can also be called on an empty list of domains. In this case the property
    returns an empty array wrapped as an Ad array.

    This functionality is implemented since general PorePy models implement equations
    in the md-setting. It makes this class compatible for models without fractures or
    interfaces.

    Notes:

        1. The ``dependencies`` are assumed to be of first order. I.e., independent
           variables. Nested dependencies are not supported.
        2. Data is stored in a grid's data dictionary. This class generates an extra
           key based on its name to store the derivative values using PorePy's AD
           utilities.
        3. This class provides management functionality to update values grid-wise and
           globally, in both iterative and time sense, as well as derivative values.
           On boundaries, it does not support derivatives and no storage of data in the
           iterative sense, only in time.

    Parameters:
        name: Assigned name of the expression. Used to name operators and to store
            values in the data dictionaries
        mdg: The mixed-dimensional grid on which the expression is defined.
        dependencies: A sequence of callables/ constructors for independent variables on
            which the expression depends. The order passed here is reflected in the
            order of stored derivative values.

            When calling the secondary expression on some grids, it is expected that the
            dependencies are defined there.
        time_dependent: ``default=False``

            If flagged as time-dependent, methods updating/progressing values in time
            will shift values backwards and store them to be accessed in time-stepping
            schemes. Otherwise time-dependent storage is deactivated and respective
            methods will only update the current time value (stored as current iterate).
        dof_info: ``default=None``

            See
            :meth:`~porepy.numerics.ad.equation_system.EquationSystem.create_variables`.

            The number of DOFs of this expression is used to validate the shape of
            values and derivative values set.
            Defaults to cell-wise, scalar expression.

    Raises:
        ValueError: If there are no ``*dependencies``. The user should use other
            solutions in ``porepy.ad`` for this case.

    """

    def __init__(
        self,
        name: str,
        mdg: pp.MixedDimensionalGrid,
        dependencies: Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]],
        time_dependent: bool = False,
        dof_info: Optional[dict[pp.ad.equation_system.GridEntity, int]] = None,
    ) -> None:

        if len(dependencies) == 0:
            raise ValueError("Secondary expressions must have dependencies.")

        if dof_info is None:
            dof_info = {"cells": 1}

        # help mypy with default values
        dof_info = cast(dict[pp.ad.equation_system.GridEntity, int], dof_info)

        self._dependencies: Sequence[
            Callable[[pp.GridLikeSequence], pp.ad.Variable]
        ] = dependencies
        """Sequence of callable first order dependencies. Called when constructing
        operators on domains."""

        self._time_dependent: bool = time_dependent
        """Flag to activate storage of data backwards in time, passed at instantiation.
        """
        self._name: str = name
        """See :meth:`name`."""

        self._dof_info: dict[pp.ad.equation_system.GridEntity, int] = dof_info
        """Passed at insantiation, with default value leading to scalar, cell-wise dofs.
        """

        self.mdg: pp.MixedDimensionalGrid = mdg
        """Mixed-dimensional domain passed at instantiation."""

    def __call__(self, domains: pp.GridLikeSequence) -> pp.ad.Operator:
        """Call this operator on a subset of grid in the md-grid to get a property
        restricted to the passed domains.

        1. If called on an empty list, a wrapped, empty array is returned.
        2. If called on boundary grids, a time-dependent dense array is returned.
        3. If called on subdomains on interfaces, a :class:`SecondaryOperator` is
           returned.

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
            return pp.ad.TimeDependentDenseArray(self.name, domains)
        # On subdomains or interfaces, create the secondary operators
        elif all(isinstance(g, pp.Grid) for g in domains) or all(
            isinstance(g, pp.MortarGrid) for g in domains
        ):
            # for mypy
            domains_ = cast(Sequence[pp.Grid] | Sequence[pp.MortarGrid], domains)
            children = [child(domains_) for child in self._dependencies]

            # Check if first-order dependency
            assert all(isinstance(child, pp.ad.Variable) for child in children), (
                "Secondary expressions must depend on independent variables, not"
                + f" {[type(c) for c in children]}."
            )

            # always start with operator at current time step, current iterate
            op = SecondaryOperator(
                name=self.name,
                domains=domains_,
                children=children,
            )

            # assign the function which extracts the data
            op.fetch_data = self.fetch_data

            return op
        else:
            raise ValueError(
                f"Unsupported domain configuration {[type(g) for g in domains]}."
            )

    def fetch_data(
        self, op: SecondaryOperator, grid: pp.GridLike, get_derivatives: bool = False
    ) -> np.ndarray:
        """Function fetching the data stored for this secondary expression, represented
        in Ad form by ``op``.

        ``op`` has a time step or iterate index, which specifies which data should be
        fetched.

        This function is assigned to :attr:`SecondaryOperator.fetch_data`
        (a work-around such that the Ad operator has no reference to the md-grid like
        the other operators).

        Note:
            This method uses PorePy's utility functions to get data stored in a grid's
            data dictionary. Respective errors will be raised if no data stored.

        Parameters:
            op: An AD representation of this expression at a specific time and iterate.
            grid: A grid in the md-grid for which the data should be fetched.
            get_derivatives: If True, it fetches the derivative values, otherwise it
                fetches the values.

        Returns:
            If ``get_derivatives==True``, the resulting array is of shape
            ``(num_dependencies, num_dofs)``, with ``num_dofs`` being the number of DOFs
            of this expression.

            Otherwise the resulting array is of shape ``(num_dofs,)``.

        """
        # arguments for utility function
        kwargs: dict[Any, Any] = dict()
        if get_derivatives:
            kwargs["name"] = self._name_diffs
        else:
            kwargs["name"] = self.name

        kwargs["data"] = self._data_of(grid)

        if op.is_previous_time:
            kwargs["time_step_index"] = op.time_step_index
        else:
            kwargs["iterate_index"] = op.iterate_index

        return pp.get_solution_values(**kwargs)

    def _data_of(self, grid: pp.GridLike) -> dict:
        """Convenience function to get the data dictionary of any grid."""
        if isinstance(grid, pp.Grid):
            data = self.mdg.subdomain_data(grid)
        elif isinstance(grid, pp.MortarGrid):
            data = self.mdg.interface_data(grid)
        elif isinstance(grid, pp.BoundaryGrid):
            data = self.mdg.boundary_grid_data(grid)
        else:
            raise ValueError(f"Unknown grid type {type(grid)}.")
        return data

    @property
    def num_dependencies(self) -> int:
        """Number of first order dependencies of this operator, passed at instantiation.

        This determines the number of required derivative values."""
        return len(self._dependencies)

    def num_dofs_on_grid(self, grid: pp.GridLike) -> int:
        """Computes the numbe of DOFs based on the information provided during
        instantiation.

        Note:
            On boundary grids and interfaces, only cell-wise DOFs are expected.
            On subdomains, DOFs can be cell-, node- and face-wise.

        Parameters:
            grid: Any grid in the md-grid.

        Returns:
            For boundary grids and interfaces it returns ``grid.num_cells`` multiplied
            with ``dof_info['cells']``.
            For subdomains, it returns information based on ``dof_info``,
            number of cells, number of faces and number of nodes in ``grid``.

        Raises:
            TypeError: If ``grid`` is neither a boundary, interface or subdomain.

        """
        if isinstance(grid, (pp.BoundaryGrid, pp.MortarGrid)):
            # NOTE using default value of 1, because this is the general default value
            # and only cells are supported on boundaries and interfaces
            return self._dof_info.get("cells", 1) * grid.num_cells
        elif isinstance(grid, pp.Grid):
            # NOTE cannot use default value of scalar, cell-wise, to not mess with
            # cases where the user defines only node- or face-wise dofs.
            return (
                self._dof_info.get("cells", 0) * grid.num_cells
                + self._dof_info.get("faces", 0) * grid.num_faces
                + self._dof_info.get("nodes", 0) * grid.num_nodes
            )
        else:
            raise TypeError(f"Unsupported type of grid {type(grid)}.")

    @property
    def name(self) -> str:
        """Name of this expression assigned at instantiation."""
        return self._name

    @property
    def _name_diffs(self) -> str:
        """Private property giving the name under which the **derivative** values of
        this expression are stored in the grid data dictionaries"""
        return f"{self.name}_DERIVATIVES"

    # Convenience properties/methods to access and progress values collectively

    @property
    def boundary_values(self) -> np.ndarray:
        """Property to access and set the boundary value at the current time step,
        on **all** boundaries in the md grid.

        The getter fetches the values in the order imposed by the md-grid.
        The setter slices the values according to :meth:`num_dofs_on_grid` for each
        boundary grid in the md-grid.

        Note:
            This is a convenience functionality for :meth:`update_boundary_values`,
            which operates on all boundary grids.

            Hence, the setter shifts **time step** values backwards and sets the given
            value as the **curent iterate** (current time).

            Most importantly, it has a different paradigm than the property setters and
            getters for subdomains and interfaces, which operate only with iterate
            values.

        Parameters:
            val: A new value with shape ``(N,)`` to be set. Note that ``N`` is
                calculated from the information passed in ``dof_info`` during
                instantiation, and the number of cells in all boundary grids.

        Returns:
            If boundary values are set, it returns the one for the current time
            (iterate index = 0). The shape is consistent with ``val``.

        """
        vals = []
        for _, data in self.mdg.boundaries(return_data=True):
            vals.append(pp.get_solution_values(self.name, data, iterate_index=0))
        if len(vals) > 0:
            return np.hstack(vals)
        else:
            return np.zeros(0, dtype=float)

    @boundary_values.setter
    def boundary_values(self, val: np.ndarray) -> None:

        i = 0
        for grid in self.mdg.boundaries():
            n = self.num_dofs_on_grid(grid)
            self.update_boundary_values(val[i : i + n], grid)
            i += n

    @property
    def subdomain_values(self) -> np.ndarray:
        """Property to access and store the value at the current time step, current
        iterate on **all** subdomains in the md-grid for convenience.

        The getter fetches the values in the order imposed by the md-grid.
        The setter shifts iterate values backwards and sets the given value as the most
        recent iterate.

        Parameters:
            val: A new value with shape ``(N,)`` to be set. Note that ``N`` is
                calculated from the information passed in ``dof_info`` during
                instantiation, and the number of cells, faces and nodes in all
                subdomains.

        Returns:
            The current iterate value of same shape as ``val``.

        """
        vals = []
        for _, data in self.mdg.subdomains(return_data=True):
            vals.append(pp.get_solution_values(self.name, data, iterate_index=0))
        if len(vals) > 0:
            return np.hstack(vals)
        else:
            return np.zeros(0, dtype=float)

    @subdomain_values.setter
    def subdomain_values(self, val: np.ndarray) -> None:

        i = 0
        for grid in self.mdg.subdomains():
            n = self.num_dofs_on_grid(grid)
            self.progress_iterate_values_on_grid(val[i : i + n], grid)
            i += n

    @property
    def subdomain_derivatives(self) -> np.ndarray:
        """Property to access and store the derivatives at the current iteration of the
        current time step, on **all** subdomains in the md grid for convenience.

        The derivatives values are stored row-wise per dependency, column wise per
        subdomain.

        The getter fetches the values in the order imposed by the md-grid.
        The setter shifts iterate values backwards and sets the given value as the
        current iterate.

        Important:
            The order of derivatives should reflect the order of ``dependencies``
            passed at instantiation.

        Parameters:
            val: ``shape=(num_dependencies, N)``

                A new value to be set. Note that ``N`` must be consistent with the shape
                of values for :meth:`subdomain_values`.

        Returns:
            The current iterate derivative values of same shape as ``val``.

        """
        vals = []
        for _, data in self.mdg.subdomains(return_data=True):
            vals.append(pp.get_solution_values(self._name_diffs, data, iterate_index=0))
        if len(vals) > 0:
            return np.hstack(vals)
        else:
            return np.zeros(self.num_dependencies, dtype=float)

    @subdomain_derivatives.setter
    def subdomain_derivatives(self, val: np.ndarray) -> None:
        i = 0
        for grid in self.mdg.subdomains():
            n = grid.num_cells
            self.progress_iterate_derivatives_on_grid(val[:, i : i + n], grid)
            i += n

    @property
    def interface_values(self) -> np.ndarray:
        """Analogous to :meth:`subdomain_values`, but for all interfaces in the md-grid.

        Parameters:
            val: ``shape=(M,)``

                A new value to be set. Note that ``M`` is
                calculated from the information passed in ``dof_info`` during
                instantiation, and the number of cells on all interfaces.

        """
        vals = []
        for _, data in self.mdg.interfaces(return_data=True):
            vals.append(pp.get_solution_values(self.name, data, iterate_index=0))
        if len(vals) > 0:
            return np.hstack(vals)
        else:
            return np.zeros(0, dtype=float)

    @interface_values.setter
    def interface_values(self, val: np.ndarray) -> None:
        i = 0
        for grid in self.mdg.interfaces():
            n = grid.num_cells
            self.progress_iterate_values_on_grid(val[i : i + n], grid)
            i += n

    @property
    def interface_derivatives(self) -> np.ndarray:
        """Analogous to :meth:`subdomain_derivatives`, but for all interfaces in the
        md-grid.

        Parameters:
            val: ``shape=(num_dependencies, M)``

                A new value to be set, with ``M`` as for :meth:`interface_values`.

        """
        vals = []
        for _, data in self.mdg.interfaces(return_data=True):
            vals.append(pp.get_solution_values(self._name_diffs, data, iterate_index=0))
        if len(vals) > 0:
            return np.hstack(vals)
        else:
            return np.zeros(self.num_dependencies, dtype=float)

    @interface_derivatives.setter
    def interface_derivatives(self, val: np.ndarray) -> None:

        i = 0
        for grid in self.mdg.interfaces():
            n = grid.num_cells
            self.progress_iterate_derivatives_on_grid(val[:, i : i + n], grid)
            i += n

    # Methods to progress values and derivatives in time on subdomains or
    # interfaces (don't need new values)

    def progress_values_in_time(
        self, domains: Sequence[pp.Grid | pp.MortarGrid]
    ) -> None:
        """Shifts timestepping values backwards in times and sets the current
        iterate value as the (first) previous time step value.

        Does this only if the expression was instatiated as time-dependent.

        Parameters:
            domains: Performs the progress on given sequence of domains.

        """
        if self._time_dependent:
            for grid in domains:
                data = self._data_of(grid)
                pp.shift_solution_values(self.name, data, pp.TIME_STEP_SOLUTIONS)
                current_vals = pp.get_solution_values(self.name, data, iterate_index=0)
                pp.set_solution_values(self.name, current_vals, data, time_step_index=1)

    def progress_derivatives_in_time(
        self, domains: Sequence[pp.Grid | pp.MortarGrid]
    ) -> None:
        """Analogous to :meth:`progress_values_in_time`, but for derivatives."""
        if self._time_dependent:
            for grid in domains:
                data = self._data_of(grid)
                pp.shift_solution_values(self._name_diffs, data, pp.TIME_STEP_SOLUTIONS)
                current_vals = pp.get_solution_values(
                    self._name_diffs, data, iterate_index=0
                )
                pp.set_solution_values(
                    self._name_diffs, current_vals, data, time_step_index=1
                )

    # Methods operating on single grids

    def update_boundary_values(
        self, values: np.ndarray, boundary_grid: pp.BoundaryGrid
    ) -> None:
        """Function to update the value of the secondary expression on the boundary.

        The update process of boundary values is different from the process on
        subdomains and interfaces.

        1. Boundary values have only a single iterate value (current time step)
        2. Boundary values can have multiple previous time steps, if expression is
           time-dependent.

        Therefore, values for the current time can always be set, but values will be
        shifted backwards in time only if the quantity is time-dependent.

        Parameters:
            values: ``shape=(N,)``

                A new value to be set for the boundary. ``N`` is computed from the
                information passed in ``dof_info`` at instantiation and the number of
                cells in ``boundary_grid``.
            boundary_grid: A boundary grid in the mixed-dimensional domain.

        Raises:
            AssertionError: If ``values`` is not of the expected shape.

        """
        shape = (self.num_dofs_on_grid(boundary_grid),)
        assert values.shape == shape, (
            f"Need array of shape {shape}," + f" but {values.shape} given."
        )
        data = self._data_of(boundary_grid)

        # If boundary values stored in time, shift them and store the current time step
        # (iterate idx 0) as the most recent previous time step
        if self._time_dependent:
            pp.shift_solution_values(self.name, data, pp.TIME_STEP_SOLUTIONS)
            # NOTE avoid pp.get_solution_values for the case when values are set
            # the first time.
            try:
                new_prev_val = data[pp.ITERATE_SOLUTIONS][self.name][0]
            except KeyError:
                pass
            else:
                pp.set_solution_values(self.name, new_prev_val, data, time_step_index=1)
        # set the given value as the new single iterate value (current time)
        pp.set_solution_values(self.name, values, data, iterate_index=0)

    def progress_iterate_values_on_grid(
        self,
        values: np.ndarray,
        grid: pp.Grid | pp.MortarGrid,
    ) -> None:
        """Shifts the iterate values backwards in the iterate sense and sets ``values``
        as the current iterate values.

        Parameters:
            values: ``shape=(N,)``

                A new value to be set on the ``grid``. ``N`` is computed from the
                information passed in ``dof_info`` at instantiation and the number of
                cells in ``grid`` (and faces and nodes if subdomain).
            grid: A subdomain or interface in the mixed-dimensional domain.

        Raises:
            AssertionError: If ``values`` is not of the expected shape.

        """
        shape = (self.num_dofs_on_grid(grid),)
        assert values.shape == shape, (
            f"Need array of shape {shape}," + f" but {values.shape} given."
        )
        data = self._data_of(grid)
        pp.shift_solution_values(self.name, data, pp.ITERATE_SOLUTIONS)
        pp.set_solution_values(self.name, values, data, iterate_index=0)

    def progress_iterate_derivatives_on_grid(
        self,
        values: np.ndarray,
        grid: pp.Grid | pp.MortarGrid,
    ) -> None:
        """Analogous to :meth:`progress_iterate_values_on_grid`, but for derivative
        values.

        Parameters:
            values: ``shape=(num_dependencies, N)``

                A new value to be set on the ``grid``. ``N`` must be consistent with
                the shape in :meth:`progress_iterate_values_on_grid`.
            grid: A subdomain or interface in the mixed-dimensional domain.

        Raises:
            AssertionError: If ``values`` is not of the expected shape.

        """
        shape = (self.num_dependencies, self.num_dofs_on_grid(grid))
        assert values.shape == shape, (
            f"Need array of shape {shape}," + f" but {values.shape} given."
        )
        data = self._data_of(grid)
        pp.shift_solution_values(self._name_diffs, data, pp.ITERATE_SOLUTIONS)
        pp.set_solution_values(self._name_diffs, values, data, iterate_index=0)
