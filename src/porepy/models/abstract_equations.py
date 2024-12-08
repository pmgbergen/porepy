"""Module containing classes representing generalized equations and variables.

- :class:`EquationMixin`: Base equation class providing the interface for setting
    equations. This interface is used by the solution strategy mixin.
- :class:`BalanceEquation`: Intermediate base class for vector and scalar balance
    equations.
- :class:`LocalElimination`: Intermediate base class for local equations which
    represent the elimination of some variable by a correlation/function dependent on
    other variables.
- :class:`VariableMixin`: Base class for variables, providing the interface for
    creating variables. This interface is used by the solution strategy mixin.

"""

from __future__ import annotations

from functools import cached_property
from typing import Callable, Sequence, Union, cast

import numpy as np

import porepy as pp
from porepy.numerics.ad.equation_system import GridEntity


class EquationMixin(pp.PorePyModel):
    """General class for equations defining an interface to introduce equations into
    a model.

    Example:
        When creating multi-physics models, the order of equations will reflect
        the order of inheritance of individual equation classes.

        Say we have to equation classes combined into one multi-physics model:

        .. code::python

            class Equation1(EquationMixin):

                def set_equations(self):
                    super().set_equation()
                    # proceed to set equation 1

            class Equation2(EquationMixin):

                def set_equations(self):
                    super().set_equation()
                    # proceed to set equation 2

            class ModelEquations(Equation1, Equation2):
                pass

        Notice that in both ``set_equations`` the super-call comes first.
        The way ``super()`` works now is, that ``Equation1`` will be called first,
        which in return executes the code of ``Equation2`` first.
        I.e., the order of equations introduced into the system is the reverse order
        of equation classes in the inheritance tree of the combined class.

        Notice also, that the combined class ``ModelEquations`` does not need an
        implementation of ``set_equations``.

    """

    def set_equations(self) -> None:
        """Method to be overridden to set equations on some grid.

        The base class method does nothing and is implemented to provide the right
        signature and to help the super call to resolve the setting of equations in
        multi-physics models.

        """
        pass


class BalanceEquation(EquationMixin):
    """Generic class for vector balance equations.

    In the only known use case, the balance equation is the momentum balance equation,

        d_t(momentum) + div(stress) - source = 0,

    with momentum frequently being zero. All terms need to be specified in order to
    define an equation.

    """

    def balance_equation(
        self,
        subdomains: list[pp.Grid],
        accumulation: pp.ad.Operator,
        surface_term: pp.ad.Operator,
        source: pp.ad.Operator,
        dim: int,
    ) -> pp.ad.Operator:
        """Balance equation that combines an accumulation and a surface term.

        The balance equation is given by
        .. math::
            d_t(accumulation) + div(surface_term) - source = 0.

        Parameters:
            subdomains: List of subdomains where the balance equation is defined.
            accumulation: Operator for the cell-wise accumulation term, integrated over
                the cells of the subdomains.
            surface_term: Operator for the surface term (e.g. flux, stress), integrated
                over the faces of the subdomains.
            source: Operator for the source term, integrated over the cells of the
                subdomains.
            dim: Spatial dimension of the balance equation.

        Returns:
            Operator for the balance equation.

        """
        dt_operator = pp.ad.time_derivatives.dt
        dt = self.ad_time_step
        div = pp.ad.Divergence(subdomains, dim=dim)
        return dt_operator(accumulation, dt) + div @ surface_term - source

    def volume_integral(
        self,
        integrand: pp.ad.Operator,
        grids: Union[list[pp.Grid], list[pp.MortarGrid]],
        dim: int,
    ) -> pp.ad.Operator:
        """Numerical volume integral over subdomain or interface cells.

        Includes cell volumes and specific volume.

        Parameters:
            integrand: Operator for the integrand. Assumed to be a cell-wise scalar
                or vector quantity, cf. :code:`dim` argument.
            grids: List of subdomains or interfaces to be integrated over.
            dim: Spatial dimension of the integrand. dim = 1 for scalar problems,
                dim > 1 for vector problems.

        Returns:
            Operator for the volume integral.

        Raises:
            ValueError: If the grids are not all subdomains or all interfaces.

        """
        assert all(isinstance(g, pp.MortarGrid) for g in grids) or all(
            isinstance(g, pp.Grid) for g in grids
        ), "Grids must be either all subdomains or all interfaces."

        # First account for cell volumes.
        cell_volumes = self.wrap_grid_attribute(grids, "cell_volumes", dim=1)

        # Next, include the effects of reduced dimensions, expressed as specific
        # volumes.
        if dim == 1:
            # No need to do more for scalar problems
            return cell_volumes * self.specific_volume(grids) * integrand
        else:
            # For vector problems, we need to expand the volume array from cell-wise
            # scalar values to cell-wise vectors. We do this by left multiplication with
            # e_i and summing over i.
            basis = self.basis(grids, dim=dim)

            volumes_nd = pp.ad.sum_operator_list(
                [e @ (cell_volumes * self.specific_volume(grids)) for e in basis]
            )

            return volumes_nd * integrand


class LocalElimination(EquationMixin):
    """Mixin to introduce local equations on some grid.

    Provides functionality to close a model with dangling variables by introducing
    a closure of form :math:`x - \\tilde{x}(\\dots) = 0`, where :math:`\\tilde{x}` is
    some function to be evaluated depending on specified variables.

    Note:
        This elimination happens locally in time and space. No values backwards in time
        and iterate sense are stored for :math:`\\tilde{x}`. This class assumes that
        previous iterate and time steps exist only for :math:`x` in some other equation.

    Important:
        For this mixin to work reliably, it must be above IC, BC and solution strategy
        mixins in the inheritance order.
        Since it fetches values for variables, update routines for IC, BC and solution
        strategies (iterate values) must be executed before any routine here is
        attached.

    Example:
        Considering a two-phase flow, the gas saturation could be given by some
        external function :math:`\\tilde{s}_g(p, T)`. While the liquid saturation is
        naturally eliminated by :math:`1 - s_g``, the system still has a dangling
        variable :math:`s_g`. This class can be used to close the system by using
        :meth:`eliminate_locally` and a numerical function :math:`\\tilde{s}_g(p, T)`.
        That function can be a table look-up, some correlations, or even pre-computed
        equilibrium values.

    """

    @cached_property
    def __local_eliminations(self) -> dict[
        str,
        tuple[
            pp.ad.MixedDimensionalVariable,
            pp.ad.SurrogateFactory,
            Callable[..., tuple[np.ndarray, np.ndarray]],
            Sequence[pp.Grid] | Sequence[pp.MortarGrid],
            Sequence[pp.BoundaryGrid],
        ],
    ]:
        """Storage of configurations of local eliminations.

        The key is the name of the local equation introduced into the model.
        The value is a tuple containing

        1. references to the eliminated variable,
        2. the surrogate factory itself,
        3. a callable which is the functional representation of the elimination,
        4. a sequence of internal grids on which the variable was eliminated,
        5. and a sequence of boundary grids for consistent BC.

        """
        return {}

    def eliminate_locally(
        self,
        independent_quantity: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator],
        dependencies: Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]],
        func: Callable[..., tuple[np.ndarray, np.ndarray]],
        domains: Sequence[pp.Grid | pp.MortarGrid | pp.BoundaryGrid],
        equations_per_grid_entity: dict[GridEntity, int] = {"cells": 1},
    ) -> None:
        """Method to add a secondary equation eliminating a variable by some
        constitutive law depending on *other* variables.

        The passed variables are assumed to be formally independent quantities in the
        AD sense.

        For a formally independent quantity :math:`\\varphi`, this method introduces
        a secondary equation :math:`\\varphi - \\hat{\\varphi}(x) = 0`, with :math:`x`
        denoting the ``dependencies``.

        It uses the :class:`~porepy.numerics.ad.surrogate_operator.SurrogateFactory` to
        provide AD representations of :math:`\\hat{\\varphi}` and to update its values
        and derivatives using ``func`` in the solutionstrategy.

        Note:
            While any type of grid can be passed in ``domains``, the equation is
            formally introduced **only** on subdomains and boundaries, if any.
            On boundary grids, the framework of local elimination provides a convenience
            functionality to automatically compute values for the eliminated variable
            on the boundary using ``func`` and store them.

            This is for consistency reasons and to help the user with not having to
            write methods for this step.

        Parameters:
            independent_quantity: AD representation :math:`\\varphi`, callable on some
                grids.
            dependencies: First order dependencies (variables) through which
                :math:`\\varphi` is expressed locally.
            func: A numerical function which computes the values of
                :math:`\\hat{\\varphi}(x)` and its derivatives.

                The return value must be a 2-tuple, containing a 1D value array and a
                2D derivative value array. The shape of the value array must be
                ``(N,)``, where ``N`` is consistent with ``dofs``, and the shape of
                the derivative value array must be ``(M,N)``, where ``M`` denotes the
                number of first order dependencies (``M == len(dependencies)``).

                The order of arguments for ``func`` must correspond with the order in
                ``dependencies``.
            domains: A Sequence of grids on which the quantity and its dependencies are
                defined and on which the equation should be introduces.
                Used to call ``independent_quantity`` and ``dependencies``.
            equations_per_grid_entity: ``default={'cells':1}``

                Argument for when adding above equation to the equation system and
                creating a surrogate factory.

        """
        # separate these two because Boundary values for independent quantities are
        # stored differently
        non_boundaries = cast(
            pp.SubdomainsOrBoundaries,
            [g for g in domains if isinstance(g, (pp.Grid, pp.MortarGrid))],
        )

        # cast to Variable, because most models have functions returning variables
        # like pressure and temperature, typed as returning Operator
        sec_var = cast(
            pp.ad.MixedDimensionalVariable, independent_quantity(non_boundaries)
        )
        g_ids = [d.id for d in non_boundaries]

        sec_expr = pp.ad.SurrogateFactory(
            name=f"surrogate_for_{sec_var.name}_on_grids_{g_ids}",
            mdg=self.mdg,
            dependencies=dependencies,
            dof_info=equations_per_grid_entity,
        )

        equ_name = f"elimination_of_{sec_var.name}_on_grids_{g_ids}"
        local_equ = sec_var - sec_expr(non_boundaries)
        local_equ.set_name(equ_name)
        self.equation_system.set_equation(
            local_equ,
            cast(list[pp.Grid] | list[pp.MortarGrid], non_boundaries),
            equations_per_grid_entity,
        )

        self._add_local_elimination(
            equ_name, sec_var, sec_expr, func, cast(pp.GridLikeSequence, domains)
        )

    def _add_local_elimination(
        self,
        local_equation_name: str,
        primary: pp.ad.MixedDimensionalVariable,
        expression: pp.ad.SurrogateFactory,
        func: Callable[[tuple[np.ndarray, ...]], tuple[np.ndarray, np.ndarray]],
        grids: pp.GridLikeSequence,
    ) -> None:
        """Register a surrogate factory used for local elimination with the model
        framework to have it's update automatized.

        Regular updates are performed on boundaries (at the beginning of every time
        step), and on internal domains before every non-linear solver iteration.

        See also:

            - :meth:`update_all_boundary_conditions`
            - :meth:`initial_condition`
            - :meth:`before_nonlinear_iteration`

        Parameters:
            local_equation_name: The name of the local equation eliminating ``primary``.
            primary: The formally independent Ad operator which was eliminated by the
                expression.
            expression: The secondary expression which eliminates ``primary``, with
                some dependencies on primary variables.
            func: A numerical function returning value and derivative values to be
                inserted into ``expression`` when updateing.

                The derivative values must be a 2D array with rows consistent with the
                number of dependencies in ``expression``.
            grids: A sequence of grids on which it was eliminated.

        """
        boundaries = [g for g in grids if isinstance(g, pp.BoundaryGrid)]
        domains = cast(
            list[pp.Grid] | list[pp.MortarGrid],
            [g for g in grids if isinstance(g, (pp.Grid, pp.MortarGrid))],
        )
        self.__local_eliminations.update(
            {local_equation_name: (primary, expression, func, domains, boundaries)}
        )

    def update_all_boundary_conditions(self) -> None:
        """Attaches to the BC update routine via super call and provides an update for
        surrogate operators as local representations, after all other boundary values
        are updated.

        This functionality is here to help with BC values for eliminated variables, s.t.
        they are consistent with the local elimination.

        """

        # Mypy complains about the parent (the protocol) having a trivial body.
        # We ignore the safe-super check here, but do not comprosie safety by explicitly
        # checking the inheritance tree.
        if isinstance(self, pp.BoundaryConditionMixin):
            super().update_all_boundary_conditions()  # type:ignore[safe-super]
        else:
            raise TypeError(
                f"Model class {type(self)} does not have the BoundaryConditionMixin"
                + " included."
            )

        for elimination in self.__local_eliminations.values():
            eliminatedvar, expr, func, _, bgs = elimination

            # skip if not eliminated on boundary
            if not bgs:
                continue

            def bc_values_prim(bg: pp.BoundaryGrid) -> np.ndarray:
                bc_vals: np.ndarray

                if bg in bgs:
                    X = [
                        d([bg]).value(self.equation_system) for d in expr._dependencies
                    ]
                    bc_vals, _ = func(*X)
                else:
                    bc_vals = np.zeros(bg.num_cells)

                return bc_vals

            self.update_boundary_condition(eliminatedvar.name, bc_values_prim)

    def initial_condition(self) -> None:
        """Attaches to the initialization routine via super-call and computes the
        initial values for constitutive expressions, after values for variables have
        been set.

        Provides initial values for the surrogate operator, but also for the variable
        which was eliminated. I.e., the local equation is fulfilled at the beginning.

        """

        # Same remark as in override of update_all_boundary_conditions
        if isinstance(self, pp.SolutionStrategy):
            super().initial_condition()  # type:ignore[safe-super]
        else:
            raise TypeError(
                f"Model class {type(self)} does not have a SolutionStrategy included."
            )

        for elimination in self.__local_eliminations.values():
            eliminatedvar, expr, f, domains, _ = elimination
            # Initialization is performed grid-wise.
            for grid in domains:
                X = [
                    d(cast(list[pp.Grid] | list[pp.MortarGrid], [grid])).value(
                        self.equation_system
                    )
                    for d in expr._dependencies
                ]
                val, diff = f(*X)
                # Update values and derivatives of surrogate operator.
                expr.set_values_on_grid(val, grid)
                expr.set_derivatives_on_grid(diff, grid)

                # Update value of the variable which was eliminated.
                self.equation_system.set_variable_values(
                    val,
                    [v for v in eliminatedvar.sub_vars if v.domain == grid],
                    iterate_index=0,
                )

    def before_nonlinear_iteration(self) -> None:
        """Attaches to the non-linear iteration routines and performes an update
        of the constitutive expressions before an iteration of the non-linear solver
        is performed.

        Updates both value and derivatives for the surrogate operators used in local
        eliminations.

        """

        # Same remark as in override of update_all_boundary_conditions
        if isinstance(self, pp.SolutionStrategy):
            super().before_nonlinear_iteration()  # type:ignore[safe-super]
        else:
            raise TypeError(
                f"Model class {type(self)} does not have a SolutionStrategy included."
            )

        for elimination in self.__local_eliminations.values():
            _, expr, func, domains, _ = elimination

            for grid in domains:
                X = [
                    d(cast(list[pp.Grid] | list[pp.MortarGrid], [grid])).value(
                        self.equation_system
                    )
                    for d in expr._dependencies
                ]

                vals, diffs = func(*X)

                expr.set_values_on_grid(vals, grid)
                expr.set_derivatives_on_grid(diffs, grid)


class VariableMixin(pp.PorePyModel):
    """Mixin class for variables.

    This class is intended to be used together with the other model classes providing
    generic functionality for variables.

    TODO: Refactor depending on whether other abstract classes are needed. Also, not
    restricted to variables, but also to operators  (e.g. representing a secondary
    variable) having a reference value.

    """

    def create_variables(self) -> None:
        """Method to be overridden to introduce variables on subdomains on interfaces.

        The base class method does nothing and is implemented to provide the right
        signature and to help the super call to resolve the creation of variables in
        multi-physics models.

        Note:
            When creating multi-physics models, the order of variables will reflect
            the order of inheritance of individual variable classes.

        """
        pass

    def perturbation_from_reference(self, name: str, grids: list[pp.Grid]):
        """Perturbation of some quantity ``name`` from its reference value.

        The parameter ``name`` should be the name of a mixed-in method, returning an
        AD operator for given ``grids``.

        ``name`` should also be defined in the model's :attr:`reference_values`.

        This method calls the model method with given ``name`` on given ``grids`` to
        create an operator ``A``. It then fetches the respective reference value and
        wraps it into an AD scalar ``A_0``. The return value is an operator ``A - A_0``.

        Parameters:
            name: Name of the quantity to be perturbed from a reference value.
            grids: List of subdomain or interface grids on which the quantity is
                defined.

        Returns:
            Operator for the perturbation.

        """
        quantity = getattr(self, name)
        # This will throw an error if the attribute is not callable
        quantity_op = cast(pp.ad.Operator, quantity(grids))
        # the reference values are a data class instance storing only numbers
        quantity_ref = cast(pp.number, getattr(self.reference_variable_values, name))
        # The casting reflects the expected outcome, and is used to help linters find
        # the set_name method
        quantity_perturbed = quantity_op - pp.ad.Scalar(quantity_ref)
        quantity_perturbed.set_name(f"{name}_perturbation")
        return quantity_perturbed
