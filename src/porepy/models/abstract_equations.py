"""Abstract equation classes.

Contains:
    - BalanceEquation: Base class for vector and scalar balance equations.
    - VariableMixin: Base class for variables.
"""

from __future__ import annotations

from typing import Callable, Sequence, Union

import porepy as pp


class BalanceEquation:
    """Generic class for vector balance equations.

    In the only known use case, the balance equation is the momentum balance equation,

        d_t(momentum) + div(stress) - source = 0,

    with momentum frequently being zero. All terms need to be specified in order to
    define an equation.

    """

    nd: int
    """Ambient dimension of the problem. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """
    wrap_grid_attribute: Callable[[Sequence[pp.GridLike], str, int], pp.ad.DenseArray]
    """Wrap a grid attribute as a DenseArray. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """
    specific_volume: Callable[
        [Union[list[pp.Grid], list[pp.MortarGrid]]], pp.ad.Operator
    ]

    """Function that returns the specific volume of a subdomain or interface.

    Normally provided by a mixin of instance
    :class:`~porepy.models.constitutive_laws.DimensionReduction`.

    """
    basis: Callable[[Sequence[pp.GridLike], int], list[pp.ad.SparseArray]]
    """Basis for the local coordinate system. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """
    time_manager: pp.TimeManager
    """Time manager. Normally set by a mixin instance of
    :class:`porepy.models.solution_strategy.SolutionStrategy`.

    """
    ad_time_step: pp.ad.Scalar
    """Time step as an automatic differentiation scalar. Normally set in
    :class:`porepy.models.solution_strategy.SolutionStrategy`.

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
            integrand: Operator for the integrand. Assumed to be a cell-wise scalar or
                vector quantity, cf. :code:`dim` argument.
            grids: List of subdomains or interfaces to be integrated over.
            dim: Spatial dimension of the integrand. dim = 1 for scalar problems, dim >
                1 for vector problems.

        Returns:
            Operator for the volume integral.

        Raises:
            ValueError: If the grids are not all subdomains or all interfaces.

        """

        assert all(isinstance(g, pp.MortarGrid) for g in grids) or all(
            isinstance(g, pp.Grid) for g in grids
        ), "Grids must be either all subdomains or all interfaces."

        # First account for cell volumes.
        # Ignore mypy complaint about unexpected keyword arguments.
        cell_volumes = self.wrap_grid_attribute(  # type: ignore[call-arg]
            grids, "cell_volumes", dim=1
        )

        # Next, include the effects of reduced dimensions, expressed as specific
        # volumes.
        if dim == 1:
            # No need to do more for scalar problems
            return cell_volumes * self.specific_volume(grids) * integrand
        else:
            # For vector problems, we need to expand the volume array from cell-wise
            # scalar values to cell-wise vectors. We do this by left multiplication with
            #  e_i and summing over i.
            basis: list[pp.ad.SparseArray] = self.basis(
                grids, dim=dim  # type: ignore[call-arg]
            )
            volumes_nd = pp.ad.sum_operator_list(
                [e @ (cell_volumes * self.specific_volume(grids)) for e in basis]
            )

            return volumes_nd * integrand


class VariableMixin:
    """Mixin class for variables.

    This class is intended to be used together with the other model classes providing
    generic functionality for variables.

    TODO: Refactor depending on whether other abstract classes are needed. Also, not
    restricted to variables, but also to operators  (e.g. representing a secondary
    variable) having a reference value.

    """

    def perturbation_from_reference(self, variable_name: str, grids: list[pp.Grid]):
        """Perturbation of a variable from its reference value.

        The parameter :code:`variable_name` should be the name of a variable so that
        :code:`self.variable_name()` and `self.reference_variable_name()` are valid
        calls. These methods will be provided by mixin classes; normally this will be a
        subclass of :class:`VariableMixin`.

        The returned operator will be of the form
        :code:`self.variable_name(grids) - self.reference_variable_name(grids)`.

        Parameters:
            variable_name: Name of the variable.
            grids: List of subdomain or interface grids on which the variable is defined.

        Returns:
            Operator for the perturbation.

        """
        var = getattr(self, variable_name)
        var_ref = getattr(self, "reference_" + variable_name)
        d_var = var(grids) - var_ref(grids)
        d_var.set_name(variable_name + "_perturbation")
        return d_var
