"""Abstract equation classes.

Contains:
    - BalanceEquation: Base class for vector and scalar balance equations.
    - VariableMixin: Base class for variables.
"""

from __future__ import annotations

from typing import Callable, Union

import porepy as pp
from porepy.models.protocol import PorePyModel


class BalanceEquation(PorePyModel):
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
            #  e_i and summing over i.
            basis = self.basis(grids, dim=dim)
            volumes_nd = pp.ad.sum_operator_list(
                [e @ (cell_volumes * self.specific_volume(grids)) for e in basis]
            )

            return volumes_nd * integrand


class VariableMixin(PorePyModel):
    """Mixin class for variables.

    This class is intended to be used together with the other model classes providing
    generic functionality for variables.

    TODO: Refactor depending on whether other abstract classes are needed. Also, not
    restricted to variables, but also to operators  (e.g. representing a secondary
    variable) having a reference value.

    """

    def perturbation_from_reference(self, variable_name: str, grids: list[pp.Grid]):
        var = getattr(self, variable_name)
        var_ref = getattr(self, "reference_" + variable_name)
        d_var = var(grids) - var_ref(grids)
        d_var.set_name(variable_name + "_perturbation")
        return d_var
