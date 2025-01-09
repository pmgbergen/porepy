"""Abstract equation classes.

Contains:
    - BalanceEquation: Base class for vector and scalar balance equations.
    - VariableMixin: Base class for variables.
"""

from __future__ import annotations

from typing import Union, cast

import porepy as pp


class BalanceEquation(pp.PorePyModel):
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


class VariableMixin(pp.PorePyModel):
    """Mixin class for variables.

    This class is intended to be used together with the other model classes providing
    generic functionality for variables.

    TODO: Refactor depending on whether other abstract classes are needed. Also, not
    restricted to variables, but also to operators  (e.g. representing a secondary
    variable) having a reference value.

    """

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
