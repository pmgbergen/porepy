"""Abstract equation classes.

Contains:
    - BalanceEquation: Base class for vector and scalar balance equations.
    - VariableMixin: Base class for variables.
"""

from typing import Union

import porepy as pp


class BalanceEquation:
    """Generic class for vector balance equations.

    In the only known use case, the balance equation is the momentum balance equation,

        d_t(momentum) + div(stress) - source = 0,

    with momentum frequently being zero. All terms need to be specified in order to
    define an equation.

    """

    def balance_equation(
        self,
        subdomains: list[pp.Grid],
        accumulation,
        stress,
        source,
        dim: int,
    ) -> pp.ad.Operator:
        """Balance equation for a vector variable.

        Parameters:
            subdomains: List of subdomains where the balance equation is defined.
            accumulation: Operator for the accumulation term. stress: Operator for the
            stress term. source: Operator for the source term.

        Returns:
            Operator for the balance equation.

        """

        dt_operator = pp.ad.time_derivatives.dt
        dt = self.time_manager.dt
        div = pp.ad.Divergence(subdomains, dim=dim)
        return dt_operator(accumulation, dt) + div * stress - source

    def volume_integral(
        self,
        integrand: pp.ad.Operator,
        grids: Union[list[pp.Grid], list[pp.MortarGrid]],
        dim: int,
    ) -> pp.ad.Operator:
        """Numerical volume integral over subdomain or interface cells.

        Includes cell volumes and specific volume.

        Parameters:
            integrand: Operator for the integrand. grids: List of subdomain or interface
            grids over which the integral is to be
                computed.

        Returns:
            Operator for the volume integral.

        """
        cell_volumes = self.wrap_grid_attribute(grids, "cell_volumes")
        # TODO: Extend specific volume to mortar grids.
        if len(grids) == 0:
            # No need for a scaling here
            volumes = cell_volumes
        elif all(isinstance(g, pp.Grid) for g in grids):
            volumes = cell_volumes * self.specific_volume(grids)
        elif not all(isinstance(g, pp.MortarGrid) for g in grids):
            raise ValueError("Grids must be either all subdomains or all interfaces.")
        elif all(g.dim == self.nd - 1 for g in grids):
            # If dim is nd-1, specific volume is 1.
            volumes = cell_volumes
        else:
            raise NotImplementedError(
                "Only implemented for interfaces of dimension nd-1."
            )

        if dim == 1:
            return volumes * integrand
        else:
            # Expand from cell scalar to vector by left and right multiplication with e_i
            # and e_i.T
            basis = self.basis(grids, dim=dim)
            volumes_nd = sum([e * volumes * e.T for e in basis])

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
