"""Abstract equation classes.

Contains:
    - BalanceEquation: Base class for vector and scalar balance equations.
    - VariableMixin: Base class for variables.
"""

from typing import Union, Callable, Sequence, Optional

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
    wrap_grid_attribute: Callable[
        [Sequence[pp.GridLike], str, Optional[int], bool], pp.ad.Operator
    ]
    """Wrap grid attributes as Ad operators. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """
    specific_volume: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Function that returns the specific volume of a subdomain. Normally provided by a
    mixin of instance :class:`~porepy.models.constitutive_laws.DimensionReduction`.

    """
    basis: Callable[[Sequence[pp.GridLike], Optional[int]], list[pp.ad.Matrix]]
    """Basis for the local coordinate system. Normally set by a mixin instance of
    :class:`porepy.models.geometry.ModelGeometry`.

    """
    time_manager: pp.TimeManager
    """Time manager. Normally set by a mixin instance of
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
        dt = pp.ad.Scalar(self.time_manager.dt)
        div = pp.ad.Divergence(subdomains, dim=dim)
        return dt_operator(accumulation, dt) + div * surface_term - source

    def volume_integral(
        self,
        integrand: pp.ad.Operator,
        grids: Union[list[pp.Grid], list[pp.MortarGrid]],
        dim: int,
    ) -> pp.ad.Operator:
        """Numerical volume integral over subdomain or interface cells.

        Includes cell volumes and specific volume.

        Parameters:
            integrand: Operator for the integrand.
            grids: List of subdomain or interface to be integrated over.
            dim: Spatial dimension of the integrand.

        Returns:
            Operator for the volume integral.

        Raises:
            ValueError: If the grids are not all subdomains or all interfaces.
            NotImplementedError: If the grids are not all interfaces of dimension nd-1.

        """
        # First construct a volume integral for scalar equations, then extend to vector
        # (below).

        # First account for cell volumes.
        cell_volumes = self.wrap_grid_attribute(grids, "cell_volumes")

        # Next, include the effects of reduced dimensions, expressed as specific
        # volumes.
        if len(grids) == 0:
            # No need for a scaling here
            volumes = cell_volumes
        elif all(isinstance(g, pp.Grid) for g in grids):
            # For grids, we can use the specific volume method.
            # make mypy happy
            subdomains: list[pp.Grid] = [g for g in grids if isinstance(g, pp.Grid)]
            volumes = cell_volumes * self.specific_volume(subdomains)
        elif not all(isinstance(g, pp.MortarGrid) for g in grids):
            # We cannot deal with a combination of subdomains and interfaces.
            raise ValueError("Grids must be either all subdomains or all interfaces.")
        elif all(g.dim == self.nd - 1 for g in grids):
            # Subdomain grids are dealt with above. Here we only need to account for
            # interfaces.
            # For interfaces of dimension nd-1, there is no need for further scalings.
            volumes = cell_volumes
        else:
            # TODO: Extend specific volume to mortar grids.
            # EK: I am not sure what specific volume means for a mortar grid.
            raise NotImplementedError(
                "Only implemented for interfaces of dimension nd-1."
            )

        if dim == 1:
            # No need to do more for scalar problems
            return volumes * integrand
        else:
            # For vector problems, we need to expand the integrand to a vector. Do this
            # by left and right multiplication with e_i and e_i.T
            basis: list[pp.ad.Matrix] = self.basis(grids, dim)
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
