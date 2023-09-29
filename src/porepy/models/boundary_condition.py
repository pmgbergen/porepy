import warnings
from typing import Callable, Sequence
from functools import cached_property

import numpy as np

import porepy as pp


class BoundaryConditionMixin:
    """Mixin class for bounray conditions.

    This class is intended to be used together with the other model classes providing
    generic functionality for boundary conditions.

    """

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid for the current model. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """

    domain_boundary_sides: Callable[[pp.Grid], pp.domain.DomainSides]
    """Boundary sides of the domain. Normally defined in a mixin instance of
    :class:`~porepy.models.geometry.ModelGeometry`.

    """

    time_manager: pp.TimeManager
    """Time manager. Normally set by an instance of a subclass of
    :class:`porepy.models.solution_strategy.SolutionStrategy`.

    """

    subdomains_to_boundary_grids: Callable[
        [Sequence[pp.Grid]], Sequence[pp.BoundaryGrid]
    ]

    units: "pp.Units"
    """Units object, containing the scaling of base magnitudes."""

    def update_all_boundary_conditions(self) -> None:
        """This method is called before a new time step to set the values of the
        boundary conditions.

        This implementation updates only the filters for Dirichlet and Neumann
        values. The specific boundary condition values should be updated in
        overrides by models.

        Note:
            One can use the convenience method `update_boundary_condition` for each
            boundary condition value.

        """
        for name, bc_type_callable in self.__bc_type_storage.items():
            # Note: transposition is unavoidable to treat vector values correctly.
            def dirichlet(bg: pp.BoundaryGrid):
                is_dir = bc_type_callable(bg.parent).is_dir.T
                is_dir = bg.projection @ is_dir
                return is_dir.T.ravel("F")

            def neumann(bg: pp.BoundaryGrid):
                is_neu = bc_type_callable(bg.parent).is_neu.T
                is_neu = bg.projection @ is_neu
                return is_neu.T.ravel("F")

            self.update_boundary_condition(
                name=(name + "_filter_dir"), function=dirichlet
            )
            self.update_boundary_condition(
                name=(name + "_filter_neu"), function=neumann
            )

    def update_boundary_condition(
        self,
        name: str,
        function: Callable[[pp.BoundaryGrid], np.ndarray],
    ) -> None:
        """This method is the unified procedure of updating a boundary condition.
        It moves the boundary condition values used on the previous time step from
        iterate data (current time step) to previous time step data.
        Next, it evaluates the boundary condition values for the new time step and
        stores them in the iterate data.

        Note:
            This implementation assumes that only one time step and iterate layers are
            used. Otherwise, it prints a warning.

        Parameters:
            name: Name of the operator defined onon the boundary.
            function: A callable that provides the boundary condition values on a given
                boundary grid.

        """
        for bg, data in self.mdg.boundaries(return_data=True):
            # Set the known time step values.
            if name in data[pp.ITERATE_SOLUTIONS]:
                # Use the values at the unknown time step from the previous time step.
                vals = pp.get_solution_values(name=name, data=data, time_step_index=0)
            else:
                # No previous time step exists. The method was called during
                # the initialization.
                vals = function(bg)
            pp.set_solution_values(name=name, values=vals, data=data, time_step_index=0)

            # Set the unknown time step values.
            vals = function(bg)
            pp.set_solution_values(name=name, values=vals, data=data, iterate_index=0)

            # If more than one time step or iterate values are stored, the user should
            # override this method to handle boundary data replacement properly.
            max_steps_back = max(
                len(data[pp.ITERATE_SOLUTIONS][name]),
                len(data[pp.TIME_STEP_SOLUTIONS][name]),
            )
            if max_steps_back > 1:
                warnings.warn(
                    "The default implementation of update_boundary_condition does not"
                    " consider having more than one time step or iterate data layers."
                )

    def create_boundary_operator(
        self, name: str, domains: Sequence[pp.BoundaryGrid]
    ) -> pp.ad.TimeDependentDenseArray:
        """
        Parameters:
            name: Name of the variable or operator to be represented on the boundary.
            domains: A sequence of boundary grids on which the operator is defined.

        Raises:
            ValueError: If the passed sequence of domains does not consist entirely
                of instances of boundary grid.

        Returns:
            An operator of given name representing time-dependent value on given
            sequence of boundary grids.

        """
        if not all(isinstance(x, pp.BoundaryGrid) for x in domains):
            raise ValueError("domains must consist entirely of the boundary grids.")
        return pp.ad.TimeDependentDenseArray(name=name, domains=domains)

    def _make_boundary_operator(
        self,
        subdomains: Sequence[pp.Grid],
        dirichlet_operator: Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
        neumann_operator: Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
        bc_type: Callable[[pp.Grid], pp.BoundaryCondition],
        name: str,
        dim: int = 1,
    ) -> pp.ad.Operator:
        """Creates an operator representing Dirichlet and Neumann boundary conditions
        and projects it to the subdomains from boundary grids.

        Parameters:
            subdomains: List of subdomains.

            dirichlet_operator: Function that returns the Dirichlet boundary condition
                operator.

            neumann_operator: Function that returns the Neumann boundary condition
                operator.

            dim: Dimension of the equation. Defaults to 1.

            name: Name of the resulting operator.

        Returns:
            Boundary condition representation operator.

        """
        boundary_grids = self.subdomains_to_boundary_grids(subdomains)
        dirichlet = dirichlet_operator(boundary_grids)
        neumann = neumann_operator(boundary_grids)

        self.__bc_type_storage[name] = bc_type
        dir_filter = pp.ad.TimeDependentDenseArray(
            name=(name + "_filter_dir"), domains=boundary_grids
        )
        neu_filter = pp.ad.TimeDependentDenseArray(
            name=(name + "_filter_neu"), domains=boundary_grids
        )

        boundary_to_subdomain = pp.ad.BoundaryProjection(
            self.mdg, subdomains=subdomains, dim=dim
        ).boundary_to_subdomain

        dirichlet *= dir_filter
        neumann *= neu_filter
        result = boundary_to_subdomain @ (dirichlet + neumann)
        result.set_name(name)

        # self.update_all_boundary_conditions()  # TODO: Remove this line

        return result

    @cached_property
    def __bc_type_storage(self) -> dict[str, Callable[[pp.Grid], pp.BoundaryCondition]]:
        """TODO"""
        return {}
