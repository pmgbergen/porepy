from functools import cached_property
from typing import Callable, Optional, Sequence, Union

import numpy as np

import porepy as pp


class BoundaryConditionMixin:
    """Mixin class for boundary conditions.

    This class is intended to be used together with the other model classes providing
    generic functionality for boundary conditions.

    """

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid for the current model. Normally defined in a mixin
    instance of :class:`~porepy.models.geometry.ModelGeometry`.

    """

    domain_boundary_sides: Callable[[pp.Grid | pp.BoundaryGrid], pp.domain.DomainSides]
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
    """Function that maps a sequence of subdomains to a sequence of boundary grids.
    Normally defined in a mixin instance of
    :class:`~porepy.models.geometry.ModelGeometry`.

    """

    units: "pp.Units"
    """Units object, containing the scaling of base magnitudes."""

    time_step_indices: np.ndarray
    """See :meth:`~porepy.models.solution_strategy.SolutionStragey.time_step_indices`.
    """

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
            self._update_bc_type_filter(name=name, bc_type_callable=bc_type_callable)

    def update_boundary_condition(
        self,
        name: str,
        function: Callable[[pp.BoundaryGrid], np.ndarray],
    ) -> None:
        """This method is the unified procedure of updating a boundary condition.

        It shifts the boundary condition values in time and stores the current iterate
        data (current time step) as the most recent previous time step data.
        Next, it evaluates the boundary condition values for the new time step and
        stores them in the iterate data.

        Parameters:
            name: Name of the operator defined on the boundary.
            function: A callable that provides the boundary condition values on a given
                boundary grid.

        """
        for bg, data in self.mdg.boundaries(return_data=True):
            # Get the current time step values.
            if pp.ITERATE_SOLUTIONS in data and name in data[pp.ITERATE_SOLUTIONS]:
                # Use the values at the unknown time step from the previous time step.
                vals = pp.get_solution_values(name=name, data=data, iterate_index=0)
            else:
                # No current value stored. The method was called during the
                # initialization.
                vals = function(bg)

            # Before setting the new, most recent time step, shift the stored values
            # backwards in time.
            pp.shift_solution_values(
                name=name,
                data=data,
                location=pp.TIME_STEP_SOLUTIONS,
                max_index=len(self.time_step_indices),
            )
            # Set the values of current time to most recent previous time.
            pp.set_solution_values(name=name, values=vals, data=data, time_step_index=0)

            # Set the unknown time step values.
            vals = function(bg)
            pp.set_solution_values(name=name, values=vals, data=data, iterate_index=0)

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

    def _combine_boundary_operators(
        self,
        subdomains: Sequence[pp.Grid],
        dirichlet_operator: Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
        neumann_operator: Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator],
        robin_operator: Optional[
            Union[None, Callable[[Sequence[pp.BoundaryGrid]], pp.ad.Operator]]
        ],
        bc_type: Callable[[pp.Grid], pp.BoundaryCondition],
        name: str,
        dim: int = 1,
    ) -> pp.ad.Operator:
        """Creates an operator representing Dirichlet, Neumann and Robin boundary
        conditions and projects it to the subdomains from boundary grids.

        Parameters:
            subdomains: List of subdomains.
            dirichlet_operator: Function that returns the Dirichlet boundary condition
                operator.
            neumann_operator: Function that returns the Neumann boundary condition
                operator.
            robin_operator: Function that returns the Robin boundary condition operator.
                Expected to be None for e.g. advective fluxes.
            dim: Dimension of the equation. Defaults to 1.
            name: Name of the resulting operator. Must be unique for an operator.

        Returns:
            Boundary condition representation operator.

        """
        boundary_grids = self.subdomains_to_boundary_grids(subdomains)

        # Create dictionaries to hold the Dirichlet and Neumann operators and filters
        operators = {
            "dirichlet": dirichlet_operator(boundary_grids),
            "neumann": neumann_operator(boundary_grids),
        }
        filters = {
            "dirichlet": pp.ad.TimeDependentDenseArray(
                name=(name + "_filter_dir"), domains=boundary_grids
            ),
            "neumann": pp.ad.TimeDependentDenseArray(
                name=(name + "_filter_neu"), domains=boundary_grids
            ),
        }

        # If the Robin operator is not None, it is also included in the operator and
        # filter dictionaries
        if robin_operator is not None:
            operators["robin"] = robin_operator(boundary_grids)
            filters["robin"] = pp.ad.TimeDependentDenseArray(
                name=(name + "_filter_rob"), domains=boundary_grids
            )

        # Adding bc_type function to local storage to evaluate it before every time step
        # in case if the type changes in the runtime.
        self.__bc_type_storage[name] = bc_type

        # Setting the values of the filters for the first time.
        self._update_bc_type_filter(name=name, bc_type_callable=bc_type)

        boundary_to_subdomain = pp.ad.BoundaryProjection(
            self.mdg, subdomains=subdomains, dim=dim
        ).boundary_to_subdomain

        # Apply filters to the operators. This ensures that the Dirichlet operator only
        # assigns (non-zero) values to faces that are marked as having Dirichlet
        # conditions, that the Neumann operator assigns (non-zero) values only to
        # Neumann faces, and that the Robin operator assigns (non-zero) values only to
        # Robin faces.
        for key in operators:
            operators[key] *= filters[key]

        # Combine the operators and project from the boundary grid to the subdomain.
        values = [val for val in operators.values()]  # Get list of values from dict
        combined_operator = pp.ad.sum_operator_list(values)
        result = boundary_to_subdomain @ combined_operator

        result.set_name(name)
        return result

    def _update_bc_type_filter(
        self, name: str, bc_type_callable: Callable[[pp.Grid], pp.BoundaryCondition]
    ):
        """Update the filters for Dirichlet, Neumann and Robin values.

        This is done to discard the data related to Dirichlet and Robin boundary
        condition in cells where the ``bc_type`` is Neumann and vice versa.

        """

        # Note: transposition is unavoidable to treat vector values correctly.
        def dirichlet(bg: pp.BoundaryGrid):
            # Transpose to get a n_face x nd array with shape compatible with
            # the projection matrix.
            is_dir = bc_type_callable(bg.parent).is_dir.T
            is_dir = bg.projection() @ is_dir
            # Transpose back, then ravel (in that order).
            return is_dir.T.ravel("F")

        def neumann(bg: pp.BoundaryGrid):
            is_neu = bc_type_callable(bg.parent).is_neu.T
            is_neu = bg.projection() @ is_neu
            return is_neu.T.ravel("F")

        def robin(bg: pp.BoundaryGrid):
            is_rob = bc_type_callable(bg.parent).is_rob.T
            is_rob = bg.projection() @ is_rob
            return is_rob.T.ravel("F")

        self.update_boundary_condition(name=(name + "_filter_dir"), function=dirichlet)
        self.update_boundary_condition(name=(name + "_filter_neu"), function=neumann)
        self.update_boundary_condition(name=(name + "_filter_rob"), function=robin)

    @cached_property
    def __bc_type_storage(self) -> dict[str, Callable[[pp.Grid], pp.BoundaryCondition]]:
        """Storage of functions that determine the boundary condition type on the given
        grid.

        Used in :meth:`update_all_boundary_conditions` for Dirichlet and Neumann
        filters.

        Stores per operator name (key) a callable (value) returning an operator
        representing the BC type per subdomain.

        """
        return {}
