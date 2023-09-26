from abc import ABC, abstractmethod
from typing import Callable, Sequence

import numpy as np

import porepy as pp


class BoundaryConditionMixin(ABC):
    """TODO"""

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

    units: "pp.Units"
    """Units object, containing the scaling of base magnitudes."""

    @abstractmethod
    def update_all_boundary_conditions(self) -> None:
        """This method is called before a new time step to set the values of the
        boundary conditions. The specific boundary condition values should be updated in
        concrete overrides. By default, it does nothing.

        Note:
            One can use the convenience method `update_boundary_condition` for each
            boundary condition value.

        """
        pass

    def update_boundary_condition(
        self,
        name: str,
        function: Callable[[pp.BoundaryGrid], np.ndarray],
    ) -> None:
        """TODO

        TODO: Somewhere we must have an assertion that there is only 1 time step behind
        stored, and 1 iterate.
        """
        for bg, data in self.mdg.boundaries(return_data=True):
            # Set the known time step values.
            if name in data[pp.ITERATE_SOLUTIONS]:
                # Use the values at the unknown time step from the previous time step.
                vals = data[pp.ITERATE_SOLUTIONS][name][0]
            else:
                # No previous time step exists. The method was called during
                # the initialization.
                vals = function(bg)
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
            AssertionError: If the passed sequence of domains does not consist entirely
                of instances of boundary grid.

        Returns:
            An operator of given name representing time-dependent value on given
            sequence of boundary grids.

        """
        assert all(isinstance(x, pp.BoundaryGrid) for x in domains)
        return pp.ad.TimeDependentDenseArray(name=name, domains=domains)
